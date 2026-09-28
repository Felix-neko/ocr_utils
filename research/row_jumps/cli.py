"""Команды стенда: ``measure`` — перескоки по сохранённым осям, ``select`` — P и N, ``run`` — пересчёт полос с резкой двухрядных сгустков и без, ``compare`` — «было | стало»."""

from __future__ import annotations

import csv
import json
import logging
import random
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, header_strip, legend_strip
from ocr_utils.page_layout.pack_analysis.final import page_record, text_blocks
from ocr_utils.page_layout.pack_analysis.stages import PageTask, load_image, page_key
from ocr_utils.page_layout.text_blocks import RENDER_DPI
from ocr_utils.page_layout.text_blocks import overlay as blocks_overlay
from ocr_utils.page_layout.text_blocks.metrics import _axis_points, metrics_of, pitch_of, row_jumps_of
from ocr_utils.page_layout.text_blocks.page import AxisKind
from ocr_utils.page_layout.text_blocks.report import page_json
from ocr_utils.page_layout.text_blocks.sides_overlay import side_by_side
from research.gutter_crossing.cli import _init_worker, read_keys

logger = logging.getLogger(__name__)

# Ширина оверлея полосы (как у разбора пака).
OVERLAY_WIDTH = 1600
# Ось-перескок — толсто красным поверх разбора; её сосед, по которому мерили, — оранжевым.
COLOUR_JUMP = (60, 60, 220)
COLOUR_NEIGHBOUR = (0, 165, 255)
# Вырезка у находки: поле вокруг оси по высоте (px рабочей копии) и ширина склейки.
ZOOM_PAD_PX = 60
# Обводка места перескока: эллипс вокруг участка, где расстояние до соседа меняется быстрее всего.
COLOUR_SPOT = (0, 0, 255)
SPOT_HALF_WIDTH_PITCHES = 3.0
SPOT_HALF_HEIGHT_PITCHES = 1.6
ZOOM_WIDTH = 1400


class Variant(str, Enum):
    """Вариант пересчёта: без резки двухрядных сгустков и с ней; ``marks`` — с резкой на коде с верхними знаками.

    Варианты различаются папкой выхода: ``marks`` считается тем же флагом, что ``split``, но на коде,
    где верхние знаки («й», «ё», индексы) сажаются на базовую линию (``pieces.baselines_of``), —
    так прошлые ``plain`` и ``split`` остаются для сравнения «было | стало».
    """

    PLAIN = "plain"
    SPLIT = "split"
    MARKS = "marks"
    STACKED = "stacked"  # + кружок «%» как знак над буквой и многопроходная резка (сгусток на 3+ строк)


def jumps_of(payload: dict) -> list[dict]:
    """Перескоки по JSON разбора (``report.page_json``): :func:`metrics.row_jumps_of` словарями.

    Args:
        payload: Разбор полосы с осями и блоками.

    Returns:
        Находки ``RowJump.to_json()``.
    """
    axes = [_axis_points(axis) for axis in payload.get("axes", [])]
    return [jump.to_json() for jump in row_jumps_of(axes, pitch_of(payload.get("blocks", [])))]


def _measure_file(path: Path) -> dict:
    """Мера одной сохранённой полосы: ключ, шаг, находки."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"key": payload["key"], "folder": "", "jumps": len(jumps_of(payload))}


def jump_spot(axis: np.ndarray, neighbour: np.ndarray) -> tuple[float, float]:
    """Место перескока: точка оси, где её расстояние до соседа меняется быстрее всего.

    Args:
        axis: Ось-перескок ``(n, 2)``.
        neighbour: Сосед, по которому мерили.

    Returns:
        ``(x, y)`` в пикселях рабочей копии.
    """
    left, right = max(axis[0, 0], neighbour[0, 0]), min(axis[-1, 0], neighbour[-1, 0])
    xs = np.linspace(left, right, 200)
    gap = np.interp(xs, axis[:, 0], axis[:, 1]) - np.interp(xs, neighbour[:, 0], neighbour[:, 1])
    # Скорость изменения расстояния, сглаженная по 9 точкам: одиночный зубец оси не в счёт.
    speed = np.abs(np.convolve(np.gradient(gap), np.ones(9) / 9.0, mode="same"))
    x = float(xs[int(np.argmax(speed))])
    return x, float(np.interp(x, axis[:, 0], axis[:, 1]))


def draw_page(analysis, gray300: np.ndarray, jumps: list[dict], axes: list[dict], title: list[str]) -> np.ndarray:
    """Оверлей полосы: разбор текстовых блоков, оси-перескоки и их соседи; шапка и легенда в поле.

    Args:
        analysis: ``PageAnalysis`` полосы.
        gray300: Серый рендер ``RENDER_DPI``.
        jumps: Находки :func:`jumps_of`.
        axes: Оси из JSON разбора (для отрисовки находок).
        title: Строки шапки.

    Returns:
        Картинка BGR.
    """
    scale = OVERLAY_WIDTH / analysis.width
    small = cv2.resize(gray300, (OVERLAY_WIDTH, int(round(analysis.height * scale))), interpolation=cv2.INTER_AREA)
    canvas = blocks_overlay.draw(analysis, small, scale=scale)
    for jump in jumps:
        for index, colour, width in ((jump["neighbour"], COLOUR_NEIGHBOUR, 2), (jump["axis"], COLOUR_JUMP, 4)):
            points = np.round(np.asarray(axes[index]["points"], dtype=np.float64) * scale).astype(np.int32)
            cv2.polylines(canvas, [points], False, colour, width)
    # Обводка места перескока — эллипсом в шаг строк страницы.
    pitch = pitch_of(page_json(analysis)["blocks"]) or 20.0
    for jump in jumps:
        x, y = jump_spot(
            np.asarray(axes[jump["axis"]]["points"], dtype=np.float64),
            np.asarray(axes[jump["neighbour"]]["points"], dtype=np.float64),
        )
        size = (int(SPOT_HALF_WIDTH_PITCHES * pitch * scale), int(SPOT_HALF_HEIGHT_PITCHES * pitch * scale))
        cv2.ellipse(canvas, (int(x * scale), int(y * scale)), size, 0, 0, 360, COLOUR_SPOT, 3)
    width = canvas.shape[1]
    entries = [
        LegendEntry("ось-перескок", COLOUR_JUMP),
        LegendEntry("сосед, по которому мерили", COLOUR_NEIGHBOUR),
        LegendEntry("место перескока", COLOUR_SPOT),
    ]
    return np.vstack(
        [
            header_strip(title, width),
            canvas,
            legend_strip(entries, width, "стенд"),
            legend_strip(blocks_overlay.legend_entries(body_axis=True), width, "текстовые блоки"),
        ]
    )


def run_page(key: str, pack_dir: Path, sharpened_dir: Path, out_dir: Path, variant: Variant) -> dict:
    """Пересчитать текстовые блоки полосы входом разбора пака и замерить перескоки.

    Args:
        key: Полоса ``год/выпуск/полоса``.
        pack_dir: Готовый разбор пака (объекты, надписи, ориентация — из ``pages/*.json``).
        sharpened_dir: Заострённые сканы.
        out_dir: Выход варианта (``pages/``, ``overlays/``).
        variant: С резкой двухрядных сгустков или без.

    Returns:
        Строка сводки: ключ, папка, осей, блоков, перескоков, ступенек, скрещиваний, покрытие.
    """
    name = page_key(key)
    final = json.loads((pack_dir / "pages" / f"{name}.json").read_text(encoding="utf-8"))
    record = page_record(final)
    image = load_image(PageTask(key, sharpened_dir / f"{key}.jpg"), record["rotate_cw"])
    analysis, _ = text_blocks(image, record, final["objects"], AxisKind.BODY, split_rows=variant is not Variant.PLAIN)
    payload = page_json(analysis)
    payload["key"] = key
    jumps = jumps_of(payload)
    payload["jumps"] = jumps
    target = out_dir / "pages" / f"{name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    title = [
        f"{key}   вариант: {variant.value}",
        f"осей {len(analysis.axes)}, блоков {len(analysis.blocks)}, перескоков {len(jumps)}",
    ]
    picture = draw_page(analysis, image.gray_at(RENDER_DPI), jumps, payload["axes"], title)
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "overlays" / f"{name}.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
    measures = metrics_of(payload)
    return {
        "key": key,
        "folder": final.get("folder", ""),
        "axes": len(analysis.axes),
        "blocks": len(analysis.blocks),
        "jumps": len(jumps),
        "steps": measures.steps,
        "crossed_axes": measures.crossings,
        "ink_share": round(measures.ink_share, 4),
    }


def _run_one(args: tuple) -> dict | None:
    """Обёртка для пула: ошибка одной полосы не рушит прогон."""
    try:
        return run_page(*args)
    except Exception:  # noqa: BLE001 — полоса с ошибкой уходит в лог
        logger.exception("Полоса %s: ошибка разбора", args[0])
        return None


def _write_csv(rows: list[dict], path: Path) -> None:
    """CSV по строкам-словарям (заголовок — ключи первой строки)."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@click.group()
def main() -> None:
    """Стенд «перескок оси на соседнюю строку»."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@main.command()
@click.option(
    "--pages-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Папка JSON разбора с осями (report.page_json), например pack1_gutter_crossing/legacy/pages.",
)
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True, help="CSV «ключ, перескоков».")
@click.option("--jobs", default=16, show_default=True)
def measure(pages_dir: Path, out: Path, jobs: int) -> None:
    """Перескоки по сохранённым осям всего пака, без пересчёта разбора."""
    paths = sorted(pages_dir.glob("*.json"))
    with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        rows = list(pool.map(_measure_file, paths, chunksize=16))
    rows.sort(key=lambda row: row["key"])
    _write_csv(rows, out)
    flagged = sum(1 for row in rows if row["jumps"])
    logger.info(
        "Полос %d, с перескоками %d, перескоков %d → %s", len(rows), flagged, sum(r["jumps"] for r in rows), out
    )


@main.command()
@click.option("--measures", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--seed", default=0, show_default=True)
def select(measures: Path, out_dir: Path, seed: int) -> None:
    """P — все полосы с перескоками, N — столько же случайных без них: ``problem.txt``, ``normal.txt``."""
    rows = list(csv.DictReader(measures.open(encoding="utf-8")))
    problem = [row["key"] for row in rows if int(row["jumps"])]
    clean = [row["key"] for row in rows if not int(row["jumps"])]
    normal = sorted(random.Random(seed).sample(clean, min(len(problem), len(clean))))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "problem.txt").write_text("\n".join(problem) + "\n", encoding="utf-8")
    (out_dir / "normal.txt").write_text("\n".join(normal) + "\n", encoding="utf-8")
    logger.info("P: %d полос, N: %d → %s", len(problem), len(normal), out_dir)


@main.command()
@click.option("--pack-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--sharpened-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--keys", "keys_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--variant", type=click.Choice([v.value for v in Variant]), required=True)
@click.option("--jobs", default=16, show_default=True)
def run(pack_dir: Path, sharpened_dir: Path, out_dir: Path, keys_file: Path, variant: str, jobs: int) -> None:
    """Пересчитать полосы в варианте ``--variant``: ``<out-dir>/<вариант>/pages``, ``overlays``, ``summary.csv``."""
    chosen = Variant(variant)
    keys = read_keys(keys_file, pack_dir)
    target = out_dir / chosen.value
    target.mkdir(parents=True, exist_ok=True)
    tasks = [(key, pack_dir, sharpened_dir, target, chosen) for key in keys]
    rows = []
    with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        for row in pool.map(_run_one, tasks, chunksize=2):
            if row is not None:
                rows.append(row)
    _write_csv(rows, target / "summary.csv")
    flagged = sum(1 for row in rows if row["jumps"])
    logger.info("Вариант %s: полос %d, с перескоками %d → %s", chosen.value, len(rows), flagged, target)


def _zoom(picture: np.ndarray, payload: dict, jumps: list[dict]) -> np.ndarray | None:
    """Вырезка из оверлея вокруг находок (оверлей — с шапкой в две строки сверху)."""
    if not jumps:
        return None
    scale = OVERLAY_WIDTH / payload["width"]
    offset = header_strip(["", ""], picture.shape[1]).shape[0]
    ys = [np.asarray(payload["axes"][jump["axis"]]["points"])[:, 1] for jump in jumps]
    y0 = max(0, int((min(float(y.min()) for y in ys) - ZOOM_PAD_PX) * scale))
    y1 = int((max(float(y.max()) for y in ys) + ZOOM_PAD_PX) * scale)
    return picture[offset + y0 : offset + y1]


@main.command()
@click.option("--out-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--before", "before_name", default=Variant.PLAIN.value, show_default=True)
@click.option("--after", "after_name", default=Variant.SPLIT.value, show_default=True)
def compare(out_dir: Path, before_name: str, after_name: str) -> None:
    """Склейки «было | стало» (вся полоса и вырезка у находок) и CSV по полосам: ``<out-dir>/compare[_<было>_<стало>]``."""
    before, after = out_dir / before_name, out_dir / after_name
    default = before_name == Variant.PLAIN.value and after_name == Variant.SPLIT.value
    target = out_dir / ("compare" if default else f"compare_{before_name}_{after_name}")
    (target / "pages").mkdir(parents=True, exist_ok=True)
    (target / "zoom").mkdir(parents=True, exist_ok=True)
    old_rows = {row["key"]: row for row in csv.DictReader((before / "summary.csv").open(encoding="utf-8"))}
    new_rows = {row["key"]: row for row in csv.DictReader((after / "summary.csv").open(encoding="utf-8"))}
    rows = []
    for key in sorted(set(old_rows) & set(new_rows)):
        name = page_key(key)
        old = json.loads((before / "pages" / f"{name}.json").read_text(encoding="utf-8"))
        new = json.loads((after / "pages" / f"{name}.json").read_text(encoding="utf-8"))
        changed = old["axes"] != new["axes"] or [b["rows"] for b in old["blocks"]] != [b["rows"] for b in new["blocks"]]
        row = {"key": key, "folder": old_rows[key]["folder"], "changed": int(changed)}
        for field in ("jumps", "axes", "blocks", "steps", "crossed_axes"):
            row[f"{field}_before"] = int(old_rows[key][field])
            row[f"{field}_after"] = int(new_rows[key][field])
        rows.append(row)
        if not changed:
            continue
        old_pic = cv2.imread(str(before / "overlays" / f"{name}.jpg"))
        new_pic = cv2.imread(str(after / "overlays" / f"{name}.jpg"))
        titles = [f"было: {before_name}", f"стало: {after_name}"]
        cv2.imwrite(str(target / "pages" / f"{name}.jpg"), side_by_side([old_pic, new_pic], titles))
        # Вырезка — у находок «было» (если их нет — «стало»): одно и то же место на обеих картинках.
        spots = old["jumps"] or new["jumps"]
        source = old if old["jumps"] else new
        parts = [_zoom(old_pic, source, spots), _zoom(new_pic, source, spots)]
        if parts[0] is None:
            continue
        scale = ZOOM_WIDTH / max(1, sum(part.shape[1] for part in parts))
        parts = [cv2.resize(part, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for part in parts]
        cv2.imwrite(str(target / "zoom" / f"{name}.jpg"), side_by_side(parts, titles), [cv2.IMWRITE_JPEG_QUALITY, 85])
    _write_csv(rows, target / "compare.csv")
    fixed = sum(1 for r in rows if r["jumps_before"] and not r["jumps_after"])
    broke = sum(1 for r in rows if not r["jumps_before"] and r["jumps_after"])
    logger.info(
        "Полос %d: перескок снят %d, появился %d, изменились оси или ряды %d → %s",
        len(rows),
        fixed,
        broke,
        sum(r["changed"] for r in rows),
        target,
    )
