"""Команды стенда: ``select`` — множества P/N по готовому прогону пака, ``run`` — пересчёт текстовых блоков по ключам со сращиванием на отточиях или без, ``compare`` — «было | стало»."""

from __future__ import annotations

import csv
import json
import logging
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, header_strip, legend_strip
from ocr_utils.page_layout.pack_analysis.final import text_blocks
from ocr_utils.page_layout.pack_analysis.stages import PageTask, load_image, page_key
from ocr_utils.page_layout.text_blocks import RENDER_DPI, WORK_DPI
from ocr_utils.page_layout.text_blocks import overlay as blocks_overlay
from ocr_utils.page_layout.text_blocks.columns import GutterMode
from ocr_utils.page_layout.text_blocks.metrics import metrics_of
from ocr_utils.page_layout.text_blocks.page import AxisKind
from ocr_utils.page_layout.text_blocks.report import page_json
from ocr_utils.page_layout.text_blocks.sides_overlay import side_by_side
from research.gutter_crossing.cli import _init_worker, read_keys
from research.leader_merge.pairs import LEADER_TAIL_HEIGHTS, shared_pairs

logger = logging.getLogger(__name__)

# Ширина оверлея полосы (как у оверлеев разбора пака).
OVERLAY_WIDTH = 1600
# Находка на оверлее: перекрытие двух осей — жирный отрезок на высоте ряда.
COLOUR_PAIR = (60, 60, 220)
# Вырезка вокруг находок на склейке «было | стало»: поле вокруг (мм) и ширина склейки.
ZOOM_PAD_MM = 12.0
ZOOM_WIDTH = 1400
# Полоса «с отточием» для слоя N: у какой-нибудь оси столько низких меток подряд с ровным шагом.
LEADER_MARKS = 4
LEADER_STEP_SPREAD = 0.25
# Высокая строка (заголовок) для слоя N: выше стольких медиан высот строк полосы.
TITLE_HEIGHTS = 1.8
# Слои беспроблемного множества N (порядок — порядок добора).
LAYERS = ("отточия", "не_только_текст", "колонки", "заголовки", "обычная")


def _has_leader(page: dict) -> bool:
    """Есть ли на полосе ось с отточием: ``LEADER_MARKS`` низких меток подряд с почти ровным шагом."""
    for axis in page["axes"]:
        centres = np.array([(a + b) / 2.0 for a, b in axis["mark_spans"]])
        if centres.size < LEADER_MARKS:
            continue
        steps = np.diff(np.sort(centres))
        # Скользящее окно из LEADER_MARKS−1 шагов: разброс шага мал против самого шага.
        for start in range(steps.size - LEADER_MARKS + 2):
            window = steps[start : start + LEADER_MARKS - 1]
            if window.size and np.ptp(window) <= LEADER_STEP_SPREAD * np.median(window):
                return True
    return False


def _layer_of(page: dict, folder: str) -> str:
    """Слой беспроблемной полосы для стратифицированной выборки N (первый подходящий из ``LAYERS``)."""
    if _has_leader(page):
        return "отточия"
    if folder.startswith("не_только_текст"):
        return "не_только_текст"
    if any(len(zone["columns"]) >= 2 for zone in page["zones"]):
        return "колонки"
    heights = np.array([axis["height"] for axis in page["axes"]]) if page["axes"] else np.zeros(0)
    if heights.size and heights.max() >= TITLE_HEIGHTS * np.median(heights):
        return "заголовки"
    return "обычная"


def _scan_page(path: Path) -> tuple[str, int, str]:
    """Воркер отбора: ключ полосы, число находок и слой (для чистых полос)."""
    page = json.loads(path.read_text(encoding="utf-8"))
    pairs = shared_pairs(page["axes"])
    return page["key"], len(pairs), "" if pairs else _layer_of(page, "")


@click.group()
def main() -> None:
    """Стенд «оси, сошедшиеся на отточии»."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@main.command()
@click.option(
    "--source",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="прогон с осями полос (pages/*.json из report.page_json), например pack1_gutter_crossing/legacy",
)
@click.option("--pack-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--problem", default=20, show_default=True, help="Сколько полос в P.")
@click.option("--normal", default=80, show_default=True, help="Сколько полос в N.")
@click.option(
    "--leaders-from",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="прогон run по полосам с находками (с боксами глифов): P берётся только из пар на отточии (SharedPair.is_leader)",
)
@click.option("--must", multiple=True, help="Ключи, обязательно входящие в P (если на них есть находки).")
@click.option("--seed", default=0, show_default=True)
@click.option("--jobs", default=16, show_default=True, help="Воркеров пула (чтение JSON).")
def select(
    source: Path,
    pack_dir: Path,
    out_dir: Path,
    problem: int,
    normal: int,
    leaders_from: Path | None,
    must: tuple,
    seed: int,
    jobs: int,
) -> None:
    """Множества P (находки ``shared_pairs``) и N (без находок, по слоям): ``problem.txt``, ``normal.txt``, ``scan.csv``."""
    paths = sorted((source / "pages").glob("*.json"))
    with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        rows = list(pool.map(_scan_page, paths, chunksize=32))
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "scan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "pairs", "layer"])
        writer.writerows(rows)
    generator = random.Random(seed)
    # P: обязательные, затем по одной полосе на выпуск по кругу (разброс по годам), остальное — случайно.
    flagged = {key: count for key, count, _ in rows if count}
    any_pairs = set(flagged)
    if leaders_from is not None:
        # Только пары на отточии: по сохранённым осям без глифов их не отличить от обрывков текста на запятой.
        leader_pairs = {}
        for path in (leaders_from / "pages").glob("*.json"):
            page = json.loads(path.read_text(encoding="utf-8"))
            count = sum(1 for pair in page["pairs"] if pair["tail"] is not None and pair["tail"] >= LEADER_TAIL_HEIGHTS)
            if count:
                leader_pairs[page["key"]] = count
        logger.info("Из %d полос с находками на отточии — %d", len(flagged), len(leader_pairs))
        flagged = leader_pairs
    chosen = [key for key in must if key in flagged]
    by_issue: dict[str, list[str]] = defaultdict(list)
    for key in sorted(flagged):
        if key not in chosen:
            by_issue[key.rsplit("/", 1)[0]].append(key)
    issues = sorted(by_issue)
    generator.shuffle(issues)
    while len(chosen) < problem and any(by_issue.values()):
        for issue in issues:
            if by_issue[issue] and len(chosen) < problem:
                chosen.append(by_issue[issue].pop(generator.randrange(len(by_issue[issue]))))
    # N: поровну по слоям, недобор — из обычных. Слой «не только текст» — по папке итогового JSON пака.
    layers: dict[str, list[str]] = defaultdict(list)
    for key, count, layer in rows:
        if count or key in any_pairs:
            continue
        if layer != "отточия":
            final = json.loads((pack_dir / "pages" / f"{page_key(key)}.json").read_text(encoding="utf-8"))
            if final.get("folder", "").startswith("не_только_текст"):
                layer = "не_только_текст"
        layers[layer].append(key)
    quota = max(1, normal // len(LAYERS))
    picked: list[tuple[str, str]] = []
    for layer in LAYERS:
        picked += [(key, layer) for key in generator.sample(layers[layer], min(quota, len(layers[layer])))]
    taken = {key for key, _ in picked}
    rest = [key for key in layers["обычная"] if key not in taken]
    picked += [(key, "обычная") for key in generator.sample(rest, max(0, min(len(rest), normal - len(picked))))]
    (out_dir / "problem.txt").write_text("".join(f"{key},находок {flagged[key]}\n" for key in chosen), encoding="utf-8")
    (out_dir / "normal.txt").write_text("".join(f"{key},{layer}\n" for key, layer in picked), encoding="utf-8")
    sizes = ", ".join(f"{layer} {len(layers[layer])}" for layer in LAYERS)
    logger.info(
        "Полос %d, с находками %d (пар %d); P: %d, N: %d (чистых по слоям: %s) → %s",
        len(rows),
        len(flagged),
        sum(flagged.values()),
        len(chosen),
        len(picked),
        sizes,
        out_dir,
    )


def draw_page(analysis, gray300: np.ndarray, pairs: list[dict], title: list[str]) -> np.ndarray:
    """Оверлей полосы: разбор текстовых блоков и находки (перекрытие двух осей — жирным), шапка и легенда в поле.

    Args:
        analysis: ``PageAnalysis`` полосы.
        gray300: Серый рендер ``RENDER_DPI``.
        pairs: Находки :func:`pairs.shared_pairs` словарями.
        title: Строки шапки.

    Returns:
        Картинка BGR.
    """
    scale = OVERLAY_WIDTH / analysis.width
    small = cv2.resize(gray300, (OVERLAY_WIDTH, int(round(analysis.height * scale))), interpolation=cv2.INTER_AREA)
    canvas = blocks_overlay.draw(analysis, small, scale=scale)
    # Перекрытие — отрезок чуть ниже ряда, чтобы не закрывать сами оси.
    for item in pairs:
        y = int((item["y"] + 4) * scale)
        cv2.line(canvas, (int(item["x0"] * scale), y), (int(item["x1"] * scale), y), COLOUR_PAIR, 5)
    width = canvas.shape[1]
    entries = [LegendEntry("две оси сошлись на общем глифе и не срослись: перекрытие", COLOUR_PAIR)]
    return np.vstack(
        [
            header_strip(title, width),
            canvas,
            legend_strip(entries, width, "стенд"),
            legend_strip(blocks_overlay.legend_entries(body_axis=True), width, "текстовые блоки"),
        ]
    )


def run_page(key: str, pack_dir: Path, sharpened_dir: Path, out_dir: Path, join: bool, overlays: bool) -> dict:
    """Пересчитать текстовые блоки полосы тем же входом, что в разборе пака v3, и замерить находки.

    Args:
        key: Полоса ``год/выпуск/полоса``.
        pack_dir: Выход разбора пака (запись кандидатов ``work/pages`` и объекты ``pages``).
        sharpened_dir: Заострённые сканы.
        out_dir: Выход прогона (``pages/``, ``overlays/``).
        join: Сращивать ли оси, сошедшиеся на отточии (``segment.join_on_leaders``).
        overlays: Писать ли оверлей.

    Returns:
        Строка сводки: ключ, число осей, блоков, находок и меры ``metrics.py``.
    """
    name = page_key(key)
    record = json.loads((pack_dir / "work" / "pages" / f"{name}.json").read_text(encoding="utf-8"))
    final = json.loads((pack_dir / "pages" / f"{name}.json").read_text(encoding="utf-8"))
    image = load_image(PageTask(key, sharpened_dir / f"{key}.jpg"), record["rotate_cw"])
    analysis, _ = text_blocks(image, record, final["objects"], AxisKind.BODY, GutterMode.LEGACY, join_leaders=join)
    payload = page_json(analysis)
    payload["key"] = key
    # Боксы глифов — для меры по любому общему глифу, а не только по низким меткам.
    for item, axis in zip(payload["axes"], analysis.axes):
        item["glyphs"] = [] if axis.glyphs is None else np.round(axis.glyphs, 1).tolist()
    pairs = [pair.to_json() for pair in shared_pairs(payload["axes"])]
    payload["pairs"] = pairs
    target = out_dir / "pages" / f"{name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    if overlays:
        title = [
            f"{key}   сращивание на отточиях: {'да' if join else 'нет'}",
            f"осей {len(analysis.axes)}, блоков {len(analysis.blocks)}, находок {len(pairs)}",
        ]
        picture = draw_page(analysis, image.gray_at(RENDER_DPI), pairs, title)
        (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "overlays" / f"{name}.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
    measures = metrics_of(payload)
    return {
        "key": key,
        "axes": len(analysis.axes),
        "blocks": len(analysis.blocks),
        "pairs": len(pairs),
        "steps": measures.steps,
        "crossed_axes": measures.crossings,
        "escaping": measures.escaping,
        "outside": measures.outside,
        "half_rows": measures.half_rows,
        "ink_share": round(measures.ink_share, 4),
    }


def _run_one(args: tuple) -> dict | None:
    """Обёртка для пула: ошибка одной полосы не рушит прогон."""
    try:
        return run_page(*args)
    except Exception:  # noqa: BLE001 — полоса с ошибкой уходит в лог, прогон идёт дальше
        logger.exception("Полоса %s: ошибка разбора", args[0])
        return None


@main.command()
@click.option("--pack-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--sharpened-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--keys", "keys_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--join/--no-join", default=True, help="Сращивать оси, сошедшиеся на отточии.")
@click.option("--overlays/--no-overlays", default=True, help="Писать оверлеи полос.")
@click.option("--jobs", default=16, show_default=True, help="Воркеров пула.")
def run(
    pack_dir: Path, sharpened_dir: Path, out_dir: Path, keys_file: Path, join: bool, overlays: bool, jobs: int
) -> None:
    """Пересчитать полосы по ключам: ``<out-dir>/pages``, ``overlays``, ``summary.csv``."""
    keys = read_keys(keys_file, pack_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(key, pack_dir, sharpened_dir, out_dir, join, overlays) for key in keys]
    with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        rows = [row for row in pool.map(_run_one, tasks) if row is not None]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(
        "Полос %d (ошибок %d), с находками %d, находок %d → %s",
        len(rows),
        len(tasks) - len(rows),
        sum(1 for row in rows if row["pairs"]),
        sum(row["pairs"] for row in rows),
        out_dir,
    )


def _same_geometry(old: dict, new: dict) -> bool:
    """Совпадают ли оси и полигоны блоков двух прогонов полосы (точно, по сохранённым числам)."""
    if [a["points"] for a in old["axes"]] != [a["points"] for a in new["axes"]]:
        return False
    return [b["envelope"]["polygon"] for b in old["blocks"]] == [b["envelope"]["polygon"] for b in new["blocks"]]


def _zoom_box(pairs: list[dict], width: int, height: int) -> tuple[int, int, int, int] | None:
    """Рамка вокруг всех находок полосы с полем ``ZOOM_PAD_MM`` (по x — вчетверо шире); ``None`` — находок нет."""
    if not pairs:
        return None
    pad = ZOOM_PAD_MM * WORK_DPI / 25.4
    x0 = max(0, min(p["x0"] for p in pairs) - 4 * pad)
    x1 = min(width, max(p["x1"] for p in pairs) + 4 * pad)
    y0 = max(0, min(p["y"] for p in pairs) - pad)
    y1 = min(height, max(p["y"] for p in pairs) + pad)
    return int(x0), int(y0), int(x1), int(y1)


def _crop(picture: np.ndarray, box: tuple[int, int, int, int], page_width: int) -> np.ndarray:
    """Вырезка из оверлея по рамке в пикселях рабочей копии (над холстом — шапка в две строки)."""
    scale = OVERLAY_WIDTH / page_width
    offset = header_strip(["", ""], picture.shape[1]).shape[0]
    x0, y0, x1, y1 = (int(v * scale) for v in box)
    return picture[offset + y0 : offset + y1, x0:x1]


@main.command()
@click.option("--before", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--after", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--keys", "keys_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
def compare(before: Path, after: Path, keys_file: Path, out_dir: Path) -> None:
    """Склейки «было | стало», вырезки у находок и CSV мер: ``<out-dir>/pages``, ``zoom``, ``compare.csv``."""
    (out_dir / "pages").mkdir(parents=True, exist_ok=True)
    (out_dir / "zoom").mkdir(parents=True, exist_ok=True)
    old_rows = {row["key"]: row for row in csv.DictReader((before / "summary.csv").open(encoding="utf-8"))}
    new_rows = {row["key"]: row for row in csv.DictReader((after / "summary.csv").open(encoding="utf-8"))}
    labels = {}
    for line in keys_file.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            key, _, label = line.partition(",")
            labels[key.strip()] = label.strip()
    rows = []
    for key, label in labels.items():
        if key not in old_rows or key not in new_rows:
            continue
        name = page_key(key)
        old_page = json.loads((before / "pages" / f"{name}.json").read_text(encoding="utf-8"))
        new_page = json.loads((after / "pages" / f"{name}.json").read_text(encoding="utf-8"))
        row = {"key": key, "label": label}
        for field in ("pairs", "axes", "blocks", "steps", "crossed_axes", "escaping", "outside", "half_rows"):
            row[f"{field}_before"] = int(old_rows[key][field])
            row[f"{field}_after"] = int(new_rows[key][field])
        row["same"] = int(_same_geometry(old_page, new_page))
        rows.append(row)
        if row["same"]:
            continue
        old_pic = cv2.imread(str(before / "overlays" / f"{name}.jpg"))
        new_pic = cv2.imread(str(after / "overlays" / f"{name}.jpg"))
        if old_pic is None or new_pic is None:
            continue
        cv2.imwrite(
            str(out_dir / "pages" / f"{name}.jpg"),
            side_by_side([old_pic, new_pic], ["было", "стало"]),
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )
        box = _zoom_box(old_page["pairs"] or new_page["pairs"], old_page["width"], old_page["height"])
        if box is None:
            continue
        parts = [_crop(old_pic, box, old_page["width"]), _crop(new_pic, box, new_page["width"])]
        scale = ZOOM_WIDTH / max(1, sum(part.shape[1] for part in parts))
        parts = [cv2.resize(part, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for part in parts]
        cv2.imwrite(
            str(out_dir / "zoom" / f"{name}.jpg"),
            side_by_side(parts, ["было", "стало"]),
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
    with (out_dir / "compare.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    total = {
        field: (sum(r[f"{field}_before"] for r in rows), sum(r[f"{field}_after"] for r in rows))
        for field in ("pairs", "axes", "blocks", "steps", "crossed_axes", "half_rows")
    }
    summary = ", ".join(f"{field} {old}→{new}" for field, (old, new) in total.items())
    changed = [r["key"] for r in rows if not r["same"]]
    logger.info("Полос %d, изменились %d: %s", len(rows), len(changed), summary)
    for key in changed:
        logger.info("  изменилась: %s", key)


__all__ = ["main", "run_page"]
