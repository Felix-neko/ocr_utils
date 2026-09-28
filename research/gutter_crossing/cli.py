"""Команды стенда: ``run`` — пересчёт текстовых блоков по ключам в заданном режиме межколонников с мерой «ось через межколонник», ``compare`` — картинки «было | стало»."""

from __future__ import annotations

import csv
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.page_layout.orientation.detectors.ink_axis import glyph_mask
from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, header_strip, legend_strip
from ocr_utils.page_layout.pack_analysis.final import text_blocks
from ocr_utils.page_layout.pack_analysis.stages import PageTask, load_image, page_key
from ocr_utils.page_layout.text_blocks import RENDER_DPI, WORK_DPI
from ocr_utils.page_layout.text_blocks import overlay as blocks_overlay
from ocr_utils.page_layout.text_blocks.columns import GutterMode
from ocr_utils.page_layout.text_blocks.metrics import gutter_crossings_of, metrics_of
from ocr_utils.page_layout.text_blocks.page import AxisKind
from ocr_utils.page_layout.text_blocks.report import page_json
from ocr_utils.page_layout.text_blocks.sides_overlay import side_by_side

logger = logging.getLogger(__name__)

# Ширина оверлея полосы (как у оверлеев разбора пака: ``pack_analysis.final.OVERLAY_WIDTH``).
OVERLAY_WIDTH = 1600
# Находка «ось через межколонник»: столбец пустоты — полупрозрачной заливкой, отрезок под осью — жирно.
COLOUR_CROSSING = (60, 60, 220)
CROSSING_ALPHA = 0.35
# Вырезка вокруг находок на склейке «было | стало»: поле вокруг (мм) и во сколько раз увеличить.
ZOOM_PAD_MM = 15.0
ZOOM_WIDTH = 1400


def _init_worker() -> None:
    """Воркер пула: без hugepage-пометки и без потоков BLAS/OpenCV (см. .claude/rules/gpu_and_pools.md)."""
    import threadpoolctl

    try:
        np._core.multiarray._set_madvise_hugepage(False)  # type: ignore[attr-defined]
    except AttributeError:
        pass
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)


def read_keys(path: Path | None, pack_dir: Path) -> list[str]:
    """Ключи полос ``год/выпуск/полоса``: из файла (строка на ключ, ``#`` — комментарий) или все полосы разбора.

    Args:
        path: Файл со списком; ``None`` — все ``pages/*.json`` разбора пака.
        pack_dir: Выход разбора пака (``pack1_page_analysis_v3``).

    Returns:
        Ключи в порядке файла или по алфавиту.
    """
    if path is None:
        return sorted(json.loads(p.read_text(encoding="utf-8"))["page"] for p in (pack_dir / "pages").glob("*.json"))
    keys = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            keys.append(line.split(",")[0].strip())
    return keys


def page_record(pack_dir: Path, key: str) -> tuple[dict, dict]:
    """Запись кандидатов полосы (``work/pages``) и итоговый JSON разбора пака.

    Разбору текстовых блоков из записи кандидатов нужны только ``page``, ``size``, ``dpi``,
    ``loose_rules`` и ``rotate_cw`` — всё это есть и в итоговом JSON (поворот — из вердикта
    ориентации, если он применён). Поэтому, когда ``work/pages`` нет (рабочая папка разбора
    почищена), запись собирается из итогового JSON.

    Args:
        pack_dir: Выход разбора пака.
        key: Полоса.

    Returns:
        Пара: запись кандидатов и итоговый JSON.
    """
    name = page_key(key)
    final = json.loads((pack_dir / "pages" / f"{name}.json").read_text(encoding="utf-8"))
    work = pack_dir / "work" / "pages" / f"{name}.json"
    if work.exists():
        return json.loads(work.read_text(encoding="utf-8")), final
    orientation = final.get("orientation") or {}
    record = {
        "page": final["page"],
        "size": final["size"],
        "dpi": final["dpi"],
        "loose_rules": final.get("loose_rules", []),
        "rotate_cw": orientation.get("rotate_cw", 0) if orientation.get("apply") else 0,
    }
    return record, final


def draw_page(analysis, gray300: np.ndarray, crossings: list[dict], title: list[str]) -> np.ndarray:
    """Оверлей полосы: разбор текстовых блоков и находки «ось через межколонник», шапка и легенда в поле.

    Args:
        analysis: ``PageAnalysis`` полосы.
        gray300: Серый рендер ``RENDER_DPI``.
        crossings: Находки :func:`metrics.gutter_crossings_of` (словари ``to_json``).
        title: Строки шапки.

    Returns:
        Картинка BGR.
    """
    scale = OVERLAY_WIDTH / analysis.width
    small = cv2.resize(gray300, (OVERLAY_WIDTH, int(round(analysis.height * scale))), interpolation=cv2.INTER_AREA)
    canvas = blocks_overlay.draw(analysis, small, scale=scale)
    # Столбец пустоты — полупрозрачно, чтобы под ним читались буквы.
    layer = canvas.copy()
    for item in crossings:
        p0 = (int(item["x0"] * scale), int(item["top"] * scale))
        p1 = (int(item["x1"] * scale), int(item["bottom"] * scale))
        cv2.rectangle(layer, p0, p1, COLOUR_CROSSING, -1)
    cv2.addWeighted(layer, CROSSING_ALPHA, canvas, 1 - CROSSING_ALPHA, 0, canvas)
    # Сам перескок под осью — жирный отрезок на высоте оси.
    for item in crossings:
        y = int(item["y"] * scale)
        cv2.line(canvas, (int(item["x0"] * scale), y), (int(item["x1"] * scale), y), COLOUR_CROSSING, 4)
    width = canvas.shape[1]
    entries = [
        LegendEntry("ось через межколонник: пустой столбец", COLOUR_CROSSING, CROSSING_ALPHA, SampleStyle.BOX),
        LegendEntry("ось через межколонник: пустота под осью", COLOUR_CROSSING),
    ]
    return np.vstack(
        [
            header_strip(title, width),
            canvas,
            legend_strip(entries, width, "стенд"),
            legend_strip(blocks_overlay.legend_entries(body_axis=True), width, "текстовые блоки"),
        ]
    )


def crossings_of(payload: dict, gray300: np.ndarray) -> list[dict]:
    """Мера «ось через межколонник» по разбору полосы.

    Маска глифов — по всей краске скана, без запретов разбора: растр и таблицы перегораживают пустой
    столбец, а не продлевают его (в запретах нет осей, поэтому сами они находок не дают).

    Args:
        payload: JSON разбора (:func:`report.page_json`): оси, размер рабочей копии.
        gray300: Серый рендер ``RENDER_DPI``.

    Returns:
        Находки :func:`metrics.gutter_crossings_of` словарями.
    """
    work = cv2.resize(gray300, (payload["width"], payload["height"]), interpolation=cv2.INTER_AREA)
    return [c.to_json() for c in gutter_crossings_of(payload["axes"], glyph_mask(work), WORK_DPI)]


def run_page(key: str, pack_dir: Path, sharpened_dir: Path, out_dir: Path, mode: GutterMode, overlays: bool) -> dict:
    """Пересчитать текстовые блоки полосы тем же входом, что в разборе пака, и замерить пересечения межколонников.

    Вход — ровно как в ``pack_analysis.final.final_page``: запись кандидатов ``work/pages``, объекты
    итогового JSON (растр, таблицы, line art — запреты и барьеры), вторая ось строки.

    Args:
        key: Полоса ``год/выпуск/полоса``.
        pack_dir: Выход разбора пака.
        sharpened_dir: Заострённые сканы.
        out_dir: Выход стенда для этого режима (``pages/``, ``overlays/``).
        mode: Режим межколонников.
        overlays: Писать ли оверлей.

    Returns:
        Строка сводки: ключ, число осей, блоков, межколонников и находок, меры ``metrics.py``.
    """
    name = page_key(key)
    record, final = page_record(pack_dir, key)
    image = load_image(PageTask(key, sharpened_dir / f"{key}.jpg"), record["rotate_cw"])
    analysis, hints = text_blocks(image, record, final["objects"], AxisKind.BODY, mode)
    payload = page_json(analysis)
    payload["key"] = key
    crossings = crossings_of(payload, image.gray_at(RENDER_DPI))
    payload["crossings"] = crossings
    target = out_dir / "pages" / f"{name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    if overlays:
        title = [
            f"{key}   режим межколонников: {mode.value}",
            f"осей {len(analysis.axes)}, блоков {len(analysis.blocks)}, межколонников {len(analysis.gutters)}, "
            f"осей через межколонник {len(crossings)}",
        ]
        picture = draw_page(analysis, image.gray_at(RENDER_DPI), crossings, title)
        (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "overlays" / f"{name}.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
    measures = metrics_of(payload)
    return {
        "key": key,
        "folder": final.get("folder", ""),
        "axes": len(analysis.axes),
        "blocks": len(analysis.blocks),
        "gutters": len(analysis.gutters),
        "crossings": len(crossings),
        "steps": measures.steps,
        "crossed_axes": measures.crossings,
        "escaping": measures.escaping,
        "outside": measures.outside,
        "half_rows": measures.half_rows,
        "ink_share": round(measures.ink_share, 4),
    }


def _run_one(args: tuple) -> dict | None:
    """Обёртка для пула: ошибка одной полосы не рушит прогон."""
    key = args[0]
    try:
        return run_page(*args)
    except Exception:  # noqa: BLE001 — полоса с ошибкой уходит в лог, прогон идёт дальше
        logger.exception("Полоса %s: ошибка разбора", key)
        return None


@click.group()
def main() -> None:
    """Стенд «строки через межколонник»."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@main.command()
@click.option("--pack-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--sharpened-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--keys", "keys_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--mode", type=click.Choice([m.value for m in GutterMode]), default=GutterMode.LEGACY.value)
@click.option("--overlays/--no-overlays", default=True, help="Писать оверлеи полос.")
@click.option("--jobs", default=16, show_default=True, help="Воркеров пула.")
def run(
    pack_dir: Path, sharpened_dir: Path, out_dir: Path, keys_file: Path | None, mode: str, overlays: bool, jobs: int
) -> None:
    """Пересчитать полосы в режиме ``--mode``: ``<out-dir>/<mode>/pages``, ``overlays``, ``summary.csv``."""
    gutter_mode = GutterMode(mode)
    keys = read_keys(keys_file, pack_dir)
    target = out_dir / gutter_mode.value
    target.mkdir(parents=True, exist_ok=True)
    tasks = [(key, pack_dir, sharpened_dir, target, gutter_mode, overlays) for key in keys]
    rows = []
    with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        for done, row in enumerate(pool.map(_run_one, tasks, chunksize=4), start=1):
            if row is not None:
                rows.append(row)
            if done % 200 == 0:
                logger.info("Полос %d / %d", done, len(tasks))
    with (target / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    flagged = sum(1 for row in rows if row["crossings"])
    logger.info("Режим %s: полос %d, с осями через межколонник %d → %s", mode, len(rows), flagged, target)


def measure_page(key: str, pack_dir: Path, sharpened_dir: Path, out_dir: Path) -> tuple[str, int]:
    """Пересчитать меру по сохранённому JSON полосы (оси те же) и записать её обратно.

    Args:
        key: Полоса.
        pack_dir: Выход разбора пака: оттуда поворот полосы, с которым её разбирали.
        sharpened_dir: Заострённые сканы.
        out_dir: Выход стенда для режима.

    Returns:
        Ключ и число находок.
    """
    path = out_dir / "pages" / f"{page_key(key)}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    record, _ = page_record(pack_dir, key)
    image = load_image(PageTask(key, sharpened_dir / f"{key}.jpg"), record["rotate_cw"])
    payload["crossings"] = crossings_of(payload, image.gray_at(RENDER_DPI))
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return key, len(payload["crossings"])


def _measure_one(args: tuple) -> tuple[str, int] | None:
    """Обёртка для пула: ошибка одной полосы не рушит пересчёт."""
    try:
        return measure_page(*args)
    except Exception:  # noqa: BLE001 — полоса с ошибкой уходит в лог
        logger.exception("Полоса %s: ошибка пересчёта меры", args[0])
        return None


@main.command()
@click.option("--pack-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--sharpened-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--mode", default=GutterMode.LEGACY.value, show_default=True)
@click.option("--jobs", default=16, show_default=True, help="Воркеров пула.")
def measure(pack_dir: Path, sharpened_dir: Path, out_dir: Path, mode: str, jobs: int) -> None:
    """Пересчитать меру «ось через межколонник» по готовому прогону (после правки порогов меры), без разбора."""
    target = out_dir / mode
    rows = list(csv.DictReader((target / "summary.csv").open(encoding="utf-8")))
    tasks = [(row["key"], pack_dir, sharpened_dir, target) for row in rows]
    counts: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        for result in pool.map(_measure_one, tasks, chunksize=8):
            if result is not None:
                counts[result[0]] = result[1]
    for row in rows:
        row["crossings"] = counts.get(row["key"], row["crossings"])
    with (target / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    flagged = sum(1 for row in rows if int(row["crossings"]))
    logger.info("Режим %s: мера пересчитана по %d полосам, с находками %d", mode, len(rows), flagged)


def _zoom_box(crossings: list[dict], width: int, height: int) -> tuple[int, int, int, int] | None:
    """Рамка вокруг всех находок полосы с полем ``ZOOM_PAD_MM``; ``None`` — находок нет."""
    if not crossings:
        return None
    pad = ZOOM_PAD_MM * WORK_DPI / 25.4
    x0 = max(0, min(c["x0"] for c in crossings) - 4 * pad)
    x1 = min(width, max(c["x1"] for c in crossings) + 4 * pad)
    y0 = max(0, min(c["top"] for c in crossings) - pad)
    y1 = min(height, max(c["bottom"] for c in crossings) + pad)
    return int(x0), int(y0), int(x1), int(y1)


def _crop(picture: np.ndarray, box: tuple[int, int, int, int], page_width: int) -> np.ndarray:
    """Вырезка из оверлея по рамке в пикселях рабочей копии (оверлей — с шапкой сверху)."""
    scale = OVERLAY_WIDTH / page_width
    # Над холстом оверлея — шапка в две строки (:func:`draw_page`); её высота та же, что у пустой
    # шапки в две строки той же ширины.
    offset = header_strip(["", ""], picture.shape[1]).shape[0]
    x0, y0, x1, y1 = (int(v * scale) for v in box)
    return picture[offset + y0 : offset + y1, x0:x1]


@main.command()
@click.option("--out-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--before", default=GutterMode.LEGACY.value, show_default=True)
@click.option("--after", default=GutterMode.SEGMENTED.value, show_default=True)
@click.option("--keys", "keys_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
def compare(out_dir: Path, before: str, after: str, keys_file: Path | None) -> None:
    """Склейки «было | стало» и CSV мер по полосам, посчитанным в обоих режимах: ``<out-dir>/compare_<before>_<after>``."""
    old_dir, new_dir = out_dir / before, out_dir / after
    target = out_dir / f"compare_{before}_{after}"
    (target / "pages").mkdir(parents=True, exist_ok=True)
    (target / "zoom").mkdir(parents=True, exist_ok=True)
    old_rows = {row["key"]: row for row in csv.DictReader((old_dir / "summary.csv").open(encoding="utf-8"))}
    new_rows = {row["key"]: row for row in csv.DictReader((new_dir / "summary.csv").open(encoding="utf-8"))}
    keys = read_keys(keys_file, out_dir) if keys_file else sorted(set(old_rows) & set(new_rows))
    rows = []
    for key in keys:
        if key not in old_rows or key not in new_rows:
            continue
        name = page_key(key)
        old_page = json.loads((old_dir / "pages" / f"{name}.json").read_text(encoding="utf-8"))
        new_page = json.loads((new_dir / "pages" / f"{name}.json").read_text(encoding="utf-8"))
        row = {"key": key, "folder": old_rows[key]["folder"]}
        for field in ("crossings", "blocks", "gutters", "steps", "crossed_axes", "escaping", "outside", "half_rows"):
            row[f"{field}_before"] = int(old_rows[key][field])
            row[f"{field}_after"] = int(new_rows[key][field])
        # Выключка по блокам — строкой видов, чтобы видеть, что поменялось.
        row["align_before"] = " ".join(b["alignment"]["kind"] for b in old_page["blocks"])
        row["align_after"] = " ".join(b["alignment"]["kind"] for b in new_page["blocks"])
        row["changed"] = int(
            row["blocks_before"] != row["blocks_after"]
            or row["align_before"] != row["align_after"]
            or len(old_page["axes"]) != len(new_page["axes"])
        )
        rows.append(row)
        old_pic = cv2.imread(str(old_dir / "overlays" / f"{name}.jpg"))
        new_pic = cv2.imread(str(new_dir / "overlays" / f"{name}.jpg"))
        if old_pic is None or new_pic is None:
            continue
        cv2.imwrite(
            str(target / "pages" / f"{name}.jpg"),
            side_by_side([old_pic, new_pic], [f"было: {before}", f"стало: {after}"]),
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )
        # Вырезка вокруг находок «было» (а если их нет — «стало»): то же место на обеих картинках.
        box = _zoom_box(old_page["crossings"] or new_page["crossings"], old_page["width"], old_page["height"])
        if box is None:
            continue
        parts = [_crop(old_pic, box, old_page["width"]), _crop(new_pic, box, new_page["width"])]
        scale = ZOOM_WIDTH / max(1, sum(part.shape[1] for part in parts))
        parts = [cv2.resize(part, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for part in parts]
        cv2.imwrite(
            str(target / "zoom" / f"{name}.jpg"),
            side_by_side(parts, [f"было: {before}", f"стало: {after}"]),
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
    with (target / "compare.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fixed = sum(1 for r in rows if r["crossings_before"] and not r["crossings_after"])
    broke = sum(1 for r in rows if not r["crossings_before"] and r["crossings_after"])
    changed = sum(r["changed"] for r in rows)
    logger.info(
        "Полос %d: исправлено %d, появилось %d, изменились блоки %d → %s", len(rows), fixed, broke, changed, target
    )


def _dropped(page: dict) -> bool:
    """Выпадает ли на полосе хоть один межколонник из запретов прежнего хода (общая часть ломаной по x пуста)."""
    for gutter in page.get("gutters", []):
        points = gutter["points"]
        if max(point[1] for point in points) >= min(point[2] for point in points):
            return True
    return False


def _ends_above_text(page: dict) -> bool:
    """Кончается ли какой-нибудь межколонник выше низа полосы больше чем на 20 мм (под ним может быть короткий фрагмент)."""
    margin = 20.0 * WORK_DPI / 25.4
    return any(gutter["points"][-1][0] < page["height"] - margin for gutter in page.get("gutters", []))


@main.command()
@click.option("--out-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--mode", default=GutterMode.LEGACY.value, show_default=True, help="Прогон, по которому выбирать.")
@click.option(
    "--min-crossings",
    default=2,
    show_default=True,
    help="Проблемная полоса — от стольких осей через межколонник (одна — чаще всего случайная пустота).",
)
@click.option("--seed", default=0, show_default=True)
def select(out_dir: Path, mode: str, min_crossings: int, seed: int) -> None:
    """Проблемное множество P (оси через межколонник) и нормальное N того же размера по слоям: ``problem.txt``, ``normal.txt``.

    Слои N (поровну, сколько наберётся): межколонник выпадает из запретов по наклону, межколонник
    кончается выше низа полосы, таблицы и прочее «не только текст», обычные полосы с межколонником.
    """
    import random

    source = out_dir / mode
    rows = list(csv.DictReader((source / "summary.csv").open(encoding="utf-8")))
    problem = [row["key"] for row in rows if int(row["crossings"]) >= min_crossings]
    clean = [row for row in rows if int(row["crossings"]) == 0 and int(row["gutters"]) > 0]
    layers: dict[str, list[str]] = {"наклон": [], "межколонник_кончается": [], "не_только_текст": [], "обычная": []}
    for row in clean:
        page = json.loads((source / "pages" / f"{page_key(row['key'])}.json").read_text(encoding="utf-8"))
        if _dropped(page):
            layers["наклон"].append(row["key"])
        elif row["folder"].startswith("не_только_текст"):
            layers["не_только_текст"].append(row["key"])
        elif _ends_above_text(page):
            layers["межколонник_кончается"].append(row["key"])
        else:
            layers["обычная"].append(row["key"])
    generator = random.Random(seed)
    quota = max(1, len(problem) // len(layers))
    normal: list[tuple[str, str]] = []
    for name, keys in layers.items():
        normal += [(key, name) for key in generator.sample(keys, min(quota, len(keys)))]
    # Недобор по слоям добирается из обычных полос.
    rest = [key for key in layers["обычная"] if key not in {k for k, _ in normal}]
    normal += [(key, "обычная") for key in generator.sample(rest, max(0, min(len(rest), len(problem) - len(normal))))]
    (out_dir / "problem.txt").write_text("\n".join(problem) + "\n", encoding="utf-8")
    (out_dir / "normal.txt").write_text("\n".join(f"{key},{name}" for key, name in normal) + "\n", encoding="utf-8")
    sizes = ", ".join(f"{name} {len(keys)}" for name, keys in layers.items())
    logger.info("P: %d полос; N: %d полос (всего чистых по слоям: %s) → %s", len(problem), len(normal), sizes, out_dir)
