"""CLI стенда закраса пограничных областей: мера шлейфов и ступеньки шва по прогону v2 (``select``, ``sheets``, ``select-seam``, ``sheets-seam``), варианты закраса по множествам (``run``), листы «варианты в ряд» (``compare``)."""

from __future__ import annotations

import csv
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from enum import Enum
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, framed
from ocr_utils.scan_cropping.v2.pipeline import init_worker
from research.border_fill.cases import load_case, load_case_from_output, read_set
from research.border_fill.frames import key_of, load_masks, read_image
from research.border_fill.measure import MEASURE_SCALE, SEAM_STEP, blot_area, seam_step, trail_score
from research.border_fill.variants import Variant, fill_variant

log = logging.getLogger(__name__)

# Плотность итогов пака-6, px/мм (``--px-per-mm 13.8`` в ``v2_pack6_args.sh``).
PACK6_PX_PER_MM = 13.8
# Цвета (BGR, палитра ``/draw-overlay``): шлейф — красный полупрозрачно, край бумаги — зелёный, защита — оранжевый.
COLOUR_TRAIL = (0, 0, 220)
COLOUR_PAPER = (0, 150, 0)
COLOUR_KEEP = (0, 165, 255)
TRAIL_ALPHA = 0.45
# Ширина миниатюры на листе, px.
THUMB_WIDTH = 480
# Миниатюр в ряду и рядов на листе.
SHEET_COLS = 3
SHEET_ROWS = 5
# Сторона вырезки варианта на листе сравнения, px.
CROP_SIDE = 420
# Цвет резкого шва на миниатюрах (BGR): закрас светлее каймы — красный, темнее — синий.
COLOUR_SEAM_LIGHT = (0, 0, 230)
COLOUR_SEAM_DARK = (230, 80, 0)
# Сторона вырезки вокруг худшего участка шва, мм.
SEAM_CROP_MM = 30.0


# Легенда листов шлейфа.
TRAIL_LEGEND = [
    LegendEntry("шлейф закраса: ΔE > 15 от бумаги у края", COLOUR_TRAIL, alpha=TRAIL_ALPHA),
    LegendEntry("край бумаги (опора закраса)", COLOUR_PAPER),
    LegendEntry("край защиты контента", COLOUR_KEEP),
]


class Focus(str, Enum):
    """По чему выбирается вырезка на листе сравнения."""

    # Самое крупное пятно шлейфа.
    TRAIL = "trail"
    # Самый длинный резкий участок шва.
    SEAM = "seam"


@dataclass(frozen=True)
class SelectRow:
    """Строка меры шлейфа кадра (CSV ``select``)."""

    key: str
    rel: str
    area_mm2: float
    reach_mm: float
    mean_de: float
    fill_mm2: float
    message: str = ""


def thumbnail(bgr: np.ndarray, paper: np.ndarray, keep: np.ndarray, spots: np.ndarray) -> np.ndarray:
    """Миниатюра прямоугольника: шлейф полупрозрачно, край бумаги и защиты — контуром.

    Args:
        bgr: Итоговый прямоугольник BGR (полный масштаб).
        paper: Маска бумаги прямоугольника.
        keep: Маска защиты прямоугольника.
        spots: Маска шлейфа на масштабе ``MEASURE_SCALE``.

    Returns:
        Миниатюра шириной ``THUMB_WIDTH``.
    """
    f = THUMB_WIDTH / bgr.shape[1]
    small = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    size = (small.shape[1], small.shape[0])
    layer = small.copy()
    layer[cv2.resize(spots.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0] = COLOUR_TRAIL
    cv2.addWeighted(layer, TRAIL_ALPHA, small, 1 - TRAIL_ALPHA, 0, small)
    for mask, colour in ((paper, COLOUR_PAPER), (keep, COLOUR_KEEP)):
        m = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(small, cs, -1, colour, 1)
    if not spots.any():
        return small
    # Рядом — увеличенная вырезка вокруг пятна шлейфа (с полем 30 мм), той же высоты, что миниатюра.
    k = 1 / MEASURE_SCALE
    # Вырезка — вокруг самого крупного пятна.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(spots.astype(np.uint8), connectivity=8)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(labels == biggest)
    margin = int(30 * PACK6_PX_PER_MM)
    y0, y1 = max(int(ys.min() * k) - margin, 0), min(int(ys.max() * k) + margin, bgr.shape[0])
    x0, x1 = max(int(xs.min() * k) - margin, 0), min(int(xs.max() * k) + margin, bgr.shape[1])
    zoom = bgr[y0:y1, x0:x1]
    z = min(small.shape[0] / zoom.shape[0], THUMB_WIDTH / zoom.shape[1])
    zoom = cv2.resize(zoom, None, fx=z, fy=z, interpolation=cv2.INTER_AREA)
    m = cv2.resize(spots[int(y0 / k) : int(y1 / k), int(x0 / k) : int(x1 / k)].astype(np.uint8),
                   (zoom.shape[1], zoom.shape[0]), interpolation=cv2.INTER_NEAREST)  # fmt: skip
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(zoom, cs, -1, COLOUR_TRAIL, 1)
    canvas = np.full((small.shape[0], small.shape[1] + 8 + THUMB_WIDTH, 3), 255, np.uint8)
    canvas[:, : small.shape[1]] = small
    canvas[: zoom.shape[0], small.shape[1] + 8 : small.shape[1] + 8 + zoom.shape[1]] = zoom
    return canvas


def select_one(args: tuple[Path, Path, Path, Path]) -> SelectRow:
    """Мера шлейфа одного итога (для пула) и миниатюра в ``thumbs/<ключ>.jpg``.

    Args:
        args: ``(итог, корень итогов, корень кэша, папка выхода)``.

    Returns:
        Строка меры.
    """
    path, output_root, cache_root, out_dir = args
    rel = path.relative_to(output_root)
    key = key_of(rel)
    try:
        masks = load_masks(cache_root / key)
        if masks.box is None:
            return SelectRow(key, str(rel), 0, 0, 0, 0, "нет бумаги")
        img = read_image(path)
        paper, keep = masks.crop(masks.paper), masks.crop(masks.protected)
        if img.shape[:2] != paper.shape:
            return SelectRow(key, str(rel), 0, 0, 0, 0, f"размер {img.shape[:2]} ≠ рамке {paper.shape}")
        score = trail_score(img, paper, keep, PACK6_PX_PER_MM)
        thumb = thumbnail(img, paper, keep, score.spots)
        cv2.imwrite(str(out_dir / "thumbs" / f"{key}.jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return SelectRow(key, str(rel), score.area_mm2, score.reach_mm, score.mean_de, score.fill_mm2)
    except Exception as exc:  # кадр с ошибкой не роняет прогон
        return SelectRow(key, str(rel), 0, 0, 0, 0, repr(exc))


@click.group()
def main() -> None:
    """Стенд закраса пограничных областей scan_cropping v2."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@main.command("select")
@click.option("--run", "run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Папка прогона v2 (output/ и cache/ внутри).")  # fmt: skip
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Куда писать CSV и миниатюры.")  # fmt: skip
@click.option("--jobs", default=16, show_default=True, help="Число процессов.")
def select_cmd(run_dir: Path, out_dir: Path, jobs: int) -> None:
    """Мера шлейфа закраса по всем итогам прогона: ``trails.csv`` и миниатюры ``thumbs/``."""
    output_root, cache_root = run_dir / "output", run_dir / "cache"
    (out_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    paths = sorted(output_root.rglob("*.tiff"))
    log.info("итогов: %d", len(paths))
    rows = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        tasks = [(p, output_root, cache_root, out_dir) for p in paths]
        for i, row in enumerate(pool.map(select_one, tasks, chunksize=4), 1):
            rows.append(row)
            if i % 200 == 0:
                log.info("%d / %d", i, len(paths))
    rows.sort(key=lambda r: -r.area_mm2)
    with open(out_dir / "trails.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SelectRow.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)
    bad = [r for r in rows if r.message]
    log.info("готово: %d кадров, с сообщением %d", len(rows), len(bad))


def sheet_page(keys: list[str], captions: list[str], thumbs: Path, title: str, legend: list[LegendEntry]) -> np.ndarray:
    """Лист миниатюр с подписью над каждой.

    Args:
        keys: Ключи кадров (имена миниатюр).
        captions: Подписи миниатюр.
        thumbs: Папка миниатюр.
        title: Шапка листа.
        legend: Легенда в поле листа.

    Returns:
        Лист BGR со шапкой и легендой в полях.
    """
    cells = []
    for key, caption in zip(keys, captions):
        img = cv2.imread(str(thumbs / f"{key}.jpg"))
        if img is None:
            img = np.full((400, 2 * THUMB_WIDTH + 8, 3), 255, np.uint8)
        if img.shape[1] < 2 * THUMB_WIDTH + 8:
            img = np.hstack([img, np.full((img.shape[0], 2 * THUMB_WIDTH + 8 - img.shape[1], 3), 255, np.uint8)])
        cells.append(framed(img, [caption], []))
    height = max(c.shape[0] for c in cells)
    cells = [np.vstack([c, np.full((height - c.shape[0], c.shape[1], 3), 255, np.uint8)]) for c in cells]
    while len(cells) % SHEET_COLS:
        cells.append(np.full_like(cells[0], 255))
    pad = np.full((height, 8, 3), 255, np.uint8)
    rows_img = [
        np.hstack([x for c in cells[i : i + SHEET_COLS] for x in (c, pad)]) for i in range(0, len(cells), SHEET_COLS)
    ]
    return framed(np.vstack(rows_img), [title], legend)


@main.command("sheets")
@click.option("--select-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Выход ``select``.")  # fmt: skip
@click.option("--min-area", default=50.0, show_default=True, help="Нижняя граница меры, мм².")
def sheets_cmd(select_dir: Path, min_area: float) -> None:
    """Листы миниатюр кадров с мерой шлейфа ≥ ``--min-area``, по убыванию меры: ``sheets/NN.jpg``."""
    with open(select_dir / "trails.csv") as f:
        rows = [r for r in csv.DictReader(f) if float(r["area_mm2"]) >= min_area]
    out = select_dir / "sheets"
    out.mkdir(exist_ok=True)
    per = SHEET_COLS * SHEET_ROWS
    for n, i in enumerate(range(0, len(rows), per), 1):
        part = rows[i : i + per]
        title = f"Шлейфы закраса, кадры {i + 1}–{i + len(part)} из {len(rows)} (мера ≥ {min_area:g} мм²)"
        captions = [f"{r['rel']}  {float(r['area_mm2']):.0f} мм², {float(r['reach_mm']):.0f} мм" for r in part]
        page = sheet_page([r["key"] for r in part], captions, select_dir / "thumbs", title, TRAIL_LEGEND)
        cv2.imwrite(str(out / f"{n:02d}.jpg"), page, [cv2.IMWRITE_JPEG_QUALITY, 85])
    log.info("листов: %d, кадров: %d", -(-len(rows) // per), len(rows))


@dataclass(frozen=True)
class SeamRow:
    """Строка меры шва кадра (CSV ``select-seam``)."""

    key: str
    rel: str
    sharp_mm: float
    edge_mm: float
    p90: float
    trail_mm2: float
    message: str = ""


def seam_thumbnail(bgr: np.ndarray, paper: np.ndarray, step: np.ndarray) -> np.ndarray:
    """Миниатюра прямоугольника с резким швом (закрас светлее каймы — красный, темнее — синий) и вырезка вокруг
    худшего участка ``SEAM_CROP_MM`` на полном масштабе.

    Args:
        bgr: Итоговый прямоугольник BGR (полный масштаб).
        paper: Маска бумаги прямоугольника.
        step: Ступенька шва на масштабе ``MEASURE_SCALE`` (NaN вне края).

    Returns:
        Миниатюра шириной ``THUMB_WIDTH`` и вырезка рядом.
    """
    f = THUMB_WIDTH / bgr.shape[1]
    small = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    size = (small.shape[1], small.shape[0])
    st = np.nan_to_num(step)
    for mask, colour in ((st > SEAM_STEP, COLOUR_SEAM_LIGHT), (st < -SEAM_STEP, COLOUR_SEAM_DARK)):
        m = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
        small[cv2.resize(m, size, interpolation=cv2.INTER_NEAREST) > 0] = colour
    canvas = np.full((small.shape[0], small.shape[1] + 8 + THUMB_WIDTH, 3), 255, np.uint8)
    canvas[:, : small.shape[1]] = small
    # Вырезка: тот же выбор участка, что на листах сравнения (по ½ масштабу), переведённый в полный.
    box = seam_crop(np.abs(st) > SEAM_STEP, (bgr.shape[0] // 2, bgr.shape[1] // 2))
    if box is None:
        return canvas
    x0, y0, x1, y1 = (2 * v for v in box)
    zoom = bgr[y0:y1, x0:x1].copy()
    edge = cv2.morphologyEx(paper[y0:y1, x0:x1].astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    z = min(small.shape[0] / zoom.shape[0], THUMB_WIDTH / zoom.shape[1])
    zoom = cv2.resize(zoom, None, fx=z, fy=z, interpolation=cv2.INTER_AREA)
    # Край бумаги — короткими штрихами слева от линии, чтобы не закрывать сам шов: только отметки по рамке вырезки.
    ys, xs = np.nonzero(cv2.resize(edge, (zoom.shape[1], zoom.shape[0]), interpolation=cv2.INTER_NEAREST))
    for y, x in zip(ys[::40], xs[::40]):
        cv2.circle(zoom, (int(x), int(y)), 1, COLOUR_PAPER, -1)
    canvas[: zoom.shape[0], small.shape[1] + 8 : small.shape[1] + 8 + zoom.shape[1]] = zoom
    return canvas


def select_seam_one(args: tuple[Path, Path, Path, Path]) -> SeamRow:
    """Мера шва одного итога (для пула) и миниатюра в ``thumbs/<ключ>.jpg``.

    Args:
        args: ``(итог, корень итогов, корень кэша, папка выхода)``.

    Returns:
        Строка меры.
    """
    path, output_root, cache_root, out_dir = args
    rel = path.relative_to(output_root)
    key = key_of(rel)
    try:
        masks = load_masks(cache_root / key)
        if masks.box is None:
            return SeamRow(key, str(rel), 0, 0, 0, 0, "нет бумаги")
        img = read_image(path)
        paper, keep = masks.crop(masks.paper), masks.crop(masks.protected)
        if img.shape[:2] != paper.shape:
            return SeamRow(key, str(rel), 0, 0, 0, 0, f"размер {img.shape[:2]} ≠ рамке {paper.shape}")
        seam = seam_step(img, paper, keep, PACK6_PX_PER_MM)
        trail = trail_score(img, paper, keep, PACK6_PX_PER_MM)
        thumb = seam_thumbnail(img, paper, seam.step)
        cv2.imwrite(str(out_dir / "thumbs" / f"{key}.jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return SeamRow(key, str(rel), seam.sharp_mm, seam.edge_mm, seam.p90, trail.area_mm2)
    except Exception as exc:  # кадр с ошибкой не роняет прогон
        return SeamRow(key, str(rel), 0, 0, 0, 0, repr(exc))


@main.command("select-seam")
@click.option("--run", "run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Папка прогона v2 (output/ и cache/ внутри).")  # fmt: skip
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Куда писать CSV и миниатюры.")  # fmt: skip
@click.option("--jobs", default=16, show_default=True, help="Число процессов.")
def select_seam_cmd(run_dir: Path, out_dir: Path, jobs: int) -> None:
    """Ступенька шва «край бумаги → закрас» по всем итогам прогона: ``seam.csv`` и миниатюры ``thumbs/``."""
    output_root, cache_root = run_dir / "output", run_dir / "cache"
    (out_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    paths = sorted(output_root.rglob("*.tiff"))
    log.info("итогов: %d", len(paths))
    rows = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        tasks = [(p, output_root, cache_root, out_dir) for p in paths]
        for i, row in enumerate(pool.map(select_seam_one, tasks, chunksize=4), 1):
            rows.append(row)
            if i % 200 == 0:
                log.info("%d / %d", i, len(paths))
    rows.sort(key=lambda r: -r.sharp_mm)
    with open(out_dir / "seam.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SeamRow.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)
    log.info("готово: %d кадров, с сообщением %d", len(rows), sum(1 for r in rows if r.message))


@main.command("sheets-seam")
@click.option("--select-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Выход ``select-seam``.")  # fmt: skip
@click.option("--bands", default="300,150,75,30", show_default=True,
              help="Нижние границы поясов длины резкого шва, мм, по убыванию.")  # fmt: skip
@click.option("--per-band", default=15, show_default=True, help="Кадров на пояс (случайно, детерминированно).")
def sheets_seam_cmd(select_dir: Path, bands: str, per_band: int) -> None:
    """Листы по поясам длины резкого шва: ``sheets/<пояс>.jpg``, по ``--per-band`` кадров на пояс."""
    with open(select_dir / "seam.csv") as f:
        rows = [r for r in csv.DictReader(f) if not r["message"]]
    out = select_dir / "sheets"
    out.mkdir(exist_ok=True)
    edges = [float(b) for b in bands.split(",")]
    uppers = [float("inf")] + edges[:-1]
    rng = np.random.default_rng(0)
    for lo, hi in zip(edges, uppers):
        band = [r for r in rows if lo <= float(r["sharp_mm"]) < hi]
        pick = [band[i] for i in sorted(rng.choice(len(band), min(per_band, len(band)), replace=False))] if band else []
        if not pick:
            continue
        captions = [f"{r['rel']}  шов {float(r['sharp_mm']):.0f} мм, p90 {float(r['p90']):.0f}" for r in pick]
        title = f"Резкий шов закраса {lo:g}–{hi:g} мм: {len(pick)} из {len(band)} кадров пояса"
        legend = [
            LegendEntry("закрас светлее каймы листа больше чем на 8 уровней", COLOUR_SEAM_LIGHT),
            LegendEntry("закрас темнее каймы листа больше чем на 8 уровней", COLOUR_SEAM_DARK),
            LegendEntry("край бумаги (на вырезке справа — точками)", COLOUR_PAPER),
        ]
        page = sheet_page([r["key"] for r in pick], captions, select_dir / "thumbs", title, legend)
        cv2.imwrite(str(out / f"seam_{lo:04.0f}.jpg"), page, [cv2.IMWRITE_JPEG_QUALITY, 85])
        log.info("пояс %g–%g мм: %d кадров, на листе %d", lo, hi, len(band), len(pick))


@dataclass(frozen=True)
class VariantRow:
    """Строка меры варианта закраса на кадре (CSV ``run``)."""

    key: str
    rel: str
    set: str
    variant: str
    seconds: float
    area_mm2: float
    reach_mm: float
    mean_de: float
    seam_sharp_mm: float = 0.0
    seam_p90: float = 0.0
    blot_k20_mm2: float = 0.0
    blot_k05_mm2: float = 0.0
    message: str = ""


def run_one(args: tuple[Path, str, Path, Path, Path, Path, list[str], Path | None]) -> list[VariantRow]:
    """Все варианты закраса на одном кадре (для пула): мера шлейфа, время, итог в ``<вариант>/<ключ>.jpg`` (½).

    Args:
        args: ``(кадр, имя множества, кэш масок, кэш кадров без пальцев, корень пака, папка выхода, варианты, прогон v2
            для входа по итогу или None)``.

    Returns:
        Строки по вариантам.
    """
    rel, set_name, mask_cache, clean_cache, src_root, out_dir, variants, from_run = args
    key = key_of(rel)
    try:
        if from_run is not None:
            case = load_case_from_output(rel, from_run, PACK6_PX_PER_MM)
        else:
            case = load_case(rel, mask_cache, clean_cache, src_root, PACK6_PX_PER_MM)
    except Exception as exc:  # кадр с ошибкой не роняет прогон
        return [VariantRow(key, str(rel), set_name, "", 0, 0, 0, 0, message=repr(exc))]
    if case is None:
        return [VariantRow(key, str(rel), set_name, "", 0, 0, 0, 0, message="нет бумаги")]
    rows = []
    for name in variants:
        variant = Variant(name)
        t0 = time.perf_counter()
        out = fill_variant(case.inp, variant)
        seconds = time.perf_counter() - t0
        score = trail_score(out, case.inp.paper, case.inp.keep, PACK6_PX_PER_MM)
        seam = seam_step(out, case.inp.paper, case.inp.keep, PACK6_PX_PER_MM)
        blots = blot_area(out, case.inp.paper, case.inp.keep, PACK6_PX_PER_MM)
        half = cv2.resize(out, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / name / f"{key}.jpg"), half, [cv2.IMWRITE_JPEG_QUALITY, 90])
        np.save(out_dir / "spots" / f"{key}__{name}.npy", np.packbits(score.spots, axis=None))
        sharp = np.nan_to_num(np.abs(seam.step)) > SEAM_STEP
        np.save(out_dir / "spots" / f"{key}__{name}__seam.npy", np.packbits(sharp, axis=None))
        rows.append(
            VariantRow(key, str(rel), set_name, name, seconds, score.area_mm2, score.reach_mm, score.mean_de,
                       seam.sharp_mm, seam.p90, *blots)  # fmt: skip
        )
    np.save(out_dir / "spots" / f"{key}__shape.npy", np.array(score.spots.shape))
    return rows


@main.command("run")
@click.option("--set", "set_paths", type=click.Path(exists=True, dir_okay=False, path_type=Path), multiple=True,
              required=True, help="Файл множества (sets/*.txt); можно несколько.")  # fmt: skip
@click.option("--mask-cache", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="Кэш прогона v2 с масками бумаги, защитой, страницами и блоками surya.")  # fmt: skip
@click.option("--clean-cache", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="Кэш прогона v2 с --save-fingers (clean.png, fingers.npz).")  # fmt: skip
@click.option("--src-root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="Корень пака (исходные кадры, только чтение).")  # fmt: skip
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True, help="Куда писать.")
@click.option("--variants", default=",".join(v.value for v in Variant), show_default=True,
              help="Варианты через запятую.")  # fmt: skip
@click.option("--from-run", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="Вход — готовые итоги и маски этого прогона v2 (вместо --mask-cache/--clean-cache/--src-root).")  # fmt: skip
@click.option("--jobs", default=8, show_default=True, help="Число процессов (кадр ~0.5 ГБ ОЗУ на вариант).")
def run_cmd(set_paths, mask_cache: Path | None, clean_cache: Path | None, src_root: Path | None, out_dir: Path,variants: str, from_run: Path | None, jobs: int):  # fmt: skip
    """Варианты закраса по кадрам множеств: ``variants.csv``, итоги вариантов (½) и маски шлейфа.

    Вход — либо кэши прогона с ``--save-fingers`` и корень пака, либо ``--from-run`` (итоги и маски готового прогона).
    """
    if from_run is None and None in (mask_cache, clean_cache, src_root):
        raise click.UsageError("нужны --mask-cache, --clean-cache и --src-root или --from-run")
    names = [v.strip() for v in variants.split(",") if v.strip()]
    for name in names:
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    (out_dir / "spots").mkdir(parents=True, exist_ok=True)
    tasks = [
        (rel, p.stem, mask_cache, clean_cache, src_root, out_dir, names, from_run)
        for p in set_paths
        for rel in read_set(p)
    ]
    log.info("кадров: %d, вариантов: %d", len(tasks), len(names))
    rows = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for i, part in enumerate(pool.map(run_one, tasks), 1):
            rows.extend(part)
            if i % 10 == 0:
                log.info("%d / %d", i, len(tasks))
    with open(out_dir / "variants.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(VariantRow.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)
    summarize(out_dir / "variants.csv")


def summarize(path: Path) -> None:
    """Сводка ``variants.csv`` в лог: по множеству и варианту — медиана и сумма меры, доля кадров со шлейфом, время."""
    import pandas as pd

    d = pd.read_csv(path)
    d = d[d.message.isna()]
    table = d.groupby(["set", "variant"]).agg(
        frames=("key", "count"),
        area_sum=("area_mm2", "sum"),
        area_median=("area_mm2", "median"),
        with_trail=("area_mm2", lambda a: float((a >= 100).mean())),
        reach_median=("reach_mm", "median"),
        mean_de=("mean_de", "mean"),
        seam_sharp_sum=("seam_sharp_mm", "sum"),
        seam_p90_median=("seam_p90", "median"),
        blot_k20_sum=("blot_k20_mm2", "sum"),
        blot_k05_sum=("blot_k05_mm2", "sum"),
        seconds=("seconds", "median"),
    )
    log.info("сводка:\n%s", table.round(2).to_string())
    table.round(3).to_csv(path.with_name("summary.csv"))


def crop_region(spots: np.ndarray, shape_half: tuple[int, int], margin: int) -> tuple[int, int, int, int]:
    """Вырезка (px ½ итога) вокруг самого крупного пятна шлейфа маски ¼ копии.

    Args:
        spots: Маска шлейфа на ¼ масштабе.
        shape_half: Размер ½ итога ``(H, W)``.
        margin: Поле вокруг пятна, px ½ итога.

    Returns:
        ``(x0, y0, x1, y1)``.
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats(spots.astype(np.uint8), connectivity=8)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(labels == biggest)
    h, w = shape_half
    return (max(xs.min() * 2 - margin, 0), max(ys.min() * 2 - margin, 0),
            min(xs.max() * 2 + margin, w), min(ys.max() * 2 + margin, h))  # fmt: skip


@main.command("compare")
@click.option("--run-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Выход ``run``.")  # fmt: skip
@click.option("--by", "by_variant", default=Variant.TELEA_KEEP.value, show_default=True,
              help="Вариант, по пятну которого выбирается вырезка.")  # fmt: skip
@click.option("--variants", default=None, help="Варианты в ряд через запятую (по умолчанию — все из run).")
@click.option("--focus", type=click.Choice([f.value for f in Focus]), default=Focus.TRAIL.value, show_default=True,
              help="Вырезка: вокруг пятна шлейфа или худшего участка шва варианта --by.")  # fmt: skip
@click.option("--sets", "only_sets", default=None, help="Только эти множества, через запятую.")
def compare_cmd(run_dir: Path, by_variant: str, variants: str | None, focus: str, only_sets: str | None) -> None:
    """Листы «варианты в ряд» по кадрам: ``compare/<множество>[_seam]/NN.jpg``, по 5 кадров на лист."""
    import pandas as pd

    d = pd.read_csv(run_dir / "variants.csv")
    d = d[d.message.isna()]
    names = variants.split(",") if variants else list(dict.fromkeys(d.variant))
    focus = Focus(focus)
    for set_name, part in d.groupby("set"):
        if only_sets and set_name not in only_sets.split(","):
            continue
        out = run_dir / "compare" / (set_name if focus == Focus.TRAIL else f"{set_name}_seam")
        out.mkdir(parents=True, exist_ok=True)
        order = "area_mm2" if focus == Focus.TRAIL else "seam_sharp_mm"
        keys = part[part.variant == by_variant].sort_values(order, ascending=False)
        strips = []
        for _, row in keys.iterrows():
            strips.append(compare_strip(run_dir, row, part, names, by_variant, focus))
        what = "пятну шлейфа" if focus == Focus.TRAIL else "худшему участку шва"
        for n, i in enumerate(range(0, len(strips), 5), 1):
            title = (
                f"{set_name}: кадры {i + 1}–{min(i + 5, len(strips))} из {len(strips)}; вырезка по {what} {by_variant}"
            )
            page = stack_strips(strips[i : i + 5])
            legend = [
                LegendEntry("подпись: шлейф мм², его длина мм | резкий шов мм (ступенька > 8) | чёрное Sauvola k=0.05 у края мм²",
                            (0, 0, 0)),
            ]  # fmt: skip
            cv2.imwrite(str(out / f"{n:02d}.jpg"), framed(page, [title], legend), [cv2.IMWRITE_JPEG_QUALITY, 85])
    log.info("листы: %s", run_dir / "compare")


def seam_crop(sharp: np.ndarray, shape_half: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Квадрат ``SEAM_CROP_MM`` (px ½ итога) с центром в середине самого длинного резкого участка шва.

    Args:
        sharp: Маска резкого шва на ¼ масштабе.
        shape_half: Размер ½ итога ``(H, W)``.

    Returns:
        ``(x0, y0, x1, y1)`` или None, если резкого шва нет.
    """
    if not sharp.any():
        return None
    # Соседние пиксели края с разрывами в 1–2 px — один участок.
    joined = cv2.dilate(sharp.astype(np.uint8), np.ones((5, 5), np.uint8))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero((labels == biggest) & sharp)
    # Середина участка — его пиксель, ближайший к центру масс.
    i = int(np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2))
    cy, cx = ys[i] * 2, xs[i] * 2
    half = int(SEAM_CROP_MM * PACK6_PX_PER_MM / 2 / 2)
    h, w = shape_half
    x0, y0 = min(max(cx - half, 0), max(w - 2 * half, 0)), min(max(cy - half, 0), max(h - 2 * half, 0))
    return x0, y0, min(x0 + 2 * half, w), min(y0 + 2 * half, h)


def load_bits(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Булева маска из ``np.packbits`` стенда."""
    return np.unpackbits(np.load(path))[: shape[0] * shape[1]].reshape(shape).astype(bool)


def compare_strip(run_dir: Path, row, part, names: list[str], by_variant: str, focus: Focus) -> np.ndarray:
    """Ряд вырезок одного кадра по вариантам с подписью меры над каждой.

    Args:
        run_dir: Выход ``run``.
        row: Строка кадра (вариант ``by_variant``).
        part: Строки множества.
        names: Варианты в ряд.
        by_variant: Вариант, по пятну (шву) которого выбирается вырезка.
        focus: Вырезка по шлейфу или по шву.

    Returns:
        Ряд BGR.
    """
    key = row.key
    shape = tuple(np.load(run_dir / "spots" / f"{key}__shape.npy"))
    first = cv2.imread(str(run_dir / names[0] / f"{key}.jpg"))
    box = None
    if focus == Focus.SEAM:
        box = seam_crop(load_bits(run_dir / "spots" / f"{key}__{by_variant}__seam.npy", shape), first.shape[:2])
    else:
        spots = load_bits(run_dir / "spots" / f"{key}__{by_variant}.npy", shape)
        if spots.any():
            box = crop_region(spots, first.shape[:2], int(25 * PACK6_PX_PER_MM / 2))
    x0, y0, x1, y1 = box if box is not None else (0, 0, first.shape[1], first.shape[0])
    cells = []
    for name in names:
        img = cv2.imread(str(run_dir / name / f"{key}.jpg"))
        crop = img[y0:y1, x0:x1]
        f = min(CROP_SIDE / crop.shape[1], CROP_SIDE / crop.shape[0])
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        cell = np.full((CROP_SIDE, CROP_SIDE, 3), 255, np.uint8)
        cell[: crop.shape[0], : crop.shape[1]] = crop
        r = part[(part.key == key) & (part.variant == name)].iloc[0]
        caption = (
            f"{name}: {r.area_mm2:.0f} мм², {r.reach_mm:.0f} мм | {r.seam_sharp_mm:.0f} мм | {r.blot_k05_mm2:.0f} мм²"
        )
        cells.append(framed(cell, [caption], []))
    pad = np.full((cells[0].shape[0], 6, 3), 255, np.uint8)
    strip = np.hstack([x for c in cells for x in (c, pad)])
    return framed(strip, [row.rel], [])


def stack_strips(strips: list[np.ndarray]) -> np.ndarray:
    """Ряды друг под другом, по ширине — до самого широкого белым полем."""
    width = max(s.shape[1] for s in strips)
    rows = [np.hstack([s, np.full((s.shape[0], width - s.shape[1], 3), 255, np.uint8)]) for s in strips]
    return np.vstack(rows)
