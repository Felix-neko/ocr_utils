"""CLI стенда закраса пограничных областей: мера шлейфов по прогону v2 (``select``, ``sheets``), варианты закраса по множествам (``run``), листы «варианты в ряд» (``compare``)."""

from __future__ import annotations

import csv
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, framed
from ocr_utils.scan_cropping.v2.pipeline import init_worker
from research.border_fill.cases import load_case, read_set
from research.border_fill.frames import key_of, load_masks, read_image
from research.border_fill.measure import MEASURE_SCALE, trail_score
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


def sheet_page(rows: list[dict], thumbs: Path, title: str) -> np.ndarray:
    """Лист миниатюр с подписью кадра и меры под каждой.

    Args:
        rows: Строки CSV ``select``.
        thumbs: Папка миниатюр.
        title: Шапка листа.

    Returns:
        Лист BGR со шапкой и легендой в полях.
    """
    cells = []
    for r in rows:
        img = cv2.imread(str(thumbs / f"{r['key']}.jpg"))
        if img is None:
            img = np.full((400, 2 * THUMB_WIDTH + 8, 3), 255, np.uint8)
        if img.shape[1] < 2 * THUMB_WIDTH + 8:
            img = np.hstack([img, np.full((img.shape[0], 2 * THUMB_WIDTH + 8 - img.shape[1], 3), 255, np.uint8)])
        caption = f"{r['rel']}  {float(r['area_mm2']):.0f} мм², {float(r['reach_mm']):.0f} мм"
        cells.append(framed(img, [caption], []))
    height = max(c.shape[0] for c in cells)
    cells = [np.vstack([c, np.full((height - c.shape[0], c.shape[1], 3), 255, np.uint8)]) for c in cells]
    while len(cells) % SHEET_COLS:
        cells.append(np.full_like(cells[0], 255))
    pad = np.full((height, 8, 3), 255, np.uint8)
    rows_img = [
        np.hstack([x for c in cells[i : i + SHEET_COLS] for x in (c, pad)]) for i in range(0, len(cells), SHEET_COLS)
    ]
    legend = [
        LegendEntry("шлейф закраса: ΔE > 15 от бумаги у края", COLOUR_TRAIL, alpha=TRAIL_ALPHA),
        LegendEntry("край бумаги (опора закраса)", COLOUR_PAPER),
        LegendEntry("край защиты контента", COLOUR_KEEP),
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
        cv2.imwrite(str(out / f"{n:02d}.jpg"), sheet_page(part, select_dir / "thumbs", title),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])  # fmt: skip
    log.info("листов: %d, кадров: %d", -(-len(rows) // per), len(rows))


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
    message: str = ""


def run_one(args: tuple[Path, str, Path, Path, Path, Path, list[str]]) -> list[VariantRow]:
    """Все варианты закраса на одном кадре (для пула): мера шлейфа, время, итог в ``<вариант>/<ключ>.jpg`` (½).

    Args:
        args: ``(кадр, имя множества, кэш масок, кэш кадров без пальцев, корень пака, папка выхода, варианты)``.

    Returns:
        Строки по вариантам.
    """
    rel, set_name, mask_cache, clean_cache, src_root, out_dir, variants = args
    key = key_of(rel)
    try:
        case = load_case(rel, mask_cache, clean_cache, src_root, PACK6_PX_PER_MM)
    except Exception as exc:  # кадр с ошибкой не роняет прогон
        return [VariantRow(key, str(rel), set_name, "", 0, 0, 0, 0, repr(exc))]
    if case is None:
        return [VariantRow(key, str(rel), set_name, "", 0, 0, 0, 0, "нет бумаги")]
    rows = []
    for name in variants:
        variant = Variant(name)
        t0 = time.perf_counter()
        out = fill_variant(case.inp, variant)
        seconds = time.perf_counter() - t0
        score = trail_score(out, case.inp.paper, case.inp.keep, PACK6_PX_PER_MM)
        half = cv2.resize(out, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / name / f"{key}.jpg"), half, [cv2.IMWRITE_JPEG_QUALITY, 90])
        np.save(out_dir / "spots" / f"{key}__{name}.npy", np.packbits(score.spots, axis=None))
        rows.append(VariantRow(key, str(rel), set_name, name, seconds, score.area_mm2, score.reach_mm, score.mean_de))
    np.save(out_dir / "spots" / f"{key}__shape.npy", np.array(score.spots.shape))
    return rows


@main.command("run")
@click.option("--set", "set_paths", type=click.Path(exists=True, dir_okay=False, path_type=Path), multiple=True,
              required=True, help="Файл множества (sets/*.txt); можно несколько.")  # fmt: skip
@click.option("--mask-cache", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Кэш прогона v2 с масками бумаги, защитой, страницами и блоками surya.")  # fmt: skip
@click.option("--clean-cache", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Кэш прогона v2 с --save-fingers (clean.png, fingers.npz).")  # fmt: skip
@click.option("--src-root", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Корень пака (исходные кадры, только чтение).")  # fmt: skip
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True, help="Куда писать.")
@click.option("--variants", default=",".join(v.value for v in Variant), show_default=True,
              help="Варианты через запятую.")  # fmt: skip
@click.option("--jobs", default=8, show_default=True, help="Число процессов (кадр ~0.5 ГБ ОЗУ на вариант).")
def run_cmd(set_paths, mask_cache: Path, clean_cache: Path, src_root: Path, out_dir: Path, variants: str, jobs: int):
    """Варианты закраса по кадрам множеств: ``variants.csv``, итоги вариантов (½) и маски шлейфа."""
    names = [v.strip() for v in variants.split(",") if v.strip()]
    for name in names:
        (out_dir / name).mkdir(parents=True, exist_ok=True)
    (out_dir / "spots").mkdir(parents=True, exist_ok=True)
    tasks = [(rel, p.stem, mask_cache, clean_cache, src_root, out_dir, names) for p in set_paths for rel in read_set(p)]
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
def compare_cmd(run_dir: Path, by_variant: str, variants: str | None) -> None:
    """Листы «варианты в ряд» по кадрам: ``compare/<множество>/NN.jpg``, по 5 кадров на лист."""
    import pandas as pd

    d = pd.read_csv(run_dir / "variants.csv")
    d = d[d.message.isna()]
    names = variants.split(",") if variants else list(dict.fromkeys(d.variant))
    for set_name, part in d.groupby("set"):
        out = run_dir / "compare" / set_name
        out.mkdir(parents=True, exist_ok=True)
        keys = part[part.variant == by_variant].sort_values("area_mm2", ascending=False)
        strips = []
        for _, row in keys.iterrows():
            strips.append(compare_strip(run_dir, row, part, names, by_variant))
        for n, i in enumerate(range(0, len(strips), 5), 1):
            title = (
                f"{set_name}: кадры {i + 1}–{min(i + 5, len(strips))} из {len(strips)}; вырезка по пятну {by_variant}"
            )
            page = stack_strips(strips[i : i + 5])
            legend = [LegendEntry("шлейф закраса: ΔE > 15 от бумаги у края", COLOUR_TRAIL)]
            cv2.imwrite(str(out / f"{n:02d}.jpg"), framed(page, [title], legend), [cv2.IMWRITE_JPEG_QUALITY, 85])
    log.info("листы: %s", run_dir / "compare")


def compare_strip(run_dir: Path, row, part, names: list[str], by_variant: str) -> np.ndarray:
    """Ряд вырезок одного кадра по вариантам с подписью меры над каждой.

    Args:
        run_dir: Выход ``run``.
        row: Строка кадра (вариант ``by_variant``).
        part: Строки множества.
        names: Варианты в ряд.
        by_variant: Вариант, по пятну которого выбирается вырезка.

    Returns:
        Ряд BGR.
    """
    key = row.key
    shape = tuple(np.load(run_dir / "spots" / f"{key}__shape.npy"))
    first = cv2.imread(str(run_dir / names[0] / f"{key}.jpg"))
    spots = np.unpackbits(np.load(run_dir / "spots" / f"{key}__{by_variant}.npy"))[: shape[0] * shape[1]]
    spots = spots.reshape(shape).astype(bool)
    if spots.any():
        x0, y0, x1, y1 = crop_region(spots, first.shape[:2], int(25 * PACK6_PX_PER_MM / 2))
    else:
        x0, y0, x1, y1 = 0, 0, first.shape[1], first.shape[0]
    cells = []
    for name in names:
        img = cv2.imread(str(run_dir / name / f"{key}.jpg"))
        crop = img[y0:y1, x0:x1]
        f = min(CROP_SIDE / crop.shape[1], CROP_SIDE / crop.shape[0])
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        cell = np.full((CROP_SIDE, CROP_SIDE, 3), 255, np.uint8)
        cell[: crop.shape[0], : crop.shape[1]] = crop
        r = part[(part.key == key) & (part.variant == name)].iloc[0]
        cells.append(framed(cell, [f"{name}: {r.area_mm2:.0f} мм², {r.reach_mm:.0f} мм, {r.seconds:.1f} с"], []))
    pad = np.full((cells[0].shape[0], 6, 3), 255, np.uint8)
    strip = np.hstack([x for c in cells for x in (c, pad)])
    return framed(strip, [row.rel], [])


def stack_strips(strips: list[np.ndarray]) -> np.ndarray:
    """Ряды друг под другом, по ширине — до самого широкого белым полем."""
    width = max(s.shape[1] for s in strips)
    rows = [np.hstack([s, np.full((s.shape[0], width - s.shape[1], 3), 255, np.uint8)]) for s in strips]
    return np.vstack(rows)
