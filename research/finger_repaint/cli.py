"""CLI стенда тёмных пятен на месте пальцев: мера по итогам прогона v2 (``select``), листы отбора (``sheets``), точный стенд осветления тени (``edge``), copy back с припуском ядра пальца (``core-margin``)."""

from __future__ import annotations

import csv
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed
from ocr_utils.scan_cropping.finger_removal.text_protection import intersecting_polygons, polygons_to_mask
from ocr_utils.scan_cropping.v2.copy_back import LAYOUT_PAD_PX
from ocr_utils.scan_cropping.v2.pipeline import init_worker, read_layout, to_frame
from research.border_fill.frames import key_of, load_masks, read_image
from research.finger_repaint.core_margin import CORE_GROW_PX, load_margin_inputs, margin_rows
from research.finger_repaint.edge import baseline_path, evaluate_frame, load_edge_inputs
from research.finger_repaint.measure import MEASURE_SCALE, spot_score
from research.finger_repaint.repaint import Variant, load_input, repaint, stage_cache

log = logging.getLogger(__name__)

# Плотность итогов пака-6, px/мм (``--px-per-mm 13.8`` в ``v2_pack6_args.sh``).
PACK6_PX_PER_MM = 13.8
# Цвета (BGR, палитра ``/draw-overlay``): пятно — красный полупрозрачно, граница зоны пальца — оранжевый.
COLOUR_SPOT = (0, 0, 220)
COLOUR_ZONE = (0, 165, 255)
SPOT_ALPHA = 0.4
# Высота миниатюры на листе, px; поле вокруг зоны пальца на вырезке, мм.
THUMB_HEIGHT = 360
ZOOM_MARGIN_MM = 15
# Миниатюр в ряду и рядов на листе.
SHEET_COLS = 4
SHEET_ROWS = 6
# Высота вырезки варианта на листе сравнения, px.
COMPARE_HEIGHT = 520


@dataclass(frozen=True)
class SelectRow:
    """Строка меры тёмного пятна кадра (CSV ``select``)."""

    key: str
    rel: str
    area_mm2: float
    mean_drop: float
    max_drop: float
    zone_mm2: float
    finger_back_mm2: float
    message: str = ""


def zone_box(mask: np.ndarray, margin: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Рамка ``(x0, y0, x1, y1)`` маски масштаба меры, переведённая в полный масштаб, с полем ``margin`` px.

    Args:
        mask: Маска на масштабе ``MEASURE_SCALE``.
        margin: Поле, px полного масштаба.
        shape: ``(H, W)`` полного масштаба (рамка обрезается по нему).

    Returns:
        Рамка в px полного масштаба.
    """
    k = 1 / MEASURE_SCALE
    ys, xs = np.nonzero(mask)
    x0, y0 = max(int(xs.min() * k) - margin, 0), max(int(ys.min() * k) - margin, 0)
    x1, y1 = min(int((xs.max() + 1) * k) + margin, shape[1]), min(int((ys.max() + 1) * k) + margin, shape[0])
    return x0, y0, x1, y1


def largest(mask: np.ndarray) -> np.ndarray:
    """Крупнейшая 8-связная компонента маски (bool); пустая маска — как есть."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n < 2:
        return mask
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def spot_view(bgr: np.ndarray, zone: np.ndarray, spots: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Вырезка итога ``box`` высотой ``THUMB_HEIGHT``: как есть | с пятнами (полупрозрачно) и границей зоны пальца.

    Args:
        bgr: Итоговый прямоугольник BGR.
        zone: Зона пальца на бумаге, масштаб меры.
        spots: Пятна, масштаб меры.
        box: Рамка вырезки, px полного масштаба.

    Returns:
        Картинка BGR.
    """
    x0, y0, x1, y1 = box
    crop = bgr[y0:y1, x0:x1]
    f = THUMB_HEIGHT / crop.shape[0]
    plain = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    size = (plain.shape[1], plain.shape[0])
    s = MEASURE_SCALE
    sy0, sy1, sx0, sx1 = int(y0 * s), int(y1 * s), int(x0 * s), int(x1 * s)
    marked = plain.copy()
    layer = marked.copy()
    spot = cv2.resize(spots[sy0:sy1, sx0:sx1].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    layer[spot] = COLOUR_SPOT
    cv2.addWeighted(layer, SPOT_ALPHA, marked, 1 - SPOT_ALPHA, 0, marked)
    z = cv2.resize(zone[sy0:sy1, sx0:sx1].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours(z, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(marked, cs, -1, COLOUR_ZONE, 1)
    return np.hstack([plain, np.full((plain.shape[0], 6, 3), 255, np.uint8), marked])


@dataclass(frozen=True)
class FrameInputs:
    """Итог кадра и маски в координатах итогового прямоугольника.

    Поля:
        key: Ключ кэша.
        rel: Путь итога относительно корня итогов.
        img: Итог BGR.
        paint: Зона закраса пальца.
        core: Ядро пальца (сам палец по SAM).
        returned: Возвращено с оригинала copy back блоков surya (блоки, задетые зоной пальца, внутри «не фона»).
        visible: Бумага ∪ защита контента.
    """

    key: str
    rel: Path
    img: np.ndarray
    paint: np.ndarray
    core: np.ndarray
    returned: np.ndarray
    visible: np.ndarray


def copy_back_region(cache: Path, paint: np.ndarray) -> np.ndarray:
    """Где финал v2 вернул пиксели с оригинала: блоки surya, задетые зоной пальца (припуск ``LAYOUT_PAD_PX``), внутри
    грубой маски книги — как ``copy_back_blocks`` с ``--copy-back-over-fingers`` (ключ пака-6).

    Args:
        cache: Папка кадра в кэше v2.
        paint: Зона закраса пальца (bool) размера кадра.

    Returns:
        Булева маска размера кадра.
    """
    polys = read_layout(cache / "layout.json") or []
    hits = intersecting_polygons(polys, paint.astype(np.uint8) * 255, LAYOUT_PAD_PX) if polys else []
    if not hits:
        return np.zeros(paint.shape, bool)
    with np.load(cache / "book.npz") as data:
        book = to_frame(data["book"], paint.shape)
    return (polygons_to_mask(paint.shape, hits, LAYOUT_PAD_PX) > 0) & book


def frame_inputs(path: Path, output_root: Path, cache_root: Path) -> FrameInputs:
    """Итог кадра и маски в его координатах: зона и ядро пальца, возвращённое copy back, «бумага ∪ защита».

    Args:
        path: Итоговый TIFF.
        output_root: Корень итогов прогона.
        cache_root: Корень кэша прогона.

    Returns:
        Вход меры.

    Raises:
        ValueError: Бумаги нет или размер итога не совпал с рамкой.
    """
    rel = path.relative_to(output_root)
    key = key_of(rel)
    masks = load_masks(cache_root / key)
    if masks.box is None:
        raise ValueError("нет бумаги")
    with np.load(cache_root / key / "fingers.npz") as data:
        paint_full, core_full = data["paint"], data["core"]
    returned = copy_back_region(cache_root / key, paint_full)
    img = read_image(path)
    visible = masks.crop(masks.paper | masks.protected)
    if img.shape[:2] != visible.shape:
        raise ValueError(f"размер {img.shape[:2]} ≠ рамке {visible.shape}")
    return FrameInputs(key, rel, img, masks.crop(paint_full), masks.crop(core_full), masks.crop(returned), visible)


def select_one(args: tuple[Path, Path, Path, Path]) -> SelectRow:
    """Мера тёмного пятна одного итога (для пула) и миниатюра ``thumbs/<ключ>.jpg`` у крупнейшей зоны.

    Args:
        args: ``(итог, корень итогов, корень кэша, папка выхода)``.

    Returns:
        Строка меры.
    """
    path, output_root, cache_root, out_dir = args
    rel = path.relative_to(output_root)
    key = key_of(rel)
    try:
        inp = frame_inputs(path, output_root, cache_root)
        img = inp.img
        # Мера — только по закрасу LaMa: возвращённое copy back (с пальцем, если блок surya его задел) не меряется.
        score = spot_score(img, inp.paint & ~inp.returned, inp.visible, PACK6_PX_PER_MM)
        # Палец, возвращённый в итог copy back: ядро пальца ∩ возвращённое ∩ видимое, мм².
        finger_back = float((inp.core & inp.returned & inp.visible).sum() / PACK6_PX_PER_MM**2)
        if score.zone.any():
            # Вырезка — вокруг крупнейшего пятна, а без пятен — вокруг крупнейшей компоненты зоны пальца на бумаге.
            target = largest(score.spots if score.spots.any() else score.zone)
            box = zone_box(target, int(ZOOM_MARGIN_MM * PACK6_PX_PER_MM), img.shape[:2])
            view = spot_view(img, score.zone, score.spots, box)
            cv2.imwrite(str(out_dir / "thumbs" / f"{key}.jpg"), view, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return SelectRow(key, str(rel), score.area_mm2, score.mean_drop, score.max_drop, score.zone_mm2, finger_back)
    except Exception as exc:  # кадр с ошибкой не роняет прогон
        return SelectRow(key, str(rel), 0, 0, 0, 0, 0, repr(exc))


@click.group()
def main() -> None:
    """Стенд тёмных пятен LaMa на месте пальцев (scan_cropping v2)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@main.command("select")
@click.option("--run", "run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Папка прогона v2 (output/ и cache/ внутри).")  # fmt: skip
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Куда писать CSV и миниатюры.")  # fmt: skip
@click.option("--jobs", default=16, show_default=True, help="Число процессов.")
def select_cmd(run_dir: Path, out_dir: Path, jobs: int) -> None:
    """Мера тёмного пятна по всем итогам прогона: ``spots.csv`` и миниатюры ``thumbs/``."""
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
    with open(out_dir / "spots.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SelectRow.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)
    with_zone = sum(r.zone_mm2 > 0 for r in rows)
    log.info("готово: %d кадров, с пальцем на бумаге %d, с пятном %d, с сообщением %d", len(rows), with_zone,
             sum(r.area_mm2 > 0 for r in rows), sum(bool(r.message) for r in rows))  # fmt: skip


def pad_to(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """Картинка, добитая белым полем справа и снизу до ``height × width``."""
    out = np.full((height, width, 3), 255, np.uint8)
    out[: img.shape[0], : img.shape[1]] = img[:height, :width]
    return out


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
            img = np.full((THUMB_HEIGHT, THUMB_HEIGHT, 3), 255, np.uint8)
        caption = f"{r['rel']}  {float(r['area_mm2']):.0f} мм², провал {float(r['mean_drop']):.0f}"
        cells.append(framed(img, [caption], []))
    height = max(c.shape[0] for c in cells)
    width = max(c.shape[1] for c in cells)
    cells = [pad_to(c, height, width) for c in cells]
    while len(cells) % SHEET_COLS:
        cells.append(np.full_like(cells[0], 255))
    pad = np.full((height, 8, 3), 255, np.uint8)
    rows_img = [
        np.hstack([x for c in cells[i : i + SHEET_COLS] for x in (c, pad)]) for i in range(0, len(cells), SHEET_COLS)
    ]
    legend = [
        LegendEntry("тёмное пятно: L ниже кольца бумаги вокруг на 8+", COLOUR_SPOT, alpha=SPOT_ALPHA),
        LegendEntry("граница зоны закраса пальца на бумаге", COLOUR_ZONE),
    ]
    return framed(np.vstack(rows_img), [title, "в каждой ячейке: итог как есть | с разметкой"], legend)


@main.command("sheets")
@click.option("--select-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Выход ``select``.")  # fmt: skip
@click.option("--min-area", default=20.0, show_default=True, help="Нижняя граница меры, мм².")
@click.option("--max-area", default=1e9, show_default=True, help="Верхняя граница меры, мм².")
@click.option("--name", default="sheets", show_default=True, help="Подпапка листов.")
def sheets_cmd(select_dir: Path, min_area: float, max_area: float, name: str) -> None:
    """Листы миниатюр кадров с мерой пятна в поясе ``[min, max)``, по убыванию меры: ``<name>/NN.jpg``."""
    with open(select_dir / "spots.csv") as f:
        rows = [r for r in csv.DictReader(f) if min_area <= float(r["area_mm2"]) < max_area]
    out = select_dir / name
    out.mkdir(exist_ok=True)
    per = SHEET_COLS * SHEET_ROWS
    for n, i in enumerate(range(0, len(rows), per), 1):
        part = rows[i : i + per]
        title = f"Тёмные пятна на месте пальцев, кадры {i + 1}–{i + len(part)} из {len(rows)} (мера {min_area:g}–{max_area:g} мм²)"
        cv2.imwrite(str(out / f"{n:02d}.jpg"), sheet_page(part, select_dir / "thumbs", title),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])  # fmt: skip
    log.info("листов: %d, кадров: %d", -(-len(rows) // per), len(rows))


def read_set(path: Path) -> list[Path]:
    """Кадры множества: по строке на кадр, путь относительно корня пака; пустые строки и ``#`` — пропуск."""
    rows = [line.split("#")[0].strip() for line in path.read_text().splitlines()]
    return [Path(r) for r in rows if r]


def pack6_params(src_root: Path, run_dir: Path) -> "RunParams":
    """Параметры финала v2 как у полного прогона пака-6 (``v2_pack6_args.sh``), папки — варианта стенда.

    Args:
        src_root: Корень пака (входные кадры, только чтение).
        run_dir: Папка варианта (``output``, ``cache``, ``debug`` внутри).

    Returns:
        Параметры прогона.
    """
    from ocr_utils.scan_cropping.v2.fill import FillMethod
    from ocr_utils.scan_cropping.v2.coons_layout import LayoutWarp
    from ocr_utils.scan_cropping.v2.pipeline import RunParams
    from ocr_utils.scan_cropping.v2.scale import Scale

    return RunParams(
        input_dir=src_root,
        output_dir=run_dir / "output",
        work_dir=run_dir / "cache",
        debug_dir=run_dir / "debug",
        scale=Scale(PACK6_PX_PER_MM),
        fill=FillMethod.PULL_PUSH,
        hole_max_px=300,
        light_increment=(20.0, 20.0),
        layout_pad_mm=(1.3043, 2.6087),  # 18, 36 px при 13.8 px/мм
        refine_depth_mm=5.0725,  # 35 px масштаба сегментации
        ink_fill=True,
        layout_warp=LayoutWarp.KEYSTONE,
        fit_sides=True,
        snap_sides=False,
        copy_back_over_fingers=True,
        fill_before_copy_back=True,
        compensate_levels=True,
        output_format="tiff",
        force_dpi=450,
        save_fingers=True,
        # Базовые прогоны стенда шли без двойной сегментации; с 2026-10-24 она у v2 по умолчанию включена.
        dual_levels=False,
    )


@main.command("run")
@click.option("--base", "base_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Базовый прогон v2 с --save-fingers (cache/<ключ>/clean.png).")  # fmt: skip
@click.option("--src-root", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Корень пака-6 (только чтение).")  # fmt: skip
@click.option("--set", "set_paths", multiple=True, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True, help="Файлы множеств (sets/*.txt).")  # fmt: skip
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Корень вариантов: <out-dir>/<вариант>/{output,cache,debug}.")  # fmt: skip
@click.option(
    "--variants", default=",".join(v.value for v in Variant), show_default=True, help="Варианты через запятую."
)
@click.option("--jobs", default=12, show_default=True, help="Процессов финала.")
def run_cmd(
    base_dir: Path, src_root: Path, set_paths: tuple[Path, ...], out_dir: Path, variants: str, jobs: int
) -> None:
    """Второй проход закраса пальца вариантами по кадрам множеств (GPU, по кадру), затем штатный финал v2 (пул)."""
    from ocr_utils.scan_cropping.gpu_models import GpuModels
    from ocr_utils.scan_cropping.v2.pipeline import FrameJob, cpu_finish
    from ocr_utils.scan_cropping.v2.scale import Scale

    chosen = [Variant(v) for v in variants.split(",")]
    rels = [rel for p in set_paths for rel in read_set(p)]
    scale = Scale(PACK6_PX_PER_MM)
    timing = []
    with GpuModels(with_detection=False, with_lama=True) as models:
        for i, rel in enumerate(rels, 1):
            key = key_of(rel)
            inp = load_input(src_root / rel, base_dir / "cache" / key)
            for v in chosen:
                clean, seconds = repaint(inp, v, models, scale)
                stage_cache(base_dir / "cache" / key, out_dir / v.value / "cache" / key, clean)
                timing.append((key, v.value, seconds))
            log.info("%d/%d %s: второй проход готов", i, len(rels), key)
    with open(out_dir / "timing.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "variant", "lama_seconds"])
        writer.writerows(timing)
    # Финал v2 штатным кодом пакета: закрас до прямоугольника, защита, copy back — по кэшу варианта.
    tasks = [(FrameJob(src_root / rel, rel), pack6_params(src_root, out_dir / v.value)) for v in chosen for rel in rels]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        stats = list(pool.map(cpu_finish, tasks))
    log.info("финал: %d кадров, не ok: %s", len(stats), [(s.key, s.status) for s in stats if s.status != "ok"])


def crop_at(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Вырезка ``box`` высотой ``COMPARE_HEIGHT`` (итоги вариантов одного размера — рамка общая)."""
    x0, y0, x1, y1 = box
    crop = img[y0:y1, x0:x1]
    f = COMPARE_HEIGHT / crop.shape[0]
    return cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


# Подписи вариантов на листах «было | стало».
VARIANT_CAPTIONS = {
    "A_v2": "как сейчас",
    Variant.REPAINT.value: "повторный LaMa по «палец ∩ книга», ROI как сейчас",
    Variant.TIGHT.value: "то же, тесный ROI",
    Variant.NO_MAT.value: "B без коврика в контексте",
    Variant.TONE.value: "A + тоновая поправка",
    Variant.REPAINT_TONE.value: "B + тоновая поправка",
    Variant.TONE_FINE.value: "A + мелкая тоновая поправка",
    Variant.SYM_HOLE.value: "LaMa по круговой зоне 3× припуска, вклейка только в асимметричную зону",
    Variant.SYM_HOLE_2X.value: "то же, круг 2× припуска",
    Variant.SYM_PASTE_ALL.value: "как H, но вклейка во весь круг, буквы возвращаются с оригинала",
    Variant.SYM_NO_LIGHT.value: "как J, без высветления зоны пальца перед LaMa",
}


@main.command("compare")
@click.option("--base", "base_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Базовый прогон (итог A).")  # fmt: skip
@click.option("--runs", "runs_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Корень вариантов ``run``.")  # fmt: skip
@click.option("--set", "set_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Файл множества.")  # fmt: skip
@click.option(
    "--variants", default=",".join(v.value for v in Variant), show_default=True, help="Варианты через запятую."
)
@click.option("--suffix", default="", help="Добавка к имени папки листов (compare/<множество><suffix>).")
def compare_cmd(base_dir: Path, runs_dir: Path, set_path: Path, variants: str, suffix: str) -> None:
    """Листы «было | стало»: на кадр — ряд вырезок A | варианты у крупнейшего пятна A (или зоны пальца), ``compare/<множество>/NN.jpg``."""
    names = ["A_v2"] + variants.split(",")
    roots = [base_dir] + [runs_dir / v for v in variants.split(",")]
    out = runs_dir / "compare" / f"{set_path.stem}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    strips = []
    for rel in read_set(set_path):
        tiff = rel.with_suffix(".tiff")
        try:
            inp = frame_inputs(base_dir / "output" / tiff, base_dir / "output", base_dir / "cache")
        except Exception as exc:  # кадр без итога — пропуск с записью в лог
            log.warning("%s: %r", rel, exc)
            continue
        score = spot_score(inp.img, inp.paint & ~inp.returned, inp.visible, PACK6_PX_PER_MM)
        target = largest(score.spots if score.spots.any() else score.zone)
        if not target.any():
            continue
        box = zone_box(target, int(ZOOM_MARGIN_MM * PACK6_PX_PER_MM), inp.img.shape[:2])
        cells = []
        for name, root in zip(names, roots):
            img = inp.img if root == base_dir else read_image(root / "output" / tiff)
            s = spot_score(img, inp.paint & ~inp.returned, inp.visible, PACK6_PX_PER_MM)
            cells.append(
                framed(crop_at(img, box), [f"{name}: пятно {s.area_mm2:.0f} мм², провал {s.mean_drop:.0f}"], [])
            )
        height = max(c.shape[0] for c in cells)
        row = np.hstack(
            [x for c in cells for x in (pad_to(c, height, c.shape[1]), np.full((height, 8, 3), 255, np.uint8))]
        )
        strips.append(framed(row, [str(rel)], []))
    per = 4
    for n, i in enumerate(range(0, len(strips), per), 1):
        part = strips[i : i + per]
        width = max(p.shape[1] for p in part)
        sheet = np.vstack([pad_to(p, p.shape[0], width) for p in part])
        title = f"Тёмные пятна на месте пальца: {set_path.stem}, кадры {i + 1}–{i + len(part)} из {len(strips)}"
        subs = [f"{name} — {VARIANT_CAPTIONS.get(name, name)}" for name in names]
        cv2.imwrite(str(out / f"{n:02d}.jpg"), framed(sheet, [title, *subs], []), [cv2.IMWRITE_JPEG_QUALITY, 88])
    log.info("листов: %d, кадров: %d", -(-len(strips) // per), len(strips))


def reach_one(args: tuple[Path, Path, Path]) -> np.ndarray | None:
    """Профиль тени одного итога (для пула): медиана L бумаги по кольцам 1 мм от ядра пальца, или None."""
    from research.finger_repaint.shadow import load_shadow_inputs, reach_profile

    path, output_root, cache_root = args
    try:
        return reach_profile(load_shadow_inputs(path, output_root, cache_root), PACK6_PX_PER_MM)
    except ValueError as exc:
        log.warning("%s: %r", path, exc)
        return None


@main.command("reach")
@click.option("--base", "base_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Прогон (output, cache).")  # fmt: skip
@click.option("--set", "set_paths", multiple=True, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True, help="Файлы множеств.")  # fmt: skip
@click.option("--jobs", default=12, show_default=True, help="Процессов.")
def reach_cmd(base_dir: Path, set_paths: tuple[Path, ...], jobs: int) -> None:
    """Профиль тени по расстоянию от ядра пальца: провал L бумаги против дальних колец (30–40 мм), по кадрам и медиана."""
    rels = [rel for p in set_paths for rel in read_set(p)]
    tasks = [(base_dir / "output" / rel.with_suffix(".tiff"), base_dir / "output", base_dir / "cache") for rel in rels]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        profiles = [p for p in pool.map(reach_one, tasks) if p is not None]
    # Провал кольца против дальней бумаги того же кадра (медиана колец 30–40 мм).
    drops = np.array([np.nanmedian(p[30:40]) - p for p in profiles])
    med = np.nanmedian(drops, axis=0)
    p90 = np.nanpercentile(drops, 90, axis=0)
    for mm in range(drops.shape[1]):
        log.info(
            "%2d мм: провал L медиана %5.1f, 90%% %5.1f, кадров %d",
            mm,
            med[mm],
            p90[mm],
            np.isfinite(drops[:, mm]).sum(),
        )


@dataclass(frozen=True)
class ShadowJob:
    """Задание оценки тени одного кадра (для пула).

    Поля:
        rel: Путь итога относительно корня итогов (``.tiff``).
        source_dir: Прогон-источник (``output`` внутри): итог, который осветляется (A или J).
        ref_dir: Прогон A (``output``, ``cache``): опора меры и маски.
        out_dir: Корень вариантов: ``<out_dir>/<источник>+<оценщик>_<режим>/output``.
        source_name: Имя источника в имени варианта.
        methods: Оценщики.
        also: Готовые варианты (папки прогонов), которые только меряются по той же области.
    """

    rel: Path
    source_dir: Path
    ref_dir: Path
    out_dir: Path
    source_name: str
    methods: tuple[str, ...]
    also: tuple[Path, ...]


def shadow_variant_name(source_name: str, method: str, mode: str) -> str:
    """Имя варианта осветления: ``<источник>+<оценщик>_<режим>``."""
    return f"{source_name}+{method}_{mode}"


def write_tiff(path: Path, img: np.ndarray) -> None:
    """Пишет TIFF (LZW) с путём в юникоде, создавая папки."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".tiff", img, [cv2.IMWRITE_TIFF_COMPRESSION, 5])
    if not ok:
        raise ValueError(f"не закодировать {path}")
    data.tofile(str(path))


def shadow_frame(job: ShadowJob, models=None) -> list[dict]:
    """Оценка тени кадра всеми оценщиками задания, осветление в обоих режимах, запись итогов, меры.

    Args:
        job: Задание.
        models: ``GpuModels`` с LaMa — только если среди оценщиков ``T2_lama``.

    Returns:
        Строки меры: по строке на вариант (источник как есть, готовые варианты ``also``, осветлённые).
    """
    from research.finger_repaint.shadow import (
        LightenMode,
        ShadowMethod,
        estimate,
        lighten,
        load_shadow_inputs,
        shadow_score,
    )

    key = key_of(job.rel)
    ref = load_shadow_inputs(job.ref_dir / "output" / job.rel, job.ref_dir / "output", job.ref_dir / "cache")
    src = read_image(job.source_dir / "output" / job.rel)
    inp = type(ref)(ref.key, src, ref.paint, ref.core, ref.book, ref.original)
    rows = []
    masks_map = None
    for method in job.methods:
        smap = estimate(inp, ShadowMethod(method), PACK6_PX_PER_MM, models)
        masks_map = smap if masks_map is None else masks_map
        for mode in LightenMode:
            name = shadow_variant_name(job.source_name, method, mode.value)
            out = lighten(src, smap, mode)
            write_tiff(job.out_dir / name / "output" / job.rel, out)
            score = shadow_score(ref.img, out, smap, PACK6_PX_PER_MM)
            rows.append({"key": key, "variant": name, "seconds": smap.seconds, **asdict(score)})
            if mode == LightenMode.GAIN:
                write_map(job.out_dir / name / "maps" / f"{key}.jpg", out, smap)
    if masks_map is None:
        masks_map = estimate(inp, ShadowMethod.PULL_PUSH, PACK6_PX_PER_MM)
    # Источник как есть и готовые варианты — по той же области поиска.
    for name, img in [(job.source_name, src)] + [(d.name, read_image(d / "output" / job.rel)) for d in job.also]:
        score = shadow_score(ref.img, img, masks_map, PACK6_PX_PER_MM)
        rows.append({"key": key, "variant": name, "seconds": 0.0, **asdict(score)})
    return rows


def write_map(path: Path, img: np.ndarray, smap) -> None:
    """Картинка карты: итог на ¼, осветление (доля, синим цветом полупрозрачно), контур области поиска, легенда в поле."""
    small = cv2.resize(img, (smap.area.shape[1], smap.area.shape[0]), interpolation=cv2.INTER_AREA)
    gain = np.exp(smap.log_gain.max(axis=2)) - 1
    heat = cv2.applyColorMap(np.clip(gain / 0.3 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = np.clip(gain / 0.05, 0, 1)[..., None] * 0.55
    canvas = (small * (1 - alpha) + heat * alpha).astype(np.uint8)
    for mask, color in ((smap.area, (0, 150, 0)), (smap.skin, (180, 105, 255))):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, contours, -1, color, 1)
    legend = [
        LegendEntry("область поиска тени", (0, 150, 0)),
        LegendEntry("кожа (не осветляется)", (180, 105, 255)),
        LegendEntry(
            "осветление: синий — мало, красный — +30 % и больше", (255, 0, 0), alpha=0.55, style=SampleStyle.BOX
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), framed(canvas, [path.stem], legend), [cv2.IMWRITE_JPEG_QUALITY, 85])


def shadow_frame_safe(job: ShadowJob) -> list[dict]:
    """``shadow_frame`` для пула: ошибка кадра — в лог, пустой список."""
    try:
        return shadow_frame(job)
    except ValueError as exc:
        log.warning("%s: %r", job.rel, exc)
        return []


@main.command("shadow")
@click.option("--ref", "ref_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Прогон A (output, cache): маски и опора меры.")  # fmt: skip
@click.option("--source", "source_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Прогон-источник, итог которого осветляется (A или J).")  # fmt: skip
@click.option("--source-name", required=True, help="Имя источника в именах вариантов (A_v2, J).")
@click.option("--set", "set_paths", multiple=True, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True, help="Файлы множеств.")  # fmt: skip
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True, help="Корень вариантов.")
@click.option("--methods", default="T1_telea,P_pull_push,Q_robust,S_surface", show_default=True,
              help="Оценщики через запятую (T2_lama — на GPU, последовательно).")  # fmt: skip
@click.option("--also", multiple=True, type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Готовые варианты (папки прогонов), которые только меряются.")  # fmt: skip
@click.option("--jobs", default=12, show_default=True, help="Процессов (оценщики на CPU).")
def shadow_cmd(
    ref_dir: Path,
    source_dir: Path,
    source_name: str,
    set_paths: tuple[Path, ...],
    out_dir: Path,
    methods: str,
    also: tuple[Path, ...],
    jobs: int,
) -> None:
    """Карта тени и осветление итога оценщиками; меры в ``<out-dir>/shadow_<источник>.csv``."""
    import pandas as pd

    chosen = tuple(methods.split(","))
    cpu = tuple(m for m in chosen if m != "T2_lama")
    rels = [rel.with_suffix(".tiff") for p in set_paths for rel in read_set(p)]
    jobs_list = [ShadowJob(rel, source_dir, ref_dir, out_dir, source_name, cpu, also) for rel in rels]
    rows: list[dict] = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for part in pool.map(shadow_frame_safe, jobs_list):
            rows += part
    if "T2_lama" in chosen:
        from ocr_utils.scan_cropping.gpu_models import GpuModels

        with GpuModels(with_detection=False, with_lama=True) as models:
            for rel in rels:
                job = ShadowJob(rel, source_dir, ref_dir, out_dir, source_name, ("T2_lama",), ())
                rows += [r for r in shadow_frame(job, models) if "+" in r["variant"]]
    table = pd.DataFrame(rows)
    sets = {key_of(rel.with_suffix(".tiff")): p.stem for p in set_paths for rel in read_set(p)}
    table["set"] = table["key"].map(sets)
    table.to_csv(out_dir / f"shadow_{source_name}.csv", index=False)
    summary = table.groupby(["set", "variant"]).agg(
        residual_sum=("residual_mm2", "sum"),
        residual_median=("residual_mm2", "median"),
        frames_ge50=("residual_mm2", lambda v: int((v >= 50).sum())),
        over_sum=("over_mm2", "sum"),
        max_grad_p90=("max_grad", lambda v: float(np.percentile(v, 90))),
        ink_contrast=("ink_contrast", "median"),
        seconds=("seconds", "mean"),
    )
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        log.info("\n%s", summary.round(2))


@dataclass(frozen=True)
class EdgeJob:
    """Кадр точного стенда тени.

    Поля:
        cache_dir: Папка кадра в кэше прогона v2 с ``--save-before-shadow``.
        px_per_mm: Плотность итога, px/мм.
        baseline_root: Корень итогов прежнего прогона («было») или None.
        sheets_dir: Папка листов «до | было | стало» или None.
    """

    cache_dir: Path
    px_per_mm: float
    baseline_root: Path | None
    sheets_dir: Path | None


def edge_one(job: EdgeJob) -> list[dict]:
    """Меры и листы одного кадра (для пула); кадр без входа карты — пусто, ошибка — в лог, пусто."""
    try:
        inputs = load_edge_inputs(job.cache_dir)
        if inputs is None:
            return []
        baseline = None
        if job.baseline_root is not None:
            path = baseline_path(job.baseline_root, inputs.key)
            baseline = None if path is None else read_image(path)
        return evaluate_frame(inputs, job.px_per_mm, baseline, job.sheets_dir)
    except Exception:  # кадр с ошибкой не роняет стенд
        log.exception("%s: ошибка", job.cache_dir.name)
        return []


@main.command("edge")
@click.option(
    "--cache",
    "cache_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="--cache прогона v2 с --save-before-shadow (в папках кадров before_shadow.png, shadow_masks.npz).",
)
@click.option(
    "--baseline",
    "baseline_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Итоги прежнего прогона тех же кадров («было»), структура как у --output v2.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Куда писать zones.csv и листы sheets/.",
)
@click.option("--px-per-mm", default=PACK6_PX_PER_MM, show_default=True, help="Плотность итогов, px/мм.")
@click.option("--no-sheets", is_flag=True, help="Без листов — только меры.")
@click.option("--jobs", default=12, show_default=True, help="Процессов.")
def edge_cmd(
    cache_root: Path, baseline_root: Path | None, out_dir: Path, px_per_mm: float, no_sheets: bool, jobs: int
) -> None:
    """Точный стенд осветления тени: карта текущим кодом на входе из кэша; меры каймы и пересвета по зонам, листы."""
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sheets_dir = None if no_sheets else out_dir / "sheets"
    (sheets_dir or out_dir).mkdir(parents=True, exist_ok=True)
    jobs_list = [EdgeJob(d, px_per_mm, baseline_root, sheets_dir) for d in sorted(cache_root.iterdir()) if d.is_dir()]
    rows: list[dict] = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for part in pool.map(edge_one, jobs_list):
            rows += part
    if not rows:
        log.warning("нет кадров с входом карты в %s", cache_root)
        return
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "zones.csv", index=False)
    # Сводка по вариантам: 10 % / медиана / 90 % каждой меры.
    columns = [c for c in table.columns if c not in ("key", "zone")]
    summary = table[columns].quantile([0.1, 0.5, 0.9]).T
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        log.info("зон %d, кадров %d\n%s", len(table), table["key"].nunique(), summary.round(2))


@dataclass(frozen=True)
class MarginJob:
    """Кадр стенда «кайма кожи в copy back».

    Поля:
        cache_dir: Папка кадра в кэше прогона v2 с ``--save-fingers``.
        src_root: Корень исходных кадров (структура как у ``--input-dir`` v2).
        grows: Припуски ядра, px калибровочного кадра.
        px_per_mm: Плотность кадров, px/мм.
    """

    cache_dir: Path
    src_root: Path
    grows: tuple[int, ...]
    px_per_mm: float


def margin_one(job: MarginJob) -> tuple[list[dict], list[np.ndarray]]:
    """Меры и вырезки одного кадра (для пула); без входа или с ошибкой — пусто."""
    from ocr_utils.scan_cropping.v2.scale import Scale

    try:
        source = baseline_path(job.src_root, job.cache_dir.name)
        inputs = None if source is None else load_margin_inputs(job.cache_dir, source)
        if inputs is None:
            return [], []
        return margin_rows(inputs, list(job.grows), Scale(job.px_per_mm))
    except Exception:  # кадр с ошибкой не роняет стенд
        log.exception("%s: ошибка", job.cache_dir.name)
        return [], []


@main.command("core-margin")
@click.option(
    "--cache",
    "cache_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="--cache прогона v2 с --save-fingers (в папках кадров clean.png, layout.json).",
)
@click.option(
    "--src-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Корень исходных кадров (--input-dir того прогона).",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Куда писать margin.csv и листы вырезок.",
)
@click.option(
    "--grows",
    default=f"0,{CORE_GROW_PX},8",
    show_default=True,
    help="Припуски ядра пальца, px калибровочного кадра, через запятую; первый — опорный.",
)
@click.option("--px-per-mm", default=PACK6_PX_PER_MM, show_default=True, help="Плотность кадров, px/мм.")
@click.option("--jobs", default=8, show_default=True, help="Процессов (кадры полного размера — память).")
def core_margin_cmd(cache_root: Path, src_root: Path, out_dir: Path, grows: str, px_per_mm: float, jobs: int) -> None:
    """Copy back без ядра пальца, раздутого на припуск: мера каймы кожи и вырезки «оригинал | закрас | припуски»."""
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out_dir.mkdir(parents=True, exist_ok=True)
    values = tuple(int(v) for v in grows.split(","))
    jobs_list = [MarginJob(d, src_root, values, px_per_mm) for d in sorted(cache_root.iterdir()) if d.is_dir()]
    rows: list[dict] = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for job, (part, crops) in zip(jobs_list, pool.map(margin_one, jobs_list)):
            rows += part
            if crops:
                width = max(c.shape[1] for c in crops)
                sheet = np.vstack([cv2.copyMakeBorder(c, 0, 8, 0, width - c.shape[1], cv2.BORDER_CONSTANT,
                                                      value=(255, 255, 255)) for c in crops])  # fmt: skip
                cv2.imwrite(str(out_dir / f"{job.cache_dir.name}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not rows:
        log.warning("нет кадров с clean.png и разметкой в %s", cache_root)
        return
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "margin.csv", index=False)
    summary = table.groupby("grow_px").agg(
        frames=("key", "nunique"),
        returned_px=("returned_px", "sum"),
        removed_px=("removed_px", "sum"),
        rim_a_median=("rim_a", "median"),
        rim_a_p90=("rim_a", lambda v: float(np.nanpercentile(v, 90))),
        rim_next_a_median=("rim_next_a", "median"),
        rim_next_a_p90=("rim_next_a", lambda v: float(np.nanpercentile(v, 90))),
    )
    with pd.option_context("display.width", 200):
        log.info("кайма кожи по припускам (rim_a — превышение a* над бумагой в 1–3 px за ядром, rim_next_a — сразу за "
                 "припуском):\n%s", summary.round(2))  # fmt: skip
