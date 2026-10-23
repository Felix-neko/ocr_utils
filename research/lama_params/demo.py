"""Наглядный разбор параметров закраса LaMa: синтетическая полоса, закрасы по сеткам значений, листы и CSV.

Запуск без параметров (все пути зашиты ниже)::

    uv run python -m research.lama_params.demo

У самой сети ``big-lama.pt`` параметров нет: это torchscript-модуль ``(картинка, маска) -> картинка``,
ему нужна лишь кратность сторон 8. Всё, чем можно управлять, — обвязка проекта вокруг сети:
что подать (ROI вокруг маски: ``padding``, ``roi_scale``), в каком масштабе (``hole_max_px``,
``roi_max_side``), как вклеить обратно (``feather``), какую маску (дилатация) и сколько областей
закрашивать одним прогоном (``group_dilate_frac``). Лист на каждый параметр: остальные стоят на
базовых значениях, меняется только он. Подробно — ``research/lama_params/README.md``.

Входы (синтетика с эталоном) генерируются при первом запуске и дальше читаются с диска.
"""

import csv
import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_utils.inpainting.apply import inpaint_by_groups
from ocr_utils.inpainting.backends import LamaFiller
from ocr_utils.inpainting.grouping import group_masks
from ocr_utils.inpainting.roi import mask_bbox, mask_components, roi_bounds
from ocr_utils.page_layout.overlay_frame import FONT_PATH, LegendEntry, framed
from ocr_utils.scan_cropping.gpu_models import GpuModels
from research.lama_params.synth import TargetObject, make_page

logger = logging.getLogger(__name__)

# Все пути — здесь: скрипт параметров не принимает.
OUT_ROOT = Path("/mnt/hotstore/scan_processing/lama_params")
INPUT_DIR = OUT_ROOT / "input"  # page.png, clean.png, mask_<объект>.png
SHEETS_DIR = OUT_ROOT / "sheets"  # <NN>_<параметр>_<объект>.jpg
SUMMARY_CSV = OUT_ROOT / "summary.csv"
PAGE_FILE = INPUT_DIR / "page.png"
CLEAN_FILE = INPUT_DIR / "clean.png"

# Зона оценки: маска объекта, раздутая на 40 px. Одна на все варианты объекта, иначе ряд дилатации
# сравнивал бы ошибку по зонам разного размера. 40 px перекрывают самую широкую дилатацию на листах.
EVAL_DILATE_PX = 40

# Раскладка листа: ширина плитки, число колонок, поля между плитками, кегль подписи.
TILE_WIDTH = 560
GRID_COLUMNS = 4
TILE_GAP = 8
CAPTION_SIZE = 17
CAPTION_ROW = 23
# Строк подписи у каждой плитки: у всех одинаково, чтобы вырезки в ряду стояли вровень.
CAPTION_LINES = 3

# Цвета BGR по палитре навыка draw-overlay: ROI — «вспомогательное», контур маски — «отвергнуто».
COLOUR_ROI = (200, 140, 60)
COLOUR_MASK = (0, 0, 220)
COLOUR_TEXT = (20, 20, 20)


class Param(Enum):
    """Параметр обвязки LaMa, которому посвящён лист; значение — часть имени файла."""

    HOLE_MAX_PX = "hole_max_px"
    ROI_MAX_SIDE = "roi_max_side"
    ROI_CONTEXT = "roi_scale_padding"
    FEATHER = "feather"
    MASK_DILATE = "mask_dilate"
    GROUPING = "group_dilate_frac"


@dataclass(frozen=True)
class Settings:
    """Полный набор ручек одного закраса.

    Базовые значения — как у закраса разметки из CVAT (``scan_cleanup.inpaint``): предел дыры 300,
    ROI вдвое больше группы плюс поле 64, растушёвка 9, склейка областей на треть их размера.
    Дилатация 9 — запас маски вокруг объекта, равный растушёвке (почему так — лист ``feather``).
    """

    mask_dilate_px: int = 9  # раздуть маску объекта на столько пикселей перед закрасом
    hole_max_px: Optional[int] = 300  # предел длинной стороны ДЫРЫ в масштабе сети; None — не задан
    roi_max_side: int = 512  # предел длинной стороны ROI (работает, только если hole_max_px = None); 0 — нативно
    padding: int = 64  # контекстное поле вокруг рамки группы, пикс.
    roi_scale: float = 2.0  # растяжение ROI от центра после поля
    feather: int = 9  # растушёвка шва внутрь маски, пикс.
    group_dilate_frac: Optional[float] = 1.0 / 3.0  # склейка областей; None — каждая связная область отдельно


NATIVE = Settings(hole_max_px=None, roi_max_side=0)


@dataclass(frozen=True)
class Variant:
    """Один закрас ряда: подпись плитки и настройки."""

    label: str
    settings: Settings


@dataclass(frozen=True)
class Experiment:
    """Лист: какой параметр, на каком объекте, что на нём смотреть, варианты и поле вырезки вокруг объекта."""

    param: Param
    target: TargetObject
    what_to_see: str
    variants: "tuple[Variant, ...]"
    crop_margin: int = 120


@dataclass(frozen=True)
class Outcome:
    """Результат одного варианта: картинка, ROI и меры в зоне оценки."""

    image: np.ndarray  # RGB uint8 (H, W, 3)
    rois: "list[tuple[int, int, int, int]]"
    net_hole_px: int  # наибольшая длинная сторона дыры в масштабе сети по группам
    net_roi_px: int  # наибольшая длинная сторона ROI в масштабе сети
    residue: float  # остаток объекта: |закрас - эталон| / |исходник - эталон| на пикселях объекта; 1 — не тронут
    mae: float  # средняя |закрас - эталон| по каналам в зоне оценки
    tone_shift: float  # средняя яркость закраса минус эталона в зоне оценки: туман > 0, клякса < 0
    sharpness: float  # СКО лапласиана закраса / эталона в зоне оценки: < 1 — мыло, 1 — как в эталоне
    seconds: float


def hole_variants() -> "tuple[Variant, ...]":
    """Ряд ``hole_max_px``: от нативного прогона до предела 50 px."""
    return (Variant("нативно", NATIVE),) + tuple(
        Variant(f"hole_max_px={h}", Settings(hole_max_px=h)) for h in (600, 400, 300, 200, 100, 50)
    )


EXPERIMENTS: "tuple[Experiment, ...]" = (
    Experiment(
        Param.HOLE_MAX_PX,
        TargetObject.BIG_STAMP,
        "дыра 700 px поверх текста: нативно — туман в центре; 300–600 — псевдотекст; 50–100 — мыло",
        hole_variants(),
    ),
    Experiment(
        Param.HOLE_MAX_PX,
        TargetObject.FINGER,
        "палец поверх растра: 300–400 — растр продолжен; 100 и меньше — растр крупной клеткой, ступень у стола",
        hole_variants(),
    ),
    Experiment(
        Param.HOLE_MAX_PX,
        TargetObject.STAMP,
        "тонкие штрихи по краске: дыра мала, до 400 её не ужимают вовсе; ниже — буквы под штрихами расплываются",
        hole_variants(),
        crop_margin=40,
    ),
    Experiment(
        Param.ROI_MAX_SIDE,
        TargetObject.FINGER,
        "предел по ROI ужимает дыру косвенно: одно и то же число даёт разную дыру на разных зонах",
        (
            Variant("нативно", NATIVE),
            Variant("roi_max_side=1024", Settings(hole_max_px=None, roi_max_side=1024)),
            Variant("roi_max_side=512", Settings(hole_max_px=None, roi_max_side=512)),
            Variant("roi_max_side=256", Settings(hole_max_px=None, roi_max_side=256)),
            Variant("для сравнения hole_max_px=300", Settings()),
        ),
    ),
    Experiment(
        Param.ROI_CONTEXT,
        TargetObject.FINGER,
        "контекст вокруг дыры: впритык (×1, поле 0) растр за пальцем не продолжен; от ×1.5 — продолжен",
        tuple(
            Variant(f"roi_scale={s:g}, padding={p}", Settings(roi_scale=s, padding=p))
            for s, p in ((1.0, 0), (1.0, 16), (1.0, 64), (1.5, 64), (2.0, 64), (3.0, 64))
        ),
        crop_margin=300,
    ),
    Experiment(
        Param.FEATHER,
        TargetObject.STAMP,
        "растушёвка шире запаса маски (9 px) возвращает край штриха: от 20 px печать проступает",
        tuple(Variant(f"feather={f}", Settings(feather=f)) for f in (0, 3, 9, 20, 40)),
        crop_margin=40,
    ),
    Experiment(
        Param.MASK_DILATE,
        TargetObject.STAMP,
        "запас маски: 0 — печать на месте; 3 — тень; 9 — чисто; 15+ — штрихи слились, строки потеряны",
        tuple(Variant(f"дилатация {d} px", Settings(mask_dilate_px=d)) for d in (0, 3, 9, 15, 40)),
        crop_margin=40,
    ),
    Experiment(
        Param.GROUPING,
        TargetObject.HANDWRITING,
        "шесть росчерков: по одному (ROI видит соседей) или одной операцией (дыра длиннее — сильнее ужимается)",
        tuple(
            Variant(
                f"{name}, {hole_name}", Settings(mask_dilate_px=3, feather=2, group_dilate_frac=frac, hole_max_px=h)
            )
            for frac, name in ((None, "по одному"), (1.0 / 3.0, "группой 1/3"))
            for h, hole_name in ((300, "hole 300"), (100, "hole 100"))
        ),
        crop_margin=60,
    ),
)


def load_or_make_inputs() -> "tuple[np.ndarray, np.ndarray, dict[TargetObject, np.ndarray]]":
    """Синтетическая полоса, эталон и маски: с диска, а если их нет — генерирует и сохраняет.

    Возвращает ``(полоса RGB, эталон RGB, {объект: маска uint8 0/255})``.
    """
    mask_files = {t: INPUT_DIR / f"mask_{t.value}.png" for t in TargetObject}
    if not (PAGE_FILE.exists() and CLEAN_FILE.exists() and all(p.exists() for p in mask_files.values())):
        logger.info("Генерирую синтетическую полосу в %s", INPUT_DIR)
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        synthetic = make_page()
        cv2.imwrite(str(PAGE_FILE), cv2.cvtColor(synthetic.page, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(CLEAN_FILE), cv2.cvtColor(synthetic.clean, cv2.COLOR_RGB2BGR))
        for target, path in mask_files.items():
            cv2.imwrite(str(path), synthetic.masks[target])
    page = cv2.cvtColor(cv2.imread(str(PAGE_FILE)), cv2.COLOR_BGR2RGB)
    clean = cv2.cvtColor(cv2.imread(str(CLEAN_FILE)), cv2.COLOR_BGR2RGB)
    masks = {t: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for t, p in mask_files.items()}
    return page, clean, masks


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Маска, раздутая кругом радиуса ``px`` (0 — без изменений)."""
    if px <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, kernel)


def split_groups(mask: np.ndarray, frac: Optional[float]) -> "list[np.ndarray]":
    """Маска, разбитая на операции закраса: связные области по одной (``frac=None``) или склеенные.

    Аргументы:
        mask: маска uint8 0/255;
        frac: ``group_dilate_frac`` из :func:`ocr_utils.inpainting.grouping.group_masks` или None.

    Возвращает список масок групп.
    """
    return mask_components(mask) if frac is None else group_masks(mask, frac)


def network_scale(roi_shape: "tuple[int, int]", zone: np.ndarray, settings: Settings) -> float:
    """Во сколько раз ROI ужимается перед сетью — та же логика, что в ``GpuModels.lama_fill_roi``.

    Нужна только для подписи на листе (какого размера дыру реально видит сеть); сам закрас идёт
    через ``lama_fill_roi``.

    Аргументы:
        roi_shape: (высота, ширина) ROI;
        zone: маска группы внутри ROI;
        settings: настройки варианта.

    Возвращает масштаб 0..1 (1 — нативно).
    """
    if settings.hole_max_px:
        box = mask_bbox(zone)
        hole = max(box[2] - box[0], box[3] - box[1]) if box else 0
        return min(1.0, settings.hole_max_px / hole) if hole else 1.0
    if settings.roi_max_side > 0 and max(roi_shape) > settings.roi_max_side:
        return settings.roi_max_side / max(roi_shape)
    return 1.0


def net_sizes(mask: np.ndarray, settings: Settings) -> "tuple[int, int]":
    """Наибольшие длинные стороны дыры и ROI в масштабе сети по всем группам маски.

    Возвращает ``(дыра, ROI)`` в пикселях сети.
    """
    worst_hole, worst_roi = 0, 0
    for zone in split_groups(mask, settings.group_dilate_frac):
        bounds = roi_bounds(zone, settings.padding, settings.roi_scale, mask.shape[:2])
        if bounds is None:
            continue
        x1, y1, x2, y2 = bounds
        zone_roi = zone[y1:y2, x1:x2]
        scale = network_scale((y2 - y1, x2 - x1), zone_roi, settings)
        box = mask_bbox(zone_roi)
        hole = max(box[2] - box[0], box[3] - box[1])
        worst_hole = max(worst_hole, round(hole * scale))
        worst_roi = max(worst_roi, round(max(y2 - y1, x2 - x1) * scale))
    return worst_hole, worst_roi


def object_residue(page: np.ndarray, result: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> float:
    """Какая доля объекта осталась после закраса — по его пикселям (маска до дилатации).

    Разность «исходник − эталон» на пикселе объекта — это сам объект (цвет печати, кожа пальца).
    Остаток — проекция разности «закрас − эталон» на неё, нормированная на её квадрат и усреднённая.
    Выдуманное сетью содержимое (псевдобуквы под печатью) с цветом объекта не связано и в среднем
    даёт около нуля; недокрашенный объект даёт около единицы.

    MAE по зоне этого не ловит: штрихи печати — малая доля зоны, и недокрашенная печать (буквы под
    ней остались эталонными) выходит по MAE «лучше» честно закрашенной (буквы под ней выдуманы).

    Аргументы:
        page, result, clean: исходник с объектом, закрас и эталон, RGB uint8;
        mask: маска объекта до дилатации, uint8 0/255.

    Возвращает долю: 1 — объект не тронут, 0 — следа объекта нет (меньше нуля — перекрашено «в другую сторону»).
    """
    on_object = mask > 0
    object_diff = page[on_object].astype(np.float32) - clean[on_object]
    left_diff = result[on_object].astype(np.float32) - clean[on_object]
    return float((object_diff * left_diff).sum() / max(float((object_diff**2).sum()), 1e-6))


def measure(result: np.ndarray, clean: np.ndarray, zone: np.ndarray) -> "tuple[float, float, float]":
    """Меры закраса против эталона в зоне оценки.

    Аргументы:
        result, clean: закрашенная полоса и эталон, RGB uint8;
        zone: зона оценки, bool (H, W).

    Возвращает ``(MAE, сдвиг тона, резкость)`` — смысл см. :class:`Outcome`.
    """
    mae = float(np.abs(result[zone].astype(np.float32) - clean[zone].astype(np.float32)).mean())
    gray_result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_clean = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.float32)
    tone_shift = float(gray_result[zone].mean() - gray_clean[zone].mean())
    # Резкость считаем по лапласиану только в рамке зоны, чтобы не гонять фильтр по всей полосе.
    x1, y1, x2, y2 = mask_bbox(zone.astype(np.uint8))
    lap_result = cv2.Laplacian(gray_result[y1:y2, x1:x2], cv2.CV_32F)
    lap_clean = cv2.Laplacian(gray_clean[y1:y2, x1:x2], cv2.CV_32F)
    inside = zone[y1:y2, x1:x2]
    sharpness = float(lap_result[inside].std() / max(lap_clean[inside].std(), 1e-6))
    return mae, tone_shift, sharpness


def run_variant(
    models: GpuModels, page: np.ndarray, clean: np.ndarray, mask: np.ndarray, zone: np.ndarray, settings: Settings
) -> Outcome:
    """Один закрас: раздуть маску, разбить на группы, прогнать LaMa по ROI, вклеить, измерить.

    Аргументы:
        models: загруженные сети (нужна только LaMa);
        page, clean: полоса с объектами и эталон, RGB uint8;
        mask: маска объекта до дилатации, uint8 0/255;
        zone: зона оценки, bool;
        settings: настройки варианта.

    Возвращает :class:`Outcome`.
    """
    work_mask = dilate(mask, settings.mask_dilate_px)
    filler = LamaFiller(models, roi_max_side=settings.roi_max_side, hole_max_px=settings.hole_max_px)
    started = time.perf_counter()
    result, rois = inpaint_by_groups(
        page,
        work_mask,
        filler,
        groups=lambda m: split_groups(m, settings.group_dilate_frac),
        padding=settings.padding,
        feather=settings.feather,
        roi_scale=settings.roi_scale,
    )
    seconds = time.perf_counter() - started
    net_hole, net_roi = net_sizes(work_mask, settings)
    mae, tone, sharpness = measure(result, clean, zone)
    return Outcome(
        result, rois, net_hole, net_roi, object_residue(page, result, clean, mask), mae, tone, sharpness, seconds
    )


def caption_strip(lines: "list[str]", width: int) -> np.ndarray:
    """Белая полоса подписи плитки (DejaVu через PIL: кириллица и «×»). Возвращает BGR."""
    picture = Image.new("RGB", (width, 4 + CAPTION_ROW * max(CAPTION_LINES, len(lines))), (255, 255, 255))
    draw = ImageDraw.Draw(picture)
    font = ImageFont.truetype(FONT_PATH, CAPTION_SIZE)
    for i, line in enumerate(lines):
        draw.text((4, 2 + CAPTION_ROW * i), line, font=font, fill=COLOUR_TEXT[::-1])
    return cv2.cvtColor(np.asarray(picture), cv2.COLOR_RGB2BGR)


def tile(
    rgb: np.ndarray,
    crop: "tuple[int, int, int, int]",
    lines: "list[str]",
    rois: "list[tuple[int, int, int, int]]" = (),
    contour: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Плитка листа: вырезка полосы, рамки ROI и контур маски поверх, подпись над вырезкой (в поле).

    Аргументы:
        rgb: полоса RGB;
        crop: вырезка (x1, y1, x2, y2);
        lines: строки подписи;
        rois: ROI закраса в координатах полосы — рисуются тонкой рамкой;
        contour: маска, чей контур обвести (только на плитке исходника).

    Возвращает BGR шириной ``TILE_WIDTH``.
    """
    x1, y1, x2, y2 = crop
    piece = cv2.cvtColor(rgb[y1:y2, x1:x2], cv2.COLOR_RGB2BGR).copy()
    if contour is not None:
        found, _ = cv2.findContours(
            (contour[y1:y2, x1:x2] > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(piece, found, -1, COLOUR_MASK, 2)
    for rx1, ry1, rx2, ry2 in rois:
        cv2.rectangle(piece, (rx1 - x1, ry1 - y1), (rx2 - x1 - 1, ry2 - y1 - 1), COLOUR_ROI, 3)
    scale = TILE_WIDTH / piece.shape[1]
    piece = cv2.resize(piece, (TILE_WIDTH, round(piece.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return np.vstack([caption_strip(lines, TILE_WIDTH), piece])


def grid(tiles: "list[np.ndarray]") -> np.ndarray:
    """Плитки сеткой по ``GRID_COLUMNS`` в ряд; высоты в ряду выравниваются белым полем снизу."""
    padded = [np.pad(t, ((0, TILE_GAP), (0, TILE_GAP), (0, 0)), constant_values=255) for t in tiles]
    rows = []
    for start in range(0, len(padded), GRID_COLUMNS):
        row = padded[start : start + GRID_COLUMNS]
        row += [np.full((1, TILE_WIDTH + TILE_GAP, 3), 255, np.uint8)] * (GRID_COLUMNS - len(row))
        height = max(t.shape[0] for t in row)
        rows.append(
            np.hstack([np.pad(t, ((0, height - t.shape[0]), (0, 0), (0, 0)), constant_values=255) for t in row])
        )
    return np.vstack(rows)


def run_experiment(
    models: GpuModels,
    index: int,
    experiment: Experiment,
    page: np.ndarray,
    clean: np.ndarray,
    masks: "dict[TargetObject, np.ndarray]",
) -> "list[dict]":
    """Прогоняет все варианты листа, пишет лист и возвращает строки сводки.

    Аргументы:
        models: загруженные сети;
        index: номер листа (префикс имени файла);
        experiment: что гоняем;
        page, clean, masks: входы из :func:`load_or_make_inputs`.

    Возвращает список строк для ``summary.csv``.
    """
    mask = masks[experiment.target]
    zone = dilate(mask, EVAL_DILATE_PX) > 0
    bx1, by1, bx2, by2 = mask_bbox(mask)
    height, width = page.shape[:2]
    margin = experiment.crop_margin
    crop = (max(0, bx1 - margin), max(0, by1 - margin), min(width, bx2 + margin), min(height, by2 + margin))

    tiles = [
        tile(
            page,
            crop,
            ["исходник", f"контур маски с дилатацией {experiment.variants[0].settings.mask_dilate_px} px"],
            contour=dilate(mask, experiment.variants[0].settings.mask_dilate_px),
        ),
        tile(clean, crop, ["эталон", "та же полоса без объекта"]),
    ]
    rows = []
    for variant in experiment.variants:
        outcome = run_variant(models, page, clean, mask, zone, variant.settings)
        lines = [
            variant.label,
            f"остаток {outcome.residue:.2f} · MAE {outcome.mae:.1f} · тон {outcome.tone_shift:+.1f} · резк. {outcome.sharpness:.2f}",
            f"дыра в сети {outcome.net_hole_px} px · ROI в сети {outcome.net_roi_px} px",
        ]
        tiles.append(tile(outcome.image, crop, lines, rois=outcome.rois))
        logger.info("%s / %s / %s: %s", experiment.param.value, experiment.target.value, variant.label, lines[1])
        rows.append(
            {
                "sheet": index,
                "param": experiment.param.value,
                "target": experiment.target.value,
                "variant": variant.label,
                **asdict(variant.settings),
                "net_hole_px": outcome.net_hole_px,
                "net_roi_px": outcome.net_roi_px,
                "residue": round(outcome.residue, 3),
                "mae": round(outcome.mae, 2),
                "tone_shift": round(outcome.tone_shift, 2),
                "sharpness": round(outcome.sharpness, 3),
                "seconds": round(outcome.seconds, 3),
            }
        )

    base = Settings()
    header = [
        f"{experiment.param.value} — объект «{experiment.target.value}»",
        experiment.what_to_see,
        (
            f"база: дилатация {base.mask_dilate_px}, hole_max_px {base.hole_max_px}, roi_scale {base.roi_scale:g}, "
            f"padding {base.padding}, feather {base.feather}, группы {base.group_dilate_frac:.2f}"
        ),
        "остаток — доля объекта, оставшаяся на его пикселях (проекция «закрас−эталон» на «исходник−эталон»): 1 — не тронут, 0 — стёрт",
        "MAE — к эталону в зоне маска+40 px; тон — яркость минус эталон; резк. — СКО лапласиана к эталону (1 = как было)",
    ]
    legend = [LegendEntry("ROI, поданный в сеть", COLOUR_ROI), LegendEntry("контур маски (на исходнике)", COLOUR_MASK)]
    sheet = framed(grid(tiles), header, legend)
    path = SHEETS_DIR / f"{index:02d}_{experiment.param.value}_{experiment.target.value}.jpg"
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    logger.info("Лист: %s", path)
    return rows


def main() -> None:
    """Генерирует (или читает) входы, гоняет все листы ``EXPERIMENTS``, пишет листы и ``summary.csv``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    page, clean, masks = load_or_make_inputs()
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    rows: "list[dict]" = []
    with GpuModels(with_detection=False, with_lama=True) as models:
        for index, experiment in enumerate(EXPERIMENTS, start=1):
            rows = rows + run_experiment(models, index, experiment, page, clean, masks)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Сводка: %s (%d вариантов)", SUMMARY_CSV, len(rows))


if __name__ == "__main__":
    main()
