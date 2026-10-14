"""Закрас пальцев по набору подшивок в вариантах защиты печати (Surya, глифы, оба) и вырезки «было | зона | стало»."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed
from research.bound_volumes.frames import OUT_ROOT, Frame
from ocr_utils.scan_cropping.v2.fingers import (  # noqa: F401 — ядро, ре-экспорт для стенда
    ASYM_RATIO,
    INK_BG_CLOSE_PX,
    INK_BG_MEDIAN_PX,
    INK_CORE_BAND_PX,
    INK_DARK_RAMP,
    INK_MAX_CHROMA,
    TEXT_BLOCK_GUARD_PX,
    TIGHT_DILATE_PX,
    copy_back_ink,
    text_free_context,
)
from ocr_utils.scan_cropping.v2.glyphs import GLYPH_SCALE, glyph_protect  # noqa: F401

log = logging.getLogger(__name__)

# Корень выхода закраса: <FINGER_ROOT>/<вариант>/{clean,masks}/<плоский ключ>.*
FINGER_ROOT = OUT_ROOT / "fingers"
# Параметры закраса — как в боевом прогоне МТС 450 dpi (run_scripts/scan_cropping/run_detect_and_crop_mts_3.sh).
DILATE_PX = 90
LAYOUT_PAD_PX = (18, 72)
# Поле у ядра пальца, где защита глифов не действует, в варианте PAPER, px кадра: ореол кожи на 350 dpi до ~50 px.
PAPER_CORE_MARGIN_PX = 48

# Цвета оверлея (BGR, палитра проекта): зона закраса — оранжевый, ядро пальца — розовый, защита — зелёный.
COLOUR_PAINT = (0, 165, 255)
COLOUR_CORE = (180, 105, 255)
COLOUR_PROTECT = (40, 170, 40)
AREA_ALPHA = 0.45


class Guard(str, Enum):
    """Вариант защиты печати при закрасе пальцев."""

    # Как в боевом scan_cropping МТС: Surya layout, copy-back блоков, высветление +20.
    SURYA = "surya"
    # Газетный: глифы цепочкой с зазором (``text_guard.finger_protect``), без высветления.
    GLYPH = "glyph"
    # Оба: глифы урезают зону закраса, Surya возвращает блоки после LaMa; без высветления.
    BOTH = "both"
    # Глифы, как GLYPH, но LaMa видит кадр без блоков текста (``text_free_context``): палец на тексте закрашивается
    # бумагой, а не полосами-продолжениями обрезанных пальцем строк; поле у ядра пальца шире.
    PAPER = "paper"
    # Красится вся раздутая зона пальца (без вычитания защиты) по контексту без блоков текста, затем в пределах
    # защиты глифов с оригинала возвращается только краска (``copy_back_ink``): кожа пальца, попавшая в зазор
    # защиты, не становится известным контекстом LaMa и не протекает в дыру.
    INK_BACK = "inkback"


@dataclass(frozen=True)
class FingerResult:
    """Итог закраса кадра: картинка без пальцев и маски (зона закраса, ядро пальца, защита)."""

    clean: np.ndarray
    paint: np.ndarray
    core: np.ndarray
    protect: np.ndarray | None
    info: str








def run_frame(bgr: np.ndarray, models, guard: Guard, guard_px: int, log_name: str) -> FingerResult:
    """Закрас пальцев одного кадра в заданном варианте защиты.

    Args:
        bgr: Кадр BGR в исходном разрешении.
        models: ``GpuModels`` с LaMa (и с Surya layout для вариантов SURYA и BOTH).
        guard: Вариант защиты.
        guard_px: Зазор защиты по глифам (для GLYPH и BOTH).
        log_name: Имя кадра для логов.

    Returns:
        Картинка без пальцев и маски.
    """
    from ocr_utils.scan_cropping.finger_removal.removal import (
        FINGER_PADDING,
        FINGER_ROI_SCALE,
        LAMA_ROI_MAX_SIDE,
        remove_fingers,
    )
    from ocr_utils.scan_cropping.finger_removal.text_protection import PROTECT_COPY_BACK

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    protect = glyph_protect(rgb, guard_px) if guard in (Guard.GLYPH, Guard.BOTH, Guard.PAPER, Guard.INK_BACK) else None
    clean, paint, _, _, info, core, _ = remove_fingers(
        bgr,
        models,
        want_boxes=True,
        dilate_px=TIGHT_DILATE_PX if guard in (Guard.PAPER, Guard.INK_BACK) else DILATE_PX,
        asymmetric_dilation_ratio=ASYM_RATIO,
        light_increment=20 if guard == Guard.SURYA else 0,
        protect_text=guard in (Guard.SURYA, Guard.BOTH),
        protect_mode=PROTECT_COPY_BACK,
        layout_pad_px=LAYOUT_PAD_PX,
        log_name=log_name,
        protect_mask=None if guard == Guard.INK_BACK else protect,
        protect_core_margin_px=PAPER_CORE_MARGIN_PX if guard == Guard.PAPER else 16,
    )
    core = np.zeros(paint.shape, np.uint8) if core is None else core
    if guard in (Guard.PAPER, Guard.INK_BACK) and paint.any():
        # Второй проход LaMa по кадру без блоков текста: в итог берётся только зона закраса.
        context = text_free_context(bgr, glyph_protect(rgb, TEXT_BLOCK_GUARD_PX) > 0, paint > 0)
        filled = models.inpaint(
            cv2.cvtColor(context, cv2.COLOR_BGR2RGB),
            paint,
            padding=FINGER_PADDING,
            roi_scale=FINGER_ROI_SCALE,
            roi_max_side=LAMA_ROI_MAX_SIDE,
        )
        clean = bgr.copy()
        clean[paint > 0] = cv2.cvtColor(filled, cv2.COLOR_RGB2BGR)[paint > 0]
        if guard == Guard.INK_BACK:
            k = 2 * INK_CORE_BAND_PX + 1
            band = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
            clean, _ = copy_back_ink(clean, bgr, (paint > 0) & (protect > 0) & ~band)
    return FingerResult(clean, paint, core, protect, info)


def out_paths(guard: Guard, frame: Frame) -> tuple[Path, Path]:
    """Пути выхода закраса: картинка без пальцев и папка масок.

    Args:
        guard: Вариант защиты.
        frame: Кадр.

    Returns:
        ``(путь картинки, префикс масок)``; маски — ``<префикс>_paint.png``, ``_core.png``, ``_protect.png``.
    """
    root = FINGER_ROOT / guard.value
    return root / "clean" / f"{frame.flat_key}.jpg", root / "masks" / frame.flat_key


def save(result: FingerResult, guard: Guard, frame: Frame) -> None:
    """Пишет картинку без пальцев и маски варианта на SSD.

    Args:
        result: Итог закраса.
        guard: Вариант защиты.
        frame: Кадр.
    """
    img_path, mask_prefix = out_paths(guard, frame)
    img_path.parent.mkdir(parents=True, exist_ok=True)
    mask_prefix.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(img_path), result.clean, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(f"{mask_prefix}_paint.png", (result.paint > 0).astype(np.uint8) * 255)
    cv2.imwrite(f"{mask_prefix}_core.png", (result.core > 0).astype(np.uint8) * 255)
    if result.protect is not None:
        cv2.imwrite(f"{mask_prefix}_protect.png", result.protect)


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Читает маску 0/255 или отдаёт пустую, если файла нет.

    Args:
        path: Путь PNG.
        shape: Размер кадра ``(H, W)`` для пустой маски.

    Returns:
        Булева маска.
    """
    if not path.exists():
        return np.zeros(shape, bool)
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0


def zone_overlay(bgr: np.ndarray, paint: np.ndarray, core: np.ndarray, protect: np.ndarray) -> np.ndarray:
    """Кадр с полупрозрачными зоной закраса, ядром пальца и защитой печати.

    Args:
        bgr: Кадр BGR (исходный, с пальцами).
        paint: Зона закраса (bool).
        core: Ядро пальца до дилатации (bool).
        protect: Защита печати (bool).

    Returns:
        Холст BGR.
    """
    canvas = bgr.copy()
    layer = canvas.copy()
    layer[protect] = COLOUR_PROTECT
    layer[paint] = COLOUR_PAINT
    layer[core & paint] = COLOUR_CORE
    cv2.addWeighted(layer, AREA_ALPHA, canvas, 1 - AREA_ALPHA, 0, canvas)
    return canvas


LEGEND = [
    LegendEntry("зона закраса LaMa", COLOUR_PAINT, alpha=AREA_ALPHA, style=SampleStyle.BOX),
    LegendEntry("ядро пальца (SAM до дилатации)", COLOUR_CORE, alpha=AREA_ALPHA, style=SampleStyle.BOX),
    LegendEntry("защита печати (глифы с зазором)", COLOUR_PROTECT, alpha=AREA_ALPHA, style=SampleStyle.BOX),
]


def finger_boxes(paint: np.ndarray, margin: int) -> list[tuple[int, int, int, int]]:
    """Рамки компонент зоны закраса с полем вокруг (для вырезок крупно).

    Args:
        paint: Зона закраса (объединение вариантов), bool.
        margin: Поле вокруг компоненты, px.

    Returns:
        Список ``(x0, y0, x1, y1)`` в пределах кадра.
    """
    h, w = paint.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(paint.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if area < 500:
            continue
        out.append((max(0, x - margin), max(0, y - margin), min(w, x + bw + margin), min(h, y + bh + margin)))
    return out


def label(img: np.ndarray, text: str) -> np.ndarray:
    """Подпись над вырезкой: белая полоса с текстом (DejaVu через ``header_strip``).

    Args:
        img: Вырезка BGR.
        text: Подпись.

    Returns:
        Вырезка с полосой подписи сверху.
    """
    from ocr_utils.page_layout.overlay_frame import header_strip

    return np.vstack([header_strip([text], img.shape[1]), img])


def crop_fit(img: np.ndarray, box: tuple[int, int, int, int], width: int) -> np.ndarray:
    """Вырезка по рамке, приведённая к заданной ширине.

    Args:
        img: Картинка BGR.
        box: Рамка ``(x0, y0, x1, y1)``.
        width: Ширина результата, px.

    Returns:
        Вырезка BGR шириной ``width``.
    """
    x0, y0, x1, y1 = box
    scale = width / (x1 - x0)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=interp)


def finger_sheet(frame: Frame, guards: list[Guard], tile_width: int = 640, margin: int = 160) -> np.ndarray | None:
    """Лист кадра: по каждому пальцу ряд вырезок «было», затем «зона | стало» по каждому варианту.

    Args:
        frame: Кадр.
        guards: Варианты, у которых есть выход.
        tile_width: Ширина вырезки на листе, px.
        margin: Поле вокруг зоны закраса, px кадра.

    Returns:
        Лист BGR или None, если пальцев не нашлось ни в одном варианте.
    """
    from research.bound_volumes.frames import read_bgr

    src = read_bgr(frame.work_path)
    shape = src.shape[:2]
    per_guard = {}
    union = np.zeros(shape, bool)
    for g in guards:
        img_path, prefix = out_paths(g, frame)
        if not img_path.exists():
            continue
        paint = read_mask(Path(f"{prefix}_paint.png"), shape)
        core = read_mask(Path(f"{prefix}_core.png"), shape)
        protect = read_mask(Path(f"{prefix}_protect.png"), shape)
        per_guard[g] = (read_bgr(img_path), zone_overlay(src, paint, core, protect))
        union |= paint
    rows = []
    for x0, y0, x1, y1 in finger_boxes(union, margin):
        box = (x0, y0, x1, y1)
        tiles = [label(crop_fit(src, box, tile_width), "было")]
        for g, (clean, over) in per_guard.items():
            tiles += [
                label(crop_fit(over, box, tile_width), f"{g.value}: зона"),
                label(crop_fit(clean, box, tile_width), f"{g.value}: стало"),
            ]
        rows.append(np.hstack(tiles))
    if not rows:
        return None
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 8), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    return framed(np.vstack(rows), [f"МТО 1990 {frame.key} — закрас пальцев по вариантам защиты"], LEGEND)
