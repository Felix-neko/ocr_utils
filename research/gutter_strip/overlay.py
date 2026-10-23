"""Оверлей стенда полосы-вклейки: кадр с полупрозрачной маской полосы, вспомогательными линиями, линией корешка, краями книги и рамками страниц «было | стало»; справа — верх и низ полосы у корешка крупно."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed
from ocr_utils.scan_cropping.v2.overlay import COLOUR_BOOK_LINE, COLOUR_SPINE, _draw_line
from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM
from ocr_utils.scan_cropping.v2.spread_lines import Line, SpreadLines
from research.gutter_strip.fit import PageFit
from ocr_utils.scan_cropping.v2.gutter_strip import Strip

# Цвета (BGR, палитра ``/draw-overlay``): полоса — оранжевая полупрозрачная область, вспомогательные линии — розовый
# пунктир, рамка «было» — серая справочная, «стало» — зелёная «принято».
COLOUR_STRIP = (0, 165, 255)
COLOUR_AUX = (180, 105, 255)
COLOUR_BEFORE = (170, 170, 170)
COLOUR_AFTER = (0, 150, 0)
STRIP_ALPHA = 0.4
# Ширина общего вида, px; вырезки у корешка: полуширина и высота, мм, увеличение.
VIEW_WIDTH = 1300
ZOOM_HALF_MM = 25.0
ZOOM_HEIGHT_MM = 40.0
ZOOM = 2.5


def _cut(canvas: np.ndarray, spine: Line, y_centre: float) -> np.ndarray:
    """Вырезка ``±ZOOM_HALF_MM`` у корешка высотой ``ZOOM_HEIGHT_MM`` вокруг строки ``y_centre``, увеличенная в ``ZOOM``."""
    h, w = canvas.shape[:2]
    cx = int(spine.x_at(y_centre))
    half, hh = int(ZOOM_HALF_MM * SEG_PX_PER_MM), int(ZOOM_HEIGHT_MM * SEG_PX_PER_MM / 2)
    y0, y1 = max(int(y_centre) - hh, 0), min(int(y_centre) + hh, h)
    x0, x1 = max(cx - half, 0), min(cx + half, w)
    return cv2.resize(canvas[y0:y1, x0:x1], None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)


def draw_strip_overlay(
    img: np.ndarray,
    lines: SpreadLines,
    spine: Line,
    spine_source: str,
    strip: Strip | None,
    fits: list[PageFit],
    title: str,
) -> np.ndarray:
    """Картинка кадра стенда.

    Args:
        img: Кадр масштаба сегментации, BGR.
        lines: Прямые разворота v2 (края книги).
        spine: Линия корешка стенда.
        spine_source: Откуда наклон корешка (для шапки).
        strip: Полоса или None.
        fits: Рамки страниц «было | стало».
        title: Шапка (кадр, метка).

    Returns:
        Картинка BGR с шапкой и легендой.
    """
    canvas = img.copy()
    if strip is not None:
        layer = canvas.copy()
        layer[strip.mask] = COLOUR_STRIP
        cv2.addWeighted(layer, STRIP_ALPHA, canvas, 1 - STRIP_ALPHA, 0, canvas)
    for edge in (lines.book_left, lines.book_right):
        if edge.line is not None:
            _draw_line(canvas, edge.line, COLOUR_BOOK_LINE, 2, dashed=not edge.visible)
    for fit in fits:
        cv2.polylines(canvas, [np.round(fit.before.polygon()).astype(np.int32)], True, COLOUR_BEFORE, 2, cv2.LINE_AA)
        cv2.polylines(canvas, [np.round(fit.after.polygon()).astype(np.int32)], True, COLOUR_AFTER, 2, cv2.LINE_AA)
    if strip is not None:
        _draw_line(canvas, strip.left, COLOUR_AUX, 2, dashed=True)
        _draw_line(canvas, strip.right, COLOUR_AUX, 2, dashed=True)
    _draw_line(canvas, spine, COLOUR_SPINE, 2, dashed=False)
    s = VIEW_WIDTH / canvas.shape[1]
    view = cv2.resize(canvas, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    # Справа — верх и низ полосы у корешка крупно (там всплески края у неплотно приклеенной полосы).
    top = min(f.before.top[:, 1].min() for f in fits)
    bot = max(f.before.bottom[:, 1].max() for f in fits)
    cuts = [_cut(canvas, spine, top + 8 * SEG_PX_PER_MM), _cut(canvas, spine, bot - 8 * SEG_PX_PER_MM)]
    gap = np.full((6, cuts[0].shape[1], 3), 255, np.uint8)
    column = np.vstack([cuts[0], gap, cuts[1]])
    if column.shape[0] > view.shape[0]:
        k = view.shape[0] / column.shape[0]
        column = cv2.resize(column, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    column = np.vstack([column, np.full((view.shape[0] - column.shape[0], column.shape[1], 3), 255, np.uint8)])
    picture = np.hstack([view, np.full((view.shape[0], 8, 3), 255, np.uint8), column])
    sub = f"корешок {spine.tilt_deg:+.2f}°: {spine_source}"
    if strip is not None:
        sub += (f"; полоса {strip.width_mm:.1f} мм, края {strip.left.tilt_deg:+.2f}° / {strip.right.tilt_deg:+.2f}°; "
                f"расхождение рамок у корешка {max(f.spike_mm for f in fits):.1f} мм")  # fmt: skip
    else:
        sub += "; полосы нет"
    legend = [
        LegendEntry("полоса-вклейка (маска)", COLOUR_STRIP, alpha=STRIP_ALPHA, style=SampleStyle.BOX),
        LegendEntry("вспомогательные линии: края полосы", COLOUR_AUX),
        LegendEntry(
            "линия корешка: без полосы — разделитель v2, с полосой — средний наклон краёв полосы через её середину",
            COLOUR_SPINE,
        ),
        LegendEntry("края книги (пунктир — за кадром)", COLOUR_BOOK_LINE),
        LegendEntry("рамка страницы было: прежний сплайн v2 по маске до корешка", COLOUR_BEFORE),
        LegendEntry(
            "подгоночная область стало: сплайн с узлами на изгибах, у корешка — линия корешка, у полосы — до вспомогательных линий и отрезок",
            COLOUR_AFTER,
        ),
    ]
    return framed(picture, [f"{title} — полоса-вклейка у корешка", sub, "справа: верх и низ у корешка ×2.5"], legend)
