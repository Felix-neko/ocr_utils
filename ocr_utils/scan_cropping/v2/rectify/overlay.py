"""Оверлей ``09_pages`` выхода страниц: выпрямленный разворот с полупрозрачными областями страниц, кривой линией разреза и рамками записанных частей; игнорируемые флагами страницы — серой заливкой."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed
from ocr_utils.scan_cropping.v2.fill import paper_bbox
from ocr_utils.scan_cropping.v2.rectify.pages import PagePart, cut_line, page_areas, part_areas
from ocr_utils.scan_cropping.v2.rectify.text_run import TextResult

# Высота оверлея по большей стороне, px: разворот целиком, буквы ещё различимы.
OVERLAY_SIDE = 2000
# Цвета (BGR, палитра навыка draw-overlay).
COLOUR_LEFT = (220, 90, 20)  # область левой страницы — синий
COLOUR_RIGHT = (40, 170, 40)  # область правой страницы — зелёный
COLOUR_IGNORED = (120, 120, 120)  # страница, игнорируемая флагом кадра — серый
COLOUR_CUT_LINE = (180, 105, 255)  # линия разреза — розовый (``COLOUR_TAIL``: линия отсечки)
COLOUR_BOX = (200, 60, 200)  # рамка записанной части — фиолетовый (``overlay.COLOUR_BOX``)
AREA_ALPHA = 0.18  # прозрачность заливки областей: буквы читаются
IGNORED_ALPHA = 0.45  # игнорируемая страница — заметно серее


def pages_overlay(result: TextResult, parts: list[PagePart], header: list[str]) -> np.ndarray:
    """Оверлей выхода страниц кадра.

    Args:
        result: Итог выпрямления кадра.
        parts: Записанные части (остальные стороны — игнорируемые флагами).
        header: Строки шапки.

    Returns:
        Картинка BGR со шапкой и легендой в полях.
    """
    areas = page_areas(result.layout, result.dense)
    h, w = result.final.shape[:2]
    f = OVERLAY_SIDE / max(h, w)
    size = (max(1, int(w * f)), max(1, int(h * f)))
    canvas = cv2.resize(result.final, size, interpolation=cv2.INTER_AREA)
    written = set(parts)
    spread = PagePart.SPREAD in written
    # Одна страница кадра — одна область (цвет левой); записана, если записана хоть одна часть.
    sides = ((areas[0], None, COLOUR_LEFT),) if result.layout.single else (
        (areas[0], PagePart.LEFT, COLOUR_LEFT), (areas[1], PagePart.RIGHT, COLOUR_RIGHT))  # fmt: skip
    for area, part, colour in sides:
        small = cv2.resize(area.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        keep = spread or part in written or (part is None and bool(written))
        layer = canvas.copy()
        layer[small] = colour if keep else COLOUR_IGNORED
        alpha = AREA_ALPHA if keep else IGNORED_ALPHA
        cv2.addWeighted(layer, alpha, canvas, 1 - alpha, 0, canvas)
    line = cut_line(result.layout, result.dense)
    if not spread and len(line):
        cv2.polylines(canvas, [np.round(line * f).astype(np.int32)], False, COLOUR_CUT_LINE, 2, cv2.LINE_AA)
    area_of = part_areas(result, parts)
    for part in parts:
        box = paper_bbox(area_of[part])
        if box is not None:
            x0, y0, x1, y1 = (int(round(v * f)) for v in box)
            cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), COLOUR_BOX, 2)
    legend = [
        LegendEntry(
            "область левой страницы (одной страницы кадра)", COLOUR_LEFT, alpha=AREA_ALPHA, style=SampleStyle.BOX
        ),
        LegendEntry("область правой страницы", COLOUR_RIGHT, alpha=AREA_ALPHA, style=SampleStyle.BOX),
        LegendEntry("страница игнорируется флагом кадра", COLOUR_IGNORED, alpha=IGNORED_ALPHA, style=SampleStyle.BOX),
        LegendEntry("линия разреза", COLOUR_CUT_LINE),
        LegendEntry("рамка записанной страницы (закрас до неё)", COLOUR_BOX),
    ]
    return framed(canvas, header, legend)
