"""Оверлей областей «левая страница — полоса — правая страница» кадра: заливки и контуры, линия корешка, подгоночные области; «было» (страницы и полоса v2) | «стало» (исправленные страницы и полоса-область); справа — верх и низ у корешка крупно."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, header_strip, legend_strip
from ocr_utils.scan_cropping.v2.page_frames import PageFrame
from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM
from ocr_utils.scan_cropping.v2.spread_lines import Line

COLOUR_LEFT = (220, 90, 20)
COLOUR_STRIP = (0, 165, 255)
COLOUR_RIGHT = (0, 150, 0)
COLOUR_SPINE = (200, 60, 200)
COLOUR_FRAME = (0, 0, 200)
COLOUR_EXTENSION = (140, 215, 255)
ALPHA = 0.4
ZOOM = 3.0
ZOOM_HALF_MM = 25.0
ZOOM_HEIGHT_MM = 36.0
VIEW_WIDTH = 1100


@dataclass(frozen=True)
class RegionView:
    """Что рисуется в одной половине картинки.

    Поля:
        title: Подпись половины.
        left: Маска левой страницы.
        right: Маска правой страницы.
        strip: Маска полосы (пустая — полосы нет).
        spine: Линия корешка.
        frames: Подгоночные области страниц.
        extension: Продолжение полосы вдоль корешка (или None).
    """

    title: str
    left: np.ndarray
    right: np.ndarray
    strip: np.ndarray
    spine: Line
    frames: list[PageFrame]
    extension: np.ndarray | None = None


def paint_view(img: np.ndarray, view: RegionView) -> np.ndarray:
    """Кадр с полупрозрачными областями, их контурами, линией корешка и подгоночными областями."""
    canvas = img.copy()
    layer = canvas.copy()
    for mask, colour in ((view.left, COLOUR_LEFT), (view.right, COLOUR_RIGHT), (view.strip, COLOUR_STRIP)):
        layer[mask] = colour
    if view.extension is not None:
        layer[view.extension] = COLOUR_EXTENSION
    cv2.addWeighted(layer, ALPHA, canvas, 1 - ALPHA, 0, canvas)
    for mask, colour in ((view.left, COLOUR_LEFT), (view.right, COLOUR_RIGHT), (view.strip, COLOUR_STRIP)):
        cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, cnt, -1, colour, 1)
    y0, y1 = view.spine.y_top, view.spine.y_bot
    cv2.line(canvas, (int(view.spine.x_at(y0)), int(y0)), (int(view.spine.x_at(y1)), int(y1)), COLOUR_SPINE, 1,
             cv2.LINE_AA)  # fmt: skip
    for f in view.frames:
        cv2.polylines(canvas, [np.round(f.polygon()).astype(np.int32)[:, None]], True, COLOUR_FRAME, 1, cv2.LINE_AA)
    return canvas


def zoom_at(canvas: np.ndarray, spine: Line, y: float) -> np.ndarray:
    """Вырезка ±``ZOOM_HALF_MM`` у линии корешка высотой ``ZOOM_HEIGHT_MM`` вокруг строки ``y``, ×``ZOOM``."""
    h, w = canvas.shape[:2]
    cx = int(spine.x_at(y))
    half, hh = int(ZOOM_HALF_MM * SEG_PX_PER_MM), int(ZOOM_HEIGHT_MM * SEG_PX_PER_MM / 2)
    y0, y1 = max(int(y) - hh, 0), min(int(y) + hh, h)
    x0, x1 = max(cx - half, 0), min(cx + half, w)
    return cv2.resize(canvas[y0:y1, x0:x1], None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)


def half_picture(img: np.ndarray, view: RegionView, ends: tuple[float, float]) -> np.ndarray:
    """Половина картинки: подпись, кадр целиком (ширина ``VIEW_WIDTH``) и справа верх и низ у корешка крупно."""
    canvas = paint_view(img, view)
    s = VIEW_WIDTH / canvas.shape[1]
    whole = cv2.resize(canvas, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    zooms = [zoom_at(canvas, view.spine, y) for y in ends]
    zw = max(z.shape[1] for z in zooms)
    zooms = [np.pad(z, ((0, 6), (0, zw - z.shape[1]), (0, 0)), constant_values=255) for z in zooms]
    right = np.vstack(zooms)
    hgt = max(whole.shape[0], right.shape[0])
    whole, right = (np.pad(p, ((0, hgt - p.shape[0]), (0, 0), (0, 0)), constant_values=255) for p in (whole, right))
    body = np.hstack([whole, np.full((hgt, 8, 3), 255, np.uint8), right])
    return np.vstack([header_strip([view.title], body.shape[1]), body])


def regions_overlay(img: np.ndarray, before: RegionView, after: RegionView, header: list[str],ends: tuple[float, float]) -> np.ndarray:  # fmt: skip
    """Картинка «было | стало» областей кадра с общей шапкой и легендой в полях.

    Args:
        img: Кадр масштаба сегментации, BGR.
        before: Страницы и полоса v2.
        after: Исправленные страницы и полоса-область.
        header: Строки шапки.
        ends: Строки верха и низа у корешка для вырезок.

    Returns:
        Картинка BGR.
    """
    halves = [half_picture(img, v, ends) for v in (before, after)]
    hgt = max(p.shape[0] for p in halves)
    halves = [np.pad(p, ((0, hgt - p.shape[0]), (0, 0), (0, 0)), constant_values=255) for p in halves]
    body = np.hstack([halves[0], np.full((hgt, 16, 3), 255, np.uint8), halves[1]])
    legend = [
        LegendEntry("левая страница", COLOUR_LEFT, alpha=ALPHA, style=SampleStyle.BOX),
        LegendEntry("полоса-вклейка", COLOUR_STRIP, alpha=ALPHA, style=SampleStyle.BOX),
        LegendEntry("правая страница", COLOUR_RIGHT, alpha=ALPHA, style=SampleStyle.BOX),
        LegendEntry(
            "продолжение полосы вдоль корешка (отсекает мостики страниц)",
            COLOUR_EXTENSION,
            alpha=ALPHA,
            style=SampleStyle.BOX,
        ),
        LegendEntry("линия корешка", COLOUR_SPINE),
        LegendEntry("подгоночная область страницы", COLOUR_FRAME),
    ]
    return np.vstack([header_strip(header, body.shape[1]), body, legend_strip(legend, body.shape[1], None)])
