"""Оверлеи зоны копирования: лист, четырёхугольник копирования, дорисовка, отрезанное квантилем; шапка и легенда в полях."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed
from research.newspaper_crop.zone import Zone, quad_mask

# Цвета BGR (палитра проекта, навык draw-overlay): не бумага — красный «отвергнуто», четырёхугольник
# копирования — синий «граница блока», дорисовка — оранжевый, лист за четырёхугольником — фиолетовый.
COLOUR_NOT_PAPER = (0, 0, 220)
COLOUR_QUAD = (220, 90, 20)
COLOUR_FILL = (0, 165, 255)
COLOUR_CUT = (200, 60, 200)
COLOUR_FINGER = (180, 105, 255)
# Складка корешка — зелёный «принято» палитры (линия, по которой отрезано).
COLOUR_FOLD = (0, 150, 0)
# Трапеция листа до отрезки — фиолетовый «крупная огибающая» палитры, пунктиром.
COLOUR_DETECTED = (200, 60, 200)
AREA_ALPHA = 0.45


def _tint(canvas: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int], alpha: float) -> None:
    """Полупрозрачно закрашивает область маски на холсте (на месте).

    Args:
        canvas: BGR-холст.
        mask: Булева маска того же размера.
        colour: Цвет BGR.
        alpha: Непрозрачность.
    """
    if not mask.any():
        return
    canvas[mask] = (canvas[mask] * (1 - alpha) + np.array(colour) * alpha).astype(np.uint8)


def zone_overlay(
    bgr: np.ndarray,
    zone: Zone,
    header: list[str],
    finger: np.ndarray | None = None,
    max_height: int = 2208,
    folds: list[np.ndarray] | None = None,
    detected_quad: np.ndarray | None = None,
) -> np.ndarray:
    """Оверлей зоны поверх кадра на рабочем масштабе.

    Args:
        bgr: Кадр BGR (любой масштаб — приводится к размеру маски зоны).
        zone: Зона кадра.
        header: Строки шапки.
        finger: Маска пальцев на рабочем масштабе или None.
        max_height: Наибольшая высота картинки (ужимается с сохранением пропорций).
        folds: Принятые складки корешка — ломаные в пикселях рабочей маски — или None.
        detected_quad: Трапеция листа, найденная до отрезки за складкой (пиксели маски), — рисуется пунктиром
            под зоной копирования; None — не рисовать.

    Returns:
        BGR-картинка с шапкой и легендой в полях.
    """
    h, w = zone.paper.shape
    canvas = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    qm = quad_mask(zone.quad, (h, w)) if zone.quad is not None else np.zeros((h, w), bool)
    _tint(canvas, ~zone.paper & ~qm, COLOUR_NOT_PAPER, AREA_ALPHA)
    _tint(canvas, qm & ~zone.paper, COLOUR_FILL, AREA_ALPHA)
    _tint(canvas, zone.paper & ~qm, COLOUR_CUT, AREA_ALPHA)
    if finger is not None:
        contours, _ = cv2.findContours(finger.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, COLOUR_FINGER, 2)
    for line in folds or []:
        cv2.polylines(canvas, [np.round(line).astype(np.int32)], False, COLOUR_FOLD, 2)
    if detected_quad is not None:
        _dashed_poly(canvas, detected_quad, COLOUR_DETECTED, 2)
    if zone.quad is not None:
        cv2.polylines(canvas, [np.round(zone.quad).astype(np.int32)], True, COLOUR_QUAD, 3)
    if h > max_height:
        canvas = cv2.resize(canvas, (int(w * max_height / h), max_height), interpolation=cv2.INTER_AREA)
    legend = [
        LegendEntry("не бумага (вне четырёхугольника)", COLOUR_NOT_PAPER, AREA_ALPHA, SampleStyle.BOX),
        LegendEntry("дорисовать: в четырёхугольнике, но не бумага", COLOUR_FILL, AREA_ALPHA, SampleStyle.BOX),
        LegendEntry("бумага за четырёхугольником (отрезано)", COLOUR_CUT, AREA_ALPHA, SampleStyle.BOX),
        LegendEntry("зона копирования" if detected_quad is not None else "четырёхугольник копирования", COLOUR_QUAD),
    ]
    if detected_quad is not None:
        legend.append(LegendEntry("трапеция листа (до отрезки за складкой)", COLOUR_DETECTED, style=SampleStyle.DASHED))
    if finger is not None:
        legend.append(LegendEntry("палец (детектор)", COLOUR_FINGER))
    if folds:
        legend.append(LegendEntry("складка корешка: за ней — соседняя полоса, отрезано", COLOUR_FOLD))
    return framed(canvas, header, legend)


def _dashed_poly(
    canvas: np.ndarray, poly: np.ndarray, colour: tuple[int, int, int], thickness: int, dash: int = 18
) -> None:
    """Замкнутый многоугольник пунктиром (на месте).

    Args:
        canvas: BGR-холст.
        poly: Вершины (N, 2).
        colour: Цвет BGR.
        thickness: Толщина, px.
        dash: Длина штриха и пропуска, px.
    """
    for i in range(len(poly)):
        a, b = np.asarray(poly[i], float), np.asarray(poly[(i + 1) % len(poly)], float)
        length = float(np.linalg.norm(b - a))
        steps = max(int(length // dash), 1)
        for k in range(0, steps, 2):
            p = a + (b - a) * (k / steps)
            q = a + (b - a) * (min(k + 1, steps) / steps)
            cv2.line(canvas, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), colour, thickness)
