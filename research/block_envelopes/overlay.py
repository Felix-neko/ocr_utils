"""Оверлеи стенда: блоки своим цветом каждый (полупрозрачная заливка и контур), оси строк, для сравнения — прежние границы тонкой линией; шапка и легенда в полях."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed

# Ширина оверлея полосы — как у разбора пака (``pack_analysis.final.OVERLAY_WIDTH``).
OVERLAY_WIDTH = 1600
# Блоки — по кругу из шести различимых цветов (BGR): соседние блоки должны отличаться, чтобы ложный
# разрез или слияние было видно сразу. Синий главной границы проекта идёт первым.
BLOCK_COLOURS = ((220, 90, 20), (0, 140, 0), (160, 60, 160), (0, 120, 230), (150, 150, 0), (40, 40, 200))
# Заливка блока — полупрозрачно: под ней читаются буквы.
FILL_ALPHA = 0.18
# Контур блока — толще и насыщеннее заливки, но тоже в слой, чтобы не закрывать буквы.
OUTLINE_ALPHA = 0.8
# Ось строки — зелёная проекта (``text_blocks.overlay.COLOUR_AXIS``).
COLOUR_AXIS = (40, 170, 40)
# Прежняя граница (боевой алгоритм) для сравнения — тонкий приглушённый серый.
COLOUR_REFERENCE = (120, 120, 120)
# Номер блока — чёрным на белой подложке.
COLOUR_TEXT = (20, 20, 20)


def _points(polygon: np.ndarray, scale: float) -> np.ndarray:
    """Контур в пикселях холста."""
    return np.round(np.asarray(polygon, dtype=np.float64) * scale).astype(np.int32)


def draw_blocks(
    gray: np.ndarray,
    polygons: list[np.ndarray],
    axes: list[np.ndarray] | None = None,
    reference: list[np.ndarray] | None = None,
    labels: list[str] | None = None,
) -> tuple[np.ndarray, float]:
    """Холст полосы: блоки, оси строк и прежние границы.

    Args:
        gray: Серая рабочая копия (пиксели рабочей копии — в них же все контуры).
        polygons: Контуры блоков ``(N, 2)``.
        axes: Ломаные осей строк или ``None``.
        reference: Контуры прежних блоков для сравнения или ``None``.
        labels: Подписи блоков (по одной на контур) или ``None`` — тогда номера по порядку.

    Returns:
        Пара ``(холст BGR, масштаб «рабочая копия → холст»)``.
    """
    scale = OVERLAY_WIDTH / gray.shape[1]
    small = cv2.resize(gray, (OVERLAY_WIDTH, int(round(gray.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    canvas = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    # Заливки блоков — одним слоем с малой альфой.
    fill = canvas.copy()
    for index, polygon in enumerate(polygons):
        if polygon is None or len(polygon) < 3:
            continue
        cv2.fillPoly(fill, [_points(polygon, scale)], BLOCK_COLOURS[index % len(BLOCK_COLOURS)])
    cv2.addWeighted(fill, FILL_ALPHA, canvas, 1.0 - FILL_ALPHA, 0, canvas)
    # Оси строк — тонко и непрозрачно: они не закрывают буквы.
    for axis in axes or []:
        if axis is not None and len(axis) >= 2:
            cv2.polylines(canvas, [_points(axis, scale)], False, COLOUR_AXIS, 1, cv2.LINE_AA)
    # Прежние границы — тонкой серой линией.
    for polygon in reference or []:
        if polygon is not None and len(polygon) >= 3:
            cv2.polylines(canvas, [_points(polygon, scale)], True, COLOUR_REFERENCE, 1, cv2.LINE_AA)
    # Контуры блоков — слоем, полупрозрачно.
    outline = canvas.copy()
    for index, polygon in enumerate(polygons):
        if polygon is None or len(polygon) < 3:
            continue
        colour = BLOCK_COLOURS[index % len(BLOCK_COLOURS)]
        cv2.polylines(outline, [_points(polygon, scale)], True, colour, 3, cv2.LINE_AA)
    cv2.addWeighted(outline, OUTLINE_ALPHA, canvas, 1.0 - OUTLINE_ALPHA, 0, canvas)
    # Номера блоков у левого верхнего угла контура, на белой подложке.
    for index, polygon in enumerate(polygons):
        if polygon is None or len(polygon) < 3:
            continue
        text = labels[index] if labels else str(index)
        x, y = _points(polygon, scale).min(axis=0)
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x, y = max(0, int(x) - w - 4), max(h + 2, int(y) + h)
        cv2.rectangle(canvas, (x - 1, y - h - 2), (x + w + 1, y + 3), (255, 255, 255), -1)
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLOCK_COLOURS[index % 6], 1, cv2.LINE_AA)
    return canvas, scale


def legend(with_reference: bool, block_word: str = "блок") -> list[LegendEntry]:
    """Строки легенды оверлея стенда.

    Args:
        with_reference: Нарисованы ли прежние границы.
        block_word: Как называть области (``блок`` у алгоритмов, ``регион движка`` у чужих движков).

    Returns:
        Строки легенды.
    """
    entries = [
        LegendEntry(
            f"{block_word}: заливка (цвета по кругу, соседние различаются)",
            BLOCK_COLOURS[0],
            FILL_ALPHA,
            SampleStyle.BOX,
        ),
        LegendEntry(f"{block_word}: граница", BLOCK_COLOURS[0], OUTLINE_ALPHA),
        LegendEntry("ось строки", COLOUR_AXIS),
    ]
    if with_reference:
        entries.append(LegendEntry("граница блока боевого алгоритма (для сравнения)", COLOUR_REFERENCE))
    return entries


def page_picture(
    gray: np.ndarray,
    polygons: list[np.ndarray],
    header: list[str],
    axes: list[np.ndarray] | None = None,
    reference: list[np.ndarray] | None = None,
    labels: list[str] | None = None,
    block_word: str = "блок",
) -> np.ndarray:
    """Готовая картинка полосы: холст :func:`draw_blocks`, шапка сверху и легенда снизу — в полях.

    Args:
        gray: Серая рабочая копия.
        polygons: Контуры блоков.
        header: Строки шапки.
        axes: Оси строк.
        reference: Прежние границы.
        labels: Подписи блоков.
        block_word: Как называть области в легенде.

    Returns:
        Картинка BGR.
    """
    canvas, _ = draw_blocks(gray, polygons, axes, reference, labels)
    return framed(canvas, header, legend(reference is not None, block_word))


__all__ = ["OVERLAY_WIDTH", "draw_blocks", "legend", "page_picture"]
