"""Выпрямление вырезки находки детектора таблиц вдоль осей её изогнутых горизонтальных линеек — признаки таблицы считаются по прямым линейкам.

ЗАЧЕМ. На холсте Кунса ``scan_cropping.v2`` (выход страниц) низ страницы может остаться изогнутым: у таблицы ХиП 1991/02
IMG_1161, стр. 116, горизонтальные линейки прогнуты «улыбкой» на ~2 мм на 107 мм, разброс углов отрезков 3.5°. Затравку
находит поиск линеек ``rules.find_rules`` (он идёт вслед за изгибом), а проверка ``verify.table_features`` ищет в вырезке
прямые линейки (``find_lines``): изогнутые рвутся на куски на разной высоте, сетка получает лишние ряды, ``edge_fill``
падал до 0.42 при пороге 0.45 — «не таблица». Здесь вырезка выпрямляется по осям длинных горизонтальных линеек, и
признаки считаются уже по ней (``detector._find``, TABLES_VERSION 8). Стенд ``research/line_art_boxes``: таблиц ХиП 1991
8 → 11 без ложных, пак-1 с surya (3073 страницы, 221 таблица) — без смен.
"""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.tables import rules
from ocr_utils.page_layout.tables.ruling import binarize
from ocr_utils.units import mm_to_px

# Горизонтальная линейка годится в опору выпрямления, если она не короче этой доли ширины вырезки.
TRACE_MIN_SPAN = 0.3
# Полоса поиска линейки поперёк её рамки, мм: рамка изогнутой линейки — охват дуги, ось ищется внутри.
TRACE_BAND_MM = 1.0
# Изгиб меньше этого (размах оси по высоте), мм, — выпрямлять незачем.
MIN_SAG_MM = 0.4
# Степень многочлена оси линейки по x: дуга «улыбки» и лёгкий S — до третьей степени.
TRACE_DEGREE = 3


def rule_axis(binary: np.ndarray, box, dpi: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Ось горизонтальной линейки в её рамке: по каждому столбцу — середина краски в полосе, затем многочлен.

    Args:
        binary: Краска вырезки (255 — краска).
        box: Рамка линейки ``Box`` (охват дуги).
        dpi: Разрешение.

    Returns:
        ``(xs, ys)`` — столбцы рамки и сглаженная ось, или None — краски мало.
    """
    band = mm_to_px(TRACE_BAND_MM, dpi)
    y0, y1 = max(0, box.y0 - band), min(binary.shape[0], box.y1 + band)
    patch = binary[y0:y1, box.x0 : box.x1] > 0
    if patch.size == 0:
        return None
    cols = np.flatnonzero(patch.any(axis=0))
    if len(cols) < 0.5 * patch.shape[1] or len(cols) < 8:
        return None
    ys = np.array([np.mean(np.flatnonzero(patch[:, c])) for c in cols]) + y0
    xs = cols + box.x0
    coef = np.polyfit(xs, ys, min(TRACE_DEGREE, max(1, len(xs) // 20)))
    grid = np.arange(box.x0, box.x1)
    return grid, np.polyval(coef, grid)


def straighten(gray: np.ndarray, dpi: int) -> tuple[np.ndarray, float]:
    """Выпрямить вырезку таблицы вдоль её длинных горизонтальных линеек.

    Оси линеек (:func:`rule_axis`) дают отклонение ``d(x)`` от их средней высоты; между линейками отклонение
    интерполируется по y линейно, за крайними — как у крайней. Пиксель ``(x, y)`` выпрямленной вырезки берётся из
    ``(x, y + d(x, y))``.

    Args:
        gray: Серая вырезка находки.
        dpi: Разрешение.

    Returns:
        ``(выпрямленная вырезка, наибольший изгиб линеек, мм)``; без изгиба — вырезка как есть.
    """
    height, width = gray.shape[:2]
    binary = binarize(gray)
    found = rules.find_rules(gray, dpi, binary, rules.fragments(binary, dpi))
    axes = []
    for segment in found.horizontal:
        if segment.box.width < TRACE_MIN_SPAN * width:
            continue
        traced = rule_axis(binary, segment.box, dpi)
        if traced is None:
            continue
        xs, ys = traced
        dev = np.full(width, np.nan)
        dev[xs.clip(0, width - 1)] = ys - ys.mean()
        # За концами линейки — отклонение крайней точки (дуга продолжается плоско).
        valid = np.flatnonzero(np.isfinite(dev))
        dev = np.interp(np.arange(width), valid, dev[valid])
        axes.append((float(ys.mean()), dev))
    if not axes:
        return gray, 0.0
    sag = max(float(np.ptp(d)) for _, d in axes) * 25.4 / dpi
    if sag < MIN_SAG_MM:
        return gray, sag
    axes.sort(key=lambda a: a[0])
    levels = np.array([a[0] for a in axes])
    devs = np.stack([a[1] for a in axes])
    rows = np.arange(height, dtype=np.float64)
    # Отклонение в каждой клетке: линейная интерполяция между линейками по y, за крайними — как у крайней.
    weights = np.clip(np.interp(rows, levels, np.arange(len(levels))), 0, len(levels) - 1)
    lower = np.floor(weights).astype(int)
    upper = np.minimum(lower + 1, len(levels) - 1)
    t = (weights - lower)[:, None]
    field = (1 - t) * devs[lower] + t * devs[upper]
    map_x = np.repeat(np.arange(width, dtype=np.float32)[None, :], height, axis=0)
    map_y = (rows[:, None] + field).astype(np.float32)
    flat = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return flat, sag


def straightened(gray: np.ndarray, dpi: int) -> np.ndarray:
    """Вырезка находки для признаков таблицы: выпрямленная вдоль изогнутых линеек, если изгиб не меньше ``MIN_SAG_MM``.

    Args:
        gray: Серая вырезка находки.
        dpi: Разрешение.

    Returns:
        Выпрямленная вырезка или сама ``gray`` (изгиба нет или линеек-опор нет).
    """
    flat, sag = straighten(gray, dpi)
    return flat if sag >= MIN_SAG_MM else gray


__all__ = ["MIN_SAG_MM", "TRACE_BAND_MM", "TRACE_DEGREE", "TRACE_MIN_SPAN", "rule_axis", "straighten", "straightened"]
