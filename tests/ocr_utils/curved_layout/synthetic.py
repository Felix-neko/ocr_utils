"""Синтетические страницы с известной вёрсткой: колонки, выключка, кривизна строк."""

from __future__ import annotations

import random
from typing import Callable

import cv2
import numpy as np

from tests.ocr_utils.page_layout.orientation.synthetic import (
    DESCENDER_H,
    GAP_RANGE,
    GLYPH_H,
    GLYPH_W_RANGE,
    INK,
    LINE_STEP,
    MARGIN,
    PAGE_SHAPE,
    paper,
)
from tests.ocr_utils.scan_markup.synthetic import PAPER

# Ширина межколонника в пикселях 300 dpi (≈ 4 мм) — как в журнале пака-1.
GUTTER_PX = 48


def column_page(
    columns: int = 2, shape: tuple[int, int] = PAGE_SHAPE, justify: str = "both", seed: int = 0, indent: int = 40
) -> np.ndarray:
    """Страница в ``columns`` колонок с заданной выключкой.

    Args:
        columns: Число колонок.
        shape: Размер страницы (пиксели 300 dpi).
        justify: ``both`` — по формату, ``left`` — рваный правый край, ``right`` — рваный левый.
        seed: Зерно генератора длин слов.
        indent: Абзацный отступ первой строки каждого пятого ряда (пиксели).

    Returns:
        Серое изображение страницы: бумага ``PAPER``, краска ``INK``.
    """
    image = paper(shape)
    generator = random.Random(seed)
    usable = shape[1] - 2 * MARGIN - GUTTER_PX * (columns - 1)
    width = usable // columns
    for column in range(columns):
        x0 = MARGIN + column * (width + GUTTER_PX)
        x1 = x0 + width
        row = 0
        for y in range(MARGIN, shape[0] - MARGIN - GLYPH_H - DESCENDER_H, LINE_STEP):
            ragged = generator.randint(0, 90) if justify != "both" else 0
            start = x0 + (indent if row % 5 == 0 and justify != "right" else 0)
            end = x1
            if justify == "left":
                end = x1 - ragged
            elif justify == "right":
                start = x0 + ragged
            # Выключка вправо: правый край ровный, поэтому последнее слово прижимается к нему.
            _draw_line(image, generator, start, end, y, justify in ("both", "right"))
            row += 1
    return image


def _draw_line(image: np.ndarray, generator: random.Random, x0: int, x1: int, y: int, justify: bool) -> None:
    """Нарисовать одну строку слов между ``x0`` и ``x1``; при ``justify`` последнее слово к правому краю."""
    x = x0
    words: list[tuple[int, int]] = []
    while True:
        width = generator.randint(*GLYPH_W_RANGE) * 3
        if x + width > x1:
            break
        words.append((x, width))
        x += width + generator.randint(*GAP_RANGE) + 6
    if not words:
        return
    if justify and len(words) > 1:
        # Выключка по формату: последнее слово подвинуто вплотную к правому краю.
        last_x, last_width = words[-1]
        words[-1] = (x1 - last_width, last_width)
    for start, width in words:
        image[y : y + GLYPH_H, start : start + width] = INK


def warp(image: np.ndarray, dy: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
    """Сдвинуть строки по вертикали полем ``dy(x, y)`` (как в синтетике ``curved_lines``)."""
    height, width = image.shape
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_y = (ys - dy(xs, ys)).astype(np.float32)
    return cv2.remap(image, xs, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=PAPER)


def bowed(image: np.ndarray, amplitude_px: float = 30.0) -> np.ndarray:
    """Дуга: середина строк ниже краёв на ``amplitude_px``."""
    width = image.shape[1]
    return warp(image, lambda xs, ys: amplitude_px * (1.0 - ((xs - width / 2.0) / (width / 2.0)) ** 2))


def sheared(image: np.ndarray, rise_px: float = 26.0) -> np.ndarray:
    """Перекос: правый край строк ниже левого на ``rise_px`` (наклон постоянный по всей странице).

    Такой перекос ломает жадную сборку: сгустки соседних строк начинают чередоваться по x, и
    цепочка уходит на соседнюю строку.
    """
    width = image.shape[1]
    return warp(image, lambda xs, ys: rise_px * xs / width)


def inset_page(shape: tuple[int, int] = PAGE_SHAPE, seed: int = 0) -> np.ndarray:
    """Одна колонка на всю ширину внизу и врезка справа вверху: межколонник живёт на части высоты."""
    image = paper(shape)
    generator = random.Random(seed)
    split = shape[0] // 2
    main_right = shape[1] - MARGIN - 600
    for y in range(MARGIN, split - GLYPH_H, LINE_STEP):
        _draw_line(image, generator, MARGIN, main_right, y, True)
        _draw_line(image, generator, main_right + GUTTER_PX * 3, shape[1] - MARGIN, y, True)
    for y in range(split, shape[0] - MARGIN - GLYPH_H - DESCENDER_H, LINE_STEP):
        _draw_line(image, generator, MARGIN, shape[1] - MARGIN, y, True)
    return image


__all__ = ["GUTTER_PX", "bowed", "column_page", "inset_page", "warp"]


def leader_page(
    rows: int = 14, shape: tuple[int, int] = PAGE_SHAPE, dot_step: int = 36, dot_px: int = 6, seed: int = 3
) -> np.ndarray:
    """Страница-таблица: слева слова, дальше отточие «. . . . .», справа число.

    Геометрия взята с 1971/10 с.93: шаг между точками 36 px при 300 dpi (3 мм), точка 6 × 6 px.
    Между отточием и числом остаётся широкая пустота — она не должна стать межколонником.

    Args:
        rows: Сколько строк таблицы нарисовать.
        shape: Размер страницы (пиксели 300 dpi).
        dot_step: Шаг между точками отточия.
        dot_px: Размер точки.
        seed: Зерно генератора слов.

    Returns:
        Серую страницу.
    """
    page = paper(shape)
    generator = random.Random(seed)
    x_dots, x_number = MARGIN + 700, shape[1] - MARGIN - 200
    for index in range(rows):
        y = MARGIN + index * LINE_STEP * 2
        _draw_line(page, generator, MARGIN, MARGIN + 520, y, False)
        for x in range(x_dots, x_number - 3 * dot_step, dot_step):
            page[y + GLYPH_H - dot_px : y + GLYPH_H, x : x + dot_px] = INK
        _draw_line(page, generator, x_number, x_number + 150, y, False)
    return page


def single_line_page(shape: tuple[int, int] = PAGE_SHAPE, seed: int = 5) -> np.ndarray:
    """Страница с одной-единственной строкой: для проверки контура однострочного блока."""
    page = paper(shape)
    _draw_line(page, random.Random(seed), MARGIN, shape[1] - MARGIN, MARGIN + 200, False)
    return page
