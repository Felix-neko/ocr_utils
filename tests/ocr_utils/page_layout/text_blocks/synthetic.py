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


def stuck_leader_page(
    rows: int = 12,
    number_w: int = 100,
    shift: int = 0,
    shape: tuple[int, int] = PAGE_SHAPE,
    dot_step: int = 36,
    dot_px: int = 6,
    stuck_gap: int = 14,
    seed: int = 3,
) -> np.ndarray:
    """Таблица с отточием, чьи две последние точки прилипли к числу: «слова . . . . ..число».

    Так на 1966/03 IMG_0131_2R: последние точки отточия стоят теснее обычного шага и при смыкании RLSA
    сливаются с числом, и ряд распадается на две строки над одними и теми же точками.

    Args:
        rows: Сколько строк таблицы.
        number_w: Ширина числа (цифры по 30 px с шагом 36): 100 — три цифры, 60 — две.
        shift: На сколько пикселей ниже ряда стоят число и прилипшие к нему точки (0 — на своём ряду).
        shape: Размер страницы (пиксели 300 dpi).
        dot_step: Шаг точек отточия.
        dot_px: Размер точки.
        stuck_gap: Шаг двух последних точек и зазор до числа.
        seed: Зерно генератора слов.

    Returns:
        Серую страницу.
    """
    page = paper(shape)
    generator = random.Random(seed)
    x_dots, x_number = MARGIN + 700, shape[1] - MARGIN - 300
    for index in range(rows):
        y = MARGIN + index * LINE_STEP * 2
        _draw_line(page, generator, MARGIN, MARGIN + 520, y, False)
        # Отточие обычным шагом, затем две точки вплотную к числу — на высоте числа.
        for x in range(x_dots, x_number - 2 * stuck_gap - dot_step, dot_step):
            page[y + GLYPH_H - dot_px : y + GLYPH_H, x : x + dot_px] = INK
        y_number = y + shift
        for x in (x_number - 2 * stuck_gap, x_number - stuck_gap):
            page[y_number + GLYPH_H - dot_px : y_number + GLYPH_H, x : x + dot_px] = INK
        for x in range(x_number, x_number + number_w, 36):
            page[y_number : y_number + GLYPH_H, x : x + 30] = INK
    return page


def single_line_page(shape: tuple[int, int] = PAGE_SHAPE, seed: int = 5) -> np.ndarray:
    """Страница с одной-единственной строкой: для проверки контура однострочного блока."""
    page = paper(shape)
    _draw_line(page, random.Random(seed), MARGIN, shape[1] - MARGIN, MARGIN + 200, False)
    return page


# Строчная «буква» синтетики с базовой линией (пиксели 300 dpi): высота строчной, прописной и
# выносных, ширина буквы и пробелы.
LETTER_X_H = 22
LETTER_CAP_H = 32
LETTER_TAIL = 10
LETTER_W = 16
LETTER_GAP = 4
WORD_GAP = 22


def glyph_line_page(
    shape: tuple[int, int] = (1600, 1400), step: int = 58, seed: int = 3, amplitude: float = 18.0
) -> tuple[np.ndarray, list[float]]:
    """Страница из строк с КЛАССАМИ букв и известной осью: строчные, прописные и цифры, выносные вниз
    («р», запятая) и вверх («б»), верхние индексы («м²»), точки; строки изогнуты дугой.

    При ``step`` около 58 px выносной вниз одной строки и выносной вверх следующей почти касаются:
    плотный набор, где ось по краске перескакивала бы или проседала.

    Args:
        shape: Размер страницы (пиксели 300 dpi).
        step: Межстрочный шаг.
        seed: Зерно генератора.
        amplitude: Прогиб дуги (пиксели 300 dpi): середина строк ниже краёв.

    Returns:
        ``(картинка, верхи строчных по строкам до изгиба)``: истинная ось строки ``i`` в точке ``x`` —
        ``верх_i + LETTER_X_H / 2 + bow_dy(x, ширина, amplitude)`` (пиксели 300 dpi).
    """
    image = paper(shape)
    generator = random.Random(seed)
    tops: list[float] = []
    for top in range(MARGIN, shape[0] - MARGIN - LETTER_CAP_H, step):
        base = top + LETTER_X_H
        tops.append(float(top))
        x = MARGIN
        while x < shape[1] - MARGIN - 6 * LETTER_W:
            for _ in range(generator.randint(3, 7)):
                kind = generator.random()
                if kind < 0.12:  # прописная или цифра: низ на базовой линии, верх выше строчной
                    image[base - LETTER_CAP_H : base, x : x + LETTER_W] = INK
                elif kind < 0.22:  # «р»: выносной вниз
                    image[top : base + LETTER_TAIL, x : x + LETTER_W] = INK
                elif kind < 0.32:  # «б»: выносной вверх
                    image[base - LETTER_CAP_H - 4 : base, x : x + LETTER_W] = INK
                elif kind < 0.36:  # верхний индекс: мелкий глиф высоко над базовой линией
                    image[top - 6 : top + 6, x : x + 8] = INK
                else:  # строчная без выносных
                    image[top:base, x : x + LETTER_W] = INK
                x += LETTER_W + LETTER_GAP
            if generator.random() < 0.3:  # точка или запятая на конце слова
                image[base - 5 : base + (6 if generator.random() < 0.5 else 0), x : x + 5] = INK
                x += 5 + LETTER_GAP
            x += WORD_GAP
    return bowed(image, amplitude), tops


def bow_dy(xs: np.ndarray, width: int, amplitude: float) -> np.ndarray:
    """Сдвиг строк вниз дугой :func:`bowed` в точках ``xs`` (пиксели 300 dpi)."""
    return amplitude * (1.0 - ((xs - width / 2.0) / (width / 2.0)) ** 2)
