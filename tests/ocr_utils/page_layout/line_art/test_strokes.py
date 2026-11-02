"""Тесты добора отдельных штрихов рисунка к рамке line art (``line_art.strokes``, LINE_ART_VERSION 8)."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.geometry import Box
from ocr_utils.page_layout.line_art.features import letter_boxes, params_for_dpi
from ocr_utils.page_layout.line_art.strokes import cover_strokes, drawing_shaped, is_text_component, mark_rows, page_ink
from ocr_utils.units import mm_to_px

DPI = 150


def _frame(ink: np.ndarray, box: tuple[int, int, int, int]) -> None:
    """Рамка-«рисунок» штрихом 3 px на битональной полосе (краска 0)."""
    x0, y0, x1, y1 = box
    ink[y0:y1, x0 : x0 + 3] = 0
    ink[y0:y1, x1 - 3 : x1] = 0
    ink[y0 : y0 + 3, x0:x1] = 0
    ink[y1 - 3 : y1, x0:x1] = 0


def test_cover_takes_dots_and_ray_but_not_dash():
    """К рамке добираются точки над и под ней и луч сбоку; дефис рядом — нет."""
    page = np.full((300, 400), 255, np.uint8)
    _frame(page, (100, 100, 200, 200))
    page[85:93, 146:154] = 0  # точка над рисунком, 8 px ≈ 1.4 мм, зазор 7 px
    page[208:216, 146:154] = 0  # точка под рисунком
    page[140:143, 208:230] = 0  # луч справа 22×3 px, зазор 8 px
    page[250:253, 120:128] = 0  # дефис ниже (8×3 px ≈ 1.4×0.5 мм), зазор 50 px — вне досягаемости
    page[170:173, 92:98] = 0  # дефис слева вплотную (6×3 px)
    ((box, added),) = cover_strokes([Box(100, 100, 200, 200)], page, DPI)
    assert added == 3
    assert box.as_tuple() == (100, 85, 230, 216)


def test_flat_box_is_not_grown():
    """Плоская рамка (кусок таблицы, штамп поверх строки) не дорастает."""
    page = np.full((300, 600), 255, np.uint8)
    _frame(page, (50, 100, 550, 160))
    page[85:93, 300:308] = 0
    box = Box(50, 100, 550, 160)
    assert not drawing_shaped(box, DPI)
    assert cover_strokes([box], page, DPI) == [(box, 0)]


def test_barrier_blocks_growth():
    """Компонента в растре или таблице не добирается."""
    page = np.full((300, 400), 255, np.uint8)
    _frame(page, (100, 100, 200, 200))
    page[140:143, 208:230] = 0
    ((box, added),) = cover_strokes([Box(100, 100, 200, 200)], page, DPI, [Box(205, 120, 260, 160)])
    assert added == 0 and box.as_tuple() == (100, 100, 200, 200)


def test_mark_rows_flags_asterisks_not_lone_dot():
    """Звёздочки «* * *» в ряд — знаки строки; одиночная точка — нет."""
    spots = np.array([[100, 50, 106, 56], [125, 50, 131, 56], [150, 50, 156, 56], [300, 200, 307, 207]], np.int64)
    assert mark_rows(spots, np.ones(4, bool), DPI).tolist() == [True, True, True, False]


def test_dash_in_text_row_is_text():
    """Тире в строке у рисунка — текст (буквы строки накрывают его центр), отдельный штрих — нет."""
    ink = np.zeros((200, 400), np.uint8)
    height = mm_to_px(2.5, DPI)
    for x0, x1 in ((100, 108), (112, 120)):
        ink[50 : 50 + height, x0:x1] = 1
        ink[50 + height // 2, x0 + 1 : x1 - 1] = 0
    ink[57:59, 124:142] = 1
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    letters = letter_boxes(stats[1:], params_for_dpi(DPI))
    assert is_text_component((124, 57, 142, 59), letters, DPI)
    assert not is_text_component((300, 150, 318, 152), letters, DPI)


def test_letters_inside_drawing_are_labels():
    """Буквы глубоко внутри рамки — надпись рисунка: штрих за кромкой на их высоте добирается (не «буква их строки»)."""
    page = np.full((300, 400), 255, np.uint8)
    _frame(page, (100, 100, 260, 220))
    for x0 in (200, 215, 230):  # надпись внутри рисунка, буквы 2.5 мм, до кромки 20 px ≈ 3.4 мм
        page[150:165, x0 : x0 + 10] = 0
        page[156:159, x0 + 2 : x0 + 8] = 255
    page[150:165, 263:266] = 0  # линия движения за кромкой на высоте надписи, 15×3 px
    ((box, added),) = cover_strokes([Box(100, 100, 260, 220)], page, DPI)
    assert added == 1 and box.x1 == 266


def test_text_column_stops_growth_but_axis_labels_are_taken():
    """Рисунок вплотную к колонке текста: первая буква строки в 4.4 мм от следующей не добирается (строки колонки —
    граница стороны), а засечки оси под рисунком добираются."""
    page = np.full((600, 700), 255, np.uint8)
    _frame(page, (100, 100, 300, 400))
    # Колонка справа: 6 строк, первая буква в 5 px от рамки, дальше разгонка 26 px (4.4 мм) и слова.
    for row in range(6):
        y = 150 + row * 30
        page[y : y + 12, 305:316] = 0  # первая буква строки
        page[y + 4 : y + 7, 307:314] = 255
        for x0 in (342, 354, 366, 378):
            page[y : y + 12, x0 : x0 + 10] = 0
            page[y + 4 : y + 7, x0 + 2 : x0 + 8] = 255
    # Засечки оси (штрихи 3×12 px) и под ними трёхзначные числа оси на одной базовой линии.
    for x0 in (120, 180, 240):
        page[403:415, x0 : x0 + 3] = 0
        for d in range(3):
            page[418:430, x0 - 14 + d * 10 : x0 - 6 + d * 10] = 0
            page[422:425, x0 - 12 + d * 10 : x0 - 8 + d * 10] = 255
    ((box, added),) = cover_strokes([Box(100, 100, 300, 400)], page, DPI)
    assert box.x1 == 300  # колонка справа не взята
    assert box.y1 == 415 and added == 3  # засечки оси взяты


def test_column_rows_counts_baselines_not_rows():
    """Ряды на одной базовой линии (числа оси через широкие пробелы) — одна строка; ряды друг под другом — колонка."""
    from ocr_utils.page_layout.line_art.strokes import column_rows

    one_line = np.array([[110, 410, 118, 422, 0], [170, 410, 178, 422, 1], [230, 411, 238, 422, 2]], np.int64)
    stacked = np.array([[305, y, 315, y + 12, k] for k, y in enumerate((150, 180, 210))], np.int64)
    assert column_rows(one_line, Box(100, 100, 300, 400), Box(100, 100, 300, 425), line_gap=6) == 1
    assert column_rows(stacked, Box(100, 100, 300, 400), Box(100, 100, 320, 400), line_gap=6) == 3
