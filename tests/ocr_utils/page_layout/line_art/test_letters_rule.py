"""Тесты правила A′ — слипшиеся буквы строки без текстовых блоков surya (``line_art.features.letters_in_row``, LINE_ART_VERSION 7)."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.line_art.features import analyse_gray, letter_boxes, letters_in_row, params_for_dpi
from ocr_utils.units import mm_to_px

DPI = 150
PARAMS = params_for_dpi(DPI)


def _row(ink: np.ndarray, y0: int, height: int, xs: list[tuple[int, int]]) -> None:
    """Нарисовать буквы-прямоугольники одной строки с просветом посередине (холст теста)."""
    for x0, x1 in xs:
        ink[y0 : y0 + height, x0:x1] = 1
        ink[y0 + height // 2, x0 + 1 : x1 - 1] = 0


def _letters(ink: np.ndarray) -> np.ndarray:
    """Буквы синтетической полосы."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    return letter_boxes(stats[1:], PARAMS)


def test_glued_heading_letters_are_text():
    """Слипшиеся буквы внутри строки заголовка (соседи с обеих сторон той же высоты) — текст."""
    ink = np.zeros((200, 600), np.uint8)
    height = mm_to_px(5.0, DPI)
    _row(ink, 50, height, [(70, 90), (96, 116), (122, 142), (268, 288), (294, 314)])
    ink[50 : 50 + height, 150:260] = 1
    assert letters_in_row((150, 50, 260, 50 + height), _letters(ink), PARAMS)


def test_isolated_spot_is_not_text():
    """Пятно без букв в своей строке — не текст."""
    ink = np.zeros((200, 600), np.uint8)
    ink[50:90, 150:260] = 1
    assert not letters_in_row((150, 50, 260, 90), _letters(ink), PARAMS)


def test_drawing_piece_among_hatching_is_not_text():
    """Кусок контура рисунка среди мелкой штриховки (соседи низкие и не на одной линии) — не текст."""
    ink = np.zeros((300, 600), np.uint8)
    ink[100:156, 200:260] = 1
    for x, y in ((180, 110), (186, 130), (270, 120), (276, 140)):
        ink[y : y + 8, x : x + 3] = 1
    assert not letters_in_row((200, 100, 260, 156), _letters(ink), PARAMS)


def test_ornament_beside_heading_is_kept():
    """Орнамент рубрики сбоку от заголовка (буквы с одной стороны, орнамент вдвое выше букв) — не текст."""
    ink = np.zeros((200, 600), np.uint8)
    _row(ink, 70, mm_to_px(5.0, DPI), [(200, 220), (226, 246), (252, 272)])
    ornament = (130, 55, 190, 55 + mm_to_px(10.0, DPI))
    ink[ornament[1] : ornament[3], ornament[0] : ornament[2]] = 1
    assert not letters_in_row(ornament, _letters(ink), PARAMS)


def _outline(gray: np.ndarray, x0: int, y0: int, x1: int, y1: int, columns: int = 0) -> None:
    """Контур буквы штрихом 3 px (и внутренние вертикали — у слипшихся букв) на серой полосе теста."""
    gray[y0:y1, x0 : x0 + 3] = 0
    gray[y0:y1, x1 - 3 : x1] = 0
    gray[y0 : y0 + 3, x0:x1] = 0
    gray[y1 - 3 : y1, x0:x1] = 0
    for k in range(1, columns + 1):
        x = x0 + k * (x1 - x0) // (columns + 1)
        gray[y0:y1, x : x + 3] = 0


def _heading_page() -> np.ndarray:
    """Серая полоса 150 dpi: строка заголовка 7.8 мм, в ней пятно слипшихся букв 110×46 px, прошедшее ворота пятна."""
    gray = np.full((400, 700), 255, np.uint8)
    height = 46  # 7.8 мм: выше ворот min_side (2 шага строк ≈ 7.5 мм), но ещё буква (до 8 мм)
    # Крупные буквы заголовка сами проходят ворота пятна (как на ХиП 1991) — их тоже снимает A′.
    for x0 in (90, 120, 150, 300, 330):
        _outline(gray, x0, 100, x0 + 20, 100 + height)
    _outline(gray, 180, 100, 290, 100 + height, columns=2)
    return gray


def test_analyse_gray_drops_glued_letters_without_surya_only():
    """Без текстовых блоков пятно слипшихся букв отсеивается правилом A′; с блоками surya A′ не включается."""
    gray = _heading_page()
    without = analyse_gray(gray, PARAMS)
    assert without.dropped.get("слипшиеся буквы строки (без surya)", 0) >= 1
    assert not without.candidates
    # Блок surya далеко от строки: правило A буквы не берёт, A′ выключено — пятно слипшихся букв остаётся кандидатом.
    with_blocks = analyse_gray(gray, PARAMS, text_boxes=[(0, 350, 50, 399)])
    assert (180, 100, 290, 146) in [c.box for c in with_blocks.candidates]
