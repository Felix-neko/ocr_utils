"""Тесты стенда research/line_art_boxes: варианты-подмены и добор отдельных штрихов к рамке (правила ядра — в tests/ocr_utils/page_layout)."""

from __future__ import annotations

import numpy as np

import ocr_utils.page_layout.analysis as layout_analysis
import ocr_utils.page_layout.line_art.features as art_features
import ocr_utils.page_layout.tables.detector as table_detector
from ocr_utils.page_layout.geometry import Box
from ocr_utils.units import mm_to_px
from research.line_art_boxes.cli import parse_px_per_mm, px_per_mm_of
from research.line_art_boxes.fix import (
    RING_REACH_MM,
    FixVariant,
    close_cover,
    grow_strokes,
    is_text_component,
    letter_boxes,
    mark_rows,
    page_ink,
    patched,
    ring_addable,
    stroke_like,
)

DPI = 150


def _row(ink: np.ndarray, y0: int, height: int, xs: list[tuple[int, int]]) -> None:
    """Нарисовать буквы-прямоугольники одной строки с просветом посередине (холст теста)."""
    for x0, x1 in xs:
        ink[y0 : y0 + height, x0:x1] = 1
        ink[y0 + height // 2, x0 + 1 : x1 - 1] = 0


def test_patched_restores_functions():
    """Подмены legacy и grow снимаются по выходу из контекста."""
    strips, letters, detect = table_detector.edge_strips, art_features.letters_in_row, layout_analysis.detect_line_art
    with patched(FixVariant.LEGACY):
        assert table_detector.edge_strips is not strips and art_features.letters_in_row is not letters
    with patched(FixVariant.GROW):
        assert layout_analysis.detect_line_art is not detect
    assert table_detector.edge_strips is strips
    assert art_features.letters_in_row is letters
    assert layout_analysis.detect_line_art is detect


def test_legacy_restores_fold_shadow_growth():
    """В legacy тень сгиба снова штрих (прежний line_art_mask), в base — нет."""
    binary = np.zeros((600, 400), np.uint8)
    binary[100:300, 100:103] = 255
    binary[60:500, 399] = 255
    assert table_detector.line_art_mask(binary, DPI)[:, 399].max() == 0
    with patched(FixVariant.LEGACY):
        assert table_detector.line_art_mask(binary, DPI)[:, 399].max() == 255


def test_px_per_mm_table():
    """Плотность по префиксам выпусков и значение по умолчанию."""
    table = parse_px_per_mm("01=17.9, 02=17.1")
    assert table == {"01": 17.9, "02": 17.1}
    assert px_per_mm_of("02 оплачено", table, None) == 17.1
    assert px_per_mm_of("13 оплачено", table, 16.0) == 16.0


def _spots(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Рамки и площади компонент краски синтетической полосы."""
    import cv2

    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    body = stats[1:]
    x, y = body[:, cv2.CC_STAT_LEFT], body[:, cv2.CC_STAT_TOP]
    spots = np.stack([x, y, x + body[:, cv2.CC_STAT_WIDTH], y + body[:, cv2.CC_STAT_HEIGHT]], axis=1).astype(np.int64)
    return spots, body[:, cv2.CC_STAT_AREA]


def test_grow_strokes_takes_ray_but_not_text_or_hyphen():
    """Луч у рамки добирается; строка букв и дефис переноса рядом — нет."""
    ink = np.zeros((400, 600), np.uint8)
    ink[100:300, 200:203] = 1  # левая сторона рисунка (сама рамка — box)
    ink[60:80, 250:253] = 1  # луч над рисунком: 20×3 px ≈ 3.4 мм, зазор 20 px ≈ 3.4 мм
    letter_h = mm_to_px(2.5, DPI)  # 15 px
    _row(ink, 150, letter_h, [(150, 158), (162, 170), (174, 182), (186, 194)])  # строка слева, вплотную к рамке
    ink[320:322, 240:250] = 1  # дефис под рисунком: 10×2 px ≈ 1.7 мм
    spots, areas = _spots(ink)
    letters = letter_boxes(ink, DPI)
    text = np.array([is_text_component(tuple(int(v) for v in s), letters, DPI) for s in spots])
    addable = stroke_like(spots, areas, DPI) & ~text
    grown, added = grow_strokes(Box(200, 100, 400, 300), spots, addable, DPI)
    assert added == 1
    assert grown.as_tuple() == (200, 60, 400, 300)


def test_text_component_dash_in_row():
    """Тире в строке у рисунка — текст (строка накрывает его центр), отдельный штрих — нет."""
    ink = np.zeros((200, 400), np.uint8)
    letter_h = mm_to_px(2.5, DPI)
    _row(ink, 50, letter_h, [(100, 108), (112, 120)])
    dash = (124, 57, 142, 59)
    ink[57:59, 124:142] = 1
    letters = letter_boxes(ink, DPI)
    assert is_text_component(dash, letters, DPI)
    assert not is_text_component((300, 150, 318, 152), letters, DPI)


def test_mark_rows_flags_asterisks_not_lone_dot():
    """Звёздочки «* * *» в ряд — знаки строки; одиночная точка у пиктограммы — нет."""
    spots = np.array([[100, 50, 106, 56], [125, 50, 131, 56], [150, 50, 156, 56], [300, 200, 307, 207]], np.int64)
    rows = mark_rows(spots, np.ones(4, bool), DPI)
    assert rows.tolist() == [True, True, True, False]


def test_ring_takes_dots_but_not_dash():
    """Вариант ring добирает точки у знака вопроса (над и под рамкой), но не дефис строки рядом."""
    ink = np.full((300, 400), 255, np.uint8)
    ink[100:200, 100:200] = 0  # «рисунок» — его рамка
    ink[103:197, 103:197] = 255
    ink[85:93, 146:154] = 0  # точка над рисунком (8 px ≈ 1.4 мм), зазор 7 px
    ink[208:216, 146:154] = 0  # точка под рисунком
    ink[150:153, 210:218] = 0  # дефис справа (8×3 px ≈ 1.4×0.5 мм)
    box = Box(100, 100, 200, 200)
    ink_info = page_ink(ink, DPI, [box])
    grown, added = grow_strokes(box, ink_info.spots, ring_addable(ink_info, DPI), DPI, (), RING_REACH_MM)
    assert added == 2
    assert grown.as_tuple() == (100, 85, 200, 216)


def test_close_cover_joins_nearby_ray():
    """Вариант close берёт луч в 2 мм от рамки по связности раздутой краски."""
    ink = np.full((300, 400), 255, np.uint8)
    ink[100:200, 100:103] = 0
    ink[100:200, 197:200] = 0
    ink[100:103, 100:200] = 0
    ink[197:200, 100:200] = 0
    ink[140:143, 208:230] = 0  # луч 22×3 px справа, зазор 8 px ≈ 1.4 мм
    box = Box(100, 100, 200, 200)
    info = page_ink(ink, DPI, [box])
    grown, added = close_cover(box, info, ring_addable(info, DPI), DPI)
    assert added == 1 and grown.x1 == 230
