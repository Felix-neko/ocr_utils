"""Тесты признаков стенда ложных сирот: настоящая линейка, линейка с прилипшими буквами, стопка стволов через межстрочье, строка засечек."""

import numpy as np

from ocr_utils.page_layout.geometry import LooseRule
from research.loose_rules.features import page_ink, rule_features

DPI = 150


def _page() -> np.ndarray:
    """Чистая серая полоса 150 dpi, бумага 240."""
    return np.full((600, 600), 240, np.uint8)


def test_clean_rule_has_no_glyph_share():
    """Отбивка на чистой бумаге: вся трасса — одна вытянутая компонента, доля букв ноль."""
    gray = _page()
    gray[300:302, 100:400] = 20
    rule = LooseRule(((100.0, 300.5), (399.0, 300.5)), True, 2.0)
    features = rule_features(page_ink(gray), rule, DPI, [])
    assert features.glyph_share_20 == 0.0
    assert features.coverage > 0.95
    assert features.rule_run_mm > 45


def test_underline_touching_descenders_stays_rule():
    """Подчёркивание, к которому прилипли хвосты букв: компонента всё равно вытянута — не буквы."""
    gray = _page()
    gray[300:302, 100:400] = 20
    for x in range(110, 390, 30):
        gray[285:302, x : x + 3] = 20
    rule = LooseRule(((100.0, 300.5), (399.0, 300.5)), True, 2.0)
    assert rule_features(page_ink(gray), rule, DPI, []).glyph_share_20 == 0.0


def test_stacked_stems_across_rows_are_glyphs():
    """Стволы двух «Н» заголовка друг над другом через межстрочье: обе компоненты компактны, разрыв на межстрочье."""
    gray = _page()
    for top in (200, 245):
        # «Н» высотой 30 px (5 мм), шириной 24 px, стволы по 6 px, перекладина посередине.
        gray[top : top + 30, 300:306] = 20
        gray[top : top + 30, 318:324] = 20
        gray[top + 13 : top + 17, 300:324] = 20
        # Соседняя буква той же строки — межстрочье видно по сторонам.
        gray[top : top + 30, 340:360] = 20
    rule = LooseRule(((303.0, 200.0), (303.0, 274.0)), False, 6.0)
    features = rule_features(page_ink(gray), rule, DPI, [])
    assert features.glyph_share_20 == 1.0
    assert features.gaps == 1
    assert features.interline_gaps == 1


def test_serif_row_is_glyphs():
    """Низы засечек жирных букв одной строки, сшитые через просветы: каждая буква компактна."""
    gray = _page()
    for x in range(100, 380, 40):
        gray[270:300, x : x + 30] = 20
        gray[275:295, x + 8 : x + 22] = 240  # просвет внутри буквы
    rule = LooseRule(((100.0, 298.0), (369.0, 298.0)), True, 3.0)
    assert rule_features(page_ink(gray), rule, DPI, []).glyph_share_20 > 0.9


def test_formula_box_marks_rule():
    """Центр линейки внутри рамки формулы — флаг ``in_formula``."""
    gray = _page()
    gray[300:302, 100:200] = 20
    rule = LooseRule(((100.0, 300.5), (199.0, 300.5)), True, 2.0)
    assert rule_features(page_ink(gray), rule, DPI, [(80, 250, 250, 350)]).in_formula
