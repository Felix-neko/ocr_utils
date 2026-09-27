"""Заголовки и подписи автора: балл разницы набора, потоки бок о бок, разрыв в кеглях, черта под рядами, кегль у стыка."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ocr_utils.page_layout.text_blocks import blocks, zones
from ocr_utils.page_layout.text_blocks.pieces import end_kegl
from ocr_utils.page_layout.text_blocks.segment import SCALES, Rule, _crosses
from tests.ocr_utils.page_layout.text_blocks.test_curved_layout import _row
from tests.ocr_utils.page_layout.text_blocks.test_zones import _piece

DPI = 150.0


def _styled(x0: float, x1: float, y: float, height: float, glyph: float, weight: float) -> blocks.Row:
    """Ряд с заданными кеглем (высота глифа) и жирностью."""
    return replace(_row(x0, x1, [y, y], height), glyph_h=glyph, weight=weight)


def test_style_distance_sums_excess_of_size_and_weight():
    """Балл — сумма превышений отношений над единицей: одинаковый набор даёт ноль."""
    same = _styled(0, 100, 50, 20, 19.0, 4.6)
    assert blocks.style_distance(same, same) == 0.0
    author = _styled(0, 100, 70, 18, 15.0, 2.2)
    # «современную» / «АЛФЕРЬЕВ» (1966/01 IMG_0017_1L): 19/15 − 1 + 4.6/2.2 − 1.
    expected = (19.0 / 15.0 - 1.0) + (4.6 / 2.2 - 1.0)
    assert abs(blocks.style_distance(same, author) - expected) < 1e-9


def test_silent_glyph_size_falls_back_to_row_height():
    """Кегль, который не намерился (слипшиеся жирные буквы), берётся долей высоты ряда."""
    silent = _styled(0, 100, 50, 30, 0.0, 4.6)
    assert blocks.row_glyph_size(silent) == blocks.FALLBACK_HEIGHT * 30


def test_bold_headline_beside_author_makes_a_pair_but_table_cells_do_not():
    """Заголовок и подпись того же кегля, но вдвое жирнее — пара бок о бок; графы таблицы — нет."""
    headline = _styled(400, 700, 150, 26, 19.0, 4.6)
    author = _styled(120, 260, 160, 18, 15.0, 2.2)
    assert blocks._side_by_side([headline, author], DPI) == {0, 1}
    # Цифры графы крупнее строчных в 1.4 раза при той же жирности — балл ниже порога.
    text = _styled(100, 400, 300, 14, 10.0, 2.0)
    number = _styled(600, 650, 300, 14, 14.0, 2.1)
    assert blocks._side_by_side([text, number], DPI) == set()


def test_same_style_headline_gap_is_measured_in_glyphs():
    """Разрыв строк одного заголовка меряется кеглями: скачущая высота ряда его не режет."""
    upper = _styled(395, 546, 134, 28, 19.75, 4.54)
    lower = _styled(395, 729, 202, 21, 19.0, 4.55)
    assert blocks._large_gap_limit(upper, lower) >= blocks.SAME_STYLE_GAP_GLYPHS * 19.75
    logo = _styled(395, 546, 134, 60, 40.0, 9.0)
    assert blocks._large_gap_limit(logo, lower) == blocks.GAP_HEIGHTS * 21


def test_rule_under_the_author_does_not_cut_the_headline_beside():
    """Подчёркивание фамилии слева не режет заголовок справа: черта обязана лежать под рядами."""
    first = _styled(395, 657, 76, 35, 19.0, 4.8)
    second = _styled(395, 546, 134, 28, 19.0, 4.5)
    underline = Rule(x0=70, y0=110, x1=290, y1=112)
    assert not blocks._rule_between(first, second, [underline], (0, 897))
    separator = Rule(x0=40, y0=110, x1=500, y1=112)  # черта через колонку, под обоими рядами
    assert blocks._rule_between(first, second, [separator], (0, 897))


def test_end_kegl_uses_the_edge_letters():
    """Кегль у конца куска — по крайним буквам (вторая по малости высота), а не по усреднённому иксу."""
    piece = _piece(0.0, 100.0, letters=6, x_h=11.0)
    sizes = piece.sizes.copy()
    sizes[-4:, 1] = 15.0
    piece = replace(piece, sizes=sizes)
    assert end_kegl(piece, at_start=False) == 15.0
    assert end_kegl(piece, at_start=True) == 11.0
    # По одной букве с точкой кегль не мерится.
    assert end_kegl(_piece(0.0, 100.0, letters=1, x_h=11.0), at_start=True) is None


def test_mixed_kegl_is_not_linked_across_a_wide_gap():
    """Разный кегль у стыка: через широкий зазор не сцепляется, через обычный пробел — да."""
    ink = np.zeros((10, 10), dtype=bool)
    small = _piece(0.0, 100.0, letters=4, x_h=15.0)
    far = _piece(small.x1 + 90.0, 100.0, letters=5, x_h=20.0)
    near = _piece(small.x1 + 12.0, 100.0, letters=5, x_h=20.0)
    verdict = zones._verdict_of(small, far, SCALES[0], [], None, ink, 2.0, _crosses)
    assert verdict is zones.LinkVerdict.MIXED_GAP
    assert zones._verdict_of(small, near, SCALES[0], [], None, ink, 2.0, _crosses) is zones.LinkVerdict.ACCEPTED
