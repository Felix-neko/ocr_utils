"""Итоговый оверлей полосы (``pack_analysis.final``): блоки по второй оси и дополнительные линии левой и правой стороны."""

from __future__ import annotations

import numpy as np
import pytest

from ocr_utils.page_layout.image import PageImage, Variant
from ocr_utils.page_layout.pack_analysis.final import SIDE_COLOR, draw, side_lines, text_blocks
from ocr_utils.page_layout.text_blocks.page import AxisKind
from ocr_utils.page_layout.text_blocks.sides import SideKind
from tests.ocr_utils.page_layout.text_blocks.synthetic import column_page

# Синтетическая страница рисуется в 300 dpi — это и родное разрешение кадра.
DPI = 300


@pytest.fixture(scope="module")
def page():
    """Колонка по формату как полоса: картинка, запись стадии кандидатов без объектов, разбор по второй оси."""
    image = PageImage.from_array(column_page(columns=1, justify="both"), DPI, Variant.SHARPENED, "1973/06/SYN")
    record = {"page": "1973/06/SYN", "loose_rules": []}
    analysis, hints = text_blocks(image, record, [], AxisKind.BODY)
    return image, record, analysis, hints


def test_blocks_are_built_on_the_second_axis(page):
    """При ``AxisKind.BODY`` основной осью строк стала вторая: прежняя лежит в ``centre_points``."""
    _, _, analysis, _ = page
    assert analysis.blocks
    assert any(axis.centre_points is not None for axis in analysis.axes)


def test_side_lines_are_built_for_both_sides_of_a_justified_column(page):
    """У колонки по формату обе дополнительные линии строятся и без заплаток."""
    _, _, analysis, _ = page
    lines = side_lines(analysis)
    assert len(lines) == len(analysis.blocks)
    block = max(range(len(lines)), key=lambda i: len(analysis.blocks[i].rows))
    for side in (SideKind.LEFT, SideKind.RIGHT):
        line = lines[block][side]
        assert line is not None
        assert not line.filled.any()


def test_overlay_shows_side_lines_in_their_colours(page):
    """На оверлее с линиями сторон появляются их цвета (смешанные с бумагой), без линий — картинка другая."""
    image, record, analysis, hints = page
    lines = side_lines(analysis)
    with_sides = draw(image, analysis, hints, record, [], [], {}, lines, body_axis=True)
    without = draw(image, analysis, hints, record, [], [], {}, None, body_axis=True)
    # Легенда сторон добавляет строки в поле под страницей.
    assert with_sides.shape[0] > without.shape[0]
    # Линия подмешана полупрозрачно к белой бумаге: у левой (золотистой) синий канал низкий, красный высокий.
    top = with_sides[: without.shape[0] - 200]
    gold = (top[..., 0] < 150) & (top[..., 2] > 200) & (top[..., 1] > 150)
    assert gold.sum() > 50
    assert SIDE_COLOR[SideKind.LEFT] != SIDE_COLOR[SideKind.RIGHT]
