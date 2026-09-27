"""Пометки на полях: обрывки карандашной черты не сцепляются со строкой и не становятся краем ряда."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ocr_utils.page_layout.text_blocks import blocks, zones
from ocr_utils.page_layout.text_blocks.segment import SCALES, _crosses
from tests.ocr_utils.page_layout.text_blocks.test_zones import _piece

INK = np.zeros((10, 10), dtype=bool)


def _speck(x0: float, y: float, size: float = 3.0):
    """Кусок из одной крошки ``size × size`` (обрывок карандаша)."""
    piece = _piece(x0, y, letters=1)
    return replace(piece, sizes=np.array([[size, size]]), x0=x0, x1=x0 + size)


def test_speck_is_not_linked_across_the_margin():
    """Крошка через пустоту в две строчные к строке не цепляется; точка у слова — цепляется."""
    word = _piece(100.0, 100.0, letters=6)
    far = _speck(word.x0 - 25.0, 100.0)
    near = _speck(word.x1 + 2.0, 104.0)
    assert zones._verdict_of(far, word, SCALES[0], [], None, INK, 2.0, _crosses) is zones.LinkVerdict.SPECK
    assert zones._verdict_of(word, near, SCALES[0], [], None, INK, 2.0, _crosses) is not zones.LinkVerdict.SPECK


def test_row_edge_stops_before_a_narrow_mark():
    """Край ряда не уходит на узкий клочок за пустотой, но берёт целое слово за пустотой."""
    ink = np.zeros((60, 300), dtype=np.uint8)
    ink[20:40, 160:260] = 1  # текст строки, ось начинается на 160
    ink[20:40, 120:122] = 1  # обрывок карандашной черты на поле
    assert blocks.connected_edge(ink, 120.0, 160.0, 15, 45, 12.0, "left") == 160.0
    ink[20:40, 100:140] = 1  # целое слово без своей оси за той же пустотой
    assert blocks.connected_edge(ink, 100.0, 160.0, 15, 45, 12.0, "left") == 100.0


def test_row_edge_inside_the_axis_is_kept():
    """Край по краске, лежащий внутри оси, не трогается."""
    ink = np.zeros((60, 300), dtype=np.uint8)
    ink[20:40, 160:260] = 1
    assert blocks.connected_edge(ink, 170.0, 150.0, 15, 45, 12.0, "left") == 170.0
