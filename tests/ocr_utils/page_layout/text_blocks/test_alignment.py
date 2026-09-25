"""Вердикт выключки (``text_blocks.alignment``) и деление блока по жирности набора (``blocks._heavy_break``)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from ocr_utils.page_layout.text_blocks import blocks, zones
from ocr_utils.page_layout.text_blocks.alignment import AlignKind, is_centered, verdict
from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
from ocr_utils.page_layout.text_blocks.page import analyse_gray
from tests.ocr_utils.page_layout.text_blocks.synthetic import column_page, single_line_page
from tests.ocr_utils.page_layout.text_blocks.test_curved_layout import _row

DPI = 150.0


def _rows(spans: list[tuple[float, float]], weights: list[float] | None = None) -> list:
    """Ряды с заданными краями по x, шагом 22 px и (необязательно) жирностью."""
    out = []
    for index, (x0, x1) in enumerate(spans):
        row = _row(x0, x1, [100.0 + 22.0 * index] * 8, height=12.0)
        if weights is not None:
            row = replace(row, weight=weights[index], stroke=2.5, glyph_w=8.0, glyph_h=9.5)
        out.append(row)
    return out


def test_centered_rows_are_center_and_equal_rows_are_not():
    """Строки разной длины с общей серединой — центр; строки одной длины — не центр (это формат)."""
    centered = _rows([(300, 500), (250, 550), (320, 480), (280, 520)])
    assert is_centered(centered, DPI)
    assert verdict(False, False, True) is AlignKind.CENTER
    justified = _rows([(100, 700)] * 5)
    assert not is_centered(justified, DPI)
    shifted = _rows([(300, 500), (100, 330), (320, 480), (500, 700)])
    assert not is_centered(shifted, DPI)


def test_verdict_order_prefers_sides_over_center():
    """Формат и выключка по краю важнее центра."""
    assert verdict(True, True, True) is AlignKind.BOTH
    assert verdict(True, False, True) is AlignKind.LEFT
    assert verdict(False, True, True) is AlignKind.RIGHT
    assert verdict(False, False, False) is AlignKind.RAGGED


def test_single_line_block_has_no_alignment():
    """Однострочный блок — ``ragged`` (на оверлее ``none``), а не ``both``."""
    analysis = analyse_gray(single_line_page(), InkEngine())
    assert analysis.alignments and analysis.alignments[0].kind is AlignKind.RAGGED


def test_justified_column_is_still_both():
    """Колонка по формату по-прежнему ``both``: центр её не перебивает."""
    analysis = analyse_gray(column_page(columns=1, justify="both"), InkEngine())
    assert analysis.alignments[0].kind is AlignKind.BOTH


def test_bold_signature_under_a_column_is_split_off():
    """Полужирная подпись в три строки под колонкой уходит в свой блок — ровно по своей границе."""
    weights = [2.0, 2.1, 1.95, 2.05, 2.0, 2.1, 2.0, 1.95, 2.05, 2.0] + [2.65, 2.6, 2.7]
    rows = _rows([(100, 700)] * 10 + [(300, 500)] * 3, weights)
    groups = blocks._split_by_style(rows)
    assert [len(group) for group in groups] == [10, 3]


def test_single_bold_last_line_of_a_big_block_is_split_off():
    """Подпись одной строкой («Э. САВИНА»): крайний ряд большого блока жирнее соседей в 1.6 раза."""
    weights = [2.0, 2.1, 1.95, 2.05, 2.0, 2.1, 2.0, 1.95, 2.05, 2.0, 3.3]
    rows = _rows([(100, 700)] * 10 + [(400, 600)], weights)
    assert [len(group) for group in blocks._split_by_style(rows)] == [10, 1]


def test_weight_noise_does_not_split():
    """Разброс жирности внутри колонки (до 10 %) блок не делит."""
    weights = [2.0, 2.2, 1.95, 2.1, 2.0, 2.15, 1.9, 2.05, 2.2, 2.0, 1.95, 2.1]
    rows = _rows([(100, 700)] * 12, weights)
    assert len(blocks._split_by_style(rows)) == 1


def test_overlapping_pieces_are_not_linked():
    """Слово и черта под ним (перекрытие по x 100 %) не сцепляются; соседние слова — сцепляются."""
    scale = SimpleNamespace(link_height_ratio=1.8)

    def piece(x0, x1, cy):
        return SimpleNamespace(x0=x0, x1=x1, cy=cy, x_h=10.0, height=18.0, leader_dots=0)

    word, next_word, rule = piece(500, 600, 143.5), piece(606, 663, 143.7), piece(109, 909, 145.0)
    assert zones._allowed(word, next_word, scale, [], [], None, 2.0, lambda *args: False)
    assert not zones._allowed(rule, next_word, scale, [], [], None, 2.0, lambda *args: False)
