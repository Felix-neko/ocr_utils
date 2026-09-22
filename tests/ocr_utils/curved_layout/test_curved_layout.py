"""Разбор текста с кривыми строками: оси строк, колонки, огибающие блоков и выключка."""

from __future__ import annotations

import numpy as np

from ocr_utils.curved_layout import WORK_DPI
from ocr_utils.curved_layout.alignment import AlignKind
from ocr_utils.curved_layout.columns import gutters_of, zones_of
from ocr_utils.curved_layout.engines.ink import InkEngine
from ocr_utils.curved_layout.page import analyse_gray
from ocr_utils.page_layout import px_to_mm
from tests.ocr_utils.curved_layout.synthetic import bowed, column_page, inset_page

ENGINE = InkEngine()


def test_axes_follow_bow():
    """Ось строки повторяет заданную дугу: прогиб оси совпадает с заданным с точностью 15 %."""
    amplitude = 30.0  # пиксели рендера 300 dpi
    page = bowed(column_page(columns=1), amplitude)
    analysis = analyse_gray(page, ENGINE)
    long_axes = [axis for axis in analysis.axes if axis.length_mm > 80]
    assert len(long_axes) > 10

    def bow(x: float) -> float:
        """Заданное поле смещения в пикселях рабочей копии (деформация задана в 300 dpi)."""
        half = page.shape[1] / 2.0
        return amplitude / 2.0 * (1.0 - ((x * 2.0 - half) / half) ** 2)

    # Ожидаемая сагитта считается по КОНЦАМ ОСИ: строка не доходит до краёв страницы, где
    # деформация нулевая, поэтому её прогиб меньше амплитуды поля.
    errors = []
    for axis in long_axes:
        chord = (bow(axis.x0) + bow(axis.x1)) / 2.0
        expected = px_to_mm(bow((axis.x0 + axis.x1) / 2.0) - chord, axis.dpi)
        errors.append(abs(axis.sagitta_mm - expected) / max(expected, 1e-6))
    assert float(np.median(errors)) < 0.2


def test_two_columns_two_blocks():
    """Страница в две колонки: два блока, огибающие не пересекают межколонник."""
    analysis = analyse_gray(column_page(columns=2), ENGINE)
    assert len(analysis.blocks) == 2
    left, right = analysis.blocks
    assert left.envelope.right[:, 0].max() < right.envelope.left[:, 0].min()


def test_three_columns():
    """Страница в три колонки: три блока и два межколонника."""
    analysis = analyse_gray(column_page(columns=3), ENGINE)
    assert len(analysis.blocks) == 3
    assert len(analysis.gutters) == 2


def test_alignment_kinds():
    """Выключка: по формату — ``both``, рваный правый край — ``left``, рваный левый — ``right``."""
    for justify, expected in (("both", AlignKind.BOTH), ("left", AlignKind.LEFT), ("right", AlignKind.RIGHT)):
        analysis = analyse_gray(column_page(columns=1, justify=justify), ENGINE)
        assert analysis.alignments, justify
        assert analysis.alignments[0].kind is expected, justify


def test_alignment_survives_bow():
    """Выключка по формату не теряется на изогнутой странице: меры считаются от огибающей."""
    analysis = analyse_gray(bowed(column_page(columns=1), 30.0), ENGINE)
    assert analysis.alignments[0].kind is AlignKind.BOTH
    # Огибающая изогнутого блока не обязана быть прямой, но должна быть ГЛАДКОЙ: соседние узлы
    # сетки отстоят друг от друга меньше чем на десятую долю миллиметра.
    left = analysis.blocks[0].envelope.left
    assert px_to_mm(float(np.abs(np.diff(left[:, 0])).max()), analysis.dpi) < 0.1


def test_inset_gutter_lives_on_part_of_height():
    """Врезка сверху справа: межколонник живёт на части высоты, зон становится две."""
    page = inset_page()
    work = page[::2, ::2]
    gutters = gutters_of(work, WORK_DPI)
    assert gutters, "межколонник врезки не найден"
    height, width = work.shape
    assert max(g.y1 for g in gutters) < height * 0.9
    zones = zones_of(gutters, height, width, WORK_DPI)
    assert len(zones) >= 2
    assert max(len(zone.columns) for zone in zones) == 2


def test_envelope_ignores_paragraph_indent():
    """Абзацный отступ не тянет огибающую внутрь блока: он считается отдельно, как отступ."""
    analysis = analyse_gray(column_page(columns=1, indent=60), ENGINE)
    block = analysis.blocks[0]
    alignment = analysis.alignments[0]
    assert alignment.left.indent_rows >= 3
    left_edges = np.array([row.x0 for row in block.rows])
    envelope = np.interp([row.y for row in block.rows], block.envelope.left[:, 1], block.envelope.left[:, 0])
    # Огибающая идёт по телу блока, а не по отступам: медиана остатка близка к нулю.
    assert abs(float(np.median(left_edges - envelope))) < 3.0
