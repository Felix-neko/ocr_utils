"""Проверка находки считает только ячейки, куда помещается буква: параллельные штрихи ближе 2 мм — одна связка."""

from __future__ import annotations

from ocr_utils.page_layout.geometry import Box
from ocr_utils.page_layout.tables import verify
from ocr_utils.page_layout.tables.ruling import Lines, Segment, WORK_DPI

import numpy as np


def _lines(xs: list[int]) -> Lines:
    """Длинная горизонталь и вертикали в ``xs``, пересекающие её (кадр 150 dpi)."""
    empty = np.zeros((120, 500), np.uint8)
    horizontal = [Segment(Box(20, 59, 480, 62), True, 0.0)]
    vertical = [Segment(Box(x, 20, x + 3, 110), False, 0.0) for x in xs]
    return Lines(horizontal, vertical, empty, empty)


def test_strokes_closer_than_a_cell_become_one_bundle() -> None:
    """Три штриха монограммы через 1,5 мм (9 px при 150 dpi) — одна вертикаль, а не решётка узких ячеек."""
    bundled = verify.bundle_lines(_lines([40, 49, 58]), WORK_DPI)
    assert len(bundled.vertical) == 1
    assert bundled.vertical[0].box.x0 == 40 and bundled.vertical[0].box.x1 == 61
    assert len(bundled.horizontal) == 1


def test_columns_wide_enough_for_a_letter_stay_apart() -> None:
    """Графы шириной 3 мм (18 px) — настоящие ячейки: вертикали не склеиваются."""
    assert len(verify.bundle_lines(_lines([40, 58, 76]), WORK_DPI).vertical) == 3


def test_pieces_of_one_rule_one_after_another_are_not_bundled() -> None:
    """Куски одной линейки друг под другом (без нахлёста вдоль) остаются отдельными: счёт линеек не меняется."""
    empty = np.zeros((200, 200), np.uint8)
    pieces = [Segment(Box(50, 10, 53, 80), False, 0.0), Segment(Box(50, 100, 53, 180), False, 0.0)]
    assert len(verify.bundle_lines(Lines([], pieces, empty, empty), WORK_DPI).vertical) == 2
