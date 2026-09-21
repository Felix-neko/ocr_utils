"""Паритет четвёртой версии с третьей на стенде: живой детектор конвейера не теряет найденного v3.

Остальные проверки четвёртой версии переехали вместе с детектором в
``tests/ocr_utils/page_layout/tables``.
"""

from __future__ import annotations

import numpy as np

from research.legacy.table_processing.detection import ruling_v3, ruling_v4
from research.legacy.table_processing.geometry import KIND_TABLE, intersection
from tests.research.legacy.table_processing import synthetic

DPI = synthetic.DPI


def _filled_table(**kwargs) -> synthetic.SyntheticTable:
    upright = {(row, column): f"я{row}{column}" for row in range(3) for column in range(3)}
    return synthetic.make_table(upright=upright, **kwargs)


def _tables(page: np.ndarray):
    return [table for table in ruling_v4.detect(page, DPI) if table.kind == KIND_TABLE]


def test_v4_finds_what_v3_finds_on_a_table_page() -> None:
    """Четвёртая версия не теряет того, что находит третья, — с перекрытием не меньше 0.7."""
    table = _filled_table()
    page, _ = synthetic.make_page(table, lines_above=4, lines_below=4)
    old = ruling_v3.detect(page, DPI)
    new = _tables(page)
    assert old and new
    for table_box in old:
        best = max(
            (intersection(table_box.box, other.box).area for other in new if intersection(table_box.box, other.box)),
            default=0,
        )
        assert best >= 0.7 * table_box.box.area
