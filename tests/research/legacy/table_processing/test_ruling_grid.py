"""Сетка ячеек: разделители, объединения, внешняя граница."""

from __future__ import annotations

import numpy as np

from research.legacy.table_processing.structure import ruling_grid

from tests.research.legacy.table_processing.synthetic import DPI, make_table


def test_separators_match_construction():
    table = make_table(upright={(0, 0): "Наименование", (1, 1): "1790", (2, 2): "164"})
    grid = ruling_grid.extract(table.image, DPI)
    assert grid.n_cols == 3 and grid.n_rows == 3
    for expected, found in zip(table.xs, grid.xs):
        assert abs(expected - found) <= 3
    for expected, found in zip(table.ys, grid.ys):
        assert abs(expected - found) <= 3


def test_missing_vertical_rule_makes_spanning_cell():
    # В шапке (строка 0) нет разделителя между графами 1 и 2 — значит, заголовок общий.
    table = make_table(
        upright={(0, 0): "Области", (0, 1): "Хвойные породы", (1, 1): "22030", (1, 2): "24853"},
        missing_vertical={(0, 2)},
    )
    grid = ruling_grid.extract(table.image, DPI)
    spanning = [cell for cell in grid.cells if cell.col_span > 1]
    assert len(spanning) == 1
    assert spanning[0].row == 0 and spanning[0].col == 1 and spanning[0].col_span == 2


def test_cells_cover_every_slot():
    table = make_table(upright={(1, 1): "1790"})
    grid = ruling_grid.extract(table.image, DPI)
    covered = {
        (row, column)
        for cell in grid.cells
        for row in range(cell.row, cell.row + cell.row_span)
        for column in range(cell.col, cell.col + cell.col_span)
    }
    assert covered == {(row, column) for row in range(grid.n_rows) for column in range(grid.n_cols)}


def test_interior_survives_a_narrow_cell():
    from research.legacy.table_processing.geometry import Box, Cell

    narrow = Cell(row=0, col=0, box=Box(10, 10, 13, 200))
    inner = ruling_grid.interior(narrow, DPI)
    assert inner.width >= 1 and inner.height >= 1


def test_strip_rules_removes_the_rule_and_keeps_text():
    table = make_table(upright={(0, 0): "Наименование"})
    grid = ruling_grid.extract(table.image, DPI)
    cell = grid.cell_at(0, 0)
    raw = table.image[ruling_grid.interior(cell, DPI).slice]
    cleaned = ruling_grid.strip_rules(raw, DPI)
    ink_before = int((raw < 128).sum())
    ink_after = int((cleaned < 128).sum())
    assert ink_after <= ink_before
    assert ink_after > ink_before * 0.5, "текст ячейки не должен исчезать вместе с линейками"


def test_cell_edges_follow_the_rule_on_a_skewed_table():
    """На перекошенной таблице ребро ячейки идёт за линейкой своей полосы.

    Один разделитель на всю таблицу этого не умеет по построению: при наклоне 1.5° и высоте
    600 px линейка уходит на 16 px, и полосы получают рамку то слева, то справа от неё.
    """
    table = make_table(
        upright={(0, 0): "Области", (1, 0): "Архангельская", (2, 0): "Вологодская"},
        row_heights=[200, 200, 200],
        skew_deg=1.5,
    )
    grid = ruling_grid.extract(table.image, DPI)
    lines = ruling_grid.find_lines(table.image, DPI)
    for cell in grid.cells:
        if cell.col == 0:
            continue
        middle = (cell.box.y0 + cell.box.y1) // 2
        row = lines.vertical_mask[middle]
        columns = np.nonzero(row[max(0, cell.box.x0 - 30) : cell.box.x0 + 30])[0]
        if columns.size == 0:
            continue
        true_x = max(0, cell.box.x0 - 30) + int(columns.mean())
        assert abs(cell.box.x0 - true_x) <= 3, f"ячейка r{cell.row}c{cell.col}: рамка {cell.box.x0}, линейка {true_x}"


def test_inner_box_has_no_rule_ink_on_a_thick_rule():
    table = make_table(upright={(0, 0): "Наименование", (1, 1): "22030"}, rule_px=9)
    grid = ruling_grid.extract(table.image, DPI)
    lines = ruling_grid.find_lines(table.image, DPI)
    for cell in grid.cells:
        inner = ruling_grid.interior(cell, DPI)
        assert lines.mask[inner.slice].sum() == 0, f"в ячейке r{cell.row}c{cell.col} остался кусок линейки"


def test_stray_rule_above_the_table_makes_no_extra_row():
    plain = ruling_grid.extract(make_table(upright={(0, 0): "Области"}).image, DPI)
    stray = ruling_grid.extract(make_table(upright={(0, 0): "Области"}, stray_rule_above=True).image, DPI)
    assert stray.n_rows == plain.n_rows
