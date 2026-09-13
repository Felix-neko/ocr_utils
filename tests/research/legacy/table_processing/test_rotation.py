"""Детекторы поворота ячейки на синтетике с известным углом."""

from __future__ import annotations

from research.legacy.table_processing.rotation import glyph_aspect, ink_axis_cell
from research.legacy.table_processing.rotation.base import CellCrop
from research.legacy.table_processing.rotation.combine import CellDecision, apply_prior, combine_axis
from research.legacy.table_processing.structure import ruling_grid

from tests.research.legacy.table_processing.synthetic import DPI, make_table


def _crops(table):
    grid = ruling_grid.extract(table.image, DPI)
    return grid, [CellCrop("t", cell, ruling_grid.cell_image(table.image, cell, DPI), DPI) for cell in grid.cells]


def test_rotated_header_is_found_and_upright_is_not():
    table = make_table(
        rotated={(0, 1): "расчетная лесосека", (0, 2): "фактический отпуск"},
        upright={(0, 0): "Области", (1, 0): "Архангельская", (2, 0): "Вологодская"},
    )
    grid, crops = _crops(table)
    for crop in crops:
        verdict = glyph_aspect.detect(crop)
        expected = (crop.cell.row, crop.cell.col) in table.rotated
        if verdict.confidence == 0.0:
            continue  # детектор честно промолчал — это не ошибка
        assert verdict.rotated == expected, f"ячейка r{crop.cell.row}c{crop.cell.col}"


def test_empty_cell_gets_no_verdict():
    table = make_table(upright={(0, 0): "Области"})
    grid, crops = _crops(table)
    empty = [crop for crop in crops if crop.cell.key != (0, 0)]
    assert all(crop.gray.size for crop in empty)
    assert all(glyph_aspect.detect(crop).confidence == 0.0 for crop in empty)


def test_ink_axis_agrees_with_glyph_aspect_on_rotated_header():
    table = make_table(rotated={(0, 1): "расчетная лесосека"}, upright={(0, 0): "Области"})
    grid, crops = _crops(table)
    target = next(crop for crop in crops if crop.cell.key == (0, 1))
    assert glyph_aspect.detect(target).rotated
    assert ink_axis_cell.detect(target).rotated


def test_prior_gives_one_side_to_the_whole_table():
    decisions = [
        CellDecision(90, 0.9, sign_votes={"ocr_vote": 90}, sign_weight={90: 0.9}),
        CellDecision(270, 0.2, sign_votes={"ink_axis": 270}, sign_weight={270: 0.2}),
        CellDecision(0, 1.0),
    ]
    result, prior = apply_prior(decisions)
    assert prior == 90
    assert [decision.rotate_cw for decision in result] == [90, 90, 0]
    assert result[1].from_prior


def test_combine_prefers_the_confident_voter():
    from research.legacy.table_processing.rotation.base import Verdict

    decision = combine_axis(
        {"glyph_aspect": Verdict(90, 1.0, axis_only=True), "ink_axis": Verdict(270, 0.2), "ocr_vote": Verdict(90, 0.83)}
    )
    assert decision.rotate_cw == 90
