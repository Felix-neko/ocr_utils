"""Укладка текста, расширение колонок и стирание ячейки."""

from __future__ import annotations

import numpy as np

from research.legacy.table_processing.geometry import Box, Cell, Grid
from research.legacy.table_processing.ocr.evaluate import cer
from research.legacy.table_processing.render.compose import Replacement, compose, erase, widen
from research.legacy.table_processing.render.fonts import fit_text, required_width

from tests.research.legacy.table_processing.synthetic import DPI, make_table


def test_fit_shrinks_until_it_fits():
    big = fit_text("количество в сутко-комплекте", 400, 200, 40, 14)
    small = fit_text("количество в сутко-комплекте", 160, 200, 40, 14)
    assert big and small
    assert big.font_px > small.font_px


def test_fit_gives_up_below_the_minimum():
    assert fit_text("очень длинный заголовок графы", 20, 20, 40, 14) is None


def test_required_width_covers_the_longest_word():
    text = "Специфицированная норма"
    width = required_width(text, 24, 300)
    fit = fit_text(text, width, 300, 24, 24)
    assert fit is not None
    assert all("-" not in line[:-1] for line in fit.lines), "разрывов посреди слова быть не должно"


def test_widen_grows_the_canvas_and_keeps_the_rest():
    table = make_table(upright={(0, 0): "Области", (1, 1): "22030"})
    from research.legacy.table_processing.structure.ruling_grid import extract

    grid = extract(table.image, DPI)
    widened, moved = widen(table.image.copy(), grid, {1: 40})
    assert widened.shape[1] == table.image.shape[1] + 40
    assert moved.xs[-1] == grid.xs[-1] + 40
    # Первая колонка не сдвинулась и не изменилась ни на пиксель.
    assert np.array_equal(widened[:, : grid.xs[1]], table.image[:, : grid.xs[1]])


def test_widen_keeps_horizontal_rules_continuous():
    table = make_table(upright={(0, 0): "Области"})
    from research.legacy.table_processing.structure.ruling_grid import extract

    grid = extract(table.image, DPI)
    widened, moved = widen(table.image.copy(), grid, {1: 60})
    rule_row = grid.ys[1] + 1
    strip = widened[rule_row, moved.xs[1] : moved.xs[2]]
    assert (strip < 128).mean() > 0.9, "линейка обязана продолжаться через вставку"


def test_erase_clears_ink_inside_the_box():
    image = np.full((100, 100), 245, np.uint8)
    image[40:60, 40:60] = 20
    erase(image, Box(30, 30, 70, 70), 245)
    assert (image < 128).sum() == 0


def test_compose_replaces_rotated_text():
    table = make_table(rotated={(0, 1): "расчетная лесосека"}, upright={(0, 0): "Области", (1, 0): "Архангельская"})
    from research.legacy.table_processing.structure.ruling_grid import extract

    grid = extract(table.image, DPI)
    cell = grid.cell_at(0, 1)
    result = compose(table.image, grid, [Replacement(cell, "расчетная лесосека", 90)], DPI)
    assert result.image.shape[1] >= table.image.shape[1]
    assert not result.failed


def test_cer_counts_characters():
    assert cer("норма на изделие", "норма на изделие") == 0.0
    assert 0.0 < cer("норма на изделие", "норма на излелие") < 0.1
    assert cer("", "") == 0.0
    assert cer("", "мусор") == 1.0
