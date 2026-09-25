"""Сетка по кривым: ячейки изогнутой таблицы, объединения, двойная линейка, внешние графы, внутренность."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.tables.curved_grid import curved_table
from tests.ocr_utils.page_layout.tables import synthetic

DPI = synthetic.DPI

CURVED_SAMPLE = Path(__file__).parents[4] / "ocr_utils/page_layout/tables/example_table_curved.jpg"


def _layout(grid) -> list[tuple[int, int, int, int]]:
    return [(cell.row, cell.col, cell.row_span, cell.col_span) for cell in grid.cells]


def test_curved_table_cells_and_merged_header() -> None:
    """Изогнутая таблица 4 × 3 с объединённой шапкой над двумя правыми графами: 11 ячеек, шапка 1 × 2."""
    table = synthetic.make_curved_table(missing_vertical={(0, 2)})
    _traces, grid = curved_table(table.image, DPI)
    assert (grid.n_rows, grid.n_cols) == (4, 3)
    layout = _layout(grid)
    assert len(layout) == 11
    assert (0, 1, 1, 2) in layout
    assert all(not cell.irregular for cell in grid.cells)


def test_double_rule_marks_the_header() -> None:
    """Двойная линейка — одна граница строк, и она отбивает шапку."""
    table = synthetic.make_curved_table(double_under_row=0)
    _traces, grid = curved_table(table.image, DPI)
    assert grid.n_rows == 4
    assert [boundary.is_double for boundary in grid.rows] == [False, True, False, False, False]
    assert grid.header_rows == 1


def test_wavy_rule_is_not_a_double_rule() -> None:
    """Сильно изогнутая (волнистая) таблица без двойных линеек двойной не получает."""
    table = synthetic.make_curved_table(amplitude_px=9.0, period_px=450.0, skew_deg=1.0)
    _traces, grid = curved_table(table.image, DPI)
    assert not any(boundary.is_double for boundary in grid.rows)
    assert (grid.n_rows, grid.n_cols) == (4, 3)


def test_outer_columns_without_rules_are_kept() -> None:
    """Без внешних вертикалей крайние графы с текстом остаются графами (граница — виртуальная кривая)."""
    table = synthetic.make_curved_table(outer_verticals=False)
    _traces, grid = curved_table(table.image, DPI)
    assert grid.n_cols == 3
    assert grid.cols[0].is_virtual and grid.cols[-1].is_virtual
    assert len(grid.cells) == 12


def test_inner_polygons_hold_no_rule_ink_and_cells_tile_the_table() -> None:
    """Внутренность ячейки не задевает линеек; соседние ячейки сходятся по общей кривой."""
    table = synthetic.make_curved_table(missing_vertical={(0, 2)})
    traces, grid = curved_table(table.image, DPI)
    rules = traces.mask > 0
    shape = table.image.shape[:2]
    for cell in grid.cells:
        assert not (cell.mask(shape, inner=True).astype(bool) & rules).any()
    # Площадь ячеек складывается в площадь таблицы с точностью до штрихов линеек на границах.
    union = np.zeros(shape, np.uint8)
    total = 0
    for cell in grid.cells:
        mask = cell.mask(shape)
        total += int((mask > 0).sum())
        union |= mask
    assert total - int((union > 0).sum()) < 0.03 * total


def test_real_curved_table_is_ten_cells() -> None:
    """1968/03 с.4: верх наклонён на 2,5–5°, низ почти ровный — сетка 4 × 3, 10 ячеек, без лишних строк."""
    gray = cv2.imread(str(CURVED_SAMPLE), cv2.IMREAD_GRAYSCALE)
    crop = gray[220:1610, 205:1570]
    _traces, grid = curved_table(crop, DPI)
    assert (grid.n_rows, grid.n_cols) == (4, 3)
    assert sorted(_layout(grid)) == [
        (0, 0, 2, 1),
        (0, 1, 1, 2),
        (1, 1, 1, 1),
        (1, 2, 1, 1),
        (2, 0, 1, 1),
        (2, 1, 1, 1),
        (2, 2, 1, 1),
        (3, 0, 1, 1),
        (3, 1, 1, 1),
        (3, 2, 1, 1),
    ]


def _with_marks(image: np.ndarray, draw_marks) -> np.ndarray:
    """Копия картинки с помехами, нарисованными ``draw_marks(canvas)`` (OpenCV, серый)."""
    canvas = image.copy()
    draw_marks(canvas)
    return canvas


def _stamp_and_bold_glyphs(canvas: np.ndarray) -> None:
    """Рамка штампа толщиной 6 px в ячейке, касающаяся линейки, и жирные «буквы» — штрихи по 12 px."""
    cv2.rectangle(canvas, (380, 230), (560, 300), synthetic.INK, 6)
    # «Буквы» — внутри ячейки строки 2 (линейки строк на y ≈ 260 и 350), не на линейке.
    for x in (120, 160, 200):
        cv2.rectangle(canvas, (x, 278), (x + 11, 332), synthetic.INK, -1)
    cv2.rectangle(canvas, (110, 298), (230, 309), synthetic.INK, -1)


def test_stamp_and_bold_glyphs_are_not_cell_borders() -> None:
    """Штамп и жирные буквы в ячейках не дают границ: раскладка та же, что без них."""
    table = synthetic.make_curved_table(missing_vertical={(0, 2)})
    _clean, reference = curved_table(table.image, DPI)
    traces, grid = curved_table(_with_marks(table.image, _stamp_and_bold_glyphs), DPI)
    assert _layout(grid) == _layout(reference)
    assert len(traces.all) == len(_clean.all)


def _pen_mark(canvas: np.ndarray) -> None:
    """Карандашная черта 8 мм у левой рамки, слегка наклонная, касается верхней линейки."""
    cv2.line(canvas, (70, 55), (80, 150), synthetic.INK, 2)


def test_pen_mark_by_the_frame_is_not_a_sliver_column() -> None:
    """Черта в полутора миллиметрах от рамки не делает графу шириной в штрих."""
    table = synthetic.make_curved_table()
    _traces, grid = curved_table(_with_marks(table.image, _pen_mark), DPI)
    assert (grid.n_rows, grid.n_cols) == (4, 3)


def test_cell_region_is_the_curved_polygon() -> None:
    """Область ячейки — ровно многоугольник по кривым: маска совпадает с ним, вне — бумага."""
    table = synthetic.make_curved_table()
    _traces, grid = curved_table(table.image, DPI)
    cell = grid.cell_at(1, 1)
    region = cell.extract(table.image)
    full = cell.mask(table.image.shape[:2], inner=True) > 0
    assert region.mask.sum() == full[region.box.slice].sum()
    assert (region.image[~region.mask] >= synthetic.PAPER - 2).all()
    assert region.box.width < cell.box.width  # внутренность уже ячейки на линейки
