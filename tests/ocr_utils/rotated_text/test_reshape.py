"""Поворот сетки и расширение колонок."""

import numpy as np

from ocr_utils.scan_markup.rotation import rotate_cw
from ocr_utils.page_layout.geometry import Box, Cell, Grid

from ocr_utils.rotated_text.tables.reshape import rotate_grid, widen_columns
from ocr_utils.rotated_text.tables.structure import scale_grid


def _grid() -> Grid:
    xs, ys = [10, 110, 210, 310], [20, 70, 120]
    cells = [
        Cell(0, 0, Box(10, 20, 110, 70), inner=Box(14, 24, 106, 66)),
        Cell(0, 1, Box(110, 20, 310, 70), col_span=2, inner=Box(114, 24, 306, 66)),
        Cell(1, 0, Box(10, 70, 110, 120), inner=Box(14, 74, 106, 116)),
        Cell(1, 1, Box(110, 70, 210, 120), inner=Box(114, 74, 206, 116)),
        Cell(1, 2, Box(210, 70, 310, 120), inner=Box(214, 74, 306, 116)),
    ]
    return Grid(xs=xs, ys=ys, cells=cells, column_cuts=[9, 109, 209, 309])


def _image() -> np.ndarray:
    image = np.full((140, 320), 240, np.uint8)
    grid = _grid()
    for x in grid.xs:
        # Разделитель 210 внутри объединённой шапки (0,1) не рисуется — иначе это не объединение.
        image[(70 if x == 210 else 20) : 121, x - 1 : x + 2] = 20
    for y in grid.ys:
        image[y - 1 : y + 2, 10:311] = 20
    return image


def test_rotate_grid_matches_rotate_box_and_keeps_order():
    grid = _grid()
    width, height = 320, 140
    for degrees in (90, 180, 270):
        rotated = rotate_grid(grid, width, height, degrees)
        assert len(rotated.cells) == len(grid.cells)
        assert rotated.n_rows * rotated.n_cols == grid.n_rows * grid.n_cols
        # Ячейка не выходит за пределы повёрнутого кадра и покрывает те же индексы решётки.
        new_w, new_h = (height, width) if degrees in (90, 270) else (width, height)
        covered = set()
        for cell in rotated.cells:
            assert 0 <= cell.box.x0 < cell.box.x1 <= new_w and 0 <= cell.box.y0 < cell.box.y1 <= new_h
            assert cell.inner is not None and cell.box.x0 <= cell.inner.x0 < cell.inner.x1 <= cell.box.x1
            for r in range(cell.row, cell.row + cell.row_span):
                for c in range(cell.col, cell.col + cell.col_span):
                    covered.add((r, c))
        assert covered == {(r, c) for r in range(rotated.n_rows) for c in range(rotated.n_cols)}
        # Ячейки идут в прежнем порядке — по нему сопоставляются записи до и после.
        assert [c.col_span for c in rotated.cells] == [c.row_span if degrees != 180 else c.col_span for c in grid.cells]


def test_rotate_grid_cw_top_row_becomes_right_column():
    grid = _grid()
    rotated = rotate_grid(grid, 320, 140, 90)
    top_left = rotated.cells[0]  # была (0,0), стала правой колонкой, первой строкой
    assert (top_left.row, top_left.col) == (0, grid.n_rows - 1)
    image = _image()
    turned = rotate_cw(image, 90)
    # Внутренность ячейки после поворота — та же бумага, без линеек.
    assert (turned[top_left.inner.y0 : top_left.inner.y1, top_left.inner.x0 : top_left.inner.x1] > 200).all()


def test_widen_columns_extends_rules_and_moves_cells():
    image, grid = _image(), _grid()
    widened, moved = widen_columns(image, grid, {1: 50})
    assert widened.shape == (140, 370)
    assert moved.xs == [10, 110, 260, 360]
    # Горизонтальные линейки продолжились через вставку, а бумага осталась бумагой.
    assert (widened[20, 209:259] < 60).all() and (widened[45, 209:259] > 200).all()
    assert (widened[95, 259:262] < 60).all()  # линейка колонки уехала за вставку
    by_key = {cell.key: cell for cell in moved.cells}
    assert by_key[(1, 1)].box == Box(110, 70, 260, 120)
    assert by_key[(1, 1)].inner == Box(114, 74, 256, 116)  # внутренность выросла вместе с рамкой
    assert by_key[(1, 2)].box == Box(260, 70, 360, 120)  # соседняя колонка просто сдвинулась
    assert by_key[(0, 1)].box == Box(110, 20, 360, 70)  # объединённая накрывает обе
    assert by_key[(1, 0)].box == Box(10, 70, 110, 120)  # левее вставки ничего не изменилось


def test_scale_grid_scales_everything():
    grid = _grid()
    scaled = scale_grid(grid, 2.0)
    assert scaled.xs == [20, 220, 420, 620] and scaled.column_cuts == [18, 218, 418, 618]
    assert scaled.cells[1].box == Box(220, 40, 620, 140) and scaled.cells[1].inner == Box(228, 48, 612, 132)


def test_widen_transplants_spanning_cell_content():
    image, grid = _image(), _grid()
    # Текст объединённой шапки — тёмный квадрат посередине, ровно там, куда ляжет вставка.
    image[40:50, 205:215] = 30
    widened, moved = widen_columns(image, grid, {1: 50})
    cell = {c.key: c for c in moved.cells}[(0, 1)]
    inner = widened[cell.inner.y0 : cell.inner.y1, cell.inner.x0 : cell.inner.x1]
    dark_columns = np.where((inner < 100).any(axis=0))[0]
    assert dark_columns.size == 10  # квадрат не разрезан
    assert abs((dark_columns[0] + dark_columns[-1]) / 2 - inner.shape[1] / 2) < 30  # и стоит по центру


def test_split_overlapping_breaks_l_shaped_cell_into_units():
    from ocr_utils.rotated_text.tables.structure import split_overlapping

    xs, ys = [0, 100, 200, 300], [0, 50, 100]
    l_shaped = Cell(0, 0, Box(0, 0, 300, 100), row_span=2, col_span=3)  # накрывает (1,0) и (1,1)
    grid = Grid(xs=xs, ys=ys, cells=[l_shaped, Cell(1, 0, Box(0, 50, 100, 100)), Cell(1, 1, Box(100, 50, 200, 100))])
    fixed = split_overlapping(grid)
    keys = sorted(c.key for c in fixed.cells)
    assert keys == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert all(c.row_span == 1 and c.col_span == 1 for c in fixed.cells)
    assert {c.key: c for c in fixed.cells}[(0, 2)].box == Box(200, 0, 300, 50)
