"""Перекройка таблицы, когда увеличения страницы не хватает.

ПОВОРОТ ВСЕЙ ТАБЛИЦЫ. Если боковых ячеек много (таблица напечатана боком целиком), проще
повернуть её всю: бывшие боковые ячейки становятся прямыми пикселями и не требуют ни
чтения, ни набора, а бывшие прямые — читаются и набираются заново в транспонированной
геометрии. Сетка переводится тем же ``rotation.rotate_box``, что и разметка полос:
второе соглашение о стороне поворота по соседству повернуло бы не туда.

РАСШИРЕНИЕ КОЛОНОК. Вставка полосы перед правой линейкой колонки, по краю штриха, а не
по центру линейки — иначе половина штриха остаётся слева от разреза, половина справа, и
линейка двоится. Вставка заливается построчной медианой колонки: на строке горизонтальной
линейки медиана — краска, и линейка продолжается сама; на строке текста тёмных пикселей
меньшинство, и вставка выходит бумагой. Ячейки над и под расширяемой растут вместе с
колонкой по построению, объединённые — тоже.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ocr_utils.scan_markup.rotation import rotate_box
from ocr_utils.scan_markup.table_detection.geometry import Box, Cell, Grid

# Что считать краской при разборе вставки и какая доля строки должна быть краской, чтобы
# строка считалась линейкой. Линейка идёт через всю колонку — доля близка к единице;
# строка по перекладине цифр набирает до половины, отсюда 0.8, а не 0.5.
GAP_INK_LEVEL = 0.6
GAP_RULE_SHARE = 0.8


def rotate_grid(grid: Grid, width: int, height: int, degrees: int) -> Grid:
    """Сетка кадра ``width × height`` после поворота кадра на ``degrees`` по часовой."""
    turns = (degrees // 90) % 4
    if turns == 0:
        return grid
    rows, cols = grid.n_rows, grid.n_cols

    def moved(box: "Box | None") -> "Box | None":
        return Box(*rotate_box(box.as_tuple(), width, height, degrees)) if box is not None else None

    cells: list[Cell] = []
    for cell in grid.cells:
        if turns == 1:  # по часовой: верхняя строка становится правой колонкой
            row, col = cell.col, rows - cell.row - cell.row_span
            row_span, col_span = cell.col_span, cell.row_span
        elif turns == 2:
            row, col = rows - cell.row - cell.row_span, cols - cell.col - cell.col_span
            row_span, col_span = cell.row_span, cell.col_span
        else:  # против часовой: левая колонка становится верхней строкой
            row, col = cols - cell.col - cell.col_span, cell.row
            row_span, col_span = cell.col_span, cell.row_span
        cells.append(
            replace(
                cell,
                row=row,
                col=col,
                row_span=row_span,
                col_span=col_span,
                box=moved(cell.box),
                inner=moved(cell.inner),
            )
        )
    # Порядок ячеек сохраняется: по нему вызывающий код сопоставляет ячейки до и после поворота.
    if turns == 1:
        xs, ys = [height - y for y in reversed(grid.ys)], list(grid.xs)
    elif turns == 2:
        xs, ys = [width - x for x in reversed(grid.xs)], [height - y for y in reversed(grid.ys)]
    else:
        xs, ys = list(grid.ys), [width - x for x in reversed(grid.xs)]
    # Шапка, двойные линейки и края штрихов вертикалей после поворота теряют смысл.
    return Grid(xs=xs, ys=ys, cells=cells, header_rows=0, double_rule_ys=[], source=grid.source, column_cuts=[])


def widen_columns(image: np.ndarray, grid: Grid, extra: dict[int, int]) -> tuple[np.ndarray, Grid]:
    """Раздвинуть колонки на ``extra[колонка]`` пикселей; линейки продолжаются сами."""
    extra = {column: gap for column, gap in extra.items() if gap > 0 and 0 <= column < grid.n_cols}
    if not extra:
        return image, grid
    cuts = grid.cuts
    pieces: list[np.ndarray] = []
    shift, previous = 0, 0
    new_xs: list[int] = []
    new_cuts: list[int] = []
    for column in range(grid.n_cols):
        left, right = cuts[column], cuts[column + 1]
        new_xs.append(grid.xs[column] + shift)
        new_cuts.append(left + shift)
        pieces.append(image[:, previous:right])
        gap = extra.get(column, 0)
        if gap:
            pieces.append(np.repeat(_gap_column(image[:, left:right])[:, None], gap, axis=1))
            shift += gap
        previous = right
    pieces.append(image[:, previous:])
    new_xs.append(grid.xs[-1] + shift)
    new_cuts.append(cuts[-1] + shift)
    widened = np.concatenate(pieces, axis=1)

    def shift_x(value: int) -> int:
        # По положению, а не по словарю: рёбра ячеек меряются по своей полосе и с общими
        # разделителями не совпадают (на паке до 20 px).
        return value + sum(gap for column, gap in extra.items() if cuts[column + 1] <= value)

    def moved(cell: Cell) -> Cell:
        box = Box(shift_x(cell.box.x0), cell.box.y0, shift_x(cell.box.x1), cell.box.y1)
        inner = cell.inner
        if inner is not None:
            # Внутренность едет ЗА СВОИМИ краями рамки: вставка ложится между текстом и
            # линейкой, и прямой сдвиг внутренности не расширил бы место под текст.
            inner = Box(inner.x0 + (box.x0 - cell.box.x0), inner.y0, inner.x1 + (box.x1 - cell.box.x1), inner.y1)
        return replace(cell, box=box, inner=inner)

    moved_cells = [moved(cell) for cell in grid.cells]
    for cell, new_cell in zip(grid.cells, moved_cells):
        # Вставка попала ВНУТРЬ объединённой ячейки — её текст разрезан пополам («По і причинам»).
        # Содержимое переносится целиком из исходника и ставится по центру новой внутренности.
        inside = any(
            cuts[column + 1] < cell.box.x1 for column in extra if cell.col <= column < cell.col + cell.col_span - 1
        )
        if cell.col_span > 1 and inside and cell.inner is not None and new_cell.inner is not None:
            _transplant(image, cell.inner, widened, new_cell.inner)
    moved_grid = Grid(
        xs=new_xs,
        ys=list(grid.ys),
        cells=moved_cells,
        header_rows=grid.header_rows,
        double_rule_ys=list(grid.double_rule_ys),
        source=grid.source,
        column_cuts=new_cuts,
    )
    return widened, moved_grid


def _gap_column(strip: np.ndarray) -> np.ndarray:
    """Столбец-заполнитель вставки: линейки сохранены, остальное — бумага."""
    if strip.size == 0:
        return np.full((strip.shape[0],), 255, np.uint8)
    paper = float(np.percentile(strip, 90))
    dark_share = (strip < paper * GAP_INK_LEVEL).mean(axis=1)
    median = np.median(strip, axis=1)
    return np.where(dark_share >= GAP_RULE_SHARE, median, paper).astype(strip.dtype)


def _transplant(source: np.ndarray, old_inner: Box, target: np.ndarray, new_inner: Box) -> None:
    """Перенести внутренность ячейки из исходника в расширенную по центру."""
    patch = source[old_inner.slice].copy()
    if patch.size == 0:
        return
    paper = int(np.percentile(patch, 90))
    target[new_inner.slice] = paper
    left = new_inner.x0 + max(0, (new_inner.width - patch.shape[1]) // 2)
    width = min(patch.shape[1], new_inner.x1 - left)
    target[new_inner.y0 : new_inner.y0 + patch.shape[0], left : left + width] = patch[:, :width]
