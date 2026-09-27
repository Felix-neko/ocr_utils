"""Линейки у края кадра: морфология не укорачивает линейку, упирающуюся в край вырезки."""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.tables import ruling
from ocr_utils.page_layout.tables.grid import grid_from_lines

DPI = ruling.WORK_DPI


def _header_row(columns: int = 5, height: int = 64, column_width: int = 140) -> np.ndarray:
    """Шапка таблицы вплотную к кадру: две горизонтали по краям и вертикали во всю высоту.

    Args:
        columns: Число граф.
        height: Высота кадра, px (64 px при 150 dpi — шапка ~11 мм, как 1968/05 с. 86).
        column_width: Ширина графы, px.

    Returns:
        Серый кадр: бумага 255, линейки 0.
    """
    width = columns * column_width + 2
    page = np.full((height, width), 255, np.uint8)
    page[1:3, :] = 0
    page[height - 3 : height - 1, :] = 0
    for column in range(columns + 1):
        x = min(width - 2, column * column_width)
        page[:, x : x + 2] = 0
    return page


def test_vertical_touching_frame_edges_keeps_full_length() -> None:
    """Вертикаль от верхнего края до нижнего выходит из поиска во всю высоту кадра.

    Раньше закрытие с нулём за краем срезало с каждого конца половину ядра (4 мм), и от
    вертикали высотой 64 px оставалось 18 px.
    """
    page = _header_row()
    lines = ruling.find_lines(page, DPI)
    inner = [s for s in lines.vertical if 10 < s.box.x0 < page.shape[1] - 10]
    assert inner, "внутренние вертикали не найдены"
    assert min(s.length for s in inner) >= page.shape[0] - 4


def test_header_row_splits_into_columns() -> None:
    """Однострочная шапка из пяти граф даёт пять ячеек, а не одну склеенную."""
    page = _header_row(columns=5)
    lines = ruling.find_lines(page, DPI)
    grid = grid_from_lines(lines, page.shape[:2], DPI)
    assert len(grid.cells) == 5


def test_rule_does_not_spread_to_frame_edge() -> None:
    """Поле вокруг кадра не тянет линейку до края: начало в 30 px от края остаётся на месте.

    Это то, ради чего у морфологии стоит ``borderValue=0`` (см. ``ruling._axis_mask``).
    """
    page = np.full((200, 400), 255, np.uint8)
    page[100:102, 30:370] = 0
    lines = ruling.find_lines(page, DPI)
    assert len(lines.horizontal) == 1
    box = lines.horizontal[0].box
    assert abs(box.x0 - 30) <= 1 and abs(box.x1 - 370) <= 1
