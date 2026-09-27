"""Мост: две полные таблицы одна под другой, связанные одной длинной линейкой, — два ядра."""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.tables import refine, rules
from ocr_utils.page_layout.tables.ruling import WORK_DPI, binarize, mm_to_px

DPI = WORK_DPI


def _grid(page: np.ndarray, top: int, rows: int, columns: list[int]) -> int:
    """Нарисовать полную таблицу: горизонтали на всю ширину через 60 px и вертикали на ``columns``.

    Args:
        page: Кадр, рисуется на месте.
        top: y верхней линейки.
        rows: Число строк.
        columns: x вертикалей (включая боковые).

    Returns:
        y нижней линейки.
    """
    bottom = top + rows * 60
    for y in range(top, bottom + 1, 60):
        page[y : y + 2, 80:800] = 0
    for x in columns:
        page[top:bottom, x : x + 2] = 0
    return bottom


def _page(gap_mm: float, lower_columns: list[int], bridge: bool = True) -> np.ndarray:
    """Две таблицы одна под другой; правая вертикаль-мост идёт вдоль обеих."""
    page = np.full((1300, 900), 255, np.uint8)
    upper_bottom = _grid(page, 100, 4, [80, 300, 520])
    lower_top = upper_bottom + mm_to_px(gap_mm, DPI)
    lower_bottom = _grid(page, lower_top, 4, lower_columns)
    if bridge:
        page[100:lower_bottom, 798:800] = 0
    return page


def _groups(page: np.ndarray) -> list[list]:
    lines = refine.drop_border_rules(rules.find_rules(page, DPI, binarize(page)), page.shape[:2], DPI)
    return [g for g in rules.cores(lines, DPI) if any(s.horizontal for s in g)]


def test_two_tables_joined_by_one_long_rule_split() -> None:
    """Графы разные, зазор 8 мм: по правилу стопок это две таблицы — мост их не склеивает.

    Левая боковая нижней таблицы сдвинута на 10 px: на одной оси с верхней она сшилась бы с ней
    через зазор как порванная линейка (``CHAIN_GAP_MM``), и мост был бы не единственной связью.
    """
    groups = _groups(_page(8.0, [90, 200, 650]))
    assert len(groups) == 2
    boxes = sorted((rules._extent(g) for g in groups), key=lambda box: box.y0)
    assert boxes[0].y1 < boxes[1].y0
    # Обе части сохранили правую боковую линейку (обрезок моста).
    for group in groups:
        assert any(not s.horizontal and s.box.x0 >= 790 for s in group)


def test_header_and_body_with_shared_columns_stay_one() -> None:
    """Графы совпадают и зазор 5 мм: это шапка и тело одной таблицы — не режем."""
    groups = _groups(_page(5.0, [80, 300, 520]))
    assert len(groups) == 1
