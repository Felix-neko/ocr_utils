"""Открытая таблица: шапка в линейках, тело — только куски вертикалей граф, разорванные подзаголовками."""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.tables import refine, rules
from ocr_utils.page_layout.tables.ruling import WORK_DPI, binarize, mm_to_px

DPI = WORK_DPI


def _page(body_columns: list[int], pieces: int = 3, gap_mm: float = 9.0) -> tuple[np.ndarray, int]:
    """Полоса с шапкой таблицы и телом из кусков вертикалей.

    Args:
        body_columns: x вертикалей тела, px; шапка имеет вертикали на x = 300 и 450.
        pieces: Сколько кусков у каждой вертикали тела.
        gap_mm: Разрыв между шапкой и телом и между кусками (строка подзаголовка).

    Returns:
        Кадр (бумага 255, краска 0) и y низа последнего куска.
    """
    page = np.full((1400, 900), 255, np.uint8)
    top, header_bottom = 200, 260
    page[top : top + 2, 80:820] = 0
    page[header_bottom : header_bottom + 2, 80:820] = 0
    for x in (300, 450):
        page[top:header_bottom, x : x + 2] = 0
    gap, piece = mm_to_px(gap_mm, DPI), mm_to_px(40.0, DPI)
    y = header_bottom + gap
    for _ in range(pieces):
        for x in body_columns:
            page[y : y + piece, x : x + 2] = 0
        y += piece + gap
    return page, y - gap


def _groups(page: np.ndarray) -> list[list]:
    lines = refine.drop_border_rules(rules.find_rules(page, DPI, binarize(page)), page.shape[:2], DPI)
    return [g for g in rules.cores(lines, DPI) if any(s.horizontal for s in g)]


def test_body_pieces_under_header_columns_join_the_header() -> None:
    """Куски вертикалей на графах шапки, разорванные на 9 мм, дотягивают ядро до низа тела."""
    page, bottom = _page([300, 450])
    groups = _groups(page)
    assert len(groups) == 1
    box = rules._extent(groups[0])
    assert box.y0 <= 202 and box.y1 >= bottom - 2


def test_single_column_rule_under_header_is_not_taken() -> None:
    """Одна вертикаль под шапкой — межколонная линейка вёрстки, а не тело: ядро не растёт."""
    page, _bottom = _page([300])
    groups = _groups(page)
    assert len(groups) == 1
    assert rules._extent(groups[0]).y1 <= 265


def test_body_after_too_wide_gap_is_not_taken() -> None:
    """Тело дальше порога зазора под шапкой не берётся."""
    page, _bottom = _page([300, 450], gap_mm=rules.OPEN_BODY_GAP_MM + 5)
    groups = _groups(page)
    assert rules._extent(groups[0]).y1 <= 265


def _section_with_total(page: np.ndarray, top: int, columns: list[int]) -> int:
    """Раздел тела, ставший своим ядром: куски вертикалей пересекают двойную линейку над «Итого».

    Args:
        page: Кадр, рисуется на месте.
        top: y начала раздела.
        columns: x вертикалей раздела.

    Returns:
        y низа раздела.
    """
    rule_y, bottom = top + 150, top + 200
    page[rule_y : rule_y + 2, 80:820] = 0
    page[rule_y + 4 : rule_y + 6, 80:820] = 0
    for x in columns:
        page[top:bottom, x : x + 2] = 0
    return bottom


def test_next_section_with_its_own_rule_joins_the_open_table() -> None:
    """Раздел под подзаголовком со своей двойной линейкой над «Итого» — продолжение той же таблицы."""
    page, bottom = _page([300, 450], pieces=1)
    end = _section_with_total(page, bottom + mm_to_px(9.0, DPI), [300, 450])
    groups = _groups(page)
    assert len(groups) == 1
    assert rules._extent(groups[0]).y1 >= end - 2


def test_framed_table_below_with_other_columns_stays_apart() -> None:
    """Под шапкой через 9 мм — своя таблица той же ширины с другими графами: не приклеивается."""
    page, _bottom = _page([300, 450], pieces=0)
    _section_with_total(page, 260 + mm_to_px(9.0, DPI), [80, 200, 620, 818])
    groups = _groups(page)
    assert len(groups) == 2


def test_next_table_with_own_top_rule_stays_apart() -> None:
    """Те же графы, но у нижней части своя верхняя линейка во всю ширину — это «Таблица 2», не раздел."""
    page, bottom = _page([300, 450], pieces=1)
    top = bottom + mm_to_px(9.0, DPI)
    page[top : top + 2, 80:820] = 0
    _section_with_total(page, top, [300, 450])
    groups = _groups(page)
    assert len(groups) == 2
