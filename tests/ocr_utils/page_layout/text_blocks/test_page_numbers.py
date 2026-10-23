"""Колонка номеров страниц оглавления: номера за отточиями, одна вертикаль правых краёв, блок с выровненной правой стороной; сторона по концам отточий — не выключка."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ocr_utils.page_layout.text_blocks.alignment import AlignKind, alignment_of, leader_side
from ocr_utils.page_layout.text_blocks.leaders import Leader
from ocr_utils.page_layout.text_blocks.page_numbers import column_block, leader_numbers, number_columns

# Рендер — 300 dpi, рабочая копия — 150 dpi (k = 2).
K = 2.0
DPI = 150.0
MM = 300 / 25.4


def _toc(rights: list[float], ys: list[float], digits: list[int] | None = None) -> tuple[np.ndarray, list[Leader]]:
    """Краска оглавления (300 dpi): у строки отточие до x 600 px рендера и номер высотой 2 мм с правым краем ``rights``.

    Args:
        rights: Правые края номеров, px рендера.
        ys: Высоты строк (базовая линия номера), px рендера.
        digits: Число цифр номера (ширина 1.2 мм на цифру); по умолчанию 2.

    Returns:
        Краска ``bool`` и отточия в px рабочей копии.
    """
    ink = np.zeros((1400, 1000), bool)
    leaders = []
    for i, (right, y) in enumerate(zip(rights, ys)):
        for x in range(300, 600, 25):
            ink[int(y) - 3 : int(y) + 1, x : x + 4] = True  # точки отточия
        w, h = int((digits[i] if digits else 2) * 1.2 * MM), int(2.0 * MM)
        ink[int(y) - h : int(y), int(right) - w : int(right)] = True  # номер страницы
        leaders.append(Leader(x0=150.0, x1=301.0, y=(y - 1) / K, thickness=2.0, dots=12))
    return ink, leaders


def test_column_through_section_headings():
    """Номера 1–3 цифр с общим правым краем (дрожание 1 px), разбитые «заголовками» (пропуски строк), — одна колонка;
    её блок выровнен вправо, слева — нет."""
    ys = [100.0, 160.0, 220.0, 400.0, 460.0, 700.0, 760.0, 820.0, 1100.0, 1160.0]
    rights = [720.0 + (i % 2) for i in range(len(ys))]
    ink, leaders = _toc(rights, ys, digits=[1, 2, 3, 2, 2, 3, 1, 2, 2, 3])
    marks = leader_numbers(ink, leaders, K)
    assert len(marks) == len(ys)
    columns = number_columns(marks, DPI)
    assert len(columns) == 1 and len(columns[0]) == len(ys)
    block = column_block(columns[0], 0, DPI)
    verdict = alignment_of(block)
    assert verdict.kind is AlignKind.RIGHT and verdict.right.aligned and not verdict.left.aligned


def test_no_column_without_common_edge():
    """Номера без общей вертикали (шаг правых краёв 1.1 мм) и три номера — колонки нет."""
    ys = [100.0 + 60 * i for i in range(10)]
    ink, leaders = _toc([700.0 + 13 * i for i in range(10)], ys)
    assert number_columns(leader_numbers(ink, leaders, K), DPI) == []
    ink, leaders = _toc([720.0] * 3, ys[:3])
    assert number_columns(leader_numbers(ink, leaders, K), DPI) == []


def test_no_numbers_without_leaders():
    """Без отточий номеров не ищем: краска справа от текста ничего не даёт."""
    ink, _ = _toc([720.0] * 6, [100.0 + 60 * i for i in range(6)])
    assert leader_numbers(ink, [], K) == []


def test_leader_side():
    """Правая сторона блока, ряды которого кончаются концами отточий, — упор точек; без отточий — нет."""
    rows = [SimpleNamespace(y=50.0 + 15 * i, height=10.0, x1=301.0) for i in range(4)]
    block = SimpleNamespace(rows=rows, dpi=DPI)
    leaders = [Leader(x0=150.0, x1=301.0, y=50.0 + 15 * i, thickness=2.0, dots=12) for i in range(3)]
    assert leader_side(block, leaders)
    assert not leader_side(block, [])


def test_sparse_regular_leader_found():
    """Отточие с шагом 5.5 мм (1968/04 IMG_0052_1L) находится, неравномерная россыпь точек с тем же средним шагом — нет."""
    from ocr_utils.page_layout.text_blocks.leaders import leaders_of

    mm = DPI / 25.4
    work = np.full((300, 1000), 255, np.uint8)
    for i in range(8):
        x = int(100 + i * 5.5 * mm)
        work[100:104, x : x + 3] = 0  # точка 0.5 мм
    rng = np.random.default_rng(1)
    x = 100.0
    for _ in range(8):
        x += rng.choice([2.5, 8.5]) * mm  # шаги 2.5 и 8.5 мм вперемешку
        work[200:204, int(x) : int(x) + 3] = 0
    assert leaders_of(work, DPI)[0] == []  # без ``sparse`` редкие отточия не ищутся
    found, _ = leaders_of(work, DPI, sparse=True)
    assert len(found) == 1 and 100 <= found[0].y <= 104
