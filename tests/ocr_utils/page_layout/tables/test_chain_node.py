"""Сшивка фрагментов линейки через узел таблицы: допуск 1 мм только там, где перпендикуляр проходит через стык насквозь."""

from __future__ import annotations

from ocr_utils.page_layout.geometry import Box
from ocr_utils.page_layout.tables.rules import Fragment, continues

# 150 dpi: обычный допуск 0,35 мм — 2 px, через узел 1 мм — 6 px, поле узла 1 мм — 6 px.
DPI = 150

# Вертикаль графы над жирной горизонталью: x 100–103, y 0–400.
UPPER = Fragment(Box(100, 0, 103, 400), False, 0.0, 1200)


def lower(shift_px: int) -> Fragment:
    """Обрывок той же графы под горизонталью (y 406–430), сдвинутый вправо на ``shift_px``."""
    return Fragment(Box(100 + shift_px, 406, 103 + shift_px, 430), False, 0.0, 72)


def rule(x0: int, x1: int) -> Fragment:
    """Горизонталь в зазоре между обрывками (y 401–404) от ``x0`` до ``x1``."""
    return Fragment(Box(x0, 401, x1, 404), True, 0.0, 3 * (x1 - x0))


def test_сдвиг_в_пределах_обычного_допуска_сшивается_и_без_узла() -> None:
    assert continues(UPPER, lower(1), DPI)


def test_сдвиг_больше_обычного_без_узла_не_сшивается() -> None:
    assert not continues(UPPER, lower(3), DPI)


def test_сдвиг_через_узел_сшивается() -> None:
    """Строка номеров граф под жирной линейкой: риска на 2,5 px правее, горизонталь проходит насквозь."""
    assert continues(UPPER, lower(3), DPI, crossers=[rule(0, 300)])


def test_угол_таблицы_не_узел() -> None:
    """Горизонталь кончается на стыке (угол рамки): подчёркивание соседней колонки не пришивается (1966/05 с.29)."""
    assert not continues(UPPER, lower(3), DPI, crossers=[rule(0, 103)])


def test_сдвиг_больше_миллиметра_не_сшивается_и_через_узел() -> None:
    assert not continues(UPPER, lower(8), DPI, crossers=[rule(0, 300)])


def test_перпендикуляр_вне_зазора_не_узел() -> None:
    """Горизонталь далеко ниже стыка — к этой паре фрагментов она отношения не имеет."""
    far = Fragment(Box(0, 470, 300, 473), True, 0.0, 900)
    assert not continues(UPPER, lower(3), DPI, crossers=[far])
