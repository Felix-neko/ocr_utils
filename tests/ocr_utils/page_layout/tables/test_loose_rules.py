"""Непристроенные линейки: детектор таблиц отдаёт линейки, не вошедшие ни в одну находку."""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.geometry import KIND_TABLE
from ocr_utils.page_layout.tables import detector
from tests.ocr_utils.page_layout.tables import synthetic
from tests.ocr_utils.page_layout.tables.synthetic import INK

DPI = synthetic.DPI


def _mm(value: float) -> int:
    return int(round(value * DPI / 25.4))


def _page_with_rules() -> tuple[np.ndarray, tuple[int, int, int, int], int, tuple[int, int, float]]:
    """Полоса: таблица, под ней прямая отбивка и ниже — изогнутая (дуга) линейка.

    Returns:
        Кадр, рамка таблицы, y прямой отбивки и параметры дуги (y середины, x середины, прогиб px).
    """
    upright = {(row, column): f"я{row}{column}" for row in range(3) for column in range(3)}
    table = synthetic.make_table(upright=upright)
    page, box = synthetic.make_page(table, lines_above=2, lines_below=2)
    height, width = page.shape
    # Дописываем снизу поле под две линейки.
    extra = np.full((_mm(60), width), synthetic.PAPER, np.uint8)
    page = np.vstack([page, extra])
    straight_y = height + _mm(15)
    page[straight_y : straight_y + 4, _mm(15) : width - _mm(15)] = INK
    # Дуга: прогиб 12 px на всю длину (изгиб полосы у корешка).
    arc_y, sag = height + _mm(40), 12
    x0, x1 = _mm(15), width - _mm(15)
    middle = (x0 + x1) / 2
    for x in range(x0, x1):
        y = int(round(arc_y + sag * (1 - ((x - middle) / ((x1 - x0) / 2)) ** 2)))
        page[y : y + 4, x] = INK
    return page, box, straight_y, (arc_y, int(middle), float(sag))


def test_detect_is_detect_all_found() -> None:
    """``detect`` отдаёт ровно находки ``detect_all``."""
    page, _box, _y, _arc = _page_with_rules()
    assert [t.box for t in detector.detect(page, DPI)] == [t.box for t in detector.detect_all(page, DPI).found]


def test_rules_outside_findings_are_loose_and_table_rules_are_not() -> None:
    """Отбивка и дуга вне таблицы — в непристроенных; ни одна линейка таблицы туда не попала."""
    page, _box, straight_y, (arc_y, _middle, _sag) = _page_with_rules()
    result = detector.detect_all(page, DPI)
    tables = [t for t in result.found if t.kind == KIND_TABLE]
    assert len(tables) == 1
    table_box = tables[0].box
    for rule in result.loose_rules:
        cx = (rule.box.x0 + rule.box.x1) / 2
        cy = (rule.box.y0 + rule.box.y1) / 2
        assert not (table_box.x0 <= cx <= table_box.x1 and table_box.y0 <= cy <= table_box.y1)
    horizontal = [r for r in result.loose_rules if r.horizontal]
    assert any(abs(r.box.y0 - straight_y) <= 4 for r in horizontal)
    assert any(arc_y - 4 <= r.box.y0 and r.box.y1 <= arc_y + 20 for r in horizontal)


def test_curved_rule_is_a_polyline_along_the_curve() -> None:
    """Изогнутая линейка — ломаная из многих точек, лежащая на дуге с точностью в пару пикселей."""
    page, _box, _y, (arc_y, middle, sag) = _page_with_rules()
    result = detector.detect_all(page, DPI)
    arcs = [r for r in result.loose_rules if r.horizontal and r.box.y0 >= arc_y - 4]
    assert len(arcs) == 1
    arc = arcs[0]
    assert len(arc.points) > 10
    half = (arc.box.x1 - arc.box.x0) / 2
    for x, y in arc.points:
        expected = arc_y + 1.5 + sag * (1 - ((x - middle) / half) ** 2)
        assert abs(y - expected) <= 3.0
