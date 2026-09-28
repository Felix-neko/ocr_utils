"""Непристроенные линейки: детектор таблиц отдаёт линейки, не вошедшие ни в одну находку."""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.geometry import KIND_TABLE, LooseRule
from ocr_utils.page_layout.tables import detector, rules
from ocr_utils.page_layout.tables.ruling import WORK_DPI
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


def _rule(points: list[tuple[float, float]], horizontal: bool = True, thickness: float = 2.0) -> LooseRule:
    """Линейка-сирота из точек оси."""
    return LooseRule(tuple(points), horizontal, thickness)


def test_rule_on_clean_paper_is_kept() -> None:
    """Отбивка на чистой бумаге: краски по сторонам нет, линейка остаётся."""
    binary = np.zeros((200, 400), np.uint8)
    binary[99:102, 50:250] = 255
    rule = _rule([(50.0, 100.0), (249.0, 100.0)])
    assert rules.side_ink(binary, rule, WORK_DPI) == (0.0, 0.0)
    assert rules.loose_rule_is_clean(binary, rule, WORK_DPI)


def test_rule_along_the_bottom_of_bold_letters_is_dropped() -> None:
    """«Линейка» по низу жирных букв (буквы стоят на ней вплотную сверху) — не отбивка."""
    binary = np.zeros((200, 400), np.uint8)
    binary[99:102, 50:150] = 255
    # Жирные «буквы» высотой 20 px с промежутками в 3 px, нижний край — на линейке.
    for x in range(50, 150, 12):
        binary[80:100, x : x + 9] = 255
    rule = _rule([(50.0, 100.0), (149.0, 100.0)])
    above, below = rules.side_ink(binary, rule, WORK_DPI)
    assert above > rules.LOOSE_MAX_SIDE_INK_HORIZONTAL and below == 0.0
    assert not rules.loose_rule_is_clean(binary, rule, WORK_DPI)


def test_vertical_rule_through_letter_stems_is_dropped_and_column_rule_is_kept() -> None:
    """Ствол, сшитый через строки заголовка, отсеивается; межколонная линейка на бумаге — нет."""
    binary = np.zeros((300, 300), np.uint8)
    # Ствол: две «буквы» шириной 12 px, линейка проходит по их левому краю.
    binary[50:80, 100:112] = 255
    binary[90:120, 100:112] = 255
    binary[50:120, 99:102] = 255
    stem = _rule([(100.0, 50.0), (100.0, 119.0)], horizontal=False)
    assert not rules.loose_rule_is_clean(binary, stem, WORK_DPI)
    binary[40:200, 219:222] = 255
    column = _rule([(220.0, 40.0), (220.0, 199.0)], horizontal=False)
    assert rules.loose_rule_is_clean(binary, column, WORK_DPI)


def test_long_rule_is_kept_even_with_text_close_by() -> None:
    """Линейка от ``LOOSE_KEEP_MM``, над которой вплотную стоит текст колонтитула, не трогается."""
    length = int(rules.mm_to_px(rules.LOOSE_KEEP_MM, WORK_DPI)) + 10
    binary = np.zeros((100, length + 40), np.uint8)
    binary[50:52, 20 : 20 + length] = 255
    binary[42:49, 20 : 20 + length] = 255
    rule = _rule([(20.0, 50.5), (20.0 + length - 1, 50.5)])
    assert rules.loose_rule_is_clean(binary, rule, WORK_DPI)


def _mm150(value: float) -> int:
    """Миллиметры в пиксели 150 dpi."""
    return int(round(value * 150 / 25.4))


def test_stems_of_letters_on_adjacent_lines_are_not_a_rule() -> None:
    """Два ствола букв друг под другом (5 и 3,7 мм через разрыв 2,7 мм) — не вертикальная линейка."""
    gray = np.full((300, 200), 245, np.uint8)
    top, gap = 60, _mm150(2.7)
    gray[top : top + _mm150(5.0), 100:104] = 25
    lower = top + _mm150(5.0) + gap
    gray[lower : lower + _mm150(3.7), 100:104] = 25
    found = rules.find_rules(gray, WORK_DPI)
    assert not [s for s in found.vertical if s.box.x0 <= 102 <= s.box.x1]


def test_long_rule_torn_by_a_short_gap_stays_one_rule() -> None:
    """Длинная линейка с разрывом 2,5 мм (продавлена сканом) остаётся одной: куски длинные, цепочка их сшивает."""
    gray = np.full((200, 700), 245, np.uint8)
    gray[100:103, 50:300] = 25
    gray[100:103, 300 + _mm150(2.5) : 650] = 25
    found = rules.find_rules(gray, WORK_DPI)
    long_rules = [s for s in found.horizontal if s.box.x1 - s.box.x0 > 500]
    assert len(long_rules) == 1
