"""Непристроенные линейки: детектор таблиц отдаёт линейки, не вошедшие ни в одну находку."""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.geometry import KIND_TABLE, LooseRule
from ocr_utils.page_layout.tables import detector, loose_filter, rules
from ocr_utils.page_layout.tables.ruling import WORK_DPI, binarize
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


# --- Вторая ступень: признаки трассы (loose_filter) -------------------------------------------------


def _drop(gray: np.ndarray, rule: LooseRule) -> loose_filter.LooseDrop | None:
    """Причина отбраковки сироты на серой полосе 150 dpi (как в ``detect_all``)."""
    binary = binarize(gray)
    components = loose_filter.page_components(binary)
    return loose_filter.drop_reason(gray, components, rules.side_ink(binary, rule, WORK_DPI), rule, WORK_DPI)


def _paper(height: int = 400, width: int = 600) -> np.ndarray:
    """Чистая серая полоса 150 dpi: бумага 240."""
    return np.full((height, width), 240, np.uint8)


def test_filter_keeps_clean_rule_and_underline_with_descenders() -> None:
    """Отбивка на бумаге и подчёркивание с прилипшими хвостами букв — настоящие линейки."""
    gray = _paper()
    gray[100:102, 100:400] = 20
    assert _drop(gray, _rule([(100.0, 100.5), (399.0, 100.5)])) is None
    gray[300:302, 100:400] = 20
    for x in range(110, 390, 30):
        gray[285:302, x : x + 3] = 20
    assert _drop(gray, _rule([(100.0, 300.5), (399.0, 300.5)])) is None


def test_filter_drops_stems_stacked_across_rows() -> None:
    """Стволы двух жирных «Н» через межстрочье (IMG_0061_1L в миниатюре): буквы вразрядку, по бокам бумага."""
    gray = _paper()
    for top in (200, 245):
        gray[top : top + 30, 300:306] = 20
        gray[top : top + 30, 318:324] = 20
        gray[top + 13 : top + 17, 300:324] = 20
    stem = _rule([(303.0, 200.0), (303.0, 274.0)], horizontal=False, thickness=6.0)
    assert _drop(gray, stem) is loose_filter.LooseDrop.GLYPHS


def test_filter_drops_serif_row() -> None:
    """Низы жирных букв одной строки, сшитые через просветы, — буквы."""
    gray = _paper()
    for x in range(100, 380, 40):
        gray[270:300, x : x + 30] = 20
        gray[275:295, x + 8 : x + 22] = 240
    assert _drop(gray, _rule([(100.0, 298.0), (369.0, 298.0)], thickness=3.0)) is loose_filter.LooseDrop.GLYPHS


def test_filter_drops_dashes_chained_with_digits() -> None:
    """Тире, сшитые через цифры («6—9—12»): каждое тире вытянуто, но длинного пробега «не букв» нет."""
    gray = _paper()
    x = 100
    for _ in range(4):
        gray[200:202, x : x + 18] = 20  # тире 3 мм
        gray[190:212, x + 20 : x + 30] = 20  # «цифра», компактная, на уровне тире
        x += 32
    assert _drop(gray, _rule([(100.0, 200.5), (float(x - 3), 200.5)])) is loose_filter.LooseDrop.SHORT_RUN


def test_filter_drops_fraction_bar() -> None:
    """Черта дроби: числитель и знаменатель вплотную с обеих сторон."""
    gray = _paper()
    gray[200:202, 100:200] = 20
    for x in range(105, 195, 12):
        gray[190:198, x : x + 8] = 20
        gray[204:212, x : x + 8] = 20
    assert _drop(gray, _rule([(100.0, 200.5), (199.0, 200.5)])) is loose_filter.LooseDrop.FRACTION


def test_filter_drops_gray_edge_shelf() -> None:
    """Кромка листа: серая полка (140) шириной 6 px вместо чёрного ядра штриха."""
    gray = _paper()
    gray[50:350, 30:36] = 140
    gray[50:350, :30] = 250
    shelf = _rule([(33.0, 50.0), (33.0, 349.0)], horizontal=False, thickness=6.0)
    # Порог Otsu на почти пустой полосе проводит полку в краску — как на сканах.
    assert _drop(gray, shelf) is loose_filter.LooseDrop.EDGE
