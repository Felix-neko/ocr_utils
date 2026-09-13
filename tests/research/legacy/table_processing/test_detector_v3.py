"""Третья версия детектора: рамка не хватает чужого, не режет буквы и добирает своё.

Все проверки — на синтетике, у которой рамка таблицы известна по построению. На паке этого
не проверить: там эталон живёт только в глазах человека, и он размечен по полосам, а не по
рамкам (см. ``labels.DetectorPageLabel``).
"""

from __future__ import annotations

import numpy as np

from research.legacy.table_processing.detection import quality, refine, ruling, ruling_v3
from research.legacy.table_processing.geometry import Box
from tests.research.legacy.table_processing import synthetic

DPI = synthetic.DPI


def _filled_table(**kwargs) -> synthetic.SyntheticTable:
    """Таблица С ТЕКСТОМ в ячейках.

    Пустая решётка третьей версией отвергается намеренно — правило «мало букв и пустые
    ячейки» ловит координатные сетки графиков, — поэтому синтетика для этих тестов обязана
    быть заполненной, иначе тест проверял бы не то, что написано в его названии.
    """
    upright = {(row, column): f"я{row}{column}" for row in range(3) for column in range(3)}
    return synthetic.make_table(upright=upright, **kwargs)


def _covers(found: Box, wanted: tuple[int, int, int, int], slack: int = 40) -> bool:
    x0, y0, x1, y1 = wanted
    return found.x0 <= x0 + slack and found.y0 <= y0 + slack and found.x1 >= x1 - slack and found.y1 >= y1 - slack


def test_paragraph_above_the_table_is_not_captured() -> None:
    """Абзац над таблицей в рамку попадать не должен — это худший из дефектов по мнению человека."""
    table = _filled_table()
    page, box = synthetic.make_page(table, lines_above=8, lines_below=4)
    found = ruling_v3.detect(page, DPI)
    assert found, "таблица на синтетической полосе обязана находиться"
    biggest = max(found, key=lambda item: item.box.area)
    assert _covers(biggest.box, box), "таблица должна быть покрыта целиком"
    # Верх рамки не должен уехать в абзац: допускаем половину зазора, но не строки набора.
    assert biggest.box.y0 > box[1] - 40


def test_scan_edge_rule_does_not_stretch_the_box() -> None:
    """Тень корешка у края кадра не должна склеивать абзацы с таблицей в один кластер."""
    table = _filled_table()
    page, box = synthetic.make_page(table, lines_above=8, lines_below=6, edge_rule=True)
    found = ruling_v3.detect(page, DPI)
    assert found
    biggest = max(found, key=lambda item: item.box.area)
    assert biggest.box.height < (box[3] - box[1]) * 1.6, "рамка растянулась на абзацы вокруг таблицы"


def test_border_rules_are_dropped_only_without_crossings() -> None:
    """Выбрасывается линейка БЕЗ пересечений; внешняя вертикаль таблицы остаётся на месте."""
    table = _filled_table()
    page, _ = synthetic.make_page(table, lines_above=4, edge_rule=True)
    lines = ruling.find_lines(page, DPI)
    cleaned = refine.drop_border_rules(lines, page.shape[:2], DPI)
    assert len(cleaned.vertical) < len(lines.vertical), "паразитная вертикаль должна была уйти"
    # Ни одна вертикаль самой таблицы не потерялась: их столько же, сколько разделителей.
    table_lines = ruling.find_lines(table.image, DPI)
    assert len(cleaned.vertical) >= len(table_lines.vertical) - 1


def test_two_tables_on_one_page_stay_two() -> None:
    """Доращивание не имеет права срастить две таблицы в одну — это ломало 1972/07 с.75."""
    first = _filled_table()
    second = _filled_table()
    gap = np.full((60, max(first.image.shape[1], second.image.shape[1])), synthetic.PAPER, np.uint8)
    width = gap.shape[1]

    def padded(image: np.ndarray) -> np.ndarray:
        canvas = np.full((image.shape[0], width), synthetic.PAPER, np.uint8)
        canvas[:, : image.shape[1]] = image
        return canvas

    page = np.vstack([padded(first.image), gap, padded(second.image)])
    found = ruling_v3.detect(page, DPI)
    assert len(found) >= 2, f"ожидали две таблицы, нашли {len(found)}"
    assert all(table.box.height < page.shape[0] * 0.7 for table in found)


def test_short_bottom_rule_is_reached_by_growing() -> None:
    """Продавленное нижнее ребро короче общего порога должно добираться доращиванием."""
    table = _filled_table()
    page, box = synthetic.make_page(table, lines_below=4)
    height, width = page.shape[:2]
    # Дорисовываем короткий обрывок ребра под таблицей — так выглядит продавленная линейка.
    stub_y = min(height - 5, box[3] + 20)
    page[stub_y : stub_y + synthetic.RULE_PX, box[0] : box[0] + 60] = synthetic.INK
    lines = ruling.find_lines(page, DPI)
    grown = ruling_v3.grow(page, lines, Box(*box), DPI)
    assert grown.y1 >= box[3], "рост не имеет права уменьшать рамку"


def test_snapping_leaves_no_letters_under_the_border() -> None:
    """После прилипания ни одна сторона рамки не должна идти по буквам."""
    table = _filled_table()
    page, box = synthetic.make_page(table, lines_above=6, lines_below=6)
    lines = ruling.find_lines(page, DPI)
    from research.legacy.table_processing.structure.ruling_grid import text_ink

    ink = text_ink(page, lines)
    # Нарочно сдвигаем рамку так, чтобы верх шёл по строке набора.
    dirty = Box(box[0], max(0, box[1] - 30), box[2], box[3])
    assert quality.edge_ink(ink, dirty)["сверху"] >= 0.0  # мера считается, не падает
    snapped = refine.snap_edges(dirty, ink, lines, DPI)
    assert snapped.box.width > 0 and snapped.box.height > 0


def test_growing_takes_both_halves_of_a_block_diagram() -> None:
    """Две части схемы, разнесённые на 10 мм ПО КРАСКЕ, доращивание обязано собрать вместе.

    Проверяется именно ``grow``, а не весь детектор: блок-схему проверка отвергает, и до
    доращивания дело бы не дошло. Зазор мерится от штриха до штриха, а не от края картинки:
    у ``make_block_diagram`` сверху и снизу есть пустые поля, и с ними «зазор 10 мм» на деле
    превращался в 21 мм — больше, чем достаёт склейка.
    """
    gap_px = int(10 * DPI / 25.4)
    height = 400
    page = np.full((height, 700), synthetic.PAPER, np.uint8)
    rule = synthetic.RULE_PX

    def block(top: int, left: int, width: int = 200, tall: int = 60) -> None:
        page[top : top + rule, left : left + width] = synthetic.INK
        page[top + tall : top + tall + rule, left : left + width] = synthetic.INK

    block(20, 40)
    block(20, 300)
    second_top = 20 + 60 + rule + gap_px
    block(second_top, 40)
    block(second_top, 300)

    lines = ruling.find_lines(page, DPI)
    start = Box(30, 10, 520, 20 + 60 + rule + 5)
    grown = ruling_v3.grow(page, lines, start, DPI)
    assert grown.y1 > second_top, f"рост не дотянулся до нижней половины схемы: {grown.as_tuple()}"


def test_page_level_outside_rules_does_not_punish_finding_more_tables() -> None:
    """Мера «линеек снаружи» обязана считаться по полосе, а не складываться по находкам."""
    first = _filled_table()
    second = _filled_table()
    width = max(first.image.shape[1], second.image.shape[1])

    def padded(image: np.ndarray) -> np.ndarray:
        canvas = np.full((image.shape[0], width), synthetic.PAPER, np.uint8)
        canvas[:, : image.shape[1]] = image
        return canvas

    gap = np.full((60, width), synthetic.PAPER, np.uint8)
    page = np.vstack([padded(first.image), gap, padded(second.image)])
    lines = ruling.find_lines(page, DPI)
    boxes = [Box(0, 0, width, first.image.shape[0]), Box(0, first.image.shape[0] + 60, width, page.shape[0])]
    both = quality.outside_rules_page(lines, boxes, DPI)
    alone = quality.outside_rules_page(lines, boxes[:1], DPI)
    assert both <= alone, "найдя вторую таблицу, детектор не должен выглядеть хуже"


def test_trimming_keeps_a_header_row_with_short_verticals() -> None:
    """Шапку, у которой разделители граф короче порога поиска линеек, обрезка резать не вправе.

    Это регрессия, пойманная на приёмке: у таблицы 1973/01 с.22 вертикали шапки идут на её
    высоту, 5.8 мм, то есть короче порога 8 мм. Обрезка видела полосу без вертикалей и срезала
    шапку вместе с заголовками колонок.
    """
    width, height = 800, 400
    page = np.full((height, width), synthetic.PAPER, np.uint8)
    rule = synthetic.RULE_PX
    header_top, header_bottom, table_bottom = 40, 40 + int(6 * DPI / 25.4), 340

    for y in (header_top, header_bottom, table_bottom):  # три сквозные горизонтали
        page[y : y + rule, 40 : width - 40] = synthetic.INK
    for x in (300, 550):  # вертикали ТОЛЬКО в теле, ниже шапки
        page[header_bottom : table_bottom + rule, x : x + rule] = synthetic.INK

    lines = ruling.find_lines(page, DPI)
    box = Box(40, header_top, width - 40, table_bottom + rule)
    trimmed = refine.trim_ruleless(box, lines, DPI)
    assert trimmed.y0 <= header_top + rule, f"обрезка съела шапку: {trimmed.as_tuple()} против {box.as_tuple()}"


def test_trimming_still_cuts_a_paragraph_above_the_table() -> None:
    """Но абзац НАД таблицей обрезка резать обязана: над ним сквозной горизонтали нет."""
    table = _filled_table()
    page, box = synthetic.make_page(table, lines_above=8)
    lines = ruling.find_lines(page, DPI)
    wide = Box(20, 20, page.shape[1] - 20, box[3])
    trimmed = refine.trim_ruleless(wide, lines, DPI)
    assert trimmed.y0 > box[1] - 60, f"абзац над таблицей остался в рамке: {trimmed.as_tuple()}"
