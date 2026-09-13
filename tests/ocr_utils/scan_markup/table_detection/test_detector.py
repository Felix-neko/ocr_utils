"""Детектор таблиц и блок-схем: связные рёбра, вид объекта, граница без разрезанных букв.

Все проверки — на синтетике, у которой рамка и линейки известны по построению; каждая
отвечает одной поломке прежних версий, найденной на паке (см. README пакета).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from ocr_utils.scan_markup.table_detection import detector, kind, quality, refine, ruling, rules
from ocr_utils.scan_markup.table_detection.geometry import KIND_DIAGRAM, KIND_TABLE, Box
from ocr_utils.scan_markup.table_detection.grid import text_ink
from tests.ocr_utils.scan_markup.table_detection import synthetic
from tests.ocr_utils.scan_markup.table_detection.synthetic import load_font

DPI = synthetic.DPI


def _mm(value: float) -> int:
    return int(round(value * DPI / 25.4))


def _filled_table(**kwargs) -> synthetic.SyntheticTable:
    upright = {(row, column): f"я{row}{column}" for row in range(3) for column in range(3)}
    return synthetic.make_table(upright=upright, **kwargs)


def _covers(found: Box, wanted: tuple[int, int, int, int], slack: int = 40) -> bool:
    x0, y0, x1, y1 = wanted
    return found.x0 <= x0 + slack and found.y0 <= y0 + slack and found.x1 >= x1 - slack and found.y1 >= y1 - slack


def _tables(page: np.ndarray):
    return [table for table in detector.detect(page, DPI) if table.kind == KIND_TABLE]


def _crossed(page: np.ndarray, box: Box) -> int:
    lines = ruling.find_lines(page, DPI)
    components = quality.glyph_components(text_ink(page, lines))
    return sum(quality.crossed_glyphs(components, box).values())


# --- линейки ------------------------------------------------------------------


def test_bent_rule_is_found_as_one_rule() -> None:
    """Изогнутая линейка (сагитта 2 мм) обязана найтись одной цепочкой на всю длину."""
    table = _filled_table(col_widths=[400, 300, 300])
    bent = synthetic.bend(table.image, amplitude_px=_mm(2.0), periods=0.5)
    found = rules.find_rules(bent, DPI)
    width = table.xs[-1] - table.xs[0]
    longest = max((segment.length for segment in found.horizontal), default=0)
    assert longest >= 0.95 * width, f"самая длинная горизонталь {longest} px при ширине таблицы {width}"


def test_blob_with_dark_neighbourhood_is_not_a_rule() -> None:
    """Ядро кляксы недосвета — тонкая тёмная полоса в серой мути — линейкой не считается."""
    table = _filled_table()
    page, _ = synthetic.make_page(table, lines_above=2, lines_below=2)
    height = page.shape[0]
    # Клякса у левого края: серый градиент шириной 6 мм с чёрной сердцевиной 1 мм.
    x = 20
    for offset in range(_mm(3.0)):
        shade = int(60 + 120 * offset / _mm(3.0))
        page[40 : height - 40, x + _mm(3.0) - offset - 1] = shade
        page[40 : height - 40, x + _mm(3.0) + offset] = shade
    lines = rules.find_rules(page, DPI)
    assert not any(segment.box.x0 < x + _mm(6.0) for segment in lines.vertical), "клякса прошла как вертикаль"


# --- затравки и рост ----------------------------------------------------------


def test_running_head_rule_and_underline_do_not_join_the_table() -> None:
    """Линейка колонтитула над таблицей и подчёркивание сбоку в рамку не входят."""
    table = _filled_table()
    page, box = synthetic.make_page(table, lines_above=3, lines_below=3, gap_px=_mm(9.0), width=1100)
    # Колонтитульная линейка на 9 мм выше таблицы, во всю ширину набора.
    y = box[1] - _mm(9.0)
    page[y : y + synthetic.RULE_PX, 40 : page.shape[1] - 40] = synthetic.INK
    # Подчёркивание в соседней колонке, в 4 мм от правого нижнего угла таблицы.
    ux = box[2] + _mm(4.0)
    page[box[3] - 4 : box[3] - 4 + synthetic.RULE_PX, ux : ux + _mm(30.0)] = synthetic.INK
    found = _tables(page)
    assert found, "таблица обязана находиться"
    biggest = max(found, key=lambda item: item.box.area)
    assert biggest.box.y0 > y + 10, f"линейка колонтитула вошла в рамку: {biggest.box.as_tuple()}"
    assert biggest.box.x1 < ux, f"подчёркивание соседней колонки вошло в рамку: {biggest.box.as_tuple()}"


def test_total_row_without_rules_and_paragraph_stay_outside() -> None:
    """Строка «Итого» без линеек и абзац под таблицей остаются за рамкой: рамка — по рёбрам.

    Так решил человек по второй раскладке: строку «Итого» без рёбер лучше оставить снаружи,
    чем захватывать заголовок, подпись или номер страницы, которые проходили тот же тест.
    """
    table = _filled_table(col_widths=[260, 160, 160])
    page, box = synthetic.make_page(table, lines_below=5, gap_px=_mm(10.0))
    image = Image.fromarray(page.copy())
    draw = ImageDraw.Draw(image)
    font = load_font(20)
    y = box[3] + _mm(2.0)
    draw.text((box[0] + table.xs[0] + 40, y), "Итого", fill=synthetic.INK, font=font)
    for column in (1, 2):
        draw.text((box[0] + table.xs[column] + 40, y), "99", fill=synthetic.INK, font=font)
    page = np.asarray(image)
    found = _tables(page)
    assert found
    biggest = max(found, key=lambda item: item.box.area)
    last_rule = box[1] + table.ys[-1] + synthetic.RULE_PX
    assert biggest.box.y1 <= last_rule + 6, f"рамка ушла ниже последней линейки: {biggest.box.as_tuple()}"


def test_fit_rows_never_widens_and_cuts_a_paragraph_above() -> None:
    """Подгонка не расширяет рамку вбок и срезает абзац над таблицей до верхней линейки."""
    table = _filled_table()
    page, box = synthetic.make_page(table, lines_above=6)
    lines = ruling.find_lines(page, DPI)
    ink = text_ink(page, lines)
    wide = Box(20, 20, page.shape[1] - 20, box[3])
    fitted = refine.fit_rows(wide, lines, ink, DPI)
    assert fitted.x0 == wide.x0 and fitted.x1 == wide.x1
    assert fitted.y0 > box[1] - 60, f"абзац над таблицей остался в рамке: {fitted.as_tuple()}"


# --- граница ---------------------------------------------------------------------


def test_pushed_border_crosses_no_glyph_component() -> None:
    """После выталкивания ни одна сторона не режет компоненту краски; слово берётся целиком."""
    table = _filled_table()
    page, box = synthetic.make_page(table, lines_above=6, lines_below=6)
    lines = ruling.find_lines(page, DPI)
    components = quality.glyph_components(text_ink(page, lines))
    dirty = Box(box[0], max(0, box[1] - 12), box[2], box[3] + 6)
    pushed = refine.push_edges(dirty, components, lines, DPI, page.shape[:2])
    assert pushed.failed == 0
    assert sum(quality.crossed_glyphs(components, pushed.box).values()) == 0


def test_detector_leaves_no_letters_under_the_border() -> None:
    """Сквозная проверка: у таблицы на полосе с абзацами граница не режет ни одной буквы."""
    table = _filled_table()
    page, _ = synthetic.make_page(table, lines_above=6, lines_below=6, gap_px=12)
    for table_box in _tables(page):
        assert _crossed(page, table_box.box) == 0, f"граница режет буквы: {table_box.box.as_tuple()}"


# --- вид объекта -----------------------------------------------------------------


def test_linked_blocks_become_one_diagram() -> None:
    """Восемь коробок, часть связана пунктиром, — одна находка вида «схема», накрывающая все."""
    page = synthetic.make_linked_blocks()
    found = detector.detect(page, DPI)
    diagrams = [table for table in found if table.kind == KIND_DIAGRAM]
    assert diagrams, f"схема не найдена, находки: {[(t.kind, t.box.as_tuple()) for t in found]}"
    biggest = max(diagrams, key=lambda item: item.box.area)
    # Восемь коробок по 300x130 в два ряда по четыре: правый край 40 + 3·420 + 300, низ 40 + 250 + 130.
    assert _covers(biggest.box, (40, 40, 1600, 420), slack=20), biggest.box.as_tuple()
    assert not [table for table in found if table.kind == KIND_TABLE], "коробки схемы приняты за таблицу"


def test_table_next_to_a_drawing_stays_a_table() -> None:
    """Таблица в 12 мм от рисунка остаётся отдельной таблицей и в рисунок не сливается."""
    table = _filled_table()
    drawing = synthetic.make_linked_blocks(blocks=4, width=1000, dashed_from=9)
    width = drawing.shape[1]
    padded = np.full((table.image.shape[0], width), synthetic.PAPER, np.uint8)
    padded[:, : table.image.shape[1]] = table.image
    gap = np.full((_mm(12.0), width), synthetic.PAPER, np.uint8)
    page = np.vstack([padded, gap, drawing])
    found = detector.detect(page, DPI)
    tables = [item for item in found if item.kind == KIND_TABLE]
    assert tables, f"таблица потерялась: {[(t.kind, t.box.as_tuple()) for t in found]}"
    assert max(item.box.y1 for item in tables) <= table.image.shape[0] + 10, "рамка таблицы уехала в рисунок"


def test_kind_features_separate_grid_from_boxes() -> None:
    """Решётка таблицы — плотные пересечения; коробки схемы — редкие."""
    table = _filled_table()
    grid = ruling.find_lines(table.image, DPI)
    diagram = synthetic.make_linked_blocks()
    boxes = ruling.find_lines(diagram, DPI)
    grid_signs = kind.features_of(grid.horizontal + grid.vertical, Box(0, 0, *table.image.shape[::-1]), DPI)
    box_signs = kind.features_of(boxes.horizontal + boxes.vertical, Box(0, 0, *diagram.shape[::-1]), DPI)
    assert grid_signs.cross_density > 0.8
    assert box_signs.cross_density < kind.DIAGRAM_MAX_CROSS_DENSITY


def test_letter_protruding_past_the_rule_of_an_open_table_is_kept() -> None:
    """Буква, торчащая за крайнюю линейку открытой таблицы, попадает в рамку целиком.

    Её строка лежит внутри рамки — значит, компонента своя, и граница уходит наружу на её
    ширину (1971/03 с.38, «Министерство Грузинской ССР»). Подпись под таблицей, чья строка
    лежит снаружи целиком, в рамку по-прежнему не входит.
    """
    # Открытая таблица: во второй строке левой линейки нет, и буква строки торчит за край
    # рамки, который задают линейки соседних строк.
    table = _filled_table(col_widths=[260, 160, 160], missing_vertical={(1, 0)})
    page, box = synthetic.make_page(table, lines_below=3, gap_px=_mm(6.0))
    image = Image.fromarray(page.copy())
    draw = ImageDraw.Draw(image)
    font = load_font(20)
    # Буква поперёк левого края рамки, на высоте второй строки таблицы: большей частью снаружи.
    row_y = box[1] + table.ys[1] + 30
    letter_x = box[0] + table.xs[0] - 10
    draw.text((letter_x, row_y), "М", fill=synthetic.INK, font=font)
    page = np.asarray(image)
    found = _tables(page)
    assert found
    biggest = max(found, key=lambda item: item.box.area)
    assert biggest.box.x0 <= letter_x, f"торчащая буква отрезана: {biggest.box.as_tuple()}"
    assert _crossed(page, biggest.box) == 0
    last_rule = box[1] + table.ys[-1] + synthetic.RULE_PX
    assert biggest.box.y1 <= last_rule + 6, "подпись под таблицей попала в рамку"


def test_running_head_rule_does_not_widen_a_diagram() -> None:
    """Линейка колонтитула во всю полосу в 4 мм над схемой не растягивает её на всю ширину."""
    diagram = synthetic.make_linked_blocks(dashed_from=9)
    height, width = diagram.shape
    page = np.full((height + 120, width + 900), synthetic.PAPER, np.uint8)
    page[100 : 100 + height, 40 : 40 + width] = diagram
    # Колонтитульная линейка: во всю ширину полосы, в 4 мм над верхним рядом коробок.
    y = 100 + 40 - _mm(4.0)
    page[y : y + synthetic.RULE_PX, 40 : page.shape[1] - 40] = synthetic.INK
    found = detector.detect(page, DPI)
    diagrams = [item for item in found if item.kind == KIND_DIAGRAM]
    assert diagrams, f"схема не найдена: {[(t.kind, t.box.as_tuple()) for t in found]}"
    biggest = max(diagrams, key=lambda item: item.box.area)
    assert biggest.box.x1 <= 40 + width + 40, f"схема растянулась по линейке колонтитула: {biggest.box.as_tuple()}"


# --- регионы для базы ------------------------------------------------------------


def test_regions_scale_to_original_and_carry_kind_and_info() -> None:
    """Находки копии 1/4 уезжают в оригинал ×4, зажатые в кадр, с видом базы и JSON подробностей."""
    import json

    from ocr_utils.scan_markup.db.models import KIND_LINE_ART_SCHEMA, KIND_TABLE as DB_TABLE
    from ocr_utils.scan_markup.table_detection import to_regions
    from ocr_utils.scan_markup.table_detection.geometry import KIND_DRAWING, TableBox

    table = TableBox(Box(10, 20, 110, 220), score=0.9, skew_deg=0.4, metrics={"cells": 9.0}, kind=KIND_TABLE)
    drawing = TableBox(Box(0, 300, 200, 420), kind=KIND_DRAWING, rule_box=Box(5, 305, 190, 400))
    regions = to_regions([table, drawing], 4.0, (800, 1600))

    assert [(r.x1, r.y1, r.x2, r.y2) for r in regions] == [(40, 80, 440, 880), (0, 1200, 800, 1600)]
    assert [r.kind for r in regions] == [DB_TABLE, KIND_LINE_ART_SCHEMA]
    info = json.loads(regions[1].detector_info)
    assert info["kind"] == KIND_DRAWING and info["rule_box"] == [20, 1220, 760, 1600]
    assert json.loads(regions[0].detector_info)["metrics"] == {"cells": 9.0}


def test_detect_regions_end_to_end_on_a_synthetic_page() -> None:
    """Сквозной ход: серая копия -> регион вида ``table`` в пикселях оригинала."""
    from ocr_utils.scan_markup.db.models import KIND_TABLE as DB_TABLE
    from ocr_utils.scan_markup.table_detection import detect_regions

    table = _filled_table()
    page, box = synthetic.make_page(table, lines_above=3, lines_below=3)
    height, width = page.shape
    regions = detect_regions(page, DPI, None, 2.0, (width * 2, height * 2))
    assert [r.kind for r in regions] == [DB_TABLE]
    region = regions[0]
    left = box[0] + table.xs[0]
    bottom = box[1] + table.ys[-1] + synthetic.RULE_PX
    assert abs(region.x1 - 2 * left) <= 40 and abs(region.y2 - 2 * bottom) <= 40


def test_layout_from_surya_result_clamps_boxes() -> None:
    """Сырой ответ surya разбирается в блоки, рамки зажимаются в кадр."""
    from types import SimpleNamespace

    from ocr_utils.scan_markup.table_detection.layout import from_surya_result

    raw = SimpleNamespace(
        bboxes=[
            SimpleNamespace(bbox=[-2.4, 10.0, 50.6, 120.2], label="Table", confidence=0.91),
            SimpleNamespace(bbox=[60.0, 0.0, 999.0, 30.0], label="Text", confidence=0.5),
        ]
    )
    layout = from_surya_result(raw, 100, 200)
    assert (layout.width, layout.height) == (100, 200)
    assert [(b.label, b.box.as_tuple()) for b in layout.blocks] == [
        ("Table", (0, 10, 51, 120)),
        ("Text", (60, 0, 100, 30)),
    ]
    assert layout.blocks[0].is_table and layout.blocks[1].is_text
