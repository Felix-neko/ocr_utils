"""Линейки и рамка таблицы на синтетике."""

from __future__ import annotations

import numpy as np

from research.legacy.table_processing.detection import ruling
from research.legacy.table_processing.geometry import iou

from tests.research.legacy.table_processing.synthetic import DPI, make_table


def test_lines_found_on_plain_table():
    table = make_table(upright={(0, 0): "Наименование", (1, 0): "Тракторы"})
    lines = ruling.find_lines(table.image, DPI)
    assert len(lines.horizontal) >= 4
    assert len(lines.vertical) >= 4


def test_table_box_matches_construction():
    table = make_table(upright={(0, 0): "Наименование", (1, 1): "1790"})
    found = ruling.detect(table.image, DPI)
    assert len(found) == 1
    expected = (table.xs[0], table.ys[0], table.xs[-1] + ruling.mm_to_px(0.3, DPI), table.ys[-1])
    from research.legacy.table_processing.geometry import Box

    assert iou(found[0].box, Box(*expected)) > 0.9


def test_skew_is_measured():
    table = make_table(upright={(0, 0): "Наименование"}, skew_deg=1.2)
    found = ruling.detect(table.image, DPI)
    assert found, "таблица на слегка перекошенной полосе обязана находиться"
    assert abs(found[0].skew_deg - 1.2) < 0.4


def test_text_only_page_has_no_table():
    page = np.full((900, 700), 245, np.uint8)
    from PIL import Image, ImageDraw

    from research.legacy.table_processing.render.fonts import load_font

    canvas = Image.fromarray(page)
    draw = ImageDraw.Draw(canvas)
    font = load_font(22)
    for index in range(20):
        draw.text((40, 40 + index * 34), "обычная строка набора без всяких линеек", fill=25, font=font)
    assert ruling.detect(np.asarray(canvas), DPI) == []
