"""Тесты ядра стенда границ блоков на синтетических рядах: подчёркивания, «дотягивание», куски строки, выровненные стороны, ступенчатый контур и меры."""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.text_blocks.barriers import BarrierLines
from ocr_utils.page_layout.text_blocks.blocks import Row
from ocr_utils.page_layout.text_blocks.lines import LineAxis
from research.block_envelopes.boundary import _local_side, stepped_polygon
from research.block_envelopes.grouping import merge_same_line, reach_components, underline_free
from research.block_envelopes.measures import BlockShape, RowBox, overshoot_px, saw_px, wedge_share

DPI = 150.0


def axis(x0: float, x1: float, y: float, height: float = 12.0) -> LineAxis:
    """Прямая ось строки от ``x0`` до ``x1`` на высоте ``y``."""
    xs = np.linspace(x0, x1, 20)
    return LineAxis(
        points=np.column_stack([xs, np.full_like(xs, y)]),
        height=height,
        column=0,
        dpi=DPI,
        cross=False,
        sagitta_mm=0.0,
        slope_deg=0.0,
        bend_mm=0.0,
        resid_parabola_mm=0.0,
    )


def row(x0: float, x1: float, y: float, height: float = 12.0, weight: float = 2.0) -> Row:
    """Ряд из одной прямой оси с краями краски по её концам."""
    return Row(y=y, height=height, x0=x0, x1=x1, axes=(axis(x0, x1, y, height),), weight=weight, glyph_h=8.0)


def test_underline_under_row_is_dropped_separator_between_rows_kept() -> None:
    """Черта в 8 px под осью строки — подчёркивание (1966/01 IMG_0036_2R); черта посередине промежутка остаётся."""
    axes = [axis(50, 190, 334, 17), axis(50, 840, 358, 17)]
    underline = [(56, 342), (186, 343)]
    separator = [(50, 400), (300, 400)]
    kept = underline_free(BarrierLines.of([underline, separator]), axes + [axis(50, 840, 420, 17)])
    assert len(kept.lines) == 1
    assert np.allclose(kept.lines[0][:, 1], 400)


def test_reach_splits_rows_that_never_overlap() -> None:
    """Хвост абзаца и «* * *» правее него не перекрыты по x — две части (1966/02 IMG_0073_1L)."""
    rows = [row(70, 836, 500), row(70, 156, 524), row(406, 482, 561)]
    parts = reach_components(rows)
    assert [len(part) for part in parts] == [2, 1]


def test_same_line_pieces_merge_into_one_row() -> None:
    """Два куска одной строки заголовка на одной высоте рядом — один ряд с общими краями."""
    rows = merge_same_line([row(330, 484, 378, 28), row(455 + 40, 600, 376, 27), row(100, 800, 450, 14)], DPI)
    assert len(rows) == 2
    assert rows[0].x0 == 330 and rows[0].x1 == 600


def test_local_side_fills_short_lines_and_ignores_ragged() -> None:
    """Правая сторона выровнена — короткая строка добивается; рваная сторона — ``nan``."""
    ys = np.arange(10) * 20.0
    rights = np.full(10, 800.0)
    rights[4] = 500.0  # конец абзаца
    side = _local_side(ys, rights, +1.0, DPI)
    assert abs(side[4] - 800.0) < 1.0
    ragged = np.array([500, 700, 420, 780, 610, 530, 690, 450, 760, 580], dtype=float)
    assert np.isnan(_local_side(ys, ragged, +1.0, DPI)).all()


def test_stepped_polygon_has_no_wedge_between_indented_rows() -> None:
    """Две строки с разным отступом (подпись «Стандарты и качество») — ступень, а не клин наискосок."""
    rows = [row(610, 900, 100), row(800, 900, 124)]
    polygon = stepped_polygon(rows, DPI, fit_sides=True)
    shape = BlockShape(polygon, tuple(RowBox(r.y, r.x0, r.x1, r.height) for r in rows), pitch=24.0)
    assert wedge_share(shape, DPI) < 0.02
    assert overshoot_px(shape, DPI) < 4.0


def test_saw_measure_sees_teeth() -> None:
    """Зубья на верхней крышке дают «пилу», прямоугольник — нет."""
    box = np.array([[0, 0], [300, 0], [300, 60], [0, 60]], dtype=float)
    top = [[x, 0 if (x // 10) % 2 else 8] for x in range(0, 301, 5)]
    teeth = np.array(top + [[300, 60], [0, 60]], dtype=float)
    rows = (RowBox(30, 0, 300, 50),)
    assert saw_px(BlockShape(box, rows, 20.0), DPI) < 1.0
    assert saw_px(BlockShape(teeth, rows, 20.0), DPI) > 50.0


def test_letter_strokes_inside_heading_are_not_barriers() -> None:
    """Перекладина по верху букв и ножка «В» внутри полосы строки заголовка — не разделители (1971/10 IMG_0010_1L)."""
    axes = [axis(336, 652, 237, 33), axis(335, 488, 294, 36)]
    bar = [(388, 277), (479, 277)]
    stem = [(340, 280), (340, 308)]
    assert underline_free(BarrierLines.of([bar, stem]), axes).empty
