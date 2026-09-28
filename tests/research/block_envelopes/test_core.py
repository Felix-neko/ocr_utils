"""Тесты ядра стенда границ блоков на синтетических рядах: подчёркивания, «дотягивание», куски строки, выровненные стороны, ступенчатый контур и меры."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ocr_utils.page_layout.text_blocks.barriers import BarrierLines
from ocr_utils.page_layout.text_blocks.blocks import Row
from ocr_utils.page_layout.text_blocks.lines import LineAxis
from research.block_envelopes.axes_fix import fixed_axes, slope_of
from research.block_envelopes.boundary import (
    _axis_points,
    _guided_at,
    _guides,
    _local_side,
    _smooth_axis,
    stepped_polygon,
)
from research.block_envelopes.grouping import (
    merge_overlapping_pieces,
    merge_same_line,
    reach_components,
    resolve_shared_rows,
    underline_free,
)
from research.block_envelopes.smooth_sides import side_curve
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


def test_overlapping_pieces_of_one_line_merge_despite_style() -> None:
    """Жирное начало и обычный хвост одной строки, налезающие по x, — один ряд (1972/03 IMG_0108_2R)."""
    bold = row(64, 558, 1254, 14, weight=3.5)
    plain = row(414, 826, 1254, 14, weight=2.0)
    merged = merge_overlapping_pieces([bold, plain, row(64, 826, 1277, 14)])
    assert len(merged) == 2
    assert merged[0].x0 == 64 and merged[0].x1 == 826


def test_side_by_side_signature_far_away_is_not_merged() -> None:
    """Подпись справа через промежуток в три высоты строки остаётся отдельным рядом."""
    assert len(merge_overlapping_pieces([row(60, 300, 500, 20), row(360, 500, 500, 14, weight=3.5)])) == 2


def test_shared_row_moves_to_block_of_longer_piece() -> None:
    """Короткий кусок строки, попавший в чужой блок, уходит к блоку длинного куска той же высоты."""
    upper = [row(64, 826, 1230), row(414, 826, 1254)]
    lower = [row(64, 558, 1254), row(64, 826, 1277)]
    groups = resolve_shared_rows([upper, lower])
    assert sorted(len(group) for group in groups) == [1, 3]


def test_tilted_heading_axis_is_replaced() -> None:
    """Ось крупной строки с наклоном 5° при горизонтальном корпусе заменяется прямой по глифам с наклоном ≤ 1°."""
    body = [axis(50, 800, 100 + 20 * i, 14) for i in range(6)]
    xs = np.linspace(290, 680, 20)
    tilted = LineAxis(
        points=np.column_stack([xs, 880 + 0.087 * (xs - 290)]),
        height=57.0,
        column=0,
        dpi=DPI,
        cross=False,
        sagitta_mm=0.0,
        slope_deg=0.0,
        bend_mm=0.0,
        resid_parabola_mm=0.0,
        glyphs=np.array([[x, 870, x + 30, 910] for x in range(290, 680, 35)], dtype=float),
    )
    fixed, log = fixed_axes([*body, tilted], DPI)
    assert len(log) == 1 and log[0].source == "glyphs"
    assert abs(slope_of(fixed[-1].points)) <= 1.0 + 1e-6
    assert fixed[-1].height < 57.0


def test_short_single_line_extends_along_body_slope() -> None:
    """Короткая строка без длинного соседа добивается до стороны по наклону корпуса, а не по своему."""
    short = Row(y=100, height=12, x0=50, x1=120, axes=(axis(50, 120, 100, 12),), weight=2.0, glyph_h=8.0)
    steep = replace(short, axes=(replace(short.axes[0], points=np.array([[50, 98.0], [120, 102.0]])),))
    guides = _guides([steep], [(50.0, 700.0)], DPI, reference_deg=0.0)
    ys = _guided_at(_smooth_axis(_axis_points(steep), DPI), np.array([700.0]), guides[0])
    assert abs(ys[0] - 102.0) < 1.0


def _side(edges: np.ndarray, pitch: float = 24.0, height: float = 14.0):
    """Правая сторона по краям строк с шагом ``pitch``: ``SideCurve`` и ординаты строк."""
    ys = 100.0 + pitch * np.arange(edges.size)
    return side_curve(ys, edges, ys - height / 2, ys + height / 2, pitch, DPI), ys


def test_trapezoid_side_is_smooth_line() -> None:
    """Трапеция: края по наклонной с шумом ±2 px — сторона без скачков, шершавость около нуля, все края внутри."""
    rng = np.random.default_rng(0)
    edges = 800.0 + 0.08 * 24.0 * np.arange(20) + rng.uniform(-2, 2, 20)
    curve, ys = _side(edges)
    assert not curve.unreliable
    grid = curve.us[:: int(round(DPI / 25.4))]  # узлы через 1 мм
    assert np.abs(np.diff(grid)).max() < 0.5 * DPI / 25.4
    assert np.all(np.interp(ys, curve.ys, curve.us) >= edges - 1e-6)


def test_hanging_hyphen_is_bump_not_step() -> None:
    """Дефис на 1 мм наружу — гладкий горб: край строки внутри, скачков больше 0.5 мм нет, участок не помечен."""
    edges = np.full(12, 800.0)
    edges[6] += 6.0
    curve, ys = _side(edges)
    assert not curve.unreliable
    assert np.interp(ys[6], curve.ys, curve.us) >= edges[6] - 1e-6
    assert np.abs(np.diff(curve.us)).max() < 0.5 * DPI / 25.4


def test_margin_note_is_step_and_unreliable() -> None:
    """Вынос на 5 мм (пометка на полях) — ступенька по краю строки и недостоверный участок."""
    edges = np.full(12, 800.0)
    edges[5] += 30.0
    curve, ys = _side(edges)
    assert len(curve.unreliable) == 1
    assert np.interp(ys[5], curve.ys, curve.us) >= edges[5] - 1e-6


def test_ragged_edge_keeps_steps() -> None:
    """Рваный край со строками разной длины (разница 10 мм) — ступеньки сохраняются."""
    edges = np.array([800, 740, 790, 700, 760, 720, 800, 690], dtype=float)
    curve, ys = _side(edges)
    assert np.allclose(np.interp(ys, curve.ys, curve.us), edges, atol=1.0)


def test_short_last_line_filled_to_side() -> None:
    """Короткая последняя строка на выровненной стороне добивается до стороны."""
    edges = np.full(10, 800.0)
    edges[-1] = 500.0
    curve, ys = _side(edges)
    assert np.interp(ys[-1], curve.ys, curve.us) >= 799.0


def test_two_indented_rows_inside_aligned_side_are_filled() -> None:
    """Конец абзаца и отступ следующего — две строки внутрь подряд между выровненными: сторона не проваливается."""
    edges = np.full(14, 800.0)
    edges[6], edges[7] = 740.0, 760.0
    curve, ys = _side(edges)
    assert np.interp(ys[6], curve.ys, curve.us) >= 799.0 and np.interp(ys[7], curve.ys, curve.us) >= 799.0
