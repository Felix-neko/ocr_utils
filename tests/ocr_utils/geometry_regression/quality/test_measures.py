"""Меры качества строк и краёв v17 на синтетических разборах."""

from __future__ import annotations

import numpy as np
import pytest

from ocr_utils.page_layout.text_blocks.store import AxisRecord, BlockRecord, PageGeometry, PointKind, SideCode, SideRecord
from ocr_utils.geometry_regression.quality.edges import edge_metrics, match_edges
from ocr_utils.geometry_regression.quality.lines import clean_points, line_metrics, match_lines, spread_mm

DPI = 150.0
MM = DPI / 25.4


def _axis(y0: float, tilt_px: float, x0=100.0, x1=900.0, height=20.0, block=0, jumps=()) -> AxisRecord:
    xs = np.linspace(x0, x1, 200)
    ys = y0 + tilt_px * (xs - x0) / (x1 - x0)
    return AxisRecord(np.column_stack([xs, ys]), np.zeros((0, 2)), height, 0, False, block, 0, tuple(jumps))


def _page(axes=(), blocks=()) -> PageGeometry:
    return PageGeometry("синтетика", DPI, (5000, 7000), 600.0, tuple(axes), tuple(blocks), (), (), ())


def test_jump_spans_are_dropped_from_the_axis():
    """Точки в участке перескока не входят в меру: ступенька на соседнюю строку не выдаёт себя за наклон."""
    axis = _axis(500.0, 0.0)
    points = axis.points.copy()
    points[(points[:, 0] > 400) & (points[:, 0] < 500), 1] += 60.0  # ось уехала на соседнюю строку
    jumped = AxisRecord(points, axis.body_points, axis.height, 0, False, 0, 0, ((395.0, 505.0),))
    assert spread_mm(clean_points(jumped), DPI) == pytest.approx(0.0, abs=1e-9)


def test_tilted_line_in_a_is_damage():
    """Строка, наклонённая в A на 1.5 мм, — порча 1.5 мм; прямая в обоих — ноль."""
    b = _page([_axis(500.0, 0.0), _axis(800.0, 0.0)])
    a = _page([_axis(500.0, 1.5 * MM), _axis(800.0, 0.0)])
    metrics, culprits = line_metrics(match_lines(b, a, None, []))
    assert metrics["line_pairs"] == 2
    assert metrics["line_quality_mm"] == pytest.approx(1.5, abs=0.05)
    assert "line_quality_mm" in culprits


def test_straightened_line_is_gain():
    """Строка, выпрямленная в A, — выигрыш, а не порча."""
    b = _page([_axis(500.0, 2.0 * MM)])
    a = _page([_axis(500.0, 0.0)])
    metrics, _ = line_metrics(match_lines(b, a, None, []))
    assert metrics["line_quality_mm"] == 0.0
    assert metrics["line_quality_gain_mm"] == pytest.approx(2.0, abs=0.05)


def _block(x_top: float, x_bottom: float, rows: int, aligned=True, kinds=None) -> BlockRecord:
    ys = np.linspace(200.0, 200.0 + rows * 25.0, 60)
    xs = x_top + (x_bottom - x_top) * (ys - ys[0]) / (ys[-1] - ys[0])
    line = np.column_stack([xs, ys])
    kinds = np.zeros(len(ys), np.int8) if kinds is None else kinds
    side = SideRecord(0, SideCode.LEFT, line, line, kinds)
    return BlockRecord(0, np.zeros((4, 2)), rows, "left", aligned, False, (), (), {SideCode.LEFT: side})


def test_slanted_aligned_side_is_damage_weighted_by_rows():
    """Сторона, ушедшая в A на 2 мм, — порча 2 мм у большого блока и вдвое меньше у блока в 4 ряда."""
    big = edge_metrics(match_edges(_page(blocks=[_block(100, 100, 16)]), _page(blocks=[_block(100, 100 + 2 * MM, 16)]), None))[0]
    small = edge_metrics(match_edges(_page(blocks=[_block(100, 100, 4)]), _page(blocks=[_block(100, 100 + 2 * MM, 4)]), None))[0]
    assert big["edge_quality_mm"] == pytest.approx(2.0, abs=0.05)
    assert small["edge_quality_mm"] == pytest.approx(1.0, abs=0.05)


def test_ragged_side_is_not_measured():
    """Невыровненная сторона не меряется."""
    b = _page(blocks=[_block(100, 100, 16, aligned=False)])
    a = _page(blocks=[_block(100, 100 + 3 * MM, 16, aligned=False)])
    assert edge_metrics(match_edges(b, a, None))[0]["edge_pairs"] == 0
