"""Аппроксимации центр-линии: парабола восстанавливается, прямая даёт нулевой остаток."""

from __future__ import annotations

import numpy as np
import pytest

from ocr_utils.scan_markup.curved_lines import fitting


def test_straight_line_has_zero_residual_and_correct_slope():
    xs = np.arange(0, 400, dtype=np.float64)
    ys = 100.0 + np.tan(np.radians(1.5)) * xs
    fit = fitting.fit_line(xs, ys)
    assert fit is not None
    assert fit.slope_deg == pytest.approx(1.5, abs=1e-6)
    assert fit.resid_lin == pytest.approx(0.0, abs=1e-9)
    assert fit.sagitta == pytest.approx(0.0, abs=1e-9)
    assert fit.length == 399.0


def test_parabola_recovers_curvature_and_sagitta():
    xs = np.arange(0, 601, dtype=np.float64)
    curvature = 4e-5  # прогиб на длине 600: 4e-5 * 300² = 3.6 px
    ys = 50.0 + curvature * (xs - 300.0) ** 2
    fit = fitting.fit_line(xs, ys)
    assert fit is not None
    assert fit.curvature == pytest.approx(curvature, rel=1e-6)
    assert fit.sagitta == pytest.approx(3.6, rel=1e-6)
    assert fit.resid_quad == pytest.approx(0.0, abs=1e-6)
    assert fit.resid_lin > 1.0


def test_too_few_points_is_none():
    xs = np.arange(5, dtype=np.float64)
    assert fitting.fit_line(xs, xs) is None


def test_centreline_is_ink_centre_of_mass_per_column():
    ink = np.zeros((10, 6), np.uint8)
    ink[2:5, 1] = 255  # центр 3.0
    ink[6, 3] = 255  # центр 6.0
    xs, ys, weights = fitting.centreline(ink)
    assert xs.tolist() == [1.0, 3.0]
    assert ys.tolist() == [3.0, 6.0]
    assert weights.tolist() == [3.0, 1.0]


def test_page_stats_normalises_by_height_and_uses_long_lines_only():
    xs = np.arange(0, 601, dtype=np.float64)
    curved = fitting.fit_line(xs, 10 + 4e-5 * (xs - 300) ** 2)
    straight = fitting.fit_line(xs, 10 + 0.0 * xs)
    short = fitting.fit_line(xs[:40], 10 + 1e-3 * (xs[:40] - 20) ** 2)  # короткая, но очень кривая
    stats = fitting.page_stats([curved, straight, short], [10.0, 10.0, 10.0])
    assert stats["lines"] == 3.0
    assert stats["lines_long"] == 2.0
    # Короткая строка в сводку не попала: p90 по двум длинным, а не по трём.
    assert stats["sagitta_rel_p90"] <= 0.36 + 1e-9
    assert stats["sagitta_rel_max3"] == pytest.approx(0.18, rel=1e-3)


def test_plane_fit_recovers_gradients():
    rng = np.random.default_rng(0)
    xs, ys = rng.random(50), rng.random(50)
    values = 1.0 + 2.0 * xs - 0.5 * ys
    a, bx, by, rms = fitting.plane_fit(xs, ys, values)
    assert (a, bx, by) == pytest.approx((1.0, 2.0, -0.5), abs=1e-9)
    assert rms == pytest.approx(0.0, abs=1e-9)


def test_clusters_split_on_gaps_and_drop_small_ones():
    values = np.array([10, 11, 12, 50, 51, 90])
    clusters = fitting.clusters_1d(values, tol=5, min_size=2)
    assert [sorted(values[c].tolist()) for c in clusters] == [[10, 11, 12], [50, 51]]


def test_edge_fit_sagitta():
    ys = np.linspace(0, 200, 21)
    xs = 100 + 1e-3 * (ys - 100) ** 2  # прогиб 1e-3 * 100² = 10 px
    fit = fitting.edge_fit(xs, ys)
    assert fit is not None
    assert fit[0] == pytest.approx(10.0, rel=1e-6)
