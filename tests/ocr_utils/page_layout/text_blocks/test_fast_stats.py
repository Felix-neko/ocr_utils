"""Быстрые медиана, квантиль, ``polyfit`` и центр-линия (``fast_stats``) дают те же числа, что numpy, бит в бит."""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.text_blocks import fast_stats


def test_median_and_percentile_match_numpy_bitwise():
    """Случайные массивы: дробные, целые, округлённые до десятых; квантили на краях и внутри."""
    rng = np.random.default_rng(0)
    for index in range(3000):
        size = int(rng.integers(1, 60))
        kind = index % 3
        if kind == 0:
            values = rng.normal(20.0, 5.0, size)
        elif kind == 1:
            values = rng.integers(0, 40, size)
        else:
            values = np.round(rng.normal(20.0, 5.0, size), 1)
        q = float(rng.choice([0.0, 10.0, 25.0, 50.0, 75.0, 99.5, 100.0, rng.uniform(0.0, 100.0)]))
        assert fast_stats.median(values) == float(np.median(values))
        assert fast_stats.percentile(values, q) == float(np.percentile(values, q))


def test_median_falls_back_to_numpy():
    """Список, NaN, многомерный вход — как у numpy."""
    assert fast_stats.median([3.0, 1.0, 2.0]) == 2.0
    assert np.isnan(fast_stats.median(np.array([1.0, np.nan])))
    grid = np.arange(6.0).reshape(2, 3)
    assert fast_stats.median(grid) == float(np.median(grid))


def test_polyfit_matches_numpy_bitwise():
    """Прямая и парабола на абсциссах страницы (тысячи px) — те же коэффициенты, что у ``np.polyfit``."""
    rng = np.random.default_rng(1)
    for index in range(1000):
        degree = 1 + index % 2
        size = int(rng.integers(degree + 1, 50))
        xs = rng.uniform(-3000.0, 3000.0, size)
        ys = 0.01 * xs + 1e-5 * xs * xs + rng.normal(500.0, 3.0, size)
        assert np.array_equal(fast_stats.polyfit(xs, ys, degree), np.polyfit(xs, ys, degree))


def test_refined_columns_centre_of_band():
    """Столбец с краской в полосе — центр масс в полосе; столбец без краски в полосе — первый проход."""
    mask = np.zeros((20, 2), dtype=bool)
    mask[4:7, 0] = True  # в полосе вокруг 5
    mask[15, 0] = True  # за полосой — не считается
    mask[15, 1] = True  # у второго столбца краска только за полосой
    ys, weights = fast_stats.refined_columns(
        mask, np.array([0, 1]), np.array([5.0, 5.0]), 3.0, np.array([7.0, 9.0]), np.array([4.0, 1.0])
    )
    assert ys.tolist() == [5.0, 9.0]
    assert weights.tolist() == [3.0, 1.0]
