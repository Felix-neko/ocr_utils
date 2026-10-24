"""Тесты подгонки формы амплитуды сжатия по высоте страницы."""

from __future__ import annotations

import numpy as np

from research.gutter_compression.shape import HeightShape, SideData, cv_errors, fit_shape, spline_basis


def synthetic_side(amp_top: float, amp_bottom: float, length_mm: float, seed: int = 0) -> SideData:
    """Окна стороны с линейной по высоте амплитудой и шумом окна 0.03."""
    rng = np.random.default_rng(seed)
    lines = np.repeat(np.arange(40), 25)
    y = (lines + 0.5) / 40
    x = np.tile(np.arange(25, dtype=np.float64), 40)
    amp = amp_top + (amp_bottom - amp_top) * y
    r = 1.0 - amp * np.exp(-x / length_mm) + rng.normal(0, 0.03, len(x))
    return SideData(x, y, r, lines, x)


def test_linear_fit_recovers_amplitude_and_length() -> None:
    data = synthetic_side(0.5, 0.1, 6.0)
    fit = fit_shape(data.x, data.y, data.r, HeightShape.LINEAR)
    assert abs(fit.length_mm - 6.0) < 0.6
    assert np.allclose(fit.amplitude(np.array([0.0, 1.0])), [0.5, 0.1], atol=0.04)


def test_cv_prefers_linear_over_const_on_tilted_side() -> None:
    data = synthetic_side(0.5, 0.1, 6.0, seed=1)
    const, _ = cv_errors(data, HeightShape.CONST)
    linear, parts = cv_errors(data, HeightShape.LINEAR)
    assert linear < const
    assert np.isfinite(parts).all()


def test_spline_basis_is_partition_of_unity() -> None:
    basis = spline_basis(np.linspace(0.0, 1.0, 11))
    assert np.allclose(basis.sum(axis=1), 1.0)


def test_model_amplitude_from_spline() -> None:
    from research.gutter_compression.profile import SPLINE_KNOTS, CompressionModel

    # Постоянные коэффициенты B-сплайна дают постоянную амплитуду; отрицательная обрезается нулём.
    flat = CompressionModel(amplitude=0.3, spline_coef=(0.3,) * (SPLINE_KNOTS + 2))
    assert np.allclose(flat.amplitude_at(np.linspace(0.0, 1.0, 5)), 0.3)
    assert np.allclose(CompressionModel(spline_coef=(-0.1,) * (SPLINE_KNOTS + 2)).amplitude_at(0.5), 0.0)
