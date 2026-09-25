"""Тесты пересборки оси строки по сетке (``lines.resample``)."""

from __future__ import annotations

import numpy as np
import pytest

from ocr_utils.page_layout.text_blocks.lines import resample


@pytest.mark.parametrize("x_end", [497.0, 499.0, 500.5, 502.9])
def test_resample_ends_stay_on_data(x_end: float):
    """Концы пересобранной оси — ровно на концах данных при любом положении конца внутри окна сетки.

    Раньше крайняя точка вставала в центр окна и уходила за конец строки до полушага
    (1973/07 с.88: краска до 497, ось до 500 — выступ боковой кромки блока).
    """
    # Точки через полпикселя и ровно в конце: конец попадает в разные доли окна сетки 6 px.
    xs = np.append(np.arange(130.0, x_end, 0.5), x_end)
    points = np.column_stack([xs, np.full(xs.shape, 250.0)])
    out = resample(points, 6.0)
    assert out[0, 0] == pytest.approx(130.0)
    assert out[-1, 0] == pytest.approx(x_end)
    # Внутренние точки идут по возрастанию и не выходят за концы.
    assert np.all(np.diff(out[:, 0]) > 0)


def test_resample_sparse_polyline_does_not_overshoot():
    """Редкая ломаная (интерполяционная ветка) тоже не выходит за последнюю точку."""
    points = np.array([[100.0, 10.0], [150.0, 12.0], [203.0, 11.0]])
    out = resample(points, 6.0)
    assert out[0, 0] == pytest.approx(100.0)
    assert out[-1, 0] == pytest.approx(203.0)
    assert np.all(np.diff(out[:, 0]) > 0)
