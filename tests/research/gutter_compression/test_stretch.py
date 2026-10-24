"""Тесты растяжения у корешка, модели сжатия и мер окна на синтетике."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ocr_utils.scan_cropping.v2.rectify.coons import Layout, PageRect
from research.gutter_compression.measure import Method, acf_pitch, stem_pitch
from research.gutter_compression.profile import CompressionModel, fit_model
from research.gutter_compression.stretch import (
    StretchLimits,
    capped,
    compress_image,
    output_of_source,
    plan_side,
    plan_widen,
    side_displacement,
    text_scale,
    widen_displacement,
    widen_output_of_source,
    Y_NODES,
)

PPM = 10.0  # px/мм синтетики


def bars(period: float, width: int, duty: float = 0.3) -> np.ndarray:
    """Профиль вертикальных штрихов с шагом ``period`` px (1 — краска)."""
    x = np.arange(width)
    return ((x % period) < duty * period).astype(np.float64)


def test_acf_pitch_finds_period() -> None:
    assert abs(acf_pitch(bars(12.0, 120), 3, 30) - 12.0) < 0.6


def test_stem_pitch_finds_period() -> None:
    assert abs(stem_pitch(bars(10.0, 100, 0.4), 0.7, 20) - 10.0) < 0.5


def test_output_of_source_keeps_outer_edge_and_is_monotone() -> None:
    model = CompressionModel(amplitude=0.25, length_mm=6.0, edge_mm=8.0)
    plan = plan_side(model, width_mm=100.0, edge_mm=8.0, limits=StretchLimits())
    u = np.arange(0.0, 100.0 + 0.05, 0.05)
    t = output_of_source(plan, u, 0.5)
    assert abs(t[-1] - 100.0) < 1e-6
    assert np.all(np.diff(t) > 0)
    # Край текста у корешка не ближе запаса к корешку.
    t_edge = np.interp(8.0, u, t)
    assert t_edge >= StretchLimits().edge_safety_mm - 1e-6
    # Местное растяжение у края — ≈ 1/c.
    k = np.searchsorted(u, 10.0)
    assert abs((t[k + 1] - t[k]) / (u[k + 1] - u[k]) - 1.0 / float(model.scale(10.025))) < 2e-3


def test_body_squeeze_when_margin_is_short() -> None:
    model = CompressionModel(amplitude=0.25, length_mm=10.0)
    plan = plan_side(model, width_mm=100.0, edge_mm=1.5, limits=StretchLimits())
    assert plan.excess_mm > 0.5 and plan.body_squeeze > 0
    u = np.arange(0.0, 100.0 + 0.05, 0.05)
    t = output_of_source(plan, u, 0.5)
    assert np.interp(1.5, u, t) >= StretchLimits().edge_safety_mm - 1e-6
    assert np.all(np.diff(t) > 0)


def test_capped_limits_stretch() -> None:
    model = capped(CompressionModel(amplitude=0.45, length_mm=5.0), StretchLimits(max_stretch=1.25))
    assert 1.0 / float(model.scale(0.0)) <= 1.25 + 1e-9


def test_side_displacement_signs() -> None:
    layout = Layout(400, 50, (PageRect(0, 0, 200, 50), PageRect(200, 0, 400, 50)), 200.0)
    plan = plan_side(CompressionModel(amplitude=0.3, length_mm=5.0, edge_mm=5.0), 20.0, 5.0, StretchLimits())
    left = side_displacement(plan, layout, True, PPM)
    right = side_displacement(plan, layout, False, PPM)
    # Источник дальше от корешка, чем точка итога: у левой страницы dx < 0, у правой > 0; внешний край — 0.
    assert left[25, 190] < 0 and right[25, 210] > 0
    assert abs(left[25, 0]) < 0.05 and abs(right[25, 399]) < 0.05


def test_compress_then_measure_pitch() -> None:
    period = 12.0
    row = (1.0 - bars(period, 600)) * 255
    image = np.repeat(row[None, :], 40, axis=0).astype(np.uint8)
    model = CompressionModel(amplitude=0.3, length_mm=5.0)
    squeezed = compress_image(image, 300.0, (model, model), PPM)
    near = acf_pitch(1.0 - squeezed[20, 301:341] / 255.0, 3, 30)  # окно 0–4 мм от корешка
    expected = period * float(np.mean(model.scale(np.linspace(0.0, 4.0, 9))))
    assert abs(near - expected) < 1.2


def test_fit_model_recovers_parameters() -> None:
    rng = np.random.default_rng(1)
    truth = CompressionModel(amplitude=0.2, length_mm=6.0, edge_mm=2.0)
    rows = []
    for line in range(20):
        d = np.arange(3.0, 40.0, 1.0)
        r = truth.scale(d) + rng.normal(0, 0.05, len(d))
        rows.append(pd.DataFrame({"d_mm": d, "from_edge_mm": d - 2.0, "y": 100.0 * line, "line": line,
                                  "acf_ratio": r, "edge_mm": 2.0}))  # fmt: skip
    model = fit_model(pd.concat(rows), Method.ACF, 2000.0, 0.0)
    assert abs(model.edge_mm - 2.0) < 1e-9
    grid = np.linspace(2.0, 20.0, 50)
    assert np.max(np.abs(model.scale(grid) - truth.scale(grid))) < 0.03


def test_small_excess_is_not_applied() -> None:
    plan = plan_side(CompressionModel(amplitude=0.3, length_mm=1.5, edge_mm=3.0), 100.0, 3.0, StretchLimits())
    assert not plan.applied


def test_displacement_without_tilt_keeps_verticals() -> None:
    layout = Layout(400, 60, (PageRect(0, 0, 200, 60), PageRect(200, 0, 400, 60)), 200.0)
    plan = plan_side(CompressionModel(amplitude=0.3, length_mm=5.0, edge_mm=3.0), 20.0, 3.0, StretchLimits())
    dx = side_displacement(plan, layout, False, PPM)
    # Смещение не зависит от высоты: вертикальная сторона блока остаётся вертикальной.
    assert np.allclose(dx, dx[:1], atol=1e-4)


def test_tilted_model_keeps_text_edge_straight() -> None:
    model = CompressionModel(amplitude=0.25, length_mm=6.0, tilt=-0.2, edge_mm=4.0)
    plan = plan_side(model, 100.0, 4.0, StretchLimits(max_body_squeeze=0.05), outer_mm=85.0)
    u = np.arange(0.0, 100.0 + 0.05, 0.05)
    edges = [np.interp(4.0, u, output_of_source(plan, u, y)) for y in (0.0, 0.5, 1.0)]
    outers = [np.interp(85.0, u, output_of_source(plan, u, y)) for y in (0.0, 0.5, 1.0)]
    # Оба края текста на одном месте при любой высоте страницы (стороны блока прямые), край страницы на месте.
    margins = [np.interp(2.0, u, output_of_source(plan, u, y)) for y in (0.0, 0.5, 1.0)]
    assert np.ptp(edges) < 1e-6
    # Поле у корешка сдвигается целиком, одинаково на любой высоте (висячие дефисы не гнут сторону блока).
    assert np.ptp(margins) < 1e-6
    assert max(abs(o - 85.0) for o in outers) < 1e-6
    assert all(abs(output_of_source(plan, u, y)[-1] - 100.0) < 1e-6 for y in (0.0, 1.0))


def test_widen_map_stretches_text_only_and_keeps_sides_straight() -> None:
    model = CompressionModel(amplitude=0.4, length_mm=6.0, tilt=-0.4)
    plan = plan_widen(model, width_mm=100.0, edge_mm=3.0, limits=StretchLimits(max_stretch=2.0), outer_mm=90.0,
                      text_starts_mm=(2.5,) * Y_NODES)  # fmt: skip
    assert plan.applied and plan.text_start(0.3) == 2.5
    # Вверху сжатие сильнее — прибавка больше; расширение — медиана прибавки по высоте.
    assert plan.excesses_mm[0] > plan.excesses_mm[-1]
    assert abs(plan.widen_mm - float(np.median(plan.excesses_mm))) < 1e-9
    k_lo, k_hi = plan.scale_range()
    assert k_lo < 1.0 < k_hi
    u = np.arange(0.0, 100.0 + 0.01, 0.01)
    for y in (0.0, 0.5, 1.0):
        t = widen_output_of_source(plan, u, y)
        assert np.all(np.diff(t) > 0)
        # Поле у корешка не трогается, начало зоны текста на месте на любой высоте.
        assert np.allclose(t[u < 2.5], u[u < 2.5])
        assert abs(np.interp(2.5, u, t) - 2.5) < 1e-6
        # Внешняя сторона блока и край страницы сдвинуты на расширение на любой высоте.
        assert abs(np.interp(90.0, u, t) - (90.0 + plan.widen_mm)) < 1e-6
        assert abs(np.interp(95.0, u, t) - (95.0 + plan.widen_mm)) < 1e-6
        # В тексте местное растяжение — k(y)/c модели.
        k = np.searchsorted(u, 8.0)
        slope = (t[k + 1] - t[k]) / (u[k + 1] - u[k])
        c = float(text_scale(plan.model, np.array([8.005]), y, plan.amp_cap, plan.text_start(y))[0])
        assert abs(slope - plan.line_scale(y) / c) < 2e-3


def test_widen_inner_side_follows_line_edges() -> None:
    # Край строк меняется по высоте (вверху 3.4 мм, внизу 2.8 мм, медиана 3.1): начало зоны текста на месте на любой
    # высоте, у самого края строки растяжение — модельное 1/c у медианного края (а не растяжение поля).
    model = CompressionModel(amplitude=0.2, length_mm=6.0)
    starts = tuple(np.linspace(3.4, 2.8, Y_NODES))
    plan = plan_widen(model, 100.0, 3.1, StretchLimits(max_stretch=2.0), 90.0, text_starts_mm=starts)
    u = np.arange(0.0, 100.0 + 0.01, 0.01)
    for y in (0.0, 1.0):
        t = widen_output_of_source(plan, u, y)
        e0 = plan.text_start(y)
        assert abs(np.interp(e0, u, t) - e0) < 1e-6
        k = np.searchsorted(u, e0 + 0.02)
        slope = (t[k + 1] - t[k]) / (u[k + 1] - u[k])
        c = 1.0 - 0.2 * np.exp(-max(e0 + 0.025 - 3.1, 0.0) / 6.0)
        assert abs(slope - plan.line_scale(y) / c) < 3e-3


def test_widen_not_applied_on_small_excess() -> None:
    plan = plan_widen(CompressionModel(amplitude=0.05, length_mm=2.0), 100.0, 3.0, StretchLimits(), 90.0)
    assert not plan.applied and plan.widen_mm == 0.0
    u = np.linspace(0.0, 100.0, 11)
    assert np.allclose(widen_output_of_source(plan, u, 0.5), u)


def test_widen_displacement_layout() -> None:
    layout = Layout(400, 300, (PageRect(20.0, 10.0, 200.0, 290.0), PageRect(200.0, 12.0, 380.0, 288.0)), 200.0)
    limits = StretchLimits(max_stretch=2.0)
    left = plan_widen(CompressionModel(amplitude=0.4, length_mm=5.0), 18.0, 1.0, limits, 16.0)
    right = plan_widen(CompressionModel(amplitude=0.3, length_mm=5.0), 18.0, 1.0, limits, 16.0)
    wide, dx = widen_displacement((left, right), layout, PPM)
    wl, wr = left.widen_mm * PPM, right.widen_mm * PPM
    assert abs(wide.spine_x - (200.0 + wl)) < 1e-9
    assert wide.width == int(np.ceil(400 + wl + wr))
    assert dx.shape == (300, wide.width)
    # Внешний край левой на месте, правой — сдвинут на W_L + W_R; линия корешка берёт линию корешка.
    assert abs(dx[150, 20]) < 1e-3
    col = int(380 + wl + wr)
    assert abs(col + dx[150, col] - (col - wl - wr)) < 0.05
    assert abs(int(np.ceil(wide.spine_x)) + dx[150, int(np.ceil(wide.spine_x))] - 200.0) < 0.5


def test_widen_weak_long_tail_is_not_applied() -> None:
    # Слабая амплитуда на длинном спаде: по всей строке лишней длины много, у края — мало.
    plan = plan_widen(CompressionModel(amplitude=0.07, length_mm=30.0), 100.0, 0.2, StretchLimits(), 90.0)
    assert max(plan.excesses_mm) > 0.8
    assert not plan.applied


def test_widen_decay_at_fit_bound_is_not_applied() -> None:
    # Спад у верхнего предела подгонки (30 мм) — не сжатие у корешка, даже если у края лишней длины хватает.
    plan = plan_widen(CompressionModel(amplitude=0.3, length_mm=30.0), 100.0, 5.0, StretchLimits(), 90.0)
    assert not plan.applied
