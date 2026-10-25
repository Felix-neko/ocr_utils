"""Тесты растяжения у корешка (``scan_cropping.v2.gutter_decompression``): подгонка по буквам, карта расширения, синтетический разворот целиком."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from ocr_utils.scan_cropping.v2.gutter_decompression.apply import page_plans
from ocr_utils.scan_cropping.v2.gutter_decompression.fit import LetterData, letter_prediction, residuals
from ocr_utils.scan_cropping.v2.gutter_decompression.model import CompressionModel, HeightShape, amplitude_of
from ocr_utils.scan_cropping.v2.gutter_decompression.widen import (
    Y_NODES,
    WidenLimits,
    plan_widen,
    text_scale,
    text_zone,
    widen_displacement,
    widen_output_of_source,
)
from ocr_utils.scan_cropping.v2.rectify.coons import Layout, PageRect

PPM = 17.0  # px/мм синтетики (как ХиП 1991)


def const_model(amplitude: float, length_mm: float, origin_mm: float) -> CompressionModel:
    """Модель с постоянной по высоте амплитудой (линия без наклона)."""
    return CompressionModel(HeightShape.LINEAR, (amplitude, 0.0), length_mm, origin_mm)


def test_spline_and_line_amplitudes() -> None:
    """Сплайн с равными коэффициентами — константа; линия Лежандра — B₀ ± B₁ на краях высоты."""
    y = np.linspace(0.0, 1.0, 5)
    assert np.allclose(amplitude_of(HeightShape.SPLINE, np.full(9, 0.3), y), 0.3)
    assert np.allclose(amplitude_of(HeightShape.LINEAR, np.array([0.3, 0.1]), np.array([0.0, 1.0])), [0.2, 0.4])


def test_letter_prediction_is_harmonic_mean_over_letter() -> None:
    """Отношение буквы у края — между сжатием на её ближнем и дальнем концах."""
    data = LetterData(np.array([2.5]), np.array([0.5]), np.array([0.5]), np.array([1.0]))
    r = letter_prediction(1.5, np.array([0.4]), data, 2.0)[0]
    assert 1 - 0.4 < r < 1 - 0.4 * np.exp(-1.0 / 1.5)


def test_residuals_zero_on_model_letters() -> None:
    """Буквы, построенные по модели, дают нулевые остатки на её параметрах."""
    centre = np.r_[2.6:30:1.2]
    data = LetterData(centre, np.full(len(centre), 0.5), np.full(len(centre), 0.5), np.ones(len(centre)))
    r = letter_prediction(1.5, np.full(len(centre), 0.4), data, 2.0)
    data = LetterData(data.centre, data.half, data.y, r)
    assert np.allclose(residuals(np.array([1.5, 0.4, 0.0]), HeightShape.LINEAR, data, 2.0, 0.0), 0.0)


def test_widen_map_stretches_text_only_and_keeps_sides_straight() -> None:
    """Поле у корешка тождественно, обе стороны блока прямые, в тексте растяжение k(y)/c."""
    model = CompressionModel(HeightShape.LINEAR, (0.4, -0.2), 6.0, 2.5)
    plan = plan_widen(model, 100.0, 90.0, WidenLimits(), (2.5,) * Y_NODES, (0.0, 1.0))
    assert plan.applied
    assert abs(plan.widen_mm - float(np.median(plan.excesses_mm))) < 1e-9
    u = np.arange(0.0, 100.0 + 0.01, 0.01)
    for y in (0.0, 0.5, 1.0):
        t = widen_output_of_source(plan, u, y)
        assert np.all(np.diff(t) > 0)
        assert np.allclose(t[u < 2.5], u[u < 2.5])
        assert abs(np.interp(90.0, u, t) - (90.0 + plan.widen_mm)) < 1e-6
        k = np.searchsorted(u, 8.0)
        slope = (t[k + 1] - t[k]) / (u[k + 1] - u[k])
        c = float(text_scale(plan.model, np.array([8.005]), y, plan.amp_cap, 2.5)[0])
        assert abs(slope - plan.line_scale(y) / c) < 2e-3


def test_short_strong_decay_passes_threshold_and_weak_does_not() -> None:
    """Порог 0.5 мм: короткий сильный спад (0.35 на 1.5 мм, 0.65 мм) применяется, слабый (0.05 на 2 мм) — нет."""
    strong = plan_widen(const_model(0.35, 1.5, 2.0), 100.0, 90.0, WidenLimits(), (), (0.0, 1.0))
    weak = plan_widen(const_model(0.05, 2.0, 2.0), 100.0, 90.0, WidenLimits(), (), (0.0, 1.0))
    assert strong.applied and 0.5 < strong.near_mm < 0.8
    assert not weak.applied and weak.widen_mm == 0.0


def test_decay_at_fit_bound_is_not_applied() -> None:
    """Спад у верхнего предела подгонки (30 мм) — не сжатие у корешка."""
    assert not plan_widen(const_model(0.3, 30.0, 5.0), 100.0, 90.0, WidenLimits(), (), (0.0, 1.0)).applied


def test_null_model_is_not_applied() -> None:
    """Без модели — тождество."""
    plan = plan_widen(CompressionModel(), 100.0, 90.0, WidenLimits(), (), (0.0, 1.0))
    assert not plan.applied
    u = np.linspace(0.0, 100.0, 11)
    assert np.allclose(widen_output_of_source(plan, u, 0.5), u)


def test_text_zone_skips_stray_line_near_spine() -> None:
    """Огибающая краёв строк не идёт к одиночной «строке» у самой складки (сор)."""
    rows = [{"line": i, "y": 100.0 + 40 * i, "edge_mm": 3.0 + 0.4 * (i % 2), "outer_mm": 95.0} for i in range(20)]
    rows[10]["edge_mm"] = 0.2
    starts, y_range, outer = text_zone(pd.DataFrame(rows), PageRect(0.0, 50.0, 1000.0, 1000.0))
    assert len(starts) == Y_NODES and min(starts) >= 3.0 - 1e-9
    assert outer == 95.0 and 0.0 < y_range[0] < y_range[1] < 1.0


def test_widen_displacement_layout() -> None:
    """Корешок сдвинут на W_L, холст шире на W_L + W_R, внешний край левой на месте."""
    layout = Layout(400, 300, (PageRect(20.0, 10.0, 200.0, 290.0), PageRect(200.0, 12.0, 380.0, 288.0)), 200.0)
    left = plan_widen(const_model(0.4, 3.0, 1.0), 180.0 / PPM, 160.0 / PPM, WidenLimits(), (), (0.0, 1.0))
    right = plan_widen(const_model(0.3, 3.0, 1.0), 180.0 / PPM, 160.0 / PPM, WidenLimits(), (), (0.0, 1.0))
    wide, dx = widen_displacement((left, right), layout, PPM)
    wl, wr = left.widen_mm * PPM, right.widen_mm * PPM
    assert abs(wide.spine_x - (200.0 + wl)) < 1e-9
    assert wide.width == int(np.ceil(400 + wl + wr)) and dx.shape == (300, wide.width)
    assert abs(dx[150, 20]) < 1e-3


def synthetic_spread(compress_right: bool) -> tuple[np.ndarray, Layout]:
    """Разворот из строк cv2.putText; у правой страницы буквы у корешка сжаты по модели 0.4 на 3 мм от края текста."""
    h, page_w = int(150 * PPM), int(110 * PPM)
    image = np.full((h, 2 * page_w, 3), 245, np.uint8)
    spine = float(page_w)
    # Случайные буквы в каждой строке: у края строки — разные буквы, как в настоящем тексте (иначе ширина «краевых»
    # букв зависит от того, какие это буквы, а не от сжатия).
    rng = np.random.default_rng(0)
    alphabet = np.array(list("abcdeghknopqsuvxyz"))
    for row in range(40):
        y = int((15 + 3.2 * row) * PPM)
        for page in (0, 1):
            x0 = int(8 * PPM) if page == 0 else int(spine + 3 * PPM)
            text = "".join(rng.choice(alphabet, 58))
            cv2.putText(image, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2, cv2.LINE_AA)
    layout = Layout(2 * page_w, h, (PageRect(0.0, 0.0, spine, float(h)), PageRect(spine, 0.0, 2.0 * page_w, float(h))),
                    spine)  # fmt: skip
    if not compress_right:
        return image, layout
    # Сжатие: пиксель на расстоянии u от корешка берётся из исходника на расстоянии e + ∫ₑᵘ 1/c (у края — уже).
    cols = np.arange(image.shape[1], dtype=np.float64)
    u = (cols - spine) / PPM
    grid = np.arange(0.0, 120.0, 0.01)
    edge = 3.0
    inv = np.where(grid < edge, 1.0, 1.0 / (1.0 - 0.4 * np.exp(-(grid - edge) / 3.0)))
    t = np.concatenate([[0.0], np.cumsum((inv[1:] + inv[:-1]) / 2 * np.diff(grid))])
    src = np.where(cols >= spine, spine + np.interp(u, grid, t) * PPM, cols).astype(np.float32)
    map_x = np.repeat(src[None, :], h, axis=0)
    map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], image.shape[1], axis=1)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderValue=(245, 245, 245)), layout


def test_page_plans_on_synthetic_spread() -> None:
    """Сжатая у корешка правая страница растягивается, несжатая левая — нет; без сжатия не растягивается ничего."""
    image, layout = synthetic_spread(True)
    left, right = page_plans(image, layout, PPM, (True, True))
    assert right.applied and right.widen_mm > 0.8
    assert float(right.model.amplitude_at(0.5)) > 0.25
    assert not left.applied
    clean, layout = synthetic_spread(False)
    assert not any(p.applied for p in page_plans(clean, layout, PPM, (True, True)))
    # Выключенная страница не растягивается.
    assert not page_plans(image, layout, PPM, (True, False))[1].applied


def test_trim_stray_drops_fold_fragment_near_spine() -> None:
    """Узкое пятно у корешка за большим промежутком отбрасывается, однобуквенное слово и знак у текста — нет."""
    from ocr_utils.scan_cropping.v2.gutter_decompression.letters import trim_stray

    xh = 20.0
    text = np.arange(100, 400)
    fold = np.arange(2, 5)
    # Правая страница: корешок слева; обрывок складки в 95 px от текста.
    assert trim_stray(np.r_[fold, text], xh, near_right=False)[0] == 100
    # Левая страница: корешок справа.
    assert trim_stray(np.r_[text, fold + 495], xh, near_right=True)[-1] == 399
    # Однобуквенное слово (шире половины строчной) за обычным пробелом остаётся.
    word = np.arange(80, 92)
    assert trim_stray(np.r_[word, text], xh, near_right=False)[0] == 80
    # Узкий знак вплотную к тексту (промежуток меньше 1.5 строчной) остаётся.
    assert trim_stray(np.r_[np.arange(85, 88), text], xh, near_right=False)[0] == 85


def test_page_plans_ignores_fold_line_at_spine() -> None:
    """Линия складки у корешка несжатого разворота не даёт растяжения."""
    clean, layout = synthetic_spread(False)
    spine = int(layout.spine_x)
    # Тёмная линия складки в 0.2 мм от корешка на правой странице, порванная на обрывки высотой со строчную в поясе
    # каждой строки (как на ХиП 1991/01 IMG_0987 R).
    x = spine + int(0.2 * PPM)
    for row in range(40):
        base = int((15 + 3.2 * row) * PPM)
        clean[base - 15 : base + 1, x : x + 2] = 40
    assert not any(p.applied for p in page_plans(clean, layout, PPM, (True, True)))
