"""Форма зависимости амплитуды сжатия от высоты страницы ``B(y)``: семейства плавных функций (константа, линия, полиномы Лежандра, P-сплайн, логистическая ступень), робастная подгонка с общей длиной спада стороны, перекрёстная проверка с отложенными группами строк, амплитуда по поясам высоты без допущения о форме и листы кривых."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import partial

import cv2
import numpy as np
import pandas as pd
from numpy.polynomial import legendre
from scipy.optimize import least_squares

from research.gutter_compression.measure import Method
from research.gutter_compression.profile import FIT_SCALE, LENGTH_MM, SPLINE_KNOTS, ratio_column, spline_basis

# Пределы амплитуды в любой точке высоты при подгонке: снизу — небольшое «расширение» (у поясов без сжатия текст в
# 4–20 мм от корешка шире опоры на 2–4 %), сверху — как у модели стороны с запасом.
AMPLITUDE_RANGE = (-0.3, 0.75)
# Сетка штрафа P-сплайна за вторую разность коэффициентов (подбирается по перекрёстной проверке на всём паке).
SPLINE_LAMBDAS = (0.3, 3.0, 30.0, 300.0)
# Окон в подгонке не меньше (иначе сторона пропускается) и строк в группе для перекрёстной проверки.
MIN_WINDOWS = 120
LINES_PER_GROUP = 4
FOLDS = 5
# Ошибка на отложенных окнах считается ближе этого к краю текста, мм: там растяжение и решает.
EVAL_NEAR_MM = 10.0
# Окна дальше этого от края текста в подгонку не идут (как у ``profile.fit_model``), мм.
MAX_FROM_EDGE_MM = 30.0
# Поясов высоты для амплитуды без допущения о форме и для ошибки по частям высоты.
HEIGHT_BANDS = 7
EVAL_PARTS = 5


class HeightShape(StrEnum):
    """Семейство зависимости амплитуды сжатия от доли высоты страницы ``y`` (0 — верх, 1 — низ)."""

    CONST = "const"  # одна амплитуда на всю высоту (прежнее «без наклона»)
    LINEAR = "linear"  # линия без предела наклона (прежнее «с наклоном» с |tilt| ≤ 0.3)
    QUADRATIC = "quadratic"  # полином Лежандра степени 2
    CUBIC = "cubic"  # полином Лежандра степени 3
    SPLINE = "spline"  # кубический P-сплайн с штрафом за вторую разность
    LOGISTIC = "logistic"  # монотонная ступень B₀ + ΔB·σ((y − y₀)/w)


@dataclass(frozen=True)
class ShapeFit:
    """Подогнанная зависимость амплитуды от высоты.

    Поля:
        shape: Семейство.
        coef: Параметры семейства (смысл — у :func:`amplitude_of`).
        length_mm: Общая длина спада сжатия стороны, мм.
        penalty: Штраф P-сплайна (0 у прочих).
        residual: Робастный разброс остатков (1.4826·MAD) по окнам подгонки.
        windows: Окон в подгонке.
    """

    shape: HeightShape
    coef: tuple[float, ...]
    length_mm: float
    penalty: float = 0.0
    residual: float = float("nan")
    windows: int = 0

    def amplitude(self, y_frac: np.ndarray | float) -> np.ndarray:
        """Амплитуда на долях высоты (без обрезки снизу нулём)."""
        return amplitude_of(self.shape, np.asarray(self.coef, np.float64), np.asarray(y_frac, np.float64))


def n_params(shape: HeightShape) -> int:
    """Число параметров амплитуды у семейства (без длины спада)."""
    return {HeightShape.CONST: 1, HeightShape.LINEAR: 2, HeightShape.QUADRATIC: 3, HeightShape.CUBIC: 4,
            HeightShape.SPLINE: SPLINE_KNOTS + 2, HeightShape.LOGISTIC: 4}[shape]  # fmt: skip


def amplitude_of(shape: HeightShape, coef: np.ndarray, y_frac: np.ndarray) -> np.ndarray:
    """Амплитуда сжатия на долях высоты.

    Параметры по семействам:
    - полиномы (CONST, LINEAR, QUADRATIC, CUBIC) — коэффициенты Лежандра по ``z = 2y − 1`` (нулевой — среднее по
      высоте);
    - SPLINE — коэффициенты B-сплайна (:func:`spline_basis`);
    - LOGISTIC — ``(B₀, ΔB, y₀, w)``: ``B₀ + ΔB·σ((y − y₀)/w)``.

    Args:
        shape: Семейство.
        coef: Параметры.
        y_frac: Доли высоты.

    Returns:
        Амплитуда той же формы, что ``y_frac``.
    """
    y = np.asarray(y_frac, np.float64)
    if shape is HeightShape.SPLINE:
        return (spline_basis(y.ravel()) @ coef).reshape(y.shape)
    if shape is HeightShape.LOGISTIC:
        return coef[0] + coef[1] / (1.0 + np.exp(-(y - coef[2]) / coef[3]))
    return legendre.legval(2.0 * y - 1.0, coef)


def initial_and_bounds(shape: HeightShape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Начальная точка и пределы параметров амплитуды семейства.

    Args:
        shape: Семейство.

    Returns:
        (начальные параметры, нижние пределы, верхние пределы).
    """
    lo_a, hi_a = AMPLITUDE_RANGE
    if shape is HeightShape.SPLINE:
        k = n_params(shape)
        return np.full(k, 0.1), np.full(k, lo_a), np.full(k, hi_a)
    if shape is HeightShape.LOGISTIC:
        # Ступень может быть и плавной (w до 1 — почти линия), и резкой (w от 0.05 высоты страницы).
        return (np.array([0.0, 0.1, 0.5, 0.2]), np.array([lo_a, -1.0, -0.2, 0.05]),
                np.array([hi_a, 1.0, 1.2, 1.0]))  # fmt: skip
    k = n_params(shape)
    # Старшие коэффициенты Лежандра ограничены так, чтобы форма не выходила далеко за пределы амплитуды.
    return np.r_[0.1, np.zeros(k - 1)], np.r_[lo_a, np.full(k - 1, -0.6)], np.r_[hi_a, np.full(k - 1, 0.6)]


def second_diff(k: int) -> np.ndarray:
    """Матрица вторых разностей (k − 2, k) — штраф P-сплайна."""
    return np.diff(np.eye(k), 2, axis=0)


def residuals(params: np.ndarray, shape: HeightShape, x: np.ndarray, y: np.ndarray, r: np.ndarray,
              penalty: float) -> np.ndarray:  # fmt: skip
    """Остатки подгонки: ``r − (1 − B(y)·exp(−x/L))`` по окнам и штраф P-сплайна.

    Args:
        params: ``(L, параметры амплитуды…)``.
        shape: Семейство.
        x: Расстояние окон от края текста стороны, мм (не меньше 0).
        y: Доли высоты окон.
        r: Отношения меры окон к опоре строки.
        penalty: Штраф за вторую разность коэффициентов (только SPLINE).

    Returns:
        Остатки окон, у P-сплайна — с хвостом штрафа.
    """
    length, coef = params[0], params[1:]
    out = r - (1.0 - amplitude_of(shape, coef, y) * np.exp(-x / length))
    if shape is HeightShape.SPLINE and penalty > 0:
        # Штраф масштабирован к числу окон: λ одинаково сглаживает сторону из 300 и из 3000 окон.
        out = np.r_[out, np.sqrt(penalty * len(r) / 1000.0) * (second_diff(len(coef)) @ coef)]
    return out


def fit_shape(x: np.ndarray, y: np.ndarray, r: np.ndarray, shape: HeightShape, penalty: float = 0.0,
              length_mm: float | None = None) -> ShapeFit:  # fmt: skip
    """Робастная подгонка амплитуды семейства и общей длины спада стороны (мягкая L1, как ``profile.fit_model``).

    Args:
        x: Расстояние окон от края текста стороны, мм.
        y: Доли высоты окон.
        r: Отношения меры.
        shape: Семейство.
        penalty: Штраф P-сплайна.
        length_mm: Длина спада задана (подгоняется только амплитуда) или None — подгоняется вместе.

    Returns:
        Подгонка.
    """
    start, lo, hi = initial_and_bounds(shape)
    if length_mm is None:
        start, lo, hi = np.r_[6.0, start], np.r_[LENGTH_MM[0], lo], np.r_[LENGTH_MM[1], hi]
    else:
        # Длина спада закреплена узким интервалом вокруг заданной.
        start, lo, hi = np.r_[length_mm, start], np.r_[length_mm - 1e-6, lo], np.r_[length_mm + 1e-6, hi]
    fit = least_squares(residuals, x0=start, bounds=(lo, hi), loss="soft_l1", f_scale=FIT_SCALE,
                        args=(shape, x, y, r, penalty))  # fmt: skip
    res = residuals(fit.x, shape, x, y, r, 0.0)
    mad = 1.4826 * float(np.median(np.abs(res - np.median(res))))
    return ShapeFit(shape, tuple(float(v) for v in fit.x[1:]), float(fit.x[0]), penalty, mad, len(r))


@dataclass(frozen=True)
class SideData:
    """Окна стороны, готовые к подгонке.

    Поля:
        x: Расстояние от края текста стороны, мм.
        y: Доля высоты страницы.
        r: Отношение меры к опоре строки.
        line: Номер строки окна.
        from_edge: Расстояние окна от края текста своей строки, мм (для отбора окон оценки).
    """

    x: np.ndarray
    y: np.ndarray
    r: np.ndarray
    line: np.ndarray
    from_edge: np.ndarray

    def subset(self, mask: np.ndarray) -> SideData:
        """Окна по маске."""
        return SideData(self.x[mask], self.y[mask], self.r[mask], self.line[mask], self.from_edge[mask])


def side_data(part: pd.DataFrame, method: Method, page_top: float, page_height: float) -> SideData:
    """Окна одной стороны для подгонки формы (те же отборы, что у ``profile.fit_model``).

    Args:
        part: Окна стороны (поправленные на место в строке).
        method: Мера.
        page_top: Верх страницы на холсте, px.
        page_height: Высота страницы, px.

    Returns:
        Окна стороны.
    """
    column = ratio_column(method)
    edge = float(part.edge_mm.median())
    data = part[part.from_edge_mm <= MAX_FROM_EDGE_MM][["d_mm", "y", "line", "from_edge_mm", column]].dropna()
    data = data[(data[column] > 0.3) & (data[column] < 2.0)]
    return SideData(np.maximum(data.d_mm.to_numpy() - edge, 0.0),
                    np.clip((data.y.to_numpy() - page_top) / max(page_height, 1.0), 0.0, 1.0),
                    data[column].to_numpy(np.float64), data.line.to_numpy(), data.from_edge_mm.to_numpy())  # fmt: skip


def line_folds(data: SideData) -> np.ndarray:
    """Номер части перекрёстной проверки для каждого окна.

    Строки упорядочены по высоте и собраны в группы по :data:`LINES_PER_GROUP` соседних; группа ``i`` идёт в часть
    ``i mod FOLDS``. Так каждая часть покрывает всю высоту, а соседние строки (сжатие у них почти одно) не попадают
    одновременно в подгонку и в оценку.

    Args:
        data: Окна стороны.

    Returns:
        int (N,) — часть окна.
    """
    order = pd.Series(data.y).groupby(data.line).median().sort_values()
    group = {line: (i // LINES_PER_GROUP) % FOLDS for i, line in enumerate(order.index)}
    return np.array([group[v] for v in data.line])


def cv_errors(data: SideData, shape: HeightShape, penalty: float = 0.0) -> tuple[float, np.ndarray]:
    """Ошибка семейства на отложенных строках: средний модуль отклонения окон ближе :data:`EVAL_NEAR_MM` к краю.

    Args:
        data: Окна стороны.
        shape: Семейство.
        penalty: Штраф P-сплайна.

    Returns:
        (ошибка по всем отложенным окнам; ошибки по :data:`EVAL_PARTS` частям высоты, NaN — окон нет).
    """
    folds = line_folds(data)
    errors = np.full(len(data.r), np.nan)
    for k in range(FOLDS):
        test = folds == k
        if not test.any() or (~test).sum() < MIN_WINDOWS // 2:
            continue
        fit = fit_shape(data.x[~test], data.y[~test], data.r[~test], shape, penalty)
        model = 1.0 - fit.amplitude(data.y[test]) * np.exp(-data.x[test] / fit.length_mm)
        errors[test] = np.abs(data.r[test] - model)
    near = (data.from_edge <= EVAL_NEAR_MM) & np.isfinite(errors)
    parts = np.minimum((data.y * EVAL_PARTS).astype(int), EVAL_PARTS - 1)
    by_part = np.array([errors[near & (parts == p)].mean() if (near & (parts == p)).any() else np.nan
                        for p in range(EVAL_PARTS)])  # fmt: skip
    return float(errors[near].mean()) if near.any() else float("nan"), by_part


def band_amplitudes(data: SideData, length_mm: float) -> np.ndarray:
    """Амплитуда сжатия в поясах высоты без допущения о форме: в каждом поясе своя, длина спада — общая стороны.

    Args:
        data: Окна стороны.
        length_mm: Длина спада стороны (модель с постоянной амплитудой), мм.

    Returns:
        float (HEIGHT_BANDS,) — амплитуда по поясам сверху вниз; NaN — окон в поясе мало.
    """
    bands = np.minimum((data.y * HEIGHT_BANDS).astype(int), HEIGHT_BANDS - 1)
    out = np.full(HEIGHT_BANDS, np.nan)
    for b in range(HEIGHT_BANDS):
        sel = bands == b
        if sel.sum() >= 15:
            out[b] = fit_shape(data.x[sel], data.y[sel], data.r[sel], HeightShape.CONST, length_mm=length_mm).coef[0]
    return out


def edge_excursion(fit: ShapeFit) -> float:
    """«Выстрел» формы на краях высоты: наибольшая разница амплитуды на краю (0 и 1) и в 0.1 высоты от края.

    Там строк мало, и гибкая форма может уйти вверх или вниз без опоры.

    Args:
        fit: Подгонка.

    Returns:
        Наибольший модуль разницы.
    """
    a = fit.amplitude(np.array([0.0, 0.1, 0.9, 1.0]))
    return float(max(abs(a[0] - a[1]), abs(a[3] - a[2])))


# Палитра кривых листа (BGR): константа серая, линия синяя, полиномы — зелёный и бирюзовый, сплайн красный,
# ступень оранжевая; точки поясов — чёрные.
SHAPE_COLORS = {HeightShape.CONST: (150, 150, 150), HeightShape.LINEAR: (200, 90, 30),
                HeightShape.QUADRATIC: (40, 160, 40), HeightShape.CUBIC: (160, 160, 0),
                HeightShape.SPLINE: (40, 40, 220), HeightShape.LOGISTIC: (0, 140, 255)}  # fmt: skip


# Поля панели кривых (лево, право, верх, низ), px, и диапазон амплитуды по горизонтали.
PANEL_PAD = (30, 10, 28, 22)
PANEL_AMPLITUDE = (-0.2, 0.75)


def panel_points(a: np.ndarray, y: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Точки (амплитуда, доля высоты) → пиксели панели кривых.

    Args:
        a: Амплитуды (N,).
        y: Доли высоты (N,), 0 — верх панели.
        size: ``(ширина, высота)`` панели, px.

    Returns:
        int32 (N, 2) — пиксели ``(x, y)``.
    """
    w, h = size
    pad_l, pad_r, pad_t, pad_b = PANEL_PAD
    a_lo, a_hi = PANEL_AMPLITUDE
    px = pad_l + (np.clip(a, a_lo, a_hi) - a_lo) / (a_hi - a_lo) * (w - pad_l - pad_r)
    py = pad_t + np.asarray(y, np.float64) * (h - pad_t - pad_b)
    return np.stack([px, py], axis=1).round().astype(np.int32)


def curve_panel(title: str, bands: np.ndarray, fits: list[ShapeFit], size: tuple[int, int] = (360, 300)) -> np.ndarray:
    """Панель листа: амплитуда по высоте (вертикаль — высота страницы сверху вниз, горизонталь — амплитуда).

    Высота идёт вниз, как на странице: верх страницы — верх панели.

    Args:
        title: Подпись (ключ кадра и сторона).
        bands: Амплитуда по поясам высоты (точки).
        fits: Подгонки семейств (кривые).
        size: ``(ширина, высота)`` панели, px.

    Returns:
        BGR-панель.
    """
    w, h = size
    pad_t, pad_b = PANEL_PAD[2], PANEL_PAD[3]
    panel = np.full((h, w, 3), 255, np.uint8)
    to_px = partial(panel_points, size=size)
    # Сетка: нулевая амплитуда и 0.25/0.5.
    for a in (0.0, 0.25, 0.5):
        x0 = to_px(np.array([a]), np.array([0.0]))[0, 0]
        cv2.line(panel, (x0, pad_t), (x0, h - pad_b), (225, 225, 225) if a else (180, 180, 180), 1)
        cv2.putText(panel, f"{a:.2f}", (x0 - 12, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1, cv2.LINE_AA)
    ys = np.linspace(0.0, 1.0, 101)
    for fit in fits:
        cv2.polylines(panel, [to_px(fit.amplitude(ys), ys)], False, SHAPE_COLORS[fit.shape], 2, cv2.LINE_AA)
    centers = (np.arange(HEIGHT_BANDS) + 0.5) / HEIGHT_BANDS
    for a, y in zip(bands, centers):
        if np.isfinite(a):
            cv2.circle(panel, tuple(to_px(np.array([a]), np.array([y]))[0]), 4, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.putText(panel, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(panel, "top", (2, pad_t + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(panel, "bottom", (2, h - pad_b), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (90, 90, 90), 1, cv2.LINE_AA)
    return panel


def legend_strip(width: int) -> np.ndarray:
    """Полоса легенды листа кривых: цвет каждого семейства и точки поясов."""
    strip = np.full((26, width, 3), 255, np.uint8)
    x = 8
    items = [(s.value, SHAPE_COLORS[s]) for s in HeightShape] + [("bands (points)", (0, 0, 0))]
    for label, color in items:
        cv2.line(strip, (x, 13), (x + 22, 13), color, 3, cv2.LINE_AA)
        cv2.putText(strip, label, (x + 26, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        x += 40 + 9 * len(label)
    return strip


__all__ = ["HeightShape", "ShapeFit", "SideData", "band_amplitudes", "curve_panel", "cv_errors", "edge_excursion",
           "fit_shape", "legend_strip", "side_data"]  # fmt: skip
