"""Модель горизонтального сжатия страницы у корешка ``c(d, y) = 1 − B(y)·exp(−(d − e)/L)``: амплитуда по высоте страницы — P-сплайн (по умолчанию) или линия, переключение — константой :data:`HEIGHT_SHAPE`."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.interpolate import BSpline


class HeightShape(StrEnum):
    """Зависимость амплитуды сжатия ``B(y)`` от доли высоты страницы ``y`` (0 — верх, 1 — низ)."""

    LINEAR = "linear"  # линия ``B₀ + B₁·(2y − 1)`` (коэффициенты Лежандра)
    SPLINE = "spline"  # кубический P-сплайн на SPLINE_KNOTS равномерных узлах со штрафом SPLINE_PENALTY


# Форма амплитуды по высоте. Стенд research/gutter_compression, шаг shape (834 сжатые стороны ХиП 1991): сплайн со
# штрафом 30 хуже линии на 1 % сторон и лучше на 8 % (ступень сжатия у низа — 01/IMG_1052 R); на 30 кадрах растяжения
# по буквам разница с линией — в шуме (буквы у края после: 0.964 у обоих). Линия — HeightShape.LINEAR.
HEIGHT_SHAPE = HeightShape.SPLINE
# Узлов P-сплайна по высоте и штраф за вторую разность коэффициентов (масштабируется к числу букв, см. ``fit``).
SPLINE_KNOTS = 7
SPLINE_PENALTY = 30.0
# Пределы амплитуды в любой точке высоты при подгонке: снизу — небольшое «расширение» (у страниц без сжатия текст в
# 4–20 мм от корешка шире опоры на 2–4 %), сверху — с запасом над сильнейшим сжатием пака (0.6).
AMPLITUDE_RANGE = (-0.3, 0.75)


def spline_basis(y_frac: np.ndarray) -> np.ndarray:
    """Базис кубического B-сплайна по доле высоты: равномерные узлы на [0, 1].

    Args:
        y_frac: Доли высоты (N,).

    Returns:
        float (N, SPLINE_KNOTS + 2) — значения базисных функций.
    """
    inner = np.linspace(0.0, 1.0, SPLINE_KNOTS)
    knots = np.r_[[0.0] * 3, inner, [1.0] * 3]
    y = np.clip(np.asarray(y_frac, np.float64), 0.0, 1.0)
    return BSpline.design_matrix(y, knots, 3).toarray()


def n_params(shape: HeightShape) -> int:
    """Число параметров амплитуды у формы (без длины спада)."""
    return SPLINE_KNOTS + 2 if shape is HeightShape.SPLINE else 2


def amplitude_of(shape: HeightShape, coef: np.ndarray, y_frac: np.ndarray) -> np.ndarray:
    """Амплитуда сжатия на долях высоты (без обрезки нулём).

    Args:
        shape: Форма: SPLINE — коэффициенты B-сплайна (:func:`spline_basis`), LINEAR — коэффициенты Лежандра по
            ``z = 2y − 1`` (нулевой — амплитуда в середине высоты).
        coef: Параметры формы.
        y_frac: Доли высоты.

    Returns:
        Амплитуда той же формы, что ``y_frac``.
    """
    y = np.asarray(y_frac, np.float64)
    if shape is HeightShape.SPLINE:
        return (spline_basis(y.ravel()) @ np.asarray(coef, np.float64)).reshape(y.shape)
    return coef[0] + coef[1] * (2.0 * y - 1.0)


@dataclass(frozen=True)
class CompressionModel:
    """Модель сжатия страницы у корешка: ``c(d, y) = 1 − B(y)·exp(−max(d − e, 0)/L)``.

    ``d`` — расстояние от линии корешка по холсту (мм), ``e`` — начало спада (начало зоны текста стороны, мм),
    ``y`` — доля высоты страницы. ``c`` — во сколько раз отрезок строки на холсте короче, чем на бумаге; растяжение
    ``1/c`` уходит в 1 с удалением от корешка.

    Поля:
        shape: Форма амплитуды по высоте.
        coef: Параметры формы (:func:`amplitude_of`); пусто — сжатия нет.
        length_mm: Длина спада ``L``, мм.
        origin_mm: Начало спада ``e`` от линии корешка, мм.
        letters: Букв в подгонке.
        lines: Строк в подгонке.
        residual: Робастный разброс остатков логарифма отношения (1.4826·MAD).
    """

    shape: HeightShape = HEIGHT_SHAPE
    coef: tuple[float, ...] = ()
    length_mm: float = 5.0
    origin_mm: float = 0.0
    letters: int = 0
    lines: int = 0
    residual: float = float("nan")

    def amplitude_at(self, y_frac: np.ndarray | float) -> np.ndarray:
        """Амплитуда на доле высоты страницы, не меньше нуля (у нулевой модели — 0)."""
        y = np.asarray(y_frac, np.float64)
        if not self.coef:
            return np.zeros_like(y)
        return np.maximum(amplitude_of(self.shape, np.asarray(self.coef), y), 0.0)

    @property
    def null(self) -> bool:
        """Модель без сжатия (подгонки не было)."""
        return not self.coef


__all__ = ["HEIGHT_SHAPE", "CompressionModel", "HeightShape", "amplitude_of", "spline_basis"]
