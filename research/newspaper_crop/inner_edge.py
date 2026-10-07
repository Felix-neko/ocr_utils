"""Видимый край верхнего листа у стороны четырёхугольника: самая внутренняя длинная прямая кромка между стороной и печатью полосы."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Глубина поиска внутрь от стороны, px превью: торчащая стопка и кромка корешка ЭГ — до ~200 px (эталон: до 186).
MAX_DEPTH = 260
# Запас снаружи стороны, px превью: сторона уже на квантиле 0.995, кромка может лежать чуть снаружи.
OUTSIDE = 12
# Наклон кромки к стороне, который перебирается, градусы и шаг.
MAX_TILT, TILT_STEP = 3.0, 0.05
# Поле от печати, px превью: кромка не может лежать ближе к своей печати (≈ 1 мм).
PRINT_MARGIN = 10
# Сглаживание серого перед градиентом поперёк стороны (σ, px превью).
SMOOTH = 1.5
# Полуширина перепада, px превью: ступень края листа держится шире линейки (тонкая тёмная полоса 2–4 px гасится).
STEP_HALF = 2


@dataclass(frozen=True)
class EdgeLine:
    """Найденная кромка: сдвиг внутрь у начала и у конца стороны (px превью), сила (средний перепад), доля длины с перепадом."""

    inward0: float
    inward1: float
    strength: float
    coverage: float


def side_strip(gray: np.ndarray, p: np.ndarray, q: np.ndarray, inward: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Полоса кадра вдоль стороны: строки — глубина от −:data:`OUTSIDE` до :data:`MAX_DEPTH` внутрь, столбцы — вдоль стороны.

    Args:
        gray: Серое превью (float32).
        p: Начало стороны (px превью).
        q: Конец стороны.
        inward: Единичная нормаль внутрь.

    Returns:
        (полоса float32 (D, L), глубины строк (D,)).
    """
    length = int(np.linalg.norm(q - p))
    along = (q - p) / np.linalg.norm(q - p)
    depths = np.arange(-OUTSIDE, MAX_DEPTH, dtype=np.float32)
    u = np.arange(length, dtype=np.float32)
    map_x = p[0] + u[None, :] * along[0] + depths[:, None] * inward[0]
    map_y = p[1] + u[None, :] * along[1] + depths[:, None] * inward[1]
    strip = cv2.remap(gray, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)  # fmt: skip
    return strip, depths


def find_inner_edge(
    gray: np.ndarray, p: np.ndarray, q: np.ndarray, inward: np.ndarray, limit: np.ndarray, min_coverage: float = 0.6
) -> EdgeLine | None:
    """Самая внутренняя прямая кромка в полосе у стороны (до печати), с перепадом «темнее снаружи — светлее внутри».

    Перебираются прямые ``v(u) = v0 + u·tg(α)`` (α в ±:data:`MAX_TILT`); сила прямой — средний положительный перепад
    яркости поперёк неё, доля покрытия — часть длины, где перепад выше половины силы сильнейшей прямой. Берётся
    самая внутренняя прямая с покрытием не ниже ``min_coverage`` и силой не меньше половины сильнейшей; прямая не
    должна заходить за предел ``limit`` (печать минус поле).

    Args:
        gray: Серое превью (float32).
        p: Начало стороны.
        q: Конец стороны.
        inward: Нормаль внутрь.
        limit: Предел глубины для каждого столбца полосы (px от стороны; inf — печати нет).
        min_coverage: Наименьшая доля длины с перепадом.

    Returns:
        :class:`EdgeLine` или None (кромки нет).
    """
    strip, depths = side_strip(gray, p, q, inward)
    valid = np.isfinite(strip)
    strip = np.where(valid, strip, np.nanmedian(strip) if valid.any() else 0).astype(np.float32)
    strip = cv2.GaussianBlur(strip, (0, 0), SMOOTH)
    # Перепад внутрь: светлее на STEP_HALF px глубже, чем на STEP_HALF px мельче (край верхнего листа — светлая бумага
    # над тенью).
    h = STEP_HALF
    grad = np.zeros_like(strip)
    grad[h:-h] = strip[2 * h :] - strip[: -2 * h]
    grad = np.clip(grad, 0, None) * valid
    d_count, length = grad.shape
    u = np.arange(length)
    tilts = np.arange(-MAX_TILT, MAX_TILT + 1e-9, TILT_STEP)
    # Проход 1: сила каждой прямой (наклон × строка) — средний перепад вдоль неё.
    strength = np.zeros((len(tilts), d_count), np.float32)
    for k, tilt in enumerate(tilts):
        strength[k] = _line_values(grad, u, np.tan(np.radians(tilt))).mean(axis=1)
    top = float(strength.max())
    if top <= 0:
        return None
    # Проход 2: кандидаты сильнее половины лучшей — от самых внутренних; первая с покрытием и в пределе печати.
    ks, rs = np.nonzero(strength >= 0.5 * top)
    slopes = np.tan(np.radians(tilts[ks]))
    mean_depth = depths[rs] + slopes * (length - 1) / 2
    for i in np.argsort(-mean_depth):
        v = depths[rs[i]] + u * slopes[i]
        if np.mean(v > limit) > 0.02:
            continue
        vals = _line_values(grad, u, slopes[i])[rs[i]]
        coverage = float(np.mean(vals > 0.5 * top))
        if coverage >= min_coverage:
            return EdgeLine(float(v[0]), float(v[-1]), float(strength[ks[i], rs[i]]), coverage)
    return None


def _line_values(grad: np.ndarray, u: np.ndarray, slope: float) -> np.ndarray:
    """Перепад вдоль всех прямых одного наклона: строка r — значения по прямой ``v = r + u·slope`` (0 за полосой).

    Args:
        grad: Перепад (D, L).
        u: Индексы столбцов (L,).
        slope: Тангенс наклона.

    Returns:
        Массив (D, L).
    """
    d_count = grad.shape[0]
    rows = np.arange(d_count)[:, None] + np.round(u * slope).astype(int)[None, :]
    inside = (rows >= 0) & (rows < d_count)
    return np.where(inside, grad[np.clip(rows, 0, d_count - 1), u[None, :]], 0.0)
