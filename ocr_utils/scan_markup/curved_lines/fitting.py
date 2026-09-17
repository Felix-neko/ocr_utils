"""Математика, общая для детекторов: центр-линия строки, её аппроксимации и сводки по полосе.

Строка здесь — это набор точек (x, y) с весами: ордината центра масс краски в каждом
столбце. По ним считаются прямая и парабола, а дальше три величины, которыми меряется
кривизна: наклон прямой, ПРОГИБ параболы на длине строки (сагитта) и остаток от прямой.

Почему сагитта, а не коэффициент при x². Коэффициент — величина в px⁻¹, и его смысл зависит
от разрешения и длины строки: одна и та же кривизна на короткой подписи и на строке во всю
колонку — это разный прогиб, а глаз и FineReader видят именно прогиб. Сагитта в долях высоты
строки — то, что видно: 0.5 значит «строка ушла на полкегля».

Leptonica (dewarp) меряет ровно коэффициент при x² в микро-единицах; он тоже выдаётся,
для сверки с её порогами (модель годна при кривизне строки ≤ 150 микро-единиц на её
рабочем разрешении).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

# Меньше этого числа столбцов с краской строку не аппроксимировать: парабола по горстке
# точек ловит форму букв, а не строки.
MIN_POINTS = 12

# Строка считается «длинной», если её длина не меньше этой доли от p90 длин по полосе.
# Только по длинным строкам считаются сводки кривизны: сагитта растёт с квадратом длины,
# и короткие строки (концы абзацев, подписи) заведомо дают малые прогибы, размывая
# перцентили. Leptonica берёт 0.8 от максимума; здесь мягче, потому что заголовок или
# таблица во всю полосу задирают максимум.
LONG_LINE_FRACTION = 0.6

# Сколько самых больших прогибов усредняется в «устойчивый максимум». Один максимум —
# всегда выброс (слипшаяся с линейкой строка), три — уже форма.
TOP_K = 3


@dataclass(frozen=True)
class LineFit:
    """Аппроксимации одной строки.

    Attributes:
        slope_deg: Наклон прямой в градусах; ось y вниз, поэтому положительный наклон —
            строка опускается слева направо.
        curvature: Коэффициент при x² параболы, px⁻¹.
        sagitta: Прогиб параболы на длине строки, px, со знаком (положительный — середина
            ниже концов).
        resid_lin: RMS остаток сглаженной центр-линии от прямой, px.
        resid_quad: RMS остаток от параболы, px.
        length: Длина строки по x, px.
        n: Число столбцов с краской.
        x0: Левый край строки.
        y0: Ордината прямой на левом краю.
        x1: Правый край.
        y1: Ордината прямой на правом краю.
    """

    slope_deg: float
    curvature: float
    sagitta: float
    resid_lin: float
    resid_quad: float
    length: float
    n: int
    x0: float
    y0: float
    x1: float
    y1: float


def centreline(ink: np.ndarray, step: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Центр масс краски по столбцам.

    Args:
        ink: Маска краски строки (ненулевое — краска), прямоугольный вырез.
        step: Брать каждый ``step``-й столбец.

    Returns:
        ``(xs, ys, weights)``: абсциссы столбцов с краской, ордината центра масс в каждом
        и число пикселей краски (вес). Координаты — в пикселях выреза.
    """
    mask = ink > 0
    if step > 1:
        mask = mask[:, ::step]
    counts = mask.sum(axis=0)
    xs = np.nonzero(counts)[0]
    if xs.size == 0:
        return xs.astype(np.float64), np.empty(0), np.empty(0)
    rows = np.arange(mask.shape[0], dtype=np.float64)[:, None]
    ys = (mask * rows).sum(axis=0)[xs] / counts[xs]
    return xs.astype(np.float64) * step, ys, counts[xs].astype(np.float64)


def smooth_median(ys: np.ndarray, window: int) -> np.ndarray:
    """Медианное сглаживание центр-линии.

    Центр масс столбца прыгает на выносных элементах и знаках препинания; окно порядка
    двух высот строки убирает форму букв и оставляет форму строки.
    """
    window = int(window) | 1
    if window < 3 or ys.size < window:
        return ys
    return median_filter(ys, size=window, mode="nearest")


def fit_line(xs: np.ndarray, ys: np.ndarray, weights: np.ndarray | None = None) -> LineFit | None:
    """Прямая и парабола через точки центр-линии.

    Returns:
        ``LineFit`` либо None, если точек мало или строка вырождена по длине.
    """
    if xs.size < MIN_POINTS:
        return None
    length = float(xs.max() - xs.min())
    if length <= 0.0:
        return None
    centre = float(xs.mean())
    xc = xs - centre  # центрирование — ради обусловленности полинома
    w = None if weights is None else np.sqrt(np.maximum(weights, 1e-9))
    lin = np.polyfit(xc, ys, 1, w=w)
    quad = np.polyfit(xc, ys, 2, w=w)
    resid_lin = float(np.sqrt(np.mean((ys - np.polyval(lin, xc)) ** 2)))
    resid_quad = float(np.sqrt(np.mean((ys - np.polyval(quad, xc)) ** 2)))
    curvature = float(quad[0])
    half = length / 2.0
    x0, x1 = float(xs.min()), float(xs.max())
    return LineFit(
        slope_deg=float(np.degrees(np.arctan(lin[0]))),
        curvature=curvature,
        sagitta=curvature * half * half,
        resid_lin=resid_lin,
        resid_quad=resid_quad,
        length=length,
        n=int(xs.size),
        x0=x0,
        y0=float(np.polyval(lin, x0 - centre)),
        x1=x1,
        y1=float(np.polyval(lin, x1 - centre)),
    )


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else 0.0


def _top_mean(values: np.ndarray, k: int = TOP_K) -> float:
    if values.size == 0:
        return 0.0
    return float(np.sort(values)[-k:].mean())


def long_mask(fits: list[LineFit]) -> np.ndarray:
    """Какие строки «длинные» — по ним считаются сводки кривизны."""
    lengths = np.array([fit.length for fit in fits], dtype=np.float64)
    if lengths.size == 0:
        return np.zeros(0, dtype=bool)
    return lengths >= LONG_LINE_FRACTION * np.percentile(lengths, 90)


def page_stats(fits: list[LineFit], heights: list[float]) -> dict[str, float]:
    """Сводка метрик по полосе из аппроксимаций строк.

    Все величины — «больше — кривее». Прогибы и остатки нормированы на высоту своей строки,
    наклоны — в градусах.
    """
    if not fits:
        return {"lines": 0.0, "lines_long": 0.0}
    height = np.maximum(np.array(heights, dtype=np.float64), 1.0)
    long = long_mask(fits)
    sag_rel = np.abs(np.array([fit.sagitta for fit in fits])) / height
    resid_lin_rel = np.array([fit.resid_lin for fit in fits]) / height
    resid_quad_rel = np.array([fit.resid_quad for fit in fits]) / height
    curv_micro = np.abs(np.array([fit.curvature for fit in fits])) * 1e6
    slopes = np.array([fit.slope_deg for fit in fits])
    slopes_long = slopes[long]
    return {
        "lines": float(len(fits)),
        "lines_long": float(int(long.sum())),
        "sagitta_rel_p90": _percentile(sag_rel[long], 90),
        "sagitta_rel_max3": _top_mean(sag_rel[long]),
        "sagitta_rel_med": _percentile(sag_rel[long], 50),
        "curv_micro_p90": _percentile(curv_micro[long], 90),
        "resid_lin_rel_p90": _percentile(resid_lin_rel[long], 90),
        "resid_quad_rel_p90": _percentile(resid_quad_rel[long], 90),
        "slope_p50": _percentile(slopes, 50),
        "slope_spread_deg": _percentile(slopes_long, 90) - _percentile(slopes_long, 10) if slopes_long.size else 0.0,
    }


def plane_fit(
    xs: np.ndarray, ys: np.ndarray, values: np.ndarray, weights: np.ndarray | None = None
) -> tuple[float, float, float, float]:
    """Плоскость ``value = a + bx·x + by·y`` по точкам, взвешенно.

    Координаты подаются нормированными на размер полосы (0..1), тогда ``bx`` и ``by`` —
    изменение величины поперёк всей полосы. Возвращает ``(a, bx, by, rms остатка)``.
    Меньше трёх точек — плоскость не строится, остаток равен разбросу вокруг среднего.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    w = np.ones_like(values) if weights is None else np.sqrt(np.maximum(np.asarray(weights, dtype=np.float64), 1e-9))
    if values.size < 3:
        mean = float(np.average(values, weights=w * w))
        return mean, 0.0, 0.0, float(np.sqrt(np.average((values - mean) ** 2, weights=w * w)))
    design = np.stack([np.ones_like(values), np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)], 1)
    coef, *_ = np.linalg.lstsq(design * w[:, None], values * w, rcond=None)
    resid = values - design @ coef
    rms = float(np.sqrt(np.average(resid * resid, weights=w * w)))
    return float(coef[0]), float(coef[1]), float(coef[2]), rms


def slope_field_stats(
    xs: np.ndarray, ys: np.ndarray, slopes: np.ndarray, weights: np.ndarray | None = None
) -> dict[str, float]:
    """Как наклон строк меняется по полосе: градиенты плоскости и остаток от неё.

    Остаток — главный признак «волн»: у прямой, пусть и перекошенной полосы наклон всюду
    один, у трапеции меняется линейно, а у волнистой не описывается плоскостью вовсе.
    """
    _, grad_x, grad_y, rms = plane_fit(xs, ys, slopes, weights)
    return {"slope_grad_x": grad_x, "slope_grad_y": grad_y, "slope_resid_deg": rms}


def clusters_1d(values: np.ndarray, tol: float, min_size: int) -> list[np.ndarray]:
    """Кластеры по одной координате: разрыв больше ``tol`` начинает новый кластер.

    Returns:
        Список массивов индексов исходных точек; кластеры меньше ``min_size`` отброшены.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return []
    order = np.argsort(values, kind="mergesort")
    clusters: list[list[int]] = [[int(order[0])]]
    for previous, current in zip(order[:-1], order[1:]):
        if values[current] - values[previous] > tol:
            clusters.append([])
        clusters[-1].append(int(current))
    return [np.array(cluster) for cluster in clusters if len(cluster) >= min_size]


def edge_fit(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float, float] | None:
    """Парабола ``x = a + b·y + c·y²`` через край колонки (начала или концы строк).

    Returns:
        ``(sagitta_x, a, b, c)`` — прогиб края по x на его высоте, со знаком; None, если
        точек мало.
    """
    if ys.size < 4:
        return None
    height = float(ys.max() - ys.min())
    if height <= 0.0:
        return None
    centre = float(ys.mean())
    coef = np.polyfit(ys - centre, xs, 2)
    return float(coef[0] * (height / 2.0) ** 2), float(coef[2]), float(coef[1]), float(coef[0])
