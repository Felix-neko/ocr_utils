"""Быстрые медиана, квантиль, `polyfit` и центр-линия строки (Numba): те же числа, что у numpy, без накладных расходов.

Разбор текстовых блоков зовёт ``np.median`` и ``np.percentile`` сотни тысяч раз на массивах из десятков чисел
(кегль и низ глифов оси, опорные строчные базовой линии): по профилю пересборки блоков трёх полос — 231 тыс. вызовов,
треть всего времени, и почти всё это — проверки и разбор аргументов внутри numpy, а не счёт. Здесь то же самое
считается ядрами ``@njit``:

* :func:`median` и :func:`percentile` дают **те же числа**, что numpy (способ ``linear``, та же формула
  интерполяции ``_lerp`` с ветвью ``t >= 0.5``, NaN во входе — NaN на выходе). Порядковые статистики не зависят от
  того, сортировать или делить массив, поэтому совпадение — бит в бит;
* :func:`polyfit` повторяет ``np.polyfit(x, y, степень)`` по шагам — матрица Вандермонда, масштаб столбцов, тот же
  ``lstsq`` LAPACK (``gelsd``) с тем же ``rcond`` — и совпадает бит в бит (проверено на 12 тыс. случайных подгонок
  степеней 1 и 2), но без обвязки numpy вокруг: 2.8 мкс против 22 мкс на прямой по 12 точкам;
* :func:`refined_columns` — цикл по столбцам второго прохода центр-линии строки
  (``segment.refined_centreline``); суммы там целочисленные, совпадение бит в бит.

Что ядро не берёт (не одномерный массив, пустой, ``float32``, комплексный тип, NaN и бесконечности в подгонке),
уходит в numpy как прежде.
Скомпилированные ядра кэшируются на диске (``cache=True``) рядом с модулем: воркеры пула не компилируют их заново.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Типы, которые ядра считают без потери совпадения с numpy: float64 как есть, целые и беззнаковые — через float64
# (numpy у целых считает медиану и квантиль тоже во float64; целые высоты и координаты глифов переводятся точно).
_EXACT_KINDS = frozenset("iu")
# Машинное эпсилон float64: ``np.polyfit`` берёт ``rcond = len(x) * eps``.
_EPS = float(np.finfo(np.float64).eps)


@njit(cache=True)
def _median_kernel(values: np.ndarray) -> float:
    """Медиана одномерного ``float64`` как у ``np.median``: средний элемент или полусумма двух средних.

    Args:
        values: Непустой одномерный массив ``float64`` (не меняется: сортируется копия).

    Returns:
        Медиана; NaN, если во входе есть NaN.
    """
    ordered = np.sort(values)
    size = ordered.size
    # NaN при сортировке уходит в конец: numpy в этом случае отдаёт NaN.
    if np.isnan(ordered[size - 1]):
        return np.nan
    half = size // 2
    if size % 2 == 1:
        return ordered[half]
    # numpy берёт ``mean`` двух средних: сумма, затем деление на 2 — та же операция.
    return (ordered[half - 1] + ordered[half]) / 2.0


@njit(cache=True)
def _percentile_kernel(values: np.ndarray, q: float) -> float:
    """Квантиль одномерного ``float64`` как у ``np.percentile(values, q)`` (способ ``linear``).

    Повторяет numpy по шагам: доля ``q / 100``, виртуальный индекс ``(n − 1)·доля``, соседние порядковые
    статистики и интерполяция ``_lerp``: ``a + (b − a)·t`` при ``t < 0.5`` и ``b − (b − a)·(1 − t)`` иначе.

    Args:
        values: Непустой одномерный массив ``float64`` (не меняется).
        q: Процентиль от 0 до 100.

    Returns:
        Квантиль; NaN, если во входе есть NaN.
    """
    ordered = np.sort(values)
    size = ordered.size
    if np.isnan(ordered[size - 1]):
        return np.nan
    share = q / 100.0
    virtual = (size - 1) * share
    # Индекс за последним элементом — сам последний элемент (у numpy оба соседа становятся последним).
    if virtual >= size - 1:
        return ordered[size - 1]
    previous = np.floor(virtual)
    gamma = virtual - previous
    low = ordered[int(previous)]
    high = ordered[int(previous) + 1]
    diff = high - low
    if gamma >= 0.5:
        return high - diff * (1.0 - gamma)
    return low + diff * gamma


@njit(cache=True)
def _polyfit_kernel(xs: np.ndarray, ys: np.ndarray, degree: int) -> np.ndarray:
    """Коэффициенты полинома МНК как у ``np.polyfit(xs, ys, degree)`` — те же шаги и тот же ``lstsq``.

    Args:
        xs: Абсциссы ``float64``.
        ys: Ординаты той же длины.
        degree: Степень полинома.

    Returns:
        Коэффициенты от старшей степени к свободному члену (для ``np.polyval``).
    """
    size = xs.size
    # Матрица Вандермонда: столбцы x^степень … x^0, степени — последовательным умножением, как ``np.vander``.
    lhs = np.empty((size, degree + 1))
    for row in range(size):
        value = 1.0
        for column in range(degree, -1, -1):
            lhs[row, column] = value
            value = value * xs[row]
    # Масштаб столбцов к единичной норме (обусловленность), затем МНК с ``rcond = n·eps``, обратный масштаб.
    scale = np.sqrt((lhs * lhs).sum(axis=0))
    lhs = lhs / scale
    coefficients, _, _, _ = np.linalg.lstsq(lhs, ys, size * _EPS)
    return coefficients / scale


@njit(cache=True)
def _refined_columns_kernel(
    mask: np.ndarray, xs: np.ndarray, anchor: np.ndarray, half: float, ys: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Центр масс краски каждого столбца в полосе ``±half`` вокруг ``anchor`` (см. :func:`refined_columns`)."""
    out_y = np.empty(xs.size)
    out_w = np.empty(xs.size)
    height = mask.shape[0]
    for index in range(xs.size):
        x = xs[index]
        centre = anchor[index]
        count = 0
        total = 0.0
        for row in range(height):
            if mask[row, x] and abs(row - centre) <= half:
                count += 1
                total += row
        if count == 0:
            out_y[index] = ys[index]
            out_w[index] = weights[index]
        else:
            out_y[index] = total / count
            out_w[index] = count
    return out_y, out_w


def _kernel_input(values) -> np.ndarray | None:
    """Массив ``float64`` для ядра или ``None``, если вход ядру не подходит и считать надо numpy.

    Args:
        values: Что угодно, что принимает ``np.median`` (массив, список, кортеж).

    Returns:
        Одномерный непустой ``float64``-массив (у ``float64``-входа — он сам, без копии) или ``None``.
    """
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        return None
    if array.dtype == np.float64:
        return array
    if array.dtype.kind in _EXACT_KINDS:
        return array.astype(np.float64)
    return None


def median(values) -> float:
    """Медиана как ``float(np.median(values))``, но в разы быстрее на малых массивах.

    Args:
        values: Одномерные числа (массив, список). Многомерный, пустой или ``float32``-вход считается numpy.

    Returns:
        Медиана.
    """
    array = _kernel_input(values)
    if array is None:
        return float(np.median(values))
    return float(_median_kernel(array))


def percentile(values, q: float) -> float:
    """Квантиль как ``float(np.percentile(values, q))`` (способ ``linear``), но в разы быстрее на малых массивах.

    Args:
        values: Одномерные числа. Многомерный, пустой или ``float32``-вход считается numpy.
        q: Процентиль от 0 до 100 (скаляр).

    Returns:
        Квантиль.
    """
    array = _kernel_input(values)
    if array is None:
        return float(np.percentile(values, q))
    return float(_percentile_kernel(array, float(q)))


def polyfit(xs, ys, degree: int) -> np.ndarray:
    """Коэффициенты полинома МНК как ``np.polyfit(xs, ys, degree)`` (без весов), бит в бит, но в разы быстрее.

    Args:
        xs: Абсциссы (одномерные).
        ys: Ординаты той же длины (одномерные).
        degree: Степень полинома.

    Returns:
        Коэффициенты от старшей степени к свободному члену (для ``np.polyval``). Многомерный, пустой или
        неконечный вход считается numpy.
    """
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size == 0 or x.size != y.size:
        return np.polyfit(xs, ys, degree)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return np.polyfit(xs, ys, degree)
    return _polyfit_kernel(np.ascontiguousarray(x), np.ascontiguousarray(y), int(degree))


def refined_columns(
    mask: np.ndarray, xs: np.ndarray, anchor: np.ndarray, half: float, ys: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Второй проход центр-линии строки: центр масс краски столбца только в полосе ``±half`` вокруг ``anchor``.

    Args:
        mask: Краска сегмента ``(h, w)``, ``bool``.
        xs: Номера столбцов (целые).
        anchor: Ордината сглаженной линии первого прохода в каждом столбце.
        half: Полувысота полосы, px.
        ys: Ординаты первого прохода (остаются у столбцов без краски в полосе).
        weights: Веса первого прохода (так же).

    Returns:
        ``(ординаты, веса)``: у столбца с краской в полосе — центр масс и число пикселей краски, иначе — первый проход.
    """
    return _refined_columns_kernel(
        np.ascontiguousarray(mask),
        np.ascontiguousarray(xs, dtype=np.int64),
        np.ascontiguousarray(anchor, dtype=np.float64),
        float(half),
        np.ascontiguousarray(ys, dtype=np.float64),
        np.ascontiguousarray(weights, dtype=np.float64),
    )
