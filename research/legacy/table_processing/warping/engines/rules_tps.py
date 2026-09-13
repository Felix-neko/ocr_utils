"""Тонкая пластина по пересечениям линеек: самый гибкий и самый ломкий способ.

ЗАМЫСЕЛ. Пересечения линеек — готовые контрольные точки: мы знаем, где они сейчас, и знаем,
где им следует быть (в узлах прямоугольной решётки). Тонкая пластина строит по ним гладкое
поле смещений. Это стандартный приём (в scikit-image есть ``ThinPlateSplineTransform``,
в проекте — ``RBFInterpolator(kernel="thin_plate_spline")`` в ``dewarp.engines.textline``).

ЧЕМ ЛОМКИЙ. Опор мало и они внутри таблицы; за их пределами пластина расходится. Поэтому
поле обрезается выпуклой оболочкой опор, а за ней смещение считается постоянным — тот же
приём, что в ``textline``, где это уже проверено на паке.
"""

from __future__ import annotations

import cv2
import numpy as np

from research.legacy.table_processing.detection.ruling import find_lines
from research.legacy.table_processing.warping.engines.base import Warped, Warper
from research.legacy.table_processing.warping.trace import crossings, trace

# Меньше этого числа пересечений тонкой пластине строить нечего.
MIN_POINTS = 6

# Сглаживание: то же, что в ``dewarp.engines.textline``, где оно подобрано на паке.
SMOOTHING = 0.01

# Шаг сетки, на которой считается поле, в пикселях. Считать в каждой точке незачем: поле
# гладкое, а расчёт тонкой пластины дорог.
GRID_STEP = 16


def run(gray: np.ndarray, dpi: int) -> Warped:
    from scipy.interpolate import RBFInterpolator

    height, width = gray.shape[:2]
    horizontal, vertical = trace(find_lines(gray, dpi), dpi)
    points = crossings(horizontal, vertical)
    if len(points) < MIN_POINTS:
        return Warped(gray, note=f"пересечений всего {len(points)}")

    # Цель: у всех пересечений одной колонки общий x, у всех пересечений одной строки — общий y.
    xs: dict[int, list[float]] = {}
    ys: dict[int, list[float]] = {}
    for h_index, v_index, x, y in points:
        xs.setdefault(v_index, []).append(x)
        ys.setdefault(h_index, []).append(y)
    target_x = {index: float(np.mean(values)) for index, values in xs.items()}
    target_y = {index: float(np.mean(values)) for index, values in ys.items()}

    source = np.array([[x, y] for _, _, x, y in points], dtype=float)
    shift = np.array([[x - target_x[v_index], y - target_y[h_index]] for h_index, v_index, x, y in points], dtype=float)
    if float(np.abs(shift).max()) < 0.5:
        return Warped(gray, note="пересечения уже на решётке")

    grid_x = np.arange(0, width, GRID_STEP, dtype=float)
    grid_y = np.arange(0, height, GRID_STEP, dtype=float)
    mesh = np.stack(np.meshgrid(grid_x, grid_y), axis=-1).reshape(-1, 2)
    try:
        interpolator = RBFInterpolator(source, shift, kernel="thin_plate_spline", smoothing=SMOOTHING)
        values = interpolator(mesh).reshape(grid_y.size, grid_x.size, 2)
    except Exception as error:
        return Warped(gray, note=f"тонкая пластина не построилась: {error}")

    # За выпуклой оболочкой опор пластина расходится — держим смещение постоянным, обрезая
    # его по величине, наблюдаемой в самих опорах.
    limit = float(np.abs(shift).max()) * 1.5
    values = np.clip(values, -limit, limit)
    field_x = cv2.resize(values[:, :, 0].astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    field_y = cv2.resize(values[:, :, 1].astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)

    base_x, base_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    fixed = cv2.remap(gray, base_x + field_x, base_y + field_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Warped(fixed, np.stack([field_x, field_y]), note=f"по {len(points)} пересечениям", changed=True)


ALGORITHM = Warper(name="rules_tps", summary="тонкая пластина по пересечениям линеек", run=run)
