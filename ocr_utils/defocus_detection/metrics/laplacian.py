"""Наивная базовая метрика — дисперсия Лапласиана (variance of Laplacian).

Классика из всех туториалов: свернуть с лапласианом и взять дисперсию. Оставлена в наборе
НЕ потому, что хороша, а как база для сравнения: она честно демонстрирует ровно те
проблемы, ради которых написаны остальные метрики.

- Зависит от количества краски: малотекстовая полоса «проваливается» без всякого расфокуса.
- Зависит от кегля: на крупном заголовке переходов мало, дисперсия низкая, фокус при этом
  идеальный.
- Зависит от экспозиции и ISO: контраст масштабирует лапласиан квадратично, а шум высокого
  ISO, наоборот, задирает дисперсию — размытый шумный кадр может обогнать резкий чистый.

Работа по тайлам и отбор печатных тайлов первые две проблемы частично лечат, но третья
остаётся, поэтому для боевого ранжирования лучше ``edge_width`` или ``combo``.
"""

import cv2
import numpy as np

from ocr_utils.defocus_detection.metrics.base import Algorithm
from ocr_utils.defocus_detection.tiles import Grid


def _tile_sharpness(gray: np.ndarray, grid: Grid) -> np.ndarray:
    """Карта дисперсии Лапласиана по тайлам (больше = резче).

    Args:
        gray: Полутоновый кадр.
        grid: Сетка тайлов.

    Returns:
        Массив (ny, nx) с дисперсией лапласиана.
    """
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    out = np.full((grid.ny, grid.nx), np.nan)
    for iy in range(grid.ny):
        for ix in range(grid.nx):
            y1, y2, x1, x2 = grid.bounds(iy, ix)
            out[iy, ix] = float(lap[y1:y2, x1:x2].var())
    return out


ALGORITHM = Algorithm(
    name="laplacian",
    summary="дисперсия лапласиана — наивная база для сравнения (зависит от контента, кегля и ISO)",
    tile_sharpness=_tile_sharpness,
    unit="lapvar",
)
