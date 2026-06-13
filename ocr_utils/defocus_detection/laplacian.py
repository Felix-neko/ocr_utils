"""Альтернативный метод резкости — дисперсия Лапласиана по тайлам.

Классическая метрика (variance of Laplacian). Оставлена как опция (`--method laplacian`):
проще и быстрее муара, но сильнее зависит от контента (поля → 0, заголовки → мало,
фото → много), поэтому для тонкого зонального расфокуса хуже основного метода.
"""

import cv2
import numpy as np

from ocr_utils.defocus_detection.grid import tile_bounds


def laplacian_tile_maps(gray: np.ndarray, grid_x: int, grid_y: int) -> tuple[np.ndarray, np.ndarray]:
    """Считает по сетке карту дисперсии Лапласиана и карту контраста.

    Args:
        gray: Полутоновое изображение.
        grid_x: Число тайлов по горизонтали.
        grid_y: Число тайлов по вертикали.

    Returns:
        Кортеж (sharp, structure) — два массива shape (grid_y, grid_x):
        sharp — дисперсия Лапласиана в тайле,
        structure — std тайла (мера наличия печатного контента).
    """
    g = gray.astype(np.float32)
    h, w = g.shape
    sharp = np.zeros((grid_y, grid_x), dtype=np.float64)
    structure = np.zeros((grid_y, grid_x), dtype=np.float64)
    for ry in range(grid_y):
        y1, y2 = tile_bounds(h, grid_y, ry)
        for rx in range(grid_x):
            x1, x2 = tile_bounds(w, grid_x, rx)
            tile = g[y1:y2, x1:x2]
            sharp[ry, rx] = cv2.Laplacian(tile, cv2.CV_32F).var()
            structure[ry, rx] = tile.std()
    return sharp, structure
