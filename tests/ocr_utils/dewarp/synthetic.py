"""Синтетическая кривая полоса: прямой текст, искривлённый известным полем смещений."""

from __future__ import annotations

import cv2
import numpy as np

from tests.ocr_utils.page_layout.orientation.synthetic import PAGE_SHAPE, text_page


def warp(image: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """``dst(x, y) = src(x, y - dy(x, y))``: строка уходит вниз на dy."""
    height, width = image.shape[:2]
    map_x = np.tile(np.arange(width, dtype=np.float32), (height, 1))
    map_y = np.arange(height, dtype=np.float32)[:, None] - dy.astype(np.float32)
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def curl_page(amplitude_px: float = 40.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Прогиб у правого края, растущий книзу (как у корешка): полоса и поле смещений."""
    image = text_page(PAGE_SHAPE, seed=seed)
    height, width = image.shape
    xs = np.arange(width, dtype=np.float64) / width
    ys = np.arange(height, dtype=np.float64) / height
    dy = amplitude_px * np.outer(ys, np.clip(xs - 0.5, 0, None) ** 2 * 4)
    return warp(image, dy), dy


def sine_page(amplitude_px: float = 9.0, waves: float = 1.5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Волны вдоль строки одинаковой формы по всей полосе.

    Амплитуда — порядка трети высоты строки, как на настоящих волнистых полосах; при
    амплитуде в полшага строк соседние строки смыкаются, и это уже не dewarp, а каша.
    """
    image = text_page(PAGE_SHAPE, seed=seed)
    height, width = image.shape
    xs = np.arange(width, dtype=np.float64) / width
    dy = np.tile(amplitude_px * np.sin(2 * np.pi * waves * xs), (height, 1))
    return warp(image, dy), dy
