"""Синтетические «газетные полосы» для проверки метрик резкости.

Все страницы рисуются из одного набора штрихов, чтобы отличаться ровно одним свойством:
плотностью текста, кеглем, экспозицией или степенью размытия. Именно на таких парах и
проверяется главное требование к метрике — реагировать на фокус, а не на вёрстку.
"""

import cv2
import numpy as np
import pytest

PAPER = 235
INK = 35


def draw_page(
    height: int = 1024, width: int = 1024, *, fill: float = 1.0, stroke: int = 2, line_height: int = 12, seed: int = 0
) -> np.ndarray:
    """Рисует страницу со строками текста заданного кегля и плотности.

    Args:
        height: Высота кадра.
        width: Ширина кадра.
        fill: Доля площади, занятая текстом (1.0 — вся страница, 0.25 — четверть).
        stroke: Толщина штриха в пикселях (кегль).
        line_height: Расстояние между строками.
        seed: Зерно генератора для воспроизводимости.

    Returns:
        Полутоновый кадр uint8.
    """
    rng = np.random.default_rng(seed)
    page = np.full((height, width), float(PAPER))
    text_bottom = int(height * fill)
    for y in range(line_height, text_bottom - line_height, line_height):
        x = 40
        while x < width - 60:
            glyph = int(rng.integers(stroke * 2, stroke * 5))
            page[y : y + stroke * 4, x : x + stroke] = INK  # вертикальный штрих
            page[y : y + stroke, x : x + glyph] = INK  # горизонтальный элемент
            x += glyph + int(rng.integers(stroke, stroke * 3))
    return np.clip(page, 0, 255).astype(np.uint8)


def blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Размывает кадр гауссианой (имитация расфокуса).

    Args:
        image: Полутоновый кадр.
        sigma: Сигма размытия в пикселях; 0 — вернуть кадр как есть.

    Returns:
        Размытый кадр uint8.
    """
    if sigma <= 0:
        return image
    return cv2.GaussianBlur(image, (0, 0), sigma)


def expose(image: np.ndarray, gain: float, offset: float = 0.0, noise: float = 0.0, seed: int = 1) -> np.ndarray:
    """Меняет экспозицию и добавляет шум (имитация другой ISO и освещения).

    Args:
        image: Полутоновый кадр.
        gain: Множитель контраста относительно средней яркости.
        offset: Сдвиг яркости в уровнях.
        noise: СКО аддитивного гауссова шума.
        seed: Зерно генератора шума.

    Returns:
        Кадр uint8 с изменённой экспозицией.
    """
    rng = np.random.default_rng(seed)
    mean = float(image.mean())
    out = (image.astype(np.float64) - mean) * gain + mean + offset
    if noise > 0:
        out = out + rng.normal(0.0, noise, image.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


@pytest.fixture
def sharp_page() -> np.ndarray:
    """Плотно набранная резкая полоса мелким шрифтом."""
    return draw_page()
