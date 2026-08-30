"""Синтетические развороты для тестов детектора ухода текста под корешок."""

import numpy as np
import pytest

import cv2

WIDTH, HEIGHT = 1600, 1100
PITCH = 26
TOP, BOTTOM = 150, 950


def draw_spread(inner_margin: int, tilt: float = 0.0, rules: bool = False, shadow: bool = True) -> np.ndarray:
    """Рисует разворот с заданным внутренним полем обеих полос.

    Args:
        inner_margin: Пробел между концом строки и сгибом, в пикселях.
        tilt: Наклон разворота, пикселей по всей высоте.
        rules: Рисовать ли вертикальные линейки таблицы у корешка.
        shadow: Затемнять ли бумагу к корешку (как на реальной съёмке).

    Returns:
        Полутоновый кадр float32.
    """
    img = np.full((HEIGHT, WIDTH), 250.0, np.float32)
    fold = WIDTH // 2
    rng = np.random.default_rng(7)
    for y in range(TOP, BOTTOM, PITCH):
        shift = int(round(tilt * (y - HEIGHT / 2) / HEIGHT))
        f = fold + shift
        # строки набора: короткие штрихи, имитирующие буквы
        for x0, x1 in ((f - 620, f - inner_margin), (f + inner_margin, f + 620)):
            x = x0
            while x < x1 - 6:
                w = int(rng.integers(3, 7))
                if x + w > x1:
                    break
                img[y : y + 12, x : x + w] = 20.0
                x += w + int(rng.integers(2, 5))
    if shadow:
        ramp = np.abs(np.arange(WIDTH) - fold).astype(np.float32)
        img *= np.clip(0.78 + 0.22 * np.minimum(ramp / 120.0, 1.0), 0, 1)[None, :]
    for y in range(HEIGHT):
        shift = int(round(tilt * (y - HEIGHT / 2) / HEIGHT))
        img[y, fold + shift - 1 : fold + shift + 2] = 15.0  # тёмный след сгиба
    if rules:
        for dx in (-300, -220, -140, 140, 220, 300):
            for y in range(TOP, BOTTOM):
                shift = int(round(tilt * (y - HEIGHT / 2) / HEIGHT))
                img[y, fold + shift + dx] = 25.0
    return img


@pytest.fixture
def clean_spread() -> np.ndarray:
    """Разворот с нормальным внутренним полем (около двух шагов строк)."""
    return draw_spread(inner_margin=55)


@pytest.fixture
def bitten_spread() -> np.ndarray:
    """Разворот, у которого строки упираются в сгиб."""
    return draw_spread(inner_margin=3)
