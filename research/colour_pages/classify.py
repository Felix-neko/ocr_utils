"""Уровень цветности страницы: медиана хромы и доля насыщенных пикселей по маске страницы — целиком цветная, цветные элементы, следы цвета, обычная."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

# Целиком цветная страница: медиана хромы (a*, b* OpenCV от 128) по маске страницы не меньше этого. Пак-6, 4921
# страница: медиана в два горба — 0–12 (4809 страниц) и 28+ (112), между ними пусто; порог где угодно в 12…25.
FULL_MIN_MEDIAN = 20.0
# Насыщенный пиксель — хрома больше этого; доли насыщенного для уровней «цветные элементы» и «следы цвета».
SATURATED = 30.0
ELEMENTS_MIN_SHARE = 0.03
TRACE_MIN_SHARE = 0.01


class Colour(Enum):
    """Уровень цветности страницы."""

    FULL = "full"  # целиком на цветном фоне (обложки и их обороты)
    ELEMENTS = "elements"  # белая страница с цветными плашками
    TRACE = "trace"  # тонкие цветные рамки, линии, мелкие логотипы
    PLAIN = "plain"  # обычная


@dataclass(frozen=True)
class PageColour:
    """Цветность страницы.

    Поля:
        median_chroma: Медиана хромы по маске страницы.
        saturated_share: Доля пикселей с хромой больше ``SATURATED``.
        level: Уровень.
    """

    median_chroma: float
    saturated_share: float
    level: Colour


def chroma_of(bgr: np.ndarray) -> np.ndarray:
    """Хрома Lab (OpenCV, a* и b* от 128) по пикселям, float32."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)


def page_colour(chroma: np.ndarray, page: np.ndarray) -> PageColour:
    """Цветность страницы по маске.

    Args:
        chroma: Хрома кадра (``chroma_of``).
        page: Маска страницы (bool).

    Returns:
        Цветность; пустая маска — обычная.
    """
    if not page.any():
        return PageColour(0.0, 0.0, Colour.PLAIN)
    values = chroma[page]
    median, share = float(np.median(values)), float((values > SATURATED).mean())
    if median >= FULL_MIN_MEDIAN:
        level = Colour.FULL
    elif share >= ELEMENTS_MIN_SHARE:
        level = Colour.ELEMENTS
    elif share >= TRACE_MIN_SHARE:
        level = Colour.TRACE
    else:
        level = Colour.PLAIN
    return PageColour(median, share, level)
