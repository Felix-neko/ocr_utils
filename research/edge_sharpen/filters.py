"""Фильтры кадра сегментации перед SAM 3: нерезкое маскирование, CLAHE по светлоте, локальный контраст, bilateral + нерезкая маска, усиление морфологического градиента, гамма; размеры — в мм бумаги."""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np


class EdgeFilter(str, Enum):
    """Вариант фильтра кадра сегментации."""

    # Без фильтра (повтор SAM 3 по тому же кадру — мера разброса самого SAM 3).
    NONE = "none"
    # Нерезкое маскирование, радиус 0.5 мм, сила 1.5: тонкая кромка листа.
    UNSHARP_FINE = "unsharp_fine"
    # Нерезкое маскирование, радиус 2 мм, сила 1.0: ступенька яркости между листами.
    UNSHARP_WIDE = "unsharp_wide"
    # CLAHE по L в Lab, клетка 20 мм, предел 3.
    CLAHE = "clahe"
    # Локальный контраст ``cv2.detailEnhance`` (доменный фильтр).
    DETAIL = "detail"
    # Bilateral (сглаживание бумаги, края сохраняются) + нерезкое маскирование 1 мм.
    BILATERAL_UNSHARP = "bilateral_unsharp"
    # Прибавка морфологического градиента светлоты (затемнение у перепадов): кромка листа темнеет.
    MORPH_GRADIENT = "morph_gradient"
    # Гамма 1.8 по светлоте: тёмные торцы нижних листов темнеют сильнее светлой страницы.
    GAMMA = "gamma"


# Параметры (мм бумаги; на масштабе сегментации ~6.9 px/мм).
UNSHARP_FINE_MM, UNSHARP_FINE_AMOUNT = 0.5, 1.5
UNSHARP_WIDE_MM, UNSHARP_WIDE_AMOUNT = 2.0, 1.0
CLAHE_TILE_MM, CLAHE_CLIP = 20.0, 3.0
DETAIL_SIGMA_S_MM, DETAIL_SIGMA_R = 1.5, 0.15
BILATERAL_MM, BILATERAL_SIGMA_COLOR, BILATERAL_UNSHARP_MM, BILATERAL_AMOUNT = 1.0, 25.0, 1.0, 1.5
MORPH_GRADIENT_MM, MORPH_GRADIENT_GAIN = 0.6, 1.5
GAMMA = 1.8


def unsharp(bgr: np.ndarray, sigma_px: float, amount: float) -> np.ndarray:
    """Нерезкое маскирование: ``img + amount·(img − гаусс(img))``.

    Args:
        bgr: Кадр BGR uint8.
        sigma_px: Радиус размытия, px.
        amount: Сила.

    Returns:
        Кадр BGR uint8.
    """
    blur = cv2.GaussianBlur(bgr, (0, 0), sigma_px)
    return cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)


def on_lightness(bgr: np.ndarray, func) -> np.ndarray:
    """Применить ``func`` к каналу L в Lab (цвет не трогается)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[..., 0] = func(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def darken_edges(lum: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Светлота минус ``MORPH_GRADIENT_GAIN`` × морфологический градиент: перепады (кромка листа) темнеют.

    Args:
        lum: Канал светлоты uint8.
        kernel: Структурный элемент градиента.

    Returns:
        Канал uint8.
    """
    grad = cv2.morphologyEx(lum, cv2.MORPH_GRADIENT, kernel).astype(np.float32)
    return np.clip(lum.astype(np.float32) - MORPH_GRADIENT_GAIN * grad, 0, 255).astype(np.uint8)


def apply_filter(bgr: np.ndarray, filt: EdgeFilter, px_per_mm: float) -> np.ndarray:
    """Кадр сегментации после фильтра.

    Args:
        bgr: Кадр BGR uint8 (масштаб сегментации).
        filt: Вариант фильтра.
        px_per_mm: Плотность кадра сегментации, px/мм бумаги.

    Returns:
        Кадр BGR uint8 того же размера.
    """
    if filt is EdgeFilter.NONE:
        return bgr.copy()
    if filt is EdgeFilter.UNSHARP_FINE:
        return unsharp(bgr, UNSHARP_FINE_MM * px_per_mm, UNSHARP_FINE_AMOUNT)
    if filt is EdgeFilter.UNSHARP_WIDE:
        return unsharp(bgr, UNSHARP_WIDE_MM * px_per_mm, UNSHARP_WIDE_AMOUNT)
    if filt is EdgeFilter.CLAHE:
        tiles = max(2, int(round(max(bgr.shape[:2]) / (CLAHE_TILE_MM * px_per_mm))))
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(tiles, tiles))
        return on_lightness(bgr, clahe.apply)
    if filt is EdgeFilter.DETAIL:
        return cv2.detailEnhance(bgr, sigma_s=float(DETAIL_SIGMA_S_MM * px_per_mm), sigma_r=DETAIL_SIGMA_R)
    if filt is EdgeFilter.BILATERAL_UNSHARP:
        d = max(3, int(round(BILATERAL_MM * px_per_mm)) | 1)
        smooth = cv2.bilateralFilter(bgr, d, BILATERAL_SIGMA_COLOR, d)
        return unsharp(smooth, BILATERAL_UNSHARP_MM * px_per_mm, BILATERAL_AMOUNT)
    if filt is EdgeFilter.MORPH_GRADIENT:
        k = max(3, int(round(MORPH_GRADIENT_MM * px_per_mm)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return on_lightness(bgr, lambda lum: darken_edges(lum, kernel))
    if filt is EdgeFilter.GAMMA:
        lut = (255.0 * (np.arange(256) / 255.0) ** GAMMA).astype(np.uint8)
        return on_lightness(bgr, lambda lum: cv2.LUT(lum, lut))
    raise ValueError(filt)
