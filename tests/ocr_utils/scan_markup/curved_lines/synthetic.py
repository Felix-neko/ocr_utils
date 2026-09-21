"""Синтетические полосы с известной геометрией строк.

Строятся на ``orientation.synthetic.text_page`` (масштаб 300 dpi) и искажаются полем
вертикального смещения ``dy(x, y)`` через ``cv2.remap``. Три вида кривизны — как три
вида на настоящих полосах: дуга («прогиб»), волна и локальный поворот блока.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from ocr_utils.scan_markup.curved_lines.detectors.base import Frame
from tests.ocr_utils.page_layout.orientation.synthetic import PAGE_SHAPE, text_page
from tests.ocr_utils.scan_markup.synthetic import PAPER


def warp(image: np.ndarray, dy: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
    """Смещает строки по вертикали: пиксель результата (x, y) берётся из (x, y − dy(x, y))."""
    height, width = image.shape
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_y = (ys - dy(xs, ys)).astype(np.float32)
    return cv2.remap(image, xs, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=PAPER)


def straight_page(seed: int = 0) -> np.ndarray:
    return text_page(seed=seed)


def bow_page(amplitude_px: float = 30.0, seed: int = 0) -> np.ndarray:
    """Все строки — дуги одного знака: середина ниже краёв на ``amplitude_px``."""
    width = PAGE_SHAPE[1]
    return warp(text_page(seed=seed), lambda xs, ys: amplitude_px * (1.0 - ((xs - width / 2.0) / (width / 2.0)) ** 2))


def sine_page(amplitude_px: float = 6.0, wavelength_px: float = 800.0, seed: int = 0) -> np.ndarray:
    """Волны: смещение синусом вдоль строки, одинаковое для всех строк."""
    return warp(text_page(seed=seed), lambda xs, ys: amplitude_px * np.sin(2.0 * np.pi * xs / wavelength_px))


def curl_page(amplitude_px: float = 40.0, width_px: float = 400.0, seed: int = 0) -> np.ndarray:
    """Прогиб у правого края (корешок): строки загибаются вниз на последних ``width_px``."""
    width = PAGE_SHAPE[1]
    return warp(text_page(seed=seed), lambda xs, ys: amplitude_px * np.exp(-(width - xs) / width_px))


def tilted_block_page(angle_deg: float = 2.0, seed: int = 0) -> np.ndarray:
    """Нижняя половина полосы повёрнута на угол: строки прямые, но наклон меняется скачком."""
    image = text_page(seed=seed)
    height, width = image.shape
    top = height // 2
    block = image[top:].copy()
    matrix = cv2.getRotationMatrix2D((width / 2.0, block.shape[0] / 2.0), angle_deg, 1.0)
    image[top:] = cv2.warpAffine(block, matrix, (width, block.shape[0]), flags=cv2.INTER_LINEAR, borderValue=PAPER)
    return image


def frame_of(image: np.ndarray, rel_path: str = "1967/01/IMG_0001_1L.png") -> Frame:
    """Кадр из массива в масштабе 300 dpi — как его собрал бы ``read_frame``."""
    gray300 = np.ascontiguousarray(image)
    gray150 = cv2.resize(gray300, (gray300.shape[1] // 2, gray300.shape[0] // 2), interpolation=cv2.INTER_AREA)
    from pathlib import Path

    return Frame(
        rel_path, Path(rel_path), gray300.shape[1], gray300.shape[0], 300, np.ascontiguousarray(gray150), gray300
    )
