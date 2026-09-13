"""Чтение полос и вырезание кусков — в одном месте, потому что дорого и легко испортить.

ДВА ФОКУСА, оба замерены. Первый: ``Image.draft`` просит JPEG-декодер сразу выдать
уменьшенную картинку, и разжатие полосы 600 dpi до 150 dpi стоит 0.1 с вместо 0.9 с —
декодер пропускает старшие коэффициенты, а не ужимает готовый растр. Второй: масштаб
берётся только степенями двойки (draft умеет 1/2, 1/4, 1/8), а доводка до точного
разрешения делается ``INTER_AREA``.

DPI НЕ ЧИТАЕТСЯ ИЗ ФАЙЛА. В заострённых копиях пака он записан не везде, а ошибка в dpi
сдвигает все пороги пакета разом. Поэтому исходное разрешение задаётся снаружи и по
умолчанию равно ``paths.SOURCE_DPI``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from research.legacy.table_processing.geometry import Box
from research.legacy.table_processing.paths import SOURCE_DPI


def load_gray(path: Path, target_dpi: int, source_dpi: int = SOURCE_DPI) -> np.ndarray:
    """Серая копия полосы в заданном разрешении."""
    image = Image.open(path)
    if target_dpi < source_dpi:
        scale = source_dpi / target_dpi
        image.draft("L", (max(1, round(image.width / scale)), max(1, round(image.height / scale))))
    gray = np.asarray(image.convert("L"))
    wanted_width = max(1, round(gray.shape[1] * target_dpi / source_dpi * (source_dpi / source_dpi)))
    exact_width = max(1, round(_original_width(path) * target_dpi / source_dpi))
    if gray.shape[1] != exact_width:
        exact_height = max(1, round(gray.shape[0] * exact_width / gray.shape[1]))
        interpolation = cv2.INTER_AREA if exact_width < gray.shape[1] else cv2.INTER_CUBIC
        gray = cv2.resize(gray, (exact_width, exact_height), interpolation=interpolation)
    del wanted_width
    return gray


def _original_width(path: Path) -> int:
    """Ширина исходника без разжатия — из заголовка JPEG."""
    with Image.open(path) as image:
        return image.width


def crop(gray: np.ndarray, box: Box) -> np.ndarray:
    """Кусок с обрезкой по краям кадра."""
    height, width = gray.shape[:2]
    safe = box.clipped(width, height)
    return gray[safe.slice]


def to_rgb(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def paper_level(gray: np.ndarray) -> int:
    """Уровень бумаги: верхний дециль яркости. Медиана уводит вниз на плотно набранной
    ячейке, где краски больше половины площади."""
    return int(np.percentile(gray, 90))
