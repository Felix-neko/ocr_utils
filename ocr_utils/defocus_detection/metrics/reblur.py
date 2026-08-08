"""Re-blur метрика Crete (The Blur Effect) — «сколько ещё осталось терять».

ИДЕЯ. Размоем кадр ещё раз и посмотрим, насколько упали перепады между соседними
пикселями. Резкий кадр теряет много, уже размытый — почти ничего (терять нечего).

    blur = Σ max(0, D − D_reblur) / Σ D,   D = |I[x] − I[x−1]|

Величина нормирована на собственную энергию перепадов кадра, поэтому по построению
не зависит ни от количества краски на полосе, ни от контраста/экспозиции, и лежит
в [0, 1]. Резкость = 1 − blur.

Ограничение: метрика видит «относительную мягкость», а не абсолютную ширину штриха,
поэтому на очень мелком тексте (штрих ~1 px) она ближе к насыщению, чем ``edge_width``.
Зато она вообще не требует настройки порогов и хорошо работает как второй голос.

Литература: Crete et al., «The Blur Effect: Perception and Estimation with a New
No-Reference Perceptual Blur Metric» (SPIE 2007); реализация той же идеи есть в
``skimage.measure.blur_effect``, но нам нужна карта по тайлам, а не число на кадр.
"""

import cv2
import numpy as np

from ocr_utils.defocus_detection.metrics.base import Algorithm
from ocr_utils.defocus_detection.tiles import Grid, tile_reduce

# Размер ядра пере-размытия: у Crete оптимум 9, и на размеченной папке 1979 года это
# подтвердилось (AUC 0.828 против 0.813 при ядре 5 и 0.743 при ядре 3 — короткое ядро
# слишком близко к шагу растра и меряет уже не штрихи, а зерно).
DEFAULT_KERNEL = 9


def _sharpness_axis(img: np.ndarray, grid: Grid, kernel: np.ndarray) -> np.ndarray:
    """Считает долю перепадов, потерянных при пере-размытии, вдоль одной оси.

    Это ровно `1 − blur` из статьи Crete: доля энергии перепадов, которую кадр теряет
    от дополнительного размытия. Резкому терять есть что (значение ближе к 1), уже
    размытому — нечего (ближе к 0).

    Args:
        img: Полутоновый кадр (float32).
        grid: Сетка тайлов.
        kernel: Усредняющее ядро формы (1, k) или (k, 1) — оно же задаёт ось.

    Returns:
        Массив (ny, nx) со значениями в [0, 1] (больше = резче).
    """
    blurred = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    axis = 1 if kernel.shape[0] == 1 else 0
    d_orig = np.abs(np.diff(img, axis=axis))
    d_blur = np.abs(np.diff(blurred, axis=axis))
    lost = np.maximum(0.0, d_orig - d_blur)

    # Дифференцирование съедает один пиксель — дополняем нулём, чтобы карта совпадала
    # по размеру с кадром и легла на ту же сетку тайлов.
    pad = ((0, 0), (0, 1)) if axis == 1 else ((0, 1), (0, 0))
    lost = np.pad(lost, pad)
    d_orig = np.pad(d_orig, pad)

    lost_sum = tile_reduce(lost, grid, kind="sum")
    orig_sum = tile_reduce(d_orig, grid, kind="sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(orig_sum > 0, lost_sum / np.maximum(orig_sum, 1e-9), np.nan)


def _tile_sharpness(gray: np.ndarray, grid: Grid, kernel_size: int = DEFAULT_KERNEL) -> np.ndarray:
    """Карта резкости 1 − blur по тайлам (больше = резче).

    Args:
        gray: Полутоновый кадр.
        grid: Сетка тайлов.
        kernel_size: Длина усредняющего ядра пере-размытия.

    Returns:
        Массив (ny, nx) со значениями в [0, 1].
    """
    img = gray.astype(np.float32)
    k = np.ones((1, kernel_size), dtype=np.float32) / kernel_size
    sharp_h = _sharpness_axis(img, grid, k)
    sharp_v = _sharpness_axis(img, grid, k.T)
    # Берём худшее из двух направлений: смаз/расфокус часто анизотропен, и достаточно
    # одной «поплывшей» оси, чтобы кадр пришлось переснимать.
    return np.fmin(sharp_h, sharp_v)


ALGORITHM = Algorithm(
    name="reblur",
    summary="re-blur метрика Crete: нормирована по построению, без порогов и настроек",
    tile_sharpness=_tile_sharpness,
    unit="1−blur",
)
