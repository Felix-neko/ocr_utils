"""Компенсация уровней внутри маски страницы (контраст-стретч)."""

import cv2
import numpy as np
from skimage.exposure import rescale_intensity

from ocr_utils.scan_cropping.page_detection import WORK_SIDE

# Компенсация уровней: перцентили по общей интенсивности внутри маски (минус эрозия)
N_EROSION_PX = 20
LEVELS_LOW_PCT = 1.0
LEVELS_HIGH_PCT = 98.0


def compensate_levels(
    bgr: np.ndarray,
    mask: np.ndarray,
    erosion_px: int,
    low_pct: float = LEVELS_LOW_PCT,
    high_pct: float = LEVELS_HIGH_PCT,
    work_side: int = WORK_SIDE,
) -> np.ndarray:
    """Растягивает уровни по общей интенсивности (одинаково для всех каналов).

    Перцентили считаются по пикселям внутри маски страницы, эрозированной на
    ``erosion_px`` (чтобы не захватывать край страницы/фон). Диапазон общий для
    B/G/R — это не независимая цветокоррекция по каналам, а контраст-стретч,
    сохраняющий цветовой баланс.

    Эрозия и ``np.percentile`` считаются на копии, уменьшенной до ``work_side``
    (как и в ``page_mask``) — это лишь ОЦЕНКА перцентилей, полное разрешение ей
    не нужно, а на кадрах 30-48 Мп percentile по маске занимал секунды (см.
    профилирование ``detect_and_crop`` на медленных прогонах). Сам контраст-стретч
    (``rescale_intensity``) применяется к исходному кадру полного разрешения —
    только на нём и формируется итоговый результат.
    """
    h, w = mask.shape[:2]
    scale = work_side / max(h, w) if max(h, w) > work_side else 1.0
    if scale < 1.0:
        small_mask = cv2.resize(mask, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
        small_bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        small_erosion_px = max(1, int(round(erosion_px * scale)))
    else:
        small_mask, small_bgr, small_erosion_px = mask, bgr, erosion_px

    eroded = small_mask
    if small_erosion_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (small_erosion_px * 2 + 1, small_erosion_px * 2 + 1))
        eroded = cv2.erode(small_mask, k)
    sel = eroded > 0
    if not np.any(sel):
        return bgr

    small_bgr_f = small_bgr.astype(np.float32) / 255.0
    lo, hi = np.percentile(small_bgr_f[sel], (low_pct, high_pct))
    if hi <= lo:
        return bgr

    bgr_f = bgr.astype(np.float32) / 255.0
    out = rescale_intensity(bgr_f, in_range=(lo, hi), out_range=(0.0, 1.0))
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)
