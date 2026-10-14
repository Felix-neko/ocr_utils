"""Доводка края маски бумаги SAM 3 наружу до настоящей кромки: доводка по краю, низкий порог SAM 3, раздутие на сдвиг."""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.book import disk
from ocr_utils.scan_cropping.v2.edge import (  # noqa: F401 — ядро, ре-экспорт для стенда
    REACH_SMOOTH,
    REF_FAR,
    REF_NEAR,
    SNAP_AB,
    SNAP_BAND,
    SNAP_BG_REACH,
    SNAP_DROP,
    snap_outward,
)

# Раздутие маски на средний сдвиг внутрь, px ½ кадра.
DILATE_PX = 6
# Порог вероятности маски SAM 3 вместо штатного 0.5.
LOW_THR = 0.3


class EdgeFix(str, Enum):
    """Способ доводки края."""

    SNAP = "snap"
    SNAP_BG = "snap_bg"
    LOW_THR = "lowthr"
    DILATE = "dilate"




def dilate_fixed(paper: np.ndarray, book: np.ndarray, bg: np.ndarray, px: int = DILATE_PX) -> np.ndarray:
    """Раздутие маски на постоянный сдвиг, не выходя на фон и за книгу.

    Args:
        paper: Маска бумаги.
        book: Маска книги.
        bg: Маска фона.
        px: Радиус, px.

    Returns:
        Маска.
    """
    return paper | ((cv2.dilate(paper.astype(np.uint8), disk(px)) > 0) & book & ~bg)


class _ShiftedF:
    """Подмена модуля ``torch.nn.functional`` в постобработке SAM 3: ``interpolate`` возвращает вероятность + сдвиг.

    Штатная постобработка режет маску как ``interpolate(...) > 0.5``; со сдвигом ``0.5 − thr`` это равносильно
    ``> thr``. Остальные функции модуля отдаются как есть.
    """

    def __init__(self, functional, shift: float) -> None:
        self._f = functional
        self._shift = shift

    def interpolate(self, *args, **kwargs):
        return self._f.interpolate(*args, **kwargs) + self._shift

    def __getattr__(self, name):
        return getattr(self._f, name)


def make_low_thr_predictor(overrides: dict, thr: float = LOW_THR):
    """``SAM3SemanticPredictor`` с порогом вероятности маски ``thr`` вместо 0.5.

    Args:
        overrides: Настройки предиктора.
        thr: Порог.

    Returns:
        Предиктор.
    """
    import torch.nn.functional as F
    import ultralytics.models.sam.predict as predict_module
    from ultralytics.models.sam import SAM3SemanticPredictor

    shim = _ShiftedF(F, 0.5 - thr)

    class LowThr(SAM3SemanticPredictor):
        """SAM 3 по тексту с пониженным порогом маски."""

        def postprocess(self, preds, img, orig_imgs):
            """Штатная постобработка с подменённым ``F`` на время вызова."""
            saved = predict_module.F
            predict_module.F = shim
            try:
                return super().postprocess(preds, img, orig_imgs)
            finally:
                predict_module.F = saved

    return LowThr(overrides=overrides)
