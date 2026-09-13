"""Распознавание ячейки через surya: детектор строк плюс распознаватель.

Surya в проекте уже используется (``gutter_loss_restoration.pageocr``), веса скачаны,
кириллицу она знает: 88.8% на её собственном многоязычном замере.

ДЕТЕКТОР СТРОК ПОДАЁТСЯ ЯВНО. Без него распознаватель читает вырезку как одну строку и
склеивает две-три строки заголовка в кашу; с ним строки возвращаются по отдельности —
а рендеру нужно знать, где были переносы.

ОТКАТ ПО ПАМЯТИ — как в ``pageocr._recognize``: при нехватке видеопамяти партия делится
пополам, пока не влезет. Ячеек в таблице десятки, но подряд идут таблицы, и без отката
прогон падал бы на самой широкой.
"""

from __future__ import annotations

import time
from typing import Sequence

import cv2
import numpy as np

from research.legacy.table_processing.ocr.base import Engine, OcrResult, prepare

DEFAULT_BATCH = 16
BATCH_MIN = 1

_SHARED = None


def surya_available() -> bool:
    try:
        import surya.recognition  # noqa: F401
    except Exception:
        return False
    return True


def _predictors():
    global _SHARED
    if _SHARED is None:
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        foundation = FoundationPredictor()
        recognizer = RecognitionPredictor(foundation)
        recognizer.disable_tqdm = True
        detector = DetectionPredictor()
        detector.disable_tqdm = True
        _SHARED = (recognizer, detector)
    return _SHARED


class SuryaRecognizer:
    def __init__(self, batch_size: int = DEFAULT_BATCH) -> None:
        self._batch_size = batch_size

    def __call__(self, images: Sequence[np.ndarray]) -> list[OcrResult]:
        if not images:
            return []
        from PIL import Image

        recognizer, detector = _predictors()
        pages = [Image.fromarray(cv2.cvtColor(prepare(gray), cv2.COLOR_GRAY2RGB)) for gray in images]
        started = time.time()
        predictions = _recognize(recognizer, detector, pages, self._batch_size)
        elapsed = (time.time() - started) / max(1, len(images))

        results: list[OcrResult] = []
        for prediction in predictions:
            lines = [line.text.strip() for line in prediction.text_lines if line.text.strip()]
            scores = [float(getattr(line, "confidence", 0.0) or 0.0) for line in prediction.text_lines]
            results.append(
                OcrResult(lines=lines, confidence=float(np.mean(scores)) if scores else 0.0, seconds=elapsed)
            )
        return results


def _recognize(recognizer, detector, pages, batch: int):
    """Партия с откатом по видеопамяти."""
    import torch

    while True:
        try:
            return recognizer(pages, det_predictor=detector, math_mode=False, recognition_batch_size=batch)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= BATCH_MIN:
                raise
            batch = max(BATCH_MIN, batch // 2)


ALGORITHM = Engine(
    name="surya",
    summary="surya 0.17: детектор строк + распознаватель (GPU)",
    stage="gpu",
    make=SuryaRecognizer,
    available=surya_available,
)
