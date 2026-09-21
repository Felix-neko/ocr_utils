"""Единственная в репозитории обёртка над surya LayoutPredictor: ленивая загрузка, пачки, парент-процесс.

ПРАВИЛО. GPU-модель живёт в одном процессе — в родителе пула (видеопамять одна на всех, см.
``.claude/rules/gpu_and_pools.md``), грузится лениво при первом обращении (пустой список
страниц не должен стоить загрузки весов) и после создания пула, а не до: CUDA, поднятая до
``fork``, в дочерних процессах виснет. Воркерам surya не нужна — они читают блоки из кэша.

Раньше конструкторов ``LayoutPredictor`` было пять (``background_smoothing``, ``scan_cropping``,
``line_art_detection``, стенд таблиц, скрипт в legacy), и кадр каждому готовили по-своему.
Теперь кадр готовит :class:`~ocr_utils.page_layout.image.PageImage` (``surya_frame``), а
модель зовётся отсюда.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from ocr_utils.page_layout.surya.blocks import LayoutBlocks, from_surya_result

logger = logging.getLogger(__name__)

# Сколько кадров подаётся модели за раз: при 150 dpi кадры мелкие, восемь укладываются в память.
BATCH = 8


class SuryaLayoutModel:
    """Surya LayoutPredictor с ленивой загрузкой; один объект на прогон."""

    def __init__(self, batch: int = BATCH) -> None:
        self._predictor = None
        self.batch = batch

    def _load(self):
        if self._predictor is None:
            from surya.foundation import FoundationPredictor
            from surya.layout import LayoutPredictor
            from surya.settings import settings

            logger.info("Загружаю surya layout")
            predictor = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))
            predictor.disable_tqdm = True  # иначе на каждый кадр рвётся прогресс-бар пачки
            self._predictor = predictor
        return self._predictor

    @property
    def loaded(self) -> bool:
        return self._predictor is not None

    @staticmethod
    def name() -> str:
        """Имя и версия модели — пишутся в кэш, чтобы было видно, чем набит."""
        try:
            from importlib.metadata import version

            return f"surya-ocr {version('surya-ocr')}"
        except Exception:  # noqa: BLE001 — отсутствие метаданных не повод падать
            return "surya-ocr"

    def predict(self, frames: Sequence[np.ndarray]) -> list[LayoutBlocks]:
        """Блоки для каждого кадра ``RGB uint8 (H, W, 3)`` — в пикселях этого же кадра.

        Args:
            frames: Кадры surya (``PageImage.surya_frame``), любое число; идут пачками по ``batch``.

        Returns:
            По :class:`LayoutBlocks` на кадр, в том же порядке.
        """
        if not frames:
            return []
        from PIL import Image as PILImage

        predictor = self._load()
        out: list[LayoutBlocks] = []
        for start in range(0, len(frames), self.batch):
            chunk = frames[start : start + self.batch]
            results = predictor([PILImage.fromarray(np.ascontiguousarray(frame)) for frame in chunk])
            for frame, result in zip(chunk, results):
                height, width = frame.shape[:2]
                out.append(from_surya_result(result, width, height))
        return out

    def predict_one(self, frame: np.ndarray) -> LayoutBlocks:
        return self.predict([frame])[0]

    def close(self) -> None:
        """Отпустить модель и кэш аллокатора CUDA (когда GPU нужен следующему шагу)."""
        self._predictor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — torch может отсутствовать, тогда чистить нечего
            pass


__all__ = ["BATCH", "SuryaLayoutModel"]
