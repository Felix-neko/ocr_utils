"""Строки по краске: свои межколонники, сегментация сгустков и центр масс краски вдоль строки."""

from __future__ import annotations

import time

import cv2
import numpy as np

from ocr_utils.curved_layout import LINKING_DEFAULT, RENDER_DPI, WORK_DPI
from ocr_utils.curved_layout.columns import gutters_of, rule_separators, separators_for_segmentation
from ocr_utils.curved_layout.engines.base import EngineLine, EngineResult
from ocr_utils.curved_layout.leaders import leaders_of
from ocr_utils.curved_layout.segment import segments_of


class InkEngine:
    """Свой ход: строки и центр-линии по краске рендера, без моделей и GPU.

    Порядок важен: сначала межколонники (:mod:`ocr_utils.curved_layout.columns`), потом
    сегментация с ними (:mod:`ocr_utils.curved_layout.segment`) — иначе сборка кусков строки
    сшивает две колонки через межколонник шириной 3–4 мм.
    """

    name = "ink"

    def __init__(self, linking: str = LINKING_DEFAULT) -> None:
        """Args:
        linking: Способ сцепки кусков в строки — ``zones`` или ``greedy`` (см. ``curved_layout``).
        """
        self.linking = linking

    def segment(self, gray300: np.ndarray, dpi: float = WORK_DPI) -> EngineResult:
        """Строки страницы в пикселях рабочей копии ``dpi``.

        Args:
            gray300: Серый рендер страницы в ``RENDER_DPI``.
            dpi: Разрешение рабочей копии, в котором нужен результат.

        Returns:
            :class:`EngineResult`: строки с центр-линиями (``baseline=False``) и межколонники.
        """
        started = time.monotonic()
        size = (max(1, round(gray300.shape[1] * dpi / RENDER_DPI)), max(1, round(gray300.shape[0] * dpi / RENDER_DPI)))
        work = cv2.resize(gray300, size, interpolation=cv2.INTER_AREA)
        # Отточия ищутся до межколонников: поле точек иначе принимается за межколонник и режет
        # таблицу на колонки (1971/10 с.93).
        leaders, _ = leaders_of(work, dpi)
        gutters = gutters_of(work, dpi, leaders)
        # Границей строки служат и пустые межколонники, и вертикальные линейки таблицы.
        separators = separators_for_segmentation(gutters) + rule_separators(work, dpi)
        segments, rules = segments_of(gray300, separators, dpi, leaders, self.linking)
        lines = [
            EngineLine(
                points=np.column_stack([segment.xs, segment.ys]),
                polygon=None,
                height=float(segment.height),
                baseline=False,
                mark_spans=segment.mark_spans,
            )
            for segment in segments
        ]
        return EngineResult(
            lines=lines,
            gutters=gutters,
            rules=rules,
            leaders=leaders,
            engine=self.name,
            seconds=time.monotonic() - started,
        )


__all__ = ["InkEngine"]
