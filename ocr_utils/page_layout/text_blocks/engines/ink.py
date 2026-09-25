"""Строки по краске: свои межколонники, сегментация сгустков и центр масс краски вдоль строки."""

from __future__ import annotations

import time

import cv2
import numpy as np

from ocr_utils.page_layout.text_blocks import LINKING_DEFAULT, RENDER_DPI, WORK_DPI
from ocr_utils.page_layout.text_blocks.columns import gutters_of, rule_separators, separators_for_segmentation
from ocr_utils.page_layout.text_blocks.engines.base import EngineLine, EngineResult
from ocr_utils.page_layout.text_blocks.hints import LayoutHints, barrier_rules, barrier_separators, masked_ink
from ocr_utils.page_layout.text_blocks.leaders import leaders_of
from ocr_utils.page_layout.text_blocks.segment import segments_of


class InkEngine:
    """Свой ход: строки и центр-линии по краске рендера, без моделей и GPU.

    Порядок важен: сначала межколонники (:mod:`ocr_utils.page_layout.text_blocks.columns`), потом
    сегментация с ними (:mod:`ocr_utils.page_layout.text_blocks.segment`) — иначе сборка кусков строки
    сшивает две колонки через межколонник шириной 3–4 мм.
    """

    name = "ink"

    def __init__(self, linking: str = LINKING_DEFAULT, hints: LayoutHints | None = None) -> None:
        """Args:
        linking: Способ сцепки кусков в строки — ``zones`` или ``greedy`` (см. пакет ``text_blocks``).
        hints: Вспомогательная информация внешних детекторов (``hints.LayoutHints``): маска
            разрешённого текста и рамки таблиц и блок-схем. ``None`` — разбор как прежде.
        """
        self.linking = linking
        self.hints = hints or LayoutHints()

    def segment(self, gray300: np.ndarray, dpi: float = WORK_DPI) -> EngineResult:
        """Строки страницы в пикселях рабочей копии ``dpi``.

        Args:
            gray300: Серый рендер страницы в ``RENDER_DPI``.
            dpi: Разрешение рабочей копии, в котором нужен результат.

        Returns:
            :class:`EngineResult`: строки с центр-линиями (``baseline=False``) и межколонники.
        """
        started = time.monotonic()
        # Краска вне разрешённой маски гасится ДО всего: тогда ни межколонников, ни строк, ни
        # блоков в запретной области не возникает — гасить их потом пришлось бы в четырёх местах.
        gray300 = masked_ink(gray300, self.hints.text_allowed)
        size = (max(1, round(gray300.shape[1] * dpi / RENDER_DPI)), max(1, round(gray300.shape[0] * dpi / RENDER_DPI)))
        work = cv2.resize(gray300, size, interpolation=cv2.INTER_AREA)
        # Отточия ищутся до межколонников: поле точек иначе принимается за межколонник и режет
        # таблицу на колонки (1971/10 с.93).
        leaders, _ = leaders_of(work, dpi)
        gutters = gutters_of(work, dpi, leaders)
        # Границей строки служат и пустые межколонники, и вертикальные линейки таблицы.
        # Рёбра рамок таблиц и блок-схем — такие же запреты: вертикальные ложатся к
        # межколонникам, горизонтальные к чертам, по которым делится блок.
        separators = (
            separators_for_segmentation(gutters) + rule_separators(work, dpi) + barrier_separators(self.hints.barriers)
        )
        segments, rules = segments_of(gray300, separators, dpi, leaders, self.linking)
        rules = list(rules) + barrier_rules(self.hints.barriers)
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
