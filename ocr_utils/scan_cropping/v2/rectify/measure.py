"""Меры прямизны текста страницы по разбору ``text_blocks``: размах осей строк по высоте (наклон плюс волна) и размах выровненных сторон блоков по горизонтали — до и после второго преобразования."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocr_utils.page_layout.text_blocks.leaders import inside_spans
from ocr_utils.units import px_to_mm
from research.page_rectify.text_field import MIN_RUN_MM, PageText, line_curve


@dataclass(frozen=True)
class TextMeasures:
    """Сводка прямизны текста кадра.

    Поля:
        lines: Строк длиной от :data:`MIN_RUN_MM`.
        line_med_mm: Медианный размах оси строки по высоте, мм.
        line_p95_mm: 95-й перцентиль того же.
        slope_med_deg: Медианный |наклон| строк по концам, градусы.
        sides: Выровненных сторон блоков.
        side_med_mm: Медианный размах выровненной стороны по горизонтали, мм.
        side_p95_mm: 95-й перцентиль того же.
    """

    lines: int = 0
    line_med_mm: float = float("nan")
    line_p95_mm: float = float("nan")
    slope_med_deg: float = float("nan")
    sides: int = 0
    side_med_mm: float = float("nan")
    side_p95_mm: float = float("nan")

    def row(self, prefix: str) -> dict:
        """Поля для строки сводки с приставкой (``before_``, ``after_``), числа — с двумя знаками."""
        return {f"{prefix}{k}": (round(v, 2) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def line_spans(pages: list[PageText], dpi: float) -> tuple[list[float], list[float]]:
    """Размахи осей строк по высоте (мм) и |наклоны| (градусы) по всем страницам.

    Args:
        pages: Разборы страниц.
        dpi: Разрешение холста.

    Returns:
        Списки размахов и наклонов строк не короче :data:`MIN_RUN_MM`.
    """
    spans, slopes = [], []
    for page in pages:
        for axis in page.analysis.axes:
            pts = line_curve(axis)
            if len(pts) < 2:
                continue
            pts = page.to_canvas(pts)
            if px_to_mm(np.ptp(pts[:, 0]), dpi) < MIN_RUN_MM:
                continue
            spans.append(px_to_mm(np.ptp(pts[:, 1]), dpi))
            # Наклон — прямая МНК по базовой линии на холсте.
            slopes.append(abs(float(np.degrees(np.arctan(np.polyfit(pts[:, 0], pts[:, 1], 1)[0])))))
    return spans, slopes


def side_spans(pages: list[PageText], dpi: float) -> list[float]:
    """Размахи выровненных сторон блоков по горизонтали (мм), без недостоверных участков.

    Args:
        pages: Разборы страниц.
        dpi: Разрешение холста.

    Returns:
        Размахи сторон не короче :data:`MIN_RUN_MM`.
    """
    spans = []
    for page in pages:
        for block, alignment in zip(page.analysis.blocks, page.analysis.alignments):
            envelope = block.envelope
            for stats, core, unreliable in (
                (alignment.left, envelope.core_left, getattr(envelope, "unreliable_left", ())),
                (alignment.right, envelope.core_right, getattr(envelope, "unreliable_right", ())),
            ):
                if not stats.aligned or core is None or len(core) < 2:
                    continue
                pts = np.asarray(core, np.float64)
                if unreliable:
                    pts = pts[~inside_spans(pts[:, 1], list(unreliable))]
                if len(pts) < 2:
                    continue
                pts = page.to_canvas(pts)
                if px_to_mm(np.ptp(pts[:, 1]), dpi) < MIN_RUN_MM:
                    continue
                spans.append(px_to_mm(np.ptp(pts[:, 0]), dpi))
    return spans


def text_measures(pages: list[PageText], dpi: float) -> TextMeasures:
    """Сводка прямизны текста по разборам страниц кадра.

    Args:
        pages: Разборы страниц (без цветных).
        dpi: Разрешение холста.

    Returns:
        Меры; без строк — пустые (NaN).
    """
    spans, slopes = line_spans(pages, dpi)
    sides = side_spans(pages, dpi)
    measures = TextMeasures(lines=len(spans), sides=len(sides))
    if spans:
        measures = TextMeasures(len(spans), float(np.median(spans)), float(np.percentile(spans, 95)),
                                float(np.median(slopes)), len(sides))  # fmt: skip
    if sides:
        measures = TextMeasures(measures.lines, measures.line_med_mm, measures.line_p95_mm, measures.slope_med_deg,
                                len(sides), float(np.median(sides)), float(np.percentile(sides, 95)))  # fmt: skip
    return measures
