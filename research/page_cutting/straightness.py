"""Прямизна строк на картинке страницы по осям ``page_layout.text_blocks``: наклон, разброс наклонов, прогиб и непрямота осей длинных строк, в ° и мм."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.page_layout.text_blocks import RENDER_DPI
from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
from ocr_utils.page_layout.text_blocks.lines import LineAxis
from ocr_utils.page_layout.text_blocks.page import analyse_gray
from ocr_utils.units import MM_PER_INCH

# Строки короче в меры не идут: у коротких наклон и прогиб шумные (концы абзацев, номера, формулы), мм.
MIN_LINE_MM = 40.0


@dataclass(frozen=True)
class Straightness:
    """Прямизна строк страницы (по длинным строкам).

    Поля:
        lines: Сколько строк длиннее ``MIN_LINE_MM``.
        slope_med_deg: Медиана наклона строк, ° (общий поворот страницы).
        slope_spread_deg: Разброс наклонов: 90-й минус 10-й перцентиль, ° (веер, трапеция, волна по высоте).
        sagitta_p90_mm: 90-й перцентиль |прогиба| строки относительно хорды, мм (дуга).
        bend_p90_mm: 90-й перцентиль размаха оси от прямой, мм (любая непрямота).
        resid_p90_mm: 90-й перцентиль размаха оси от параболы, мм (S-изгиб, волна).
    """

    lines: int
    slope_med_deg: float
    slope_spread_deg: float
    sagitta_p90_mm: float
    bend_p90_mm: float
    resid_p90_mm: float


def long_axes(axes: tuple[LineAxis, ...]) -> list[LineAxis]:
    """Оси строк длиннее ``MIN_LINE_MM``, кроме кусков заголовков, разрезанных межколонником."""
    out = []
    for ax in axes:
        length_mm = (ax.x1 - ax.x0) / ax.dpi * MM_PER_INCH
        if length_mm >= MIN_LINE_MM and not ax.cross:
            out.append(ax)
    return out


def measure_straightness(bgr: np.ndarray, px_per_mm: float) -> Straightness:
    """Прямизна строк картинки страницы.

    Картинка приводится к ``RENDER_DPI`` (пороги детектора — в мм бумаги), оси — движком краски на CPU.

    Args:
        bgr: Картинка страницы (BGR или серая).
        px_per_mm: Плотность картинки, px/мм.

    Returns:
        Меры прямизны (NaN, если длинных строк нет).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    scale = RENDER_DPI / MM_PER_INCH / px_per_mm
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    axes = long_axes(analyse_gray(gray, InkEngine()).axes)
    if not axes:
        nan = float("nan")
        return Straightness(0, nan, nan, nan, nan, nan)
    slope = np.array([a.slope_deg for a in axes])
    sag = np.abs([a.sagitta_mm for a in axes])
    bend = np.array([a.bend_mm for a in axes])
    resid = np.array([a.resid_parabola_mm for a in axes])
    return Straightness(
        lines=len(axes),
        slope_med_deg=round(float(np.median(slope)), 3),
        slope_spread_deg=round(float(np.percentile(slope, 90) - np.percentile(slope, 10)), 3),
        sagitta_p90_mm=round(float(np.percentile(sag, 90)), 3),
        bend_p90_mm=round(float(np.percentile(bend, 90)), 3),
        resid_p90_mm=round(float(np.percentile(resid, 90)), 3),
    )
