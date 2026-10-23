"""Меры края бумаги варианта: рваность и волна наружных сторон страниц (верх, низ, дальняя) по маске бумаги, площадь бумаги."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocr_utils.scan_cropping.v2.page_pose import frame_pages
from ocr_utils.scan_cropping.v2.trapezoid import fit_enclosing_quad
from ocr_utils.scan_cropping.v2.page_sides import Side, side_profiles
from research.page_cutting.selection import side_stats

# Наружные стороны страницы: здесь край верхней страницы сливается с торцами нижних листов. Сторону у корешка не
# мерим — там своя задача (сгиб, research/page_cutting).
OUTER_SIDES = (Side.TOP, Side.BOTTOM, Side.FAR)


@dataclass(frozen=True)
class EdgeMeasures:
    """Меры края бумаги кадра.

    Поля:
        pages: Число страниц.
        ragged_max_mm: Наибольшая рваность наружной стороны (90-й перцентиль отклонения от сглаженного профиля), мм.
        ragged_mean_mm: Средняя рваность наружных сторон, мм.
        wave_far_mm: Наибольшая волна дальней стороны (размах сглаженного профиля), мм.
        wave_tb_mm: Наибольшая волна верха или низа, мм.
        area_mm2: Площадь бумаги, мм².
    """

    pages: int
    ragged_max_mm: float
    ragged_mean_mm: float
    wave_far_mm: float
    wave_tb_mm: float
    area_mm2: float


def edge_measures(
    paper: np.ndarray, book: np.ndarray, hulls: np.ndarray, inst: np.ndarray, exclude: np.ndarray, px_per_mm: float
) -> EdgeMeasures:
    """Меры края бумаги.

    Args:
        paper: Маска бумаги (масштаб сегментации).
        book: Грубая маска книги.
        hulls: Оболочки печати.
        inst: Экземпляры-страницы SAM 3.
        exclude: Зона пальца (точки контура там не мерятся).
        px_per_mm: Плотность масштаба сегментации, px/мм.

    Returns:
        Меры.
    """
    pages = frame_pages(paper, book, hulls, inst)
    two = len(pages) == 2
    ragged, wave_far, wave_tb = [], 0.0, 0.0
    for i, page in enumerate(pages):
        quad = fit_enclosing_quad(page)
        if quad is None:
            continue
        spine_right = (i == 0) if two else None
        for side, profile in side_profiles(page, quad, spine_right, exclude).items():
            if side not in OUTER_SIDES:
                continue
            stats = side_stats(profile, px_per_mm)
            if stats is None:
                continue
            ragged.append(stats[0])
            if side is Side.FAR:
                wave_far = max(wave_far, stats[1])
            else:
                wave_tb = max(wave_tb, stats[1])
    return EdgeMeasures(
        pages=len(pages),
        ragged_max_mm=round(max(ragged, default=0.0), 3),
        ragged_mean_mm=round(float(np.mean(ragged)) if ragged else 0.0, 3),
        wave_far_mm=round(wave_far, 3),
        wave_tb_mm=round(wave_tb, 3),
        area_mm2=round(float(paper.sum()) / px_per_mm**2, 1),
    )
