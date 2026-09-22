"""Разбор одной страницы: рендер → строки движка → оси → колонки → блоки с огибающими и выключкой."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.curved_layout import RENDER_DPI, WORK_DPI
from ocr_utils.curved_layout.alignment import Alignment, alignment_of
from ocr_utils.curved_layout.blocks import COARSE_FACTOR, SMOOTH_PITCHES, TextBlock, blocks_of
from ocr_utils.curved_layout.engines.base import Engine
from ocr_utils.curved_layout.lines import SMOOTH_HEIGHTS, LineAxis, axes_of, with_column
from ocr_utils.page_layout.orientation.detectors.ink_axis import glyph_mask
from ocr_utils.curved_layout.columns import mark_cut_lines, zones_of


class Variant(str, Enum):
    """Какой из двух рендеров FineReader разбираем."""

    GEO = "geo"  # с коррекцией геометрии
    NOGEO = "nogeo"  # без коррекции


@dataclass(frozen=True)
class PageAnalysis:
    """Результат разбора страницы: оси строк, блоки-колонки и вердикты выключки."""

    name: str  # имя PDF без расширения
    page: int  # номер страницы с единицы
    variant: str
    engine: str
    width: int  # размеры рабочей копии
    height: int
    dpi: float
    axes: tuple[LineAxis, ...]
    blocks: tuple[TextBlock, ...]
    alignments: tuple[Alignment, ...]
    gutters: tuple  # локальные межколонники (``columns.Gutter``)
    zones: tuple  # зоны вёрстки (``columns.Zone``)
    seconds: float
    note: str = ""

    @property
    def key(self) -> str:
        return f"{self.name}_p{self.page:03d}_{self.variant}_{self.engine}"


def analyse_gray(
    gray300: np.ndarray,
    engine: Engine,
    dpi: float = WORK_DPI,
    smooth_line: float = SMOOTH_HEIGHTS,
    smooth_block: float = SMOOTH_PITCHES,
    coarse_factor: float = COARSE_FACTOR,
    name: str = "page",
    page: int = 1,
    variant: str = Variant.NOGEO.value,
) -> PageAnalysis:
    """Разбор страницы по серому рендеру ``RENDER_DPI``.

    Args:
        gray300: Серый рендер страницы в ``RENDER_DPI``.
        engine: Поставщик строк (``engines.ink.InkEngine`` или адаптер чужого движка).
        dpi: Разрешение рабочей копии, в котором отдаются все координаты.
        smooth_line: Окно сглаживания оси строки в долях её высоты.
        smooth_block: Окно огибающей блока в межстрочных интервалах.
        coarse_factor: Во сколько раз шире окно крупной огибающей.
        name, page, variant: Чем подписать результат.

    Returns:
        :class:`PageAnalysis` со всеми кривыми в пикселях рабочей копии.
    """
    started = time.monotonic()
    result = engine.segment(gray300, dpi)
    axes = axes_of(result.lines, dpi, smooth_line)
    width = int(round(gray300.shape[1] * dpi / RENDER_DPI))
    height = int(round(gray300.shape[0] * dpi / RENDER_DPI))
    # Куски заголовка во всю ширину, набранные через межколонник, помечаются до сборки блоков.
    cut = mark_cut_lines(axes, result.gutters, dpi)
    axes = [with_column(axis, axis.column, cross=flag) for axis, flag in zip(axes, cut)]
    ink = text_ink(gray300, dpi)
    zones = zones_of(result.gutters, height, width, dpi)
    blocks = blocks_of(axes, zones, result.gutters, width, ink, dpi, smooth_block, coarse_factor)
    alignments = [alignment_of(block) for block in blocks]
    return PageAnalysis(
        name=name,
        page=page,
        variant=variant,
        engine=result.engine or getattr(engine, "name", "?"),
        width=width,
        height=height,
        dpi=float(dpi),
        axes=tuple(axes),
        blocks=tuple(blocks),
        alignments=tuple(alignments),
        gutters=tuple(result.gutters),
        zones=tuple(zones),
        seconds=time.monotonic() - started,
        note=result.note,
    )


def text_ink(gray300: np.ndarray, dpi: float = WORK_DPI) -> np.ndarray:
    """Краска ТЕКСТА рендера: краска, оставшаяся под маской глифов рабочей копии.

    Края рядов ищутся по ней, а не по сырой краске: тень корешка и рамка кадра — сплошные
    тёмные полосы у поля страницы, и сырой край ряда цеплялся за них (1975/05 с.97, левая
    колонка). ``glyph_mask`` оставляет только компоненты размером с глиф.
    """
    size = (max(1, round(gray300.shape[1] * dpi / RENDER_DPI)), max(1, round(gray300.shape[0] * dpi / RENDER_DPI)))
    work = cv2.resize(gray300, size, interpolation=cv2.INTER_AREA)
    glyphs = cv2.resize(glyph_mask(work), (gray300.shape[1], gray300.shape[0]), interpolation=cv2.INTER_NEAREST)
    return (gray300 < 128) & (glyphs > 0)


def render_page(pdf: Path, page: int) -> np.ndarray:
    """Серый рендер страницы ``page`` (с единицы) из PDF в ``RENDER_DPI``."""
    import fitz

    from ocr_utils.geometry_regression.render import render_gray

    with fitz.open(pdf) as document:
        return render_gray(document, page - 1)


def analyse_pdf_page(
    pdf: Path, page: int, engine: Engine, variant: str = Variant.NOGEO.value, **options
) -> PageAnalysis:
    """Разбор страницы PDF: рендер плюс :func:`analyse_gray`."""
    gray300 = render_page(pdf, page)
    return analyse_gray(gray300, engine, name=pdf.stem, page=page, variant=variant, **options)


__all__ = ["PageAnalysis", "Variant", "analyse_gray", "analyse_pdf_page", "render_page", "text_ink"]
