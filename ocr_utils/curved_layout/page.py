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
from ocr_utils.curved_layout.blocks import COARSE_FACTOR, DILATE_GLYPHS, SMOOTH_PITCHES, TextBlock, blocks_of
from ocr_utils.curved_layout.engines.base import Engine
from ocr_utils.curved_layout.lines import SMOOTH_HEIGHTS, LineAxis, axes_of, with_column
from ocr_utils.page_layout.orientation.detectors.ink_axis import glyph_mask
from ocr_utils.curved_layout.columns import gutters_of, mark_cut_lines, zones_of
from ocr_utils.curved_layout.leaders import leaders_mask, leaders_of

# Полоса строки для меры покрытия краски: ось ± столько её высот.
BAND_HEIGHTS = 0.6


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
    ink_share: float = 0.0  # доля краски текста, накрытая полосами найденных строк
    leaders: tuple = ()  # отточия страницы (``leaders.Leader``)
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
    dilate: float = DILATE_GLYPHS,
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
        dilate: На какую долю размера символа раздувать границу блока.
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
    work = _work_copy(gray300, dpi)
    # Отточия считаются один раз на разбор: они нужны и колонкам (поле точек — не межколонник), и
    # краске текста (точка мельче порога маски глифов), и рядам блока (их не берут в меры набора).
    page_leaders = list(result.leaders) if result.leaders else leaders_of(work, dpi)[0]
    gutters = result.gutters or gutters_of(work, dpi, page_leaders)
    ink = text_ink(gray300, dpi, work=work, leaders=page_leaders)
    cut = mark_cut_lines(axes, gutters, dpi, ink=gray300 < 128, k=RENDER_DPI / dpi)
    axes = [with_column(axis, axis.column, cross=flag) for axis, flag in zip(axes, cut)]
    # Межколонники — свойство страницы, а не движка: чужие сегментаторы их не отдают, и без них
    # три колонки заметки слипались в один блок (1975/05 с.97, pero). Считаем сами по рендеру.
    zones = zones_of(gutters, height, width, dpi)
    blocks = blocks_of(
        axes, zones, gutters, width, ink, result.rules, dpi, smooth_block, coarse_factor, dilate, page_leaders
    )
    alignments = [alignment_of(block) for block in blocks]
    share = ink_share(axes, ink, dpi)
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
        gutters=tuple(gutters),
        leaders=tuple(page_leaders),
        zones=tuple(zones),
        seconds=time.monotonic() - started,
        ink_share=share,
        note=result.note,
    )


def _work_copy(gray300: np.ndarray, dpi: float) -> np.ndarray:
    """Рабочая копия рендера в разрешении ``dpi``."""
    size = (max(1, round(gray300.shape[1] * dpi / RENDER_DPI)), max(1, round(gray300.shape[0] * dpi / RENDER_DPI)))
    return cv2.resize(gray300, size, interpolation=cv2.INTER_AREA)


def ink_share(axes: list[LineAxis], ink: np.ndarray, dpi: float) -> float:
    """Доля краски текста, накрытая полосами найденных строк.

    Мера полноты сегментации, одинаковая для всех движков: полоса строки — её ось плюс-минус
    ``BAND_HEIGHTS`` высоты. Чем меньше доля, тем больше текста движок не увидел.

    Args:
        axes: Оси строк страницы (пиксели рабочей копии).
        ink: Краска текста рендера ``RENDER_DPI``.
        dpi: Разрешение рабочей копии.

    Returns:
        Доля от 0 до 1; 0, если краски нет.
    """
    total = int(ink.sum())
    if total == 0:
        return 0.0
    k = RENDER_DPI / dpi
    covered = np.zeros(ink.shape, dtype=np.uint8)
    for axis in axes:
        half = max(1.0, BAND_HEIGHTS * axis.height) * k
        points = np.asarray(axis.points, dtype=np.float64) * k
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            cv2.line(covered, (int(x0), int(y0)), (int(x1), int(y1)), 1, thickness=int(round(2 * half)))
    return float((ink & (covered > 0)).sum() / total)


def text_ink(
    gray300: np.ndarray, dpi: float = WORK_DPI, work: np.ndarray | None = None, leaders: list | None = None
) -> np.ndarray:
    """Краска ТЕКСТА рендера: краска под маской глифов рабочей копии плюс точки отточий.

    Края рядов ищутся по ней, а не по сырой краске: тень корешка и рамка кадра — сплошные
    тёмные полосы у поля страницы, и сырой край ряда цеплялся за них (1975/05 с.97, левая
    колонка). ``glyph_mask`` оставляет только компоненты размером с глиф.
    """
    work = _work_copy(gray300, dpi) if work is None else work
    mask = glyph_mask(work)
    # Точки отточий (0.5 мм) ниже нижнего порога маски глифов (5 px), и без них край ряда
    # останавливается на последнем слове, а поле точек остаётся вне блока (1971/10 с.93).
    # Берутся не любые точки, а только собранные в цепочки — пыль краем ряда не станет.
    dots = leaders_mask(work, dpi, leaders) if leaders is not None else leaders_of(work, dpi)[1]
    mask = cv2.max(mask, dots)
    glyphs = cv2.resize(mask, (gray300.shape[1], gray300.shape[0]), interpolation=cv2.INTER_NEAREST)
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


__all__ = ["PageAnalysis", "Variant", "analyse_gray", "analyse_pdf_page", "ink_share", "render_page", "text_ink"]
