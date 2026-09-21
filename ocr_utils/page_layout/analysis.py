"""Фасад разбора страницы: один вход, детекторы в правильном порядке, два этапа под пул процессов.

ПОРЯДОК. Surya (кэш или модель) → ориентация → растр (обложка, полосная иллюстрация, фотографии,
подозрения на печать) → таблицы → line art ТОЛЬКО вне растра и таблиц → повёрнутый текст вне
таблиц и растра. Порядок — часть алгоритма: line art внутри фотографии или таблицы не бывает,
и то, что раньше выставлял детектор штриха без исключений, здесь отбрасывается.

ДВА ЭТАПА, как у прежнего ``scan_markup.detection.page``: :meth:`prepare` считает всё
пиксельное и годится для воркера пула (surya там нет — только кэш); если кэш промахнулся,
объект уезжает в родителя с кадром surya (``needs_surya``), родитель зовёт модель и вызывает
:meth:`finish`. При попадании в кэш всё достраивается в воркере — ничего лишнего через pickle
не ездит. :meth:`process` — оба этапа подряд для однопроцессного разбора.

ПОЛНЫЙ КАДР НЕ ЕЗДИТ НИКУДА. После ``prepare`` у картинки остаются только копии рабочего
разрешения (мегабайт-два) и результаты по полному кадру (карты клеток растра, статистика пятен).

РЕЗУЛЬТАТЫ — в родных пикселях картинки (``PageImage.dpi``), см. :class:`~ocr_utils.page_layout.regions.Region`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable

import numpy as np

from ocr_utils.background_smoothing.processing import HALFTONE_DOWNSCALE
from ocr_utils.page_layout import WORK_DPI
from ocr_utils.page_layout.geometry import Box, TableBox
from ocr_utils.page_layout.image import PageImage
from ocr_utils.page_layout.line_art.detector import LineArtInputs, detect_line_art
from ocr_utils.page_layout.orientation.analysis import combine, run_cpu_detectors
from ocr_utils.page_layout.orientation.detectors.base import ROTATIONS, Verdict
from ocr_utils.page_layout.orientation.image_io import frame_from_gray
from ocr_utils.page_layout.raster import cover as cover_module
from ocr_utils.page_layout.raster.boxes import FULL_PAGE_FRAC, MIN_REGION_FRAC, is_full_page
from ocr_utils.page_layout.raster.color_kind import (
    CHROMA_SELF_FRAC_THR,
    CHROMA_SPREAD_THR,
    CHROMA_THR,
    COLOR_FRAC_THR,
    balanced_lab,
    classify,
    paper_color,
)
from ocr_utils.page_layout.raster.dots import ScreenParams, ScreenRegions, params_for_dpi, screen_regions
from ocr_utils.page_layout.raster.regions import (
    FULL_PAGE_COLOR_FRAC,
    GROW_PAPER_MARGIN,
    LEADER_EMPTY_ROWS_THR,
    LEADER_PERIODICITY_THR,
    LEADER_TONE_SPREAD_THR,
    LINEART_FULL_PAGE_INK_FRAC,
    LINEART_MAX_DOT_FRAC,
    LINEART_PICTURE_MIN_FRAC,
    SAFETY_MIN_FRAC,
    SOURCE_LINEART,
    SURYA_LINEART_P99_PX,
    find_raster_boxes,
)
from ocr_utils.page_layout.raster.tone import (
    LINEART_ENTROPY_THR,
    LINEART_MID_FRAC_THR,
    LINEART_SCREEN_PEAK_THR,
    STAMP_INK_CONTRAST_THR,
    ToneMaps,
    tone_maps,
)
from ocr_utils.page_layout.regions import Region, RegionKind
from ocr_utils.page_layout.rotated_text.detector import rotated_zones
from ocr_utils.page_layout.surya.blocks import PICTURE_LABELS, LayoutBlocks
from ocr_utils.page_layout.surya.cache import SuryaCache
from ocr_utils.page_layout.surya.source import SuryaSource
from ocr_utils.page_layout.tables.detector import detect as detect_tables

logger = logging.getLogger(__name__)


class Find(str, Enum):
    """Что искать на странице; по умолчанию — всё."""

    RASTER = "raster"
    TABLES = "tables"
    LINE_ART = "line_art"
    ROTATED_TEXT = "rotated_text"
    ORIENTATION = "orientation"


ALL_FINDS: frozenset[Find] = frozenset(Find)


@dataclass(frozen=True)
class LayoutOptions:
    """Пороги и переключатели разбора. Обязаны переживать pickle (уезжают в воркер)."""

    # Surya: искать ли блоки вообще. Без неё детекторы работают по одним пикселям.
    use_surya: bool = True
    # Первая полоса выпуска — обложка: ответ известен заранее, пиксели не читаются.
    first_page_is_cover: bool = False

    # --- Растр (пороги как в scan_markup detect; замеры — в модулях raster.*) ---------------
    chroma_thr: float = CHROMA_THR
    color_frac_thr: float = COLOR_FRAC_THR
    chroma_spread_thr: float = CHROMA_SPREAD_THR
    chroma_self_frac_thr: float | None = CHROMA_SELF_FRAC_THR
    min_region_frac: float = MIN_REGION_FRAC
    full_page_frac: float = FULL_PAGE_FRAC
    lineart_ink_frac: float = LINEART_FULL_PAGE_INK_FRAC
    lineart_p99: int = SURYA_LINEART_P99_PX
    safety_min_frac: float = SAFETY_MIN_FRAC
    lineart_picture_min_frac: float = LINEART_PICTURE_MIN_FRAC
    full_page_color_frac: float = FULL_PAGE_COLOR_FRAC
    leader_empty_rows_thr: float = LEADER_EMPTY_ROWS_THR
    leader_periodicity_thr: float = LEADER_PERIODICITY_THR
    leader_tone_spread_thr: float = LEADER_TONE_SPREAD_THR
    grow_paper_margin: int = GROW_PAPER_MARGIN
    lineart_mid_frac: float = LINEART_MID_FRAC_THR
    lineart_entropy: float = LINEART_ENTROPY_THR
    lineart_screen_peak: float = LINEART_SCREEN_PEAK_THR
    stamp_ink_contrast: float = STAMP_INK_CONTRAST_THR
    lineart_max_dot_frac: float = LINEART_MAX_DOT_FRAC
    # None — пересчитанное от DPI (см. raster.dots.params_for_dpi).
    merge_gap: int | None = None
    min_region_side_px: int | None = None
    cell_px: int | None = None
    dot_frac_thr: float | None = None
    dot_max_area_px: int | None = None
    cell_max_area_px: int | None = None
    min_dots_per_cell: int | None = None
    min_cells: int | None = None

    # --- Ориентация -------------------------------------------------------------------------
    # CPU-детекторы, которые гоняются в prepare (GPU-детекторы зовёт родитель и отдаёт в finish).
    orientation_detectors: tuple[str, ...] = ()
    allowed_rotations: tuple[int, ...] = ROTATIONS

    def screen_params(self, dpi: int) -> ScreenParams:
        return params_for_dpi(
            dpi,
            cell_px=self.cell_px,
            dot_frac_thr=self.dot_frac_thr,
            dot_max_area_px=self.dot_max_area_px,
            cell_max_area_px=self.cell_max_area_px,
            min_dots_per_cell=self.min_dots_per_cell,
            min_cells=self.min_cells,
        )


@dataclass
class RasterPrep:
    """Пиксельная часть растра по ПОЛНОМУ кадру — считается в prepare, живёт до finish."""

    regions: ScreenRegions
    stats: np.ndarray
    centroids: np.ndarray
    tone: ToneMaps
    params: ScreenParams


class PageLayout:
    """Разбор одной страницы. Создать, ``process`` (или ``prepare`` + ``finish``), читать поля."""

    def __init__(
        self,
        image: PageImage,
        find: Iterable[Find] | None = None,
        options: LayoutOptions = LayoutOptions(),
        order_index: int | None = None,
    ) -> None:
        """
        Args:
            image: Страница (см. :class:`PageImage`).
            find: Что искать; ``None`` — всё.
            options: Пороги и переключатели.
            order_index: Номер полосы в выпуске с нуля — нужен правилу обложки; ``None`` — не первая.
        """
        self.image = image
        self.find: frozenset[Find] = frozenset(find) if find is not None else ALL_FINDS
        self.options = options
        self.order_index = order_index

        self.blocks: LayoutBlocks | None = None  # surya в пикселях кадра surya
        self.surya_used = False
        self.needs_surya = False
        self.prepared = False
        self.finished = False
        self.is_cover = False

        self.raster_pics: list[Region] = []
        self.stamp_suspects: list[Region] = []
        self.tables: list[Region] = []
        self.line_arts: list[Region] = []
        self.rotated_text_not_in_tables_regions: list[Region] = []
        self.best_page_orientation: Verdict | None = None
        self.orientation_verdicts: dict[str, Verdict] = {}

        self._raster: RasterPrep | None = None
        self._table_boxes: list[TableBox] = []  # находки детектора таблиц в пикселях копии
        self._cpu_orientation: dict[str, Verdict] = {}
        self._orientation_done = False

    # --- Свойства ------------------------------------------------------------------------

    @property
    def raw_surya_content(self) -> LayoutBlocks | None:
        """Блоки surya в РОДНЫХ пикселях картинки (``None`` — surya не использовалась)."""
        if self.blocks is None:
            return None
        return self.blocks.scaled_to(self.image.width, self.image.height)

    @property
    def regions(self) -> list[Region]:
        """Все области одним списком: растр, печати, таблицы, line art, повёрнутый текст."""
        return (
            self.raster_pics
            + self.stamp_suspects
            + self.tables
            + self.line_arts
            + self.rotated_text_not_in_tables_regions
        )

    @property
    def work_dpi(self) -> int:
        return WORK_DPI

    @property
    def raster_dpi(self) -> int:
        """Разрешение копии растрового детектора: строго 1/``HALFTONE_DOWNSCALE`` полного кадра."""
        return max(1, int(round(self.image.dpi / HALFTONE_DOWNSCALE)))

    def orientation_image(self):
        """Картинка под GPU-детектор ориентации (PIL RGB копии рабочего разрешения) или ``None``."""
        if Find.ORIENTATION not in self.find or self.is_cover:
            return None
        from PIL import Image as PILImage

        return PILImage.fromarray(self.image.surya_frame)

    # --- Этап 1: пиксели --------------------------------------------------------------------

    def prepare(self, cache: SuryaCache | None = None) -> "PageLayout":
        """Всё, что считается без GPU. Годится для воркера пула.

        Args:
            cache: Кэш surya (только чтение); ``None`` — surya в этом процессе не искать.

        Returns:
            Себя. Если ``needs_surya`` — блоков нет, ждём :meth:`finish` от родителя; иначе
            разбор уже завершён (``finished``).
        """
        options = self.options
        image = self.image
        if options.first_page_is_cover and self.order_index == 0:
            self._ready_cover()
            return self

        if Find.RASTER in self.find:
            params = options.screen_params(image.dpi)
            gray = image.gray
            regions, stats, centroids = screen_regions(gray, params)
            self._raster = RasterPrep(regions, stats, centroids, tone_maps(gray, params), params)
        if Find.ORIENTATION in self.find and options.orientation_detectors:
            path = Path(image.source.path) if image.source is not None else Path(image.cache_name or "page")
            frame = frame_from_gray(image.gray, image.dpi, image.cache_name or "", path, options.allowed_rotations)
            self._cpu_orientation = run_cpu_detectors(frame, options.orientation_detectors)
        self._orientation_done = Find.ORIENTATION in self.find
        # Копии рабочего разрешения — до того, как полный кадр будет отпущен.
        image.gray_at(self.work_dpi)
        image.bgr_at(self.work_dpi)
        if Find.RASTER in self.find:
            image.gray_at(self.raster_dpi)
            image.bgr_at(self.raster_dpi)
        if Find.LINE_ART in self.find or Find.ROTATED_TEXT in self.find:
            image.bitonal_at(self.work_dpi)
        if options.use_surya:
            image.surya_frame
        image.drop_full_frames()
        self.prepared = True

        if not options.use_surya:
            self._finish_with(None, None)
        elif cache is not None and (blocks := cache.load(image)) is not None:
            self._finish_with(blocks, None)
        else:
            self.needs_surya = True
        return self

    def _ready_cover(self) -> None:
        """Обложка: одна цветная область во весь кадр, остальное пусто, поворот не нужен."""
        self.is_cover = True
        box = Box(*cover_module.cover_region(self.image.width, self.image.height))
        self.raster_pics = [Region(box, RegionKind.COLOR, 1.0, "raster", True, {"cover": True})]
        if Find.ORIENTATION in self.find:
            self.best_page_orientation = Verdict(0, 1.0, note="обложка")
        self.prepared = self.finished = True

    # --- Этап 2: сборка -----------------------------------------------------------------------

    def finish(self, blocks: LayoutBlocks | None, gpu_orientation: dict[str, Verdict] | None = None) -> "PageLayout":
        """Достроить разбор по блокам surya (или без них) и GPU-вердиктам ориентации. В родителе."""
        if not self.prepared:
            raise RuntimeError("finish() до prepare()")
        if self.finished:
            return self
        self._finish_with(blocks, gpu_orientation)
        return self

    def process(
        self, surya: SuryaSource | None = None, gpu_orientation: dict[str, Verdict] | None = None
    ) -> "PageLayout":
        """Оба этапа подряд в одном процессе. ``surya`` — откуда брать блоки (см. :class:`SuryaSource`)."""
        cache = surya.cache if surya is not None else None
        self.prepare(cache)
        if self.finished:
            return self
        blocks = surya.resolve(self.image) if surya is not None and surya.enabled else None
        return self.finish(blocks, gpu_orientation)

    def _finish_with(self, blocks: LayoutBlocks | None, gpu_orientation: dict[str, Verdict] | None) -> None:
        self.blocks = blocks
        self.surya_used = blocks is not None
        self.needs_surya = False
        image = self.image
        work_dpi = self.work_dpi
        work_w, work_h = image.size_at(work_dpi)
        work_blocks = blocks.scaled_to(work_w, work_h) if blocks is not None else None
        native = image.scale_to_native(work_dpi)

        if Find.ORIENTATION in self.find and self._orientation_done:
            verdicts = dict(self._cpu_orientation)
            verdicts.update(gpu_orientation or {})
            self.orientation_verdicts = verdicts
            if verdicts:
                self.best_page_orientation, _source, _disputed = combine(verdicts, self.options.allowed_rotations)

        if Find.RASTER in self.find and self._raster is not None:
            self._finish_raster(blocks)
        raster_work = [r.box.scaled(1.0 / native) for r in self.raster_pics + self.stamp_suspects]

        if Find.TABLES in self.find or Find.LINE_ART in self.find:
            found = detect_tables(image.gray_at(work_dpi), work_dpi, layout=work_blocks)
            self._table_boxes = [t for t in found if not _covered_by(t.box, raster_work, 0.5)]
            if Find.TABLES in self.find:
                self.tables = [_table_region(t, native, image) for t in self._table_boxes if t.kind == TABLE_KIND_RU]
        table_work = [t.box for t in self._table_boxes if t.kind == TABLE_KIND_RU]

        if Find.LINE_ART in self.find:
            inputs = LineArtInputs(
                image.bitonal_at(work_dpi),
                work_dpi,
                raster_work,
                table_work,
                [t for t in self._table_boxes if t.kind != TABLE_KIND_RU],
                work_blocks,
            )
            self.line_arts = [_to_native(r, native, image) for r in detect_line_art(inputs)]

        if Find.ROTATED_TEXT in self.find:
            art_work = [r.box.scaled(1.0 / native) for r in self.line_arts]
            zones = rotated_zones(image.bitonal_at(work_dpi), work_dpi, raster_work + table_work, art_work)
            self.rotated_text_not_in_tables_regions = [_to_native(r, native, image) for r in zones]
        self.finished = True

    def _finish_raster(self, blocks: LayoutBlocks | None) -> None:
        """Растр: находки по клеткам + затравки Picture от surya, затем цвет и печати (как в detect)."""
        image, prep, options = self.image, self._raster, self.options
        raster_dpi = self.raster_dpi
        work = image.bgr_at(raster_dpi)
        work_gray = image.gray_at(raster_dpi)
        surya_boxes = None
        if blocks is not None:
            native_blocks = blocks.scaled_to(image.width, image.height)
            surya_boxes = [b.as_tuple() for b in native_blocks.boxes(PICTURE_LABELS, min_confidence=0.0)]
        paper = paper_color(work)
        page_a, page_b = balanced_lab(work, paper)
        page_chroma_spread = float(np.hypot(page_a.std(), page_b.std()))
        findings = find_raster_boxes(
            prep.regions,
            prep.stats,
            prep.centroids,
            work_gray,
            (image.height, image.width),
            prep.params,
            surya_boxes=surya_boxes,
            order_index=self.order_index if self.order_index is not None else 1,
            first_page_is_cover=options.first_page_is_cover,
            min_region_frac=options.min_region_frac,
            merge_gap=options.merge_gap,
            min_side=options.min_region_side_px,
            lineart_ink_frac=options.lineart_ink_frac,
            lineart_p99=options.lineart_p99,
            safety_min_frac=options.safety_min_frac,
            page_chroma_spread=page_chroma_spread,
            chroma_spread_thr=options.chroma_spread_thr,
            full_page_color_frac=options.full_page_color_frac,
            leader_empty_rows_thr=options.leader_empty_rows_thr,
            leader_periodicity_thr=options.leader_periodicity_thr,
            leader_tone_spread_thr=options.leader_tone_spread_thr,
            grow_paper_margin=options.grow_paper_margin,
            tone=prep.tone,
            lineart_mid_frac=options.lineart_mid_frac,
            lineart_entropy=options.lineart_entropy,
            lineart_screen_peak=options.lineart_screen_peak,
            lineart_max_dot_frac=options.lineart_max_dot_frac,
        )
        if findings.cover:
            self.is_cover = True
        pics, stamps = classify_raster(findings, work, paper, image.width, image.height, options)
        self.raster_pics, self.stamp_suspects = pics, stamps


# Русские виды детектора таблиц: «таблица» против схемы/рисунка.
TABLE_KIND_RU = "таблица"


def _covered_by(box: Box, others: list[Box], share: float) -> bool:
    if not others or box.area <= 0:
        return False
    covered = 0
    for other in others:
        width = min(box.x1, other.x1) - max(box.x0, other.x0)
        height = min(box.y1, other.y1) - max(box.y0, other.y0)
        if width > 0 and height > 0:
            covered += width * height
    return covered >= share * box.area


def _to_native(region: Region, native: float, image: PageImage) -> Region:
    box = region.box.scaled(native).clipped(image.width, image.height)
    return Region(box, region.kind, region.confidence, region.source, region.full_page, region.info)


def _table_region(table: TableBox, native: float, image: PageImage) -> Region:
    """Находка детектора таблиц → область ``TABLE`` в родных пикселях с JSON-подробностями (как раньше в базе)."""
    box = table.box.scaled(native).clipped(image.width, image.height)
    info = {
        "kind": table.kind,
        "score": round(float(table.score), 4),
        "skew_deg": round(float(table.skew_deg), 3),
        "rule_box": list(table.rules.scaled(native).clipped(image.width, image.height).as_tuple()),
        "metrics": {key: round(float(value), 4) for key, value in sorted(table.metrics.items())},
    }
    return Region(box, RegionKind.TABLE, round(float(table.score), 4), "tables", False, info)


def classify_raster(
    findings, work: np.ndarray, paper, width: int, height: int, options: LayoutOptions
) -> tuple[list[Region], list[Region]]:
    """Красит найденные растровые области и отделяет подозрения на печать.

    Правила те же, что были в ``scan_markup.detection.page._classify_regions``:

    * растр серый / цветной → ``grayscale`` / ``color``;
    * штрих цветной и крупный → ``color`` (цветной рисунок берётся целиком);
    * штрих чёрный → в разметку НЕ включается: он бинаризуется как текст (line art ищет свой детектор);
    * мелкий бледный штрих или мелкая цветная область без растровых клеток → ``stamp_suspect``.

    Args:
        findings: Ответ ``find_raster_boxes`` (рамки в родных пикселях).
        work: Цветная копия 1/4 (по ней меряется цвет: разброс хроматичности на ней разделяет так же).
        paper: Цвет бумаги по копии.
        width: Ширина полного кадра.
        height: Высота полного кадра.
        options: Пороги.

    Returns:
        ``(иллюстрации, подозрения на печать)``.
    """
    pics: list[Region] = []
    stamps: list[Region] = []
    if not findings.findings:
        return pics, stamps
    work_h, work_w = work.shape[:2]
    scale = HALFTONE_DOWNSCALE
    for finding in findings.findings:
        x1, y1 = finding.box[0] // scale, finding.box[1] // scale
        x2 = min(max(x1 + 1, -(-finding.box[2] // scale)), work_w)
        y2 = min(max(y1 + 1, -(-finding.box[3] // scale)), work_h)
        color = classify(
            work[y1:y2, x1:x2],
            paper,
            options.chroma_thr,
            options.color_frac_thr,
            options.chroma_spread_thr,
            options.chroma_self_frac_thr,
        )
        kind = RegionKind(color.kind)
        area = (finding.box[2] - finding.box[0]) * (finding.box[3] - finding.box[1])
        small = area < options.lineart_picture_min_frac * width * height
        line_art = finding.source == SOURCE_LINEART
        pale = finding.ink_contrast is not None and finding.ink_contrast < options.stamp_ink_contrast
        stamp = small and ((line_art and pale) or (kind is RegionKind.COLOR and not finding.has_cells))
        if stamp:
            kind = RegionKind.STAMP_SUSPECT
        elif line_art and kind is not RegionKind.COLOR:
            continue  # чёрный штрих — не растр; его найдёт детектор line art
        info = {
            "chroma_frac": color.chroma_frac,
            "chroma_spread": color.chroma_spread,
            "chroma_self_frac": color.chroma_self_frac,
            "dot_frac": finding.dot_frac,
            "mid_frac": finding.mid_frac,
            "tone_entropy": finding.tone_entropy,
            "screen_peak": finding.screen_peak,
            "ink_contrast": finding.ink_contrast,
            "source": finding.source,
        }
        region = Region(
            Box(*finding.box).clipped(width, height),
            kind,
            None,
            "raster",
            finding.full_page or is_full_page(finding.box, width, height, options.full_page_frac),
            info,
        )
        (stamps if kind is RegionKind.STAMP_SUSPECT else pics).append(region)
    return pics, stamps


__all__ = ["ALL_FINDS", "Find", "LayoutOptions", "PageLayout", "classify_raster"]
