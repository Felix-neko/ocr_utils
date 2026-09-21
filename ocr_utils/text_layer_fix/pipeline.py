"""Одна страница от растра до вердиктов и чтения зон; результат сериализуется в JSON кэша.

Порядок: растр и слой → таблицы детектора → сетка и ориентация ячеек → зоны боковых
ячеек и прямых ячеек без слоя → боковой текст вне таблиц (подписи на картинках FineReader
и схемах детектора — ``LINE_ART_LABEL``, остальное — ``STANDALONE``) → вердикты по словам
слоя → чтение зон tesseract-ом. GPU здесь не зовётся: второе мнение surya, если оно
включено, идёт в родительском процессе по вырезкам из кэша (``second_opinion``).
"""

from __future__ import annotations

import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

import fitz
import numpy as np
import pikepdf

from ocr_utils.rotated_text.tables.ocr import LANGUAGES
from ocr_utils.page_layout.geometry import KIND_TABLE, Box

from ocr_utils.text_layer_fix import VERSION, WORK_DPI
from ocr_utils.text_layer_fix.classify import VerdictCounts, WordVerdict, classify_words
from ocr_utils.page_layout.rotated_text.docstrum import glyph_components
from ocr_utils.text_layer_fix.ocr import ZoneText, read_zone
from ocr_utils.text_layer_fix.raster import downscale, page_raster, render_gray
from ocr_utils.text_layer_fix.text_layer import TextLayer, load_layer
from ocr_utils.text_layer_fix.zones import RotatedZone, TableInfo, ZoneKind, analyse_table, side_zones, upright_zones

# Прямая ячейка считается «с краской», если внутри неё не меньше стольких компонент размера глифа.
MIN_GLYPHS_FOR_MISSING = 3


@dataclass
class Options:
    """Параметры разбора страницы."""

    allowed: tuple[int, ...] = (0, 90, 180, 270)
    lang: str = LANGUAGES
    read_zones: bool = True
    # Искать боковой текст вне таблиц (docstrum): подписи на схемах и отдельный текст.
    free_text: bool = True
    # Какой это PDF FineReader — с коррекцией геометрии или без: вариант картинки в кэше surya.
    variant: str = "fr_geo"
    # Источник surya для разбора ``page_layout`` (кэш только для чтения в воркере); ``None`` —
    # разбор без surya (детектор таблиц + связные пятна).
    layout: "SuryaSourceConfig | None" = None  # ocr_utils.page_layout.surya.SuryaSourceConfig
    # Словарная проверка для слов в непрочитанных зонах (``classify_words``); ставится воркером.
    known: "Callable[[str], bool] | None" = None


@dataclass
class PageResult:
    """Всё, что известно о странице после разбора."""

    pdf: str
    page: int
    raster_dpi: float = 0.0
    raster_size: tuple[int, int] = (0, 0)
    figures: list[tuple[int, int, int, int]] = field(default_factory=list)
    tables: list[TableInfo] = field(default_factory=list)
    zones: list[RotatedZone] = field(default_factory=list)
    words: list[WordVerdict] = field(default_factory=list)
    readings: dict[int, ZoneText] = field(default_factory=dict)
    layer_words: int = 0
    unmatched: int = 0
    offpage: int = 0
    seconds: float = 0.0
    error: str = ""

    def to_json(self) -> dict:
        return {
            "version": VERSION,
            "pdf": self.pdf,
            "page": self.page,
            "raster_dpi": round(self.raster_dpi, 3),
            "raster_size": list(self.raster_size),
            "figures": [list(f) for f in self.figures],
            "tables": [t.to_json() for t in self.tables],
            "zones": [z.to_json() for z in self.zones],
            "words": [w.to_json() for w in self.words],
            "readings": {str(k): v.to_json() for k, v in self.readings.items()},
            "layer_words": self.layer_words,
            "unmatched": self.unmatched,
            "offpage": self.offpage,
            "seconds": round(self.seconds, 2),
            "error": self.error,
        }

    def counts(self) -> dict[str, int]:
        return VerdictCounts.of(self.words).counts


def _inside(box: Box, container: Box, slack: int) -> bool:
    return (
        box.x0 >= container.x0 - slack
        and box.y0 >= container.y0 - slack
        and box.x1 <= container.x1 + slack
        and box.y1 <= container.y1 + slack
    )


def missing_upright_cells(
    gray: np.ndarray, tables: list[TableInfo], layer: TextLayer, px: fitz.Matrix, dpi: float
) -> list[RotatedZone]:
    """Прямые ячейки с краской, в которых слой пуст: FineReader выбросил текст вместе с таблицей.

    Args:
        gray: Растр страницы.
        tables: Разобранные таблицы.
        layer: Слой страницы.
        px: Матрица «pt → пиксели».
        dpi: Разрешение растра.

    Returns:
        Зоны вида ``TABLE_CELL_UPRIGHT`` с поворотом 0.
    """
    centers = [((w.bbox * px).tl + (w.bbox * px).br) / 2 for w in layer.words if w.text.strip()]
    zones: list[RotatedZone] = []
    factor = dpi / WORK_DPI
    for table in tables:
        for cell in table.cells:
            if cell.rotate_cw not in (0, None) or cell.axis_sideways or cell.inner.width <= 0 or cell.inner.height <= 0:
                continue
            if any(cell.inner.x0 <= c.x <= cell.inner.x1 and cell.inner.y0 <= c.y <= cell.inner.y1 for c in centers):
                continue
            crop = gray[cell.inner.slice]
            if crop.size == 0:
                continue
            stats = glyph_components(downscale(crop, factor), WORK_DPI)
            if len(stats) < MIN_GLYPHS_FOR_MISSING:
                continue
            zones.append(
                RotatedZone(
                    cell.inner,
                    ZoneKind.TABLE_CELL_UPRIGHT,
                    0,
                    1.0,
                    table.index,
                    cell.key,
                    note="прямая ячейка без слоя",
                )
            )
    return zones


def page_structure(doc: fitz.Document, index: int, gray: np.ndarray, dpi: int, options: Options):
    """Разбор структуры страницы (``page_layout``) по растру PDF: таблицы, line art, повёрнутые цепочки.

    Args:
        doc: Документ PyMuPDF.
        index: Номер страницы с нуля.
        gray: Растр страницы (по нему считают детекторы; его же в родном разрешении отдаёт PDF).
        dpi: Разрешение растра.
        options: ``variant`` и ``layout`` — вариант картинки и источник surya.

    Returns:
        Готовый :class:`~ocr_utils.page_layout.analysis.PageLayout`.
    """
    from ocr_utils.page_layout.analysis import Find, LayoutOptions, PageLayout
    from ocr_utils.page_layout.image import PageImage, SourceStat, Variant

    path = Path(doc.name) if doc.name else None
    source = SourceStat.of(path, index) if path is not None and path.is_file() else None
    cache_name = f"{path.stem}/p{index:04d}" if path is not None else None
    image = PageImage.from_array(gray, dpi, Variant(options.variant), cache_name, source)
    surya = options.layout.open() if options.layout is not None else None
    layout_options = LayoutOptions(use_surya=surya is not None)
    return PageLayout(image, {Find.TABLES, Find.LINE_ART, Find.ROTATED_TEXT}, layout_options).process(surya)


def process_page(doc: fitz.Document, pdf: "pikepdf.Pdf | None", index: int, options: Options) -> PageResult:
    """Полный разбор одной страницы.

    Args:
        doc: Документ PyMuPDF (только чтение).
        pdf: Тот же документ в pikepdf (шрифты, дерево структуры); None — без них.
        index: Номер страницы с нуля.
        options: Параметры.

    Returns:
        Результат; при исключении — с заполненным ``error`` и тем, что успело посчитаться.
    """
    started = time.time()
    result = PageResult(doc.name.rsplit("/", 1)[-1], index)
    try:
        page = doc[index]
        raster = page_raster(page)
        result.raster_dpi, result.raster_size = raster.dpi, (raster.width, raster.height)
        px = raster.to_px()
        gray = render_gray(page, raster)
        height, width = gray.shape[:2]
        figures = [Box(*[int(round(v)) for v in (f * px)]).clipped(width, height) for f in raster.figures]
        result.figures = [f.as_tuple() for f in figures]
        layer = load_layer(page, pdf)
        result.layer_words = sum(1 for w in layer.words if w.text.strip())
        result.unmatched, result.offpage = layer.unmatched_glyphs, layer.offpage_glyphs

        # Структура страницы — тем же разбором, что на detect и в детекторе порчи геометрии:
        # таблицы, line art (схемы детектора таблиц + surya + пятна) и цепочки повёрнутого текста.
        layout = page_structure(doc, index, gray, int(round(raster.dpi)), options)
        found = layout.table_findings()
        zones: list[RotatedZone] = []
        art_boxes: list[Box] = list(figures) + [r.box for r in layout.line_arts]
        for i, table in enumerate(found):
            if table.kind == KIND_TABLE:
                info, cell_zones = analyse_table(gray, i, table, int(round(raster.dpi)), options.allowed, options.lang)
                result.tables.append(info)
                zones.extend(cell_zones)
        for region in layout.line_arts:
            result.tables.append(TableInfo(len(result.tables), region.box, str(region.info.get("kind", "рисунок"))))
        zones.extend(missing_upright_cells(gray, result.tables, layer, px, raster.dpi))
        if options.free_text:
            chains = [r.box for r in layout.rotated_text_not_in_tables_regions]
            for zone in side_zones(gray, chains, ZoneKind.STANDALONE, int(round(raster.dpi)), options.lang):
                if any(_inside(zone.box, art, max(zone.box.width, zone.box.height)) for art in art_boxes):
                    zone.kind = ZoneKind.LINE_ART_LABEL
                zones.append(zone)
        if options.free_text:
            # Прямой текст без слоя вне таблиц: на картинках FineReader слоя нет вовсе, на схемах
            # детектора и просто на странице — там, где слова слоя рамку цепочки не накрывают.
            layer_boxes = [Box(*[int(round(v)) for v in (w.bbox * px)]) for w in layer.words if w.text.strip()]
            taken = [t.box for t in result.tables if t.kind == KIND_TABLE] + [z.box for z in zones]
            zones.extend(upright_zones(gray, taken, layer_boxes, int(round(raster.dpi)), art_boxes))
        result.zones = zones
        if options.read_zones:
            # Чтение — до вердиктов: прямое чтение лучше повёрнутого снимает зону с ячейки
            # (широкие «ШТ.» по форме «лежат»), и её слова остаются в слое.
            for i, zone in enumerate(zones):
                compare = zone.kind in (ZoneKind.TABLE_CELL, ZoneKind.TABLE_CELL_MIXED)
                reading = read_zone(gray, zone, int(round(raster.dpi)), options.lang, compare_upright=compare)
                result.readings[i] = reading
                if compare and reading.rotate_cw == 0:
                    zone.rotate_cw = 0
                    zone.note = (zone.note + "; " if zone.note else "") + "снята: прямой текст читается лучше"
        readable = {i for i, r in result.readings.items() if r.accepted} if options.read_zones else None
        result.words = classify_words(layer, zones, px, raster.dpi, readable, options.known)
    except Exception as error:  # noqa: BLE001 — страница не должна валить прогон
        result.error = f"{type(error).__name__}: {error}"
    result.seconds = time.time() - started
    return result
