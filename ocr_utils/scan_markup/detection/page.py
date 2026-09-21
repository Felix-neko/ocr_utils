"""Детекция по ОДНОЙ полосе на этапе ``detect``: файл на диске → результат, без обращения к базе.

Сам разбор делает ``ocr_utils.page_layout`` — те же детекторы и те же версии, что при сборке
финальных PDF. Здесь только то, что нужно этапу ``detect``: отпечаток файла (по нему
``--skip-detected`` узнаёт, что полоса прежняя), выбор семейств для ЭТОЙ полосы (растр свежий,
нужны одни таблицы), два этапа под пул процессов и упаковка ответа для базы.

ДВА ЭТАПА — как у ``page_layout.PageLayout``: :func:`analyse_page` читает файл и считает всё
пиксельное (в воркере; surya там только из кэша), :func:`finish_page` при промахе кэша зовёт
модель и достраивает разбор (в родителе, где живёт GPU). Между ними едет :class:`PageAnalysis`
с копиями рабочего разрешения; полный кадр не ездит никуда.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ocr_utils.page_layout.analysis import Find, LayoutOptions, PageLayout
from ocr_utils.page_layout.image import PageImage, Variant
from ocr_utils.page_layout.orientation.detectors.base import ROTATIONS, Verdict
from ocr_utils.page_layout.regions import Region
from ocr_utils.page_layout.surya.cache import SuryaCache, scan_cache_name
from ocr_utils.scan_markup.hashing import FileStamp, full_stamp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageOptions:
    """Всё, что нужно детекции по одной полосе. Обязано переживать pickle."""

    default_dpi: int | None = None
    # Отпечаток файла: считать хеш или ограничиться ``stat``.
    need_digest: bool = True

    # --- Что считать на полосе (семейства page_layout) -------------------------------
    raster: bool = True
    tables: bool = True
    line_art: bool = True
    rotated_text: bool = True
    orientation: bool = False
    # Имена CPU-детекторов ориентации, которые гоняются в воркере; GPU-детектор идёт в родителе.
    orientation_detectors: tuple[str, ...] = ()
    allowed_rotations: tuple[int, ...] = ROTATIONS

    # --- Surya ---------------------------------------------------------------------------
    # Зовёт ли родитель surya вовсе. Без неё детекторы работают по одним пикселям.
    use_surya_layout: bool = True
    # Корень кэша surya (``page_layout.surya.SuryaCache``); ``None`` — без кэша.
    layout_cache_dir: Path | None = None

    # Пороги детекторов (растр и др.) и правило обложки.
    layout: LayoutOptions = LayoutOptions()

    def finds(self) -> set[Find]:
        wanted = set()
        if self.raster:
            wanted.add(Find.RASTER)
        if self.tables:
            wanted.add(Find.TABLES)
        if self.line_art:
            wanted.add(Find.LINE_ART)
        if self.rotated_text:
            wanted.add(Find.ROTATED_TEXT)
        if self.orientation:
            wanted.add(Find.ORIENTATION)
        return wanted

    def layout_options(self) -> LayoutOptions:
        """Опции фасада: пороги из ``layout`` плюс переключатели этой полосы."""
        from dataclasses import replace

        return replace(
            self.layout,
            use_surya=self.use_surya_layout,
            orientation_detectors=self.orientation_detectors if self.orientation else (),
            allowed_rotations=self.allowed_rotations,
        )


@dataclass
class PageAnalysis:
    """Разбор после первого этапа. Едет из воркера в родителя; обязан быть небольшим."""

    rel_path: str
    order_index: int
    stamp: FileStamp | None = None
    error: str = ""
    # Файл читали только потому, что разошёлся ``stat``, а содержимое оказалось прежним.
    unchanged: bool = False
    layout: PageLayout | None = None

    @property
    def needs_surya(self) -> bool:
        return self.layout is not None and self.layout.needs_surya

    def orientation_image(self):
        """Картинка под GPU-детектор ориентации или ``None`` (обложка, ошибка, не просили)."""
        return self.layout.orientation_image() if self.layout is not None else None


@dataclass
class PageResult:
    """Результат по одной полосе. ``error`` непуст — значит остальное недостоверно.

    Список семейства равен ``None``, когда его не считали (полосе нужны были другие); это
    НЕ то же самое, что пустой список «искали и не нашли».
    """

    rel_path: str
    width: int = 0
    height: int = 0
    dpi: int = 0
    stamp: FileStamp | None = None
    error: str = ""
    unchanged: bool = False
    raster: list[Region] | None = None  # иллюстрации и подозрения на печать
    tables: list[Region] | None = None
    line_art: list[Region] | None = None
    rotated_text: list[Region] | None = None
    # Ориентация: вердикты по детекторам и сводный. ``combo is None`` значит «не считали».
    orientation: dict[str, Verdict] = field(default_factory=dict)
    combo: Verdict | None = None
    surya_used: bool = False

    @property
    def regions(self) -> list[Region]:
        out: list[Region] = []
        for family in (self.raster, self.tables, self.line_art, self.rotated_text):
            out.extend(family or [])
        return out


def analyse_page(
    path: Path, rel_path: str, order_index: int, options: PageOptions, known_digest: str | None = None
) -> PageAnalysis:
    """Первый этап: отпечаток файла и весь пиксельный разбор. Исключения кладёт в ``error``.

    ``known_digest`` — хеш, записанный у полосы в базе. Совпал с посчитанным сейчас — файл
    просто переписали тем же содержимым, и разбирать его незачем: ``unchanged``.
    """
    analysis = PageAnalysis(rel_path, order_index)
    try:
        analysis.stamp = full_stamp(path) if options.need_digest else None
        if known_digest is not None and analysis.stamp is not None and analysis.stamp.digest == known_digest:
            analysis.unchanged = True
            return analysis
        image = PageImage.from_file(path, Variant.SCAN, scan_cache_name(rel_path), options.default_dpi)
        cache = SuryaCache(options.layout_cache_dir, readonly=True) if options.layout_cache_dir is not None else None
        layout = PageLayout(image, options.finds(), options.layout_options(), order_index)
        analysis.layout = layout.prepare(cache if options.use_surya_layout else None)
        return analysis
    except Exception as exc:  # noqa: BLE001 — одна битая полоса не должна валить прогон
        analysis.error = str(exc)
        return analysis


def finish_page(
    analysis: PageAnalysis, options: PageOptions, model=None, gpu_orientation: dict[str, Verdict] | None = None
) -> PageResult:
    """Второй этап: surya при промахе кэша (модель в родителе), сборка ответа для базы."""
    if analysis.error:
        return PageResult(analysis.rel_path, error=analysis.error)
    if analysis.unchanged or analysis.layout is None:
        return PageResult(analysis.rel_path, stamp=analysis.stamp, unchanged=True)
    layout = analysis.layout
    if not layout.finished:
        blocks = None
        if layout.needs_surya:
            if model is None:
                raise RuntimeError(f"{analysis.rel_path}: промах кэша surya, а модели в этом процессе нет")
            blocks = model.predict_one(layout.image.surya_frame)
            if options.layout_cache_dir is not None:
                SuryaCache(options.layout_cache_dir).save(layout.image, blocks, model.name())
        layout.finish(blocks, gpu_orientation)
    finds = layout.find
    return PageResult(
        analysis.rel_path,
        layout.image.width,
        layout.image.height,
        layout.image.dpi,
        analysis.stamp,
        raster=(layout.raster_pics + layout.stamp_suspects) if Find.RASTER in finds else None,
        tables=layout.tables if Find.TABLES in finds else None,
        line_art=layout.line_arts if Find.LINE_ART in finds else None,
        rotated_text=layout.rotated_text_not_in_tables_regions if Find.ROTATED_TEXT in finds else None,
        orientation=dict(layout.orientation_verdicts),
        combo=layout.best_page_orientation,
        surya_used=layout.surya_used,
    )


def detect_page(
    path: Path, rel_path: str, order_index: int, options: PageOptions, model=None, known_digest: str | None = None
) -> PageResult:
    """Оба этапа подряд — для однопроцессного прогона, оснастки валидации и тестов."""
    analysis = analyse_page(path, rel_path, order_index, options, known_digest)
    return finish_page(analysis, options, model)


__all__ = ["PageAnalysis", "PageOptions", "PageResult", "analyse_page", "detect_page", "finish_page"]
