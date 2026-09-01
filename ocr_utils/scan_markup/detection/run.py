"""Прогон предварительной детекции по паку: оригиналы -> SQLite.

Один проход по TIFF на полосу. Это дорогая часть конвейера: пак-1 — 12 136 файлов по
~40 МБ на медленном NTFS, то есть около полутерабайта чтения. Поэтому за одно чтение
делается всё, что вообще можно сделать из пикселей: размеры, DPI в базу, полутоновые
области, цвет бумаги, классификация областей, отпечаток файла.

СЧЁТ РАЗДАЁТСЯ ПУЛУ ПРОЦЕССОВ, запись остаётся здесь. Полосы независимы, детектор точек
считает по полному 21-мегапиксельному кадру и упирается в CPU — то есть ровно тот случай,
который CLAUDE.md требует распараллеливать. Писатель у SQLite при этом один: результаты
приходят в родителя и кладутся в базу по мере готовности.

SURYA ЖИВЁТ В РОДИТЕЛЕ, И ПУЛ ЭТОМУ НЕ МЕШАЕТ. Видеопамять одна на всех, раздать инференс
пулу нельзя — но это и не нужно. Работа делится на два этапа (см. ``detection.page``):
воркеры читают файл и считают всё пиксельное, родитель зовёт GPU и собирает области. Между
ними едет разбор на несколько мегабайт, а не 21-мегапиксельный кадр. Сериализуется таким
образом только инференс, а чтение с медленного диска и счёт по полному кадру идут в
шестнадцать процессов, и прогон по-прежнему упирается в диск.
"""

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session
from tqdm import tqdm

from ocr_utils.scan_markup.db.models import SOURCE_AUTO, Page, RasterRegion
from ocr_utils.scan_markup.db.repo import iter_pages, replace_raster_regions, upsert_pack
from ocr_utils.scan_markup.detection import DETECTOR_VERSION
from ocr_utils.scan_markup.detection.boxes import FULL_PAGE_FRAC, MIN_REGION_FRAC
from ocr_utils.scan_markup.detection.color_kind import (
    CHROMA_SELF_FRAC_THR,
    CHROMA_SPREAD_THR,
    CHROMA_THR,
    COLOR_FRAC_THR,
)
from ocr_utils.scan_markup.detection.overlay import write_debug_overlay
from ocr_utils.scan_markup.detection.tone import (
    LINEART_ENTROPY_THR,
    LINEART_MID_FRAC_THR,
    LINEART_SCREEN_PEAK_THR,
    STAMP_INK_CONTRAST_THR,
)
from ocr_utils.scan_markup.detection.regions import (
    FULL_PAGE_COLOR_FRAC,
    GROW_PAPER_MARGIN,
    LEADER_EMPTY_ROWS_THR,
    LINEART_MAX_DOT_FRAC,
    LEADER_PERIODICITY_THR,
    LEADER_TONE_SPREAD_THR,
    LINEART_PICTURE_MIN_FRAC,
    SAFETY_MIN_FRAC,
    SURYA_LINEART_P99_PX,
)
from ocr_utils.scan_markup.detection.page import (
    PageAnalysis,
    PageOptions,
    PageResult,
    analyse_page,
    finish_page,
    surya_boxes_for,
)
from ocr_utils.scan_markup.hashing import apply_stamp, stat_matches, stat_stamp
from ocr_utils.scan_markup.scan_tree import count_pages, scan_pack

logger = logging.getLogger(__name__)

# Сколько полос отдавать воркеру за раз. Полоса считается около секунды, так что накладные
# расходы на передачу задания несущественны, а мелкий кусок лучше выравнивает хвост.
CHUNK_SIZE = 4


@dataclass
class DetectParams:
    """Параметры прогона ``detect``."""

    pack_dir: Path
    db_path: Path
    pack_name: str
    default_dpi: int | None = None
    only_year: str | None = None
    only_issue: str | None = None
    limit: int | None = None
    skip_detected: bool = False
    rehash_all: bool = False
    use_surya_layout: bool = True
    first_page_is_cover: bool = False
    jobs: int = 8
    chroma_thr: float = CHROMA_THR
    color_frac_thr: float = COLOR_FRAC_THR
    chroma_spread_thr: float = CHROMA_SPREAD_THR
    chroma_self_frac_thr: float | None = CHROMA_SELF_FRAC_THR
    min_region_frac: float = MIN_REGION_FRAC
    merge_gap: int | None = None
    full_page_frac: float = FULL_PAGE_FRAC
    cell_px: int | None = None
    dot_frac_thr: float | None = None
    min_cells: int | None = None
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
    debug_dir: Path | None = None

    def page_options(self) -> PageOptions:
        """Часть параметров, которая уезжает в воркер. Обязана переживать pickle."""
        return PageOptions(
            default_dpi=self.default_dpi,
            first_page_is_cover=self.first_page_is_cover,
            chroma_thr=self.chroma_thr,
            color_frac_thr=self.color_frac_thr,
            chroma_spread_thr=self.chroma_spread_thr,
            chroma_self_frac_thr=self.chroma_self_frac_thr,
            min_region_frac=self.min_region_frac,
            merge_gap=self.merge_gap,
            full_page_frac=self.full_page_frac,
            cell_px=self.cell_px,
            dot_frac_thr=self.dot_frac_thr,
            min_cells=self.min_cells,
            lineart_p99=self.lineart_p99,
            safety_min_frac=self.safety_min_frac,
            lineart_picture_min_frac=self.lineart_picture_min_frac,
            full_page_color_frac=self.full_page_color_frac,
            leader_empty_rows_thr=self.leader_empty_rows_thr,
            leader_periodicity_thr=self.leader_periodicity_thr,
            leader_tone_spread_thr=self.leader_tone_spread_thr,
            grow_paper_margin=self.grow_paper_margin,
            lineart_mid_frac=self.lineart_mid_frac,
            lineart_entropy=self.lineart_entropy,
            lineart_screen_peak=self.lineart_screen_peak,
            stamp_ink_contrast=self.stamp_ink_contrast,
            lineart_max_dot_frac=self.lineart_max_dot_frac,
        )

    def worker_count(self) -> int:
        """Сколько процессов заводить на пиксельный этап. GPU их не ограничивает."""
        return max(1, self.jobs)


@dataclass
class DetectStats:
    """Итоги прогона — то, что печатается в конце."""

    pages: int = 0
    skipped: int = 0
    changed: int = 0
    failed: int = 0
    regions: int = 0
    color: int = 0
    grayscale: int = 0
    full_page: int = 0


@dataclass(frozen=True)
class _Job:
    """Задание воркеру: что считать и чем."""

    path: Path
    rel_path: str
    order_index: int
    options: PageOptions
    known_digest: str | None = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _worker(job: _Job) -> PageAnalysis:
    """Обёртка для ``ProcessPoolExecutor.imap`` (лямбду не запикль)."""
    return analyse_page(job.path, job.rel_path, job.order_index, job.options, job.known_digest)


def _needs_detection(page: Page, params: DetectParams, stamp) -> bool:
    """Надо ли считать эту полосу заново при ``--skip-detected``.

    Три условия, и все три обязательны. Полоса вообще считалась; считалась ТЕКУЩЕЙ версией
    детектора; файл с тех пор не менялся. Без второго условия правка алгоритма не доехала бы
    до базы вовсе: файлы-то те же, и весь пак был бы молча пропущен.
    """
    if page.detected_at is None or page.file_hash is None:
        return True
    if page.detector_version != DETECTOR_VERSION:
        return True
    if params.rehash_all:
        return True
    return not stat_matches(page, stamp)


def _apply_result(session: Session, page: Page, result: PageResult, stats: DetectStats) -> None:
    """Кладёт результат по полосе в базу и обновляет счётчики."""
    if result.stamp is not None and page.file_hash is not None and page.file_hash != result.stamp.digest:
        stats.changed += 1
        tqdm.write(f"ФАЙЛ ИЗМЕНИЛСЯ {page.rel_path}: разметка в CVAT к нему больше не относится")

    # Делитель и размеры уменьшенной копии здесь НЕ считаются: их выбирает to-cvat по
    # своему --cvat-dpi. Отсюда уходит только то, что прочитано из файла.
    page.width, page.height, page.dpi = result.width, result.height, result.dpi
    page.detected_at = _utcnow()
    page.detector_version = DETECTOR_VERSION
    # Отпечаток пишется ПОСЛЕ успешной детекции: полоса, на которой детекция упала, не
    # должна выглядеть обработанной для следующего прогона.
    if result.stamp is not None:
        apply_stamp(page, result.stamp)

    replace_raster_regions(
        session,
        page,
        [
            RasterRegion(
                x1=region.box[0],
                y1=region.box[1],
                x2=region.box[2],
                y2=region.box[3],
                kind=region.kind,
                full_page=region.full_page,
                chroma_frac=region.chroma_frac,
                chroma_spread=region.chroma_spread,
                chroma_self_frac=region.chroma_self_frac,
                dot_frac=region.dot_frac,
                mid_frac=region.mid_frac,
                tone_entropy=region.tone_entropy,
                screen_peak=region.screen_peak,
                ink_contrast=region.ink_contrast,
                source=SOURCE_AUTO,
            )
            for region in result.regions
        ],
    )

    stats.pages += 1
    stats.regions += len(result.regions)
    stats.color += sum(1 for region in result.regions if region.kind == "color")
    stats.grayscale += sum(1 for region in result.regions if region.kind == "grayscale")
    stats.full_page += sum(1 for region in result.regions if region.full_page)


def _collect_jobs(
    session: Session, params: DetectParams, stats: DetectStats, years
) -> tuple[list[_Job], dict[str, Page]]:
    """Отбирает полосы, которые надо считать, и попутно пропускает неизменившиеся."""
    pack = upsert_pack(session, params.pack_name, params.pack_dir, years)
    pages = list(iter_pages(pack, params.only_year, params.only_issue))
    if params.limit is not None:
        pages = pages[: params.limit]

    options = params.page_options()
    jobs: list[_Job] = []
    by_rel: dict[str, Page] = {}
    for _year, _issue, page in pages:
        path = params.pack_dir / page.rel_path
        try:
            stamp = stat_stamp(path)
        except OSError as exc:
            stats.failed += 1
            tqdm.write(f"ОШИБКА {page.rel_path}: {exc}")
            continue

        # Дешёвая проверка идёт первой: совпали версия детектора, размер и время правки —
        # файл не читается вовсе. Именно ради этого пропуска ``stat`` и лежит в базе.
        if params.skip_detected and not _needs_detection(page, params, stamp):
            stats.skipped += 1
            continue

        by_rel[page.rel_path] = page
        # Хеш из базы отдаём воркеру, только если полосу пересчитывают из-за разъехавшегося
        # ``stat``: совпал хеш — файл просто переписали тем же содержимым, и декодировать его
        # незачем. Два условия, и оба обязательны. Без ``--skip-detected`` короткого замыкания
        # быть не должно вовсе: прогон без флага — это требование пересчитать всё. А при смене
        # версии детектора совпадение хеша ничего не значит: файл прежний, алгоритм новый.
        stale_stat_only = (
            params.skip_detected
            and page.detected_at is not None
            and page.file_hash is not None
            and page.detector_version == DETECTOR_VERSION
        )
        jobs.append(_Job(path, page.rel_path, page.order_index, options, page.file_hash if stale_stat_only else None))
    return jobs, by_rel


def run_detect(params: DetectParams, session_factory) -> DetectStats:
    """Полный прогон: обход пака, запись дерева, детекция по каждой полосе."""
    years = scan_pack(params.pack_dir)
    if not years:
        raise ValueError(f"в {params.pack_dir} не найдено ни одного годового комплекта с картинками")
    logger.info(
        "Пак %s: лет %d, выпусков %d, полос %d",
        params.pack_name,
        len(years),
        sum(len(year.issues) for year in years),
        count_pages(years),
    )

    detector = None
    if params.use_surya_layout:
        from ocr_utils.background_smoothing.layout import LayoutDetector

        detector = LayoutDetector()

    stats = DetectStats()
    with session_factory() as session:  # type: Session
        jobs, by_rel = _collect_jobs(session, params, stats, years)
        session.commit()

        for result in _iter_results(jobs, params, detector):
            page = by_rel[result.rel_path]
            if result.error:
                stats.failed += 1
                tqdm.write(f"ОШИБКА {result.rel_path}: {result.error}")
                continue
            if result.unchanged:
                # Содержимое прежнее — обновляем только отметку ``stat``, разметку не трогаем.
                apply_stamp(page, result.stamp)
                session.commit()
                stats.skipped += 1
                continue
            _apply_result(session, page, result, stats)
            session.commit()
            if params.debug_dir is not None and result.regions:
                write_debug_overlay(
                    params.debug_dir, result.rel_path, params.pack_dir / result.rel_path, result.regions
                )

    return stats


def _iter_results(jobs: list[_Job], params: DetectParams, detector):
    """Результаты по полосам: пиксельный этап в пуле, GPU и сборка — здесь.

    ``imap`` с ``chunksize=1``, а не ``map``: воркер возвращает рабочую копию полосы на
    несколько мегабайт, и крупными кусками они копились бы в очереди пула десятками.
    """
    if not jobs:
        return

    workers = params.worker_count()
    if workers == 1:
        analyses = (_worker(job) for job in jobs)
        yield from _finish(analyses, len(jobs), params, detector)
        return

    with ProcessPoolExecutor(max_workers=workers) as pool:
        analyses = pool.map(_worker, jobs, chunksize=1)
        yield from _finish(analyses, len(jobs), params, detector)


def _finish(analyses, total: int, params: DetectParams, detector):
    """Прогон Surya и сборка областей — строго последовательно, в родителе."""
    options = params.page_options()
    for analysis in tqdm(analyses, total=total, desc="детекция", unit="полоса"):
        yield finish_page(analysis, options, surya_boxes_for(analysis, detector))
