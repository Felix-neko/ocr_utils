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
import dataclasses
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session
from tqdm import tqdm

from ocr_utils.scan_markup.db.models import SOURCE_AUTO, SOURCE_CVAT, Page, RasterRegion
from ocr_utils.scan_markup.db.repo import iter_pages, replace_raster_regions, upsert_pack
from ocr_utils.scan_markup.detection import DETECTOR_VERSION
from ocr_utils.scan_markup.orientation import ORIENTATION_VERSION
from ocr_utils.scan_markup.rotation import format_allowed
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
    orientation_image,
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

    # --- Ориентация полосы ---------------------------------------------------
    # Считается тем же прогоном, что и растр: пак читается ОДИН раз, а не два.
    orientation: bool = True
    # Через запятую, как в CLI. GPU-детектор (surya_lines) идёт в родителе, остальные —
    # в воркерах; арбитр (ocr_vote) вызывается вторым проходом только по кандидатам.
    orientation_detectors: str = "ink_axis,osd,surya_lines,ocr_vote"
    # Набор допустимых поворотов; пустая строка — взять из базы (пак/выпуск) или умолчание.
    angles: str = ""

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
            orientation=self.orientation,
            orientation_detectors=self.cpu_orientation_detectors(),
            allowed_rotations=self.allowed_rotations(),
        )

    def detector_names(self) -> tuple[str, ...]:
        return tuple(part.strip() for part in self.orientation_detectors.split(",") if part.strip())

    def cpu_orientation_detectors(self) -> tuple[str, ...]:
        """Только те, что считаются в воркере: не GPU и не арбитр."""
        from ocr_utils.scan_markup.orientation.detectors import DETECTORS

        return tuple(
            name
            for name in self.detector_names()
            if (d := DETECTORS.get(name)) is not None and d.stage == "cpu" and not d.arbiter
        )

    def gpu_orientation_detectors(self) -> tuple[str, ...]:
        from ocr_utils.scan_markup.orientation.detectors import DETECTORS

        return tuple(
            name for name in self.detector_names() if (d := DETECTORS.get(name)) is not None and d.stage == "gpu"
        )

    def arbiter_orientation_detectors(self) -> tuple[str, ...]:
        from ocr_utils.scan_markup.orientation.detectors import DETECTORS

        return tuple(name for name in self.detector_names() if (d := DETECTORS.get(name)) is not None and d.arbiter)

    def allowed_rotations(self, pack_value: "str | None" = None, issue_value: "str | None" = None):
        """Набор допустимых поворотов: ключ прогона важнее выпуска, выпуск важнее пака.

        Ключ ставит человек здесь и сейчас, значение выпуска — переопределение пакового, а
        паковое — общее условие. Порядок именно такой: явно указанное на запуске не должно
        молча перебиваться тем, что лежит в базе.
        """
        from ocr_utils.scan_markup.rotation import DEFAULT_ALLOWED, parse_allowed

        for raw in (self.angles, issue_value, pack_value):
            if raw:
                return parse_allowed(raw)
        return DEFAULT_ALLOWED

    def worker_count(self) -> int:
        """Сколько процессов заводить на пиксельный этап. GPU их не ограничивает."""
        return max(1, self.jobs)


@dataclass
class DetectStats:
    """Итоги прогона — то, что печатается в конце."""

    pages: int = 0
    skipped: int = 0
    changed: int = 0
    # Полос, которым сводный вердикт назначил поворот.
    rotated: int = 0
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


def _orientation_stale(page: Page, params: DetectParams) -> bool:
    """Устарела ли ориентация САМА ПО СЕБЕ — без оглядки на ``stat`` файла.

    Отделено от :func:`_needs_orientation` затем, что вопросы разные. Здесь: «поменялся ли
    алгоритм или полосу вовсе не считали». Там: «надо ли её пересчитывать в этом прогоне»,
    куда входит ещё и то, менялся ли файл.
    """
    if page.orientation_source == SOURCE_CVAT:
        return False
    if page.orientation_detected_at is None or page.rotate_cw is None:
        return True
    return page.orientation_version != ORIENTATION_VERSION


def _needs_orientation(page: Page, params: DetectParams, stamp) -> bool:
    """То же для ОРИЕНТАЦИИ, и версия у неё СВОЯ.

    Слить обе версии в одну нельзя: тогда правка порога ориентации заставила бы перечитать
    весь пак вместе со всей растровой детекцией — полтерабайта с медленного NTFS-3G и часы.
    И наоборот: правка детектора растра не должна отменять уже посчитанную ориентацию.

    Ручную правку из CVAT прогон не трогает: у неё ``orientation_source`` равен ``cvat``, и
    пересчитывать её значило бы затирать решение человека автоматическим.
    """
    if not params.orientation:
        return False
    if page.orientation_source == SOURCE_CVAT:
        return False
    if _orientation_stale(page, params):
        return True
    if params.rehash_all:
        return True
    return not stat_matches(page, stamp)


def _apply_orientation(page: Page, result: PageResult, stats: DetectStats) -> None:
    """Пишет ориентацию — и ТОЛЬКО когда её действительно считали.

    ``combo is None`` значит «не считали»: полосу пропустили по свежей версии ориентации или
    ориентацию не просили вовсе. Проставить в этом случае версию и время значило бы соврать —
    следующий прогон решил бы, что полоса посчитана, и больше к ней не вернулся.
    """
    if result.combo is None:
        return
    page.rotate_cw = result.combo.rotate_cw
    page.orientation_confidence = float(result.combo.confidence)
    page.orientation_version = ORIENTATION_VERSION
    page.orientation_detected_at = _utcnow()
    page.orientation_source = SOURCE_AUTO
    stats.rotated += bool(result.combo.rotate_cw)


def _apply_result(session: Session, page: Page, result: PageResult, stats: DetectStats) -> None:
    """Кладёт результат по полосе в базу и обновляет счётчики."""
    if result.stamp is not None and page.file_hash is not None and page.file_hash != result.stamp.digest:
        stats.changed += 1
        tqdm.write(f"ФАЙЛ ИЗМЕНИЛСЯ {page.source_rel_path}: разметка в CVAT к нему больше не относится")

    # Делитель и размеры уменьшенной копии здесь НЕ считаются: их выбирает to-cvat по
    # своему --cvat-dpi. Отсюда уходит только то, что прочитано из файла.
    page.width, page.height, page.dpi = result.width, result.height, result.dpi
    page.detected_at = _utcnow()
    page.detector_version = DETECTOR_VERSION
    _apply_orientation(page, result, stats)
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

    # Набор допустимых поворотов ЗАКРЕПЛЯЕТСЯ ЗА ПАКОМ. Указан ключом — записывается в базу,
    # не указан — берётся оттуда. Так следующий шаг конвейера узнаёт условие, не полагаясь
    # на память запускающего: ровно затем же в паке лежат и корни путей.
    resolved = params.allowed_rotations(pack.allowed_rotations)
    params.angles = format_allowed(resolved)
    if pack.allowed_rotations != params.angles:
        logger.info("Допустимые повороты пака: %s", params.angles)
        pack.allowed_rotations = params.angles

    pages = list(iter_pages(pack, params.only_year, params.only_issue))
    if params.limit is not None:
        pages = pages[: params.limit]

    options = params.page_options()
    jobs: list[_Job] = []
    by_rel: dict[str, Page] = {}
    for _year, _issue, page in pages:
        path = params.pack_dir / page.source_rel_path
        try:
            stamp = stat_stamp(path)
        except OSError as exc:
            stats.failed += 1
            tqdm.write(f"ОШИБКА {page.source_rel_path}: {exc}")
            continue

        # Дешёвая проверка идёт первой: совпали версия детектора, размер и время правки —
        # файл не читается вовсе. Именно ради этого пропуска ``stat`` и лежит в базе.
        need_regions = not params.skip_detected or _needs_detection(page, params, stamp)
        need_orientation = _needs_orientation(page, params, stamp) if params.skip_detected else params.orientation
        if not need_regions and not need_orientation:
            stats.skipped += 1
            continue

        by_rel[page.source_rel_path] = page
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
        # Полосе, которой нужна только ориентация, растровый этап всё равно считается: он
        # идёт по тому же разжатому кадру и почти ничего не добавляет к стоимости чтения.
        # А вот НАОБОРОТ — считать ориентацию там, где она уже свежая, — заметная трата:
        # tesseract OSD стоит две с половиной секунды на полосу.
        job_options = options if need_orientation else dataclasses.replace(options, orientation=False)

        # Короткое замыкание по хешу действует и на ориентацию: совпал хеш — содержимое
        # прежнее, а значит прежний вердикт о повороте по-прежнему верен, и декодировать
        # полосу незачем. Но только пока версия ориентации та же: при её смене совпадение
        # хеша не значит ничего — файл прежний, алгоритм новый.
        orientation_current = not params.orientation or not _orientation_stale(page, params)
        known_digest = page.file_hash if stale_stat_only and orientation_current else None
        jobs.append(_Job(path, page.source_rel_path, page.order_index, job_options, known_digest))
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

        _run_arbiter(session, params, by_rel, stats)

    return stats


def _run_arbiter(session: Session, params: DetectParams, by_rel: "dict[str, Page]", stats: DetectStats) -> None:
    """Второй проход: арбитр по кандидатам, которых отметили быстрые детекторы.

    Отдельным проходом, а не в общем цикле, потому что он ДОРОГ: распознавание на каждом
    допустимом повороте — секунды на полосу против долей секунды у остальных. По паку-1 в
    кандидаты попадает около сотни полос из двенадцати тысяч, и повторное чтение сотни файлов
    несущественно рядом с полтерабайтом первого прохода.

    Арбитр решает сам: он единственный проверяет гипотезы напрямую, читая полосу на каждом
    повороте, тогда как остальные судят об ориентации косвенно.
    """
    names = params.arbiter_orientation_detectors()
    if not params.orientation or not names:
        return

    from ocr_utils.scan_markup.orientation.detectors import DETECTORS
    from ocr_utils.scan_markup.orientation.image_io import read_frame

    candidates = [page for page in by_rel.values() if page.rotate_cw and page.orientation_source == SOURCE_AUTO]
    if not candidates:
        return
    logger.info("Кандидатов арбитру: %d", len(candidates))
    allowed = params.allowed_rotations()

    for page in tqdm(candidates, desc="арбитр", unit="полоса"):
        path = params.pack_dir / page.source_rel_path
        try:
            frame, _ = read_frame(path, page.source_rel_path, default_dpi=page.dpi or 600, allowed=allowed)
        except Exception as error:  # noqa: BLE001
            logger.warning("%s: арбитр не прочитал файл (%s)", page.source_rel_path, error)
            continue
        verdict = None
        for name in names:
            try:
                verdict = DETECTORS[name].run(frame)
            except Exception as error:  # noqa: BLE001
                logger.warning("%s: арбитр %s упал (%s)", page.source_rel_path, name, error)
                continue
            if verdict.confidence > 0.0:
                break
        if verdict is None or verdict.confidence <= 0.0:
            continue
        if verdict.rotate_cw != page.rotate_cw:
            stats.rotated -= bool(page.rotate_cw)
            stats.rotated += bool(verdict.rotate_cw)
        page.rotate_cw = verdict.rotate_cw
        page.orientation_confidence = float(verdict.confidence)
        session.commit()


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


def _orientation_gpu(params: DetectParams):
    """GPU-детекторы ориентации, поднятые по одному разу на прогон.

    Модели грузятся ЛЕНИВО, при первом вызове. Это не украшение: пул процессов создаётся
    контекстом ``fork``, и форк процесса с уже поднятой CUDA даёт зависшие воркеры. Пока
    первое обращение случается в родителе ПОСЛЕ форка, всё в порядке.
    """
    from ocr_utils.scan_markup.orientation.detectors import DETECTORS

    if not params.orientation:
        return {}
    return {name: DETECTORS[name].make_batch() for name in params.gpu_orientation_detectors()}


def _finish(analyses, total: int, params: DetectParams, detector):
    """Прогон Surya и сборка областей — строго последовательно, в родителе."""
    options = params.page_options()
    batches = _orientation_gpu(params)
    for analysis in tqdm(analyses, total=total, desc="детекция", unit="полоса"):
        gpu = _orientation_for(analysis, batches) if batches else None
        yield finish_page(analysis, options, surya_boxes_for(analysis, detector), gpu)


def _orientation_for(analysis, batches) -> "dict | None":
    """GPU-вердикты ориентации по одной полосе.

    По одной, а не пачкой: цикл ``_finish`` устроен «полоса — коммит», и буферизация ради
    батча отложила бы запись в базу на всю пачку. Так уже работает Surya layout рядом, и
    цена та же — доли секунды на полосу.
    """
    if not analysis.orientation_done or analysis.work is None:
        return None
    image = orientation_image(analysis)
    if image is None:
        return None
    verdicts = {}
    for name, batch in batches.items():
        try:
            result = batch([image])
        except Exception as error:  # noqa: BLE001 — вердикт не важнее полосы
            logger.warning("%s: детектор %s упал (%s)", analysis.rel_path, name, error)
            continue
        if result:
            verdicts[name] = result[0]
    return verdicts
