"""Детекция по ОДНОЙ полосе: файл на диске -> результат, без единого обращения к базе.

РАЗБИТА НА ДВА ЭТАПА, и это не косметика. Первичный детектор — Surya, а она живёт на GPU:
видеопамять одна на всех, и раздать инференс пулу процессов нельзя. Всё остальное — чтение
с медленного NTFS-3G, уменьшение, порог краски и связные компоненты по 21-мегапиксельному
кадру — прекрасно параллелится и как раз упирается в диск и в CPU.

Поэтому:

* :func:`analyse_page` читает файл и считает всё пиксельное — её крутит пул процессов;
* :func:`finish_page` получает готовый разбор плюс блоки Surya и собирает области — она
  работает в родителе, где и живёт модель.

Между этапами едет :class:`PageAnalysis`. Он обязан переживать pickle и быть НЕБОЛЬШИМ:
рабочая копия 1/4 (около 4 МБ), карты клеток (сотни чисел) и статистика пятен (меньше
мегабайта). Полный кадр не едет никуда — 21 мегапиксель через pickle это уже не оптимизация,
а её противоположность.

Отсюда же следует, что ЦВЕТ областей меряется по копии 1/4, а не по полному кадру. Замер
показал, что разделение от этого не страдает: разброс хроматичности по уменьшенной копии дал
у ч/б областей до 5.19 при 9.03 у цветных — тот же промежуток, что и на полном разрешении
(5.28 против 9.27).

Здесь же снимается отпечаток файла: файл всё равно читается целиком, и хеш на прогретом
кеше стоит доли секунды (см. ``scan_markup.hashing``).
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ocr_utils.background_smoothing.processing import HALFTONE_DOWNSCALE
from ocr_utils.scan_cropping.image_io import read_dpi
from ocr_utils.scan_markup.db.models import KIND_COLOR, KIND_STAMP_SUSPECT
from ocr_utils.scan_markup.detection import cover as cover_module
from ocr_utils.scan_markup.detection.boxes import (
    FULL_PAGE_FRAC,
    MIN_REGION_FRAC,
    is_full_page,
    polygons_to_boxes,
    upscale_box,
)
from ocr_utils.scan_markup.detection.color_kind import (
    CHROMA_SELF_FRAC_THR,
    CHROMA_SPREAD_THR,
    CHROMA_THR,
    COLOR_FRAC_THR,
    balanced_lab,
    classify,
    paper_color,
)
from ocr_utils.scan_markup.detection.dots import ScreenParams, ScreenRegions, params_for_dpi, screen_regions
from ocr_utils.scan_markup.detection.tone import (
    LINEART_ENTROPY_THR,
    LINEART_MID_FRAC_THR,
    LINEART_SCREEN_PEAK_THR,
    STAMP_INK_CONTRAST_THR,
    ToneMaps,
    tone_maps,
)
from ocr_utils.scan_markup.detection.regions import (
    FULL_PAGE_COLOR_FRAC,
    GROW_PAPER_MARGIN,
    LEADER_EMPTY_ROWS_THR,
    LEADER_PERIODICITY_THR,
    LEADER_TONE_SPREAD_THR,
    LINEART_FULL_PAGE_INK_FRAC,
    LINEART_PICTURE_MIN_FRAC,
    SAFETY_MIN_FRAC,
    SOURCE_LINEART,
    SURYA_LINEART_P99_PX,
    find_raster_boxes,
)
from ocr_utils.scan_markup.hashing import FileStamp, full_stamp

logger = logging.getLogger(__name__)

# Ниже этого разрешения тег считается отсутствующим. TIFF без разрешения не бывает: если
# его не записали, PIL и большинство сканеров всё равно кладут в тег 1 dpi. Взять эту
# единицу всерьёз — значит получить делитель 1, то есть залить в CVAT полноразмерные
# сканы, ради ухода от которых уменьшение и делается, причём молча. Порог 72 dpi ниже
# любого осмысленного разрешения сканирования и выше всякого мусорного.
MIN_PLAUSIBLE_DPI = 72


@dataclass(frozen=True)
class PageOptions:
    """Всё, что нужно детекции по одной полосе. Обязано переживать pickle."""

    default_dpi: int | None = None
    first_page_is_cover: bool = False
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
    # None — взять пересчитанное от DPI полосы (см. dots.params_for_dpi).
    merge_gap: int | None = None
    min_region_side_px: int | None = None
    cell_px: int | None = None
    dot_frac_thr: float | None = None
    dot_max_area_px: int | None = None
    cell_max_area_px: int | None = None
    min_dots_per_cell: int | None = None
    min_cells: int | None = None
    # Отпечаток файла: считать хеш или ограничиться ``stat``.
    need_digest: bool = True


@dataclass
class DetectedRegion:
    """Одна найденная область в координатах ОРИГИНАЛА."""

    box: tuple[int, int, int, int]
    kind: str
    full_page: bool
    chroma_frac: float | None = None
    chroma_spread: float | None = None
    chroma_self_frac: float | None = None
    dot_frac: float | None = None
    mid_frac: float | None = None
    tone_entropy: float | None = None
    screen_peak: float | None = None
    ink_contrast: float | None = None


@dataclass
class PageAnalysis:
    """Пиксельный разбор полосы: всё, что посчитано без GPU. Едет из воркера в родителя."""

    rel_path: str
    order_index: int
    width: int = 0
    height: int = 0
    dpi: int = 0
    work: np.ndarray | None = None  # копия 1/4, BGR — по ней же гоняется Surya
    regions: ScreenRegions | None = None
    stats: np.ndarray | None = None
    centroids: np.ndarray | None = None
    tone: ToneMaps | None = None
    params: ScreenParams | None = None
    stamp: FileStamp | None = None
    error: str = ""
    # Полоса, ответ по которой известен заранее (обложка при --first-page-is-cover):
    # пиксели не читались, Surya не нужна.
    ready: list[DetectedRegion] | None = None


@dataclass
class PageResult:
    """Результат по одной полосе. ``error`` непуст — значит остальное недостоверно."""

    rel_path: str
    width: int = 0
    height: int = 0
    dpi: int = 0
    regions: list[DetectedRegion] = field(default_factory=list)
    stamp: FileStamp | None = None
    error: str = ""
    # Файл читали только потому, что разошёлся ``stat``, а содержимое оказалось прежним:
    # разметку трогать не надо, достаточно обновить отметку.
    unchanged: bool = False


def resolve_dpi(path: Path, default_dpi: int | None) -> int:
    """DPI полосы из тега файла либо из ``--default-dpi``; иначе — понятная ошибка."""
    dpi = read_dpi(path)
    if dpi is not None and dpi >= MIN_PLAUSIBLE_DPI:
        return int(dpi)
    if default_dpi is None:
        raise ValueError(
            f"нет осмысленного тега разрешения (прочитано {dpi!r}), а --default-dpi не задан; "
            "подставить разрешение наугад значило бы промахнуться в разы и залить в CVAT "
            "кадры не того размера"
        )
    logger.debug("У %s разрешение %r, беру --default-dpi=%d", path.name, dpi, default_dpi)
    return int(default_dpi)


def analyse_page(
    path: Path, rel_path: str, order_index: int, options: PageOptions, known_digest: str | None = None
) -> PageAnalysis:
    """Пиксельный разбор полосы. Исключения не выпускает — кладёт их в ``error``.

    ``known_digest`` — хеш, записанный у полосы в базе. Совпал с посчитанным сейчас — файл
    просто переписали тем же содержимым (обычное дело при копировании пака), и декодировать
    его незачем: возвращается разбор с пустой рабочей копией, а результат помечается
    ``unchanged``.
    """
    try:
        dpi = resolve_dpi(path, options.default_dpi)
        stamp = full_stamp(path) if options.need_digest else None
        analysis = PageAnalysis(rel_path, order_index, dpi=dpi, stamp=stamp)
        if known_digest is not None and stamp is not None and stamp.digest == known_digest:
            return analysis

        if options.first_page_is_cover and order_index == 0:
            # Ответ известен заранее, и декодировать 40 МБ TIFF ради него незачем: размеры
            # берутся из заголовка. Чтение файла всё равно происходит — его требует хеш.
            with Image.open(path) as image:
                analysis.width, analysis.height = image.size
            box = cover_module.cover_region(analysis.width, analysis.height)
            analysis.ready = [DetectedRegion(box, KIND_COLOR, True)]
            return analysis

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("не читается как изображение")
        analysis.height, analysis.width = bgr.shape[:2]

        analysis.work = cv2.resize(
            bgr,
            (max(1, analysis.width // HALFTONE_DOWNSCALE), max(1, analysis.height // HALFTONE_DOWNSCALE)),
            interpolation=cv2.INTER_AREA,
        )
        analysis.params = params_for_dpi(
            dpi,
            cell_px=options.cell_px,
            dot_frac_thr=options.dot_frac_thr,
            dot_max_area_px=options.dot_max_area_px,
            cell_max_area_px=options.cell_max_area_px,
            min_dots_per_cell=options.min_dots_per_cell,
            min_cells=options.min_cells,
        )
        full_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        analysis.regions, analysis.stats, analysis.centroids = screen_regions(full_gray, analysis.params)
        analysis.tone = tone_maps(full_gray, analysis.params)
        return analysis
    except Exception as exc:  # noqa: BLE001 — одна битая полоса не должна валить прогон
        return PageAnalysis(rel_path, order_index, error=str(exc))


def surya_boxes_for(analysis: PageAnalysis, detector) -> list[tuple[int, int, int, int]] | None:
    """Сырые блоки ``Picture`` в координатах ОРИГИНАЛА; ``None``, если Surya не гонялась.

    Блоки берутся БЕЗ фильтра ``is_raster_block``: он меряет полутон по копии 1/4, где штрих
    от растра неотличим. Отличать их будет ``regions.find_raster_boxes`` по статистике пятен.
    """
    if detector is None or analysis.work is None:
        return None
    work_gray = cv2.cvtColor(analysis.work, cv2.COLOR_BGR2GRAY)
    polygons = detector.picture_polygons(analysis.work, work_gray, filter_raster=False)
    return [upscale_box(box, HALFTONE_DOWNSCALE) for box in polygons_to_boxes(polygons)]


def finish_page(
    analysis: PageAnalysis, options: PageOptions, surya_boxes: list[tuple[int, int, int, int]] | None
) -> PageResult:
    """Сборка областей по готовому разбору и блокам Surya. Работает в родителе."""
    if analysis.error:
        return PageResult(analysis.rel_path, error=analysis.error)
    if analysis.ready is not None:
        return PageResult(
            analysis.rel_path, analysis.width, analysis.height, analysis.dpi, analysis.ready, analysis.stamp
        )
    if analysis.work is None:  # содержимое не изменилось, пиксели не читались
        return PageResult(analysis.rel_path, stamp=analysis.stamp, unchanged=True)

    work_gray = cv2.cvtColor(analysis.work, cv2.COLOR_BGR2GRAY)

    # Цвет полосы считается ОДИН раз на всю полосу и идёт в два места: по разбросу
    # хроматичности опознаётся цветная полоса целиком (``regions._fill_colour_page``), а по
    # цвету бумаги классифицируются найденные области. Перевод копии 1/4 в Lab не бесплатен,
    # и делать его дважды незачем.
    paper = paper_color(analysis.work)
    page_a, page_b = balanced_lab(analysis.work, paper)
    page_chroma_spread = float(np.hypot(page_a.std(), page_b.std()))

    findings = find_raster_boxes(
        analysis.regions,
        analysis.stats,
        analysis.centroids,
        work_gray,
        (analysis.height, analysis.width),
        analysis.params,
        surya_boxes=surya_boxes,
        order_index=analysis.order_index,
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
        tone=analysis.tone,
        lineart_mid_frac=options.lineart_mid_frac,
        lineart_entropy=options.lineart_entropy,
        lineart_screen_peak=options.lineart_screen_peak,
    )
    regions = _classify_regions(analysis, findings, options, paper)
    return PageResult(analysis.rel_path, analysis.width, analysis.height, analysis.dpi, regions, analysis.stamp)


def detect_page(
    path: Path, rel_path: str, order_index: int, options: PageOptions, detector=None, known_digest: str | None = None
) -> PageResult:
    """Оба этапа подряд — для однопроцессного прогона, оснастки валидации и тестов."""
    analysis = analyse_page(path, rel_path, order_index, options, known_digest)
    return finish_page(analysis, options, surya_boxes_for(analysis, detector))


def _classify_regions(analysis: PageAnalysis, findings, options: PageOptions, paper) -> list[DetectedRegion]:
    """Красит найденные области и решает, какие вообще идут в разметку.

    Правила:

    * растр серый / цветной   -> ``grayscale`` / ``color``;
    * штрих цветной и крупный -> ``color``: цветной рисунок берётся целиком;
    * штрих чёрный            -> в разметку НЕ включается вовсе: он бинаризуется как текст,
      и вырезать из него картинку не нужно;
    * оттиск библиотечной печати -> ``stamp_suspect``.

    Про печати. Оттиск — это мелкий штрих, и опознаётся он по размеру: замер по паку-1 дал у
    печатей 2.1 и 2.2% полосы против почти 100% у синего рисунка обложки. Но одного размера
    мало — мелкий чёрный штрих это ещё и виньетка рубрики, которых в паке два десятка. Их
    разделяют РАСТРОВЫЕ КЛЕТКИ:

    * бледный штрих — оттиск. Мастика фиолетовая, но выгорает, и такие печати уезжали в
      разметку полноценными ``grayscale``-картинками (замер по паку-1: 12 штук сверх 33,
      опознанных по цвету). Контраст «бумага минус краска» у 10 оттисков 79..220;
    * чёрный штрих — виньетка рубрики: типографская краска, контраст 231..254 у всех 33.

    Отдельно остаётся старое условие «мелкая ЦВЕТНАЯ область без растровых клеток»: оно ловит
    оттиски, которые ``detection.tone`` штрихом не признала. Тёмная полутоновая фотография на
    внутренней полосе этого журнала в любом случае серая, не цветная, — значит, это оттиск.
    """
    if not findings.findings:
        return []

    # ``paper`` посчитан вызывающим по копии 1/4 всей полосы: он один на полосу и не зависит
    # от того, какую область мы сейчас классифицируем (мотивировка в color_kind.paper_color).
    scale = HALFTONE_DOWNSCALE
    work_h, work_w = analysis.work.shape[:2]

    regions = []
    for finding in findings.findings:
        x1, y1 = finding.box[0] // scale, finding.box[1] // scale
        x2 = min(max(x1 + 1, -(-finding.box[2] // scale)), work_w)
        y2 = min(max(y1 + 1, -(-finding.box[3] // scale)), work_h)
        color = classify(
            analysis.work[y1:y2, x1:x2],
            paper,
            options.chroma_thr,
            options.color_frac_thr,
            options.chroma_spread_thr,
            options.chroma_self_frac_thr,
        )
        kind = color.kind
        area = (finding.box[2] - finding.box[0]) * (finding.box[3] - finding.box[1])
        small = area < options.lineart_picture_min_frac * analysis.width * analysis.height
        line_art = finding.source == SOURCE_LINEART
        pale = finding.ink_contrast is not None and finding.ink_contrast < options.stamp_ink_contrast
        stamp = small and (
            (line_art and pale)  # бледная краска: оттиск мастикой, а не типографская виньетка
            or (color.kind == KIND_COLOR and not finding.has_cells)  # цветной оттиск без клеток
        )
        if stamp:
            kind = KIND_STAMP_SUSPECT
        elif line_art and color.kind != KIND_COLOR:
            continue  # чёрный штрих бинаризуется как текст, вырезать из него нечего
        regions.append(
            DetectedRegion(
                box=finding.box,
                kind=kind,
                full_page=finding.full_page
                or is_full_page(finding.box, analysis.width, analysis.height, options.full_page_frac),
                chroma_frac=color.chroma_frac,
                chroma_spread=color.chroma_spread,
                chroma_self_frac=color.chroma_self_frac,
                dot_frac=finding.dot_frac,
                mid_frac=finding.mid_frac,
                tone_entropy=finding.tone_entropy,
                screen_peak=finding.screen_peak,
                ink_contrast=finding.ink_contrast,
            )
        )
    return regions
