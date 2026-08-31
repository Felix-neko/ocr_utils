"""Прогон предварительной детекции по паку: оригиналы -> SQLite.

Один проход по TIFF на полосу. Это дорогая часть конвейера: пак-1 — 12 136 файлов по
~40 МБ на медленном NTFS, то есть около полутерабайта чтения. Поэтому за одно чтение
делается всё, что вообще можно сделать из пикселей: размеры, DPI и параметры уменьшения
в базу, рабочая копия 1/4 для детекции, цвет бумаги, классификация областей.

Заодно с пикселями снимается отпечаток файла — sha256 плюс размер и время правки
(``scan_markup.hashing``). Он нужен не детекции, а последующему обновлению пака: по нему
``to-cvat`` понимает, какие полосы разошлись с тем, что уже залито в CVAT. Считать его
здесь ничего не стоит — файл всё равно читается целиком.

Детекция считается по копии 1/``HALFTONE_DOWNSCALE``, а не по полному кадру. Surya всё
равно ужимает вход до 2048 по длинной стороне (кадр 600 dpi — 6051 px), а константы
``RASTER_*`` и ``HALFTONE_*`` в ``background_smoothing`` откалиброваны именно на копии 1/4.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
from sqlalchemy.orm import Session
from tqdm import tqdm

from ocr_utils.background_smoothing.processing import HALFTONE_DOWNSCALE
from ocr_utils.scan_cropping.image_io import read_dpi
from ocr_utils.scan_markup.db.models import SOURCE_AUTO, Page, RasterRegion
from ocr_utils.scan_markup.db.repo import iter_pages, replace_raster_regions, upsert_pack
from ocr_utils.scan_markup.detection.color_kind import CHROMA_THR, COLOR_FRAC_THR, classify, paper_color
from ocr_utils.scan_markup.detection.raster import (
    MERGE_GAP_PX,
    MIN_REGION_FRAC,
    FULL_PAGE_FRAC,
    find_raster_boxes,
    is_full_page,
    scale_box,
)
from ocr_utils.scan_markup.hashing import apply_stamp, full_stamp, stat_matches, stat_stamp
from ocr_utils.scan_markup.scan_tree import count_pages, scan_pack

logger = logging.getLogger(__name__)

# Ниже этого разрешения тег считается отсутствующим. TIFF без разрешения не бывает: если
# его не записали, PIL и большинство сканеров всё равно кладут в тег 1 dpi. Взять эту
# единицу всерьёз — значит получить делитель 1, то есть залить в CVAT полноразмерные
# сканы, ради ухода от которых уменьшение и делается, причём молча. Порог 72 dpi ниже
# любого осмысленного разрешения сканирования и выше всякого мусорного.
MIN_PLAUSIBLE_DPI = 72


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
    chroma_thr: float = CHROMA_THR
    color_frac_thr: float = COLOR_FRAC_THR
    min_region_frac: float = MIN_REGION_FRAC
    merge_gap: int = MERGE_GAP_PX
    full_page_frac: float = FULL_PAGE_FRAC
    debug_dir: Path | None = None


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


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def detect_page(
    path: Path, params: DetectParams, detector=None
) -> tuple[int, int, int, list[tuple[tuple[int, int, int, int], str, float, bool]]]:
    """Детекция по одному файлу.

    Возвращает ``(width, height, dpi, regions)``, где каждый элемент ``regions`` —
    ``((x1, y1, x2, y2) в координатах ОРИГИНАЛА, kind, chroma_frac, full_page)``.
    """
    dpi = read_dpi(path)
    if dpi is None or dpi < MIN_PLAUSIBLE_DPI:
        if params.default_dpi is None:
            raise ValueError(
                f"нет осмысленного тега разрешения (прочитано {dpi!r}), а --default-dpi не задан; "
                "подставить разрешение наугад значило бы промахнуться в разы и залить в CVAT "
                "кадры не того размера"
            )
        logger.debug("У %s разрешение %r, беру --default-dpi=%d", path.name, dpi, params.default_dpi)
        dpi = params.default_dpi

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("не читается как изображение")
    height, width = bgr.shape[:2]

    work = cv2.resize(
        bgr, (max(1, width // HALFTONE_DOWNSCALE), max(1, height // HALFTONE_DOWNSCALE)), interpolation=cv2.INTER_AREA
    )
    work_gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    boxes, cover = find_raster_boxes(work, work_gray, detector, params.min_region_frac, params.merge_gap)
    if not boxes:
        return width, height, dpi, []

    # Цвет бумаги — по рабочей копии всей полосы: он один на полосу и не зависит от того,
    # какую область мы сейчас классифицируем (мотивировка в color_kind.paper_color).
    paper = paper_color(work)
    work_h, work_w = work_gray.shape[:2]

    regions = []
    for box in boxes:
        crop = work[box[1] : box[3], box[0] : box[2]]
        kind, chroma_frac = classify(crop, paper, params.chroma_thr, params.color_frac_thr)
        full = cover or is_full_page(box, work_w, work_h, params.full_page_frac)
        original = scale_box(box, HALFTONE_DOWNSCALE)
        # Кламп: рабочая копия округлялась вниз, и умножение обратно могло вылезти за кадр
        # не больше чем на HALFTONE_DOWNSCALE - 1 пикселя.
        original = (original[0], original[1], min(original[2], width), min(original[3], height))
        regions.append((original, kind, chroma_frac, full))
    return width, height, dpi, regions


def _write_debug_overlay(debug_dir: Path, page: Page, bgr_path: Path, regions) -> None:
    """Уменьшенный оверлей с найденными областями и подписями ``kind chroma_frac``.

    Без этого порог color/grayscale не откалибровать: цифра в базе не говорит, ту ли
    область она описывает.
    """
    bgr = cv2.imread(str(bgr_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return
    scale = 1000.0 / max(bgr.shape[:2])
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for (x1, y1, x2, y2), kind, chroma_frac, full in regions:
        color = (0, 230, 118) if kind == "color" else (255, 176, 0)
        p1 = (int(x1 * scale), int(y1 * scale))
        p2 = (int(x2 * scale), int(y2 * scale))
        cv2.rectangle(small, p1, p2, color, 2)
        label = f"{kind} {chroma_frac:.3f}" + (" FULL" if full else "")
        cv2.putText(small, label, (p1[0] + 4, p1[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    out = debug_dir / f"{page.rel_path.replace('/', '__')}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), small, [cv2.IMWRITE_JPEG_QUALITY, 85])


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
        pack = upsert_pack(session, params.pack_name, params.pack_dir, years)
        pages = list(iter_pages(pack, params.only_year, params.only_issue))
        if params.limit is not None:
            pages = pages[: params.limit]

        for _year, _issue, page in tqdm(pages, desc="детекция", unit="полоса"):
            path = params.pack_dir / page.rel_path
            try:
                stamp = stat_stamp(path)
            except OSError as exc:
                stats.failed += 1
                tqdm.write(f"ОШИБКА {page.rel_path}: {exc}")
                continue

            if params.skip_detected and page.detected_at is not None and page.file_hash is not None:
                # Дешёвая проверка идёт первой: совпали размер и время правки — файл
                # считается прежним и не читается вовсе. Именно ради этого пропуска
                # ``stat`` и лежит в базе: перечитать полтерабайта пака ради хешей — часы.
                if not params.rehash_all and stat_matches(page, stamp):
                    stats.skipped += 1
                    continue
                # ``stat`` разошёлся — читаем и считаем хеш. Он вполне может совпасть:
                # файл могли просто скопировать заново, содержимое от этого не меняется.
                # Тогда обновляем отметку и идём дальше, не трогая разметку.
                stamp = full_stamp(path)
                if stamp.digest == page.file_hash:
                    apply_stamp(page, stamp)
                    session.commit()
                    stats.skipped += 1
                    continue

            if stamp.digest is None:
                stamp = full_stamp(path)
            if page.file_hash is not None and page.file_hash != stamp.digest:
                stats.changed += 1
                tqdm.write(f"ФАЙЛ ИЗМЕНИЛСЯ {page.rel_path}: разметка в CVAT к нему больше не относится")

            try:
                width, height, dpi, regions = detect_page(path, params, detector)
            except Exception as exc:  # noqa: BLE001 — одна битая полоса не должна валить прогон
                stats.failed += 1
                tqdm.write(f"ОШИБКА {page.rel_path}: {exc}")
                continue

            # Делитель и размеры уменьшенной копии здесь НЕ считаются: их выбирает to-cvat
            # по своему --cvat-dpi. Отсюда уходит только то, что прочитано из файла.
            page.width, page.height, page.dpi = width, height, dpi
            page.detected_at = _utcnow()
            # Отпечаток пишется ПОСЛЕ успешной детекции: полоса, на которой детекция
            # упала, не должна выглядеть обработанной для следующего прогона.
            apply_stamp(page, stamp)

            replace_raster_regions(
                session,
                page,
                [
                    RasterRegion(
                        x1=box[0],
                        y1=box[1],
                        x2=box[2],
                        y2=box[3],
                        kind=kind,
                        full_page=full,
                        chroma_frac=chroma_frac,
                        source=SOURCE_AUTO,
                    )
                    for box, kind, chroma_frac, full in regions
                ],
            )
            session.commit()

            stats.pages += 1
            stats.regions += len(regions)
            stats.color += sum(1 for _b, kind, _c, _f in regions if kind == "color")
            stats.grayscale += sum(1 for _b, kind, _c, _f in regions if kind == "grayscale")
            stats.full_page += sum(1 for _b, _k, _c, full in regions if full)

            if params.debug_dir is not None and regions:
                _write_debug_overlay(params.debug_dir, page, path, regions)

    return stats
