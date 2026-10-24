"""Кэш стенда: одно чтение TIFF полосы с медленного диска → серый 300 dpi, цвет 150 dpi и находки растрового детектора ``page_layout`` (surya — только из кэша) на SSD."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from research.photo_bounds.data import CacheLayout, PageRef

log = logging.getLogger(__name__)

# Разрешение серой копии, по которой работает стенд. Эталон CVAT точен до ``divisor`` = 8 px оригинала 600 dpi
# (0.34 мм) — это 4 px копии 300 dpi; растровая сетка пака-1 (~0.4–0.5 мм) на 300 dpi ещё различима, а размытие
# σ 0.35 мм её гасит. Полный кадр не хранится: 40 МБ на полосу против ~5 МБ.
GRAY_DPI = 300
# Цветная копия для оверлеев и признака цвета.
BGR_DPI = 150


@dataclass(frozen=True)
class CacheJob:
    """Задание кэша одной полосы: полоса, корень пака, раскладка кэша, корень кэша surya."""

    page: PageRef
    pack_root: Path
    layout: CacheLayout
    surya_root: Path


def init_worker() -> None:
    """Инициализатор воркера пула: без hugepage и без потоков BLAS/OpenCV (правило проекта для больших массивов)."""
    import threadpoolctl

    np._core.multiarray._set_madvise_hugepage(False)
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)


def cache_page(job: CacheJob) -> dict:
    """Читает TIFF полосы один раз, гоняет растровый детектор ``page_layout`` и пишет копии и находки на SSD.

    Детектор — ``PageLayout`` с ``Find.RASTER`` и умолчаниями ``LayoutOptions`` (как ``scan_markup detect``); блоки
    surya — только из кэша ``surya_root`` (вариант ``scan``, полон на пак-1); промах кэша — разбор по одним пикселям
    (``surya_used = false`` в ``detector.json``).

    Args:
        job: Задание.

    Returns:
        Строка сводки: полоса, статус, время, число находок.
    """
    from ocr_utils.page_layout.analysis import Find, LayoutOptions, PageLayout
    from ocr_utils.page_layout.image import PageImage, Variant
    from ocr_utils.page_layout.surya.blocks import PICTURE_LABELS
    from ocr_utils.page_layout.surya.cache import SuryaCache

    page, layout = job.page, job.layout
    row = {"page": page.page_key, "status": "ok"}
    if layout.detector(page.page_key).exists() and layout.gray(page.page_key).exists():
        return row | {"status": "skipped"}
    t0 = time.perf_counter()
    try:
        image = PageImage.from_file(job.pack_root / page.rel_path, Variant.SCAN, cache_name=page.page_key)
        analysis = PageLayout(image, {Find.RASTER}, LayoutOptions(), order_index=page.order_index)
        analysis.prepare(SuryaCache(job.surya_root, readonly=True))
        surya_used = not analysis.needs_surya
        if analysis.needs_surya:
            analysis.finish(None)
        out_dir = layout.page_dir(page.page_key)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(layout.gray(page.page_key)), image.gray_at(GRAY_DPI), [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(str(layout.bgr(page.page_key)), image.bgr_at(BGR_DPI), [cv2.IMWRITE_JPEG_QUALITY, 92])
        pictures = []
        if analysis.blocks is not None:
            native = analysis.blocks.scaled_to(image.width, image.height)
            pictures = [list(map(int, b.as_tuple())) for b in native.boxes(PICTURE_LABELS, min_confidence=0.0)]
        payload = {
            "page": page.page_key,
            "width": image.width,
            "height": image.height,
            "dpi": image.dpi,
            "surya_used": surya_used,
            "raster": [_region_json(r) for r in analysis.raster_pics],
            "stamps": [_region_json(r) for r in analysis.stamp_suspects],
            "surya_pictures": pictures,
        }
        layout.detector(page.page_key).write_text(json.dumps(payload, ensure_ascii=False, indent=1))
        row |= {"raster": len(analysis.raster_pics), "surya_used": surya_used}
    except Exception as exc:  # полоса с ошибкой не роняет прогон
        log.exception("%s: ошибка кэша", page.page_key)
        row |= {"status": "error", "message": repr(exc)}
    row["seconds"] = round(time.perf_counter() - t0, 2)
    return row


def _region_json(region) -> dict:
    """Область ``page_layout`` → JSON: рамка в px оригинала, вид, полностраничность."""
    b = region.box
    return {
        "box": [int(b.x0), int(b.y0), int(b.x1), int(b.y1)],
        "kind": region.kind.value,
        "full_page": bool(region.full_page),
    }


@dataclass(frozen=True)
class PageData:
    """Полоса из кэша стенда.

    Поля:
        page: Полоса набора.
        gray: Серый 300 dpi.
        scale: px копии на px оригинала (``GRAY_DPI / dpi``).
        detector: Находки детектора (``detector.json``).
    """

    page: PageRef
    gray: np.ndarray
    scale: float
    detector: dict


def load_page(layout: CacheLayout, page: PageRef) -> PageData:
    """Читает полосу из кэша стенда.

    Args:
        layout: Раскладка кэша.
        page: Полоса.

    Returns:
        Данные полосы.
    """
    gray = cv2.imread(str(layout.gray(page.page_key)), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(layout.gray(page.page_key))
    detector = json.loads(layout.detector(page.page_key).read_text())
    return PageData(page, gray, GRAY_DPI / float(page.dpi), detector)
