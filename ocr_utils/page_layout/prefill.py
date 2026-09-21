"""Набивка кэша surya пачкой: страницы готовятся в пуле, модель — в родителе, ответы — в кэш.

ЗАЧЕМ. Потребители разбора (сборка финальных PDF, детектор порчи геометрии, правка текстового
слоя) считают страницы в пуле процессов, а surya в пул не заворачивается: видеопамять одна.
Поэтому перед пулом кэш набивается здесь: воркеры декодируют или рендерят страницы и отдают
только кадр surya (мегабайт-два), родитель гонит модель пачками и пишет ответы. Дальше воркеры
читают кэш и модели не видят.

ЧТО НЕ ПЕРЕСЧИТЫВАЕТСЯ. Страница, чья запись в кэше подходит по отпечатку источника (размер,
mtime, номер страницы) или помечена ``legacy``, пропускается без чтения пикселей.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from tqdm import tqdm

from ocr_utils.page_layout.image import PageImage, Variant
from ocr_utils.page_layout.surya.cache import SuryaCache
from ocr_utils.page_layout.surya.model import SuryaLayoutModel

logger = logging.getLogger(__name__)

DEFAULT_JOBS = 16
# Сколько кадров готовится впереди модели: ограничивает память (по ~1-4 МБ на кадр).
DEFAULT_CHUNK = 64
WORKER_NICE = 10


@dataclass(frozen=True)
class PrefillRequest:
    """Одна страница для набивки: файл картинки либо страница PDF."""

    path: Path
    variant: Variant
    cache_name: str
    page_index: int | None = None  # None — файл картинки, иначе страница PDF
    default_dpi: int | None = None

    def image(self) -> PageImage:
        if self.page_index is None:
            return PageImage.from_file(self.path, self.variant, self.cache_name, self.default_dpi)
        return PageImage.from_pdf_file(self.path, self.page_index, self.variant, self.cache_name)


@dataclass
class PrefillStats:
    """Итог набивки."""

    requested: int = 0
    cached: int = 0  # уже были в кэше, пиксели не читались
    done: int = 0
    failed: int = 0


def pdf_requests(pdf_paths: Iterable[Path], variant: Variant) -> Iterator[PrefillRequest]:
    """Запросы на все страницы всех PDF: имя ``<stem>/p<index:04d>``."""
    import fitz

    for path in pdf_paths:
        with fitz.open(str(path)) as document:
            count = document.page_count
        for index in range(count):
            yield PrefillRequest(Path(path), variant, f"{Path(path).stem}/p{index:04d}", index)


def scan_requests(root: Path, variant: Variant, default_dpi: int | None = None) -> Iterator[PrefillRequest]:
    """Запросы на все картинки под корнем: имя — путь без расширения относительно корня."""
    from ocr_utils.scan_cropping.image_io import IMAGE_EXTS

    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and not path.name.startswith("."):
            yield PrefillRequest(path, variant, path.relative_to(root).with_suffix("").as_posix(), None, default_dpi)


def _init_worker() -> None:
    """Воркер: один поток счёта, без hugepage, пониженный приоритет (см. .claude/rules/gpu_and_pools.md)."""
    import cv2

    cv2.setNumThreads(1)
    try:
        os.nice(WORKER_NICE)
    except OSError:
        pass
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except Exception:  # noqa: BLE001 — без threadpoolctl просто чуть медленнее
        pass
    try:
        import numpy as np

        np._core.multiarray._set_madvise_hugepage(False)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 — старый numpy без этого флага
        pass


def _load_frame(request: PrefillRequest) -> "PageImage | Exception":
    """Воркер: страница → объект с посчитанным кадром surya и без остальных пикселей."""
    try:
        image = request.image()
        image.surya_frame
        image.drop_pixels()
        return image
    except Exception as error:  # noqa: BLE001 — одна битая страница не должна ронять набивку
        return error


def prefill(
    requests: Iterable[PrefillRequest],
    cache: SuryaCache,
    model: SuryaLayoutModel,
    jobs: int = DEFAULT_JOBS,
    chunk: int = DEFAULT_CHUNK,
    progress: bool = True,
) -> PrefillStats:
    """Набить кэш surya для страниц ``requests``.

    Args:
        requests: Страницы (см. :func:`pdf_requests`, :func:`scan_requests`).
        cache: Кэш на запись.
        model: Модель surya (грузится лениво в родителе после создания пула).
        jobs: Воркеров на декодирование/рендер.
        chunk: Сколько кадров готовится за раз перед моделью.
        progress: Показывать ли прогресс.

    Returns:
        Счётчики.
    """
    stats = PrefillStats()
    pending: list[PrefillRequest] = []
    for request in requests:
        stats.requested += 1
        # Попадание по отпечатку источника — без пикселей: from_file читает только заголовок.
        try:
            probe = request.image()
        except Exception as error:  # noqa: BLE001
            logger.warning("%s: не открывается (%s)", request.cache_name, error)
            stats.failed += 1
            continue
        if cache.load(probe) is not None:
            stats.cached += 1
            continue
        pending.append(request)
    if not pending:
        return stats

    context = multiprocessing.get_context("forkserver")
    bar = tqdm(total=len(pending), desc="surya", unit="стр", disable=not progress)
    with ProcessPoolExecutor(max_workers=max(1, jobs), mp_context=context, initializer=_init_worker) as pool:
        for start in range(0, len(pending), chunk):
            batch = pending[start : start + chunk]
            images: list[PageImage] = []
            for request, result in zip(batch, pool.map(_load_frame, batch)):
                if isinstance(result, Exception):
                    logger.warning("%s: не подготовлена (%s)", request.cache_name, result)
                    stats.failed += 1
                    continue
                images.append(result)
            blocks = model.predict([image.surya_frame for image in images])
            for image, found in zip(images, blocks):
                cache.save(image, found, model.name())
                stats.done += 1
            bar.update(len(batch))
    bar.close()
    return stats


__all__ = ["PrefillRequest", "PrefillStats", "pdf_requests", "prefill", "scan_requests"]
