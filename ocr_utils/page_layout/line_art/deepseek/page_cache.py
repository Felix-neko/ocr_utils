"""Кэш разбора полосы целиком DeepSeek-OCR-2 (затравки line art): кадры surya → воркер vLLM → блоки в кэш формата surya.

ЗАЧЕМ. DeepSeek-OCR-2 — второй разбор вёрстки. Его блоки ``image`` — затравки line art там, где surya
промахнулась: тонкий график 1977/04 0180_2R surya назвала ``Caption``, а DeepSeek нашёл как ``image``.
Модель живёт в отдельном окружении vLLM и в процесс детекторов не грузится, поэтому её ответы, как и
у surya, кладутся на диск заранее, а ``scan_markup detect`` читает их из кэша (``--deepseek-cache``).

КАДР И КЭШ. Модели подаётся тот же кадр, что и surya (:attr:`PageImage.surya_frame`, 150 dpi): на нём
DeepSeek находит те же рисунки, что и на кадре 290 dpi (проверено на 1977/04 0180_2R), а на полосах с
ложными находками пака-2 (строка бланка, «Примечание», рамка выходных данных) рисунков не предлагает.
Поэтому годится весь механизм кэша surya: запись ``<корень>/<вариант>/<имя>.json`` с отпечатком
файла и дайджестом кадра, блоки в пикселях кадра (:class:`~ocr_utils.page_layout.surya.cache.SuryaCache`
под СВОИМ корнем). Уверенности у DeepSeek нет — в блоках записана единица.

ПОРЯДОК. Пул готовит кадры недостающих страниц (PNG в ``<work>/frames``); воркер
``vllm_env/worker.py`` прогоняет их промптом ``markdown`` под сторожем памяти (:mod:`runner`; движок
поднимается один раз, ~2,5 мин) и дописывает ``<work>/out/markdown.jsonl`` — прерванный прогон
продолжается с того же места; ответы раскладываются в кэш.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
from tqdm import tqdm

from ocr_utils.page_layout.geometry import Box
from ocr_utils.page_layout.image import SourceStat, Variant
from ocr_utils.page_layout.line_art.deepseek.runner import run_worker
from ocr_utils.page_layout.prefill import PrefillRequest, _init_worker
from ocr_utils.page_layout.surya.blocks import LayoutBlocks, from_boxes
from ocr_utils.page_layout.surya.cache import CacheEntry, SuryaCache

logger = logging.getLogger(__name__)

# Имя модели в записи кэша.
MODEL_NAME = "DeepSeek-OCR-2"

# Уверенность, которой помечаются блоки DeepSeek (своей у модели нет; общий порог surya её пропускает).
CONFIDENCE = 1.0

# Промпт: штатный «Convert the document to markdown» в режиме grounding — рамки блоков вёрстки с метками.
PROMPT = "markdown"

# Потолок ответа на полосу целиком. Рамки блоков идут по ходу чтения, текст полосы журнала — до 3–4
# тысяч токенов; при умолчании воркера (1536, рассчитано на вырезки) рисунки внизу плотной полосы
# терялись бы. Обрезанные ответы считаются в ``truncated``.
MAX_NEW_TOKENS = 6144

DEFAULT_JOBS = 16


@dataclass(frozen=True)
class FrameMeta:
    """Что нужно записи кэша о кадре: имя страницы, отпечаток источника, дайджест и размер кадра."""

    variant: Variant
    cache_name: str
    frame_path: Path
    width: int
    height: int
    frame_dpi: int
    source: SourceStat | None
    digest: str


@dataclass
class PageCacheStats:
    """Итог набивки."""

    requested: int = 0
    cached: int = 0  # уже были в кэше, пиксели не читались
    done: int = 0
    truncated: int = 0  # ответ обрезан лимитом токенов
    failed: int = 0


def blocks_from_elements(elements: list[dict], width: int, height: int) -> LayoutBlocks:
    """Элементы ``parse.parse`` (рамки в пикселях кадра) → блоки формата surya.

    Args:
        elements: Словари ``x0, y0, x1, y1, label, text`` из ответа воркера.
        width: Ширина кадра, px.
        height: Высота кадра, px.

    Returns:
        Блоки ``(метка, CONFIDENCE, рамка)``; рамки обрезаны по кадру, пустые пропущены.
    """
    items = []
    for element in elements:
        box = Box(int(element["x0"]), int(element["y0"]), int(element["x1"]), int(element["y1"])).clipped(width, height)
        if box.area > 0:
            items.append((str(element["label"]), CONFIDENCE, box))
    return from_boxes(items, width, height)


def _frame_id(cache_name: str) -> str:
    """Имя файла кадра из имени страницы: слэши → «__» (``1977/04/0180_2R`` → ``1977__04__0180_2R``)."""
    return cache_name.replace("/", "__")


def _write_frame(job: tuple[PrefillRequest, Path]) -> "FrameMeta | Exception":
    """Воркер пула: страница → PNG кадра surya в папке кадров и его описание для записи кэша.

    Args:
        job: Запрос страницы и папка кадров.

    Returns:
        :class:`FrameMeta` либо исключение (одна битая страница не роняет набивку).
    """
    request, frames_dir = job
    try:
        image = request.image()
        frame = image.surya_frame
        path = frames_dir / f"{_frame_id(request.cache_name)}.png"
        # Кадр RGB → BGR для OpenCV; PNG без потерь — модель видит ровно то, от чего считан дайджест.
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return FrameMeta(
            request.variant,
            request.cache_name,
            path,
            int(frame.shape[1]),
            int(frame.shape[0]),
            image.surya_dpi,
            image.source,
            image.surya_frame_digest(),
        )
    except Exception as error:  # noqa: BLE001 — одна битая страница не должна ронять набивку
        return error


def prefill_page_cache(
    requests: Iterable[PrefillRequest],
    cache: SuryaCache,
    work_dir: Path,
    jobs: int = DEFAULT_JOBS,
    progress: bool = True,
) -> PageCacheStats:
    """Набить кэш разбора DeepSeek-OCR-2 для страниц ``requests``.

    Args:
        requests: Страницы (``page_layout.prefill.scan_requests`` / ``pdf_requests``).
        cache: Кэш на запись — под корнем DeepSeek, НЕ под корнем surya.
        work_dir: Рабочая папка (на SSD): кадры PNG, задания, ответы и логи воркера; после набивки
            её можно удалить.
        jobs: Воркеров на подготовку кадров (декодирование TIFF с медленного диска).
        progress: Показывать ли прогресс подготовки кадров.

    Returns:
        Счётчики.
    """
    stats = PageCacheStats()
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

    # Кадры недостающих страниц — пулом.
    frames_dir = work_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    metas: list[FrameMeta] = []
    context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=max(1, jobs), mp_context=context, initializer=_init_worker) as pool:
        results = pool.map(_write_frame, [(request, frames_dir) for request in pending], chunksize=4)
        for request, result in tqdm(
            zip(pending, results), total=len(pending), desc="кадры", unit="стр", disable=not progress
        ):
            if isinstance(result, Exception):
                logger.warning("%s: кадр не подготовлен (%s)", request.cache_name, result)
                stats.failed += 1
                continue
            metas.append(result)

    # Задания воркеру и прогон модели (ответы дописываются в out/markdown.jsonl).
    jobs_file = work_dir / "jobs.jsonl"
    jobs_file.write_text(
        "".join(json.dumps({"id": meta.cache_name, "crop": str(meta.frame_path)}) + "\n" for meta in metas),
        encoding="utf-8",
    )
    out_dir = work_dir / "out"
    run_worker(jobs_file, out_dir, PROMPT, ("--max-new-tokens", str(MAX_NEW_TOKENS)))

    # Ответы → записи кэша в пикселях кадра.
    by_name = {meta.cache_name: meta for meta in metas}
    written: set[str] = set()
    for line in (out_dir / f"{PROMPT}.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        meta = by_name.get(record["id"])
        if meta is None or meta.cache_name in written:
            continue  # ответ прошлого прогона на страницу вне этого запроса или повтор
        blocks = blocks_from_elements(record.get("elements", []), meta.width, meta.height)
        entry = CacheEntry(
            meta.variant, meta.cache_name, blocks, meta.frame_dpi, meta.source, meta.digest, False, MODEL_NAME
        )
        cache.write(entry)
        written.add(meta.cache_name)
        stats.truncated += record.get("finish_reason") == "length"
    stats.done = len(written)
    stats.failed += len(metas) - stats.done
    return stats


__all__ = ["MODEL_NAME", "PageCacheStats", "blocks_from_elements", "prefill_page_cache"]
