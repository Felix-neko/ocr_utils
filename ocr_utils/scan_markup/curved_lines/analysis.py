"""Прогон детекторов по дереву полос.

СХЕМА ИСПОЛНЕНИЯ — та же, что в ``orientation.analysis``: разжатие полосы и CPU-детекторы
едут в пул процессов, GPU-детекторы остаются в родителе и работают пачками (видеомамять
одна на всех, CLAUDE.md). Воркер отдаёт родителю не массив, а JPEG уменьшенной копии.

КЭШ. Если задан каталог кэша, воркер сперва ищет готовые измерения каждого детектора и
разжимает полосу только ради недостающих; при полном попадании полоса не читается вовсе.
Так повторный прогон по паку с другими порогами занимает минуты, а не часы.
"""

from __future__ import annotations

import ctypes
import logging
import multiprocessing
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import cv2

from ocr_utils.scan_markup.curved_lines.cache import PageCache
from ocr_utils.scan_markup.curved_lines.detectors import DETECTORS, Detector, GpuPage, Measure
from ocr_utils.scan_markup.curved_lines.flags import COMBO, Thresholds, combine
from ocr_utils.page_layout.orientation.image_io import decode_gpu_jpeg, read_frame

logger = logging.getLogger(__name__)

# Воркеры уступают дорогу интерактивной работе: прогон по паку идёт долго, и всё это время
# машина не должна быть занята под завязку.
WORKER_NICE = 10
THREAD_LIMIT_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
# Заданий на воркера в работе: двух хватает, чтобы воркер не простаивал, и мало, чтобы
# готовые JPEG не копились в памяти родителя (см. ``_bounded``).
IN_FLIGHT_PER_WORKER = 2


def _trim_heap() -> None:
    """Возвращает системе память из фрагментированных арен glibc после GPU-пачки."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


@dataclass
class PageResult:
    """Что известно об одной полосе после всех детекторов."""

    rel_path: str
    path: Path
    width: int = 0
    height: int = 0
    measures: dict[str, Measure] = field(default_factory=dict)
    combo: Measure | None = None
    label: str = ""
    seconds: dict[str, float] = field(default_factory=dict)
    error: str = ""

    def metrics_by_detector(self) -> dict[str, dict[str, float]]:
        return {name: dict(measure.metrics) for name, measure in self.measures.items() if not measure.silent}

    @property
    def flagged(self) -> bool:
        return self.combo is not None and self.combo.flag


@dataclass(frozen=True)
class Job:
    path: Path
    rel_path: str
    cpu_names: tuple[str, ...]
    gpu_names: tuple[str, ...]
    default_dpi: int
    gpu_side: int
    keep_raw: bool
    cache_root: Path | None


def _init_worker() -> None:
    cv2.setNumThreads(1)
    try:
        os.nice(WORKER_NICE)
    except OSError:
        pass
    # numpy помечает большие массивы MADV_HUGEPAGE, и ядро при defrag=madvise уплотняет
    # память под них синхронно; в одиночку незаметно, а дюжина воркеров, разом выделяющих
    # копии полосы, встаёт в очередь (замер в ocr_utils/dewarp: 0.2 с против 8-10 с).
    try:
        import numpy as np

        np._core.multiarray._set_madvise_hugepage(False)  # type: ignore[attr-defined]
    except Exception:
        pass


def _from_cache(payload: dict, keep_raw: bool) -> Measure:
    return Measure(
        metrics={k: float(v) for k, v in payload.get("metrics", {}).items()},
        note=str(payload.get("note", "")),
        silent=bool(payload.get("silent", False)),
        raw=payload.get("raw") if keep_raw else None,
    )


def _to_cache(measure: Measure, width: int, height: int) -> dict:
    return {
        "metrics": measure.metrics,
        "note": measure.note,
        "silent": measure.silent,
        "raw": measure.raw,
        "size": [width, height],
    }


def _worker(job: Job) -> tuple[PageResult, bytes | None]:
    """CPU-этап одной полосы. Не бросает: сбой одной полосы не должен ронять прогон."""
    result = PageResult(rel_path=job.rel_path, path=job.path)
    cache = PageCache(job.cache_root) if job.cache_root is not None else None

    missing_cpu: list[str] = []
    need_gpu = False
    for name in (*job.cpu_names, *job.gpu_names):
        payload = cache.load(job.path, name, DETECTORS[name].cache_key) if cache is not None else None
        if payload is not None:
            result.measures[name] = _from_cache(payload, job.keep_raw)
            result.seconds[name] = 0.0
            size = payload.get("size")
            if size and not result.width:
                result.width, result.height = int(size[0]), int(size[1])
        elif name in job.cpu_names:
            missing_cpu.append(name)
        else:
            need_gpu = True
    if not missing_cpu and not need_gpu:
        return result, None

    try:
        frame, payload = read_frame(
            job.path, job.rel_path, default_dpi=job.default_dpi, gpu_side=job.gpu_side if need_gpu else 0
        )
    except Exception as error:
        result.error = f"чтение: {error}"
        return result, None
    result.width, result.height = frame.width, frame.height
    for name in missing_cpu:
        detector = DETECTORS[name]
        started = time.perf_counter()
        try:
            measure = detector.run(frame, True)
        except Exception as error:
            result.error = f"{name}: {error}"
            measure = Measure(note=f"ошибка: {error}", silent=True)
        result.seconds[name] = time.perf_counter() - started
        if cache is not None:
            cache.store(job.path, name, detector.cache_key, _to_cache(measure, frame.width, frame.height))
        result.measures[name] = measure if job.keep_raw else measure.stripped()
    return result, payload if need_gpu else None


def _with_progress(iterator: Iterable, total: int, enabled: bool, desc: str) -> Iterator:
    if not enabled:
        yield from iterator
        return
    from tqdm import tqdm

    yield from tqdm(iterator, total=total, desc=desc, unit="полоса")


def _bounded(
    pool: ProcessPoolExecutor, jobs: Sequence[Job], in_flight: int
) -> Iterator[tuple[PageResult, bytes | None]]:
    """Результаты по порядку, но в работе не больше ``in_flight`` заданий: иначе очередь
    готовых JPEG растёт без границы, пока родитель занят GPU."""
    queue: deque = deque()
    remaining = iter(jobs)
    for job in islice(remaining, in_flight):
        queue.append(pool.submit(_worker, job))
    while queue:
        result = queue.popleft().result()
        next_job = next(remaining, None)
        if next_job is not None:
            queue.append(pool.submit(_worker, next_job))
        yield result


def _run_pool(
    jobs: Sequence[Job], workers: int, progress: bool, desc: str
) -> Iterator[tuple[PageResult, bytes | None]]:
    if workers <= 1 or len(jobs) <= 1:
        yield from _with_progress((_worker(job) for job in jobs), len(jobs), progress, desc)
        return
    for name in THREAD_LIMIT_VARS:
        os.environ.setdefault(name, "1")
    # forkserver, а не fork: в родителе может быть поднят torch, а форк процесса с
    # инициализированной CUDA — верный способ получить зависший воркер.
    context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker) as pool:
        stream = _bounded(pool, jobs, max(IN_FLIGHT_PER_WORKER * workers, 8))
        yield from _with_progress(stream, len(jobs), progress, desc)


def analyse(
    tasks: Sequence[tuple[Path, str]],
    detectors: Sequence[Detector],
    workers: int,
    default_dpi: int = 600,
    gpu_side: int = 1536,
    gpu_batch: int = 16,
    cache_root: Path | None = None,
    keep_raw: bool = False,
    gpu_options: dict | None = None,
    progress: bool = True,
    desc: str = "кривизна",
) -> list[PageResult]:
    """Прогон набора детекторов по списку полос. Порядок результатов — как на входе."""
    cpu_names = tuple(d.name for d in detectors if d.stage == "cpu")
    gpu_detectors = [d for d in detectors if d.stage == "gpu"]
    gpu_names = tuple(d.name for d in gpu_detectors)
    options = dict(gpu_options or {})
    batches = {d.name: d.make_batch(cache_dir=cache_root, **options) for d in gpu_detectors}
    cache = PageCache(cache_root) if cache_root is not None else None

    jobs = [
        Job(path, rel, cpu_names, gpu_names, default_dpi, gpu_side if gpu_detectors else 0, keep_raw, cache_root)
        for path, rel in tasks
    ]
    results: list[PageResult] = []
    pending: list[tuple[PageResult, GpuPage]] = []

    def flush() -> None:
        if not pending:
            return
        for name, batch in batches.items():
            todo = [(result, page) for result, page in pending if name not in result.measures]
            if not todo:
                continue
            started = time.perf_counter()
            measures = batch([page for _, page in todo], True)
            if len(measures) != len(todo):
                raise RuntimeError(f"{name} вернул {len(measures)} мер на {len(todo)} полос")
            spent = (time.perf_counter() - started) / len(todo)
            for (result, page), measure in zip(todo, measures):
                if cache is not None:
                    cache.store(
                        page.path, name, DETECTORS[name].cache_key, _to_cache(measure, result.width, result.height)
                    )
                result.measures[name] = measure if keep_raw else measure.stripped()
                result.seconds[name] = spent
        results.extend(result for result, _ in pending)
        pending.clear()
        _trim_heap()

    for result, payload in _run_pool(jobs, workers, progress, desc):
        if payload is None:
            results.append(result)
            continue
        pending.append((result, GpuPage(result.path, result.rel_path, decode_gpu_jpeg(payload))))
        if len(pending) >= gpu_batch:
            flush()
    flush()
    # Порядок — как на входе: полосы с GPU-этапом догоняют остальных пачками.
    order = {rel: index for index, (_, rel) in enumerate(tasks)}
    results.sort(key=lambda item: order.get(item.rel_path, len(order)))
    return results


def apply_flags(results: Sequence[PageResult], thresholds: Thresholds, votes: int, strong: float) -> None:
    """Флаги и score по порогам, затем сводный вердикт — для каждой полосы."""
    for result in results:
        for name in list(result.measures):
            result.measures[name] = thresholds.apply(name, result.measures[name])
        result.combo = combine(
            result.measures,
            votes,
            strong,
            [n for n in result.measures if DETECTORS[n].solo],
            [n for n in result.measures if DETECTORS[n].sufficient],
        )


def flagged_counts(results: Sequence[PageResult], detector_names: Sequence[str]) -> dict[str, tuple[int, int, int]]:
    """На детектор: (флагнуто, молчит, ошибок)."""
    counts: dict[str, tuple[int, int, int]] = {}
    for name in [*detector_names, COMBO]:
        flagged = silent = errors = 0
        for result in results:
            measure = result.combo if name == COMBO else result.measures.get(name)
            if measure is None:
                errors += 1
            elif measure.silent:
                silent += 1
            elif measure.flag:
                flagged += 1
        counts[name] = (flagged, silent, errors)
    return counts
