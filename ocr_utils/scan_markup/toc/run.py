"""Прогон детектора оглавлений по паку: окно полос каждого выпуска -> признаки -> решение -> база.

Читает дерево выпусков из базы (её заполнил ``detect``), меряет только полосы окна в пуле
процессов (tesseract — CPU, GPU не нужен: разметка surya берётся из кэша, при промахе
признак surya равен нулю и пишется предупреждение) и записывает решение в ``pages``:
``is_toc``, ``is_year_index``, ``toc_score``, ``toc_source``, ``toc_version``,
``toc_detected_at``. Полосам вне окна и отвергнутым пишется ``False`` в оба признака с той
же версией — «искали, не оглавление». Ручное решение из CVAT (``toc_source = cvat``) не
трогается.
"""

from __future__ import annotations

import csv
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import cv2

from ocr_utils.scan_markup.db.models import Issue, Page, YearPackage
from ocr_utils.scan_markup.db.repo import require_pack
from ocr_utils.scan_markup.toc import KIND_CONTENTS, KIND_INDEX, SOURCE_AUTO, SOURCE_CVAT, TOC_VERSION, flags_from_kind
from ocr_utils.scan_markup.toc.decide import Decision, Thresholds, decide_issue, in_window
from ocr_utils.scan_markup.toc.features import METRIC_NAMES, PageFeatures, page_features
from ocr_utils.scan_markup.toc.labels import Label, lookup

logger = logging.getLogger(__name__)

WORKER_NICE = 10
# Библиотеки, которые сами разойдутся по всем ядрам поверх пула, если их не остановить.
THREAD_LIMIT_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")

# Расширения, под которыми полоса может лежать в другом корне (``--image-root``): там
# заострённые JPEG, а в базе — имена оригиналов TIFF.
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


@dataclass
class TocParams:
    db_path: Path
    pack_name: str
    pack_dir: Path
    layout_cache_dir: Path | None = None
    # Брать полосы из этого корня (та же раскладка год/выпуск/полоса, расширение любое):
    # заострённые JPEG на SSD читаются в разы быстрее оригиналов с медленного NTFS-3G.
    image_root: Path | None = None
    jobs: int = 16
    thresholds: Thresholds = field(default_factory=Thresholds)
    only_year: str | None = None
    only_issue: str | None = None
    # Пропускать выпуски, у которых все полосы уже считаны текущей версией детектора.
    skip_detected: bool = False
    dry_run: bool = False
    csv_path: Path | None = None
    debug_dir: Path | None = None
    # Контактные листы НАЙДЕННОГО: <found_dir>/содержание/<год>_<выпуск>.jpg и
    # <found_dir>/указатели/<год>_<выпуск>.jpg (последние — только у выпусков с указателем).
    found_dir: Path | None = None
    default_dpi: int = 600
    progress: bool = True


@dataclass
class IssueResult:
    year: str
    issue: str
    number: int | None
    n_pages: int
    features: list[PageFeatures]
    decisions: list[Decision]

    @property
    def rel_dir(self) -> str:
        return f"{self.year}/{self.issue}"

    def kinds(self) -> dict[str, int]:
        counts = {KIND_CONTENTS: 0, KIND_INDEX: 0}
        for decision in self.decisions:
            if decision.kind is not None:
                counts[decision.kind] += 1
        return counts


@dataclass
class TocStats:
    issues: int = 0
    skipped: int = 0
    pages_measured: int = 0
    failed: int = 0
    contents: int = 0
    index: int = 0
    # Выпуски без единой полосы «Содержание» — красный флаг, их смотреть глазами.
    issues_without_contents: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Job:
    path: Path
    rel_path: str
    order_index: int
    idx_from_end: int
    layout_cache_dir: Path | None
    want_thumbnail: bool
    default_dpi: int


def _init_worker() -> None:
    cv2.setNumThreads(1)
    try:
        os.nice(WORKER_NICE)
    except OSError:
        pass
    # numpy помечает большие массивы MADV_HUGEPAGE, и ядро при defrag=madvise уплотняет
    # память под них синхронно; дюжина воркеров, разом выделяющих копии полосы, встаёт в очередь.
    try:
        import numpy as np

        np._core.multiarray._set_madvise_hugepage(False)  # type: ignore[attr-defined]
    except Exception:
        pass


def _worker(job: _Job) -> PageFeatures:
    return page_features(
        job.path,
        job.rel_path,
        job.order_index,
        job.idx_from_end,
        job.layout_cache_dir,
        want_thumbnail=job.want_thumbnail,
        default_dpi=job.default_dpi,
    )


def _with_progress(iterator: Iterable, total: int, enabled: bool) -> Iterator:
    if not enabled:
        yield from iterator
        return
    from tqdm import tqdm

    yield from tqdm(iterator, total=total, desc="оглавления", unit="полоса")


def measure(jobs: Sequence[_Job], workers: int, progress: bool) -> list[PageFeatures]:
    """Признаки по списку заданий, в том же порядке."""
    if workers <= 1 or len(jobs) <= 1:
        return list(_with_progress((_worker(job) for job in jobs), len(jobs), progress))
    for name in THREAD_LIMIT_VARS:
        os.environ.setdefault(name, "1")
    # forkserver, а не fork: в родителе может быть поднят torch, а форк процесса с
    # инициализированной CUDA — верный способ получить зависший воркер.
    context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker) as pool:
        return list(_with_progress(pool.map(_worker, jobs, chunksize=1), len(jobs), progress))


def resolve_image(params: TocParams, page: Page) -> Path:
    """Файл полосы: из ``--image-root`` под любым расширением, иначе оригинал из пака."""
    if params.image_root is not None:
        base = params.image_root / Path(page.source_rel_path).with_suffix("")
        for suffix in IMAGE_SUFFIXES:
            candidate = base.with_suffix(suffix)
            if candidate.is_file():
                return candidate
        logger.warning("%s: в %s полосы нет, беру оригинал из пака", page.source_rel_path, params.image_root)
    return params.pack_dir / page.source_rel_path


def _issue_done(pages: Sequence[Page]) -> bool:
    return all(page.toc_version == TOC_VERSION for page in pages)


def _apply(issue_pages: Sequence[Page], decisions: Sequence[Decision], now: datetime) -> None:
    by_index = {decision.order_index: decision for decision in decisions}
    for page in issue_pages:
        if page.toc_source == SOURCE_CVAT:
            # Решение человека: версию обновляем (полоса «просмотрена»), вид не трогаем.
            page.toc_version = TOC_VERSION
            page.toc_detected_at = now
            continue
        decision = by_index.get(page.order_index)
        page.is_toc, page.is_year_index = flags_from_kind(decision.kind if decision is not None else None)
        page.toc_score = decision.score if decision is not None else None
        page.toc_source = SOURCE_AUTO
        page.toc_version = TOC_VERSION
        page.toc_detected_at = now


def run_toc(
    params: TocParams, session_factory, labels: dict[str, Label] | None = None
) -> tuple[TocStats, list[IssueResult]]:
    """Полный прогон по паку из базы. Возвращает сводку и результаты по выпускам (для отчётов)."""
    stats = TocStats()
    results: list[IssueResult] = []
    with session_factory() as session:
        pack = require_pack(session, params.pack_name)
        plan: list[tuple[YearPackage, Issue, list[Page], list[_Job]]] = []
        for year in pack.year_packages:
            if params.only_year is not None and year.name != params.only_year:
                continue
            for issue in year.issues:
                if params.only_issue is not None and issue.name != params.only_issue:
                    continue
                pages = sorted(issue.pages, key=lambda p: p.order_index)
                if not pages:
                    continue
                if params.skip_detected and _issue_done(pages):
                    stats.skipped += 1
                    continue
                last = pages[-1].order_index
                jobs = [
                    _Job(
                        resolve_image(params, page),
                        page.source_rel_path,
                        page.order_index,
                        last - page.order_index,
                        params.layout_cache_dir,
                        params.debug_dir is not None or params.found_dir is not None,
                        params.default_dpi,
                    )
                    for page in pages
                    if in_window(page.order_index, last - page.order_index, params.thresholds)
                ]
                plan.append((year, issue, pages, jobs))

        all_jobs = [job for _, _, _, jobs in plan for job in jobs]
        logger.info("Выпусков: %d, полос в окне: %d", len(plan), len(all_jobs))
        features = measure(all_jobs, params.jobs, params.progress)
        stats.pages_measured = len(features)
        cursor = 0
        now = datetime.now(timezone.utc)
        for year, issue, pages, jobs in plan:
            issue_features = features[cursor : cursor + len(jobs)]
            cursor += len(jobs)
            stats.failed += sum(1 for f in issue_features if f.error)
            for feature in issue_features:
                if feature.error:
                    logger.warning("%s: %s", feature.rel_path, feature.error)
            decisions = decide_issue(issue_features, issue.number, params.thresholds)
            result = IssueResult(year.name, issue.name, issue.number, len(pages), issue_features, decisions)
            results.append(result)
            counts = result.kinds()
            stats.issues += 1
            stats.contents += counts[KIND_CONTENTS]
            stats.index += counts[KIND_INDEX]
            if counts[KIND_CONTENTS] == 0:
                stats.issues_without_contents.append(result.rel_dir)
            if not params.dry_run:
                _apply(pages, decisions, now)
                session.commit()

    if params.csv_path is not None:
        write_csv(params.csv_path, results, labels)
    if params.debug_dir is not None:
        write_sheets(params.debug_dir, results, labels)
    if params.found_dir is not None:
        write_found(params.found_dir, results)
    return stats, results


# --- отчёты ---------------------------------------------------------------------------

CSV_FIELDS = (
    "год",
    "выпуск",
    "полоса",
    "order_index",
    "idx_from_end",
    *METRIC_NAMES,
    "вид",
    "score",
    "почему",
    "метка",
    "ошибка",
)


def csv_rows(results: Sequence[IssueResult], labels: dict[str, Label] | None) -> Iterator[dict]:
    for result in results:
        by_index = {d.order_index: d for d in result.decisions}
        for feature in result.features:
            decision = by_index[feature.order_index]
            label = lookup(labels, feature.rel_path) if labels else None
            yield {
                "год": result.year,
                "выпуск": result.issue,
                "полоса": feature.rel_path,
                "order_index": feature.order_index,
                "idx_from_end": feature.idx_from_end,
                **feature.metrics(),
                "вид": decision.kind or "",
                "score": f"{decision.score:.3f}",
                "почему": decision.reason,
                "метка": label.label if label else "",
                "ошибка": feature.error,
            }


def write_csv(path: Path, results: Sequence[IssueResult], labels: dict[str, Label] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in csv_rows(results, labels):
            writer.writerow(row)


def write_sheets(debug_dir: Path, results: Sequence[IssueResult], labels: dict[str, Label] | None) -> None:
    """Контактный лист окна на выпуск: миниатюры с подписью «номер полосы, вид, score, метка»."""
    from ocr_utils.scan_markup.toc.sheet import contact_sheet

    for result in results:
        out = debug_dir / result.year / f"{result.issue}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        contact_sheet(result, labels).save(out, quality=85)


FOUND_SUBDIRS = {KIND_CONTENTS: "содержание", KIND_INDEX: "указатели"}


def write_found(found_dir: Path, results: Sequence[IssueResult]) -> None:
    """Контактные листы найденных полос: по выпуску на лист, «Содержание» и указатели врозь."""
    from ocr_utils.scan_markup.toc.sheet import found_sheet

    for kind, subdir in FOUND_SUBDIRS.items():
        for result in results:
            sheet = found_sheet(result, kind)
            if sheet is None:
                continue
            out = found_dir / subdir / f"{result.year}_{result.issue}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(out, quality=85)


__all__ = [
    "CSV_FIELDS",
    "FOUND_SUBDIRS",
    "IssueResult",
    "TocParams",
    "TocStats",
    "csv_rows",
    "resolve_image",
    "run_toc",
    "write_csv",
    "write_found",
]
