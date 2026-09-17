"""Обход пака по выпускам: полосы оглавления -> список статей -> остальные полосы -> fallback.

На выпуск: (1) полосы, помеченные в базе как «Содержание» или указатель, распознаются этапом
``toc`` и сливаются в оглавление выпуска (``toc.json`` / ``toc.md``); (2) остальные полосы идут
этапом ``page`` с рубриками и статьями «Содержания» в промпте; (3) если модель на обычной полосе
увидела оглавление, которого в базе нет (и нет вето ``force_is_not_toc``), — предупреждение и,
по ``--on-missed-toc``, повтор выпуска: найденные полосы распознаются как оглавление, список
пересобирается, обычные полосы идут заново (у них меняется ``toc_hash``, готовые с прежним
списком не считаются сделанными). Один круг повтора на выпуск.

Запросы к сети — в пуле потоков ``--jobs`` внутри этапа; выпуски идут последовательно, потому
что этап 2 зависит от этапа 1. В базу ничего не пишется: пропущенные оглавления складываются в
``missed_toc.txt``, а теги ставит человек в CVAT.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import click

from ocr_utils.external_ocr_services import toc as toc_module
from ocr_utils.external_ocr_services.client import OpenRouterClient
from ocr_utils.external_ocr_services.models import ModelSpec
from ocr_utils.external_ocr_services.ocr import (
    PageJob,
    RunOptions,
    is_done,
    load_result,
    recognise_page,
    recognise_with_second_pass,
)
from ocr_utils.external_ocr_services.pages import PageFlags, flags_for, group_by_issue, list_pages
from ocr_utils.external_ocr_services.schema import PageResult

logger = logging.getLogger(__name__)

MISSED_LIST = "missed_toc.txt"
TOC_JSON = "toc.json"
TOC_MD = "toc.md"
ON_MISSED_CHOICES = ("ask", "redo", "skip")

# Колонки сводки прогона; пересобирается по всем .meta.json под out-dir.
SUMMARY_FIELDS = (
    "page",
    "stage",
    "toc_kind_expected",
    "toc_hash",
    "articles_in_prompt",
    "second_pass_reason",
    "second_pass_chosen",
    "model",
    "provider",
    "json_mode_used",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cost_usd",
    "latency_s",
    "attempts",
    "page_number",
    "toc_kind",
    "title",
    "title_in_list",
    "content_chars",
    "toc_articles",
    "tags",
    "parse_error",
    "error",
)


@dataclass
class PipelineParams:
    in_dir: Path
    out_dir: Path
    options: RunOptions
    flags: dict[str, PageFlags] | None = None  # из базы или списков; None — база не задана
    jobs: int = 4
    skip_done: bool = False
    on_missed_toc: str = "ask"
    pages_file: Path | None = None
    only_year: str | None = None
    only_issue: str | None = None
    limit: int | None = None


@dataclass
class PipelineStats:
    issues: int = 0
    pages: int = 0
    requests: int = 0
    reused: int = 0
    failed: int = 0
    cost_usd: float = 0.0
    toc_pages: int = 0
    unknown_pages: int = 0  # полос без записи в базе
    second_passes: int = 0  # полос, ушедших во второй проход
    second_pass_kept_first: int = 0  # из них оставлен первый проход (страховка)
    missed: list[str] = field(default_factory=list)  # «выпуск: полосы», где оглавление не в базе
    redone_issues: list[str] = field(default_factory=list)


def _recognise_many(
    client: OpenRouterClient,
    spec: ModelSpec,
    params: PipelineParams,
    jobs: list[PageJob],
    reuse: bool,
    stats: PipelineStats,
) -> dict[Path, PageResult | None]:
    """Полосы пачкой в пуле потоков; готовые (по ``is_done``) не запрашиваются, если ``reuse``."""
    results: dict[Path, PageResult | None] = {}
    todo: list[PageJob] = []
    for job in jobs:
        if reuse and is_done(params.out_dir, job):
            results[job.rel] = load_result(params.out_dir, job)
            if results[job.rel] is not None:
                stats.reused += 1
                continue
        todo.append(job)
    if not todo:
        return results

    def work(job: PageJob) -> tuple[PageJob, dict, PageResult | None]:
        recognise = recognise_with_second_pass if params.options.second_pass else recognise_page
        meta, result = recognise(client, spec, params.in_dir / job.rel, job, params.out_dir, params.options)
        return job, meta, result

    with ThreadPoolExecutor(max_workers=max(1, params.jobs)) as pool:
        for job, meta, result in pool.map(work, todo):
            stats.requests += 1
            stats.cost_usd += float(meta.get("cost_usd") or 0.0)
            if meta.get("second_pass_reason"):
                stats.second_passes += 1
                stats.second_pass_kept_first += meta.get("second_pass_chosen") == "pass1"
            status = meta.get("error") or meta.get("parse_error") or "ok"
            if result is None:
                stats.failed += 1
            logger.info(
                "%s [%s] %s: %s tok → %s tok, $%s, %.0f с",
                job.rel,
                job.stage,
                status,
                meta.get("prompt_tokens", "-"),
                meta.get("completion_tokens", "-"),
                meta.get("cost_usd", "-"),
                meta.get("latency_s") or 0.0,
            )
            results[job.rel] = result
    return results


def build_issue_toc(
    pages: list[Path], toc_pages: dict[Path, str], results: dict[Path, PageResult | None]
) -> dict[str, toc_module.IssueToc]:
    """Слить результаты этапа toc в оглавления выпуска по видам, в порядке полос."""
    tocs: dict[str, toc_module.IssueToc] = {}
    for kind in toc_module.KINDS:
        ordered = [
            (rel.as_posix(), results[rel].toc)
            for rel in pages
            if toc_pages.get(rel) == kind and results.get(rel) is not None and results[rel].toc is not None
        ]
        if ordered:
            tocs[kind] = toc_module.merge_pages(kind, ordered)
    return tocs


def _write_issue_toc(out_dir: Path, issue_key: str, tocs: dict[str, toc_module.IssueToc]) -> None:
    issue_dir = out_dir / issue_key
    issue_dir.mkdir(parents=True, exist_ok=True)
    (issue_dir / TOC_JSON).write_text(
        json.dumps(toc_module.to_dict(tocs), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (issue_dir / TOC_MD).write_text(toc_module.to_markdown(tocs), encoding="utf-8")


def decide_redo(mode: str, issue_key: str, missed: list[tuple[Path, str]]) -> bool:
    """Перераспознавать ли выпуск после находки оглавления вне базы.

    ``ask`` спрашивает в терминале; без терминала (фоновый прогон с ``< /dev/null``) ведёт себя как
    ``skip`` — молча пересчитывать чужой счёт нельзя.
    """
    lines = "\n".join(f"  {rel.as_posix()}  # {kind}" for rel, kind in missed)
    logger.warning(
        "ВНИМАНИЕ: в выпуске %s модель считает оглавлением полосы, не помеченные в базе:\n%s\n"
        "Проставьте теги в CVAT («Оглавление»/«Годовой указатель» или вето «Не оглавление») и заберите базу from-cvat.",
        issue_key,
        lines,
    )
    if mode == "redo":
        return True
    if mode == "skip":
        return False
    if not sys.stdin.isatty():
        logger.warning("%s: терминала нет, перераспознание пропущено (см. %s)", issue_key, MISSED_LIST)
        return False
    return click.confirm(f"Перераспознать выпуск {issue_key} с этими полосами как оглавлением?", default=False)


def _append_missed(out_dir: Path, issue_key: str, missed: list[tuple[Path, str]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / MISSED_LIST).open("a", encoding="utf-8") as handle:
        for rel, kind in missed:
            handle.write(f"{rel.as_posix()}  # {kind}, выпуск {issue_key}\n")


def run_issue(
    client: OpenRouterClient,
    spec: ModelSpec,
    params: PipelineParams,
    issue_key: str,
    pages: list[Path],
    stats: PipelineStats,
    extra_toc: dict[Path, str] | None = None,
    allow_redo: bool = True,
) -> None:
    """Один выпуск целиком: этап toc, слияние, этап page, fallback."""
    year = issue_key.split("/")[0] if issue_key != "." else ""
    toc_pages: dict[Path, str] = {}
    for rel in pages:
        flags = flags_for(rel, params.flags)
        if not flags.known and params.flags is not None:
            stats.unknown_pages += 1
        kind = flags.toc_kind
        if kind is None and extra_toc and rel in extra_toc:
            kind = extra_toc[rel]
        if kind is not None:
            toc_pages[rel] = kind
    reuse = params.skip_done or extra_toc is not None
    logger.info("Выпуск %s: полос %d, из них оглавление/указатель %d", issue_key, len(pages), len(toc_pages))

    toc_jobs = [PageJob(rel, "toc", kind, year) for rel, kind in toc_pages.items()]
    toc_results = _recognise_many(client, spec, params, toc_jobs, reuse, stats)
    tocs = build_issue_toc(pages, toc_pages, toc_results)
    if toc_pages:
        _write_issue_toc(params.out_dir, issue_key, tocs)
        for kind, issue_toc in tocs.items():
            logger.info(
                "%s: %s — рубрик %d, статей %d, полос %d (продолжений %d)",
                issue_key,
                kind,
                len(issue_toc.rubrics),
                len(issue_toc.articles),
                len(issue_toc.pages),
                issue_toc.continuations,
            )
    rubrics, articles = toc_module.prompt_lists(tocs.get(toc_module.KIND_CONTENTS))
    digest = toc_module.toc_hash(rubrics, articles)

    regular = [rel for rel in pages if rel not in toc_pages]
    page_jobs = [PageJob(rel, "page", "none", year, tuple(rubrics), tuple(articles), digest) for rel in regular]
    page_results = _recognise_many(client, spec, params, page_jobs, reuse, stats)

    missed = [
        (rel, result.toc_kind)
        for rel in regular
        if (result := page_results.get(rel)) is not None
        and result.toc_kind != "none"
        and not flags_for(rel, params.flags).force_is_not_toc
    ]
    if not missed:
        return
    stats.missed.append(f"{issue_key}: " + ", ".join(rel.name for rel, _ in missed))
    if allow_redo and decide_redo(params.on_missed_toc, issue_key, missed):
        stats.redone_issues.append(issue_key)
        run_issue(client, spec, params, issue_key, pages, stats, extra_toc=dict(missed), allow_redo=False)
    else:
        _append_missed(params.out_dir, issue_key, missed)


def collect_meta(out_dir: Path) -> list[dict]:
    """Все .meta.json под out_dir: сводка строится по ним, а не по одному прогону."""
    rows: list[dict] = []
    for path in sorted(out_dir.rglob("*.meta.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            logger.warning("битый %s", path)
    return rows


def write_summary(out_dir: Path, rows: list[dict]) -> Path:
    path = out_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item.get("page", "")):
            writer.writerow(row)
    return path


def run_pipeline(client: OpenRouterClient, spec: ModelSpec, params: PipelineParams) -> PipelineStats:
    """Весь вход по выпускам; возвращает сводку. summary.csv пересобирается по всем meta под out-dir."""
    stats = PipelineStats()
    rels = list_pages(params.in_dir, params.pages_file, params.only_year, params.only_issue, params.limit)
    groups = group_by_issue(rels)
    stats.pages = len(rels)
    started = time.monotonic()
    logger.info(
        "%s (%s): выпусков %d, полос %d, потоков %d, тайл %d px → %d px",
        spec.name,
        spec.openrouter_id,
        len(groups),
        len(rels),
        params.jobs,
        params.options.max_src_tile,
        params.options.max_model_tile,
    )
    for issue_key, pages in groups.items():
        stats.issues += 1
        run_issue(client, spec, params, issue_key, pages, stats)
    stats.toc_pages = sum(1 for rel in rels if flags_for(rel, params.flags).toc_kind is not None)
    summary = write_summary(params.out_dir, collect_meta(params.out_dir))
    logger.info(
        "готово за %.0f с: выпусков %d, полос %d, запросов %d (готовых пропущено %d), сбоев %d, "
        "стоимость $%.4f, сводка %s",
        time.monotonic() - started,
        stats.issues,
        stats.pages,
        stats.requests,
        stats.reused,
        stats.failed,
        stats.cost_usd,
        summary,
    )
    if stats.unknown_pages:
        logger.warning("полос без записи в базе: %d — считались обычными", stats.unknown_pages)
    if params.options.second_pass:
        logger.info(
            "второй проход: %d полос из %d, оставлен первый у %d",
            stats.second_passes,
            stats.pages,
            stats.second_pass_kept_first,
        )
    if stats.missed:
        logger.warning("оглавления вне базы (%d выпусков): %s", len(stats.missed), "; ".join(stats.missed))
    return stats
