"""Прогон детекции по валидационной выборке и сравнение с ожиданиями.

Гоняется ТОТ ЖЕ ``detection.page.detect_page``, что и боевой прогон. Второй реализации
здесь нет намеренно: оснастка, которая проверяет не то, что работает, хуже, чем никакой.

Стоимость — около сотни файлов, четыре гигабайта, меньше минуты в пуле. Это и есть рабочий
цикл настройки порогов: полный прогон по паку идёт часы, и крутить пороги по нему нельзя.
"""

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from tqdm import tqdm

from ocr_utils.scan_markup.db.models import Issue, Page, YearPackage
from ocr_utils.scan_markup.db.repo import require_pack
from ocr_utils.scan_markup.detection.overlay import write_debug_overlay
from ocr_utils.scan_markup.detection.page import (
    PageAnalysis,
    PageOptions,
    PageResult,
    analyse_page,
    finish_page,
    surya_boxes_for,
)
from ocr_utils.scan_markup.validation.cases import Case, collect_cases
from ocr_utils.scan_markup.validation.checks import describe, expectation_holds

logger = logging.getLogger(__name__)


@dataclass
class ValidateParams:
    """Параметры прогона ``validate``."""

    pack_dir: Path
    cases_root: Path
    options: PageOptions
    out_dir: Path | None = None
    jobs: int = 8
    db_path: Path | None = None
    pack_name: str | None = None
    control_limit: int = 0
    use_surya_layout: bool = True


@dataclass
class CaseOutcome:
    """Итог по одной полосе выборки."""

    case: Case
    result: PageResult
    fixed: bool
    note: str


@dataclass
class ControlOutcome:
    """Итог по контрольной полосе: были области в базе — остались ли."""

    rel_path: str
    was: int
    now: int


@dataclass
class ValidateReport:
    """Всё, что печатает и пишет отчёт."""

    outcomes: list[CaseOutcome] = field(default_factory=list)
    control: list[ControlOutcome] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def by_defect(self) -> dict[str, list[CaseOutcome]]:
        grouped: dict[str, list[CaseOutcome]] = {}
        for outcome in self.outcomes:
            grouped.setdefault(outcome.case.defect.key, []).append(outcome)
        return grouped


@dataclass(frozen=True)
class _Job:
    path: Path
    rel_path: str
    order_index: int
    options: PageOptions


def _worker(job: _Job) -> PageAnalysis:
    """Обёртка для ``ProcessPoolExecutor.map`` (лямбду не запикль)."""
    return analyse_page(job.path, job.rel_path, job.order_index, job.options)


def _run_jobs(jobs: list[_Job], workers: int, desc: str, detector=None) -> dict[str, PageResult]:
    """Считает список заданий и возвращает результаты по ``rel_path``.

    Порядок ровно тот же, что в боевом прогоне: пиксели в пуле, Surya и сборка — здесь.
    Оснастка обязана проверять то, что работает, а не вторую реализацию того же.
    """
    if not jobs:
        return {}
    options = jobs[0].options

    def collect(analyses):
        out = {}
        for analysis in tqdm(analyses, total=len(jobs), desc=desc, unit="полоса"):
            out[analysis.rel_path] = finish_page(analysis, options, surya_boxes_for(analysis, detector))
        return out

    if workers <= 1:
        return collect(_worker(job) for job in jobs)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return collect(pool.map(_worker, jobs, chunksize=1))


def _order_index_by_rel(params: ValidateParams, rel_paths: set[str]) -> dict[str, int]:
    """Порядковые номера полос из базы: без них не проверить обложку.

    Базы может не быть — тогда порядок неизвестен, и полоса считается внутренней. Для
    выборки это верно: обложек в ней нет ни одной, кроме 1969/09 IMG_0103_2R.
    """
    if params.db_path is None or params.pack_name is None:
        return {}
    from ocr_utils.scan_markup.db.session import open_db

    # create=True не заводит базу заново, а дописывает недостающие колонки: база с прошлого
    # прогона схемы не знает про detector_version, и без миграции любой SELECT по Page падает.
    with open_db(params.db_path)() as session:
        pack = require_pack(session, params.pack_name)
        rows = session.execute(
            select(Page.rel_path, Page.order_index)
            .join(Issue, Issue.id == Page.issue_id)
            .join(YearPackage, YearPackage.id == Issue.year_package_id)
            .where(YearPackage.pack_id == pack.id)
        ).all()
    return {rel: index for rel, index in rows if rel in rel_paths}


def _control_jobs(params: ValidateParams, skip: set[str]) -> tuple[list[_Job], dict[str, int]]:
    """Случайные полосы, у которых в базе УЖЕ есть области, — страховка от потерь.

    Полосы выборки исключаются: они и так проверяются по ожиданиям, а их прежняя разметка
    как раз и была неправильной.
    """
    import random

    if params.db_path is None or params.pack_name is None or params.control_limit <= 0:
        return [], {}
    from ocr_utils.scan_markup.db.session import open_db

    with open_db(params.db_path)() as session:
        pack = require_pack(session, params.pack_name)
        pages = session.scalars(
            select(Page)
            .join(Issue, Issue.id == Page.issue_id)
            .join(YearPackage, YearPackage.id == Issue.year_package_id)
            .where(YearPackage.pack_id == pack.id)
            .where(Page.raster_regions.any())
            .order_by(Page.id)
        ).all()
        rows = [(page.rel_path, page.order_index, len(page.raster_regions)) for page in pages]

    rows = [row for row in rows if row[0] not in skip]
    random.Random(20260831).shuffle(rows)
    rows = rows[: params.control_limit]
    jobs = [_Job(params.pack_dir / rel, rel, order, params.options) for rel, order, _count in rows]
    return jobs, {rel: count for rel, _order, count in rows}


def run_validate(params: ValidateParams) -> ValidateReport:
    """Прогоняет выборку и, если задана база, контрольные полосы."""
    cases, notes = collect_cases(params.cases_root, params.pack_dir)
    report = ValidateReport(notes=notes)
    if not cases:
        return report

    order_by_rel = _order_index_by_rel(params, {case.rel_path for case in cases})
    jobs = [_Job(case.path, case.rel_path, order_by_rel.get(case.rel_path, 1), params.options) for case in cases]
    detector = None
    if params.use_surya_layout:
        from ocr_utils.background_smoothing.layout import LayoutDetector

        detector = LayoutDetector()

    results = _run_jobs(jobs, params.jobs, "эталоны", detector)

    for case in cases:
        result = results[case.rel_path]
        if result.error:
            report.outcomes.append(CaseOutcome(case, result, False, f"ОШИБКА: {result.error}"))
            continue
        fixed = expectation_holds(case.defect.key, result.regions)
        report.outcomes.append(CaseOutcome(case, result, fixed, describe(case.defect.key, result.regions)))
        if params.out_dir is not None:
            folder = params.out_dir / case.defect.key / ("починено" if fixed else "осталось")
            folder.mkdir(parents=True, exist_ok=True)
            write_debug_overlay(folder, case.rel_path, case.path, result.regions)

    control_jobs, was_counts = _control_jobs(params, {case.rel_path for case in cases})
    control_results = _run_jobs(control_jobs, params.jobs, "контроль", detector)
    for rel, was in was_counts.items():
        result = control_results.get(rel)
        now = 0 if result is None or result.error else len(result.regions)
        report.control.append(ControlOutcome(rel, was, now))
        if params.out_dir is not None and now == 0 and result is not None:
            folder = params.out_dir / "контроль" / "область-пропала"
            folder.mkdir(parents=True, exist_ok=True)
            write_debug_overlay(folder, rel, params.pack_dir / rel, result.regions)

    return report
