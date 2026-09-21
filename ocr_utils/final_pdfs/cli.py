"""Команда сборки финальных PDF: ``uv run python -m ocr_utils.final_pdfs run …``.

Три стадии в одном запуске (см. докстринг пакета): A — анализ страниц в пуле процессов,
B — второе мнение surya в родителе (GPU), C — сборка выпусков в пуле. Всё идемпотентно:
JSON анализа и готовые PDF при ``--skip-done`` не пересчитываются.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click

from ocr_utils.final_pdfs.analysis import (
    AnalysisParams,
    PageAnalysis,
    analyse_chunk,
    chunk_jobs,
    second_opinion_requests,
)
from ocr_utils.final_pdfs.assemble import (
    AssembleParams,
    IssueResult,
    merge_csv,
    assemble_issue,
    write_pages_csv,
    write_summary_csv,
)
from ocr_utils.final_pdfs.pictures import DEFAULT_DESCREEN_SIGMA_MM, DEFAULT_JPEG_QUALITY, DEFAULT_PICTURE_DPI
from ocr_utils.final_pdfs.plan import IssuePlan, load_plans
from ocr_utils.final_pdfs.sources import IssuePair, margins_px, pair_issue_pdfs
from ocr_utils.geometry_regression.scoring import DEFAULT_HARD, DEFAULT_MIN_GAIN, DEFAULT_RATIO
from ocr_utils.text_layer_fix.rewrite import DEFAULT_FONT_PATH

logger = logging.getLogger("ocr_utils.final_pdfs")

# Поля промежуточной PDF пака-1: 12 и 6 мм из intermediate_pdfs, округлённые вверх до MCU JPEG
# (16 px при 600 dpi) — фактически 288 и 144 px. Сверяются на каждой странице с иллюстрациями.
DEFAULT_MARGIN_X_MM = 12.192
DEFAULT_MARGIN_Y_MM = 6.096

# Анализ — tesseract по ячейкам и зонам, чистый CPU: 16 воркеров минус резерв под родителя.
# Сборка — запись PDF и разжатие 20-мегапиксельных JPEG: упор в диск, воркеров меньше.
DEFAULT_JOBS = 16
DEFAULT_RESERVE_CPU_CORES = 4
DEFAULT_ASSEMBLE_JOBS = 8


def _init_worker() -> None:
    """Инициализатор воркера: без hugepage и потоков BLAS (``.claude/rules/gpu_and_pools.md``)."""
    import cv2
    import numpy as np
    import threadpoolctl

    np._core.multiarray._set_madvise_hugepage(False)
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)


def physical_cpu_count() -> int:
    """Число физических ядер машины (без SMT-потоков); без psutil — логических."""
    try:
        import psutil

        return psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except ImportError:
        return os.cpu_count() or 1


def effective_jobs(jobs: int, reserve_cpu_cores: int) -> int:
    """``min(jobs, физических ядер − резерв)``, но не меньше одного."""
    if reserve_cpu_cores <= 0:
        return max(1, jobs)
    return max(1, min(jobs, physical_cpu_count() - reserve_cpu_cores))


def _pool(jobs: int) -> "ProcessPoolExecutor | None":
    if jobs <= 1:
        return None
    return ProcessPoolExecutor(
        max_workers=jobs, mp_context=multiprocessing.get_context("forkserver"), initializer=_init_worker
    )


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


def pair_all(geo_dir: Path, nogeo_dir: Path, plans: list[IssuePlan]) -> tuple[dict[int, IssuePair], list[IssueResult]]:
    """Пары PDF по выпускам; выпуски без пары или с расхождением страниц — сразу в ошибки."""
    pairs: dict[int, IssuePair] = {}
    failures: list[IssueResult] = []
    for plan in plans:
        try:
            pairs[plan.issue_id] = pair_issue_pdfs(geo_dir, nogeo_dir, plan)
        except (FileNotFoundError, ValueError) as error:
            logger.error("%s: %s", plan.full_pdf_name, error)
            failures.append(IssueResult(plan.full_pdf_name, status="error", reason=str(error)))
    return pairs, failures


def _prefill_surya(cache_root: Path, pairs, jobs: int) -> None:
    """Стадия 0: surya по всем страницам пар PDF (оба варианта) — в родителе, до пула стадии A."""
    from ocr_utils.page_layout.image import Variant
    from ocr_utils.page_layout.prefill import pdf_requests, prefill
    from ocr_utils.page_layout.surya.cache import SuryaCache
    from ocr_utils.page_layout.surya.model import SuryaLayoutModel

    model = SuryaLayoutModel()
    cache = SuryaCache(cache_root)
    items = list(pairs.values()) if isinstance(pairs, dict) else list(pairs)
    requests = list(pdf_requests([p.nogeo for p in items], Variant.FR_NOGEO)) + list(
        pdf_requests([p.geo for p in items], Variant.FR_GEO)
    )
    stats = prefill(requests, cache, model, jobs=jobs)
    model.close()  # видеопамять — стадии B (surya OCR)
    click.echo(f"  страниц {stats.requested}: уже в кэше {stats.cached}, размечено {stats.done}, ошибок {stats.failed}")


def run_analysis(
    plans: list[IssuePlan], pairs: dict[int, IssuePair], params: AnalysisParams, jobs: int
) -> list[PageAnalysis]:
    """Стадия A по всем выпускам."""
    tasks = [job for plan in plans if plan.issue_id in pairs for job in chunk_jobs(plan, pairs[plan.issue_id], params)]
    rows: list[PageAnalysis] = []
    pool = _pool(jobs)
    mapper = pool.map if pool else map
    done = 0
    try:
        for chunk_rows in mapper(analyse_chunk, tasks):
            rows.extend(chunk_rows)
            done += 1
            if done % 50 == 0 or done == len(tasks):
                logger.info("анализ: пачек %d из %d, страниц %d", done, len(tasks), len(rows))
    finally:
        if pool:
            pool.shutdown()
    return rows


def run_second_opinion(plans: list[IssuePlan], pairs: dict[int, IssuePair], params: AnalysisParams, batch: int) -> str:
    """Стадия B: surya по всем ненадёжным зонам (в этом процессе)."""
    from ocr_utils.rotated_text.tables import second_opinion as surya
    from ocr_utils.text_layer_fix.second_opinion import revise

    if not surya.surya_available():
        return "surya недоступна — второе мнение пропущено"
    requests = second_opinion_requests(params, plans, pairs)
    stats = revise(requests, batch)
    return (
        f"surya: страниц {stats.pages}, зон спрошено {stats.asked}, ответов принято {stats.accepted}, "
        f"стали пригодными {stats.newly_accepted}"
    )


def run_assembly(
    plans: list[IssuePlan], pairs: dict[int, IssuePair], params: AssembleParams, jobs: int
) -> list[IssueResult]:
    """Стадия C по всем выпускам."""
    tasks = [(plan, pairs[plan.issue_id], params) for plan in plans if plan.issue_id in pairs]
    results: list[IssueResult] = []
    pool = _pool(jobs)
    mapper = pool.map if pool else map
    try:
        for result in mapper(_assemble_one, tasks):
            results.append(result)
            logger.info(
                "%s: %s%s — страниц %d (geo %d / nogeo %d), иллюстраций %d, снято образов %d, слой −%d/+%d, "
                "сверка не прошла %d, %.0f с",
                result.pdf,
                result.status,
                f" ({result.reason})" if result.reason else "",
                result.pages,
                result.pages_geo,
                result.pages_nogeo,
                result.pictures,
                result.figures_removed,
                result.blanked,
                result.inserted,
                result.verify_failures,
                result.seconds,
            )
    finally:
        if pool:
            pool.shutdown()
    return results


def _assemble_one(task: tuple[IssuePlan, IssuePair, AssembleParams]) -> IssueResult:
    return assemble_issue(*task)


def write_analysis_csv(path: Path, rows: list[PageAnalysis]) -> None:
    """``analysis.csv``: строка на страницу после стадии A; страницы прошлых запусков сохраняются."""
    merge_csv(path, list(PageAnalysis.__dataclass_fields__), [row.to_row() for row in rows], ("pdf", "page"))


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--log-level", default="INFO", show_default=True)
def main(log_level: str) -> None:
    """Сборка финальных PDF из двух прогонов FineReader с правкой слоя и возвратом иллюстраций."""
    _setup_logging(log_level)


@main.command("run")
@click.option(
    "--geo-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="PDF с коррекцией геометрии",
)
@click.option(
    "--nogeo-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="PDF без коррекции",
)
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path), help="финальные PDF")
@click.option(
    "--pictures-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="очищенные полосы (blurred) {год}/{выпуск}/полоса.tif — источник иллюстраций",
)
@click.option(
    "--db",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="база разметки после экспорта из CVAT",
)
@click.option("--pack-name", required=True)
@click.option(
    "--work-dir", required=True, type=click.Path(file_okay=False, path_type=Path), help="JSON анализа, CSV, превью"
)
@click.option("--jpeg-quality", default=DEFAULT_JPEG_QUALITY, show_default=True, type=int)
@click.option("--picture-dpi", default=DEFAULT_PICTURE_DPI, show_default=True, type=int)
@click.option(
    "--descreen-sigma-mm",
    default=DEFAULT_DESCREEN_SIGMA_MM,
    show_default=True,
    type=float,
    help="Gaussian по исходному растру перед уменьшением; 0 — выключить",
)
@click.option(
    "--margin-x-mm",
    default=DEFAULT_MARGIN_X_MM,
    show_default=True,
    type=float,
    help="поле промежуточной PDF слева и справа",
)
@click.option("--margin-y-mm", default=DEFAULT_MARGIN_Y_MM, show_default=True, type=float, help="поле сверху и снизу")
@click.option("--only-year", default=None)
@click.option("--only-issue", default=None)
@click.option(
    "--geometry-run-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="прогон детектора геометрии с cache/; без него — всегда мерить",
)
@click.option(
    "--layout-cache",
    "layout_cache_dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="корень кэша surya page_layout: стадия 0 набивает его по всем страницам обеих PDF выпуска (GPU в "
    "родителе), дальше детектор геометрии и правка слоя читают его в воркерах. Без него — без surya.",
)
@click.option("--geometry-thr", multiple=True, help="порог детектора «имя=значение», можно несколько")
@click.option("--geometry-hard", default=DEFAULT_HARD, show_default=True, type=float)
@click.option("--geometry-min-gain", default=DEFAULT_MIN_GAIN, show_default=True, type=float)
@click.option("--geometry-ratio", default=DEFAULT_RATIO, show_default=True, type=float)
@click.option("--text-layer/--no-text-layer", default=True, show_default=True, help="править текстовый слой")
@click.option(
    "--second-opinion/--no-second-opinion", default=True, show_default=True, help="surya по ненадёжным зонам (GPU)"
)
@click.option("--surya-batch", default=16, show_default=True, type=int)
@click.option("--lang", default="rus", show_default=True, help="языки tesseract")
@click.option("--jobs", default=DEFAULT_JOBS, show_default=True, type=int, help="воркеров анализа")
@click.option(
    "--reserve-cpu-cores",
    default=DEFAULT_RESERVE_CPU_CORES,
    show_default=True,
    type=int,
    help="физических ядер оставить родителю",
)
@click.option(
    "--assemble-jobs", default=DEFAULT_ASSEMBLE_JOBS, show_default=True, type=int, help="воркеров сборки (упор в диск)"
)
@click.option("--skip-done/--redo", default=True, show_default=True, help="не пересчитывать готовые JSON и PDF")
@click.option(
    "--insert-font",
    default=DEFAULT_FONT_PATH,
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="TTF для вставок в слой",
)
@click.option(
    "--preview-pages",
    default=0,
    show_default=True,
    type=int,
    help="сколько страниц с иллюстрациями на выпуск отрендерить в work-dir/preview",
)
@click.option(
    "--assemble/--no-assemble", default=True, show_default=True, help="стадия C; --no-assemble — только анализ и surya"
)
@click.option("--reassemble", is_flag=True, help="пересобрать готовые PDF, не пересчитывая анализ (JSON из кэша)")
def run(
    geo_dir,
    nogeo_dir,
    out_dir,
    pictures_dir,
    db,
    pack_name,
    work_dir,
    jpeg_quality,
    picture_dpi,
    descreen_sigma_mm,
    margin_x_mm,
    margin_y_mm,
    only_year,
    only_issue,
    geometry_run_dir,
    layout_cache_dir,
    geometry_thr,
    geometry_hard,
    geometry_min_gain,
    geometry_ratio,
    text_layer,
    second_opinion,
    surya_batch,
    lang,
    jobs,
    reserve_cpu_cores,
    assemble_jobs,
    skip_done,
    insert_font,
    preview_pages,
    assemble,
    reassemble,
) -> None:
    """Собрать финальные PDF выпусков (анализ → surya → сборка)."""
    plans = load_plans(db, pack_name, only_year=only_year, only_issue=only_issue)
    if not plans:
        raise click.ClickException("в базе нет выпусков под фильтр")
    pairs, failures = pair_all(geo_dir, nogeo_dir, plans)
    click.echo(f"Выпусков: {len(plans)}, пар PDF: {len(pairs)}, без пары/с расхождением: {len(failures)}")
    analysis = AnalysisParams(
        work_dir=work_dir,
        geometry_run_dir=geometry_run_dir,
        geometry_thr=tuple(geometry_thr),
        geometry_hard=geometry_hard,
        geometry_min_gain=geometry_min_gain,
        geometry_ratio=geometry_ratio,
        text_layer=text_layer,
        lang=lang,
        skip_done=skip_done,
        layout_cache_dir=layout_cache_dir,
    )
    workers = effective_jobs(jobs, reserve_cpu_cores)
    if layout_cache_dir is not None and not reassemble:
        click.echo("Стадия 0: кэш surya по страницам обеих PDF")
        _prefill_surya(layout_cache_dir, pairs, workers)
    click.echo(f"Стадия A: анализ страниц, воркеров {workers}")
    rows = run_analysis(plans, pairs, analysis, workers)
    write_analysis_csv(work_dir / "analysis.csv", rows)
    errors = [r for r in rows if r.error]
    by_source = {"geo": sum(1 for r in rows if r.source == "geo"), "nogeo": sum(1 for r in rows if r.source == "nogeo")}
    by_reason: dict[str, int] = {}
    for r in rows:
        by_reason[r.reason] = by_reason.get(r.reason, 0) + 1
    click.echo(
        f"  страниц {len(rows)}, из кэша {sum(1 for r in rows if r.cached)}, источники {by_source}, причины {by_reason}, "
        f"ошибок {len(errors)}"
    )
    for row in errors[:10]:
        click.echo(f"  {row.pdf} с.{row.page + 1}: {row.error}")
    if second_opinion and text_layer:
        click.echo("Стадия B: второе мнение surya")
        click.echo("  " + run_second_opinion(plans, pairs, analysis, surya_batch))
    if not assemble:
        return
    params = AssembleParams(
        analysis=analysis,
        out_dir=out_dir,
        pictures_dir=pictures_dir,
        margins=margins_px(margin_x_mm, margin_y_mm, plans[0].pages[0].dpi),
        picture_dpi=picture_dpi,
        jpeg_quality=jpeg_quality,
        descreen_sigma_mm=descreen_sigma_mm,
        font_path=insert_font,
        skip_done=skip_done and not reassemble,
        preview_pages=preview_pages,
    )
    click.echo(f"Стадия C: сборка выпусков, воркеров {assemble_jobs}")
    results = run_assembly(plans, pairs, params, assemble_jobs) + failures
    write_pages_csv(work_dir / "pages.csv", results)
    write_summary_csv(work_dir / "summary.csv", results)
    built = [r for r in results if r.status == "ok"]
    click.echo(
        f"Готово: собрано {len(built)}, пропущено {sum(1 for r in results if r.status == 'skipped')}, "
        f"ошибок {sum(1 for r in results if r.status == 'error')}; страниц geo {sum(r.pages_geo for r in built)} / "
        f"nogeo {sum(r.pages_nogeo for r in built)}, иллюстраций {sum(r.pictures for r in built)}, снято образов "
        f"{sum(r.figures_removed for r in built)}, слой −{sum(r.blanked for r in built)}/+{sum(r.inserted for r in built)}, "
        f"сверка не прошла на {sum(r.verify_failures for r in built)} стр.; "
        f"размер {sum(r.bytes_in_geo for r in built) / 2**20:.0f} → {sum(r.bytes_out for r in built) / 2**20:.0f} МиБ. "
        f"CSV: {work_dir / 'summary.csv'}, {work_dir / 'pages.csv'}"
    )
    for result in results:
        if result.status == "error":
            click.echo(f"  {result.pdf}: {result.reason}")
