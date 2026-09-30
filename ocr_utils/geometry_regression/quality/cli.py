"""Боевой прогон детектора порчи геометрии v18 по паку: мера v16 при промахе её кэша, меры v17–v18, verdicts.csv."""

from __future__ import annotations

import csv
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import click

from ocr_utils.geometry_regression.metrics import Params
from ocr_utils.geometry_regression.quality.measure import load_cached, measure_cached
from ocr_utils.geometry_regression.quality.scoring import DEFAULT_TOTAL, Thresholds17
from ocr_utils.geometry_regression.quality.sources import PageRef, list_pages
from ocr_utils.geometry_regression.quality.verdict import ensure_v16
from ocr_utils.geometry_regression.scoring import DEFAULT_MIN_GAIN, DEFAULT_RATIO
from ocr_utils.page_layout.pack_analysis.stages import init_worker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunJob:
    """Задача воркера: страницы одного PDF (оба PDF открываются один раз на задачу) и пути прогона."""

    pdf: str
    pages: tuple[int, ...]  # номера страниц с единицы
    geo_dir: Path
    nogeo_dir: Path
    layout_root: Path
    v16_dir: Path
    run_dir: Path
    layout_cache_dir: Path | None
    redo: bool


def measure_job(job: RunJob) -> list[tuple[int, str]]:
    """Измерить страницы одного PDF в воркере: кэш v16 (при промахе — мера v16), затем меры v18 с записью кэша.

    Args:
        job: Задача (:class:`RunJob`).

    Returns:
        Пары ``(страница, текст ошибки)``; пустой текст — страница измерена или взята из кэша.
    """
    import fitz

    params = Params(layout_cache_dir=job.layout_cache_dir)
    out: list[tuple[int, str]] = []
    with (
        fitz.open(str(job.geo_dir / f"{job.pdf}.pdf")) as geo,
        fitz.open(str(job.nogeo_dir / f"{job.pdf}.pdf")) as nogeo,
    ):
        for page in job.pages:
            ref = PageRef(job.pdf, page)
            # Готовая страница текущей версии не пересчитывается (кроме --redo).
            if not job.redo and load_cached(job.run_dir, ref) is not None:
                out.append((page, ""))
                continue
            try:
                ensure_v16(job.v16_dir, job.pdf, page, geo, nogeo, params)
            except Exception as error:  # noqa: BLE001 — одна битая страница не валит прогон
                out.append((page, f"v16: {type(error).__name__}: {error}"))
                continue
            payload = measure_cached(
                ref, job.layout_root, job.v16_dir, job.run_dir, job.redo, (job.geo_dir, job.nogeo_dir)
            )
            out.append((page, payload.get("error", "")))
    return out


def write_verdicts(run_dir: Path, refs: list[PageRef], thresholds: Thresholds17) -> Counter:
    """``<run_dir>/verdicts.csv``: вердикт, правило, виновник, сумма групп и выигрыш по каждой измеренной странице.

    Args:
        run_dir: Каталог прогона v18.
        refs: Страницы прогона.
        thresholds: Пороги вердикта.

    Returns:
        Счётчик вердиктов (``bad``/``mixed``/``ok``).
    """
    counts: Counter = Counter()
    with (run_dir / "verdicts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pdf", "page", "verdict", "rule", "culprit", "total", "damage", "gain", "gain_reason"])
        for ref in refs:
            payload = load_cached(run_dir, ref)
            if payload is None:
                continue
            result = thresholds.assess(payload["metrics"])
            counts[result.verdict.value] += 1
            writer.writerow([ref.pdf, ref.page, result.verdict.value, result.rule.value, result.culprit,
                             f"{result.total:.3f}", f"{result.damage:.3f}", f"{result.gain:.3f}",
                             result.gain_reason])  # fmt: skip
    return counts


@click.group()
@click.option("--log-level", default="INFO", show_default=True)
def main(log_level: str) -> None:
    """Детектор порчи геометрии v18 (боевой): прогон по паку."""
    logging.basicConfig(level=log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@main.command()
@click.option("--geo-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="PDF FineReader с коррекцией геометрии (A).")  # fmt: skip
@click.option("--nogeo-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="PDF FineReader без коррекции (B); имена файлов — те же.")  # fmt: skip
@click.option("--layout-root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Разбор page_layout обоих вариантов (geo/pages, nogeo/pages): страницы прогона — те, где он есть.")  # fmt: skip
@click.option("--v16-dir", required=True, type=click.Path(file_okay=False, path_type=Path),
              help="Прогон движка v16 (cache/): страницы без его JSON меряются v16 здесь же.")  # fmt: skip
@click.option(
    "--run-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Прогон v18 (cache/, verdicts.csv).",
)
@click.option("--layout-cache", "layout_cache_dir", default=None, type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Кэш surya page_layout (рамки таблиц, рисунков и растра для v16); набивается заранее.")  # fmt: skip
@click.option("--pdfs", default=None, help="Только эти PDF через запятую (full_1966_01,...).")
@click.option("--jobs", default=16, show_default=True, type=int, help="Воркеров пула (CPU; по PDF на задачу).")
@click.option("--redo", is_flag=True, help="Пересчитать меры v18 и у страниц с готовым кэшем (кэш v16 не трогается).")
@click.option("--thr", multiple=True, help="Порог метрики имя=число (для verdicts.csv).")
@click.option("--hard", multiple=True, help="Жёсткий порог метрики имя=число.")
@click.option("--total", default=DEFAULT_TOTAL, show_default=True, type=float, help="Порог суммы групп S.")
@click.option("--min-gain", default=DEFAULT_MIN_GAIN, show_default=True, type=float)
@click.option("--ratio", default=DEFAULT_RATIO, show_default=True, type=float)
def run(geo_dir, nogeo_dir, layout_root, v16_dir, run_dir, layout_cache_dir, pdfs, jobs, redo, thr, hard, total,
        min_gain, ratio) -> None:  # fmt: skip
    """Мера всех страниц с разбором обоих вариантов (v16 при промахе, v17–v18) и verdicts.csv по порогам."""
    refs = list_pages(layout_root, set(pdfs.split(",")) if pdfs else None)
    by_pdf: dict[str, list[int]] = {}
    for ref in refs:
        by_pdf.setdefault(ref.pdf, []).append(ref.page)
    tasks = [RunJob(pdf, tuple(pages), geo_dir, nogeo_dir, layout_root, v16_dir, run_dir, layout_cache_dir, redo)
             for pdf, pages in sorted(by_pdf.items())]  # fmt: skip
    logger.info("Страниц с разбором обоих вариантов: %d, PDF: %d", len(refs), len(tasks))
    errors = 0
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for done, (job, results) in enumerate(zip(tasks, pool.map(measure_job, tasks)), 1):
            for page, error in results:
                if error:
                    errors += 1
                    logger.error("%s с.%d: %s", job.pdf, page, error)
            logger.info("PDF %d/%d: %s", done, len(tasks), job.pdf)
    thresholds = Thresholds17(Thresholds17.parse_pairs(thr), Thresholds17.parse_pairs(hard), total, min_gain, ratio)
    counts = write_verdicts(run_dir, refs, thresholds)
    click.echo(f"Готово: страниц {len(refs)}, ошибок {errors}; вердикты: {dict(counts)}; {run_dir / 'verdicts.csv'}")


__all__ = ["RunJob", "main", "measure_job", "write_verdicts"]
