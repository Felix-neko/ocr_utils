"""Прогон по паку: три фазы, между которыми — surya в родителе.

Фаза 1 (пул, CPU): вырезка, сетка, поворот ячеек, tesseract. Фаза 2 (родитель, GPU):
surya для сомнительных ячеек. Фаза 3 (пул, CPU): кегль, DPI, перекройка, набор, пары
«было-стало». Пул — forkserver, а не fork: в родителе к третьей фазе поднят torch, а форк
процесса с живой CUDA виснет.

ВЫХОД (``--out-dir``): ``after/<таблица>.png`` — результат в конечном dpi, ``pairs/`` —
пары «было-стало», ``info/<таблица>.json`` — всё, что известно про таблицу и ячейки,
``summary.csv`` — по строке на таблицу, ``sheets/`` — листы из пар, ``run.log``.
"""

from __future__ import annotations

import csv
import json
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import click
import cv2

from ocr_utils.scan_markup.rotation import parse_allowed

from ocr_utils.rotated_text.tables import pipeline, second_opinion, sheets
from ocr_utils.rotated_text.tables.source import TableRef, select_tables

logger = logging.getLogger("ocr_utils.rotated_text.tables")

SUMMARY_FIELDS = (
    "table_id",
    "year",
    "issue",
    "page",
    "cells",
    "rotated",
    "candidates",
    "replaced",
    "action",
    "sideways_table",
    "table_rotate_cw",
    "scale",
    "required_dpi",
    "min_font_px",
    "native_font_px",
    "widened_px",
    "surya_asked",
    "surya_used",
    "seconds",
    "error",
)


def _init_worker() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    cv2.setNumThreads(1)
    os.nice(10)


def _analyse_job(ref: TableRef, options: pipeline.Options) -> pipeline.TableAnalysis:
    return pipeline.analyse(ref, options)


def _rewrite_job(analysis: pipeline.TableAnalysis, options: pipeline.Options, out_dir: str, pairs_width: int) -> dict:
    """Третья фаза в воркере: перекройка, набор, файлы. Возвращает строку сводки."""
    out = Path(out_dir)
    ref = analysis.ref
    try:
        result, images = pipeline.rewrite(analysis, options)
    except Exception as error:  # noqa: BLE001 — одна таблица не должна валить прогон
        logger.exception("%s: перекройка упала", ref.table_id)
        result = pipeline.TableRewrite(table_id=ref.table_id, action=pipeline.ACTION_NONE, note=f"ошибка: {error}")
        images = None
    payload = result.to_json()
    payload["ref"] = asdict(ref)
    payload["analysis"] = {
        "work_dpi": analysis.work_dpi,
        "deskew_deg": round(analysis.deskew_deg, 3),
        "merged_cells": analysis.merged_cells,
        "sideways_share": analysis.sideways_share,
        "sideways_side": analysis.sideways_side,
        "error": analysis.error,
        "seconds": round(analysis.seconds, 2),
    }
    (out / "info").mkdir(parents=True, exist_ok=True)
    (out / "info" / f"{ref.table_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    if images is not None and result.action != pipeline.ACTION_NONE:
        (out / "after").mkdir(exist_ok=True)
        (out / "pairs").mkdir(exist_ok=True)
        cv2.imwrite(str(out / "after" / f"{ref.table_id}.png"), images.after)
        line_px = max(1, round(analysis.source_dpi / 300))
        before = sheets.overlay_before(images.before, images.grid_before, result.cells, line_px)
        caption = f"{ref.table_id}"
        right = f"стало: {_describe(result)}"
        sheets.pair_image(before, images.after, f"было: {caption}", right, pairs_width).save(
            out / "pairs" / f"{ref.table_id}.png"
        )

    candidates = [cell for cell in result.cells if cell.candidate]
    return {
        "table_id": ref.table_id,
        "year": ref.year,
        "issue": ref.issue,
        "page": ref.page_rel_path,
        "cells": len(result.cells),
        "rotated": sum(1 for cell in result.cells if cell.rotated),
        "candidates": len(candidates),
        "replaced": sum(1 for cell in result.cells if cell.replaced),
        "action": result.action,
        "sideways_table": int(result.sideways_table),
        "table_rotate_cw": result.table_rotate_cw,
        "scale": result.scale,
        "required_dpi": result.required_dpi if result.required_dpi is not None else "",
        "min_font_px": result.min_font_px,
        "native_font_px": result.native_font_px,
        "widened_px": sum(result.widened.values()),
        "surya_asked": sum(1 for cell in result.cells if cell.text_surya is not None),
        "surya_used": sum(1 for cell in result.cells if cell.engine == "surya"),
        "seconds": round(analysis.seconds + result.seconds, 1),
        "error": analysis.error or result.note,
    }


def _describe(result: pipeline.TableRewrite) -> str:
    parts = []
    if result.table_rotate_cw:
        parts.append(f"поворот таблицы {result.table_rotate_cw}")
    if result.widened:
        parts.append("колонки " + ", ".join(f"{col}: +{px}px" for col, px in result.widened.items()))
    if result.scale > 1.0 + 1e-6:
        parts.append(f"увеличение ×{result.scale:.2f} → {result.final_dpi} dpi")
    if not parts:
        parts.append(f"как есть, {result.final_dpi} dpi")
    parts.append(f"мин. кегль {result.min_font_px:.0f} px")
    if result.note:
        parts.append(result.note)
    return "; ".join(parts)


def _setup_logging(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(), logging.FileHandler(out_dir / "run.log", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers)


@click.group()
def main() -> None:
    """Таблицы с повёрнутым текстом: найти, прочитать, набрать прямо."""


@main.command()
@click.option("--db", "db_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--pack-name", required=True)
@click.option(
    "--images-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Корень заострённых копий; по умолчанию — из базы.",
)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--jobs", type=int, default=16, show_default=True)
@click.option("--only-year", default=None)
@click.option("--only-issue", default=None)
@click.option("--only-table", default=None, help="Подстрока идентификатора таблицы.")
@click.option("--limit", type=int, default=None)
@click.option("--angles", default="0,90,180,270", show_default=True, help="Допустимые повороты текста в ячейке.")
@click.option("--lang", default=pipeline.LANGUAGES, show_default=True)
@click.option("--work-dpi", type=int, default=300, show_default=True)
@click.option("--min-letters", type=int, default=3, show_default=True)
@click.option("--max-dpi", type=int, default=pipeline.MAX_DPI, show_default=True)
@click.option("--min-font-px", type=int, default=pipeline.MIN_FONT_EM_PX, show_default=True)
@click.option(
    "--surya/--no-surya",
    default=True,
    show_default=True,
    help="Второе мнение surya для повёрнутых ячеек, где tesseract ненадёжен.",
)
@click.option("--surya-batch", type=int, default=second_opinion.DEFAULT_BATCH, show_default=True)
@click.option("--no-deskew", is_flag=True, help="Не выравнивать вырезку по линейкам.")
@click.option("--pairs-width", type=int, default=1400, show_default=True, help="Ширина панели пары «было-стало».")
@click.option("--skip-done", is_flag=True, help="Пропускать таблицы, у которых уже есть info/*.json.")
def run(
    db_path: Path,
    pack_name: str,
    images_root: "Path | None",
    out_dir: Path,
    jobs: int,
    only_year: "str | None",
    only_issue: "str | None",
    only_table: "str | None",
    limit: "int | None",
    angles: str,
    lang: str,
    work_dpi: int,
    min_letters: int,
    max_dpi: int,
    min_font_px: int,
    surya: bool,
    surya_batch: int,
    no_deskew: bool,
    pairs_width: int,
    skip_done: bool,
) -> None:
    """Прогнать все таблицы пака из базы."""
    _setup_logging(out_dir)
    use_surya = surya and second_opinion.surya_available()
    if surya and not use_surya:
        logger.warning("surya недоступна — второго мнения не будет")
    options = pipeline.Options(
        work_dpi=work_dpi,
        allowed=parse_allowed(angles),
        lang=lang,
        min_letters=min_letters,
        deskew=not no_deskew,
        max_dpi=max_dpi,
        min_font_em_px=min_font_px,
        use_surya=use_surya,
    )
    refs = select_tables(db_path, pack_name, images_root, only_year, only_issue, limit, only_table)
    if skip_done:
        refs = [ref for ref in refs if not (out_dir / "info" / f"{ref.table_id}.json").is_file()]
    logger.info("таблиц к обработке: %d, воркеров: %d, углы: %s", len(refs), jobs, options.allowed)
    if not refs:
        return
    started = time.time()
    context = multiprocessing.get_context("forkserver")

    # Фаза 1.
    analyses: list[pipeline.TableAnalysis] = []
    with ProcessPoolExecutor(max_workers=max(1, jobs), mp_context=context, initializer=_init_worker) as pool:
        futures = {pool.submit(_analyse_job, ref, options): ref for ref in refs}
        for index, future in enumerate(as_completed(futures), start=1):
            ref = futures[future]
            try:
                analyses.append(future.result())
            except Exception as error:  # noqa: BLE001
                logger.exception("%s: разбор упал", ref.table_id)
                analyses.append(
                    pipeline.TableAnalysis(ref=ref, work_dpi=work_dpi, source_dpi=ref.dpi, error=f"ошибка: {error}")
                )
            if index % 50 == 0 or index == len(refs):
                logger.info("фаза 1: %d/%d, %.0f с", index, len(refs), time.time() - started)
    analyses.sort(key=lambda item: item.ref.table_id)
    with_rotated = [item for item in analyses if item.has_rotated_text]
    logger.info("таблиц с повёрнутым текстом: %d из %d", len(with_rotated), len(analyses))

    # Фаза 2.
    if use_surya:
        keys = [(item, key) for item in with_rotated for key in item.second_opinion]
        logger.info("фаза 2: surya на %d ячеек", len(keys))
        if keys:
            crops = [item.second_opinion[key] for item, key in keys]
            texts = second_opinion.read_cells(crops, surya_batch)
            accepted = 0
            for (item, key), text in zip(keys, texts):
                accepted += pipeline.apply_second_opinion(item, key, text, options)
            logger.info("фаза 2: принято ответов surya: %d из %d", accepted, len(keys))
    for item in analyses:
        item.second_opinion = {}

    # Фаза 3.
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, jobs), mp_context=context, initializer=_init_worker) as pool:
        futures = {pool.submit(_rewrite_job, item, options, str(out_dir), pairs_width): item for item in analyses}
        for index, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            try:
                rows.append(future.result())
            except Exception as error:  # noqa: BLE001
                logger.exception("%s: третья фаза упала", item.ref.table_id)
                rows.append({"table_id": item.ref.table_id, "error": f"ошибка: {error}"})
            if index % 50 == 0 or index == len(analyses):
                logger.info("фаза 3: %d/%d, %.0f с", index, len(analyses), time.time() - started)
    rows.sort(key=lambda row: row.get("table_id", ""))
    _write_summary(out_dir / "summary.csv", rows)
    actions = {}
    for row in rows:
        actions[row.get("action", "")] = actions.get(row.get("action", ""), 0) + 1
    logger.info("готово за %.0f с; действия: %s; сводка: %s", time.time() - started, actions, out_dir / "summary.csv")


def _write_summary(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


@main.command("sheets")
@click.option("--out-dir", type=click.Path(file_okay=False, exists=True, path_type=Path), required=True)
@click.option("--per-sheet", type=int, default=4, show_default=True)
@click.option("--width", type=int, default=2000, show_default=True)
def sheets_command(out_dir: Path, per_sheet: int, width: int) -> None:
    """Собрать пары «было-стало» из pairs/ в листы sheets/."""
    pairs = sorted((out_dir / "pairs").glob("*.png"))
    written = sheets.build_sheets(pairs, out_dir / "sheets", "sheet", per_sheet, width)
    click.echo(f"пар: {len(pairs)}, листов: {len(written)} → {out_dir / 'sheets'}")
