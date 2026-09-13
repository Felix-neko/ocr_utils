"""Команды пакета: ``uv run python -m research.legacy.table_processing <команда>``.

Разделение то же, что в остальных исследовательских пакетах: тяжёлые команды считают и
пишут CSV, дешёвые собирают из CSV отчёты и картинки. Пороги калибруются итеративно, и
повторный отчёт не должен трогать ни диск с полосами, ни GPU.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click

from research.legacy.table_processing import paths as default_paths
from research.legacy.table_processing.mining.export import (
    MinedTable,
    export_crop,
    issues_in,
    mine_issue,
    read_manifest,
    write_manifest,
)

logger = logging.getLogger(__name__)

# Порог счёта мешанины по умолчанию. Замер на 1966/01: шесть известных испорченных таблиц
# набирают 0.25-0.64, чистые — 0.01-0.19, а единственная пограничная (0.235) на скане
# оказывается таблицей без бокового текста. Порог 0.20 берёт всё известное с запасом.
DEFAULT_MIN_RANK = 0.20

# Ниже этого числа воркеров не опускаемся даже при --jobs 0.
MIN_JOBS = 1


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(levelname)s %(message)s")


def _init_worker() -> None:
    """Воркер не должен разбредаться по ядрам поверх пула: BLAS и OpenCV в один поток."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:
        pass
    os.nice(10)


def _mine_one(args: tuple) -> list[MinedTable]:
    issue, min_rank, docx_dir, pdf_dir, sharpened_dir, db_path, pack_name, source_dpi = args
    try:
        return mine_issue(issue, min_rank, docx_dir, pdf_dir, sharpened_dir, db_path, pack_name, source_dpi)
    except Exception as error:  # один сломанный выпуск не должен ронять прогон по паку
        logger.error("Выпуск %s: %s", issue, error)
        return []


def _export_one(args: tuple) -> str:
    record, sharpened_dir, out_dir, source_dpi = args
    return export_crop(record, sharpened_dir, out_dir, source_dpi)


@click.group()
@click.option("--log-level", default="INFO", show_default=True)
def main(log_level: str) -> None:
    """Таблицы с повёрнутым текстом: поиск, распознание, подмена."""
    _configure_logging(log_level)


@main.command()
@click.option("--out-dir", type=Path, default=default_paths.DEFAULT_OUT_DIR, show_default=True)
@click.option("--docx-dir", type=Path, default=default_paths.DOCX_DIR, show_default=True)
@click.option("--pdf-dir", type=Path, default=default_paths.RECOGNIZED_PDF_DIR, show_default=True)
@click.option("--sharpened-dir", type=Path, default=default_paths.SHARPENED_DIR, show_default=True)
@click.option("--db", "db_path", type=Path, default=default_paths.MARKUP_DB, show_default=True)
@click.option("--pack", "pack_name", default=default_paths.PACK_NAME, show_default=True)
@click.option("--min-rank", type=float, default=DEFAULT_MIN_RANK, show_default=True)
@click.option("--issue", "only_issues", multiple=True, help="Только эти выпуски, например 1966_01.")
@click.option("--jobs", type=int, default=default_paths.DEFAULT_JOBS, show_default=True)
@click.option("--export/--no-export", default=True, show_default=True, help="Сразу вырезать таблицы в 600 dpi.")
def mine(
    out_dir: Path,
    docx_dir: Path,
    pdf_dir: Path,
    sharpened_dir: Path,
    db_path: Path,
    pack_name: str,
    min_rank: float,
    only_issues: tuple[str, ...],
    jobs: int,
    export: bool,
) -> None:
    """Найти испорченные таблицы в выгрузке DOCX и вырезать их из сканов."""
    issues = list(only_issues) or issues_in(docx_dir)
    logger.info("Выпусков: %d, порог счёта %.2f", len(issues), min_rank)

    tasks = [
        (issue, min_rank, docx_dir, pdf_dir, sharpened_dir, db_path, pack_name, default_paths.SOURCE_DPI)
        for issue in issues
    ]
    records: list[MinedTable] = []
    workers = max(MIN_JOBS, jobs)
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            for found in pool.map(_mine_one, tasks):
                records.extend(found)
    else:
        for task in tasks:
            records.extend(_mine_one(task))

    manifest = out_dir / "manifest.csv"
    write_manifest(records, manifest)
    logger.info("Найдено таблиц: %d, манифест %s", len(records), manifest)

    if export and records:
        crops_dir = out_dir / "crops"
        export_tasks = [(record, sharpened_dir, crops_dir, default_paths.SOURCE_DPI) for record in records]
        errors: list[str] = []
        if workers > 1 and len(export_tasks) > 1:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
                errors = [message for message in pool.map(_export_one, export_tasks) if message]
        else:
            errors = [message for message in (_export_one(task) for task in export_tasks) if message]
        logger.info("Вырезано в %s, ошибок %d", crops_dir, len(errors))
        for message in errors[:20]:
            logger.error("%s", message)


def _process_one(args: tuple) -> dict:
    crop_path, dpi, source_dpi, detectors, engine, out_dir, render, overlay = args
    from research.legacy.table_processing.pipeline import process

    try:
        outcome = process(crop_path, dpi, source_dpi, detectors, engine, render)
    except Exception as error:  # одна битая вырезка не должна ронять прогон
        logger.error("%s: %s", crop_path.name, error)
        return {"crop_id": crop_path.stem, "error": str(error)[:200]}

    import cv2

    from research.legacy.table_processing.render.sheet import write_pair

    if overlay:
        from research.legacy.table_processing.pipeline import overlay as draw_overlay

        (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(out_dir / "overlays" / f"{outcome.crop_id}.png"),
            draw_overlay(outcome.before, outcome.grid, outcome.decisions),
        )
    if render and outcome.after is not None:
        write_pair(outcome.before, outcome.after, outcome.crop_id, out_dir / "pairs" / f"{outcome.crop_id}.png")
        cv2.imwrite(str(out_dir / "after" / f"{outcome.crop_id}.png"), outcome.after)
    return {
        "crop_id": outcome.crop_id,
        "rows": outcome.grid.n_rows,
        "cols": outcome.grid.n_cols,
        "cells": len(outcome.grid.cells),
        "rotated": len(outcome.rotated_cells),
        "prior": outcome.prior,
        "engine": outcome.engine,
        "widened": len(outcome.widened),
        "failed": len(outcome.failed),
        "texts": " | ".join(cell.text for cell in outcome.rotated_cells if cell.text),
        "error": "",
    }


@main.command()
@click.option("--out-dir", type=Path, default=default_paths.DEFAULT_OUT_DIR, show_default=True)
@click.option("--crops-dir", type=Path, default=None, help="По умолчанию — crops/ внутри --out-dir.")
@click.option("--crop", "only_crops", multiple=True, help="Только эти вырезки (без расширения).")
@click.option("--detectors", default="", help="Список детекторов поворота через запятую.")
@click.option("--engine", default="tesseract", show_default=True)
@click.option("--dpi", type=int, default=300, show_default=True)
@click.option("--jobs", type=int, default=default_paths.DEFAULT_JOBS, show_default=True)
@click.option("--render/--no-render", default=True, show_default=True)
@click.option("--overlay/--no-overlay", default=True, show_default=True, help="Рисовать разметку ячеек.")
def run(
    out_dir: Path,
    crops_dir: Path | None,
    only_crops: tuple[str, ...],
    detectors: str,
    engine: str,
    dpi: int,
    jobs: int,
    render: bool,
    overlay: bool,
) -> None:
    """Сквозной прогон по вырезанным таблицам: сетка, боковые ячейки, текст, замена."""
    import csv

    crops = crops_dir or (out_dir / "crops")
    paths = sorted(crops.glob("*.png"))
    if only_crops:
        wanted = set(only_crops)
        paths = [path for path in paths if path.stem in wanted]
    if not paths:
        logger.error("В %s нет вырезок", crops)
        return

    (out_dir / "after").mkdir(parents=True, exist_ok=True)
    names = tuple(name.strip() for name in detectors.split(",") if name.strip()) or None
    tasks = [(path, dpi, default_paths.SOURCE_DPI, names, engine, out_dir, render, overlay) for path in paths]

    rows: list[dict] = []
    workers = max(MIN_JOBS, jobs)
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            rows = list(pool.map(_process_one, tasks))
    else:
        rows = [_process_one(task) for task in tasks]

    report = out_dir / "tables.csv"
    fields = ["crop_id", "rows", "cols", "cells", "rotated", "prior", "engine", "widened", "failed", "texts", "error"]
    with report.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})
    rotated = sum(int(row.get("rotated") or 0) for row in rows)
    logger.info("Таблиц %d, боковых ячеек %d, отчёт %s", len(rows), rotated, report)


@main.command("pairs")
@click.option("--out-dir", type=Path, default=default_paths.DEFAULT_OUT_DIR, show_default=True)
@click.option("--only-rotated/--all", default=True, show_default=True)
@click.option("--per-sheet", type=int, default=4, show_default=True)
def pairs(out_dir: Path, only_rotated: bool, per_sheet: int) -> None:
    """Собрать готовые пары «было-стало» в листы."""
    import csv

    from research.legacy.table_processing.render.sheet import sheets

    wanted: set[str] | None = None
    report = out_dir / "tables.csv"
    if only_rotated and report.is_file():
        with report.open(encoding="utf-8", newline="") as handle:
            wanted = {row["crop_id"] for row in csv.DictReader(handle) if int(row.get("rotated") or 0) > 0}
    paths = sorted(path for path in (out_dir / "pairs").glob("*.png") if wanted is None or path.stem in wanted)
    if not paths:
        logger.error("Нет пар в %s", out_dir / "pairs")
        return
    written = sheets(paths, out_dir / "sheets", "before_after", per_sheet)
    logger.info("Пар %d, листов %d: %s", len(paths), len(written), out_dir / "sheets")


def _audit_one(args: tuple) -> list[dict]:
    """Все находки детектора на одной полосе с признаками и вердиктом."""
    path, rel_path, dpi, source_dpi, out_dir = args
    import cv2

    from research.legacy.table_processing.check import findings_of
    from research.legacy.table_processing.imaging import load_gray, to_rgb

    try:
        gray = load_gray(path, dpi, source_dpi)
    except Exception as error:
        logger.error("%s: %s", rel_path, error)
        return []

    found = findings_of(gray, dpi)
    if not found:
        return []

    overlay = to_rgb(gray)
    rows: list[dict] = []
    for finding in found:
        box, signs, good, reason, index = finding.box, finding.signs, finding.verdict, finding.reason, finding.index
        crop = gray[box.slice]
        name = f"{rel_path.replace('/', '_').replace('.jpg', '')}_{index:02d}.png"
        folder = out_dir / ("принятые" if good else "отклонённые")
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / name), crop)
        colour = (0, 150, 0) if good else (0, 0, 220)
        cv2.rectangle(overlay, (box.x0, box.y0), (box.x1, box.y1), colour, 3)
        cv2.putText(overlay, name[-6:-4], (box.x0 + 4, box.y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        rows.append(
            {
                "scan_rel_path": rel_path,
                "index": index,
                "x0": box.x0,
                "y0": box.y0,
                "x1": box.x1,
                "y1": box.y1,
                "dpi": dpi,
                "verdict": "таблица" if good else "нет",
                "reason": reason,
                "crop_file": name,
                **signs.as_row(),
            }
        )
    pages_dir = out_dir / "страницы"
    pages_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(pages_dir / f"{rel_path.replace('/', '_')}.png"), overlay)
    return rows


def _broken_one(args: tuple) -> list:
    issue, docx_dir, pdf_dir, sharpened_dir, db_path, pack_name, source_dpi = args
    from research.legacy.table_processing.mining.broken import find_issue

    try:
        return find_issue(issue, docx_dir, pdf_dir, sharpened_dir, db_path, pack_name, source_dpi)
    except Exception as error:  # один выпуск не должен ронять прогон по паку
        logger.error("Выпуск %s: %s", issue, error)
        return []


@main.command("find-broken")
@click.option("--out-dir", type=Path, default=default_paths.DEFAULT_OUT_DIR, show_default=True)
@click.option("--docx-dir", type=Path, default=default_paths.DOCX_DIR, show_default=True)
@click.option("--pdf-dir", type=Path, default=default_paths.RECOGNIZED_PDF_DIR, show_default=True)
@click.option("--sharpened-dir", type=Path, default=default_paths.SHARPENED_DIR, show_default=True)
@click.option("--db", "db_path", type=Path, default=default_paths.MARKUP_DB, show_default=True)
@click.option("--pack", "pack_name", default=default_paths.PACK_NAME, show_default=True)
@click.option("--issue", "only_issues", multiple=True)
@click.option("--jobs", type=int, default=default_paths.DEFAULT_JOBS, show_default=True)
def find_broken(
    out_dir: Path,
    docx_dir: Path,
    pdf_dir: Path,
    sharpened_dir: Path,
    db_path: Path,
    pack_name: str,
    only_issues: tuple[str, ...],
    jobs: int,
) -> None:
    """Найти таблицы, которые есть на скане, но которых нет в выгрузке DOCX."""
    import cv2

    from research.legacy.table_processing.imaging import load_gray
    from research.legacy.table_processing.mining.broken import write_csv
    from research.legacy.table_processing.mining.export import issues_in
    from research.legacy.table_processing.report import contact_sheets, downscale

    issues = list(only_issues) or issues_in(docx_dir)
    tasks = [
        (issue, docx_dir, pdf_dir, sharpened_dir, db_path, pack_name, default_paths.SOURCE_DPI) for issue in issues
    ]
    records = []
    workers = max(MIN_JOBS, jobs)
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            for found in pool.map(_broken_one, tasks):
                records.extend(found)
    else:
        for task in tasks:
            records.extend(_broken_one(task))

    report = out_dir / "broken.csv"
    write_csv(records, report)
    by_category: dict[str, int] = {}
    for record in records:
        by_category[record.category] = by_category.get(record.category, 0) + 1
    logger.info("Находок %d: %s. Отчёт %s", len(records), by_category, report)

    missed = [r for r in records if r.category == "пропущена"]
    tiles = []
    for record in sorted(missed, key=lambda item: -item.garbage_share):
        gray = load_gray(sharpened_dir / record.scan_rel_path, record.dpi)
        box = record.box.padded(6).clipped(gray.shape[1], gray.shape[0])
        caption = f"{record.issue} с.{record.page_number} мусор {record.garbage_share:.2f}"
        tiles.append((caption, downscale(gray[box.slice], 900)))
    written = contact_sheets(tiles, out_dir / "листы_пропущенных", "пропущенные")
    logger.info("Пропущенных таблиц %d, листов %d: %s", len(missed), len(written), out_dir / "листы_пропущенных")


def _diagnose_one(args: tuple) -> dict:
    record, sharpened_dir, pdf_dir, nogeo_dir, out_dir, dpi, source_dpi, draw = args
    import cv2

    from research.legacy.table_processing.geometry import Box
    from research.legacy.table_processing.imaging import load_gray
    from research.legacy.table_processing.warping import diagnose as diagnosis

    try:
        gray = load_gray(sharpened_dir / record.scan_rel_path, dpi, source_dpi)
    except Exception as error:
        logger.error("%s: %s", record.scan_rel_path, error)
        return {}
    scale = dpi / record.dpi
    box = Box(int(record.x0 * scale), int(record.y0 * scale), int(record.x1 * scale), int(record.y1 * scale)).clipped(
        gray.shape[1], gray.shape[0]
    )

    corrected = diagnosis.render_page(pdf_dir / f"full_{record.issue}.pdf", record.page_number, dpi)
    nogeo_path = nogeo_dir / f"full_{record.issue}.pdf"
    uncorrected = diagnosis.render_page(nogeo_path, record.page_number, dpi) if nogeo_path.is_file() else None

    verdict, scan_geo, corr_geo, nocorr_geo, note, boxes = diagnosis.diagnose(gray, box, corrected, uncorrected, dpi)
    result = diagnosis.Diagnosis(
        issue=record.issue,
        page_number=record.page_number,
        scan_rel_path=record.scan_rel_path,
        verdict=verdict,
        scan=scan_geo,
        corrected=corr_geo,
        uncorrected=nocorr_geo,
        note=note,
        boxes=boxes,
    )

    if draw:
        from ocr_utils.dewarp.compare import side_by_side

        panels = [gray[box.slice]]
        for page, name in ((corrected, "corr"), (uncorrected, "nocorr")):
            if page is None or name not in boxes:
                continue
            crop_box = Box(*boxes[name]).clipped(page.shape[1], page.shape[0])
            panels.append(page[crop_box.slice])
        if len(panels) > 1:
            sheet = panels[0]
            for panel in panels[1:]:
                sheet = side_by_side(cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR), cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR))
                sheet = cv2.cvtColor(sheet, cv2.COLOR_BGR2GRAY)
            folder = out_dir / "geometry_pairs" / verdict.replace(" ", "_")
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / f"{record.issue}_p{record.page_number:03d}.png"), sheet)
    return result.as_row()


@main.command("diagnose-geometry")
@click.option("--out-dir", type=Path, default=default_paths.DEFAULT_OUT_DIR, show_default=True)
@click.option("--sharpened-dir", type=Path, default=default_paths.SHARPENED_DIR, show_default=True)
@click.option("--pdf-dir", type=Path, default=default_paths.RECOGNIZED_PDF_DIR, show_default=True)
@click.option("--nogeo-dir", type=Path, default=default_paths.UNCORRECTED_PDF_DIR, show_default=True)
@click.option("--dpi", type=int, default=300, show_default=True)
@click.option("--jobs", type=int, default=default_paths.DEFAULT_JOBS, show_default=True)
@click.option("--draw/--no-draw", default=True, show_default=True, help="Писать листы из трёх панелей.")
def diagnose_geometry(
    out_dir: Path, sharpened_dir: Path, pdf_dir: Path, nogeo_dir: Path, dpi: int, jobs: int, draw: bool
) -> None:
    """Кто искорёжил таблицу: скан, типография или коррекция геометрии FineReader."""
    import csv as csv_module

    from research.legacy.table_processing.mining.broken import read_csv
    from research.legacy.table_processing.warping.metrics import HEADER

    records = [r for r in read_csv(out_dir / "broken.csv") if r.category == "пропущена"]
    if not records:
        logger.error("Нет пропущенных таблиц в %s", out_dir / "broken.csv")
        return
    logger.info("Таблиц на разбор: %d", len(records))

    tasks = [
        (record, sharpened_dir, pdf_dir, nogeo_dir, out_dir, dpi, default_paths.SOURCE_DPI, draw) for record in records
    ]
    rows: list[dict] = []
    workers = max(MIN_JOBS, jobs)
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            rows = [row for row in pool.map(_diagnose_one, tasks) if row]
    else:
        rows = [row for row in (_diagnose_one(task) for task in tasks) if row]

    fields = ["issue", "page_number", "scan_rel_path", "verdict", "note"]
    for prefix in ("scan", "corr", "nocorr"):
        fields.extend(f"{prefix}_{name}" for name in HEADER)
    report = out_dir / "geometry.csv"
    with report.open("w", encoding="utf-8", newline="") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["verdict"])] = counts.get(str(row["verdict"]), 0) + 1
    logger.info("Вердикты: %s. Отчёт %s", counts, report)


def _fix_one(args: tuple) -> list[dict]:
    record, sharpened_dir, out_dir, dpi, source_dpi, names, draw = args
    import cv2

    from research.legacy.table_processing.geometry import Box
    from research.legacy.table_processing.imaging import load_gray
    from research.legacy.table_processing.warping.engines import resolve
    from research.legacy.table_processing.warping.evaluate import apply
    from research.legacy.table_processing.warping.metrics import measure

    try:
        gray = load_gray(sharpened_dir / record.scan_rel_path, dpi, source_dpi)
    except Exception as error:
        logger.error("%s: %s", record.scan_rel_path, error)
        return []
    scale = dpi / record.dpi
    box = (
        Box(int(record.x0 * scale), int(record.y0 * scale), int(record.x1 * scale), int(record.y1 * scale))
        .padded(8)
        .clipped(gray.shape[1], gray.shape[0])
    )
    crop = gray[box.slice]
    before = measure(crop, dpi)

    rows: list[dict] = []
    images: dict[str, "object"] = {}
    for warper in resolve(names):
        image, outcome = apply(crop, dpi, warper, before)
        images[warper.name] = image
        rows.append({"issue": record.issue, "page_number": record.page_number, **outcome.as_row()})

    if draw and rows:
        from ocr_utils.dewarp.compare import side_by_side

        # Победитель выбирается по строкам отчёта: безопасный и сильнее всех снизивший сагитту.
        chosen = min(rows, key=lambda item: (item["safe"] == 0, item["sagitta_ratio"]))
        image = images.get(str(chosen["warper"]))
        if image is not None and chosen["sagitta_ratio"] < 1.0:
            folder = out_dir / "geometry_fixed"
            folder.mkdir(parents=True, exist_ok=True)
            sheet = side_by_side(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR), cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
            name = f"{record.issue}_p{record.page_number:03d}_{chosen['warper']}.png"
            cv2.imwrite(str(folder / name), sheet)
    return rows


@main.command("fix-geometry")
@click.option("--out-dir", type=Path, default=default_paths.DEFAULT_OUT_DIR, show_default=True)
@click.option("--sharpened-dir", type=Path, default=default_paths.SHARPENED_DIR, show_default=True)
@click.option("--warpers", default="", help="Список выпрямителей через запятую.")
@click.option("--only-curved/--all", default=True, show_default=True, help="Только таблицы с вердиктом «крива».")
@click.option("--dpi", type=int, default=300, show_default=True)
@click.option("--jobs", type=int, default=default_paths.DEFAULT_JOBS, show_default=True)
@click.option("--draw/--no-draw", default=True, show_default=True)
def fix_geometry(
    out_dir: Path, sharpened_dir: Path, warpers: str, only_curved: bool, dpi: int, jobs: int, draw: bool
) -> None:
    """Сравнить выпрямители геометрии на таблицах, которые FineReader пропустил."""
    import csv as csv_module

    import numpy as np

    from research.legacy.table_processing.mining.broken import read_csv
    from research.legacy.table_processing.report import markdown_table
    from research.legacy.table_processing.warping.evaluate import HEADER

    records = {(r.issue, r.page_number): r for r in read_csv(out_dir / "broken.csv") if r.category == "пропущена"}
    wanted: list = []
    geometry_csv = out_dir / "geometry.csv"
    if only_curved and geometry_csv.is_file():
        with geometry_csv.open(encoding="utf-8", newline="") as handle:
            for row in csv_module.DictReader(handle):
                if row["verdict"] in ("крива уже на скане", "FineReader искорёжил"):
                    key = (row["issue"], int(row["page_number"]))
                    if key in records:
                        wanted.append(records[key])
    else:
        wanted = list(records.values())
    if not wanted:
        logger.error("Нечего выпрямлять: в %s нет кривых таблиц", geometry_csv)
        return
    logger.info("Таблиц на выпрямление: %d", len(wanted))

    names = tuple(name.strip() for name in warpers.split(",") if name.strip()) or None
    tasks = [(record, sharpened_dir, out_dir, dpi, default_paths.SOURCE_DPI, names, draw) for record in wanted]
    rows: list[dict] = []
    workers = max(MIN_JOBS, jobs)
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            for found in pool.map(_fix_one, tasks):
                rows.extend(found)
    else:
        for task in tasks:
            rows.extend(_fix_one(task))

    report = out_dir / "warp.csv"
    fields = ["issue", "page_number", *HEADER]
    with report.open("w", encoding="utf-8", newline="") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for name in sorted({str(row["warper"]) for row in rows}):
        group = [row for row in rows if row["warper"] == name]
        ratios = np.array([float(row["sagitta_ratio"]) for row in group])
        improved = int((ratios < 0.8).sum())
        summary.append(
            [
                name,
                len(group),
                f"{np.median(ratios):.2f}",
                f"{improved}/{len(group)}",
                f"{np.median([float(r['ink_ratio']) for r in group]):.3f}",
                f"{np.median([float(r['sharpness_ratio']) for r in group]):.3f}",
                f"{np.median([float(r['seconds']) for r in group]):.2f}",
            ]
        )
    header = ("выпрямитель", "таблиц", "сагитта после/до", "улучшил на 20%+", "краска", "резкость", "с/таблица")
    text = markdown_table(header, summary)
    click.echo(text)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/table_geometry.md").write_text(f"# Выпрямители геометрии таблиц\n\n{text}\n", encoding="utf-8")
    logger.info("Отчёт %s, листы %s", report, out_dir / "geometry_fixed")


@main.command("geometry-pairs")
@click.option("--out-dir", type=Path, default=default_paths.DEFAULT_OUT_DIR, show_default=True)
@click.option("--per-sheet", type=int, default=3, show_default=True)
def geometry_pairs(out_dir: Path, per_sheet: int) -> None:
    """Собрать пары «было-стало» после выпрямления в листы."""
    from research.legacy.table_processing.render.sheet import sheets

    paths_list = sorted((out_dir / "geometry_fixed").glob("*.png"))
    if not paths_list:
        logger.error("Нет пар в %s", out_dir / "geometry_fixed")
        return
    written = sheets(paths_list, out_dir / "sheets_geometry", "geometry", per_sheet)
    logger.info("Пар %d, листов %d: %s", len(paths_list), len(written), out_dir / "sheets_geometry")


def _check_one(args: tuple) -> tuple:
    issue, out_dir, docx_dir, pdf_dir, sharpened_dir, db_path, pack_name, source_dpi, draw, layout_dir, ext = args
    from research.legacy.table_processing.check import check_issue

    try:
        return check_issue(
            issue, out_dir, docx_dir, pdf_dir, sharpened_dir, db_path, pack_name, source_dpi, draw, layout_dir, ext
        )
    except Exception as error:  # один выпуск не должен ронять сверку по паку
        logger.error("Выпуск %s: %s", issue, error)
        return [], [], []


@main.command("check-detector")
@click.option("--out-dir", type=Path, default=Path("reports/проверка детектора таблиц"), show_default=True)
@click.option("--docx-dir", type=Path, default=default_paths.DOCX_DIR, show_default=True)
@click.option("--pdf-dir", type=Path, default=default_paths.RECOGNIZED_PDF_DIR, show_default=True)
@click.option("--sharpened-dir", type=Path, default=default_paths.SHARPENED_DIR, show_default=True)
@click.option("--db", "db_path", type=Path, default=default_paths.MARKUP_DB, show_default=True)
@click.option("--pack", "pack_name", default=default_paths.PACK_NAME, show_default=True)
@click.option("--issue", "only_issues", multiple=True)
@click.option("--jobs", type=int, default=default_paths.DEFAULT_JOBS, show_default=True)
@click.option("--draw/--no-draw", default=True, show_default=True, help="Писать debug-картинки.")
@click.option("--sheet-limit", type=int, default=40, show_default=True, help="Плиток на категорию в листах.")
@click.option("--layout-dir", type=Path, default=None, help="Кэш surya layout (layout-pack); без него v4 идёт на CPU.")
@click.option("--ext", default="jpg", show_default=True, help="Расширение полос: jpg (заострённые) или tif (Готовое).")
def check_detector(
    out_dir: Path,
    docx_dir: Path,
    pdf_dir: Path,
    sharpened_dir: Path,
    db_path: Path,
    pack_name: str,
    only_issues: tuple[str, ...],
    jobs: int,
    draw: bool,
    sheet_limit: int,
    layout_dir: Path | None,
    ext: str,
) -> None:
    """Сверить старый детектор, новый и таблицы, которые нашёл сам FineReader."""
    from dataclasses import asdict

    from research.legacy.table_processing.check import (
        CATEGORY_AGREE,
        CATEGORY_BLIND,
        CATEGORY_FR_MISS,
        CATEGORY_LOST,
        CATEGORY_NOISE,
        DOCX_FIELDS,
        FINDING_FIELDS,
        PAGE_FIELDS,
        write_csv,
    )
    from research.legacy.table_processing.mining.export import issues_in

    issues = list(only_issues) or issues_in(docx_dir)
    logger.info("Выпусков на сверку: %d", len(issues))
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        (
            issue,
            out_dir,
            docx_dir,
            pdf_dir,
            sharpened_dir,
            db_path,
            pack_name,
            default_paths.SOURCE_DPI,
            draw,
            layout_dir,
            ext,
        )
        for issue in issues
    ]
    tables: list = []
    findings: list = []
    pages: list = []
    workers = max(MIN_JOBS, jobs)
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            for issue_tables, issue_findings, issue_pages in pool.map(_check_one, tasks):
                tables.extend(issue_tables)
                findings.extend(issue_findings)
                pages.extend(issue_pages)
    else:
        for task in tasks:
            issue_tables, issue_findings, issue_pages = _check_one(task)
            tables.extend(issue_tables)
            findings.extend(issue_findings)
            pages.extend(issue_pages)

    write_csv([asdict(row) for row in tables], DOCX_FIELDS, out_dir / "таблицы_docx.csv")
    write_csv([row.as_row() for row in findings], FINDING_FIELDS, out_dir / "находки.csv")
    write_csv([asdict(row) for row in pages], PAGE_FIELDS, out_dir / "страницы.csv")

    counts: dict[str, int] = {}
    for row in pages:
        counts[row.category] = counts.get(row.category, 0) + 1
    logger.info("Полос с чем-либо: %d. Категории: %s", len(pages), counts)

    _check_sheets(pages, sharpened_dir, out_dir, sheet_limit)
    _check_report(tables, findings, pages, out_dir, counts, layout_dir, sharpened_dir)
    logger.info("Отчёт %s", out_dir / "README.md")


def _check_sheets(pages: list, sharpened_dir: Path, out_dir: Path, limit: int) -> None:
    """Контактные листы по спорным категориям — их и смотрят глазами."""
    import csv as csv_module

    import cv2

    from research.legacy.table_processing.check import CATEGORY_BLIND, CATEGORY_FR_MISS, CATEGORY_LOST, CATEGORY_NOISE
    from research.legacy.table_processing.geometry import Box, union
    from research.legacy.table_processing.imaging import load_gray
    from research.legacy.table_processing.report import contact_sheets, downscale

    boxes: dict[tuple[str, int], list[Box]] = {}
    findings_csv = out_dir / "находки.csv"
    if findings_csv.is_file():
        with findings_csv.open(encoding="utf-8", newline="") as handle:
            for row in csv_module.DictReader(handle):
                key = (row["issue"], int(row["page_number"]))
                boxes.setdefault(key, []).append(Box(int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"])))

    for category in (CATEGORY_LOST, CATEGORY_BLIND, CATEGORY_FR_MISS, CATEGORY_NOISE):
        chosen = [row for row in pages if row.category == category][:limit]
        tiles = []
        for row in chosen:
            try:
                gray = load_gray(sharpened_dir / row.scan_rel_path, 150)
            except Exception:
                continue
            found = boxes.get((row.issue, row.page_number))
            region = union(found) if found else None
            crop = gray[region.padded(8).clipped(gray.shape[1], gray.shape[0]).slice] if region else gray
            tiles.append((f"{row.issue} с.{row.page_number}", downscale(crop, 900)))
        if tiles:
            contact_sheets(tiles, out_dir / "листы", category.replace(" ", "_"))


def _check_report(
    tables: list,
    findings: list,
    pages: list,
    out_dir: Path,
    counts: dict,
    layout_dir: "Path | None" = None,
    sharpened_dir: "Path | None" = None,
) -> None:
    """Отчёт: таблица два на три, разбор по годам и сверка третьей версии с четвёртой."""
    from research.legacy.table_processing.check import (
        CATEGORY_AGREE,
        CATEGORY_BLIND,
        CATEGORY_FR_MISS,
        CATEGORY_LOST,
        CATEGORY_NOISE,
    )
    from research.legacy.table_processing.report import markdown_table

    matched = sum(1 for table in tables if table.page_number)
    old_pages = sum(1 for row in pages if row.old_findings)
    new_pages = sum(1 for row in pages if row.new_findings)
    docx_page_count = sum(1 for row in pages if row.docx_tables)

    grid = markdown_table(
        ("", "оба нашли", "только старый", "ни один"),
        [
            [
                "**FineReader нашёл таблицу**",
                counts.get(CATEGORY_AGREE, 0),
                counts.get(CATEGORY_LOST, 0),
                counts.get(CATEGORY_BLIND, 0),
            ],
            ["**FineReader не нашёл**", counts.get(CATEGORY_FR_MISS, 0), counts.get(CATEGORY_NOISE, 0), "—"],
        ],
    )

    years: dict[str, dict[str, int]] = {}
    for row in pages:
        year = row.issue.split("_")[0]
        bucket = years.setdefault(year, {})
        bucket[row.category] = bucket.get(row.category, 0) + 1
    by_year = markdown_table(
        ("год", CATEGORY_AGREE, CATEGORY_LOST, CATEGORY_BLIND, CATEGORY_FR_MISS, CATEGORY_NOISE),
        [
            [
                year,
                bucket.get(CATEGORY_AGREE, 0),
                bucket.get(CATEGORY_LOST, 0),
                bucket.get(CATEGORY_BLIND, 0),
                bucket.get(CATEGORY_FR_MISS, 0),
                bucket.get(CATEGORY_NOISE, 0),
            ]
            for year, bucket in sorted(years.items())
        ],
    )

    v3_pages = sum(1 for row in pages if row.v3_findings)
    v4_pages = sum(1 for row in pages if row.v4_findings)
    docx_rows = [row for row in pages if row.docx_tables]
    kinds: dict[str, int] = {}
    for row in pages:
        for item in row.v4_kinds.split(","):
            if ":" in item:
                kind, count = item.split(":")
                kinds[kind] = kinds.get(kind, 0) + int(count)
    versions = markdown_table(
        ("", "третья версия", "четвёртая (таблицы)"),
        [
            ["полос с находкой", v3_pages, v4_pages],
            [
                "таблица есть в DOCX и найдена",
                sum(1 for row in docx_rows if row.v3_findings),
                sum(1 for row in docx_rows if row.v4_findings),
            ],
            [
                "таблицы в DOCX нет, а находка есть",
                sum(1 for row in pages if not row.docx_tables and row.v3_findings),
                sum(1 for row in pages if not row.docx_tables and row.v4_findings),
            ],
            [
                "третья нашла, четвёртая потеряла",
                "—",
                sum(1 for row in pages if row.v3_findings and not row.v4_findings),
            ],
            ["четвёртая нашла, третья нет", "—", sum(1 for row in pages if row.v4_findings and not row.v3_findings)],
        ],
    )
    kinds_text = ", ".join(f"{kind}: {count}" for kind, count in sorted(kinds.items())) or "—"
    layout_note = f"с кэшем surya `{layout_dir}`" if layout_dir else "без разметки surya (CPU)"

    text = f"""# Сверка детектора таблиц

Прогон по всему паку-1: {len(pages)} полос, на которых есть хоть что-то — таблица в выгрузке
DOCX или находка детектора. Полосы взяты из `{sharpened_dir}`.

## Третья версия против четвёртой

Четвёртая версия шла {layout_note}. Находок четвёртой версии по видам: {kinds_text}.
В сравнении ниже у четвёртой версии считаются только находки вида «таблица».

{versions}

Оверлеи четвёртой версии — `оверлеи_v4/`, рамки покрашены по виду: таблица зелёная, схема
синяя, рисунок серый.

## Что с чем сравнивалось

* **FineReader** — таблицы, которые он сам создал в DOCX: {len(tables)}, из них привязано к
  странице {matched}. Привязка идёт по редким токенам через текстовый слой распознанного PDF:
  счётчику страниц самого DOCX верить нельзя, он расходится с PDF на ±1 у большинства выпусков.
* **старый детектор** — морфология по линейкам без проверки: находок {len(findings)} на
  {old_pages} полосах.
* **новый детектор** — то же плюс проверка «а таблица ли это»: принято
  {sum(1 for row in findings if row.verdict == 'таблица')} находок на {new_pages} полосах.

Новый детектор — это старый плюс фильтр, поэтому его находки всегда подмножество старых.
Полос с таблицей в DOCX: {docx_page_count}.

## Сравнение по полосам

{grid}

Читается так: **согласие** — FineReader и новый детектор нашли таблицу на одной полосе;
**новый потерял** — старый находил, новый отверг, а таблица там по мнению FineReader есть;
**оба не увидели** — FineReader таблицу нашёл, а детектор линеек не увидел вовсе;
**пропуск FineReader** — детектор нашёл таблицу, а в DOCX её нет; **убранный шум** — старый
детектор срабатывал впустую, новый это убрал.

## По годам

{by_year}

## Артефакты

| папка | что там |
|---|---|
| `оверлеи_старый/` | полосы с рамками ВСЕХ находок старого детектора |
| `оверлеи_новый/` | те же полосы, обведены только принятые находки |
| `оверлеи_v3/` | находки третьей версии |
| `оверлеи_v4/` | находки четвёртой версии, цвет по виду |
| `страницы_finereader/` | исходные полосы, на которых таблицу нашёл FineReader |
| `листы/` | контактные листы по четырём спорным категориям |

Имя файла в трёх наборах одинаковое (`выпуск_сNNN_скан.jpg`), так что одну полосу можно
открыть в трёх папках рядом и сравнить.

| файл | что внутри |
|---|---|
| `страницы.csv` | строка на полосу: сколько таблиц в DOCX, находок старого и нового, категория |
| `таблицы_docx.csv` | все таблицы DOCX: выпуск, номер, размер, страница, уверенность привязки |
| `находки.csv` | каждая находка детектора: рамка, все признаки проверки, вердикт, причина отказа |
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


@main.command("audit-detector")
@click.option("--out-dir", type=Path, default=default_paths.DEFAULT_OUT_DIR, show_default=True)
@click.option("--sharpened-dir", type=Path, default=default_paths.SHARPENED_DIR, show_default=True)
@click.option("--sample", type=int, default=400, show_default=True, help="Сколько полос взять.")
@click.option("--seed", type=int, default=17, show_default=True)
@click.option("--dpi", type=int, default=150, show_default=True)
@click.option("--jobs", type=int, default=default_paths.DEFAULT_JOBS, show_default=True)
def audit_detector(out_dir: Path, sharpened_dir: Path, sample: int, seed: int, dpi: int, jobs: int) -> None:
    """Разбор находок детектора: что принято, что отклонено и почему.

    Пишет оверлей КАЖДОЙ полосы с находками, вырезки по папкам «принятые» и «отклонённые»,
    контактные листы и CSV со всеми признаками — чтобы пороги перебирались по CSV, не
    трогая диск с полосами.
    """
    import csv as csv_module
    import random

    import cv2

    from research.legacy.table_processing.detection.verify import HEADER
    from research.legacy.table_processing.report import contact_sheets, downscale

    audit_dir = out_dir / "detector_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Выборка стратифицирована по годам: пак охватывает 1966-1976, и вёрстка за десять лет
    # заметно менялась — брать подряд означало бы настроиться на один год.
    by_year: dict[str, list[Path]] = {}
    for path in sorted(sharpened_dir.glob("*/*/*.jpg")):
        by_year.setdefault(path.parts[-3], []).append(path)
    rng = random.Random(seed)
    chosen: list[Path] = []
    per_year = max(1, sample // max(1, len(by_year)))
    for year, files in sorted(by_year.items()):
        chosen.extend(rng.sample(files, min(per_year, len(files))))
    logger.info("Полос в выборке: %d из %d лет", len(chosen), len(by_year))

    tasks = [(path, str(path.relative_to(sharpened_dir)), dpi, default_paths.SOURCE_DPI, audit_dir) for path in chosen]
    rows: list[dict] = []
    workers = max(MIN_JOBS, jobs)
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            for found in pool.map(_audit_one, tasks):
                rows.extend(found)
    else:
        for task in tasks:
            rows.extend(_audit_one(task))

    report = audit_dir / "findings.csv"
    fields = ["scan_rel_path", "index", "x0", "y0", "x1", "y1", "dpi", "verdict", "reason", "crop_file", *HEADER]
    with report.open("w", encoding="utf-8", newline="") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    accepted = [r for r in rows if r["verdict"] == "таблица"]
    rejected = [r for r in rows if r["verdict"] != "таблица"]
    for name, group in (("принятые", accepted), ("отклонённые", rejected)):
        tiles = []
        for row in group:
            image = cv2.imread(str(audit_dir / name / row["crop_file"]), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            caption = (
                f"{row['scan_rel_path'][-16:]} {row['reason'][:34]}" if row["reason"] else row["scan_rel_path"][-16:]
            )
            tiles.append((caption, downscale(image, 900)))
        contact_sheets(tiles, audit_dir / "листы", name)
    logger.info(
        "Находок %d: принято %d, отклонено %d. Отчёт %s, листы %s",
        len(rows),
        len(accepted),
        len(rejected),
        report,
        audit_dir / "листы",
    )


@main.command("compare-rotation")
@click.option("--out-dir", type=Path, default=default_paths.DEFAULT_OUT_DIR, show_default=True)
@click.option("--detectors", default="", help="По умолчанию — все из реестра.")
@click.option("--dpi", type=int, default=300, show_default=True)
@click.option("--report", type=Path, default=None, help="Куда положить markdown-отчёт.")
def compare_rotation(out_dir: Path, detectors: str, dpi: int, report: Path | None) -> None:
    """Сравнить детекторы поворота ячейки по ручной разметке labels/cells.csv."""
    import time

    from research.legacy.table_processing.imaging import load_gray
    from research.legacy.table_processing.labels import cells_by_crop, load_cells
    from research.legacy.table_processing.report import markdown_table
    from research.legacy.table_processing.rotation import REGISTRY_ORDER, resolve
    from research.legacy.table_processing.rotation.base import CellCrop
    from research.legacy.table_processing.rotation.evaluate import HEADER, Score
    from research.legacy.table_processing.structure.ruling_grid import cell_image, extract

    truth = cells_by_crop(load_cells())
    if not truth:
        logger.error("Нет разметки в %s", default_paths.LABELS_DIR / "cells.csv")
        return

    names = tuple(name.strip() for name in detectors.split(",") if name.strip()) or REGISTRY_ORDER
    chosen = resolve(names)
    scores = {detector.name: Score(detector.name) for detector in chosen}
    counted = {detector.name: 0 for detector in chosen}

    for crop_id, cells in sorted(truth.items()):
        path = out_dir / "crops" / f"{crop_id}.png"
        if not path.is_file():
            logger.warning("Нет вырезки %s", path)
            continue
        table = load_gray(path, dpi, default_paths.SOURCE_DPI)
        grid = extract(table, dpi)
        crops = [
            CellCrop(crop_id, cell, cell_image(table, cell, dpi), dpi, table=table)
            for cell in grid.cells
            if cell.key in cells
        ]
        if not crops:
            logger.warning("Разметка %s не легла на сетку", crop_id)
            continue
        for detector in chosen:
            started = time.time()
            if detector.stage == "gpu":
                verdicts = detector.make_batch()(crops)
            else:
                verdicts = [detector.run(crop) for crop in crops]
            elapsed = time.time() - started
            scores[detector.name].seconds += elapsed
            counted[detector.name] += len(crops)
            for crop, verdict in zip(crops, verdicts):
                scores[detector.name].add(verdict, cells[crop.cell.key])

    for name, score in scores.items():
        score.seconds = score.seconds / max(1, counted[name])
    rows = [score.as_row() for score in scores.values()]
    table_text = markdown_table(HEADER, rows)
    click.echo(table_text)
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(f"# Детекторы поворота ячейки\n\n{table_text}\n", encoding="utf-8")


@main.command("compare-ocr")
@click.option("--out-dir", type=Path, default=default_paths.DEFAULT_OUT_DIR, show_default=True)
@click.option("--engines", default="", help="По умолчанию — все из реестра.")
@click.option("--dpi", type=int, default=300, show_default=True)
@click.option("--report", type=Path, default=None)
def compare_ocr(out_dir: Path, engines: str, dpi: int, report: Path | None) -> None:
    """Сравнить распознаватели боковых ячеек по эталонному тексту labels/cell_text.csv."""
    from research.legacy.table_processing.imaging import load_gray
    from research.legacy.table_processing.labels import load_texts, texts_by_crop
    from research.legacy.table_processing.ocr import REGISTRY_ORDER, resolve
    from research.legacy.table_processing.ocr.evaluate import ENGINE_HEADER, EngineScore
    from research.legacy.table_processing.report import markdown_table
    from research.legacy.table_processing.rotation.base import rotate_cw
    from research.legacy.table_processing.structure.ruling_grid import cell_image, extract

    truth = texts_by_crop(load_texts())
    if not truth:
        logger.error("Нет разметки в %s", default_paths.LABELS_DIR / "cell_text.csv")
        return

    names = tuple(name.strip() for name in engines.split(",") if name.strip()) or REGISTRY_ORDER
    chosen = resolve(names)
    scores = {engine.name: EngineScore(engine.name) for engine in chosen}

    for crop_id, cells in sorted(truth.items()):
        path = out_dir / "crops" / f"{crop_id}.png"
        if not path.is_file():
            logger.warning("Нет вырезки %s", path)
            continue
        table = load_gray(path, dpi, default_paths.SOURCE_DPI)
        grid = extract(table, dpi)
        targets = [cell for cell in grid.cells if cell.key in cells]
        if not targets:
            continue
        # Поворот берётся из разметки: сравниваем распознавание, а не детекцию поворота.
        images = [rotate_cw(cell_image(table, cell, dpi), 90) for cell in targets]
        for engine in chosen:
            results = engine.make()(images)
            for cell, result in zip(targets, results):
                scores[engine.name].add(cells[cell.key], result.text, result.seconds)

    rows = [score.as_row() for score in scores.values()]
    table_text = markdown_table(ENGINE_HEADER, rows)
    click.echo(table_text)
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(f"# Распознаватели боковых ячеек\n\n{table_text}\n", encoding="utf-8")


@main.command("sheet")
@click.option("--out-dir", type=Path, default=default_paths.DEFAULT_OUT_DIR, show_default=True)
@click.option("--top", type=int, default=100, show_default=True, help="Сколько таблиц с верха рейтинга показать.")
def sheet(out_dir: Path, top: int) -> None:
    """Контактные листы вырезанных таблиц — чтобы просмотреть находки глазами."""
    import cv2

    from research.legacy.table_processing.report import contact_sheets

    records = read_manifest(out_dir / "manifest.csv")[:top]
    tiles = []
    for record in records:
        path = out_dir / "crops" / record.crop_file
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.warning("Нет вырезки %s", path)
            continue
        tiles.append((f"{record.crop_id} r={record.rank:.2f} стр{record.page_number}", image))
    written = contact_sheets(tiles, out_dir / "sheets", "mined")
    logger.info("Листов: %d, первый %s", len(written), written[0] if written else "-")


# Где внутри папки сверки лежат раскладки по диагнозам. Порядок важен: раскладки идут от
# ранней к поздней, и при двух диагнозах на одну полосу побеждает поздний.
LABEL_SUBDIRS = (
    Path("оверлеи_новый") / "проблемные обнаружения",
    Path("нашёл finereader, новый детектор не нашёл"),
    Path("оверлеи_v3") / "проблемные обнаружения",
    Path("оверлеи_v4") / "проблемные оверлеи (не все, оверлеи отсмотрел по 1971-05 включительно)",
)

DEFAULT_LABEL_DIRS = (
    Path("reports/проверка детектора таблиц"),
    Path("reports/сверка детектора v3"),
    Path("reports/сверка детектора v4"),
)


@main.command("labels-from-folders")
@click.option(
    "--check-dir",
    "check_dirs",
    type=Path,
    multiple=True,
    help="Папки сверки с раскладками, от ранней к поздней. По умолчанию — v2 и v3.",
)
@click.option("--out", "out_path", type=Path, default=None, help="Куда писать; по умолчанию labels/detector_pages.csv.")
def labels_from_folders(check_dirs: tuple[Path, ...], out_path: Path | None) -> None:
    """Собрать разметку по полосам из папок, по которым человек разложил оверлеи.

    Разовая команда: раскладку делают руками, а CSV из неё пересобирают, когда папок стало
    больше. Все имена папок обязаны быть описаны в ``labels.DIAGNOSIS_CODES`` — незнакомая
    папка роняет команду намеренно, потому что молча пропущенный диагноз означает разметку,
    в которой чего-то не хватает, а понять этого по числам уже нельзя.
    """
    import csv as csv_module

    from research.legacy.table_processing import labels as labels_module

    folders: list[Path] = []
    for check_dir in check_dirs or DEFAULT_LABEL_DIRS:
        for subdir in LABEL_SUBDIRS:
            root = check_dir / subdir
            if root.is_dir():
                # Сама папка раскладки тоже идёт в список, если для неё есть код: человек кладёт в
                # её корень полосы, диагноз которых записан прямо в имя файла.
                if root.name in labels_module.DIAGNOSIS_CODES:
                    folders.append(root)
                folders += [path for path in sorted(root.iterdir()) if path.is_dir()]
    if not folders:
        raise click.ClickException("Не нашлось ни одной папки с разметкой")

    rows = labels_module.collect_detector_pages(folders)
    target = out_path or default_paths.LABELS_DIR / "detector_pages.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=list(labels_module.DETECTOR_PAGE_FIELDS))
        writer.writeheader()
        for label in rows:
            writer.writerow(
                {
                    "scan_rel_path": label.scan_rel_path,
                    "issue": label.issue,
                    "page": label.page,
                    "diagnosis": label.diagnosis,
                    "severity": label.severity,
                    "expect": label.expect,
                    "note": label.note,
                }
            )
    counts: dict[str, int] = {}
    for label in rows:
        counts[label.diagnosis] = counts.get(label.diagnosis, 0) + 1
    logger.info("Полос: %d, файл %s", len(rows), target)
    for name, count in sorted(counts.items(), key=lambda item: -item[1]):
        logger.info("  %-20s %d", name, count)


VERSION_COLOURS = {"v2": (0, 0, 220), "v3": (200, 90, 0), "v4": (0, 150, 0)}


def _compare_one(args: tuple) -> dict:
    """Одна размеченная полоса: что нашли версии детектора и как хороши их рамки.

    Для четвёртой версии в счёт таблиц идут только находки вида «таблица»; схемы и рисунки
    считаются отдельно (``v4_diagrams``, ``v4_drawings``), а вид проверяется против
    ``labels.KIND_OF_DIAGNOSIS`` (``v4_kind_ok``).
    """
    label, sharpened_dir, dpi, out_dir, draw, names, layout_dir = args
    import cv2

    from research.legacy.table_processing import imaging
    from research.legacy.table_processing import labels as labels_module
    from research.legacy.table_processing.check import KIND_COLOURS
    from research.legacy.table_processing.detection import quality, ruling, ruling_v3, ruling_v4, rules_v4
    from research.legacy.table_processing.layout import surya as layout_module
    from research.legacy.table_processing.structure.ruling_grid import text_ink

    gray = imaging.load_gray(sharpened_dir / label.scan_rel_path, dpi)
    lines = ruling.find_lines(gray, dpi)
    # Компоненты краски — без линеек ОБЕИХ версий: иначе обрывок линейки, который видит одна
    # версия и не видит другая, считался бы буквой под границей и портил меру той, чья рамка
    # по нему проходит.
    import cv2 as cv2_module

    both = ruling.Lines(
        lines.horizontal,
        lines.vertical,
        cv2_module.bitwise_or(lines.horizontal_mask, rules_v4.find_rules(gray, dpi).horizontal_mask),
        cv2_module.bitwise_or(lines.vertical_mask, rules_v4.find_rules(gray, dpi).vertical_mask),
    )
    components = quality.glyph_components(text_ink(gray, both))
    versions: dict = {}
    if "v2" in names:
        versions["v2"] = ruling.detect(gray, dpi)
    if "v3" in names:
        versions["v3"] = ruling_v3.detect(gray, dpi)
    if "v4" in names:
        versions["v4"] = ruling_v4.detect(gray, dpi, layout=layout_module.load(layout_dir, label.scan_rel_path))
    row: dict = {"issue": label.issue, "page": label.page, "diagnosis": label.diagnosis, "expect": label.expect}
    for name, found in versions.items():
        tables = [table for table in found if table.is_table]
        marks = [quality.measure(gray, lines, table.box, dpi) for table in tables]
        row[f"{name}_tables"] = len(tables)
        row[f"{name}_dirty"] = sum(mark.dirty_sides for mark in marks)
        row[f"{name}_crossed"] = sum(sum(quality.crossed_glyphs(components, table.box).values()) for table in found)
        row[f"{name}_outside"] = quality.outside_rules_page(lines, [table.box for table in found], dpi)
        row[f"{name}_foreign"] = round(max((mark.foreign_text for mark in marks), default=0.0), 4)
        if name == "v4":
            row["v4_diagrams"] = sum(1 for table in found if table.kind == "схема")
            row["v4_drawings"] = sum(1 for table in found if table.kind == "рисунок")
            truth = labels_module.KIND_OF_DIAGNOSIS.get(label.diagnosis, "")
            kinds = {table.kind for table in found}
            row["v4_kind_ok"] = "" if not truth or not found else int(truth in kinds and len(kinds) == 1)
    if draw:
        from research.legacy.table_processing.check import image_name
        from research.legacy.table_processing.report import downscale

        for name, found in versions.items():
            canvas = imaging.to_rgb(gray)
            for table in found:
                box = table.box
                colour = KIND_COLOURS[table.kind] if name == "v4" else VERSION_COLOURS[name]
                cv2.rectangle(canvas, (box.x0, box.y0), (box.x1 - 1, box.y1 - 1), colour, 3)
            folder = out_dir / f"оверлеи_{name}" / label.diagnosis
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(folder / image_name(label.issue, label.page, label.scan_rel_path)),
                downscale(canvas, 1100),
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )
    return row


@main.command("compare-detector")
@click.option("--out-dir", type=Path, default=Path("reports/детектор v4"), show_default=True)
@click.option("--sharpened-dir", type=Path, default=default_paths.SHARPENED_DIR, show_default=True)
@click.option("--dpi", type=int, default=150, show_default=True)
@click.option("--jobs", type=int, default=default_paths.DEFAULT_JOBS, show_default=True)
@click.option("--draw/--no-draw", default=True, show_default=True, help="Писать оверлеи по диагнозам.")
@click.option("--versions", default="v3,v4", show_default=True, help="Какие версии гонять: v2, v3, v4 через запятую.")
@click.option("--layout-dir", type=Path, default=None, help="Кэш surya layout для четвёртой версии.")
def compare_detector(
    out_dir: Path, sharpened_dir: Path, dpi: int, jobs: int, draw: bool, versions: str, layout_dir: Path | None
) -> None:
    """Сравнить версии детектора на размеченных полосах (190 полос, две раскладки человека).

    Это ежедневный цикл: он идёт секунды, тогда как полная сверка ``check-detector`` по 12 135
    полосам — приёмка релиза. Метрики считаются по классам диагноза, потому что улучшение
    «в среднем» ничего не значит: правка, которая чинит захват чужого текста и одновременно
    ломает захват целой таблицы, в общем счёте выглядит нейтрально.
    """
    import csv as csv_module

    from research.legacy.table_processing import labels as labels_module
    from research.legacy.table_processing.report import markdown_table

    marked = labels_module.load_detector_pages()
    if not marked:
        raise click.ClickException("Нет разметки: сначала соберите её командой labels-from-folders")
    out_dir.mkdir(parents=True, exist_ok=True)
    names = tuple(name.strip() for name in versions.split(",") if name.strip())
    tasks = [(label, sharpened_dir, dpi, out_dir, draw, names, layout_dir) for label in marked]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker) as pool:
            rows = list(pool.map(_compare_one, tasks, chunksize=2))
    else:
        rows = [_compare_one(task) for task in tasks]

    fields = list(rows[0])
    with (out_dir / "сравнение.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_diagnosis: dict[str, list[dict]] = {}
    for row in rows:
        by_diagnosis.setdefault(row["diagnosis"], []).append(row)
    header = ["диагноз", "полос"]
    for name in names:
        header += [f"нашёл {name}", f"таблиц {name}", f"режет букв {name}", f"чужого {name}"]
    if "v4" in names:
        header += ["схем v4", "рисунков v4", "вид верен v4"]
    table_rows = []
    for diagnosis, group in sorted(by_diagnosis.items(), key=lambda item: -len(item[1])):
        line: list = [diagnosis, len(group)]
        for name in names:
            line += [
                sum(1 for row in group if row[f"{name}_tables"]),
                sum(row[f"{name}_tables"] for row in group),
                sum(row[f"{name}_crossed"] for row in group),
                round(sum(row[f"{name}_foreign"] for row in group) / len(group), 3),
            ]
        if "v4" in names:
            judged = [row for row in group if row["v4_kind_ok"] != ""]
            line += [
                sum(row["v4_diagrams"] for row in group),
                sum(row["v4_drawings"] for row in group),
                f"{sum(row['v4_kind_ok'] for row in judged)}/{len(judged)}" if judged else "—",
            ]
        table_rows.append(line)
    text = markdown_table(header, table_rows)
    click.echo(text)
    wanted = [row for row in rows if row["expect"] == "таблица"]
    empty = [row for row in rows if row["expect"] == "нет"]
    lines = []
    for name in names:
        lines.append(
            f"{name}: ждём таблицу ({len(wanted)} полос) — нашёл {sum(1 for r in wanted if r[f'{name}_tables'])}; "
            f"ждём пусто ({len(empty)} полос) — таблиц {sum(1 for r in empty if r[f'{name}_tables'])}; "
            f"компонент краски под границей {sum(r[f'{name}_crossed'] for r in rows)}."
        )
    summary = "\n".join(lines) + "\n"
    click.echo("\n" + summary)
    layout_note = f"с кэшем surya `{layout_dir}`" if layout_dir else "без разметки surya"
    (out_dir / "README.md").write_text(
        f"# Детектор таблиц: сравнение версий на размеченных полосах\n\n"
        f"Версии: {', '.join(names)}; четвёртая {layout_note}.\n\n{summary}\n{text}\n",
        encoding="utf-8",
    )
    logger.info("Отчёт %s", out_dir / "README.md")


def _candidate_one(args: tuple) -> "str | None":
    """Есть ли на полосе хоть что-то по детектору четвёртой версии (CPU, в пуле)."""
    sharpened_dir, rel_path, dpi = args
    from research.legacy.table_processing import imaging
    from research.legacy.table_processing.detection import ruling_v4

    try:
        gray = imaging.load_gray(sharpened_dir / rel_path, dpi)
    except OSError:
        return None
    return rel_path if ruling_v4.detect(gray, dpi) else None


@main.command("layout-pack")
@click.option("--sharpened-dir", type=Path, default=default_paths.SHARPENED_DIR, show_default=True)
@click.option("--out-dir", type=Path, default=default_paths.DEFAULT_OUT_DIR / "layout_surya", show_default=True)
@click.option("--dpi", type=int, default=150, show_default=True, help="Разрешение копии, которую видит модель.")
@click.option("--mode", type=click.Choice(["all", "candidates"]), default="all", show_default=True)
@click.option("--issue", "only_issues", multiple=True, help="Только эти выпуски, например 1966_01.")
@click.option(
    "--jobs",
    type=int,
    default=default_paths.DEFAULT_JOBS,
    show_default=True,
    help="Воркеров для CPU-отбора кандидатов и чтения.",
)
@click.option("--redo/--no-redo", default=False, show_default=True, help="Пересчитать и уже сохранённые полосы.")
@click.option("--ext", default="jpg", show_default=True, help="Расширение полос в каталоге: jpg или tif.")
@click.option("--source-dpi", type=int, default=default_paths.SOURCE_DPI, show_default=True)
@click.option("--readers", type=int, default=4, show_default=True, help="Потоков чтения файлов (на NTFS-3G лучше 2-3).")
def layout_pack(
    sharpened_dir: Path,
    out_dir: Path,
    dpi: int,
    mode: str,
    only_issues: tuple[str, ...],
    jobs: int,
    redo: bool,
    ext: str,
    source_dpi: int,
    readers: int,
) -> None:
    """Surya layout по паку: pickle на полосу, структура подпапок как у заострённых копий.

    Два режима, как просил человек: ``all`` — каждая полоса пака (около секунды на полосу на
    GPU, по паку 3.5 часа); ``candidates`` — только полосы, где детектор по линейкам что-то
    нашёл, для уточнения (CPU-отбор в пуле, затем модель в родителе).

    GPU только в родителе: модель одна, видеопамять одна. Чтение файлов идёт в потоках, чтобы
    модель не ждала диска.

    ``--ext tif`` — для исходных сканов «Готовое/пак-1» (TIFF 600 dpi, LZW, 38 МБ на полосу,
    на медленном NTFS-3G): та же структура ``{год}/{выпуск}/{основа}``, и кэш пишется в ту же
    структуру, только в другой каталог.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    from research.legacy.table_processing import imaging
    from research.legacy.table_processing.layout import surya as surya_module

    years = sorted(p for p in sharpened_dir.iterdir() if p.is_dir())
    rel_paths: list[str] = []
    for year in years:
        for number in sorted(p for p in year.iterdir() if p.is_dir()):
            issue = f"{year.name}_{number.name}"
            if only_issues and issue not in only_issues:
                continue
            rel_paths += [str(path.relative_to(sharpened_dir)) for path in sorted(number.glob(f"*.{ext}"))]
    if not redo:
        rel_paths = [rel for rel in rel_paths if not surya_module.cache_path(out_dir, rel).is_file()]
    logger.info("Полос к разметке: %d (режим %s)", len(rel_paths), mode)

    if mode == "candidates" and rel_paths:
        tasks = [(sharpened_dir, rel, dpi) for rel in rel_paths]
        with ProcessPoolExecutor(max_workers=max(MIN_JOBS, jobs), initializer=_init_worker) as pool:
            rel_paths = [rel for rel in pool.map(_candidate_one, tasks, chunksize=4) if rel]
        logger.info("Полос-кандидатов: %d", len(rel_paths))

    predictor = surya_module.Predictor()
    started = time.time()
    done = 0
    chunk = surya_module.BATCH * 4

    def read(rel: str):
        try:
            return rel, imaging.load_gray(sharpened_dir / rel, dpi, source_dpi)
        except OSError as error:
            logger.warning("%s: %s", rel, error)
            return rel, None

    with ThreadPoolExecutor(max_workers=max(1, readers)) as pool_readers:
        for start in range(0, len(rel_paths), chunk):
            batch = [item for item in pool_readers.map(read, rel_paths[start : start + chunk]) if item[1] is not None]
            if not batch:
                continue
            outputs = predictor.predict_raw([gray for _, gray in batch])
            for (rel, _), (layout, raw) in zip(batch, outputs):
                surya_module.save(out_dir, rel, layout, raw, dpi)
            done += len(batch)
            if done % (chunk * 5) < len(batch):
                elapsed = time.time() - started
                logger.info("%d/%d полос, %.2f с/полосу", done, len(rel_paths), elapsed / max(1, done))
    logger.info("Готово: %d полос за %.0f с, кэш %s", done, time.time() - started, out_dir)
