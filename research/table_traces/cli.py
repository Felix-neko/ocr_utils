"""Команды стенда: ``select`` — самые перекошенные таблицы пака по базе, ``run`` — трассы, обе сетки, CSV и оверлеи."""

from __future__ import annotations

import csv
import json
import logging
import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.page_layout.tables import grid as grid_module
from ocr_utils.page_layout.tables.curved_grid import curved_table
from ocr_utils.page_layout.tables.ruling import find_lines, mm_to_px

logger = logging.getLogger(__name__)

# Разрешение сканов «Готовое» пака-1 и рабочее разрешение трасс и сеток.
SCAN_DPI = 600
WORK_DPI = grid_module.WORK_DPI

# Поле вокруг рамки таблицы при вырезке: как у ``example.CROP_PAD_MM``.
CROP_PAD_MM = 3.0


@dataclass(frozen=True)
class Candidate:
    """Таблица из базы разметки: где лежит скан, рамка в пикселях скана, наклон и разброс углов от детектора."""

    rank: int
    page_id: int
    rel_path: str
    x0: int
    y0: int
    x1: int
    y1: int
    skew_deg: float
    angle_spread: float
    named: bool  # взята по имени (известный трудный случай), а не по рангу


@dataclass(frozen=True)
class TableResult:
    """Итог по одной таблице: обе сетки и сводка по трассам."""

    rank: int
    rel_path: str
    x0: int
    y0: int
    skew_deg: float
    angle_spread: float
    axis_rows: int
    axis_cols: int
    axis_cells: int
    curved_rows: int
    curved_cols: int
    curved_cells: int
    curved_merged: int
    curved_irregular: int
    curved_header_rows: int
    doubles: int
    traces: int
    max_chord_deviation_mm: float
    max_abs_local_angle_deg: float
    max_angle_range_deg: float
    seconds: float
    overlay: str


# Известные трудные случаи: берутся всегда, даже если не попали в верх ранга.
NAMED = ("1968/03/IMG_0107_1L", "1968/05/IMG_0089_1L", "1976/08/IMG_0067_1L", "1967/10/IMG_0033_1L")


@click.group()
def main() -> None:
    """Стенд трасс линеек и сетки по кривым."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@main.command()
@click.option("--db", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="pack1.sqlite")
@click.option("--top", type=int, default=40, show_default=True, help="Сколько таблиц брать по рангу.")
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True, help="CSV кандидатов.")
def select(db: Path, top: int, out: Path) -> None:
    """Таблицы с наибольшим max(|наклон|, разброс углов) по ``detector_info`` базы плюс известные трудные случаи."""
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)  # база только на чтение
    rows = connection.execute(
        "select r.page_id, r.x1, r.y1, r.x2, r.y2, r.detector_info, p.source_rel_path "
        "from rect_regions r join pages p on p.id = r.page_id where r.kind = 'table' and r.detector_info is not null"
    ).fetchall()
    scored = []
    for page_id, x0, y0, x1, y1, info, rel in rows:
        details = json.loads(info)
        skew = float(details.get("skew_deg", 0.0))
        spread = float(details.get("metrics", {}).get("angle_spread", 0.0))
        scored.append((max(abs(skew), spread), page_id, rel, x0, y0, x1, y1, skew, spread))
    scored.sort(key=lambda item: -item[0])
    chosen = scored[:top] + [item for item in scored[top:] if item[2].rsplit(".", 1)[0] in NAMED]
    candidates = [
        Candidate(rank, page_id, rel, x0, y0, x1, y1, skew, spread, rank >= top)
        for rank, (_score, page_id, rel, x0, y0, x1, y1, skew, spread) in enumerate(chosen)
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(candidates[0])))
        writer.writeheader()
        writer.writerows(asdict(candidate) for candidate in candidates)
    logger.info("Кандидатов %d (по рангу %d, по имени %d) → %s", len(candidates), top, len(candidates) - top, out)


def _init_worker() -> None:
    """Воркер пула: без hugepage-пометки и без потоков BLAS/OpenCV (см. .claude/rules/gpu_and_pools.md)."""
    import threadpoolctl

    try:
        np._core.multiarray._set_madvise_hugepage(False)  # type: ignore[attr-defined]
    except AttributeError:
        pass
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)


def _crop(pack_dir: Path, candidate: Candidate) -> tuple[np.ndarray, int, int]:
    """Вырезка таблицы в рабочем разрешении с полем ``CROP_PAD_MM``; скан читается только на чтение.

    Returns:
        ``(вырезка 300 dpi, x0, y0 вырезки в пикселях скана)``.
    """
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(pack_dir / candidate.rel_path) as image:
        pad = mm_to_px(CROP_PAD_MM, SCAN_DPI)
        x0, y0 = max(0, candidate.x0 - pad), max(0, candidate.y0 - pad)
        x1, y1 = min(image.width, candidate.x1 + pad), min(image.height, candidate.y1 + pad)
        crop = np.asarray(image.crop((x0, y0, x1, y1)).convert("L"))
    factor = WORK_DPI / SCAN_DPI
    return cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA), x0, y0


def _process(task: tuple[Candidate, Path, Path]) -> tuple[TableResult, list[dict]]:
    """Одна таблица: вырезка → сетка по осям и сетка по кривым → оверлей и сводка.

    Args:
        task: ``(кандидат, папка пака, папка оверлеев)``.

    Returns:
        Итог по таблице и строки по каждой трассе.
    """
    from research.table_traces.overlay import draw_pair

    candidate, pack_dir, overlay_dir = task
    work, _x0, _y0 = _crop(pack_dir, candidate)
    started = time.perf_counter()
    lines = find_lines(work, WORK_DPI)
    axis = grid_module.grid_from_lines(lines, work.shape[:2], WORK_DPI, grid_module.text_ink(work, lines))
    traces, curved = curved_table(work, WORK_DPI)
    seconds = time.perf_counter() - started

    stem = candidate.rel_path.rsplit(".", 1)[0].replace("/", "_")
    overlay_path = overlay_dir / f"{candidate.rank:02d}_{stem}_{candidate.x0}_{candidate.y0}.jpg"
    cv2.imwrite(str(overlay_path), draw_pair(work, axis, lines, curved, traces), [cv2.IMWRITE_JPEG_QUALITY, 85])

    rule_rows: list[dict] = []
    deviations, angles, ranges = [0.0], [0.0], [0.0]
    for index, trace in enumerate(traces.all):
        local = trace.angle_deg_at(trace.along_grid(1.0))
        deviation, _where = trace.max_chord_deviation_px()
        deviation_mm = deviation * 25.4 / WORK_DPI
        deviations.append(deviation_mm)
        angles.append(float(np.abs(local).max()))
        ranges.append(float(local.max() - local.min()))
        rule_rows.append(
            {
                "rank": candidate.rank,
                "rel_path": candidate.rel_path,
                "rule": index,
                "axis": trace.axis.value,
                "length_mm": round(trace.length_mm, 1),
                "chord_angle_deg": round(trace.chord_angle_deg, 2),
                "angle_min_deg": round(float(local.min()), 2),
                "angle_max_deg": round(float(local.max()), 2),
                "max_chord_deviation_mm": round(deviation_mm, 3),
                "bandwidth_mm": trace.bandwidth_mm,
                "residual_rms_px": round(trace.residual_rms_px, 3),
                "residual_max_px": round(trace.residual_max_px, 2),
                "coverage": round(trace.coverage, 3),
                "gaps": len(trace.gaps),
            }
        )
    result = TableResult(
        rank=candidate.rank,
        rel_path=candidate.rel_path,
        x0=candidate.x0,
        y0=candidate.y0,
        skew_deg=round(candidate.skew_deg, 2),
        angle_spread=round(candidate.angle_spread, 2),
        axis_rows=axis.n_rows,
        axis_cols=axis.n_cols,
        axis_cells=len(axis.cells),
        curved_rows=curved.n_rows,
        curved_cols=curved.n_cols,
        curved_cells=len(curved.cells),
        curved_merged=sum(1 for cell in curved.cells if cell.row_span * cell.col_span > 1),
        curved_irregular=sum(1 for cell in curved.cells if cell.irregular),
        curved_header_rows=curved.header_rows,
        doubles=sum(1 for boundary in curved.rows if boundary.is_double),
        traces=len(traces.all),
        max_chord_deviation_mm=round(max(deviations), 3),
        max_abs_local_angle_deg=round(max(angles), 2),
        max_angle_range_deg=round(max(ranges), 2),
        seconds=round(seconds, 2),
        overlay=overlay_path.name,
    )
    return result, rule_rows


def _read_candidates(path: Path) -> list[Candidate]:
    with path.open() as handle:
        return [
            Candidate(
                int(row["rank"]),
                int(row["page_id"]),
                row["rel_path"],
                int(row["x0"]),
                int(row["y0"]),
                int(row["x1"]),
                int(row["y1"]),
                float(row["skew_deg"]),
                float(row["angle_spread"]),
                row["named"] == "True",
            )
            for row in csv.DictReader(handle)
        ]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@main.command()
@click.option("--candidates", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--pack-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option(
    "--jobs",
    type=int,
    default=8,
    show_default=True,
    help="Воркеров пула. 8, а не 16: сканы читаются с NTFS-3G (/mnt/dump3), задача упирается в диск.",
)
def run(candidates: Path, pack_dir: Path, out_dir: Path, jobs: int) -> None:
    """Трассы, сетка по осям и сетка по кривым для каждой таблицы: CSV по таблицам и по линейкам, оверлеи."""
    items = _read_candidates(candidates)
    overlay_dir = out_dir / "оверлеи"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(item, pack_dir, overlay_dir) for item in items]
    results: list[TableResult] = []
    rule_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        for result, rules in pool.map(_process, tasks):
            results.append(result)
            rule_rows.extend(rules)
            logger.info(
                "%02d %s: по осям %d×%d (%d), по кривым %d×%d (%d), откл. хорды до %.2f мм",
                result.rank,
                result.rel_path,
                result.axis_rows,
                result.axis_cols,
                result.axis_cells,
                result.curved_rows,
                result.curved_cols,
                result.curved_cells,
                result.max_chord_deviation_mm,
            )
    _write_csv(out_dir / "таблицы.csv", [asdict(result) for result in sorted(results, key=lambda r: r.rank)])
    _write_csv(out_dir / "линейки.csv", rule_rows)
    logger.info("Готово: %d таблиц, %d линеек → %s (pid %d)", len(results), len(rule_rows), out_dir, os.getpid())
