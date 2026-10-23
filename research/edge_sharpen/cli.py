"""CLI стенда заострения края: повторная сегментация по фильтрам (``run``), меры края всех вариантов (``measure``), листы вырезок спорного края (``sheets``)."""

from __future__ import annotations

import csv
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, framed, header_strip
from ocr_utils.scan_cropping.v2.pipeline import init_worker, paint_on_seg, read_bgr
from research.edge_sharpen.filters import EdgeFilter, apply_filter
from research.edge_sharpen.measure import OUTER_SIDES, EdgeMeasures, edge_measures
from research.edge_sharpen.segment import SEG_PX_PER_MM, FilterMode, cpu_stage, sam_stage, variant_dir
from ocr_utils.scan_cropping.v2.page_sides import Side, side_profiles, smooth_profile
from research.page_cutting.runs import RunSpec, SetRow, parse_run, read_set, write_image
from research.page_cutting.selection import SMOOTH_WINDOW_MM
from ocr_utils.scan_cropping.v2.page_pose import frame_pages
from ocr_utils.scan_cropping.v2.trapezoid import fit_enclosing_quad

log = logging.getLogger(__name__)

RUN_HELP = "Прогон v2: «имя=папка,подпапка,px_per_mm» (как в research.page_cutting)."
# Вариант «как в боевом прогоне v2» (маска бумаги из кэша v2) — в таблицах и листах.
V2_NAME = "v2"
# Сторона вырезки спорного края на листе, мм, и увеличение.
CROP_MM = 60.0
CROP_ZOOM = 2.0
# Цвет контура бумаги варианта (BGR, палитра /draw-overlay: принято — зелёный) и контур v2 (справочный — серый).
COLOUR_VARIANT = (0, 150, 0)
COLOUR_V2 = (170, 170, 170)
# Вырезок в ряду листа.
SHEET_COLS = 6


def set_rows(set_files: tuple[Path, ...], specs: dict[str, RunSpec]) -> list[SetRow]:
    """Строки наборов с известными прогонами."""
    return [r for f in set_files for r in read_set(f) if r.run in specs]


@click.group()
def main() -> None:
    """Стенд заострения края листа перед SAM 3."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def cpu_task(args: tuple[RunSpec, str, EdgeFilter, FilterMode, Path]) -> str:
    """CPU-часть варианта для пула; сбой — в лог."""
    run, key, filt, mode, out = args
    try:
        return cpu_stage(run, key, filt, mode, out)
    except Exception as exc:  # noqa: BLE001 — один кадр не роняет прогон
        log.warning("%s %s %s %s: %s", run.name, key, filt.value, mode.value, exc)
        return ""


@main.command("run")
@click.option("--run", "runs", multiple=True, required=True, help=RUN_HELP)
@click.option("--set", "set_files", multiple=True, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out-dir", type=click.Path(path_type=Path), required=True, help="Корень выхода стенда.")
@click.option(
    "--filter",
    "filters",
    multiple=True,
    type=click.Choice([f.value for f in EdgeFilter]),
    default=[f.value for f in EdgeFilter],
    show_default=True,
)
@click.option("--jobs", type=int, default=16, show_default=True, help="Воркеров CPU-части.")
def run_cmd(
    runs: tuple[str, ...], set_files: tuple[Path, ...], out_dir: Path, filters: tuple[str, ...], jobs: int
) -> None:
    """SAM 3 по отфильтрованным кадрам (GPU, последовательно), затем бумага и страницы в обоих режимах (CPU, пул)."""
    from ocr_utils.scan_cropping.v2.paper import load_sam3

    specs = {s.name: s for s in (parse_run(r) for r in runs)}
    rows = set_rows(set_files, specs)
    filts = [EdgeFilter(f) for f in filters]
    predictor = load_sam3()
    for filt in filts:
        for row in rows:
            sam_stage(predictor, specs[row.run], row.key, filt, out_dir)
        log.info("SAM 3: %s — %d кадров", filt.value, len(rows))
    del predictor
    tasks = [(specs[r.run], r.key, f, m, out_dir) for f in filts for r in rows for m in FilterMode]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        done = [k for k in pool.map(cpu_task, tasks) if k]
    log.info("бумага: %d из %d вариантов кадров", len(done), len(tasks))


def frame_inputs(run: RunSpec, key: str) -> tuple[dict, np.ndarray]:
    """Маски книги и зона пальца кадра на масштабе сегментации."""
    cache = run.cache_dir / key
    book = dict(np.load(cache / "book.npz"))
    with np.load(cache / "paper.npz") as data:
        shape = data["paper"].shape
    exclude = paint_on_seg(np.load(cache / "fingers.npz"), shape)
    return book, exclude


def variant_masks(run: RunSpec, key: str, out_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Маски бумаги и экземпляры SAM 3 всех вариантов кадра: ``v2`` (кэш прогона) и ``<фильтр>/<режим>``."""
    cache = run.cache_dir / key
    with np.load(cache / "paper.npz") as d, np.load(cache / "sam3.npz") as s:
        out = {V2_NAME: (d["paper"], s["inst"])}
    for filt in EdgeFilter:
        vdir = variant_dir(out_dir, filt, run, key)
        if not (vdir / "sam3.npz").exists():
            continue
        inst = np.load(vdir / "sam3.npz")["inst"]
        for mode in FilterMode:
            path = vdir / f"paper_{mode.value}.npz"
            if path.exists():
                out[f"{filt.value}/{mode.value}"] = (np.load(path)["paper"], inst)
    return out


def measure_task(args: tuple[RunSpec, str, Path]) -> list[dict]:
    """Меры края всех вариантов кадра (строки таблицы)."""
    run, key, out_dir = args
    book, exclude = frame_inputs(run, key)
    rows = []
    for name, (paper, inst) in variant_masks(run, key, out_dir).items():
        m = edge_measures(paper, book["book"], book["hulls"], inst, exclude, SEG_PX_PER_MM)
        rows.append({"run": run.name, "key": key, "variant": name, **asdict(m)})
    return rows


@main.command("measure")
@click.option("--run", "runs", multiple=True, required=True, help=RUN_HELP)
@click.option("--set", "set_files", multiple=True, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out-dir", type=click.Path(path_type=Path), required=True, help="Корень выхода стенда.")
@click.option("--jobs", type=int, default=16, show_default=True)
def measure_cmd(runs: tuple[str, ...], set_files: tuple[Path, ...], out_dir: Path, jobs: int) -> None:
    """Меры края всех вариантов по наборам → ``measures.csv`` (по строке на кадр × вариант)."""
    specs = {s.name: s for s in (parse_run(r) for r in runs)}
    tasks = [(specs[r.run], r.key, out_dir) for r in set_rows(set_files, specs)]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        rows = [row for chunk in pool.map(measure_task, tasks) for row in chunk]
    names = ["run", "key", "variant"] + [f.name for f in fields(EdgeMeasures)]
    with (out_dir / "measures.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, names)
        writer.writeheader()
        writer.writerows(rows)
    log.info("мер: %d → %s", len(rows), out_dir / "measures.csv")


def worst_point(paper: np.ndarray, book: dict, inst: np.ndarray, exclude: np.ndarray) -> tuple[int, int] | None:
    """Точка наружного края бумаги с наибольшим отклонением от сглаженного профиля стороны (центр вырезки)."""
    pages = frame_pages(paper, book["book"], book["hulls"], inst)
    best, point = -1.0, None
    two = len(pages) == 2
    for i, page in enumerate(pages):
        quad = fit_enclosing_quad(page)
        if quad is None:
            continue
        for side, prof in side_profiles(page, quad, (i == 0) if two else None, exclude).items():
            if side not in OUTER_SIDES or prof.clean.sum() < 30:
                continue
            t, d = prof.t[prof.clean], prof.d[prof.clean]
            length = float(np.linalg.norm(prof.b - prof.a))
            dev = np.abs(d - smooth_profile(t, d, length, SMOOTH_WINDOW_MM * SEG_PX_PER_MM))
            j = int(np.argmax(dev))
            if dev[j] > best:
                best = float(dev[j])
                base = prof.a + t[j] * (prof.b - prof.a)
                point = (int(base[0]), int(base[1]))
    return point


def crop_panel(seg: np.ndarray, paper: np.ndarray, v2: np.ndarray, centre: tuple[int, int], title: str) -> np.ndarray:
    """Вырезка ``CROP_MM`` вокруг точки: кадр, контур v2 (серый) и контур варианта (зелёный), увеличение, подпись."""
    half = int(CROP_MM * SEG_PX_PER_MM / 2)
    x, y = centre
    h, w = paper.shape
    x0, y0, x1, y1 = max(0, x - half), max(0, y - half), min(w, x + half), min(h, y + half)
    crop = cv2.resize(seg[y0:y1, x0:x1], None, fx=CROP_ZOOM, fy=CROP_ZOOM, interpolation=cv2.INTER_CUBIC)
    for mask, colour in ((v2, COLOUR_V2), (paper, COLOUR_VARIANT)):
        sub = cv2.resize(mask[y0:y1, x0:x1].astype(np.uint8), crop.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(crop, contours, -1, colour, 2, cv2.LINE_AA)
    return np.vstack([header_strip([title], crop.shape[1]), crop])


@main.command("sheets")
@click.option("--run", "runs", multiple=True, required=True, help=RUN_HELP)
@click.option("--set", "set_files", multiple=True, required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out-dir", type=click.Path(path_type=Path), required=True, help="Корень выхода стенда.")
@click.option(
    "--mode", type=click.Choice([m.value for m in FilterMode]), default=FilterMode.SAM.value, show_default=True
)
def sheets_cmd(runs: tuple[str, ...], set_files: tuple[Path, ...], out_dir: Path, mode: str) -> None:
    """Лист на кадр: вырезка самого рваного места края v2 у всех вариантов (кадр без фильтра и с фильтром)."""
    specs = {s.name: s for s in (parse_run(r) for r in runs)}
    for row in set_rows(set_files, specs):
        run = specs[row.run]
        book, exclude = frame_inputs(run, row.key)
        masks = variant_masks(run, row.key, out_dir)
        v2_paper, v2_inst = masks[V2_NAME]
        centre = worst_point(v2_paper, book, v2_inst, exclude)
        if centre is None:
            continue
        seg = read_bgr(run.cache_dir / row.key / "seg.png")
        panels = [crop_panel(seg, v2_paper, v2_paper, centre, "v2 (боевой прогон)")]
        for filt in EdgeFilter:
            name = f"{filt.value}/{mode}"
            if name in masks:
                shown = apply_filter(seg, filt, SEG_PX_PER_MM)
                m = edge_measures(masks[name][0], book["book"], book["hulls"], masks[name][1], exclude, SEG_PX_PER_MM)
                title = f"{filt.value}: рв {m.ragged_max_mm:.1f}, волна {m.wave_far_mm:.1f} мм"
                panels.append(crop_panel(shown, masks[name][0], v2_paper, centre, title))
        h = max(p.shape[0] for p in panels)
        panels = [np.pad(p, ((0, h - p.shape[0]), (0, 8), (0, 0)), constant_values=255) for p in panels]
        rows_img = []
        for i in range(0, len(panels), SHEET_COLS):
            chunk = panels[i : i + SHEET_COLS]
            while len(chunk) < SHEET_COLS:
                chunk.append(np.full_like(panels[0], 255))
            rows_img.append(np.hstack(chunk))
        sheet = framed(
            np.vstack(rows_img),
            [
                f"{row.run} {row.key} — край бумаги по вариантам фильтра (режим {mode}); вырезка {CROP_MM:.0f} мм, ×{CROP_ZOOM:.0f}"
            ],
            [LegendEntry("контур бумаги варианта", COLOUR_VARIANT), LegendEntry("контур бумаги v2", COLOUR_V2)],
        )
        write_image(out_dir / "sheets" / mode / f"{row.run}__{row.key}.jpg", sheet)
        log.info("лист: %s %s", row.run, row.key)
