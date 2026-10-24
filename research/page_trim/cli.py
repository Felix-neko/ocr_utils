"""CLI стенда зоны обрезки: прогон способов по страницам ``scan_cropping.v2 --output-pages`` с паспортами, сводка CSV, листы, расстояния контента до краёв для калибровки отступов."""

from __future__ import annotations

import csv
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.scan_cropping.image_io import collect_images
from ocr_utils.scan_cropping.v2.pipeline import init_worker, read_bgr
from ocr_utils.scan_cropping.v2.rectify.passport import PASSPORT_SUFFIX, read_passport
from ocr_utils.scan_markup.content_box.content import page_content
from ocr_utils.scan_markup.content_box.methods import (
    Method,
    TrimParams,
    box_of,
    junk_depths,
    measures,
    side_distances,
    strays,
)
from research.page_trim.sheets import trim_sheet

log = logging.getLogger("page_trim")

# Умолчание пула: разбор страницы ~3–5 с CPU и ~1 ГБ; страницы читаются с SSD.
DEFAULT_JOBS = 12


def trim_page(args: tuple[Path, Path, Path, Path | None, TrimParams]) -> list[dict]:
    """Все способы на одной странице: рамки, меры, расстояния до краёв, лист.

    Args:
        args: (страница, папка страниц, папка паспортов, папка листов или None, числа способов).

    Returns:
        Строки сводки (по строке на способ); ошибка — одна строка со ``status=error``.
    """
    path, pages_dir, meta_dir, sheets_dir, params = args
    rel = path.relative_to(pages_dir)
    try:
        passport = read_passport(meta_dir / rel.parent / f"{rel.stem}{PASSPORT_SUFFIX}")
        page = read_bgr(path)
        content = page_content(page, passport.dpi, passport.part)
        boxes = {m: box_of(m, content, params, passport.cover) for m in Method}
        found = {m: measures(b, content, params) for m, b in boxes.items()}
        dist = side_distances(content, params)
        junk = junk_depths(content, params)
        if sheets_dir is not None:
            header = [f"{rel} — зона обрезки: часть {passport.part}, {passport.dpi:.0f} dpi"
                      + (", обложка — без обрезки" if passport.cover else "")]  # fmt: skip
            sheet = trim_sheet(page, boxes, strays(content, params), passport.finger_zones, header, found)
            out = sheets_dir / rel.with_suffix(".jpg")
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        rows = []
        for m, r in found.items():
            row = {"page": rel.as_posix(), "method": m.value, "status": "ok", "part": passport.part,
                   "cover": int(passport.cover), "box": "/".join(map(str, boxes[m]))}  # fmt: skip
            row.update({k: v for k, v in asdict(r).items() if k != "cut_mm"})
            row.update({f"cut_{s}_mm": v for s, v in zip(("top", "bottom", "left", "right"), r.cut_mm)})
            row.update({f"content_{k}_mm": round(v, 1) for k, v in dist.items()})
            row.update({f"junk_{k}_mm": round(v, 1) for k, v in junk.items()})
            row["strays"] = len(strays(content, params))
            rows.append(row)
        return rows
    except Exception as exc:  # noqa: BLE001 — страница с ошибкой не роняет прогон
        log.exception("%s: ошибка", rel)
        return [{"page": rel.as_posix(), "status": "error", "message": repr(exc)}]


@click.group()
def main() -> None:
    """Стенд зоны обрезки выпрямленных страниц (способы A–D, меры, листы)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@main.command()
@click.option("--pages", "pages_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Папка страниц scan_cropping.v2 --output-pages.")  # fmt: skip
@click.option("--meta", "meta_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Папка паспортов --output-pages-meta.")  # fmt: skip
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Куда сводку и листы.")  # fmt: skip
@click.option("--sheets/--no-sheets", default=True, show_default=True, help="Рисовать листы.")
@click.option("--sample", default=0, show_default=True, help="Взять страниц равномерно; 0 — все.")
@click.option("--jobs", default=DEFAULT_JOBS, show_default=True, help="Процессов пула.")
@click.option("--top-mm", default=TrimParams.top_mm, show_default=True)
@click.option("--bottom-mm", default=TrimParams.bottom_mm, show_default=True)
@click.option("--outer-mm", default=TrimParams.outer_mm, show_default=True)
@click.option("--cut-mm", default=TrimParams.cut_mm, show_default=True)
@click.option("--margin-mm", default=TrimParams.margin_mm, show_default=True)
def run(pages_dir: Path, meta_dir: Path, out_dir: Path, sheets: bool, sample: int, jobs: int, top_mm: float,
        bottom_mm: float, outer_mm: float, cut_mm: float, margin_mm: float) -> None:  # fmt: skip
    """Все способы по страницам: summary.csv (строка на страницу и способ) и листы sheets/<страница>.jpg."""
    params = TrimParams(top_mm, bottom_mm, outer_mm, cut_mm, margin_mm)
    files = collect_images(pages_dir, recursive=True)
    if sample and sample < len(files):
        files = [files[i] for i in np.linspace(0, len(files) - 1, sample).round().astype(int)]
    out_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir = out_dir / "sheets" if sheets else None
    tasks = [(f, pages_dir, meta_dir, sheets_dir, params) for f in files]
    rows: list[dict] = []
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("forkserver"), initializer=init_worker) as pool:
        for done, page_rows in enumerate(pool.map(trim_page, tasks), 1):
            rows += page_rows
            if done % 20 == 0:
                log.info("%d/%d", done, len(tasks))
    names = list(dict.fromkeys(k for r in rows for k in r))
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    log.info("страниц %d, строк сводки %d → %s", len(files), len(rows), out_dir / "summary.csv")
