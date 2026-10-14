"""Типы блоков surya по кэшу прогона scan_cropping v2 (``layout.json`` версии 3 без типов): surya заново, как шаг v2, в отдельную папку, плюс счёт типов."""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import Counter
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.scan_cropping.image_io import collect_images
from ocr_utils.scan_cropping.v2.keystone import plan_warp, unwarp_polygons, warp_for_layout
from ocr_utils.scan_cropping.v2.pipeline import (
    LAYOUT_VERSION,
    FrameJob,
    WorkPaths,
    read_bgr,
    read_layout,
    read_page_quads,
    seg_shape,
)
from ocr_utils.scan_cropping.v2.scale import Scale

log = logging.getLogger(__name__)


def frame_layout(job: FrameJob, cache: Path, scale: Scale, models) -> dict:
    """Блоки surya кадра с типами — тем же путём, что ``pipeline.gpu_layout`` (выпрямление по трапециям из кэша).

    Args:
        job: Кадр.
        cache: Корень кэша прогона (читается ``pages.json``).
        scale: Масштабы прогона (размер сегментации — для пересчёта трапеций в px кадра).
        models: ``GpuModels`` с surya layout.

    Returns:
        Содержимое ``layout.json`` версии ``LAYOUT_VERSION``.
    """
    from ocr_utils.scan_cropping.finger_removal.text_protection import layout_blocks

    work = WorkPaths(cache / job.key)
    bgr = read_bgr(job.src)
    # Трапеции — в px сегментации; пересчёт в px кадра, как в шаге surya.
    ratio = bgr.shape[1] / seg_shape(bgr.shape[:2], scale)[1]
    warp = plan_warp([q * ratio for q in read_page_quads(work.pages)])
    if warp is None:
        polys, labels = layout_blocks(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), models)
    else:
        raw, labels = layout_blocks(cv2.cvtColor(warp_for_layout(bgr, warp), cv2.COLOR_BGR2RGB), models)
        polys = unwarp_polygons(raw, warp)
    return {
        "version": LAYOUT_VERSION,
        "keystone": warp is not None,
        "blocks": [np.asarray(p).tolist() for p in polys],
        "labels": labels,
    }


@click.command()
@click.option("--input-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Корень входа прогона (только чтение).")  # fmt: skip
@click.option("--cache", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Кэш прогона v2 (только чтение: pages.json, layout.json).")  # fmt: skip
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Куда писать <ключ>/layout.json с типами и labels.csv.")  # fmt: skip
@click.option("--px-per-mm", type=float, default=13.8, show_default=True, help="Плотность кадра, как в прогоне.")
def main(input_dir: Path, cache: Path, out_dir: Path, px_per_mm: float) -> None:
    """Пересчитывает surya по кадрам кэша с типами блоков; готовые кадры ``out-dir`` пропускает.

    Args:
        input_dir: Корень входа прогона.
        cache: Кэш прогона v2.
        out_dir: Папка выхода.
        px_per_mm: Плотность кадра (масштаб сегментации).
    """
    from ocr_utils.scan_cropping.gpu_models import GpuModels

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    scale = Scale(px_per_mm)
    # Кадры входа, у которых в кэше есть трапеции страниц (шаг «бумага и страницы» пройден).
    jobs = [FrameJob(f, f.relative_to(input_dir)) for f in collect_images(input_dir, True)]
    jobs = [j for j in jobs if not any(p.startswith(".") for p in j.rel.parts)]
    jobs = [j for j in jobs if WorkPaths(cache / j.key).pages.exists()]
    log.info("кадров с трапециями в кэше: %d", len(jobs))
    t0 = time.perf_counter()
    with GpuModels(with_detection=False, with_lama=False, with_layout=True) as models:
        for i, job in enumerate(jobs, 1):
            path = out_dir / job.key / "layout.json"
            if path.exists():
                continue
            data = frame_layout(job, cache, scale, models)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
            cached = read_layout(cache / job.key / "layout.json")
            if cached is not None and len(cached) != len(data["blocks"]):
                log.warning("%s: блоков в кэше %d, сейчас %d", job.key, len(cached), len(data["blocks"]))
            if i % 50 == 0:
                log.info("%d/%d, %.1f с/кадр", i, len(jobs), (time.perf_counter() - t0) / i)
    # Счёт типов и строки по блокам.
    counts: Counter[str] = Counter()
    frames: Counter[str] = Counter()
    with (out_dir / "labels.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["key", "block", "label"])
        for job in jobs:
            labels = json.loads((out_dir / job.key / "layout.json").read_text())["labels"]
            for k, label in enumerate(labels):
                writer.writerow([job.key, k, label])
            counts.update(labels)
            frames.update(set(labels))
    for label, n in counts.most_common():
        log.info("%-20s блоков %6d, кадров %5d", label, n, frames[label])


if __name__ == "__main__":
    main()
