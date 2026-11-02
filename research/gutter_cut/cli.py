"""CLI стенда «линия разреза у корешка»: мера разворотов по кэшам прогонов v2 (``measure``) и листы полос у корешка по вердиктам и поясам меры (``sheets``)."""

from __future__ import annotations

import csv
import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import header_strip
from ocr_utils.scan_cropping.image_io import collect_images
from ocr_utils.scan_cropping.v2.pipeline import FrameJob, init_worker, visible_frames
from research.gutter_cut.frame import Source, measure_frame

log = logging.getLogger(__name__)

# Воркеров по умолчанию: кадр держит ~1–2 ГБ карт и копий и читает оригинал с NTFS-3G (Я.Диск) — упор в память и диск
# раньше, чем в 16 ядер (как у стенда page_rectify).
DEFAULT_JOBS = 12
# Полос на листе и высота полосы на листе, px.
SHEET_COLUMNS = 8
SHEET_HEIGHT = 1400


@click.group()
@click.option("--log-level", default="INFO", show_default=True, help="Уровень логов.")
def main(log_level: str) -> None:
    """Стенд «линия разреза страниц у корешка»."""
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def sample_evenly(files: list[Path], n: int) -> list[Path]:
    """``n`` кадров равномерно по списку (0 — все)."""
    if n <= 0 or n >= len(files):
        return files
    return [files[i] for i in np.linspace(0, len(files) - 1, n).round().astype(int)]


def read_keys(path: Path) -> set[tuple[str, str]]:
    """Список кадров «ПОДПИСЬ КЛЮЧ  # пометка» → множество пар (пустые строки и комментарии пропускаются)."""
    out = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            label, key = line.split(None, 1)
            out.add((label, key.strip()))
    return out


@main.command()
@click.option("--set", "sets", type=(str, click.Path(path_type=Path, exists=True, file_okay=False),
              click.Path(path_type=Path, exists=True, file_okay=False),
              click.Path(path_type=Path, exists=True, file_okay=False), float, int), multiple=True, required=True,
              help="Набор: ПОДПИСЬ ВХОД КЭШ ИТОГИ_V2 PX_PER_MM N (N — сколько кадров взять равномерно, 0 — все); "
                   "повторяется, все кадры — в одном пуле.")  # fmt: skip
@click.option("--root", type=click.Path(path_type=Path), required=True,
              help="Корень выхода: measure.csv и strips/<набор>/<ключ>.jpg.")  # fmt: skip
@click.option("--only", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=None,
              help="Список кадров «ПОДПИСЬ КЛЮЧ» (после # — пометка): мерить только их; N наборов тогда не действует.")  # fmt: skip
@click.option("--strips/--no-strips", default=True, show_default=True, help="Рисовать полосы у корешка.")
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Воркеров пула.")
def measure(sets: tuple[tuple[str, Path, Path, Path, float, int], ...], root: Path, only: Path | None, strips: bool,
            jobs: int) -> None:  # fmt: skip
    """Мера «впритык к линии разреза» по кэшам прогонов v2 и предлагаемая уточнённая линия."""
    wanted = read_keys(only) if only else None
    tasks = []
    for label, input_dir, cache, v2_output, px_per_mm, n in sets:
        source = Source(label, input_dir, cache, v2_output, px_per_mm)
        files = visible_frames(collect_images(input_dir, True), input_dir)
        if wanted is None:
            files = sample_evenly(files, n)
        else:
            files = [f for f in files if (label, FrameJob(f, f.relative_to(input_dir)).key) in wanted]
        tasks += [(f, source, root / "strips" if strips else None) for f in files]
        log.info("%s: %d кадров", label, len(files))
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for i, row in enumerate(pool.map(measure_frame, tasks, chunksize=1), 1):
            rows.append(row)
            if i % 25 == 0 or i == len(tasks):
                log.info("%d/%d", i, len(tasks))
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(root / "measure.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, keys)
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        name = row.get("verdict", row["status"])
        counts[name] = counts.get(name, 0) + 1
    log.info("исходы: %s → %s", counts, root / "measure.csv")


@main.command()
@click.option("--root", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True,
              help="Корень прогона measure.")  # fmt: skip
@click.option("--per-sheet", type=int, default=SHEET_COLUMNS, show_default=True, help="Полос на листе.")
def sheets(root: Path, per_sheet: int) -> None:
    """Листы полос у корешка по вердиктам: sheets/<вердикт>/NN.jpg, полосы в ряд, подпись — набор и ключ."""
    with open(root / "measure.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("strip")]
    by_verdict: dict[str, list[dict]] = {}
    for row in rows:
        by_verdict.setdefault(row["verdict"], []).append(row)
    for verdict, group in by_verdict.items():
        # Внутри вердикта — по сдвигу и зазору: сильные случаи первыми.
        group.sort(key=lambda r: (-float(r["shift_mm"] or 0), float(r["gap_mm"] or 0) if r["gap_mm"] != "nan" else 0))
        out = root / "sheets" / verdict
        out.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(group), per_sheet):
            tiles = []
            for row in group[start : start + per_sheet]:
                strip = cv2.imread(row["strip"])
                scale = SHEET_HEIGHT / strip.shape[0]
                tile = cv2.resize(strip, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                tiles.append(np.vstack([header_strip([row["key"][-24:]], tile.shape[1]), tile]))
            height = max(t.shape[0] for t in tiles)
            tiles = [np.pad(t, ((0, height - t.shape[0]), (0, 8), (0, 0)), constant_values=255) for t in tiles]
            cv2.imwrite(
                str(out / f"{start // per_sheet + 1:02d}.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88]
            )
        log.info("%s: %d полос → %s", verdict, len(group), out)
