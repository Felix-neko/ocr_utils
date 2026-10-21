"""CLI стенда «прямые разворота»: прямые по кэшу прогона v2 всеми способами разделителя — CSV, картинки шага 07 по папке на способ, полосы у корешка со всеми кандидатами (``run``); выборка на отсмотр по поясам расхождения (``select``)."""

from __future__ import annotations

import csv
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, framed
from ocr_utils.scan_cropping.v2.overlay import draw_lines_step
from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM
from ocr_utils.scan_cropping.v2.pipeline import init_worker
from ocr_utils.scan_cropping.v2.spread_lines import EdgeKind, SpineMethod
from research.spread_lines.frames import FrameLines, load_frame

log = logging.getLogger(__name__)

# Цвета разделителя по способу на полосе у корешка (BGR, палитра ``/draw-overlay``).
METHOD_COLOUR = {
    SpineMethod.QUAD_MID: (0, 150, 0),
    SpineMethod.GAP_MID: (220, 90, 20),
    SpineMethod.GAP_MID_RUN: (0, 165, 255),
    SpineMethod.FOLD: (200, 60, 200),
}
# Полуширина полосы у корешка, мм бумаги; высота картинки полосы, px.
BAND_MM = 45.0
BAND_HEIGHT = 1400
# Пояса расхождения разделителей (наибольшее по способам, мм) для выборки на отсмотр.
SPREAD_BANDS_MM = (1.0, 3.0, 6.0)


def frame_rel(key: str) -> Path:
    """Путь кадра для картинок: ключ кэша ``a__b__IMG`` → ``a/b/IMG``."""
    return Path(*key.split("__"))


def spine_band(frame: FrameLines) -> np.ndarray:
    """Полоса ``±BAND_MM`` у корешка во всю высоту кадра с разделителями всех способов.

    Args:
        frame: Кадр с прямыми.

    Returns:
        Картинка BGR высотой ``BAND_HEIGHT`` с шапкой и легендой.
    """
    canvas = frame.img.copy()
    for method, lines in frame.lines.items():
        if method is SpineMethod.AUTO:
            continue  # совпадает с одним из способов
        a, b = np.round(lines.spine.ends()).astype(int)
        cv2.line(canvas, tuple(a), tuple(b), METHOD_COLOUR[method], 2, cv2.LINE_AA)
    base = frame.lines[SpineMethod.QUAD_MID].spine
    cx = int(base.x_at((base.y_top + base.y_bot) / 2))
    half = int(BAND_MM * SEG_PX_PER_MM)
    band = canvas[:, max(cx - half, 0) : cx + half]
    s = BAND_HEIGHT / band.shape[0]
    band = cv2.resize(band, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    legend = [
        LegendEntry(f"разделитель {m.value} {frame.lines[m].spine.tilt_deg:+.2f}°", METHOD_COLOUR[m])
        for m in frame.lines
        if m is not SpineMethod.AUTO
    ]
    return framed(band, [frame.key], legend)


def frame_row(frame: FrameLines) -> dict:
    """Строка CSV кадра: видимость краёв книги, наклоны, разделители по способам и их расхождение, мм."""
    any_lines = next(iter(frame.lines.values()))
    row = {"key": frame.key}
    for name, edge in (("left", any_lines.book_left), ("right", any_lines.book_right)):
        row[f"book_{name}_visible"] = int(edge.visible)
        row[f"book_{name}_border_share"] = round(edge.border_share, 3)
        row[f"book_{name}_tilt"] = None if edge.line is None else round(edge.line.tilt_deg, 3)
    tops, bots = [], []
    for method in SpineMethod:
        lines = frame.lines.get(method)
        if lines is None:
            row[f"{method.value}_tilt"] = row[f"{method.value}_top_mm"] = row[f"{method.value}_bot_mm"] = None
            continue
        sp = lines.spine
        top, bot = float(sp.x_at(sp.y_top)) / SEG_PX_PER_MM, float(sp.x_at(sp.y_bot)) / SEG_PX_PER_MM
        tops.append(top)
        bots.append(bot)
        row[f"{method.value}_tilt"] = round(sp.tilt_deg, 3)
        row[f"{method.value}_top_mm"], row[f"{method.value}_bot_mm"] = round(top, 2), round(bot, 2)
        row[f"{method.value}_left_kind"] = lines.left_kind.value
        row[f"{method.value}_right_kind"] = lines.right_kind.value
        row[f"{method.value}_left_tilt"] = None if lines.left is None else round(lines.left.tilt_deg, 3)
        row[f"{method.value}_right_tilt"] = None if lines.right is None else round(lines.right.tilt_deg, 3)
    row["spread_mm"] = round(max(max(tops) - min(tops), max(bots) - min(bots)), 2)
    row["virtual"] = int(any(k is EdgeKind.VIRTUAL for k in (any_lines.left_kind, any_lines.right_kind)))
    return row


def run_task(args: tuple[Path, str, Path]) -> dict | None:
    """Один кадр (для пула): прямые, картинки шага 07 по способам, полоса у корешка; строка CSV или None."""
    cache, key, out = args
    try:
        frame = load_frame(cache, key)
    except Exception:  # кадр с испорченным кэшем не роняет прогон
        log.exception("%s: ошибка", key)
        return None
    if frame is None:
        return None
    rel = frame_rel(key)
    for method, lines in frame.lines.items():
        path = out / method.value / rel.parent / f"{rel.name}_07_lines.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        picture = draw_lines_step(frame.img, frame.pages, lines, frame.book, f"{rel} [{method.value}]")
        cv2.imwrite(str(path), picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
    path = out / "compare" / rel.parent / f"{rel.name}_spine.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), spine_band(frame), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return frame_row(frame)


@click.group()
def main() -> None:
    """Стенд «прямые разворота» по кэшу прогона scan_cropping v2 (без GPU)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@main.command("run")
@click.option("--cache", "cache", type=click.Path(file_okay=False, exists=True, path_type=Path), required=True,
              help="Корень кэша прогона v2 (--cache), кадр — подпапка с pages.json и paper.npz.")  # fmt: skip
@click.option("--out", "out", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Папка выхода: lines.csv, <способ>/…_07_lines.jpg, compare/…_spine.jpg.")  # fmt: skip
@click.option("--jobs", type=int, default=16, show_default=True, help="Воркеров пула (CPU).")
@click.option("--limit", type=int, default=0, help="Первых N кадров (0 — все).")
def run_cmd(cache: Path, out: Path, jobs: int, limit: int) -> None:
    """Прямые разворота по всем кадрам кэша всеми способами разделителя."""
    keys = sorted(p.name for p in cache.iterdir() if (p / "pages.json").exists() and (p / "paper.npz").exists())
    keys = keys[:limit] if limit else keys
    out.mkdir(parents=True, exist_ok=True)
    log.info("кадров: %d", len(keys))
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        rows = [r for r in pool.map(run_task, [(cache, k, out) for k in keys], chunksize=4) if r is not None]
    fields = list(dict.fromkeys(f for r in rows for f in r))
    with open(out / "lines.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    spread = np.array([r["spread_mm"] for r in rows])
    log.info("разворотов: %d; виртуальных ограничителей: %d; расхождение разделителей, мм: медиана %.2f, p90 %.2f, "
             "max %.2f", len(rows), sum(r["virtual"] for r in rows), np.median(spread), np.quantile(spread, 0.9),
             spread.max())  # fmt: skip


@main.command("select")
@click.option("--out", "out", type=click.Path(file_okay=False, exists=True, path_type=Path), required=True,
              help="Папка выхода run (с lines.csv).")  # fmt: skip
@click.option("--per-band", type=int, default=12, show_default=True, help="Кадров на пояс.")
def select_cmd(out: Path, per_band: int) -> None:
    """Симлинки на полосы у корешка по поясам расхождения разделителей и на кадры с виртуальным ограничителем."""
    rows = list(csv.DictReader(open(out / "lines.csv")))
    rng = np.random.default_rng(0)
    edges = (0.0, *SPREAD_BANDS_MM, np.inf)
    groups = {
        f"spread_{a:g}-{b:g}mm": [r for r in rows if a <= float(r["spread_mm"]) < b] for a, b in zip(edges, edges[1:])
    }
    groups["virtual"] = [r for r in rows if r["virtual"] == "1"]
    for name, group in groups.items():
        pick = (
            [group[i] for i in sorted(rng.choice(len(group), min(per_band, len(group)), replace=False))]
            if group
            else []
        )
        folder = out / "select" / name
        folder.mkdir(parents=True, exist_ok=True)
        for r in pick:
            rel = frame_rel(r["key"])
            src = out / ("compare" if name != "virtual" else SpineMethod.AUTO.value) / rel.parent
            src = src / (f"{rel.name}_spine.jpg" if name != "virtual" else f"{rel.name}_07_lines.jpg")
            link = folder / f"{r['key']}{src.suffix}"
            if not link.exists():
                os.symlink(src, link)
        log.info("%s: %d из %d", name, len(pick), len(group))


@main.command("evaluate")
@click.option("--out", "out", type=click.Path(file_okay=False, exists=True, path_type=Path), required=True,
              help="Папка выхода run (с lines.csv).")  # fmt: skip
def evaluate_cmd(out: Path) -> None:
    """Правила виртуального наклона на кадрах, где оба края книги в кадре: край книги с одной стороны предсказывается
    по другой и разделителю — зеркально (2·разделитель − противоположный), как противоположный, как разделитель;
    ошибка — против настоящего наклона края книги, градусы (медиана и p90 модуля)."""
    rows = [
        r
        for r in csv.DictReader(open(out / "lines.csv"))
        if r["book_left_visible"] == "1" and r["book_right_visible"] == "1"
    ]
    log.info("кадров с обоими краями книги в кадре: %d", len(rows))
    for method in SpineMethod:
        errors: dict[str, list[float]] = {"зеркально": [], "как противоположный": [], "как разделитель": []}
        for r in rows:
            if not r[f"{method.value}_tilt"]:
                continue
            spine, left, right = (
                float(r[f"{method.value}_tilt"]),
                float(r["book_left_tilt"]),
                float(r["book_right_tilt"]),
            )
            for truth, other in ((left, right), (right, left)):
                errors["зеркально"].append(abs(2 * spine - other - truth))
                errors["как противоположный"].append(abs(other - truth))
                errors["как разделитель"].append(abs(spine - truth))
        summary = ", ".join(f"{name}: {np.median(e):.2f}° / {np.quantile(e, 0.9):.2f}°" for name, e in errors.items())
        log.info("разделитель %s — медиана / p90 ошибки: %s", method.value, summary)
