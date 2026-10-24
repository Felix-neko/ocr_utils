"""CLI стенда «surya по холсту Кунса»: меры кадров двух кэшей v2 «keystone | coons» (``measure``), сводка по наборам и отбор кадров с наибольшей разницей (``report``), склейки отладочных листов двух прогонов (``sheets``)."""

from __future__ import annotations

import csv
import json
import logging
import math
import statistics
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import header_strip
from ocr_utils.scan_cropping.v2.pipeline import init_worker
from research.surya_coons.measures import frame_keys, frame_measures, summary_rows, surya_times

log = logging.getLogger(__name__)

# Воркеров: кадр — несколько масок масштаба сегментации (~50 МБ), упор в CPU и чтение npz с SSD.
DEFAULT_JOBS = 16
# Листы, которые склеиваются «было | стало» (подпапки ``--debug`` v2).
SHEET_TAGS = ("surya", "main", "07_lines")
# Ширина склейки, px.
SHEET_WIDTH = 2400


def write_csv(path: Path, rows: list[dict]) -> None:
    """CSV по строкам с объединением колонок.

    Args:
        path: Файл.
        rows: Строки.
    """
    cols: list[str] = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)


def num(v) -> float:
    """Число из ячейки CSV (пусто — nan)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def measure_task(args: tuple[str, Path, Path]) -> dict | None:
    """Меры кадра в воркере (обёртка над ``frame_measures`` для пула)."""
    key, base, new = args
    try:
        return frame_measures(key, base, new)
    except Exception:  # noqa: BLE001 — кадр с ошибкой не роняет стенд
        log.exception("%s: меры не посчитались", key)
        return None


@click.group()
@click.option("--log-level", default="INFO", show_default=True, help="Уровень логов.")
def main(log_level: str) -> None:
    """Стенд «surya layout по холсту Кунса»: сравнение с холстом по трапециям."""
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@main.command()
@click.option("--base", "base", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Корень прогона «было» (keystone): в нём cache/, run.log.")  # fmt: skip
@click.option("--new", "new", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Корень прогона «стало» (coons).")  # fmt: skip
@click.option("--label", required=True, help="Имя набора (подпапка --out-dir).")
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True, help="Куда писать CSV.")
@click.option("--jobs", default=DEFAULT_JOBS, show_default=True, help="Воркеров пула.")
def measure(base: Path, new: Path, label: str, out_dir: Path, jobs: int) -> None:
    """Меры кадров двух прогонов → ``<out-dir>/<label>/frames.csv``.

    Args:
        base: Корень прогона keystone.
        new: Корень прогона coons.
        label: Имя набора.
        out_dir: Корень выхода.
        jobs: Воркеров.
    """
    keys = sorted(set(frame_keys(base / "cache")) & set(frame_keys(new / "cache")))
    log.info("%s: кадров в обоих кэшах %d", label, len(keys))
    tasks = [(k, base / "cache", new / "cache") for k in keys]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        rows = [r for r in pool.map(measure_task, tasks) if r is not None]
    # Время surya и строки сводки v2 — из логов и summary.csv прогонов.
    for prefix, root in (("base", base), ("new", new)):
        times = surya_times(root / "run.log")
        summary = summary_rows(root / "cache" / "summary.csv")
        for row in rows:
            # Меры прямизны текста страниц (проход с --output-pages) — из колонки pages_text сводки v2.
            straight = json.loads(summary.get(row["key"], {}).get("pages_text") or "{}")
            for name in ("lines", "line_med_mm", "line_p95_mm", "slope_med_deg", "side_med_mm", "side_p95_mm"):
                row[f"{prefix}_pages_{name}"] = straight.get(f"after_{name}", "")
            row[f"{prefix}_surya_s"] = times.get(row["key"], float("nan"))
            s = summary.get(row["key"], {})
            row[f"{prefix}_blocks_back"] = s.get("blocks_back", "")
            row[f"{prefix}_back_px"] = s.get("back_px", "")
            row[f"{prefix}_status"] = s.get("status", "")
    write_csv(out_dir / label / "frames.csv", rows)
    log.info("%s: записано %d строк", label, len(rows))


def diff_score(row: dict) -> float:
    """Мера разницы кадра двух прогонов для отбора на отсмотр.

    Сумма: изменение покрытия строк (×100, п. п.), непокрытых строк (/100 мм²), блоков через корешок (×2), исчезнувших
    и новых блоков, срезанного контента (/50 мм²).

    Args:
        row: Строка ``frames.csv``.

    Returns:
        Мера (больше — сильнее разница).
    """
    d_cover = abs(num(row["new_line_cover"]) - num(row["base_line_cover"])) * 100
    d_out = abs(num(row["new_lines_out_mm2"]) - num(row["base_lines_out_mm2"])) / 100
    d_spine = abs(num(row["new_spine_blocks"]) - num(row["base_spine_blocks"])) * 2
    d_cut = abs(num(row["new_cut_content_mm2"]) - num(row["base_cut_content_mm2"])) / 50
    churn = num(row["lost"]) + num(row["added"])
    return float(np.nansum([d_cover, d_out, d_spine, d_cut, churn]))


def label_summary(label: str, rows: list[dict]) -> dict:
    """Сводка набора: средние и суммы мер «было | стало», откаты по причинам, время surya.

    Args:
        label: Имя набора.
        rows: Строки ``frames.csv``.

    Returns:
        Строка сводки.
    """
    out: dict = {"label": label, "frames": len(rows)}
    for prefix in ("base", "new"):
        col = {k: [num(r[f"{prefix}_{k}"]) for r in rows] for k in
               ("line_cover", "lines_out_mm2", "text_loose", "spine_blocks", "spine_mm2", "protected_mm2",
                "cut_content_mm2", "blocks", "surya_s")}  # fmt: skip
        out[f"{prefix}_line_cover_mean"] = round(float(np.nanmean(col["line_cover"])), 4)
        out[f"{prefix}_lines_out_mm2_sum"] = round(float(np.nansum(col["lines_out_mm2"])))
        out[f"{prefix}_text_loose_mean"] = round(float(np.nanmean(col["text_loose"])), 4)
        out[f"{prefix}_spine_blocks_sum"] = int(np.nansum(col["spine_blocks"]))
        out[f"{prefix}_spine_mm2_sum"] = round(float(np.nansum(col["spine_mm2"])))
        out[f"{prefix}_protected_mm2_sum"] = round(float(np.nansum(col["protected_mm2"])))
        out[f"{prefix}_cut_content_mm2_sum"] = round(float(np.nansum(col["cut_content_mm2"])))
        out[f"{prefix}_blocks_sum"] = int(np.nansum(col["blocks"]))
        times = [t for t in col["surya_s"] if not math.isnan(t)]
        out[f"{prefix}_surya_s_median"] = round(statistics.median(times), 2) if times else ""
        for name in ("line_med_mm", "line_p95_mm", "side_med_mm", "side_p95_mm"):
            vals = [num(r.get(f"{prefix}_pages_{name}")) for r in rows]
            vals = [v for v in vals if not math.isnan(v)]
            out[f"{prefix}_pages_{name}_median"] = round(statistics.median(vals), 3) if vals else ""
    out["lost_sum"] = int(sum(num(r["lost"]) for r in rows))
    out["added_sum"] = int(sum(num(r["added"]) for r in rows))
    reasons: dict[str, int] = {}
    for r in rows:
        if r["new_warp"] != "coons":
            key = f"{r['new_warp']}:{r['new_reason'] or r['frames_reason'] or '?'}"
            reasons[key] = reasons.get(key, 0) + 1
    out["fallbacks"] = " ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
    return out


@main.command()
@click.option("--out-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Корень выхода measure (наборы — подпапки с frames.csv).")  # fmt: skip
@click.option("--top", default=20, show_default=True, help="Сколько кадров с наибольшей разницей отобрать по набору.")
def report(out_dir: Path, top: int) -> None:
    """Сводка по наборам → ``summary.csv``; отбор кадров на отсмотр → ``<набор>/select.txt`` (разница и все откаты).

    Args:
        out_dir: Корень выхода ``measure``.
        top: Кадров с наибольшей разницей на набор.
    """
    summary = []
    for path in sorted(out_dir.glob("*/frames.csv")):
        with path.open() as f:
            rows = list(csv.DictReader(f))
        label = path.parent.name
        summary.append(label_summary(label, rows))
        ranked = sorted(rows, key=diff_score, reverse=True)
        chosen = [r["key"] for r in ranked[:top]]
        fallbacks = [r["key"] for r in rows if r["new_warp"] != "coons" and r["key"] not in chosen]
        lines = [f"{k}\tразница {diff_score(next(r for r in rows if r['key'] == k)):.2f}" for k in chosen]
        lines += [f"{k}\tоткат" for k in fallbacks]
        (path.parent / "select.txt").write_text("\n".join(lines) + "\n")
    write_csv(out_dir / "summary.csv", summary)
    for row in summary:
        log.info("%s", row)


def debug_sheet(debug: Path, tag: str, key: str) -> Path | None:
    """Путь отладочного листа кадра в папке ``--debug`` v2: ключ ``a__b__IMG`` → ``<debug>/<лист>/a/b/IMG_<лист>.jpg``."""
    *parts, stem = key.split("__")
    path = debug.joinpath(tag, *parts, f"{stem}_{tag}.jpg")
    return path if path.exists() else None


def stacked(images: list[tuple[str, np.ndarray | None]], width: int) -> np.ndarray:
    """Листы друг под другом одной ширины, над каждым — шапка с подписью.

    Args:
        images: ``(подпись, картинка или None)``.
        width: Ширина, px.

    Returns:
        Склейка BGR.
    """
    parts = []
    for title, image in images:
        if image is None:
            image = np.full((200, width, 3), 255, np.uint8)
            title = f"{title}: листа нет"
        s = width / image.shape[1]
        image = cv2.resize(image, (width, max(1, int(image.shape[0] * s))), interpolation=cv2.INTER_AREA)
        parts += [header_strip([title], width), image]
    return np.vstack(parts)


@main.command()
@click.option("--base", "base", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Корень прогона keystone (debug/).")  # fmt: skip
@click.option("--new", "new", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Корень прогона coons (debug/).")  # fmt: skip
@click.option("--select", "select", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Список ключей (select.txt из report: ключ в первой колонке).")  # fmt: skip
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True, help="Папка склеек.")
def sheets(base: Path, new: Path, select: Path, out_dir: Path) -> None:
    """Склейки «keystone над coons» по листам ``surya``, ``main``, ``07_lines`` → ``<out-dir>/<кадр>_<лист>.jpg``.

    Args:
        base: Корень прогона keystone.
        new: Корень прогона coons.
        select: Список кадров.
        out_dir: Папка выхода.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = [line.split("\t")[0] for line in select.read_text().splitlines() if line.strip()]
    for key in keys:
        for tag in SHEET_TAGS:
            pair = []
            for name, root in (("было: keystone (трапеции)", base), ("стало: coons (Кунс по областям)", new)):
                path = debug_sheet(root / "debug", tag, key)
                pair.append((f"{key} — {tag} — {name}", None if path is None else cv2.imread(str(path))))
            cv2.imwrite(str(out_dir / f"{key}_{tag}.jpg"), stacked(pair, SHEET_WIDTH), [cv2.IMWRITE_JPEG_QUALITY, 85])
    log.info("склеек: %d кадров × %d листов в %s", len(keys), len(SHEET_TAGS), out_dir)
