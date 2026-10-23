"""CLI стенда «полоса-вклейка у корешка»: признаки всех способов по кэшу v2 (``features``), листы лент у корешка на отсмотр (``sheets``), сравнение способов по разметке (``evaluate``)."""

from __future__ import annotations

import csv
import logging
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM
from ocr_utils.scan_cropping.v2.pipeline import init_worker
from ocr_utils.scan_cropping.v2.spread_lines import SpineMethod
from research.gutter_strip.band import build_band, load_strip_frame
from ocr_utils.scan_cropping.v2.gutter_strip import segment_strip, spine_line
from research.gutter_strip.detect import METHODS, detect_all, detect_sam3
from research.gutter_strip.fit import fit_pages
from research.gutter_strip.overlay import draw_strip_overlay
from research.gutter_strip.labels import Label, read_labels

log = logging.getLogger(__name__)

# Воркеров по умолчанию: кадр — несколько лент и SAM 3 метки, ~0.3 ГБ ОЗУ; упор в CPU.
DEFAULT_JOBS = 12
# Лист отсмотра: кусок ленты по высоте и полуширина поперёк, мм (родной масштаб сегментации, 6.9 px/мм); лент в
# строке и строк на листе.
SHEET_TILE_MM = 90.0
SHEET_HALF_MM = 15.0
SHEET_COLS = 10
SHEET_ROWS = 2


def frame_features(args: tuple[Path, str]) -> dict | None:
    """Строка CSV кадра: балл и признаки каждого способа; None — кадр не разворот из двух страниц.

    Args:
        args: ``(кэш прогона, ключ кадра)``.

    Returns:
        Словарь или None.
    """
    cache, key = args
    try:
        sf = load_strip_frame(cache, key)
        if sf is None:
            return None
        band = build_band(sf)
        row: dict = {"key": key}
        for name, est in detect_all(sf, band).items():
            row[f"{name}_score"] = round(est.score, 4)
            row[f"{name}_left_mm"] = round(float(band.mm(est.left_u)), 2)
            row[f"{name}_right_mm"] = round(float(band.mm(est.right_u)), 2)
            for k, v in est.extra.items():
                row[f"{name}_{k}"] = round(float(v), 4)
        return row
    except Exception:  # noqa: BLE001 — кадр с ошибкой не роняет прогон
        log.exception("%s: ошибка", key)
        return None


def cache_keys(cache: Path) -> list[str]:
    """Ключи готовых кадров кэша (есть ``paper.npz`` и ``pages.json``)."""
    return sorted(p.name for p in cache.iterdir() if (p / "paper.npz").exists() and (p / "pages.json").exists())


def write_csv(path: Path, rows: list[dict]) -> None:
    """CSV по строкам с объединением колонок."""
    cols: list[str] = []
    for r in rows:
        cols += [c for c in r if c not in cols]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> dict[str, dict]:
    """CSV признаков → ключ → строка (числа — float)."""
    out = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            out[r["key"]] = {
                k: (r[k] if k == "key" else (float(r[k]) if r[k] not in ("", "nan") else float("nan"))) for k in r
            }
    return out


@click.group()
@click.option("--log-level", default="INFO", show_default=True)
def main(log_level: str) -> None:
    """Стенд «полоса-вклейка у корешка»."""
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@main.command()
@click.option("--cache", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Кэш прогона scan_cropping v2 (папки кадров).")  # fmt: skip
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True, help="CSV признаков.")
@click.option("--jobs", default=DEFAULT_JOBS, show_default=True)
@click.option("--limit", type=int, default=None, help="Только первые N кадров (отладка).")
def features(cache: Path, out: Path, jobs: int, limit: int | None) -> None:
    """Все способы обнаружения по каждому кадру кэша → CSV."""
    keys = cache_keys(cache)[:limit]
    log.info("кадров: %d, воркеров: %d", len(keys), jobs)
    rows = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for i, row in enumerate(pool.map(frame_features, [(cache, k) for k in keys], chunksize=4)):
            if row is not None:
                rows.append(row)
            if (i + 1) % 100 == 0:
                log.info("готово %d из %d", i + 1, len(keys))
    write_csv(out, rows)
    log.info("записано %d строк → %s", len(rows), out)


def band_tile(cache: Path, key: str, caption: list[str]) -> np.ndarray:
    """Лента у корешка кадра для листа: средние ``SHEET_TILE_MM`` по высоте и ±``SHEET_HALF_MM`` поперёк в родном
    масштабе сегментации (складки кромок полосы толщиной в 2–3 px не теряются), подписи сверху."""
    sf = load_strip_frame(cache, key)
    band = build_band(sf)
    h = band.bgr.shape[0]
    rows = int(round(SHEET_TILE_MM * SEG_PX_PER_MM))
    y0 = max(0, h // 2 - rows // 2)
    c = int(np.argmin(np.abs(band.us)))
    half = int(round(SHEET_HALF_MM * SEG_PX_PER_MM))
    tile = band.bgr[y0 : y0 + rows, max(c - half, 0) : c + half + 1].copy()
    # Риска предварительного корешка сверху и снизу.
    m = tile.shape[1] // 2
    cv2.line(tile, (m, 0), (m, 8), (255, 0, 140), 2)
    cv2.line(tile, (m, tile.shape[0] - 9), (m, tile.shape[0] - 1), (255, 0, 140), 2)
    head = np.full((16 * len(caption) + 4, tile.shape[1], 3), 255, np.uint8)
    for i, text in enumerate(caption):
        cv2.putText(head, text, (2, 13 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([head, tile])


@main.command()
@click.option("--cache", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--features-csv", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), required=True, help="Папка листов.")
@click.option("--top", default=40, show_default=True, help="Сколько лучших по каждому способу.")
@click.option("--random-n", default=40, show_default=True, help="Сколько случайных кадров в добавку.")
@click.option("--skip-labeled", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="Файл разметки: уже размеченные кадры не показывать.")  # fmt: skip
@click.option("--seed", default=0, show_default=True)
@click.option("--keys", "keys_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="Файл со списком ключей (по строке): показать ровно их, без отбора по способам.")  # fmt: skip
def sheets(cache: Path, features_csv: Path, out: Path, top: int, random_n: int, skip_labeled: Path | None,seed: int, keys_file: Path | None) -> None:  # fmt: skip
    """Листы лент у корешка на отсмотр: лучшие по каждому способу и случайные; номер ленты — в ``sheet_index.csv``."""
    rows = read_csv(features_csv)
    done = set(read_labels(skip_labeled)) if skip_labeled else set()
    keys: list[str] = []
    if keys_file is not None:
        keys = [k.strip() for k in keys_file.read_text().splitlines() if k.strip() and k.strip() in rows]
    for m in METHODS if keys_file is None else ():
        col = f"{m}_score"
        ranked = sorted((k for k in rows if np.isfinite(rows[k].get(col, np.nan))), key=lambda k: -rows[k][col])
        keys += [k for k in ranked[:top] if k not in keys and k not in done]
    rest = sorted(k for k in rows if k not in keys and k not in done)
    keys += random.Random(seed).sample(rest, min(random_n, len(rest)))
    out.mkdir(parents=True, exist_ok=True)
    per = SHEET_COLS * SHEET_ROWS
    index = []
    for sheet_no in range(0, len(keys), per):
        tiles = []
        for n, key in enumerate(keys[sheet_no : sheet_no + per]):
            r = rows[key]
            num = sheet_no + n + 1
            index.append({"n": num, "key": key})
            cap = [f"#{num} {key}", f"B{r['plateau_score']:.0f} F{r['creases_score']:.2f} C{r['edges_score']:.2f}"]
            tiles.append(band_tile(cache, key, cap))
        h = max(t.shape[0] for t in tiles)
        tiles = [np.vstack([t, np.full((h - t.shape[0], t.shape[1], 3), 255, np.uint8)]) for t in tiles]
        gap = np.full((h, 6, 3), 40, np.uint8)
        lines = [
            np.hstack(sum(([t, gap] for t in tiles[i : i + SHEET_COLS]), [])) for i in range(0, len(tiles), SHEET_COLS)
        ]
        w = max(x.shape[1] for x in lines)
        lines = [np.hstack([x, np.full((h, w - x.shape[1], 3), 255, np.uint8)]) for x in lines]
        cv2.imwrite(str(out / f"sheet_{sheet_no // per + 1:02d}.jpg"), np.vstack(lines), [cv2.IMWRITE_JPEG_QUALITY, 88])
    write_csv(out / "sheet_index.csv", index)
    log.info("лент: %d, листов: %d → %s", len(keys), (len(keys) + per - 1) // per, out)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Площадь под ROC: доля пар «полоса, не полоса», где у полосы балл выше (ничья — половина)."""
    if not len(pos) or not len(neg):
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0) + 0.5 * (d == 0)).mean())


@main.command()
@click.option("--features-csv", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--labels", "labels_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
def evaluate(features_csv: Path, labels_path: Path) -> None:
    """AUC и лучший порог (наибольшее F1) каждого числового признака по разметке strip / none."""
    rows = read_csv(features_csv)
    labels = {k: v for k, v in read_labels(labels_path).items() if k in rows and v is not Label.MAYBE}
    y = np.array([labels[k] is Label.STRIP for k in labels])
    log.info("размечено: %d (полоса %d, нет %d)", len(y), int(y.sum()), int((~y).sum()))
    cols = [c for c in next(iter(rows.values())) if c != "key"]
    results = []
    for c in cols:
        v = np.array([rows[k][c] for k in labels], np.float64)
        ok = np.isfinite(v)
        a = auc(v[ok & y], v[ok & ~y])
        if not np.isfinite(a):
            continue
        sign = 1 if a >= 0.5 else -1
        best = (0.0, float("nan"))
        for thr in np.unique(v[ok]):
            pred = sign * v >= sign * thr
            tp = int((pred & y & ok).sum())
            f1 = 2 * tp / max(int((pred & ok).sum()) + int((y & ok).sum()), 1)
            if f1 > best[0]:
                best = (f1, float(thr))
        results.append((max(a, 1 - a), c, sign, best))
    for a, c, sign, (f1, thr) in sorted(results, reverse=True):
        print(f"{c:28s} AUC {a:.3f} ({'больше' if sign > 0 else 'меньше'} — полоса)  F1 {f1:.3f} при {thr:.3f}")


def frame_run(args: tuple[Path, str, Path, str]) -> dict | None:
    """Сегментация полосы, линии, рамки «было | стало» и оверлей одного кадра → строка CSV.

    Args:
        args: ``(кэш прогона, ключ кадра, папка оверлеев, метка разметки или «»)``.

    Returns:
        Строка CSV или None (кадр не разворот из двух страниц, ошибка).
    """
    cache, key, out, label = args
    try:
        sf = load_strip_frame(cache, key)
        if sf is None:
            return None
        band = build_band(sf)
        sam3 = detect_sam3(band)
        strip = segment_strip(band, sam3, sf.frame.book)
        lines = sf.frame.lines[SpineMethod.AUTO]
        spine, source = spine_line(strip, sf.spine0, lines.method.value)
        fits = fit_pages(sf, strip, spine)
        title = f"{key}" + (f" [{label}]" if label else "")
        picture = draw_strip_overlay(sf.frame.img, lines, spine, source, strip, fits, title)
        sub = "strip" if strip is not None else "none"
        path = out / sub / f"{key}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return {
            "key": key,
            "label": label,
            "strip": int(strip is not None),
            "gap_mm": round(sam3.extra.get("gap_mm", float("nan")), 2),
            "width_mm": round(strip.width_mm, 2) if strip else "",
            "spine_tilt": round(spine.tilt_deg, 3),
            "spine_source": source,
            "auto_tilt": round(sf.spine0.tilt_deg, 3),
            "spike_left_mm": round(fits[0].spike_mm, 2),
            "spike_right_mm": round(fits[1].spike_mm, 2),
        }
    except Exception:  # noqa: BLE001 — кадр с ошибкой не роняет прогон
        log.exception("%s: ошибка", key)
        return None


@main.command()
@click.option("--cache", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Папка: run.csv и оверлеи strip/ и none/.")  # fmt: skip
@click.option("--labels", "labels_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None,
              help="Разметка: только её кадры, метка — в шапке оверлея; без неё — весь кэш.")  # fmt: skip
@click.option("--jobs", default=DEFAULT_JOBS, show_default=True)
def run(cache: Path, out: Path, labels_path: Path | None, jobs: int) -> None:
    """Полоса, линия корешка, вспомогательные линии, рамки страниц «было | стало» и оверлеи по кадрам."""
    labels = read_labels(labels_path) if labels_path else {}
    keys = [k for k in cache_keys(cache) if not labels or k in labels]
    jobs_args = [(cache, k, out, labels[k].value if k in labels else "") for k in keys]
    log.info("кадров: %d", len(keys))
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        rows = [r for r in pool.map(frame_run, jobs_args, chunksize=2) if r is not None]
    write_csv(out / "run.csv", rows)
    log.info("готово: %d кадров, с полосой %d → %s", len(rows), sum(r["strip"] for r in rows), out)


@main.command()
@click.option("--cache", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Кэш прогона v2: папки кадров или папки выпусков с ними (пункт множества — «выпуск/кадр»).")  # fmt: skip
@click.option("--src-root", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Корень исходных кадров (только чтение): «01__IMG_0021» → 01/IMG_0021.jpg.")  # fmt: skip
@click.option("--set", "set_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="Множество: строки «пункт # метка — заметка» (``labels.read_labels``).")  # fmt: skip
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Папка: refine.csv, json/, npz/ и band/ (лента) по кадрам, оверлеи strip/, none/, fail/.")  # fmt: skip
@click.option("--reduced-only", is_flag=True, help="Только сокращённый набор способов.")
@click.option("--limit", type=int, default=None, help="Первые N пунктов (проба).")
def refine(cache: Path, src_root: Path, set_path: Path, out: Path, reduced_only: bool, limit: int | None) -> None:
    """Уточняющая сегментация ленты у корешка в полном разрешении по пунктам множества (GPU, главный процесс)."""
    import torch

    from ocr_utils.scan_cropping.v2.bg_veto import Sam2Background
    from ocr_utils.scan_cropping.v2.paper import load_sam3
    from ocr_utils.scan_cropping.v2.pipeline import read_bgr
    from research.gutter_strip.refine import SpineRefiner, refine_spine, source_path

    labels = read_labels(set_path)
    items = list(labels)[:limit] if limit else list(labels)
    refiner = SpineRefiner(load_sam3(), Sam2Background())
    rows = []
    for n, item in enumerate(items, 1):
        frame_cache, key = (cache / item).parent, Path(item).name
        try:
            sf = load_strip_frame(frame_cache, key)
            if sf is None:
                log.info("%s: не разворот из двух страниц — пропуск", item)
                continue
            full = read_bgr(source_path(src_root, item))
            torch.cuda.reset_peak_memory_stats()
            t0 = time.monotonic()
            res, ctx = refine_spine(refiner, sf, full, reduced_only)
            seconds = time.monotonic() - t0
            vram = torch.cuda.max_memory_allocated() / 2**30
            row = save_refine(out, item, labels[item].value, res, ctx, seconds, vram, keep_band=True)
            rows.append(row)
            sub = "fail" if res.left is None else ("strip" if res.strip else "none")
            log.info("%d/%d %s [%s]: %s, %.1f с", n, len(items), item, labels[item].value,
                     sub if sub == "fail" else f"{sub} {res.width_mm:.1f} мм", seconds)  # fmt: skip
        except Exception:  # noqa: BLE001 — кадр с ошибкой не роняет прогон
            log.exception("%s: ошибка", item)
    write_csv(out / "refine.csv", rows)
    log.info("готово: %d кадров, с полосой %d → %s", len(rows), sum(r["strip"] for r in rows), out)


def save_refine(out: Path, item: str, label: str, res, ctx, seconds: float, vram_gb: float, keep_band: bool) -> dict:
    """Итоги уточнения кадра на диск: ``json/`` (сводка), ``npz/`` (края всех способов и выбранные), ``band/`` (лента,
    если ``keep_band``), оверлей в ``strip/``, ``none/`` или ``fail/``; возвращает строку ``refine.csv``.

    Args:
        out: Папка итогов.
        item: Пункт множества.
        label: Метка разметки.
        res: Итог ``refine_spine`` / ``rechoose``.
        ctx: Лента кадра.
        seconds: Время кадра.
        vram_gb: Пик видеопамяти, ГБ (0 — без сети).
        keep_band: Писать ленту полного разрешения.

    Returns:
        Строка CSV.
    """
    import json

    from research.gutter_strip.refine import summary
    from research.gutter_strip.refine_overlay import refine_overlay

    name = item.replace("/", "__")
    info = summary(res) | {"item": item, "label": label, "seconds": round(seconds, 1), "vram_gb": round(vram_gb, 2)}
    (out / "json").mkdir(parents=True, exist_ok=True)
    (out / "json" / f"{name}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    (out / "npz").mkdir(parents=True, exist_ok=True)
    arrays = {f"{m}_{s}": getattr(r, s) for m, r in res.results.items() for s in ("left", "right")}
    if res.left is not None:
        arrays |= {"chosen_left": res.left, "chosen_right": res.right}
    np.savez_compressed(out / "npz" / f"{name}.npz", ys_seg=res.band.ys_seg, half=res.band.half, ppm=res.band.ppm,
                        **arrays)  # fmt: skip
    if keep_band:
        # Лента полного разрешения — для разбора без GPU (``research.gutter_strip.refine.SpineBand``).
        (out / "band").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "band" / f"{name}.jpg"), ctx.bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    sub = "fail" if res.left is None else ("strip" if res.strip else "none")
    picture = refine_overlay(res, ctx.bgr, (ctx.coarse_left, ctx.coarse_right),
                             f"{item} [{label}] — лента у корешка, {seconds:.0f} с")  # fmt: skip
    for other in ("fail", "strip", "none"):
        (out / other / f"{name}.jpg").unlink(missing_ok=True)
    (out / sub).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / sub / f"{name}.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {"item": item, "label": label, "strip": int(res.strip), "width_mm": info["width_mm"],
            "left_by": res.left_by, "right_by": res.right_by, "full_set": int(res.full_set),
            "coarse_gap_mm": info["coarse"].get("gap_mm"),
            "tongue_left_share": info["coarse"].get("tongue_left_share"),
            "tongue_right_share": info["coarse"].get("tongue_right_share"),
            "seconds": info["seconds"], "vram_gb": info["vram_gb"]}  # fmt: skip


def frame_rechoose(args: tuple[Path, Path, str, str, Path, Path]) -> dict | None:
    """Повторная приёмка и выбор краёв кадра по ``npz`` прежнего ``refine`` (CPU), итоги — в ``out``."""
    from ocr_utils.scan_cropping.v2.pipeline import read_bgr
    from research.gutter_strip.refine import rechoose, source_path

    cache, src_root, item, label, refine_dir, out = args
    try:
        sf = load_strip_frame((cache / item).parent, Path(item).name)
        npz = refine_dir / "npz" / f"{item.replace('/', '__')}.npz"
        if sf is None or not npz.exists():
            return None
        with np.load(npz) as d:
            edges = {k: d[k] for k in d.files if k.endswith(("_left", "_right")) and not k.startswith("chosen")}
        t0 = time.monotonic()
        res, ctx = rechoose(sf, read_bgr(source_path(src_root, item)), edges)
        return save_refine(out, item, label, res, ctx, time.monotonic() - t0, 0.0, keep_band=False)
    except Exception:  # noqa: BLE001 — кадр с ошибкой не роняет прогон
        log.exception("%s: ошибка", item)
        return None


@main.command("rechoose")
@click.option("--cache", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--src-root", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--set", "set_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--refine-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Папка итогов refine: края способов берутся из её npz/.")  # fmt: skip
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option(
    "--jobs", default=6, show_default=True, help="Воркеров: исходники читаются с Я.Диска (NTFS-3G, медленно)."
)
def rechoose_cmd(cache: Path, src_root: Path, set_path: Path, refine_dir: Path, out: Path, jobs: int) -> None:
    """Повторная приёмка и выбор краёв по сохранённым краям способов (подбор порогов без GPU)."""
    labels = read_labels(set_path)
    args = [(cache, src_root, item, lab.value, refine_dir, out) for item, lab in labels.items()]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        rows = [r for r in pool.map(frame_rechoose, args) if r is not None]
    write_csv(out / "refine.csv", rows)
    log.info("готово: %d кадров, с полосой %d → %s", len(rows), sum(r["strip"] for r in rows), out)


def frame_refine_fit(args: tuple[Path, str, str, Path, Path]) -> dict | None:
    """Подгоночные области кадра «как в v2 | с уточнённой полосой» по краям команды ``refine`` и оверлей «было | стало».

    Args:
        args: ``(кэш, пункт множества, метка, папка итогов refine, папка оверлеев)``.

    Returns:
        Строка CSV или None (не разворот, нет итога refine, ошибка).
    """
    import json

    from research.gutter_strip.refine_fit import corner_steps_mm, load_edges, strip_ends, strip_from_edges, variant

    cache, item, label, refine_dir, out = args
    name = item.replace("/", "__")
    try:
        sf = load_strip_frame((cache / item).parent, Path(item).name)
        npz = refine_dir / "npz" / f"{name}.npz"
        if sf is None or not npz.exists():
            return None
        info = json.loads((refine_dir / "json" / f"{name}.json").read_text())
        band = build_band(sf)
        old = variant(sf, "v2", segment_strip(band, detect_sam3(band), sf.frame.book))
        new = variant(sf, "A", strip_from_edges(sf, load_edges(npz)) if info["strip"] else None)
        lines = sf.frame.lines[SpineMethod.AUTO]
        pics = []
        for v, tag in ((old, "как в v2 (прежняя полоса)"), (new, "A: полоса уточнения ленты")):
            width = f"полоса {v.strip.width_mm:.1f} мм" if v.strip is not None else "полосы нет"
            title = f"{item} [{label}] — {tag}: {width}"
            pics.append(draw_strip_overlay(sf.frame.img, lines, v.spine, v.spine_source, v.strip, v.fits, title))
        h = max(p.shape[0] for p in pics)
        pics = [np.pad(p, ((0, h - p.shape[0]), (0, 0), (0, 0)), constant_values=255) for p in pics]
        picture = np.hstack([pics[0], np.full((h, 16, 3), 255, np.uint8), pics[1]])
        changed = (old.strip is None) != (new.strip is None)
        sub = "changed" if changed else ("strip" if new.strip is not None else "none")
        (out / sub).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / sub / f"{name}.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
        shift = max(abs(old.corners[k] - new.corners[k]) for k in old.corners) / SEG_PX_PER_MM
        (ot, ob), (nt, nb) = corner_steps_mm(old), corner_steps_mm(new)
        ends = {k: round(v, 2) for k, v in strip_ends(sf, new.strip, new).items()} if new.strip is not None else {}
        return ends | {"item": item, "label": label, "old_strip": int(old.strip is not None),
                "old_width_mm": round(old.strip.width_mm, 2) if old.strip else "",
                "new_strip": int(new.strip is not None),
                "new_width_mm": round(new.strip.width_mm, 2) if new.strip else "",
                "corner_shift_mm": round(shift, 2), "old_step_top_mm": round(ot, 2), "old_step_bottom_mm": round(ob, 2),
                "new_step_top_mm": round(nt, 2), "new_step_bottom_mm": round(nb, 2)}  # fmt: skip
    except Exception:  # noqa: BLE001 — кадр с ошибкой не роняет прогон
        log.exception("%s: ошибка", item)
        return None


@main.command("refine-fit")
@click.option("--cache", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--set", "set_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--refine-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Папка итогов команды refine (json/, npz/).")  # fmt: skip
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Папка: fit.csv и оверлеи changed/ (решение о полосе поменялось), strip/, none/.")  # fmt: skip
@click.option("--jobs", default=DEFAULT_JOBS, show_default=True)
def refine_fit(cache: Path, set_path: Path, refine_dir: Path, out: Path, jobs: int) -> None:
    """Подгоночные области страниц «как в v2 | с полосой уточнения» по итогам ``refine`` (только CPU)."""
    labels = read_labels(set_path)
    args = [(cache, item, lab.value, refine_dir, out) for item, lab in labels.items()]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        rows = [r for r in pool.map(frame_refine_fit, args, chunksize=2) if r is not None]
    write_csv(out / "fit.csv", rows)
    log.info("готово: %d кадров, решение о полосе поменялось у %d → %s", len(rows),
             sum(r["old_strip"] != r["new_strip"] for r in rows), out)  # fmt: skip


def frame_region_fit(args: tuple[Path, str, str, Path, Path, bool]) -> dict | None:
    """Полоса-область, исправленные страницы, линия корешка и подгоночные области кадра; оверлей областей «v2 | R».

    Args:
        args: ``(кэш, пункт множества, метка, папка итогов refine/rechoose, папка выхода, прежняя подгонка у полосы)``.

    Returns:
        Строка CSV или None (не разворот, нет выбранных краёв, ошибка).
    """
    from ocr_utils.scan_cropping.v2.pipeline import BEND_CORE_MARGIN_MM
    from research.gutter_strip.fit import page_text
    from research.gutter_strip.refine import SpineBand
    from research.gutter_strip.refine_fit import corner_steps_mm, variant
    from research.gutter_strip.refine_region import (build_region, correct_pages, region_frames, region_spine,
                                                     shape_ends, snap_ends, extend_along_spine, fit_masks)  # fmt: skip
    from research.gutter_strip.refine_region_overlay import RegionView, regions_overlay

    cache, item, label, refine_dir, out, legacy = args
    name = item.replace("/", "__")
    try:
        frame_cache, key = (cache / item).parent, Path(item).name
        sf = load_strip_frame(frame_cache, key)
        npz = refine_dir / "npz" / f"{name}.npz"
        if sf is None or not npz.exists():
            return None
        with np.load(npz) as d:
            if "chosen_left" not in d.files:
                return {"item": item, "label": label, "status": "края не выбраны"}
            left, right = d["chosen_left"], d["chosen_right"]
        img = sf.frame.img
        pages = sf.frame.pages
        ratio = sf.full_shape[1] / img.shape[1]
        y_top = float(min(p.quad[:, 1].min() for p in pages))
        y_bot = float(max(p.quad[:, 1].max() for p in pages))
        band = SpineBand(sf.spine0, y_top, y_bot, ratio)
        work = frame_cache / key
        paper = np.load(work / "paper.npz")["paper"]
        # У верхних и нижних краёв страниц — объединение основного SAM 3 с SAM 3 по растянутой светлоте (решение пользователя
        # 2026-10-23 по оверлеям levels-top: на фон не выводит, выемки у корешка заделывает, 10/IMG_0007).
        from research.gutter_strip.levels_top import levels_add

        top_add = None if legacy else levels_add(work, [p.mask for p in pages], sf.frame.book, top=True)
        # И у нижних краёв (решение пользователя 2026-10-23 по оверлеям levels-top --bottom).
        bottom_add = None if legacy else levels_add(work, [p.mask for p in pages], sf.frame.book, top=False)
        for add in (top_add, bottom_add):
            if add is not None:
                paper = paper | add
        with np.load(work / ("book_veto.npz" if (work / "book_veto.npz").exists() else "book.npz")) as b:
            hulls = b["hulls"]
        region = build_region(sf, band, left, right)
        if region is not None:
            region = shape_ends(snap_ends(region, pages, paper), img, sf.frame.book)
        fixed = correct_pages(sf, band, left, right, region, paper, hulls)
        if fixed is None:
            return {"item": item, "label": label, "status": "страницы не исправились"}
        spine, source = region_spine(region, fixed.divider, fixed.pages)
        if region is not None and not legacy:
            region = extend_along_spine(region, spine)
        r = max(int(round(BEND_CORE_MARGIN_MM * SEG_PX_PER_MM)), 1)
        core_zone = (
            cv2.dilate(sf.core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0
        )
        texts = [page_text(sf, p.mask) for p in fixed.pages]
        if legacy:
            from ocr_utils.scan_cropping.v2.page_frames import STRIP_GUARD_MM

            frames = region_frames(fixed.pages, spine, region, sf.paint, core_zone, texts, STRIP_GUARD_MM, False)
        else:
            frames = region_frames(fixed.pages, spine, region, sf.paint, core_zone, texts)
        band_old = build_band(sf)
        old_strip = segment_strip(band_old, detect_sam3(band_old), sf.frame.book)
        old = variant(sf, "v2", old_strip)
        empty = np.zeros(img.shape[:2], bool)
        before = RegionView("было: страницы v2" + (f", полоса из прямых {old_strip.width_mm:.1f} мм" if old_strip
                            else ", полосы нет"), pages[0].mask, pages[1].mask,
                            old_strip.mask if old_strip is not None else empty, old.spine,
                            [f.after for f in old.fits])  # fmt: skip
        strip_mask = region.mask if region is not None else empty
        shown = fit_masks(fixed.pages, region) if not legacy else [p.mask for p in fixed.pages]
        after = RegionView(f"стало: страницы SAM 3 по уточнённым краям без полосы и её продолжения; корешок — {source}",
                           shown[0], shown[1], strip_mask, spine, frames,
                           region.extension if region is not None else None)  # fmt: skip
        corners = {"Lt": frames[0].top[-1, 1], "Lb": frames[0].bottom[-1, 1], "Rt": frames[1].top[-1, 1],
                   "Rb": frames[1].bottom[-1, 1]}  # fmt: skip
        if region is not None:
            th = region.thirds_mm()
            desc = (f"полоса-область: пролёт {region.span_mm():.0f} мм, серий {len(region.runs)}, ширина по третям "
                    f"{th[0]:.1f} / {th[1]:.1f} / {th[2]:.1f} мм; у верха {'есть' if region.touch_top else 'нет'}, у низа "
                    f"{'есть' if region.touch_bottom else 'нет'}; доведено до края листа (обтекание): "
                    f"{', '.join(region.snapped) or 'нет'}")  # fmt: skip
        else:
            desc = "полосы нет"
        shift_div = [abs(fixed.divider.x_at(y) - sf.spine0.x_at(y)) / SEG_PX_PER_MM for y in (y_top, y_bot)]
        header = [f"{item} [{label}] — области «левая страница — полоса — правая страница»", desc,
                  f"разделитель по исправленным страницам сдвинулся на {shift_div[0]:.1f} мм у верха, "
                  f"{shift_div[1]:.1f} мм у низа; бумаги перешло {fixed.moved_mm2:.0f} мм²"]  # fmt: skip
        ends = ((corners["Lt"] + corners["Rt"]) / 2, (corners["Lb"] + corners["Rb"]) / 2)
        picture = regions_overlay(img, before, after, header, ends)
        changed = (old_strip is None) != (region is None)
        sub = "changed" if changed else ("strip" if region is not None else "none")
        (out / "regions" / sub).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "regions" / sub / f"{name}.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
        shift = max(abs(old.corners[k] - corners[k]) for k in corners) / SEG_PX_PER_MM
        th = region.thirds_mm() if region is not None else (float("nan"),) * 3
        new_steps = (
            abs(corners["Lt"] - corners["Rt"]) / SEG_PX_PER_MM,
            abs(corners["Lb"] - corners["Rb"]) / SEG_PX_PER_MM,
        )
        return {"item": item, "label": label, "status": "", "old_strip": int(old_strip is not None),
                "new_strip": int(region is not None), "runs": len(region.runs) if region else 0,
                "span_mm": round(region.span_mm(), 1) if region else "",
                "width_top_mm": round(th[0], 2), "width_mid_mm": round(th[1], 2), "width_bottom_mm": round(th[2], 2),
                "touch_top": int(region.touch_top) if region else "", "touch_bottom": int(region.touch_bottom) if region else "",
                "snapped": "+".join(region.snapped) if region else "",
                "divider_shift_top_mm": round(shift_div[0], 2), "divider_shift_bottom_mm": round(shift_div[1], 2),
                "moved_mm2": round(fixed.moved_mm2, 1), "corner_shift_mm": round(shift, 2),
                "old_step_top_mm": round(corner_steps_mm(old)[0], 2), "old_step_bottom_mm": round(corner_steps_mm(old)[1], 2),
                "new_step_top_mm": round(new_steps[0], 2), "new_step_bottom_mm": round(new_steps[1], 2),
                "levels_top_mm2": round(float(top_add.sum()) / SEG_PX_PER_MM**2, 1) if top_add is not None else "",
                "levels_bottom_mm2": (round(float(bottom_add.sum()) / SEG_PX_PER_MM**2, 1)
                                      if bottom_add is not None else ""),
                **{f"corner_{k}": round(float(v), 2) for k, v in corners.items()}}  # fmt: skip
    except Exception:  # noqa: BLE001 — кадр с ошибкой не роняет прогон
        log.exception("%s: ошибка", item)
        return None


@main.command("region-fit")
@click.option("--cache", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--set", "set_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--refine-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Папка итогов refine или rechoose (npz/ с выбранными краями).")  # fmt: skip
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Папка: region.csv и оверлеи regions/{changed,strip,none}/.")  # fmt: skip
@click.option("--jobs", default=DEFAULT_JOBS, show_default=True)
@click.option("--legacy-guard", is_flag=True,
              help="Прежняя подгонка у полосы: защита 6 мм, сторона с полосой — без зоны сплайна у корешка.")  # fmt: skip
def region_fit(cache: Path, set_path: Path, refine_dir: Path, out: Path, jobs: int, legacy_guard: bool) -> None:
    """Полоса-область, исправленные страницы SAM 3, линия корешка и подгоночные области «v2 | R» (только CPU)."""
    labels = read_labels(set_path)
    args = [(cache, item, lab.value, refine_dir, out, legacy_guard) for item, lab in labels.items()]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        rows = [r for r in pool.map(frame_region_fit, args, chunksize=2) if r is not None]
    write_csv(out / "region.csv", rows)
    log.info("готово: %d кадров, с полосой %d → %s", len(rows), sum(int(r.get("new_strip", 0) or 0) for r in rows), out)


def frame_levels_top(args: tuple[Path, str, Path, bool]) -> dict | None:
    """Добавка растянутого SAM 3 у верхних краёв страниц кадра (``levels_top.frame_top_add``) → строка CSV."""
    from research.gutter_strip.levels_top import frame_top_add

    cache, item, out, top = args
    frame_cache, key = (cache / item).parent, Path(item).name
    try:
        got = frame_top_add(frame_cache, key, out, item, item.replace("/", "__"), top)
        if got is None:
            return None
        res, belt = got
        return {"item": item, "belt": belt} | {k: round(v, 1) for k, v in res.__dict__.items()}
    except Exception:  # noqa: BLE001 — кадр с ошибкой не роняет прогон
        log.exception("%s: ошибка", item)
        return None


@main.command("levels-top")
@click.option("--cache", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Кэш v2: папки кадров или папки выпусков с ними.")  # fmt: skip
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Папка: levels_top.csv и оверлеи по поясам добавки.")  # fmt: skip
@click.option("--jobs", default=DEFAULT_JOBS, show_default=True)
@click.option("--bottom", is_flag=True, help="Нижние края страниц вместо верхних.")
def levels_top(cache: Path, out: Path, jobs: int, bottom: bool) -> None:
    """Что добавил бы SAM 3 по растянутой светлоте к основному у верхних (нижних) краёв страниц (только CPU)."""
    items = [k for k in cache_keys(cache)] if any((p / "sam3.npz").exists() for p in cache.iterdir()) else [
        f"{d.name}/{k}" for d in sorted(cache.iterdir()) if d.is_dir() for k in cache_keys(d)]  # fmt: skip
    args = [(cache, item, out, not bottom) for item in items]
    log.info("кадров: %d", len(args))
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        rows = [r for r in pool.map(frame_levels_top, args, chunksize=4) if r is not None]
    write_csv(out / "levels_top.csv", rows)
    log.info("готово: %d кадров → %s", len(rows), out)


if __name__ == "__main__":
    main()
