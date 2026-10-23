"""CLI стенда «рваный внешний край»: меры по кэшам v2 (``features``), листы по поясам меры на отсмотр и разметку (``sheets``), точность и полнота порогов по разметке (``evaluate``), копия кэша (``copy-cache``), сравнение двух прогонов «было | стало» (``compare``)."""

from __future__ import annotations

import csv
import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import click
import numpy as np

from ocr_utils.scan_cropping.v2.pipeline import init_worker
from ocr_utils.page_layout.overlay_frame import LegendEntry
from ocr_utils.scan_cropping.v2.page_sides import Side
from research.ragged_edges.compare import NEW_COLOUR, OLD_COLOUR, frame_compare, pair_tile
from research.ragged_edges.features import frame_rows
from research.ragged_edges.labels import Label, read_labels
from research.ragged_edges.sheets import side_tile, write_sheets

log = logging.getLogger(__name__)

# Воркеров: кадр — маски масштаба сегментации и подгонка сплайна, ~0.3 ГБ ОЗУ; упор в CPU.
DEFAULT_JOBS = 12
# Пояса меры для листов (извилистость сырой маски SAM 3).
DEFAULT_BELTS = "1.0,1.02,1.03,1.05,1.07,1.10,1.20,9"


def cache_dirs(path: Path) -> list[Path]:
    """Кэши под путём: сам путь, если в нём кадры с ``paper.npz``, иначе его подпапки-кэши (выпуски ХиП)."""
    subs = [p for p in sorted(path.iterdir()) if p.is_dir()]
    if any((p / "paper.npz").exists() for p in subs[:50]):
        return [path]
    return [p for p in subs if any((q / "paper.npz").exists() for q in list(p.iterdir())[:50])]


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


def num(v: str) -> float:
    """Число из ячейки CSV (пусто — nan)."""
    return float(v) if v not in ("", "nan", None) else float("nan")


@click.group()
@click.option("--log-level", default="INFO", show_default=True)
def main(log_level: str) -> None:
    """Стенд «рваный внешний край страницы»."""
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@main.command()
@click.option("--run", "runs", multiple=True, required=True,
              help="Прогон: ИМЯ=ПУТЬ — кэш v2 или папка кэшей по выпускам (ХиП).")  # fmt: skip
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True, help="CSV мер.")
@click.option("--jobs", default=DEFAULT_JOBS, show_default=True)
def features(runs: tuple[str, ...], out: Path, jobs: int) -> None:
    """Меры дальних сторон по шагам (сырой SAM 3, plain, итог) по всем кадрам прогонов → CSV."""
    jobs_list = []
    for spec in runs:
        name, path = spec.split("=", 1)
        for cache in cache_dirs(Path(path)):
            jobs_list += [(name, cache, p.name) for p in sorted(cache.iterdir()) if (p / "paper.npz").exists()]
    log.info("кадров: %d, воркеров: %d", len(jobs_list), jobs)
    rows = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for i, rs in enumerate(pool.map(frame_rows, jobs_list, chunksize=4)):
            rows += rs
            if (i + 1) % 200 == 0:
                log.info("готово %d из %d", i + 1, len(jobs_list))
    write_csv(out, rows)
    log.info("сторон %d, ошибок %d → %s", len(rows), sum("error" in r for r in rows), out)


@main.command()
@click.option("--csv", "csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), required=True, help="Папка листов.")
@click.option("--measure", default="raw_tortuosity", show_default=True, help="Колонка меры для поясов.")
@click.option("--belts", default=DEFAULT_BELTS, show_default=True, help="Границы поясов через запятую.")
@click.option("--per-belt", default=16, show_default=True, help="Сторон на пояс (случайно, зерно --seed).")
@click.option("--run", "run_filter", default=None, help="Только этот прогон.")
@click.option("--seed", default=0, show_default=True)
@click.option("--first-number", default=1, show_default=True,
              help="Номер первой плитки (у листов разных прогонов номера не должны совпадать).")  # fmt: skip
def sheets(
    csv_path: Path,
    out: Path,
    measure: str,
    belts: str,
    per_belt: int,
    run_filter: str | None,
    seed: int,
    first_number: int,
) -> None:
    """Листы сторон по поясам меры с номерами плиток и ``index.csv`` (номер → сторона) для разметки."""
    rows = [r for r in csv.DictReader(csv_path.open()) if not r.get("error") and np.isfinite(num(r.get(measure, "")))]
    if run_filter:
        rows = [r for r in rows if r["set"] == run_filter]
    edges = [float(x) for x in belts.split(",")]
    rng = np.random.default_rng(seed)
    index, number = [], first_number
    for lo, hi in zip(edges[:-1], edges[1:]):
        belt = [r for r in rows if lo <= num(r[measure]) < hi]
        pick = [belt[i] for i in sorted(rng.choice(len(belt), min(per_belt, len(belt)), replace=False))] if belt else []
        tiles = []
        for r in pick:
            caption = f"{r['set']} {r['key']} {r['page']} {measure.split('_')[0]} {num(r[measure]):.3f}"
            try:
                tiles.append(side_tile(Path(r["cache"]) / r["key"], r["page"], number, caption))
            except Exception:  # noqa: BLE001
                log.exception("%s %s: плитка не построена", r["key"], r["page"])
                continue
            index.append({"number": number, "belt": f"{lo}-{hi}", "set": r["set"], "cache": r["cache"],
                          "key": r["key"], "page": r["page"], measure: r[measure]})  # fmt: skip
            number += 1
        if tiles:
            title = f"{measure} {lo}–{hi}: {len(belt)} сторон в поясе, показано {len(tiles)}; номера — для разметки"
            write_sheets(tiles, out, f"belt_{lo:.2f}-{hi:.2f}", title)
        log.info("пояс %s–%s: сторон %d, на листах %d", lo, hi, len(belt), len(tiles))
    write_csv(out / "index.csv", index)


@main.command()
@click.option("--csv", "csv_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--labels", type=click.Path(exists=True, dir_okay=False, path_type=Path), multiple=True, required=True)
@click.option("--measure", default="raw_tortuosity", show_default=True)
def evaluate(csv_path: Path, labels: tuple[Path, ...], measure: str) -> None:
    """Значения меры на размеченных сторонах и точность / полнота детектора по сетке порогов."""
    lab: dict = {}
    for path in labels:
        lab.update(read_labels(path))
    values = {(r["set"], r["key"], r["page"]): num(r.get(measure, "")) for r in csv.DictReader(csv_path.open())}
    pairs = [(lab[k], values[k]) for k in lab if k in values and np.isfinite(values[k]) and lab[k] is not Label.UNSURE]
    missing = [k for k in lab if k not in values]
    if missing:
        log.warning("нет в CSV: %s", missing)
    for label in (Label.RAGGED, Label.SMOOTH):
        v = np.array([x for l, x in pairs if l is label])
        if len(v):
            print(
                f"{label.value}: {len(v)} сторон, {measure} мин {v.min():.3f} медиана {np.median(v):.3f} макс {v.max():.3f}"
            )
    for thr in (1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.10):
        tp = sum(l is Label.RAGGED and x >= thr for l, x in pairs)
        fp = sum(l is Label.SMOOTH and x >= thr for l, x in pairs)
        fn = sum(l is Label.RAGGED and x < thr for l, x in pairs)
        print(f"порог {thr:.2f}: найдено рваных {tp}, ложных {fp}, пропущено {fn}")
    all_v = np.array([x for x in values.values() if np.isfinite(x)])
    for thr in (1.03, 1.05, 1.07, 1.10):
        print(f"доля сторон всего CSV с {measure} ≥ {thr:.2f}: {(all_v >= thr).mean():.3f}")


# Файлы кэша кадра, которые новый прогон считает заново (SAM 3 с растянутым проходом, бумага, страницы, разметка
# surya, итоги двойной сегментации и уточнения); остальное (пальцы, книга, фон SAM 2) переносится как есть.
RECOMPUTE = ("sam3.npz", "paper.npz", "pages.json", "layout.json", "dual.json", "ragged.json", "refine_in.npz",
             "refine_out.npz", "refine.json", "clean.png")  # fmt: skip


@main.command("copy-cache")
@click.option("--src", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Кэш v2 (папки кадров) или папка кэшей по выпускам.")  # fmt: skip
@click.option("--dst", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--keys", default=None, help="Только эти ключи кадров (через запятую).")
@click.option("--input-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None,
              help="Только кадры этой папки входа (ключи — как у v2 --recursive: путь с «__» вместо «/»).")  # fmt: skip
@click.option("--keep", multiple=True,
              help="Файл кадра из списка пересчитываемых, который всё же копировать (sam3.npz — когда SAM 3 и двойная "
                   "сегментация уже посчитаны прежним прогоном с теми же ключами).")  # fmt: skip
def copy_cache(src: Path, dst: Path, keys: str | None, input_dir: Path | None, keep: tuple[str, ...]) -> None:
    """Копия кэша v2 без шагов, которые новый прогон считает заново (база для сравнения остаётся нетронутой)."""
    import shutil

    recompute = set(RECOMPUTE) - set(keep)
    wanted = set(keys.split(",")) if keys else None
    if input_dir is not None:
        files = [p for p in input_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")]
        wanted = (wanted or set()) | {"__".join(p.relative_to(input_dir).with_suffix("").parts) for p in files}
    n = 0
    for cache in cache_dirs(src):
        rel = cache.relative_to(src)
        for frame in sorted(p for p in cache.iterdir() if p.is_dir()):
            if wanted is not None and frame.name not in wanted:
                continue
            out = dst / rel / frame.name
            out.mkdir(parents=True, exist_ok=True)
            for f in frame.iterdir():
                if f.is_file() and f.name not in recompute and not (out / f.name).exists():
                    shutil.copy2(f, out / f.name)
            n += 1
    log.info("кадров скопировано: %d → %s", n, dst)


@main.command()
@click.option("--old", "old_root", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Кэш прежнего прогона (папки кадров или папка кэшей по выпускам).")  # fmt: skip
@click.option("--new", "new_root", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Кэш нового прогона того же набора.")  # fmt: skip
@click.option("--name", default="", help="Имя набора в CSV.")
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), required=True, help="Папка CSV и листов.")
@click.option(
    "--min-mm2", default=20.0, show_default=True, help="Сторона на лист, если расхождение у неё не меньше, мм²."
)
@click.option("--top", default=96, show_default=True, help="Сторон на листы каждой зоны (по убыванию расхождения).")
@click.option("--jobs", default=DEFAULT_JOBS, show_default=True)
def compare(old_root: Path, new_root: Path, name: str, out: Path, min_mm2: float, top: int, jobs: int) -> None:
    """Страницы двух прогонов: расхождение масок у дальней стороны и у корешка → CSV, листы «было | стало»."""
    jobs_list = []
    for cache in cache_dirs(new_root):
        old_cache = old_root / cache.relative_to(new_root)
        jobs_list += [(name, old_cache / p.name, p) for p in sorted(cache.iterdir())
                      if (p / "paper.npz").exists() and (old_cache / p.name / "paper.npz").exists()]  # fmt: skip
    log.info("кадров в обоих прогонах: %d", len(jobs_list))
    rows = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for rs in pool.map(frame_compare, jobs_list, chunksize=4):
            rows += rs
    write_csv(out / "compare.csv", rows)
    legend = [LegendEntry("страница прежнего прогона", OLD_COLOUR), LegendEntry("страница нового прогона", NEW_COLOUR)]
    for zone, title in ((Side.FAR, "дальняя сторона"), (Side.SPINE, "у корешка")):
        col = f"{zone.value}_mm2"
        moved = sorted((r for r in rows if r[col] >= min_mm2), key=lambda r: -r[col])
        log.info("%s: сторон с расхождением ≥ %.0f мм² — %d из %d", title, min_mm2, len(moved), len(rows))
        tiles = []
        for r in moved[:top]:
            # Подпись латиницей: cv2.putText не знает кириллицы и «²».
            caption = (f"{Path(r['cache']).name}/{r['key']} {r['page']}: {r[col]:.0f} mm2\n"
                       f"{r['old_chosen'] or '-'} -> {r['new_chosen'] or '-'}")  # fmt: skip
            new_work = Path(r["cache"]) / r["key"]
            old_work = old_root / new_work.parent.relative_to(new_root) / r["key"]
            try:
                tiles.append(pair_tile(old_work, new_work, r["page"], zone, caption))
            except Exception:  # noqa: BLE001
                log.exception("%s %s: плитка не построена", r["key"], r["page"])
        if tiles:
            write_sheets(tiles, out / zone.value, f"{zone.value}",
                         f"{name}: {title}, расхождение страниц прогонов ≥ {min_mm2:.0f} мм² — {len(moved)} сторон, "
                         f"показано {len(tiles)}", legend)  # fmt: skip
