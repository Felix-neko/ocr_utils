"""CLI стенда «границы фото»: снимок набора из базы (``select``), кэш полос (``cache``), прогон способов и меры (``run``), листы (``sheets``), опоры выпрямления на кадрах стенда page_rectify (``rectify``)."""

from __future__ import annotations

import csv
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np

from research.photo_bounds.cache import CacheJob, cache_page, init_worker
from research.photo_bounds.report import sharpness_rows, summary_rows, tightness_summary
from research.photo_bounds.run import RunJob, run_page
from research.photo_bounds.sets import attach_sets
from research.photo_bounds.measure import CUT_COVER, SPARE_COVER
from research.photo_bounds.sheets import CardSpec as CardSpecT
from research.photo_bounds.sheets import (
    SheetKind,
    disagreement_groups,
    specs_for,
    tight_flags,
    verify_sheet,
    write_sheets,
)
from research.photo_bounds.data import CacheLayout, load_pack_root, load_pages, open_readonly, read_pages, save_pages

log = logging.getLogger(__name__)

# Воркеров по умолчанию: CPU-задачи стенда (разбор строк, профили краёв) — 12 из 16 ядер; кэш упирается в чтение
# TIFF с NTFS-3G шпинделя, для него в run-скрипте задаётся меньше.
DEFAULT_JOBS = 12


def write_csv(path: Path, rows: list[dict]) -> None:
    """Пишет строки в CSV с объединением ключей всех строк (порядок — первого появления).

    Args:
        path: Куда писать.
        rows: Строки-словари.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@click.group()
def main() -> None:
    """Стенд границ растровых фото на полосах журналов."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@main.command()
@click.option("--db", "db_path", type=click.Path(exists=True, path_type=Path), required=True,
              help="База разметки (открывается только на чтение).")  # fmt: skip
@click.option("--root", type=click.Path(path_type=Path), required=True, help="Корень выхода стенда (SSD).")
def select(db_path: Path, root: Path) -> None:
    """Снимок набора: полосы с эталонными фото из базы (без full_page) → ``pages.json``."""
    conn = open_readonly(db_path)
    pages = load_pages(conn)
    layout = CacheLayout(root)
    save_pages(layout.pages_json(), pages)
    (root / "pack_root.txt").write_text(str(load_pack_root(conn)))
    photos = sum(len(p.photos) for p in pages)
    click.echo(f"полос {len(pages)}, фото {photos} → {layout.pages_json()}")


@main.command()
@click.option("--root", type=click.Path(path_type=Path), required=True, help="Корень выхода стенда (SSD).")
@click.option("--surya-cache", type=click.Path(exists=True, path_type=Path), required=True,
              help="Корень кэша surya layout (LAYOUT_CACHE_DIR), только чтение.")  # fmt: skip
@click.option("--jobs", type=int, default=6, show_default=True,
              help="Воркеров: упор в чтение TIFF с NTFS-3G, больше 6 не ускоряет.")  # fmt: skip
@click.option("--limit", type=int, default=0, help="Только первые N полос (проба).")
def cache(root: Path, surya_cache: Path, jobs: int, limit: int) -> None:
    """Кэш полос: серый 300 dpi, цвет 150 dpi, растр детектора page_layout (TIFF читается один раз)."""
    layout = CacheLayout(root)
    pages = read_pages(layout.pages_json())
    if limit:
        pages = pages[:limit]
    pack_root = Path((root / "pack_root.txt").read_text().strip())
    tasks = [CacheJob(p, pack_root, layout, surya_cache) for p in pages]
    rows = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for i, row in enumerate(pool.map(cache_page, tasks), 1):
            rows.append(row)
            if i % 20 == 0 or row["status"] == "error":
                log.info("%d/%d %s", i, len(tasks), row)
    write_csv(root / "cache.csv", rows)
    bad = [r for r in rows if r["status"] == "error"]
    click.echo(f"полос {len(rows)}, ошибок {len(bad)}")


@main.command()
@click.option("--root", type=click.Path(path_type=Path), required=True, help="Корень выхода стенда (SSD).")
@click.option("--out", type=click.Path(path_type=Path), required=True, help="Папка прогона (JSON, CSV).")
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Воркеров.")
@click.option("--limit", type=int, default=0, help="Только первые N полос (проба).")
def run(root: Path, out: Path, jobs: int, limit: int) -> None:
    """Фото по затравкам детектора, меры против эталона: ``measures.csv``, ``summary.csv``, ``sharpness.csv``."""
    layout = CacheLayout(root)
    pages = read_pages(layout.pages_json())
    if limit:
        pages = pages[:limit]
    tasks = [RunJob(p, layout, out / "pages") for p in pages]
    rows: list[dict] = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for i, part in enumerate(pool.map(run_page, tasks), 1):
            rows += part
            if i % 50 == 0:
                log.info("%d/%d", i, len(tasks))
    rows = attach_sets(rows)
    write_csv(out / "measures.csv", rows)
    write_csv(out / "summary.csv", summary_rows(rows))
    photos = [r for r in rows if not r.get("extra")]
    write_csv(out / "sharpness.csv", sharpness_rows(photos))
    tight = [tightness_summary(photos, m) for m in ("ref", "det", "new")]
    write_csv(out / "tightness.csv", tight)
    for t in tight:
        click.echo(json.dumps(t, ensure_ascii=False))
    for r in summary_rows(rows):
        if r["group"] in ("все",):
            click.echo(json.dumps(r, ensure_ascii=False))


@main.command()
@click.option("--root", type=click.Path(path_type=Path), required=True, help="Корень выхода стенда (SSD).")
@click.option("--run-dir", type=click.Path(exists=True, path_type=Path), required=True, help="Папка прогона.")
@click.option(
    "--kind",
    type=click.Choice([k.value for k in SheetKind]),
    multiple=True,
    required=True,
    help="Какие листы: all — все фото подряд, disagree — по поясам расхождения, class — по классам края.",
)
@click.option("--per-sheet", type=int, default=6, show_default=True, help="Карточек на лист.")
@click.option("--cols", type=int, default=2, show_default=True, help="Карточек в ряду.")
@click.option("--max-width", type=int, default=1800, show_default=True, help="Ширина листа, px.")
@click.option("--zoom/--no-zoom", default=True, show_default=True, help="Увеличенные углы на карточках.")
def sheets(
    root: Path, run_dir: Path, kind: tuple[str, ...], per_sheet: int, cols: int, max_width: int, zoom: bool
) -> None:
    """Листы карточек фото: ``<run-dir>/sheets/<вид>/...jpg``."""
    layout = CacheLayout(root)
    pages = {p.page_key: p for p in read_pages(layout.pages_json())}
    with (run_dir / "measures.csv").open() as fh:
        rows = [r for r in csv.DictReader(fh) if not r.get("extra")]
    rows = [r for r in rows if r.get("new_iou") not in (None, "")]
    for k in map(SheetKind, kind):
        base = run_dir / "sheets" / k.value
        if k is SheetKind.ALL:
            groups = {"all": rows}
        elif k is SheetKind.DISAGREE:
            groups = disagreement_groups(rows)
        elif k is SheetKind.NEW_CUT:
            groups = {"new_cut": [r for r in rows if tight_flags(r, "new", False, CUT_COVER)]}
        elif k is SheetKind.NEW_SPARE:
            groups = {"new_spare": [r for r in rows if tight_flags(r, "new", True, SPARE_COVER)]}
        elif k is SheetKind.OBJECT:
            part = [r for r in rows if max(float(r.get(f"{m}_obj_out_mm2") or 0) for m in ("ref", "det", "new")) > 3]
            part.sort(key=lambda r: -float(r.get("new_obj_out_mm2") or 0))
            from research.photo_bounds.overlay import sheet as make_sheet
            from research.photo_bounds.sheets import object_card, object_title

            specs = specs_for(part, pages)
            base.mkdir(parents=True, exist_ok=True)
            for n, start in enumerate(range(0, len(specs), per_sheet), 1):
                cards = [object_card(layout, run_dir, CardSpecT(sp.page, sp.ref_index, object_title(r)))
                         for sp, r in zip(specs[start : start + per_sheet], part[start : start + per_sheet])]  # fmt: skip
                picture = make_sheet(cards, [f"объект за рамкой — лист {n}, фото {start + 1}–{start + len(cards)}; "
                                             "красное — объект за новой рамкой, голубое — объект внутри"], cols)  # fmt: skip
                if picture.shape[1] > max_width:
                    f = max_width / picture.shape[1]
                    picture = cv2.resize(picture, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(base / f"object_{n:02d}.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
            click.echo(f"object: фото {len(part)}")
            continue
        elif k is SheetKind.REF_SPARE:
            part = [
                r
                for r in rows
                if tight_flags(r, "ref", True, SPARE_COVER) and not tight_flags(r, "new", True, SPARE_COVER)
            ]
            # Равномерная выборка по порядку полос: 24 фото на глаза.
            groups = {"ref_spare": part[:: max(1, len(part) // 24)][:24]}
        else:
            groups = {}
            for r in rows:
                groups.setdefault(r.get("new_class") or "none", []).append(r)
        for name, part in groups.items():
            if not part:
                continue
            paths = write_sheets(layout, run_dir, specs_for(part, pages), base, name, f"{k.value}/{name}",
                                 per_sheet, cols, max_width, zoom)  # fmt: skip
            click.echo(f"{k.value}/{name}: фото {len(part)}, листов {len(paths)}")


@main.command()
@click.option("--root", type=click.Path(path_type=Path), required=True, help="Корень выхода стенда (SSD).")
@click.option("--run-dir", type=click.Path(exists=True, path_type=Path), required=True, help="Папка прогона.")
@click.option("--photos", type=int, default=30, show_default=True, help="Сколько фото в выборке (по 4 стороны).")
@click.option("--per-sheet", type=int, default=6, show_default=True, help="Фото на лист.")
@click.option("--seed", type=int, default=0, show_default=True, help="Зерно случайной выборки.")
def verify(root: Path, run_dir: Path, photos: int, per_sheet: int, seed: int) -> None:
    """Случайная выборка фото: вырезки середин сторон для оценки глазами, ``verify/verify_NN.jpg`` и ``verify.csv``."""
    import random

    layout = CacheLayout(root)
    pages = {p.page_key: p for p in read_pages(layout.pages_json())}
    with (run_dir / "measures.csv").open() as fh:
        rows = [r for r in csv.DictReader(fh) if not r.get("extra") and r.get("new_top_mm") not in (None, "")]
    sample = random.Random(seed).sample(rows, min(photos, len(rows)))
    table = []
    for n, start in enumerate(range(0, len(sample), per_sheet), 1):
        part = verify_sheet(layout, run_dir, sample[start : start + per_sheet], pages,
                            run_dir / "verify" / f"verify_{n:02d}.jpg")  # fmt: skip
        for row in part:
            row["n"] = len(table) + 1
            row["sheet"] = n
            table.append(row)
    write_csv(run_dir / "verify" / "verify.csv", table)
    click.echo(f"сторон {len(table)} → {run_dir / 'verify'}")


@main.command("text-count")
@click.option("--root", type=click.Path(path_type=Path), required=True, help="Корень выхода стенда (SSD).")
@click.option("--run-dir", type=click.Path(exists=True, path_type=Path), required=True, help="Папка прогона.")
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Воркеров.")
def text_count(root: Path, run_dir: Path, jobs: int) -> None:
    """Строки выпрямления на полосах набора: отсев по рамке детектора против области фото → ``text_count.csv``."""
    from research.photo_bounds.text_count import TextJob, count_page

    layout = CacheLayout(root)
    tasks = [TextJob(p, layout, run_dir) for p in read_pages(layout.pages_json())]
    rows = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for i, row in enumerate(pool.map(count_page, tasks), 1):
            rows.append(row)
            if i % 50 == 0:
                log.info("%d/%d", i, len(tasks))
    write_csv(run_dir / "text_count.csv", rows)
    for key in (
        "rect_lines",
        "quad_lines",
        "rect_points",
        "quad_points",
        "rect_lines_hit",
        "quad_lines_hit",
        "rect_inside_raster",
        "quad_inside_raster",
    ):
        click.echo(f"{key}: {sum(r[key] for r in rows)}")
    click.echo(f"полос с вернувшимися строками: {sum(1 for r in rows if r['returned_lines'] > 0)}; "
               f"почти без текста: {sum(r['few_text'] for r in rows)}")  # fmt: skip


@main.command("text-sheets")
@click.option("--root", type=click.Path(path_type=Path), required=True, help="Корень выхода стенда (SSD).")
@click.option("--run-dir", type=click.Path(exists=True, path_type=Path), required=True, help="Папка прогона.")
@click.option("--top", type=int, default=10, show_default=True, help="Сколько полос с наибольшим возвратом строк.")
def text_sheets(root: Path, run_dir: Path, top: int) -> None:
    """Листы «рамка | область фото» по полосам, где возвращается больше всего выборок строк."""
    from research.photo_bounds.text_count import TextJob, returned_sheet

    layout = CacheLayout(root)
    pages = {p.page_key: p for p in read_pages(layout.pages_json())}
    with (run_dir / "text_count.csv").open() as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: -int(r["returned_points"]))[:top]
    for r in rows:
        name = r["page"].replace("/", "_")
        returned_sheet(TextJob(pages[r["page"]], layout, run_dir), run_dir / "sheets" / "returned" / f"{name}.jpg")
        click.echo(f"{r['page']}: строк {r['rect_lines']} → {r['quad_lines']}, выборок +{r['returned_points']}")


@main.command("rectify-demo")
@click.option("--frame", "frames", type=(str, click.Path(exists=True, path_type=Path), click.Path(exists=True, path_type=Path),
              click.Path(exists=True, path_type=Path), click.Path(exists=True, path_type=Path), float), multiple=True,
              required=True, help="Кадр: ПОДПИСЬ ИСХОДНЫЙ_КАДР ВХОД_НАБОРА КЭШ_V2 ИТОГИ_V2 PX_НА_ММ; повторяется.")  # fmt: skip
@click.option("--out", type=click.Path(path_type=Path), required=True, help="Папка демонстрации.")
@click.option("--jobs", type=int, default=4, show_default=True,
              help="Воркеров: кадр держит ~3 ГБ (три варианта выпрямления), больше 4 упирается в память.")  # fmt: skip
def rectify_demo(frames: tuple, out: Path, jobs: int) -> None:
    """Выпрямление кадров с фото тремя вариантами (как есть / область фото четырёхугольником / плюс стороны в опорах)."""
    from research.photo_bounds.rectify import DemoJob, run_demo

    tasks = [DemoJob(label, src, inp, cache, v2, ppm, out) for label, src, inp, cache, v2, ppm in frames]
    rows = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for summary in pool.map(run_demo, tasks):
            rows.append(summary)
            log.info("%s: %s", summary.get("frame"), summary.get("status"))
    (out / "demo.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=float))
    from research.photo_bounds.rectify import demo_sheets

    for summary in rows:
        if summary.get("status") == "ok":
            frame_dir = out / summary["label"] / Path(summary["frame"]).stem
            for path in demo_sheets(frame_dir):
                click.echo(str(path))


@main.command("rectify-sheets")
@click.option("--out", type=click.Path(exists=True, path_type=Path), required=True, help="Папка демонстрации.")
def rectify_sheets(out: Path) -> None:
    """Листы демонстрации по готовым кадрам (``summary.json`` в папках кадров)."""
    from research.photo_bounds.rectify import demo_sheets

    for summary_path in sorted(out.rglob("summary.json")):
        for path in demo_sheets(summary_path.parent):
            click.echo(str(path))


@main.command("integrate")
@click.option("--frame", "frames", type=(str, click.Path(exists=True, path_type=Path), click.Path(exists=True, path_type=Path),
              click.Path(exists=True, path_type=Path), click.Path(exists=True, path_type=Path), float), multiple=True,
              required=True, help="Кадр: ПОДПИСЬ ИСХОДНЫЙ_КАДР ВХОД_НАБОРА КЭШ_V2 ИТОГИ_V2 PX_НА_ММ; повторяется.")  # fmt: skip
@click.option("--setting", "settings", type=(str, bool, float, float), multiple=True, required=True,
              help="Настройка: ИМЯ ФОТО(true/false) PHOTO_NEAR_MM PHOTO_NEAR_WEIGHT; повторяется.")  # fmt: skip
@click.option("--out", type=click.Path(path_type=Path), required=True, help="Папка выхода.")
@click.option("--jobs", type=int, default=6, show_default=True,
              help="Воркеров: кадр держит ~2–3 ГБ на настройку, больше 6 упирается в память.")  # fmt: skip
def integrate(frames: tuple, settings: tuple, out: Path, jobs: int) -> None:
    """Кадры выпрямления всеми настройками фото: итоги, меры текста и сторон фото → ``summary.csv``."""
    from research.photo_bounds.integrate import IntegrateJob, Setting, run_frame

    sets = tuple(Setting(*s) for s in settings)
    tasks = [IntegrateJob(label, src, inp, cache, v2, ppm, sets, out) for label, src, inp, cache, v2, ppm in frames]
    rows = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for summary in pool.map(run_frame, tasks):
            log.info("%s %s: %s", summary["label"], summary["frame"], summary["status"])
            if summary["status"] != "ok":
                rows.append({"label": summary["label"], "frame": summary["frame"], "status": summary["status"]})
                continue
            for name, m in summary["settings"].items():
                tilts = [x["tilt_deg"] for p in m["photo_sides"] for x in p]
                rows.append({"label": summary["label"], "frame": summary["frame"], "status": "ok", "setting": name,
                             "lines": "/".join(map(str, m["lines"])), "modes": "/".join(m["modes"]),
                             **{k: v for k, v in m.items() if k.startswith(("before_", "after_"))},
                             "photo_sides": len(tilts), "photo_tilt_max": round(max(tilts), 3) if tilts else "",
                             "photo_tilt_med": round(float(np.median(tilts)), 3) if tilts else ""})  # fmt: skip
    write_csv(out / "summary.csv", rows)


@main.command("integrate-sheets")
@click.option("--out", type=click.Path(exists=True, path_type=Path), required=True, help="Папка выхода integrate.")
@click.option("--before", default="off", show_default=True, help="Настройка «было».")
@click.option("--after", default="limited", show_default=True, help="Настройка «стало».")
@click.option("--threshold-mm", type=float, default=0.1, show_default=True,
              help="Изменение p95 строк или сторон блоков, с которого рисуется вся страница, мм.")  # fmt: skip
def integrate_sheets(out: Path, before: str, after: str, threshold_mm: float) -> None:
    """Листы «было | стало» по кадрам integrate: у каждого фото и страница целиком при заметном изменении."""
    from research.photo_bounds.integrate import compare_sheet

    for summary_path in sorted(out.rglob("summary.json")):
        settings = json.loads(summary_path.read_text()).get("settings", {})
        if before in settings and after in settings:
            for path in compare_sheet(summary_path.parent, before, after, threshold_mm):
                click.echo(str(path))


@main.command("rectify-scan")
@click.option("--set", "sets", type=(str, click.Path(exists=True, file_okay=False, path_type=Path), float), multiple=True,
              required=True, help="Набор: ПОДПИСЬ ПАПКА_ИТОГОВ_ВЫПРЯМЛЕНИЯ PX_НА_ММ; повторяется.")  # fmt: skip
@click.option("--out", type=click.Path(path_type=Path), required=True, help="CSV кадров с растром.")
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Воркеров.")
def rectify_scan(sets: tuple[tuple[str, Path, float], ...], out: Path, jobs: int) -> None:
    """Кадры с фото среди выпрямленных итогов page_rectify (растр page_layout без surya) → CSV."""
    from research.photo_bounds.rectify import scan_frame

    tasks, labels = [], []
    for label, folder, px_per_mm in sets:
        for path in sorted(folder.rglob("*.tif*")):
            tasks.append((path, px_per_mm * 25.4))
            labels.append(label)
    rows = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for i, (label, row) in enumerate(zip(labels, pool.map(scan_frame, tasks, chunksize=4)), 1):
            rows.append({"set": label, **row})
            if i % 100 == 0:
                log.info("%d/%d", i, len(tasks))
    write_csv(out, rows)
    click.echo(f"кадров {len(rows)}, с фото {sum(1 for r in rows if r.get('photos'))}")


if __name__ == "__main__":
    main()
