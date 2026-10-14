"""Команды стенда подшивок МТО 1990: превью, листы миниатюр, сборка набора по классам дефектов."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import click
import cv2
import numpy as np
import threadpoolctl

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed
from research.bound_volumes.fingers import FINGER_ROOT, Guard, finger_sheet, out_paths, run_frame, save
from research.bound_volumes.geometry import GeometryModels, cache_path, save_geometry, work_image
from research.bound_volumes.frames import (
    OUT_ROOT,
    SET_DIR,
    Frame,
    all_frames,
    make_preview,
    read_bgr,
    read_categories,
)

log = logging.getLogger(__name__)


def init_worker() -> None:
    """Инициализатор воркера: без hugepage и потоков BLAS/OpenCV (см. ``.claude/rules/gpu_and_pools.md``)."""
    np._core.multiarray._set_madvise_hugepage(False)
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)


@click.group()
@click.option("--log-level", default="INFO", show_default=True, help="Уровень логов.")
def main(log_level: str) -> None:
    """Стенд «тугие подшивки журналов»."""
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")


@main.command()
@click.option("--jobs", default=8, show_default=True, help="Воркеров (упор в NTFS-3G, не в счёт).")
def previews(jobs: int) -> None:
    """Превью всех кадров года на SSD (длинная сторона 1600 px)."""
    frames = all_frames()
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for i, _ in enumerate(pool.map(make_preview, frames, chunksize=4), 1):
            if i % 100 == 0:
                log.info("превью %d/%d", i, len(frames))
    log.info("превью готовы: %d", len(frames))


def thumb_tile(frame: Frame, height: int) -> np.ndarray:
    """Миниатюра кадра с подписью ключа под ней.

    Args:
        frame: Кадр (превью уже на SSD).
        height: Высота миниатюры, px.

    Returns:
        Плитка BGR высотой ``height + 24``.
    """
    img = read_bgr(frame.preview_path)
    scale = height / img.shape[0]
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    tile = np.full((height + 24, small.shape[1], 3), 255, np.uint8)
    tile[:height] = small
    cv2.putText(tile, frame.key, (3, height + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return tile


@main.command("contact-sheet")
@click.option("--cols", default=6, show_default=True, help="Миниатюр в ряд.")
@click.option("--height", default=300, show_default=True, help="Высота миниатюры, px.")
@click.option("--per-sheet", default=30, show_default=True, help="Миниатюр на лист.")
@click.option("--issue", default="", help="Только один выпуск (номер «04»); пусто — все.")
def contact_sheet(cols: int, height: int, per_sheet: int, issue: str) -> None:
    """Листы миниатюр по выпускам — для отбора трудных кадров глазами."""
    frames = [f for f in all_frames() if not issue or f.issue == issue]
    out_dir = OUT_ROOT / "contact_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    by_issue: dict[str, list[Frame]] = {}
    for f in frames:
        by_issue.setdefault(f.issue, []).append(f)
    for iss, chunk_all in by_issue.items():
        for start in range(0, len(chunk_all), per_sheet):
            tiles = [thumb_tile(f, height) for f in chunk_all[start : start + per_sheet]]
            # Плитки разной ширины (портрет/альбом) — выравниваем по самой широкой.
            width = max(t.shape[1] for t in tiles)
            tiles = [np.pad(t, ((0, 0), (0, width - t.shape[1]), (0, 0)), constant_values=255) for t in tiles]
            while len(tiles) % cols:
                tiles.append(np.full_like(tiles[0], 255))
            rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
            title = f"МТО 1990 / {iss}: кадры {start + 1}–{start + per_sheet}"
            sheet = framed(np.vstack(rows), [title], [])
            path = out_dir / f"{iss}_{start // per_sheet + 1:02d}.jpg"
            cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
            log.info("лист %s", path)


def copy_work(frame: Frame) -> None:
    """Копирует исходный JPEG кадра в рабочую папку SSD (байт в байт), если копии ещё нет.

    Args:
        frame: Кадр.
    """
    out = frame.work_path
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(frame.path.read_bytes())


@main.command("build-set")
def build_set() -> None:
    """Рабочие копии кадров набора и папки по классам (символьные ссылки на копии) по ``sets/categories.csv``."""
    labeled = read_categories()
    for item in labeled:
        copy_work(item.frame)
        for cat in item.categories:
            link = SET_DIR / cat.value / f"{item.frame.flat_key}.jpg"
            link.parent.mkdir(parents=True, exist_ok=True)
            if not link.exists():
                os.symlink(item.frame.work_path, link)
    counts = {c: sum(c in it.categories for it in labeled) for c in {c for it in labeled for c in it.categories}}
    log.info("набор: %d кадров; по классам %s", len(labeled), {c.value: n for c, n in counts.items()})


@main.command("class-sheets")
@click.option("--cols", default=3, show_default=True, help="Миниатюр в ряд.")
@click.option("--height", default=520, show_default=True, help="Высота миниатюры, px.")
@click.option("--per-sheet", default=9, show_default=True, help="Миниатюр на лист.")
def class_sheets(cols: int, height: int, per_sheet: int) -> None:
    """Листы кадров набора по классам (крупнее, чем contact-sheet) — для проверки меток глазами."""
    labeled = read_categories()
    out_dir = OUT_ROOT / "class_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    for cat in sorted({c for it in labeled for c in it.categories}, key=lambda c: c.value):
        items = [it.frame for it in labeled if cat in it.categories]
        for start in range(0, len(items), per_sheet):
            tiles = [thumb_tile(f, height) for f in items[start : start + per_sheet]]
            width = max(t.shape[1] for t in tiles)
            tiles = [np.pad(t, ((0, 0), (0, width - t.shape[1]), (0, 0)), constant_values=255) for t in tiles]
            while len(tiles) % cols:
                tiles.append(np.full_like(tiles[0], 255))
            rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
            sheet = framed(np.vstack(rows), [f"класс {cat.value}: {start + 1}–{start + per_sheet} из {len(items)}"], [])
            path = out_dir / f"{cat.value}_{start // per_sheet + 1:02d}.jpg"
            cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
            log.info("лист %s", path)


@main.command()
@click.option("--guard", "guards", multiple=True, type=click.Choice([g.value for g in Guard]), help="Варианты защиты.")
@click.option("--guard-px", default=10, show_default=True, help="Зазор у печати, px масштаба finger_protect (¼ кадра).")
@click.option("--keys", default="", help="Только эти кадры (ключи через запятую).")
@click.option("--redo", is_flag=True, help="Пересчитать и готовые кадры.")
def fingers(guards: tuple[str, ...], guard_px: int, keys: str, redo: bool) -> None:
    """Закрас пальцев по набору в вариантах защиты печати: картинки без пальцев и маски на SSD."""
    from ocr_utils.scan_cropping.gpu_models import GpuModels

    chosen = [Guard(g) for g in guards] or list(Guard)
    todo = [it.frame for it in read_categories()]
    if keys:
        todo = [f for f in todo if f.key in set(keys.split(","))]
    models = GpuModels(with_lama=True, with_layout=any(g != Guard.GLYPH for g in chosen))
    for frame in todo:
        bgr = read_bgr(frame.work_path)
        for guard in chosen:
            if not redo and out_paths(guard, frame)[0].exists():
                continue
            t0 = time.perf_counter()
            result = run_frame(bgr, models, guard, guard_px, frame.key)
            save(result, guard, frame)
            log.info("%s %s: %.1f с, %s", frame.key, guard.value, time.perf_counter() - t0, result.info)


def finger_sheet_one(frame: Frame) -> str:
    """Лист вариантов закраса одного кадра (для пула).

    Args:
        frame: Кадр.

    Returns:
        Путь листа или пустая строка, если пальцев нет.
    """
    sheet = finger_sheet(frame, [Guard.SURYA, Guard.GLYPH, Guard.INK_BACK])
    if sheet is None:
        return ""
    path = FINGER_ROOT / "sheets" / f"{frame.flat_key}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return str(path)


@main.command("finger-sheets")
@click.option("--jobs", default=16, show_default=True, help="Воркеров.")
def finger_sheets(jobs: int) -> None:
    """Листы «было | зона | стало» по каждому пальцу и варианту защиты."""
    frames = [it.frame for it in read_categories()]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        done = [p for p in pool.map(finger_sheet_one, frames) if p]
    log.info("листов: %d", len(done))


def source_image_path(source: str, frame: Frame) -> Path:
    """Путь входной картинки кадра по имени источника.

    Args:
        source: ``orig`` — рабочая копия кадра; иначе имя варианта закраса пальцев (``glyph``, ``both``, ``surya``).
        frame: Кадр.

    Returns:
        Путь JPEG.
    """
    return frame.work_path if source == "orig" else out_paths(Guard(source), frame)[0]


@main.command()
@click.option("--source", default="glyph", show_default=True, help="Вход: orig или вариант закраса пальцев.")
@click.option("--redo", is_flag=True, help="Пересчитать готовые.")
def geometry(source: str, redo: bool) -> None:
    """Кэш геометрии набора на ¼ масштаба: маски SAM2.1 (бокс + точки) и глубина/нормали MoGe-2."""
    models = GeometryModels()
    for item in read_categories():
        path = cache_path(source, item.frame)
        if path.exists() and not redo:
            continue
        t0 = time.perf_counter()
        geo = models(work_image(read_bgr(source_image_path(source, item.frame))))
        save_geometry(path, geo)
        log.info("%s: %.2f с, оценки SAM %s", item.frame.key, time.perf_counter() - t0, np.round(geo.sam_scores, 3))


@main.command("gt-export")
@click.option("--annotator", default="user", show_default=True, help="Пользователь CVAT для джоб.")
def gt_export(annotator: str) -> None:
    """Эталон верхних страниц: отбор 40 кадров (sets/gt_pick.csv) и задача CVAT с полигоном книги SAM2 на правку."""
    from research.bound_volumes.gt_cvat import PICK_CSV, export_task, pick_frames

    picks = pick_frames(read_categories())
    picks.to_csv(PICK_CSV, index=False)
    log.info("отобрано %d кадров: %s", len(picks), picks.category.value_counts().to_dict())
    log.info("задача CVAT %d", export_task(picks, annotator))


@main.command("gt-import")
def gt_import() -> None:
    """Забирает поправленные полигоны верхних страниц в маски sets/gt_masks/*.png (масштаб ½ кадра)."""
    from research.bound_volumes.gt_cvat import import_task

    log.info("эталон: %d кадров", import_task())


BG_ROOT = OUT_ROOT / "runs" / "bg"
# Вариант закраса пальцев, по которому идёт сегментация фона (меняется опцией ``--source``; glyph → runs/bg).
BG_SOURCE = "glyph"


def set_bg_source(source: str) -> None:
    """Выбирает вариант закраса для команд фона и папку выхода ``runs/bg`` или ``runs/bg_<вариант>``.

    Args:
        source: Вариант закраса пальцев.
    """
    global BG_ROOT, BG_SOURCE
    BG_SOURCE = source
    BG_ROOT = OUT_ROOT / "runs" / ("bg" if source == "glyph" else f"bg_{source}")


def bg_mask_path(method: str, frame: Frame) -> Path:
    """Путь маски фона способа: ``runs/bg/<способ>/<ключ>.png``.

    Args:
        method: Имя способа.
        frame: Кадр.

    Returns:
        Путь PNG.
    """
    return BG_ROOT / method / f"{frame.flat_key}.png"


def bg_cpu_one(args: tuple[Frame, str, str]) -> tuple[str, str, float]:
    """Маска фона одного кадра CPU-способом (для пула).

    Args:
        args: ``(кадр, имя способа, вариант закраса)``.

    Returns:
        ``(ключ, способ, секунды)``.
    """
    from research.bound_volumes.background import CPU_METHODS, BgMethod, bg_inputs

    frame, method, source = args
    set_bg_source(source)
    img, _, content, _ = bg_inputs(frame, source)
    t0 = time.perf_counter()
    bg = CPU_METHODS[BgMethod(method)](img, content)
    path = bg_mask_path(method, frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bg.astype(np.uint8) * 255)
    return frame.key, method, time.perf_counter() - t0


@main.command("bg-sam3")
@click.option("--source", default="glyph", show_default=True, help="Вариант закраса пальцев.")
@click.option("--target", type=click.Choice(["book", "bg", "bg1008", "points"]), default="book", show_default=True,
              help="book/bg — текст при 640 px; bg1008 — текст про фон при 1008 px; points — точки как у SAM 2, 1008 px.")  # fmt: skip
def bg_sam3(source: str, target: str) -> None:
    """Только SAM 3 по набору (текст про книгу или про фон) — дописывает маски к готовому прогону bg-run."""
    from research.bound_volumes.background import Sam3Background, Sam3Book, bg_inputs

    set_bg_source(source)
    from research.bound_volumes.background import Sam3Points, Sam3TextBg

    sam3 = {"book": Sam3Book, "bg": Sam3Background, "bg1008": Sam3TextBg, "points": Sam3Points}[target]()
    name = f"sam3_{target}"
    for it in read_categories():
        img, _, content, _ = bg_inputs(it.frame, source)
        t0 = time.perf_counter()
        bg = sam3(img, content)
        path = bg_mask_path(name, it.frame)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), bg.astype(np.uint8) * 255)
        log.info("%s: SAM 3 %.2f с, фон %.3f", it.frame.key, time.perf_counter() - t0, bg.mean())


@main.command("bg-run")
@click.option("--jobs", default=16, show_default=True, help="Воркеров для CPU-способов.")
@click.option("--source", default="glyph", show_default=True, help="Вариант закраса пальцев (glyph, inkback, …).")
def bg_run(jobs: int, source: str) -> None:
    """Маски фона по набору всеми способами: dark, gmm_grabcut, flood (пул), sam2_bg (GPU), vote (большинство)."""
    from research.bound_volumes.background import CPU_METHODS, BgMethod, Sam3Book, bg_inputs, bg_sam2

    set_bg_source(source)
    frames = [it.frame for it in read_categories()]
    tasks = [(f, m.value, source) for f in frames for m in CPU_METHODS]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        times: dict[str, list[float]] = {}
        for key, method, dt in pool.map(bg_cpu_one, tasks):
            times.setdefault(method, []).append(dt)
    log.info("CPU-способы, медиана с/кадр: %s", {m: round(float(np.median(t)), 2) for m, t in times.items()})
    models = GeometryModels()
    for f in frames:
        img, _, content, _ = bg_inputs(f, source)
        bg = bg_sam2(img, content, models)
        path = bg_mask_path(BgMethod.SAM2.value, f)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), bg.astype(np.uint8) * 255)
    sam3 = Sam3Book()
    for f in frames:
        img, _, content, _ = bg_inputs(f, source)
        path = bg_mask_path("sam3_book", f)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), sam3(img, content).astype(np.uint8) * 255)
    for f in frames:
        stack = np.stack([cv2.imread(str(bg_mask_path(m, f)), cv2.IMREAD_GRAYSCALE) > 0
                          for m in ("dark", "gmm_grabcut", "flood", "sam2_bg")])  # fmt: skip
        path = bg_mask_path(BgMethod.VOTE.value, f)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), (stack.sum(0) >= 3).astype(np.uint8) * 255)
    log.info("маски фона: %d кадров", len(frames))


BG_METHODS = ("dark", "gmm_grabcut", "flood", "sam2_bg", "sam3_bg", "sam3_bg1008", "sam3_points", "vote")
# Вырезка крупно на листах фона: сторона окна, px масштаба сегментации; увеличение.
BG_CROP_PX = 260
BG_CROP_ZOOM = 2


def bg_crop_centres(book: np.ndarray) -> list[tuple[str, int, int]]:
    """Центры вырезок: углы рамки книги и середины боковых сторон (по контуру голосования).

    Args:
        book: Маска книги.

    Returns:
        ``[(подпись, x, y), ...]``.
    """
    ys, xs = np.nonzero(book)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    ym = (y0 + y1) // 2
    contours, _ = cv2.findContours(book.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = np.concatenate([k[:, 0, :] for k in contours])
    out = []
    # Центр вырезки — точка контура книги, ближайшая к углу рамки или середине стороны.
    for name, x, y in (("лев-верх", x0, y0), ("прав-верх", x1, y0), ("лев-середина", x0, ym),
                       ("прав-середина", x1, ym), ("лев-низ", x0, y1), ("прав-низ", x1, y1)):  # fmt: skip
        i = int(np.argmin((c[:, 0] - x) ** 2 + (c[:, 1] - y) ** 2))
        out.append((name, int(c[i, 0]), int(c[i, 1])))
    return out


def bg_measures(bg: np.ndarray, vote: np.ndarray, content: np.ndarray, book: np.ndarray) -> dict:
    """Меры маски фона: согласие с голосованием, печать в фоне, гладкость контура книги.

    Args:
        bg: Маска фона способа.
        vote: Маска фона голосования.
        content: Своя печать.
        book: Книга способа.

    Returns:
        Словарь мер.
    """
    inter, union = (bg & vote).sum(), (bg | vote).sum()
    contours, _ = cv2.findContours(book.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=len)[:, 0, :].astype(np.float32)
    # Зубцы: точки контура дальше 6 px (½ кадра) от сглаженного по ±25 точкам контура.
    k = 25
    pad = np.r_[c[-k:], c, c[:k]]
    smooth = np.stack([np.convolve(pad[:, i], np.ones(2 * k + 1) / (2 * k + 1), "valid") for i in range(2)], 1)
    teeth = float((np.linalg.norm(c - smooth, axis=1) > 6).mean())
    return dict(iou_vote=float(inter / max(union, 1)), print_in_bg=float((content & bg).sum() / max(content.sum(), 1)),
                teeth=teeth, components=int(cv2.connectedComponents(book.astype(np.uint8))[0] - 1))  # fmt: skip


def bg_sheet_one(args) -> list[dict]:
    """Оверлеи и лист вырезок одного кадра по всем способам; меры.

    Args:
        args: ``(кадр набора с классами, вариант закраса)``.

    Returns:
        Строки мер (по способу).
    """
    item, source = args
    set_bg_source(source)
    from ocr_utils.page_layout.overlay_frame import header_strip
    from research.bound_volumes.background import bg_inputs, book_from_background, hand_mask
    from research.bound_volumes.overlay import background_overlay

    frame = item.frame
    _, src, content, core = bg_inputs(frame, source)
    hands = hand_mask(src, core)
    masks = {m: cv2.imread(str(bg_mask_path(m, frame)), cv2.IMREAD_GRAYSCALE) > 0 for m in BG_METHODS}
    stack = np.stack([masks[m] for m in ("dark", "gmm_grabcut", "flood", "sam2_bg")]).sum(0)
    dispute = (stack > 0) & (stack < 4)
    cats = ",".join(sorted(c.value for c in item.categories))
    rows, crops_rows = [], []
    vote_book = book_from_background(masks["vote"], hands)
    centres = bg_crop_centres(vote_book)
    half = BG_CROP_PX // 2
    h, w = content.shape
    for m in BG_METHODS:
        book = book_from_background(masks[m], hands)
        meas = bg_measures(masks[m], masks["vote"], content, book)
        rows.append(dict(key=frame.key, method=m, **meas))
        title = [f"МТО 1990 {frame.key} [{cats}] — фон способом {m}, по кадру после закраса пальцев ({source})",
                 f"доля фона {masks[m].mean():.3f}; согласие с голосованием IoU {meas['iou_vote']:.4f}; "
                 f"печать в фоне {meas['print_in_bg']:.4f}; зубцы контура {meas['teeth']:.3f}"]  # fmt: skip
        ov = background_overlay(src, masks[m], hands, book, title, dispute if m == "vote" else None)
        out = BG_ROOT / m / "overlays" / f"{frame.flat_key}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), ov, [cv2.IMWRITE_JPEG_QUALITY, 88])
        canvas = src.copy()
        layer = canvas.copy()
        layer[masks[m]] = (120, 40, 0)
        layer[hands] = (180, 105, 255)
        if m == "vote":
            layer[dispute] = (0, 165, 255)
        cv2.addWeighted(layer, 0.45, canvas, 0.55, 0, canvas)
        cs, _ = cv2.findContours(book.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, cs, -1, (220, 90, 20), 1)
        tiles = []
        for _, x, y in centres:
            x0, y0 = int(np.clip(x - half, 0, w - BG_CROP_PX)), int(np.clip(y - half, 0, h - BG_CROP_PX))
            tile = canvas[y0 : y0 + BG_CROP_PX, x0 : x0 + BG_CROP_PX]
            tiles.append(cv2.resize(tile, None, fx=BG_CROP_ZOOM, fy=BG_CROP_ZOOM, interpolation=cv2.INTER_NEAREST))
            tiles.append(np.full((tiles[-1].shape[0], 6, 3), 255, np.uint8))
        row = np.hstack(tiles[:-1])
        crops_rows.append(np.vstack([header_strip([f"способ {m}"], row.shape[1]), row]))
    names = "   ".join(n for n, _, _ in centres)
    sheet = framed(np.vstack(crops_rows), [f"МТО 1990 {frame.key} [{cats}] — фон по способам (кадр после закраса пальцев, {source}), вырезки ×{BG_CROP_ZOOM} ({names})"],
                   [LegendEntry("фон (коврик)", (120, 40, 0), alpha=0.45, style=SampleStyle.BOX),
                    LegendEntry("руки", (180, 105, 255), alpha=0.45, style=SampleStyle.BOX),
                    LegendEntry("граница книги (не фон − руки)", (220, 90, 20)),
                    LegendEntry("спорно (только у vote)", (0, 165, 255), alpha=0.45, style=SampleStyle.BOX)])  # fmt: skip
    out = BG_ROOT / "crops" / f"{frame.flat_key}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return rows


@main.command("bg-sheets")
@click.option("--jobs", default=12, show_default=True, help="Воркеров.")
@click.option("--source", default="glyph", show_default=True, help="Вариант закраса пальцев (glyph, inkback, …).")
def bg_sheets(jobs: int, source: str) -> None:
    """Оверлеи фона по способам, листы вырезок углов и сторон, сводка мер ``runs/bg/measures.csv``."""
    import pandas as pd

    set_bg_source(source)
    items = [(it, source) for it in read_categories()]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        rows = [r for rs in pool.map(bg_sheet_one, items) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(BG_ROOT / "measures.csv", index=False)
    summary = df.groupby("method").agg(
        iou_vote_med=("iou_vote", "median"), iou_vote_min=("iou_vote", "min"),
        print_in_bg_max=("print_in_bg", "max"), frames_print_lost=("print_in_bg", lambda s: int((s > 0.001).sum())),
        teeth_med=("teeth", "median"), teeth_p90=("teeth", lambda s: s.quantile(0.9)),
    )  # fmt: skip
    log.info("сводка:\n%s", summary.round(4).to_string())


POST_ROOT = OUT_ROOT / "runs" / "book_post"


def book_post_one(args) -> list[dict]:
    """Постобработка маски книги одного кадра во всех вариантах: маски, оверлеи «варианты рядом», меры.

    Args:
        args: ``(кадр набора с классами, вариант закраса, способ фона)``.

    Returns:
        Строки мер по вариантам.
    """
    from research.bound_volumes.background import bg_inputs, hand_mask
    from research.bound_volumes.book_mask import PostVariant, postprocess
    from research.bound_volumes.stripes import content_hulls

    item, source, method = args
    frame = item.frame
    img, _, content, core = bg_inputs(frame, source)
    root = OUT_ROOT / "runs" / ("bg" if source == "glyph" else f"bg_{source}")
    bg = cv2.imread(str(root / method / f"{frame.flat_key}.png"), cv2.IMREAD_GRAYSCALE) > 0
    hands = hand_mask(img, core)
    hulls = content_hulls(content, ~bg)
    rows, tiles = [], []
    cats = ",".join(sorted(c.value for c in item.categories))
    for v in PostVariant:
        r = postprocess(bg, hands, hulls, v)
        out = POST_ROOT / v.value / f"{frame.flat_key}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), r.book.astype(np.uint8) * 255)
        lost = float((content & ~r.book).sum() / max(content.sum(), 1))
        comps = int(cv2.connectedComponents(r.book.astype(np.uint8))[0] - 1)
        rows.append(dict(key=frame.key, variant=v.value, print_lost=lost, added=float(r.added.mean()),
                         removed=float(r.removed.mean()), components=comps))  # fmt: skip
        canvas = img.copy()
        layer = canvas.copy()
        layer[~r.book] = (120, 40, 0)
        layer[r.added] = (0, 170, 0)
        layer[r.removed] = (0, 0, 220)
        cv2.addWeighted(layer, 0.5, canvas, 0.5, 0, canvas)
        cs, _ = cv2.findContours(r.book.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, cs, -1, (220, 90, 20), 3)
        tile = framed(canvas, [f"{v.value}: печать вне книги {lost:.4f}; добавлено {r.added.mean():.4f}; снято {r.removed.mean():.4f}"], [])
        tiles.append(cv2.resize(tile, (1100, int(tile.shape[0] * 1100 / tile.shape[1])), interpolation=cv2.INTER_AREA))
    h = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0)), constant_values=255) for t in tiles]
    tiles.append(np.full_like(tiles[0], 255))
    sheet = framed(np.vstack([np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]),
                   [f"МТО 1990 {frame.key} [{cats}] — постобработка маски книги (фон {method}, кадр после закраса пальцев {source})"],
                   [LegendEntry("не книга", (120, 40, 0), alpha=0.5, style=SampleStyle.BOX),
                    LegendEntry("добавлено закрытием / заливкой / оболочкой печати", (0, 170, 0), alpha=0.5, style=SampleStyle.BOX),
                    LegendEntry("снято: руки (raw) или выступ", (0, 0, 220), alpha=0.5, style=SampleStyle.BOX),
                    LegendEntry("граница книги", (220, 90, 20))])  # fmt: skip
    out = POST_ROOT / "sheets" / f"{frame.flat_key}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return rows


@main.command("book-post")
@click.option("--source", default="inkback", show_default=True, help="Вариант закраса пальцев.")
@click.option("--method", default="flood", show_default=True, help="Способ маски фона.")
@click.option("--jobs", default=12, show_default=True, help="Воркеров.")
def book_post(source: str, method: str, jobs: int) -> None:
    """Постобработка маски книги (варианты book_mask.PostVariant) по набору: маски, листы, меры."""
    import pandas as pd

    items = [(it, source, method) for it in read_categories()]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        rows = [r for rs in pool.map(book_post_one, items) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(POST_ROOT / "measures.csv", index=False)
    s = df.groupby("variant").agg(print_lost_max=("print_lost", "max"), frames_lost=("print_lost", lambda x: int((x > 1e-4).sum())),
                                  added_med=("added", "median"), removed_med=("removed", "median"),
                                  frames_cut=("removed", lambda x: int((x > 1e-4).sum())), multi_comp=("components", lambda x: int((x > 1).sum())))  # fmt: skip
    log.info("сводка:\n%s", s.round(4).to_string())


PAPER_ROOT = OUT_ROOT / "runs" / "paper"
PAPER_METHODS = ("lab_paper", "grabcut_paper", "sam2_paper", "sam3_page", "dino_knn", "vote")


def paper_inputs(frame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Входы сегментации бумаги: кадр после закраса (½), книга после постобработки close_open, печать, оболочки печати.

    Args:
        frame: Кадр.

    Returns:
        ``(BGR, книга, печать, оболочки печати)``.
    """
    from research.bound_volumes.background import bg_inputs
    from research.bound_volumes.stripes import content_hulls

    img, _, content, _ = bg_inputs(frame, "inkback")
    book = cv2.imread(str(POST_ROOT / "close_open" / f"{frame.flat_key}.png"), cv2.IMREAD_GRAYSCALE) > 0
    return img, book, content, content_hulls(content, book)


def paper_cpu_one(args) -> tuple[str, float]:
    """Бумага CPU-способом (lab, grabcut) для одного кадра.

    Args:
        args: ``(кадр, способ)``.

    Returns:
        ``(способ, секунды)``.
    """
    from research.bound_volumes import paper as P

    frame, method = args
    img, book, _, hulls = paper_inputs(frame)
    t0 = time.perf_counter()
    pap = {"lab_paper": P.paper_lab, "grabcut_paper": P.paper_grabcut}[method](img, book, hulls)
    path = PAPER_ROOT / method / f"{frame.flat_key}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), pap.astype(np.uint8) * 255)
    return method, time.perf_counter() - t0


@main.command("paper-run")
@click.option("--jobs", default=12, show_default=True, help="Воркеров CPU-способов.")
def paper_run(jobs: int) -> None:
    """Сегментация бумаги внутри книги всеми способами по набору; голосование (3 из 5)."""
    from research.bound_volumes import paper as P

    frames = [it.frame for it in read_categories()]
    times: dict[str, list[float]] = {}
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for m, dt in pool.map(paper_cpu_one, [(f, m) for f in frames for m in ("lab_paper", "grabcut_paper")]):
            times.setdefault(m, []).append(dt)
    gpu = [("sam2_paper", lambda: (lambda g: lambda i, b, h: P.paper_sam2(i, b, h, g))(GeometryModels())),
           ("sam3_page", P.Sam3Page), ("dino_knn", P.DinoKnn)]  # fmt: skip
    for name, make in gpu:
        engine = make()
        for f in frames:
            img, book, _, hulls = paper_inputs(f)
            t0 = time.perf_counter()
            pap = engine(img, book, hulls)
            times.setdefault(name, []).append(time.perf_counter() - t0)
            path = PAPER_ROOT / name / f"{f.flat_key}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), pap.astype(np.uint8) * 255)
        del engine
        import torch

        torch.cuda.empty_cache()
    for f in frames:
        stack = np.stack([cv2.imread(str(PAPER_ROOT / m / f"{f.flat_key}.png"), cv2.IMREAD_GRAYSCALE) > 0
                          for m in PAPER_METHODS[:5]]).sum(0)  # fmt: skip
        path = PAPER_ROOT / "vote" / f"{f.flat_key}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), (stack >= 3).astype(np.uint8) * 255)
    log.info("секунд на кадр, медиана: %s", {m: round(float(np.median(t)), 2) for m, t in times.items()})


def paper_sheet_one(args) -> list[dict]:
    """Оверлеи бумаги по способам, лист вырезок у кромки, меры одного кадра.

    Args:
        args: ``(кадр набора с классами, способы, имя листа вырезок)``.

    Returns:
        Строки мер.
    """
    item, methods, sheet_name = args
    from ocr_utils.page_layout.overlay_frame import header_strip
    from research.bound_volumes import paper as P

    frame = item.frame
    img, book, content, hulls = paper_inputs(frame)
    lab = P.smooth_lab(img)
    model = P.PaperModel(lab, P.paper_seeds(img, hulls, book))
    colour_far = model.distance(lab) > P.MAHAL_THR * 2
    masks = {m: cv2.imread(str(PAPER_ROOT / m / f"{frame.flat_key}.png"), cv2.IMREAD_GRAYSCALE) > 0
             for m in set(methods) | {"vote"}}  # fmt: skip
    dispute = np.zeros_like(book)
    full_ink = P.ink_components(img, book) & (cv2.dilate(masks["vote"].astype(np.uint8), P.disk(P.INK_REACH)) > 0)
    gt = paper_gt(frame, book.shape)
    cats = ",".join(sorted(c.value for c in item.categories))
    centres = bg_crop_centres(masks["vote"] if masks["vote"].any() else book)
    h, w = book.shape
    half = BG_CROP_PX // 2
    rows, crop_rows = [], []
    for m in methods:
        pap = masks[m]
        meas = bg_measures(~pap, ~masks["vote"], content, pap)
        gt_err = contour_errors(pap, gt) if gt is not None else None
        rows.append(dict(key=frame.key, method=m, paper_share=float(pap.sum() / max(book.sum(), 1)),
                         print_lost=float((content & ~pap).sum() / max(content.sum(), 1)),
                         foreign_colour=float((pap & colour_far).sum() / max(pap.sum(), 1)),
                         iou_vote=meas["iou_vote"], teeth=meas["teeth"],
                         ink_lost=float((full_ink & ~pap).sum() / max(full_ink.sum(), 1)),
                         gt_med=float(np.median(np.abs(gt_err))) if gt_err is not None else np.nan,
                         gt_wider=float(np.mean(gt_err > 8)) if gt_err is not None else np.nan,
                         gt_narrower=float(np.mean(gt_err < -8)) if gt_err is not None else np.nan))  # fmt: skip
        canvas = img.copy()
        layer = canvas.copy()
        layer[~book] = (120, 40, 0)
        layer[book & ~pap] = (0, 0, 220)
        cv2.addWeighted(layer, 0.45, canvas, 0.55, 0, canvas)
        cs, _ = cv2.findContours(pap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, cs, -1, (220, 90, 20), 2)
        lost = full_ink & ~pap
        canvas[cv2.dilate(lost.astype(np.uint8), P.disk(2)) > 0] = (0, 0, 255)
        added_path = PAPER_ROOT / m / "added" / f"{frame.flat_key}.png"
        if added_path.exists():
            added = cv2.imread(str(added_path), cv2.IMREAD_GRAYSCALE) > 0
            lay2 = canvas.copy()
            lay2[added] = (0, 170, 0)
            cv2.addWeighted(lay2, 0.55, canvas, 0.45, 0, canvas)
        if gt is not None:
            gcs, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(canvas, gcs, -1, (180, 105, 255), 2)
        r = rows[-1]
        ov = framed(canvas, [f"МТО 1990 {frame.key} [{cats}] — бумага способом {m}",
                             f"бумага/книга {r['paper_share']:.3f}; печать вне бумаги {r['print_lost']:.4f}; "
                             f"краска вне бумаги {r['ink_lost']:.4f}; чужой цвет {r['foreign_colour']:.4f}; зубцы {r['teeth']:.3f}; "
                             f"ошибка к эталону, медиана {r['gt_med']:.1f} px"],
                    PAPER_LEGEND)  # fmt: skip
        out = PAPER_ROOT / m / "overlays" / f"{frame.flat_key}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), ov, [cv2.IMWRITE_JPEG_QUALITY, 88])
        tiles = []
        for _, x, y in centres:
            x0, y0 = int(np.clip(x - half, 0, w - BG_CROP_PX)), int(np.clip(y - half, 0, h - BG_CROP_PX))
            tiles.append(cv2.resize(canvas[y0 : y0 + BG_CROP_PX, x0 : x0 + BG_CROP_PX], None, fx=BG_CROP_ZOOM,
                                    fy=BG_CROP_ZOOM, interpolation=cv2.INTER_NEAREST))  # fmt: skip
            tiles.append(np.full((tiles[-1].shape[0], 6, 3), 255, np.uint8))
        row = np.hstack(tiles[:-1])
        crop_rows.append(np.vstack([header_strip([f"способ {m}"], row.shape[1]), row]))
    names = "   ".join(n for n, _, _ in centres)
    sheet = framed(np.vstack(crop_rows), [f"МТО 1990 {frame.key} [{cats}] — бумага по способам, вырезки ×{BG_CROP_ZOOM} ({names})"], PAPER_LEGEND)
    out = PAPER_ROOT / sheet_name / f"{frame.flat_key}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return rows


PAPER_LEGEND = [
    LegendEntry("не книга (фон)", (120, 40, 0), alpha=0.45, style=SampleStyle.BOX),
    LegendEntry("книга, но не бумага", (0, 0, 220), alpha=0.45, style=SampleStyle.BOX),
    LegendEntry("граница бумаги", (220, 90, 20)),
    LegendEntry("краска вне бумаги (номера, сноски)", (0, 0, 255), style=SampleStyle.BOX),
    LegendEntry("эталон пользователя (7 кадров)", (180, 105, 255)),
    LegendEntry("добавлено склейкой страниц", (0, 170, 0), alpha=0.55, style=SampleStyle.BOX),
]


def paper_gt(frame: Frame, shape: tuple[int, int]) -> np.ndarray | None:
    """Эталон верхних страниц из правки пользователя (задача CVAT 46), если кадр поправлен.

    Args:
        frame: Кадр.
        shape: ``(H, W)`` ½ кадра.

    Returns:
        Маска или None.
    """
    path = Path(__file__).resolve().parent / "sets" / "gt46.json"
    if not path.exists():
        return None
    shapes = json.loads(path.read_text()).get(f"{frame.flat_key}.jpg")
    if not shapes or not shapes[0]["changed"]:
        return None
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.asarray(shapes[0]["points"]).round().astype(np.int32)], 1)
    return mask > 0


def contour_errors(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Знаковое расстояние от точек контура эталона (не на рамке кадра) до контура маски: + — маска шире.

    Args:
        pred: Маска.
        gt: Эталон.

    Returns:
        Массив расстояний, px.
    """
    h, w = gt.shape
    cs, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=len)[:, 0, :]
    c = c[(c[:, 0] > 3) & (c[:, 0] < w - 4) & (c[:, 1] > 3) & (c[:, 1] < h - 4)][::3]
    dt_out = cv2.distanceTransform((~pred).astype(np.uint8), cv2.DIST_L2, 5)
    dt_in = cv2.distanceTransform(pred.astype(np.uint8), cv2.DIST_L2, 5)
    return np.where(pred[c[:, 1], c[:, 0]], dt_in[c[:, 1], c[:, 0]], -dt_out[c[:, 1], c[:, 0]])


@main.command("paper-sheets")
@click.option("--jobs", default=12, show_default=True, help="Воркеров.")
@click.option("--methods", default=",".join(PAPER_METHODS), show_default=True, help="Способы через запятую.")
@click.option("--name", default="crops", show_default=True, help="Имя папки листов и файла мер.")
def paper_sheets(jobs: int, methods: str, name: str) -> None:
    """Оверлеи и листы вырезок бумаги по способам, сводка мер ``runs/paper/<имя>_measures.csv``."""
    import pandas as pd

    chosen = methods.split(",")
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        rows = [r for rs in pool.map(paper_sheet_one, [(it, chosen, name) for it in read_categories()]) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(PAPER_ROOT / f"{name}_measures.csv", index=False)
    s = df.groupby("method").agg(paper_share=("paper_share", "median"), print_lost_max=("print_lost", "max"),
                                 frames_lost=("print_lost", lambda x: int((x > 0.001).sum())),
                                 foreign_med=("foreign_colour", "median"), foreign_p90=("foreign_colour", lambda x: x.quantile(0.9)),
                                 ink_lost_max=("ink_lost", "max"), frames_ink=("ink_lost", lambda x: int((x > 0.001).sum())),
                                 teeth=("teeth", "median"), gt_med=("gt_med", "median"),
                                 gt_wider=("gt_wider", "mean"), gt_narrower=("gt_narrower", "mean"))  # fmt: skip
    log.info("сводка:\n%s", s.round(4).to_string())


def gt_frames() -> list[Frame]:
    """40 кадров эталона (задача CVAT 46) — набор шага защитной маски.

    Returns:
        Кадры.
    """
    import pandas as pd

    from research.bound_volumes.frames import frame_by_key
    from research.bound_volumes.gt_cvat import PICK_CSV

    frames = all_frames()
    return [frame_by_key(k, frames) for k in pd.read_csv(PICK_CSV).key]


@main.command("content-run")
@click.option("--paper-method", default="sam3_1288_ink_concave", show_default=True, help="Способ бумаги для области paper.")
@click.option("--engines", default="surya_layout,surya_lines,page_layout,pero,deepseek", show_default=True)
@click.option("--all-frames", "use_all", is_flag=True, help="Все 95 кадров набора (по умолчанию — 40 кадров эталона).")
def content_run(paper_method: str, engines: str, use_all: bool) -> None:
    """Вырезки областей paper и book, разбор страницы движками; объекты в ``runs/content/<область>/<движок>``."""
    import subprocess

    from research.bound_volumes import content as C

    frames = [it.frame for it in read_categories()] if use_all else gt_frames()
    chosen = engines.split(",")
    crops = {}
    for area in C.Area:
        for f in frames:
            crops[(area, f.key)] = C.make_crop(f, area, paper_method)
    log.info("вырезки готовы: %d", len(crops))

    def run_each(name: str, fn) -> None:
        for (area, key), (path, meta) in crops.items():
            frame = next(f for f in frames if f.key == key)
            t0 = time.perf_counter()
            items = fn(path, meta)
            C.save_items(C.items_path(area, name, frame), items, time.perf_counter() - t0)
        log.info("%s: готово", name)

    if {"surya_layout", "page_layout"} & set(chosen):
        from ocr_utils.page_layout.surya.model import SuryaLayoutModel

        model = SuryaLayoutModel()
        if "surya_layout" in chosen:
            run_each("surya_layout", lambda p, m: C.surya_items(cv2.cvtColor(read_bgr(p), cv2.COLOR_BGR2RGB), model, m))
        if "page_layout" in chosen:
            run_each("page_layout", lambda p, m: C.page_layout_items(p, model, m))
        model.close()
    if "surya_lines" in chosen:
        from surya.detection import DetectionPredictor

        det = DetectionPredictor()
        run_each("surya_lines", lambda p, m: C.surya_line_items(cv2.cvtColor(read_bgr(p), cv2.COLOR_BGR2RGB), det, m))
        del det
        import torch

        torch.cuda.empty_cache()
    if "pero" in chosen:
        run_each("pero", lambda p, m: C.pero_items(read_bgr(p), m))
    if "deepseek" in chosen:
        work = C.CONTENT_ROOT / "deepseek_work"
        work.mkdir(parents=True, exist_ok=True)
        jobs = work / "jobs.jsonl"
        ids = {f"{area.value}__{key.replace('/', '__')}": (area, key) for (area, key) in crops}
        jobs.write_text("".join(json.dumps({"id": i, "crop": str(crops[v][0])}) + "\n" for i, v in ids.items()))
        out_dir = work / "out"
        (out_dir / "markdown.jsonl").unlink(missing_ok=True)
        ddir = Path("ocr_utils/page_layout/line_art/deepseek/vllm_env")
        t0 = time.perf_counter()
        # Окружение родительского ``uv run`` (VIRTUAL_ENV и пр.) воркеру vLLM не передаём: у него свой проект.
        env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "CUDA_VISIBLE_DEVICES")}
        code = subprocess.run(["uv", "run", "--project", str(ddir), "python", str(ddir / "worker.py"), "--jobs", str(jobs),
                               "--out-dir", str(out_dir), "--prompts", "markdown"],
                              stdin=subprocess.DEVNULL, env=env, start_new_session=True).returncode  # fmt: skip
        per = (time.perf_counter() - t0) / max(len(ids), 1)
        if code != 0:
            raise click.ClickException(f"DeepSeek: код {code}")
        for line in (out_dir / "markdown.jsonl").read_text().splitlines():
            rec = json.loads(line)
            area, key = ids[rec["id"]]
            frame = next(f for f in frames if f.key == key)
            C.save_items(C.items_path(area, "deepseek", frame), C.deepseek_items(rec, crops[(area, key)][1]), per)
        log.info("deepseek: готово")


CONTENT_ENGINES = ("glyphs", "surya_layout", "surya_lines", "pero", "page_layout", "deepseek", "page_layout+deepseek",
                   "text_blocks", "page_layout_curved", "page_layout_curved+deepseek")  # fmt: skip
# Цвета объектов по виду (BGR): текст — зелёный, заголовок — оранжевый, таблица — фиолетовый, рисунок — красный,
# строка — бирюзовый, прочее — серый.
KIND_COLOURS = [("table", (200, 60, 200)), ("pic", (0, 0, 220)), ("figure", (0, 0, 220)), ("image", (0, 0, 220)),
                ("raster", (0, 0, 220)), ("line_art", (0, 0, 220)), ("title", (0, 165, 255)), ("header", (0, 165, 255)),
                ("line", (140, 140, 0)), ("text", (40, 170, 40)), ("region", (40, 170, 40))]  # fmt: skip
# text_block содержит «text» — попадает в зелёный.
GUARD_COLOUR = (60, 200, 255)


CONTENT_LEGEND = [
    LegendEntry("защитная маска (объекты + зазор 1.5 мм)", (60, 200, 255), alpha=0.35, style=SampleStyle.BOX),
    LegendEntry("вне области анализа", (120, 40, 0), alpha=0.35, style=SampleStyle.BOX),
    LegendEntry("граница области", (220, 90, 20)), LegendEntry("текст / регион", (40, 170, 40)),
    LegendEntry("заголовок", (0, 165, 255)), LegendEntry("таблица", (200, 60, 200)),
    LegendEntry("рисунок / растр / line art", (0, 0, 220)), LegendEntry("строка", (140, 140, 0)),
    LegendEntry("прочее", (150, 150, 150)),
]


def kind_colour(kind: str) -> tuple[int, int, int]:
    """Цвет объекта по его виду.

    Args:
        kind: Метка движка.

    Returns:
        BGR.
    """
    k = kind.lower()
    return next((c for key, c in KIND_COLOURS if key in k), (150, 150, 150))


def engine_items(area, engine: str, frame: Frame, content: np.ndarray):
    """Объекты движка по кадру и области (глифы — из маски своей печати; объединение — из двух движков).

    Args:
        area: Область.
        engine: Имя движка.
        frame: Кадр.
        content: Своя печать (защита глифов) ½ кадра.

    Returns:
        ``(объекты, секунды)``.
    """
    from research.bound_volumes import content as C

    if engine == "glyphs":
        cs, _ = cv2.findContours(content.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [C.Item("text", c[:, 0, :].astype(float).tolist()) for c in cs if len(c) >= 3], 0.0
    if engine == "page_layout_curved":
        a, ta = engine_items(area, "text_blocks", frame, content)
        b, tb = engine_items(area, "page_layout", frame, content)
        return a + C.non_text_layout(b), ta + tb
    if engine == "page_layout_curved+deepseek":
        a, ta = engine_items(area, "page_layout_curved", frame, content)
        b, tb = engine_items(area, "deepseek", frame, content)
        return a + b, ta + tb
    if engine == "page_layout+deepseek":
        a, ta = engine_items(area, "page_layout", frame, content)
        b, tb = engine_items(area, "deepseek", frame, content)
        return a + b, ta + tb
    path = C.items_path(area, engine, frame)
    return C.load_items(path) if path.exists() else ([], float("nan"))


def content_sheet_one(args) -> list[dict]:
    """Лист «движок × (бумага | книга)» одного кадра и меры.

    Args:
        args: ``(кадр, способ бумаги)``.

    Returns:
        Строки мер.
    """
    from ocr_utils.page_layout.overlay_frame import header_strip
    from research.bound_volumes import content as C
    from research.bound_volumes.background import bg_inputs

    frame, paper_method = args
    img, _, content, _ = bg_inputs(frame, "inkback")
    areas = {a: C.area_mask(frame, a, paper_method) for a in C.Area}
    paper = areas[C.Area.PAPER]
    rows, tile_rows = [], []
    masks = {}
    for engine in CONTENT_ENGINES:
        tiles = []
        for area in C.Area:
            items, secs = engine_items(area, engine, frame, content & areas[area])
            mask = C.protection_mask(items, paper.shape, areas[area])
            masks[(engine, area)] = mask
            own = content & areas[area]
            missed = float((own & ~mask).sum() / max(own.sum(), 1))
            outside = float((mask & ~paper).sum() / max(mask.sum(), 1))
            rows.append(dict(key=frame.key, engine=engine, area=area.value, objects=len(items), missed=missed,
                             outside_paper=outside, mask_share=float(mask.sum() / max(areas[area].sum(), 1)), seconds=secs))  # fmt: skip
            canvas = img.copy()
            layer = canvas.copy()
            layer[mask] = GUARD_COLOUR
            layer[~areas[area]] = (120, 40, 0)
            cv2.addWeighted(layer, 0.35, canvas, 0.65, 0, canvas)
            for it in items:
                if len(it.polygon) >= 2:
                    cv2.polylines(canvas, [np.asarray(it.polygon).round().astype(np.int32)], True, kind_colour(it.kind), 2)
            cs, _ = cv2.findContours(areas[area].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(canvas, cs, -1, (220, 90, 20), 3)
            small = cv2.resize(canvas, (1000, int(canvas.shape[0] * 1000 / canvas.shape[1])), interpolation=cv2.INTER_AREA)
            cap = f"{engine} | область {area.value}: объектов {len(items)}, печать без защиты {missed:.4f}, защита вне бумаги {outside:.3f}"
            # Полноразмерный оверлей способа (½ кадра) — для отсмотра по одному.
            full = C.CONTENT_ROOT / "overlays" / engine / area.value / f"{frame.flat_key}.jpg"
            full.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(full), framed(canvas, [f"МТО 1990 {frame.key} — {cap}"], CONTENT_LEGEND), [cv2.IMWRITE_JPEG_QUALITY, 88])
            tiles.append(np.vstack([header_strip([cap], small.shape[1]), small]))
        tile_rows.append(np.hstack([tiles[0], np.full((tiles[0].shape[0], 8, 3), 255, np.uint8), tiles[1]]))
    for engine in CONTENT_ENGINES:
        a, b = masks[(engine, C.Area.PAPER)], masks[(engine, C.Area.BOOK)]
        for r in rows:
            if r["engine"] == engine:
                r["iou_paper_vs_book"] = float((a & b).sum() / max((a | b).sum(), 1))
    legend = CONTENT_LEGEND
    out_dir = C.CONTENT_ROOT / "sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(tile_rows), 4):
        sheet = framed(np.vstack(tile_rows[i : i + 4]),
                       [f"МТО 1990 {frame.key} — защитная маска контента по движкам; слева область «бумага» ({paper_method}), справа «книга»"],
                       legend)  # fmt: skip
        cv2.imwrite(str(out_dir / f"{frame.flat_key}_{i // 4 + 1}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return rows


@main.command("deepseek-import")
@click.option("--raw", "raw_path", type=click.Path(path_type=Path), required=True, help="JSONL воркера deepseek_hf.py.")
def deepseek_import(raw_path: Path) -> None:
    """Разбор вывода DeepSeek-OCR-2 (transformers) в объекты ``runs/content/<область>/deepseek``."""
    from ocr_utils.page_layout.line_art.deepseek.parse import parse
    from research.bound_volumes import content as C
    from research.bound_volumes.frames import frame_by_key

    frames = all_frames()
    n = 0
    for line in raw_path.read_text().splitlines():
        rec = json.loads(line)
        area_name, flat = rec["id"].split("__", 1)
        area = C.Area(area_name)
        frame = frame_by_key(flat.replace("__", "/"), frames)
        meta = C.read_meta(C.crop_dir(area) / f"{frame.flat_key}.png")
        # Формат vLLM-воркера уже с элементами; формат deepseek_hf.py — сырой вывод.
        elements = rec["elements"] if "elements" in rec else parse(rec["raw"], *rec["size"])
        C.save_items(C.items_path(area, "deepseek", frame), C.deepseek_items(dict(elements=elements), meta), rec.get("seconds", np.nan))
        n += 1
    log.info("DeepSeek: разобрано %d вырезок", n)


def text_blocks_one(args) -> str:
    """Огибающие text_blocks по одной вырезке (для пула).

    Args:
        args: ``(область, кадр)``.

    Returns:
        Ключ кадра.
    """
    from research.bound_volumes import content as C

    area, frame = args
    path = C.crop_dir(area) / f"{frame.flat_key}.png"
    t0 = time.perf_counter()
    items = C.text_blocks_items(path, C.read_meta(path))
    C.save_items(C.items_path(area, "text_blocks", frame), items, time.perf_counter() - t0)
    return frame.key


@main.command("content-text-blocks")
@click.option("--jobs", default=12, show_default=True, help="Воркеров (детектор на CPU).")
def content_text_blocks(jobs: int) -> None:
    """Криволинейные текстовые блоки ``page_layout.text_blocks`` по готовым вырезкам обеих областей."""
    from research.bound_volumes import content as C

    tasks = [(a, f) for a in C.Area for f in gt_frames()]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        done = list(pool.map(text_blocks_one, tasks))
    log.info("text_blocks: %d вырезок", len(done))


@main.command("content-sheets")
@click.option("--paper-method", default="sam3_1288_ink_concave", show_default=True)
@click.option("--all-frames", "use_all", is_flag=True)
@click.option("--jobs", default=10, show_default=True)
def content_sheets(paper_method: str, use_all: bool, jobs: int) -> None:
    """Листы защитных масок по движкам и областям, сводка ``runs/content/measures.csv``."""
    import pandas as pd

    from research.bound_volumes import content as C

    frames = [it.frame for it in read_categories()] if use_all else gt_frames()
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        rows = [r for rs in pool.map(content_sheet_one, [(f, paper_method) for f in frames]) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(C.CONTENT_ROOT / "measures.csv", index=False)
    s = df.groupby(["engine", "area"]).agg(missed_med=("missed", "median"), missed_p90=("missed", lambda x: x.quantile(0.9)),
                                           outside_med=("outside_paper", "median"), outside_p90=("outside_paper", lambda x: x.quantile(0.9)),
                                           mask_share=("mask_share", "median"), iou_ab=("iou_paper_vs_book", "median"),
                                           sec=("seconds", "median"))  # fmt: skip
    log.info("сводка:\n%s", s.round(4).to_string())


@main.command("sam3-variants")
@click.option("--imgsz", "sizes", multiple=True, type=int, default=(1008, 1288), show_default=True)
@click.option("--jobs", default=12, show_default=True)
def sam3_variants(sizes: tuple[int, ...], jobs: int) -> None:
    """SAM 3 по бумаге: сырые маски при разных разрешениях (GPU) и все варианты постобработки (пул) → ``runs/paper``."""
    from research.bound_volumes import paper as P
    from research.bound_volumes.background import sam3_overrides
    from ultralytics.models.sam import SAM3SemanticPredictor

    frames = [it.frame for it in read_categories()]
    for size in sizes:
        ov = sam3_overrides("finger_models/sam3/sam3.pt")
        ov["imgsz"] = size
        pred = SAM3SemanticPredictor(overrides=ov)
        for f in frames:
            path = PAPER_ROOT / f"sam3raw_{size}" / f"{f.flat_key}.png"
            if path.exists():
                continue
            img, _, _, hulls = paper_inputs(f)
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), P.sam3_raw(pred, img, hulls).astype(np.uint8) * 255)
        del pred
        log.info("SAM 3 %d: сырые маски готовы", size)
    tasks = [(f, s, v.value) for f in frames for s in sizes for v in P.PostVariant]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        list(pool.map(sam3_post_one, tasks))
    log.info("варианты: %s", sorted({f"sam3_{s}_{v.value}" for _, s, v in [(0, s, P.PostVariant(v)) for _, s, v in tasks]}))


def sam3_post_one(args) -> None:
    """Постобработка сырой маски SAM 3 одного кадра в одном варианте (для пула).

    Args:
        args: ``(кадр, разрешение, вариант)``.
    """
    from research.bound_volumes import paper as P

    frame, size, variant = args
    img, book, _, hulls = paper_inputs(frame)
    raw = cv2.imread(str(PAPER_ROOT / f"sam3raw_{size}" / f"{frame.flat_key}.png"), cv2.IMREAD_GRAYSCALE) > 0
    pap = P.postprocess_sam3(raw, img, book, hulls, P.PostVariant(variant))
    path = PAPER_ROOT / f"sam3_{size}_{variant}" / f"{frame.flat_key}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), pap.astype(np.uint8) * 255)


GLUE_BASE = "sam3_1288_plain"


def paper_glue_one(args) -> list[dict]:
    """Склейка страниц одного кадра всеми способами (и с заливкой вогнутостей); маски и меры добавленного.

    Args:
        args: ``(кадр,)``.

    Returns:
        Строки мер.
    """
    from research.bound_volumes import paper as P
    from research.bound_volumes.glue import GlueMethod, glue, split_pages

    (frame,) = args
    img, book, content, hulls = paper_inputs(frame)
    base = cv2.imread(str(PAPER_ROOT / GLUE_BASE / f"{frame.flat_key}.png"), cv2.IMREAD_GRAYSCALE) > 0
    inst = cv2.imread(str(PAPER_ROOT / "sam3inst_1288" / f"{frame.flat_key}.png"), cv2.IMREAD_GRAYSCALE)
    lab = P.smooth_lab(img)
    model = P.PaperModel(lab, P.paper_seeds(img, hulls, book))
    far = model.distance(lab) > 2 * P.MAHAL_THR
    edge = book & (cv2.distanceTransform(book.astype(np.uint8), cv2.DIST_L2, 5) < 15)
    rows = []
    for m in GlueMethod:
        for concave in (False, True):
            for filt in (True, False):
                name = f"glue_{m.value}" + ("_concave" if concave else "") + ("" if filt else "_nofilter")
                r = glue(base, book, m, concave, content, inst, between_filter=filt)
                for sub, mask in ((name, r.paper), (f"{name}/added", r.added)):
                    path = PAPER_ROOT / sub / f"{frame.flat_key}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
                added = max(int(r.added.sum()), 1)
                rows.append(dict(key=frame.key, method=name, added=float(r.added.sum() / max(book.sum(), 1)),
                                 added_far_colour=float((r.added & far).sum() / added),
                                 added_outer=float((r.added & edge & ~r.band).sum() / added),
                                 pages_before=len(split_pages(base, book, content, inst)),
                                 pages_after=int(cv2.connectedComponents(r.paper.astype(np.uint8))[0] - 1)))  # fmt: skip
    return rows


@main.command("paper-glue")
@click.option("--jobs", default=12, show_default=True, help="Воркеров.")
def paper_glue(jobs: int) -> None:
    """Склейка страниц маски бумаги ``sam3_1288_plain`` всеми способами; меры добавленного ``runs/paper/glue_measures.csv``."""
    import pandas as pd

    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        rows = [r for rs in pool.map(paper_glue_one, [(it.frame,) for it in read_categories()]) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(PAPER_ROOT / "glue_added_measures.csv", index=False)
    s = df.groupby("method").agg(added_med=("added", "median"), added_max=("added", "max"),
                                 frames_added=("added", lambda x: int((x > 0.002).sum())),
                                 far_colour=("added_far_colour", "median"), outer_max=("added_outer", "max"),
                                 one_comp=("pages_after", lambda x: int((x == 1).sum())))  # fmt: skip
    log.info("сводка (страниц до склейки ровно одна у %d кадров):\n%s",
             int((df[df.method == df.method.iloc[0]].pages_before == 1).sum()), s.round(4).to_string())  # fmt: skip


@main.command("sam3-instances")
@click.option("--imgsz", default=1288, show_default=True)
def sam3_instances_cmd(imgsz: int) -> None:
    """Экземпляры-страницы SAM 3 картой меток ``runs/paper/sam3inst_<размер>/<кадр>.png`` (для склейки страниц)."""
    from research.bound_volumes import paper as P
    from research.bound_volumes.background import sam3_overrides
    from ultralytics.models.sam import SAM3SemanticPredictor

    ov = sam3_overrides("finger_models/sam3/sam3.pt")
    ov["imgsz"] = imgsz
    pred = SAM3SemanticPredictor(overrides=ov)
    for it in read_categories():
        img, _, _, hulls = paper_inputs(it.frame)
        path = PAPER_ROOT / f"sam3inst_{imgsz}" / f"{it.frame.flat_key}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), P.sam3_instances(pred, img, hulls))
    log.info("экземпляры SAM 3 готовы")


TELEA_ROOT = OUT_ROOT / "runs" / "telea_bbox"


def telea_bbox_one(args) -> tuple[str, float]:
    """Кадр без пальцев в полном разрешении: вырезка по рамке маски бумаги, всё вне маски — дорисовано Telea.

    Args:
        args: ``(кадр, способ маски бумаги)``.

    Returns:
        ``(ключ, доля дорисованного в рамке)``.
    """
    from research.bound_volumes.fingers import Guard, out_paths
    from research.newspaper_crop.fill import fill_non_paper

    frame, method = args
    full = read_bgr(out_paths(Guard.INK_BACK, frame)[0])
    h, w = full.shape[:2]
    half = cv2.imread(str(PAPER_ROOT / method / f"{frame.flat_key}.png"), cv2.IMREAD_GRAYSCALE)
    # Маска ½ кадра → полный кадр: билинейно и порог, чтобы край не был ступенчатым.
    paper = cv2.resize(half, (w, h), interpolation=cv2.INTER_LINEAR) >= 128
    ys, xs = np.nonzero(paper)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop, mask = full[y0:y1, x0:x1], paper[y0:y1, x0:x1]
    out = fill_non_paper(crop, mask)
    root = TELEA_ROOT / method
    (root / "full").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / "full" / f"{frame.flat_key}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    # Превью «было | стало» с границей бумаги — для отсмотра.
    s = 1000 / out.shape[1]
    before = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    after = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.astype(np.uint8), (before.shape[1], before.shape[0]), interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(before, cs, -1, (220, 90, 20), 2)
    pad = np.full((before.shape[0], 10, 3), 255, np.uint8)
    prev = framed(np.hstack([before, pad, after]),
                  [f"МТО 1990 {frame.key} — рамка бумаги {method}: было | дорисовано Telea вне бумаги"],
                  [LegendEntry("граница бумаги", (220, 90, 20))])  # fmt: skip
    (root / "preview").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(root / "preview" / f"{frame.flat_key}.jpg"), prev, [cv2.IMWRITE_JPEG_QUALITY, 85])
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    # Тёмное в дорисовке: пиксели вне бумаги после Telea темнее медианы бумаги на DARK_DROP — пятна торца, обложки.
    dark = (~mask) & (gray < float(np.median(gray[mask])) - DARK_DROP)
    return frame.key, float((~mask).mean()), float(dark.mean())


DARK_DROP = 40


@main.command("telea-bbox")
@click.option("--method", default="glue_hull_bridge_nofilter", show_default=True, help="Способ маски бумаги.")
@click.option("--jobs", default=8, show_default=True, help="Воркеров (полные кадры 5813×4419 — память).")
def telea_bbox(method: str, jobs: int) -> None:
    """Вырезка кадров по рамке маски бумаги, дорисовка не-бумаги Telea → ``runs/telea_bbox/<способ>/{full,preview}``."""
    tasks = [(it.frame, method) for it in read_categories()]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        res = list(pool.map(telea_bbox_one, tasks))
    share = np.array([r[1] for r in res])
    dark = np.array([r[2] for r in res])
    import pandas as pd

    pd.DataFrame(res, columns=["key", "filled", "dark"]).to_csv(TELEA_ROOT / method / "measures.csv", index=False)
    log.info("готово %d кадров; дорисовано в рамке: медиана %.3f; тёмное в дорисовке: медиана %.5f, p90 %.5f, макс %.5f, "
             "кадров > 0.001: %d", len(res), np.median(share), np.median(dark), np.quantile(dark, 0.9), dark.max(),
             int((dark > 0.001).sum()))  # fmt: skip


def glue_refine_one(args) -> list[dict]:
    """Уточнение склейки одного кадра всеми способами и глубинами; маски и меры.

    Args:
        args: ``(кадр, исходная склейка, глубины)``.

    Returns:
        Строки мер.
    """
    from research.bound_volumes.glue import split_pages
    from research.bound_volumes.glue_refine import RefineMethod, refine

    frame, base_name, depths = args
    img, book, content, hulls = paper_inputs(frame)
    base = cv2.imread(str(PAPER_ROOT / "sam3_1288_plain" / f"{frame.flat_key}.png"), cv2.IMREAD_GRAYSCALE) > 0
    glued = cv2.imread(str(PAPER_ROOT / base_name / f"{frame.flat_key}.png"), cv2.IMREAD_GRAYSCALE) > 0
    inst = cv2.imread(str(PAPER_ROOT / "sam3inst_1288" / f"{frame.flat_key}.png"), cv2.IMREAD_GRAYSCALE)
    pages = split_pages(base, book, content, inst)
    rows = []
    for m in RefineMethod:
        for d in depths:
            r = refine(img, book, base, glued, hulls, pages, m, d)
            name = f"glue_refined_{m.value}_d{d}"
            for sub, mask in ((name, r.paper), (f"{name}/added", r.added), (f"{name}/cut", r.cut)):
                path = PAPER_ROOT / sub / f"{frame.flat_key}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
            rows.append(dict(key=frame.key, method=name, added_before=int((glued & ~base).sum()),
                             band=int(r.band.sum()), cut=int(r.cut.sum())))  # fmt: skip
    return rows


@main.command("glue-refine")
@click.option("--base", "base_name", default="glue_hull_bridge_nofilter", show_default=True, help="Исходная склейка.")
@click.option("--depth", "depths", default="70,100", show_default=True, help="Глубины полосы уточнения, px ½ кадра.")
@click.option("--jobs", default=12, show_default=True, help="Воркеров.")
def glue_refine(base_name: str, depths: str, jobs: int) -> None:
    """Уточнение зоны склейки у края книги (бумага / не бумага / фон) → ``runs/paper/glue_refined_<способ>_d<D>``."""
    import pandas as pd

    ds = [int(x) for x in depths.split(",")]
    tasks = [(it.frame, base_name, ds) for it in read_categories()]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        rows = [r for rs in pool.map(glue_refine_one, tasks) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(PAPER_ROOT / "glue_refine_measures.csv", index=False)
    s = df.groupby("method").agg(frames_cut=("cut", lambda x: int((x > 200).sum())), cut_px=("cut", "sum"),
                                 band_px=("band", "sum"), added_px=("added_before", "sum"))  # fmt: skip
    s["cut_of_band"] = s.cut_px / s.band_px.clip(lower=1)
    log.info("сводка:\n%s", s.round(3).to_string())


@main.command("edge-fix")
@click.option("--jobs", default=12, show_default=True, help="Воркеров.")
@click.option("--skip-sam3", is_flag=True, help="Не пересчитывать SAM 3 с низким порогом (маски уже есть).")
def edge_fix(jobs: int, skip_sam3: bool) -> None:
    """Доводка края маски ``sam3_1288_plain`` → ``runs/paper/sam3plain_{snap,snap_bg,lowthr,dilate}``."""
    from research.bound_volumes import paper as P
    from research.bound_volumes.background import sam3_overrides
    from research.bound_volumes.edge_fix import make_low_thr_predictor

    if not skip_sam3:
        ov = sam3_overrides("finger_models/sam3/sam3.pt")
        ov["imgsz"] = 1288
        pred = make_low_thr_predictor(ov)
        for it in read_categories():
            img, _, _, hulls = paper_inputs(it.frame)
            path = PAPER_ROOT / "sam3raw_1288_lowthr" / f"{it.frame.flat_key}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), P.sam3_raw(pred, img, hulls).astype(np.uint8) * 255)
        del pred
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        list(pool.map(edge_fix_one, [it.frame for it in read_categories()]))
    log.info("доводка края готова")


def edge_fix_one(frame: Frame) -> None:
    """Три доводки края одного кадра (CPU-часть).

    Args:
        frame: Кадр.
    """
    from research.bound_volumes import paper as P
    from research.bound_volumes.edge_fix import SNAP_BG_REACH, dilate_fixed, snap_outward

    img, book, _, hulls = paper_inputs(frame)
    k = frame.flat_key
    plain = cv2.imread(str(PAPER_ROOT / "sam3_1288_plain" / f"{k}.png"), cv2.IMREAD_GRAYSCALE) > 0
    bg = cv2.imread(str(OUT_ROOT / "runs" / "bg_inkback" / "flood" / f"{k}.png"), cv2.IMREAD_GRAYSCALE) > 0
    raw_low = cv2.imread(str(PAPER_ROOT / "sam3raw_1288_lowthr" / f"{k}.png"), cv2.IMREAD_GRAYSCALE) > 0
    out = {
        "sam3plain_snap": snap_outward(img, plain, book, bg),
        "sam3plain_snap_bg": snap_outward(img, plain, book, bg, bg_reach=SNAP_BG_REACH),
        "sam3plain_lowthr": P.postprocess_sam3(raw_low, img, book, hulls, P.PostVariant.PLAIN),
        "sam3plain_dilate": dilate_fixed(plain, book, bg),
    }
    for name, m in out.items():
        path = PAPER_ROOT / name / f"{k}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), m.astype(np.uint8) * 255)


EDGE_BASES = ("sam3_1288_plain", "sam3plain_snap", "sam3plain_snap_bg", "sam3plain_lowthr", "sam3plain_dilate")


def edge_chain_one(frame: Frame) -> None:
    """Склейка ``hull_bridge`` без фильтра и уточнение ``colour`` D=100 поверх каждой доводки края одного кадра.

    Пишет ``runs/paper/<доводка>_gr/<кадр>.png`` — маска бумаги, готовая к дорисовке до рамки.

    Args:
        frame: Кадр.
    """
    from research.bound_volumes.glue import GlueMethod, glue, split_pages
    from research.bound_volumes.glue_refine import RefineMethod, refine

    img, book, content, hulls = paper_inputs(frame)
    inst = cv2.imread(str(PAPER_ROOT / "sam3inst_1288" / f"{frame.flat_key}.png"), cv2.IMREAD_GRAYSCALE)
    for name in EDGE_BASES:
        base = cv2.imread(str(PAPER_ROOT / name / f"{frame.flat_key}.png"), cv2.IMREAD_GRAYSCALE) > 0
        glued = glue(base, book, GlueMethod.HULL_BRIDGE, False, content, inst, between_filter=False).paper
        pages = split_pages(base, book, content, inst)
        out = refine(img, book, base, glued, hulls, pages, RefineMethod.COLOUR, 100).paper
        path = PAPER_ROOT / f"{name}_gr" / f"{frame.flat_key}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), out.astype(np.uint8) * 255)


@main.command("edge-chain")
@click.option("--jobs", default=12, show_default=True, help="Воркеров.")
def edge_chain(jobs: int) -> None:
    """Склейка и уточнение поверх всех доводок края → ``runs/paper/<доводка>_gr`` (вход ``telea-bbox``)."""
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        list(pool.map(edge_chain_one, [it.frame for it in read_categories()]))
    log.info("склейка поверх доводок края готова")


# Сторона вырезки листа дорисовки, px полного кадра, и её показ на листе, px.
TELEA_CROP = 520
TELEA_SHOW = 260


def telea_canvas(frame: Frame, method: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Дорисовка Telea способа ``method``, положенная на место в полном кадре (вне рамки — серое), и маска бумаги.

    Args:
        frame: Кадр.
        method: Способ маски бумаги (папка ``runs/telea_bbox/<способ>/full``).
        shape: ``(высота, ширина)`` полного кадра.

    Returns:
        ``(холст BGR, маска бумаги полного кадра)``.
    """
    h, w = shape
    half = cv2.imread(str(PAPER_ROOT / method / f"{frame.flat_key}.png"), cv2.IMREAD_GRAYSCALE)
    paper = cv2.resize(half, (w, h), interpolation=cv2.INTER_LINEAR) >= 128
    ys, xs = np.nonzero(paper)
    x0, y0 = xs.min(), ys.min()
    out = read_bgr(TELEA_ROOT / method / "full" / f"{frame.flat_key}.jpg")
    canvas = np.full((h, w, 3), 90, np.uint8)
    canvas[y0 : y0 + out.shape[0], x0 : x0 + out.shape[1]] = out
    return canvas, paper


def telea_spots(paper: np.ndarray, added: np.ndarray) -> list[tuple[str, int, int]]:
    """Места у кромки для вырезок: 4 угла рамки бумаги, середины левой и правой стороны, верх и низ корешка.

    Args:
        paper: Маска бумаги полного кадра (опорный способ).
        added: Добавленное склейкой (полный кадр) — по нему ищется корешок.

    Returns:
        Список ``(подпись, x центра, y центра)``.
    """
    ys, xs = np.nonzero(paper)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    q = TELEA_CROP // 2 - 40
    spots = [("лев-верх", x0 + q, y0 + q), ("прав-верх", x1 - q, y0 + q), ("лев-сер", x0 + q, (y0 + y1) // 2),
             ("прав-сер", x1 - q, (y0 + y1) // 2), ("лев-низ", x0 + q, y1 - q), ("прав-низ", x1 - q, y1 - q)]  # fmt: skip
    ay, ax = np.nonzero(added)
    if len(ax):
        sx = int(np.median(ax))
        col = paper[:, max(sx - 5, 0) : sx + 5].any(1)
        top, bottom = np.nonzero(col)[0][[0, -1]]
        spots += [("корешок-верх", sx, top + q), ("корешок-низ", sx, bottom - q)]
    return spots


def telea_sheet_one(args) -> None:
    """Лист одного кадра: строки — способы, столбцы — места у кромки, дорисовка Telea в полном разрешении.

    Args:
        args: ``(кадр, способы, папка выхода)``.
    """
    frame, methods, out_dir = args
    full = read_bgr(out_paths(Guard.INK_BACK, frame)[0])
    h, w = full.shape[:2]
    added_half = cv2.imread(str(PAPER_ROOT / "glue_refined_colour_d100" / "added" / f"{frame.flat_key}.png"), 0)
    added = cv2.resize(added_half, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    canvases = [telea_canvas(frame, m, (h, w)) for m in methods]
    spots = telea_spots(canvases[0][1], added)
    r = TELEA_CROP // 2
    rows = []
    for (canvas, paper), m in zip(canvases, methods):
        tiles = []
        for _, cx, cy in spots:
            cx, cy = int(np.clip(cx, r, w - r)), int(np.clip(cy, r, h - r))
            sl = (slice(cy - r, cy + r), slice(cx - r, cx + r))
            tile = canvas[sl].copy()
            # Граница бумаги тонкой линией: видно, что дорисовано, а что своё.
            cs, _ = cv2.findContours(paper[sl].astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(tile, cs, -1, (220, 90, 20), 2)
            tile = cv2.resize(tile, (TELEA_SHOW, TELEA_SHOW), interpolation=cv2.INTER_AREA)
            tiles.append(np.pad(tile, ((0, 0), (0, 4), (0, 0)), constant_values=255))
        row = np.hstack(tiles)
        label = np.full((28, row.shape[1], 3), 255, np.uint8)
        cv2.putText(label, m, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        rows += [label, row]
    sheet = framed(np.vstack(rows),
                   [f"МТО 1990 {frame.key} — дорисовка Telea до рамки по способам доводки края; вырезки "
                    f"{TELEA_CROP} px полного кадра: " + "  ".join(s[0] for s in spots)],
                   [LegendEntry("граница бумаги (вне — дорисовано)", (220, 90, 20))])  # fmt: skip
    cv2.imwrite(str(out_dir / f"{frame.flat_key}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


@main.command("telea-sheets")
@click.option("--methods", default=",".join(f"{b}_gr" for b in EDGE_BASES), show_default=True, help="Способы.")
@click.option("--name", default="edge_telea", show_default=True, help="Папка листов в runs/telea_bbox.")
@click.option("--jobs", default=8, show_default=True, help="Воркеров (полные кадры — память).")
def telea_sheets(methods: str, name: str, jobs: int) -> None:
    """Листы «способ × место у кромки» по дорисовке Telea до рамки → ``runs/telea_bbox/<name>/<кадр>.jpg``."""
    out_dir = TELEA_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(it.frame, methods.split(","), out_dir) for it in read_categories()]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        list(pool.map(telea_sheet_one, tasks))
    log.info("листы дорисовки: %s", out_dir)
