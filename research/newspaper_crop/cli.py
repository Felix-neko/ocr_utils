"""Команды стенда зоны копирования газет: выборка, превью, листы миниатюр, прогоны движков, меры, оверлеи."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
import threadpoolctl
import torch
from PIL import Image

from ocr_utils.page_layout.overlay_frame import framed
from research.newspaper_crop.engines import EngineName, make_engine
import research.newspaper_crop.frames as frames_module
from research.newspaper_crop.frames import (
    ALL_CSV,
    NOT_PAGES,
    OUT_ROOT,
    SAMPLE_CSV,
    SOURCE_FOLDERS,
    Frame,
    ensure_preview,
    even_pick,
    list_raf,
    read_preview_rgb,
    read_sample,
    write_sample,
)

from research.newspaper_crop.ink_guard import print_cluster, print_guard, text_lines
from research.newspaper_crop.overlay import zone_overlay
from research.newspaper_crop.zone import Zone, build_zone, ink_in_fill, ink_mask, zone_of

log = logging.getLogger(__name__)


def _init_worker() -> None:
    """Инициализатор воркера: без hugepage и потоков BLAS/OpenCV (см. ``.claude/rules/gpu_and_pools.md``)."""
    np._core.multiarray._set_madvise_hugepage(False)
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)


def _all_frames() -> list[Frame]:
    """Все RAF исходных папок как кадры без пометок.

    Returns:
        Список кадров по группам и именам.
    """
    return [Frame(group, p) for group, folder in SOURCE_FOLDERS.items() for p in list_raf(folder)]


@click.group()
@click.option("--log-level", default="INFO", show_default=True, help="Уровень логов.")
@click.option(
    "--set", "frame_set", type=click.Choice(["sample", "all"]), default="sample", show_default=True,
    help="Набор кадров: выборка (sets/sample.csv) или все кадры двух папок (sets/all.csv, команда select-all).",
)  # fmt: skip
def main(log_level: str, frame_set: str) -> None:
    """Стенд «зона копирования на фотосканах газет»."""
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")
    frames_module.ACTIVE_SET = ALL_CSV if frame_set == "all" else SAMPLE_CSV


@main.command("select-all")
def select_all() -> None:
    """Все кадры двух папок → ``sets/all.csv``; обложки и форзацы (``frames.NOT_PAGES``) помечены skip=1."""
    frames = [Frame(f.group, f.path, f.key in NOT_PAGES, NOT_PAGES.get(f.key, "")) for f in _all_frames()]
    write_sample(frames, ALL_CSV)
    log.info("все кадры: %d, из них не брать %d → %s", len(frames), sum(f.skip for f in frames), ALL_CSV)


@main.command()
@click.option("--all-frames", is_flag=True, help="Все RAF исходных папок, а не только выборка.")
@click.option("--jobs", default=8, show_default=True, help="Воркеров; 8 — упор в NTFS-3G /mnt/dump3, не в счёт.")
def previews(all_frames: bool, jobs: int) -> None:
    """Кэширует встроенные JPEG-превью RAF на SSD."""
    frames = _all_frames() if all_frames else read_sample()
    done = 0
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        futures = {pool.submit(ensure_preview, f): f for f in frames}
        for fut in as_completed(futures):
            try:
                fut.result()
                done += 1
            except Exception as exc:  # noqa: BLE001 — битый RAF не должен ронять прогон
                log.warning("превью %s: %s", futures[fut].key, exc)
    log.info("превью готово: %d из %d", done, len(frames))


@main.command()
@click.option("--per-group", default=25, show_default=True, help="Сколько кадров равномерно из каждой группы.")
@click.option("--extra", multiple=True, help="Доп. кадры «группа/имя» (трудные случаи), можно несколько.")
def select(per_group: int, extra: tuple[str, ...]) -> None:
    """Собирает выборку: равномерно по группам плюс трудные кадры; прежние пометки skip/note сохраняются."""
    old = {f.key: f for f in read_sample()} if SAMPLE_CSV.exists() else {}
    picked: list[Frame] = []
    for group, folder in SOURCE_FOLDERS.items():
        picked += [Frame(group, p) for p in even_pick(list_raf(folder), per_group)]
    for item in extra:
        group, stem = item.split("/", 1)
        picked.append(Frame(group, SOURCE_FOLDERS[group] / f"{stem}.RAF", note="трудный"))
    # Без дублей, с переносом прежних пометок.
    seen: dict[str, Frame] = {}
    for f in picked:
        if f.key not in seen:
            seen[f.key] = old.get(f.key, f)
    write_sample(list(seen.values()))
    log.info("выборка: %d кадров → %s", len(seen), SAMPLE_CSV)


@main.command("contact-sheet")
@click.option("--group", required=True, help="Группа кадров.")
@click.option("--cols", default=10, show_default=True, help="Миниатюр в ряд.")
@click.option("--thumb", default=220, show_default=True, help="Высота миниатюры, px.")
@click.option("--per-sheet", default=60, show_default=True, help="Миниатюр на лист.")
def contact_sheet(group: str, cols: int, thumb: int, per_sheet: int) -> None:
    """Листы миниатюр всех превью группы с подписями — для отбора трудных кадров и обложек глазами."""
    from ocr_utils.page_layout.overlay_frame import framed

    frames = [f for f in _all_frames() if f.group == group and f.preview_path.exists()]
    out_dir = OUT_ROOT / "contact_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(frames), per_sheet):
        chunk = frames[start : start + per_sheet]
        tiles = []
        for f in chunk:
            img = Image.open(f.preview_path)
            img.thumbnail((thumb * 2, thumb))
            tile = np.full((thumb + 22, int(thumb * 0.7) + 6, 3), 255, np.uint8)
            arr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w = arr.shape[:2]
            w = min(w, tile.shape[1])
            tile[:h, :w] = arr[:, :w]
            cv2.putText(tile, f.path.stem, (2, thumb + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            tiles.append(tile)
        while len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 255))
        rows = [np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)]
        sheet = framed(np.vstack(rows), [f"{group}: кадры {start + 1}–{start + len(chunk)}"], [])
        path = out_dir / f"{group}_{start // per_sheet + 1:02d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        log.info("лист %s", path)


RUNS_DIR = OUT_ROOT / "runs"


def run_dir(engine: str, tag: str) -> Path:
    """Папка прогона движка: ``<OUT_ROOT>/runs/<движок>[_<метка>]``.

    Args:
        engine: Имя движка.
        tag: Метка варианта (например, ``nofinger``) или пустая строка.

    Returns:
        Путь папки.
    """
    return RUNS_DIR / (f"{engine}_{tag}" if tag else engine)


@main.command()
@click.option("--engine", "engine_name", type=click.Choice([e.value for e in EngineName]), required=True)
@click.option("--tag", default="", help="Метка варианта прогона (суффикс папки).")
@click.option(
    "--source",
    type=click.Choice(["preview", "nofinger"]),
    default="preview",
    show_default=True,
    help="Вход: превью как есть или превью с закрашенными пальцами (команда fingers).",
)
@click.option("--sam3-weights", default="finger_models/sam3/sam3.pt", show_default=True, help="Веса SAM 3.")
@click.option("--limit", default=0, show_default=True, help="Только первые N кадров выборки (0 — все).")
def run(engine_name: str, tag: str, source: str, sam3_weights: str, limit: int) -> None:
    """Прогон движка по выборке (кадры skip=1 пропускаются): маска листа, четырёхугольник, оверлей, сводка."""
    engine = make_engine(EngineName(engine_name), sam3_weights)
    out = run_dir(engine_name, tag)
    frames = [f for f in read_sample() if not f.skip]
    frames = frames[:limit] if limit else frames
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for frame in frames:
        rgb = read_source_rgb(frame, source)
        started = time.time()
        raw = engine(rgb)
        seconds = time.time() - started
        rows.append(
            finish_frame(out, frame, rgb, raw, seconds, f"{engine_name}{' / ' + tag if tag else ''}, вход {source}")
        )
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    log.info("готово: %d кадров, пик VRAM %.2f ГБ → %s", len(rows), torch.cuda.max_memory_allocated() / 1e9, out)


def finish_frame(
    out: Path,
    frame: Frame,
    rgb: np.ndarray,
    raw: np.ndarray,
    seconds: float,
    title: str,
    zone: Zone | None = None,
    folds: list[np.ndarray] | None = None,
    note: str = "",
) -> dict:
    """Общий хвост кадра: зона из сырой маски, запись масок и четырёхугольника, оверлей, строка сводки.

    Args:
        out: Папка прогона.
        frame: Кадр.
        rgb: Кадр RGB (вход движка).
        raw: Сырая маска движка на рабочем масштабе.
        seconds: Время движка на кадр, с.
        title: Подпись варианта в шапке оверлея.
        zone: Готовая зона (после правки) или None — собрать из ``raw``.
        folds: Принятые складки корешка для оверлея (ломаные в пикселях маски) или None.
        note: Дополнительная строка шапки.

    Returns:
        Строка сводки.
    """
    zone = zone if zone is not None else build_zone(raw)
    save_zone(out, frame, zone, raw)
    ink_fill = ink_in_fill(zone, ink_mask(rgb, zone.paper.shape))
    print_cut, print_fill = print_guard(zone, print_cluster(text_lines(rgb, zone.paper.shape)))
    finger = read_finger_mask(frame, zone.paper.shape)
    header = [
        f"{frame.key} — движок {title}",
        f"лист {zone.paper_frac:.3f} кадра, дорисовка {zone.fill_frac:.3f}, отрезано {zone.outside_frac:.4f}, "
        f"строки: вне {print_cut:.4f}, в дорисовке {print_fill:.4f}, трапеция {zone.keystone_deg:.1f}°, {seconds:.2f} с",
    ] + ([note] if note else [])
    picture = zone_overlay(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), zone, header, finger, folds=folds)
    ov = out / "overlays" / (frame.key.replace("/", "__") + ".jpg")
    ov.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ov), picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
    log.info(
        "%s: лист %.3f, дорисовка %.3f, краска в дорисовке %.4f", frame.key, zone.paper_frac, zone.fill_frac, ink_fill
    )
    return dict(
        key=frame.key,
        ok=zone.ok,
        paper_frac=zone.paper_frac,
        fill_frac=zone.fill_frac,
        outside_frac=zone.outside_frac,
        ink_in_fill=ink_fill,
        print_cut=print_cut,
        print_fill=print_fill,
        keystone_deg=zone.keystone_deg,
        seconds=seconds,
    )


@main.command()
@click.option("--run", "run_name", required=True, help="Имя папки прогона в runs/.")
@click.option("--source", type=click.Choice(["preview", "nofinger"]), default="preview", show_default=True)
def rezone(run_name: str, source: str) -> None:
    """Пересобирает зоны, оверлеи и сводку прогона из сохранённых сырых масок (без GPU)."""
    out = RUNS_DIR / run_name
    old = pd.read_csv(out / "summary.csv").set_index("key")
    rows = []
    for frame in [f for f in read_sample() if not f.skip]:
        raw_path = out / "zones" / (frame.key + "_raw.png")
        if not raw_path.exists():
            continue
        raw = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE) > 0
        seconds = float(old.loc[frame.key, "seconds"]) if frame.key in old.index else float("nan")
        rows.append(
            finish_frame(out, frame, read_source_rgb(frame, source), raw, seconds, f"{run_name}, вход {source}")
        )
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    log.info("пересобрано: %d кадров → %s", len(rows), out)


def read_source_rgb(frame: Frame, source: str) -> np.ndarray:
    """Вход движка: превью или превью с закрашенными пальцами.

    Args:
        frame: Кадр.
        source: ``preview`` или ``nofinger``.

    Returns:
        RGB-массив кадра.
    """
    if source == "nofinger":
        path = OUT_ROOT / "nofinger" / (frame.key + ".jpg")
        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return read_preview_rgb(frame)


def finger_mask_path(frame: Frame) -> Path:
    """Путь маски пальцев кадра (пишет команда ``fingers``).

    Args:
        frame: Кадр.

    Returns:
        Путь PNG.
    """
    return OUT_ROOT / "fingers" / (frame.key + ".png")


def read_finger_mask(frame: Frame, shape: tuple[int, int]) -> np.ndarray | None:
    """Маска пальцев на рабочем масштабе, если команда ``fingers`` уже прогнана.

    Args:
        frame: Кадр.
        shape: Рабочий (H, W).

    Returns:
        Булева маска или None.
    """
    path = finger_mask_path(frame)
    if not path.exists():
        return None
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def save_zone(out: Path, frame: Frame, zone: Zone, raw: np.ndarray) -> None:
    """Пишет маску листа, сырую маску движка (PNG) и четырёхугольник в нормированных координатах (JSON).

    Args:
        out: Папка прогона.
        frame: Кадр.
        zone: Зона.
        raw: Сырая маска движка.
    """
    base = out / "zones" / frame.key
    base.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(base) + "_paper.png", zone.paper.astype(np.uint8) * 255)
    cv2.imwrite(str(base) + "_raw.png", raw.astype(np.uint8) * 255)
    h, w = zone.paper.shape
    quad = None if zone.quad is None else (zone.quad / np.array([w, h])).round(5).tolist()
    (Path(str(base) + "_zone.json")).write_text(
        json.dumps(dict(quad_norm=quad, paper_frac=zone.paper_frac, fill_frac=zone.fill_frac), ensure_ascii=False)
    )


@main.command()
@click.option("--redo", is_flag=True, help="Пересчитать и уже готовые кадры (по умолчанию — пропускать).")
def fingers(redo: bool) -> None:
    """Пальцы по набору стеком scan_cropping (YOLO-World hand → SAM → LaMa): маска и превью с закрасом."""
    from ocr_utils.scan_cropping.finger_removal.removal import remove_fingers
    from ocr_utils.scan_cropping.gpu_models import GpuModels

    todo = [f for f in read_sample() if not f.skip]
    if not redo:
        todo = [
            f
            for f in todo
            if not (finger_mask_path(f).exists() and (OUT_ROOT / "nofinger" / (f.key + ".jpg")).exists())
        ]
    log.info("пальцы: %d кадров к расчёту", len(todo))
    models = GpuModels(with_lama=True)
    for frame in todo:
        bgr = cv2.cvtColor(read_preview_rgb(frame), cv2.COLOR_RGB2BGR)
        cleaned, mask, *_ = remove_fingers(bgr, models, log_name=frame.key)
        out_img = OUT_ROOT / "nofinger" / (frame.key + ".jpg")
        out_img.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img), cleaned, [cv2.IMWRITE_JPEG_QUALITY, 95])
        mpath = finger_mask_path(frame)
        mpath.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mpath), (mask > 0).astype(np.uint8) * 255)
        log.info("%s: пальцы %.4f кадра", frame.key, float((mask > 0).mean()))


@main.command()
@click.option("--run", "run_name", required=True, help="Имя папки прогона в runs/.")
@click.option("--cols", default=6, show_default=True, help="Оверлеев в ряд.")
@click.option("--height", default=520, show_default=True, help="Высота миниатюры, px.")
@click.option("--per-sheet", default=12, show_default=True, help="Оверлеев на лист.")
def sheets(run_name: str, cols: int, height: int, per_sheet: int) -> None:
    """Листы миниатюр оверлеев прогона для отсмотра глазами: ``runs/<прогон>/sheets/NN.jpg``."""
    src = RUNS_DIR / run_name / "overlays"
    paths = sorted(src.glob("*.jpg"))
    out = RUNS_DIR / run_name / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(paths), per_sheet):
        tiles = []
        for p in paths[start : start + per_sheet]:
            img = cv2.imread(str(p))
            tiles.append(cv2.resize(img, (int(img.shape[1] * height / img.shape[0]), height), cv2.INTER_AREA))
        width = max(t.shape[1] for t in tiles)
        tiles = [np.pad(t, ((0, 0), (0, width - t.shape[1]), (0, 0)), constant_values=255) for t in tiles]
        while len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 255))
        grid = np.vstack([np.hstack(tiles[i : i + cols]) for i in range(0, len(tiles), cols)])
        cv2.imwrite(str(out / f"{start // per_sheet + 1:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    log.info("листы: %s", out)


@main.command()
@click.option("--runs", "run_names", multiple=True, required=True, help="Прогоны-участники (не меньше трёх).")
@click.option("--name", required=True, help="Имя прогона-итога.")
@click.option("--source", type=click.Choice(["preview", "nofinger"]), default="nofinger", show_default=True)
def vote(run_names: tuple[str, ...], name: str, source: str) -> None:
    """Комбинация: лист — пиксели, которые бумагой считает большинство прогонов (по их маскам листа)."""
    out = RUNS_DIR / name
    rows = []
    for frame in [f for f in read_sample() if not f.skip]:
        masks = []
        for run_name in run_names:
            path = RUNS_DIR / run_name / "zones" / (frame.key + "_paper.png")
            if path.exists():
                masks.append(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0)
        if len(masks) < 2:
            continue
        raw = np.sum(masks, axis=0) * 2 > len(masks)
        rows.append(
            finish_frame(
                out, frame, read_source_rgb(frame, source), raw, float("nan"), f"голосование {'+'.join(run_names)}"
            )
        )
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    log.info("голосование: %d кадров → %s", len(rows), out)


@main.command("corner-sheets")
@click.option("--run", "run_name", required=True, help="Имя папки прогона в runs/.")
@click.option("--source", type=click.Choice(["preview", "nofinger"]), default="nofinger", show_default=True)
@click.option("--crop", default=320, show_default=True, help="Сторона вырезки угла, px рабочего масштаба.")
@click.option("--tile", default=230, show_default=True, help="Сторона вырезки на листе, px.")
@click.option("--per-sheet", default=8, show_default=True, help="Кадров на лист.")
def corner_sheets(run_name: str, source: str, crop: int, tile: int, per_sheet: int) -> None:
    """Листы «четыре угла четырёхугольника крупно» по каждому кадру прогона (плюс миниатюра кадра) для отсмотра."""
    out_dir = RUNS_DIR / run_name / "corner_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [f for f in read_sample() if not f.skip]
    rows = []
    for frame in frames:
        base = RUNS_DIR / run_name / "zones" / frame.key
        paper = cv2.imread(str(base) + "_paper.png", cv2.IMREAD_GRAYSCALE) > 0
        info = json.loads(Path(str(base) + "_zone.json").read_text())
        zone_quad = (
            None if info["quad_norm"] is None else np.array(info["quad_norm"]) * [paper.shape[1], paper.shape[0]]
        )
        zone = build_zone(paper) if zone_quad is None else Zone(paper, zone_quad, 0, 0, 0, 0)
        canvas = zone_overlay(
            cv2.cvtColor(read_source_rgb(frame, source), cv2.COLOR_RGB2BGR), zone, [frame.key], None, 10**6
        )
        # Отрезаем шапку и легенду framed: берём центр холста по размеру маски.
        h, w = paper.shape
        top = (canvas.shape[0] - h) // 2
        body = canvas[top : top + h, :w] if canvas.shape[1] >= w else canvas
        thumb = cv2.resize(body, (int(tile * w / h), tile), interpolation=cv2.INTER_AREA)
        tiles = [thumb]
        corners = zone_quad if zone_quad is not None else np.array([[0, 0], [w, 0], [w, h], [0, h]])
        for cx, cy in corners:
            x0 = int(np.clip(cx - crop // 2, 0, w - crop))
            y0 = int(np.clip(cy - crop // 2, 0, h - crop))
            tiles.append(cv2.resize(body[y0 : y0 + crop, x0 : x0 + crop], (tile, tile), interpolation=cv2.INTER_AREA))
        row = np.hstack([np.pad(t, ((0, 0), (0, 4), (0, 0)), constant_values=255) for t in tiles])
        cv2.putText(row, frame.key, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(row, frame.key, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rows.append(np.pad(row, ((0, 4), (0, 0), (0, 0)), constant_values=255))
    for start in range(0, len(rows), per_sheet):
        chunk = rows[start : start + per_sheet]
        width = max(r.shape[1] for r in chunk)
        chunk = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in chunk]
        cv2.imwrite(
            str(out_dir / f"{start // per_sheet + 1:02d}.jpg"), np.vstack(chunk), [cv2.IMWRITE_JPEG_QUALITY, 88]
        )
    log.info("листы углов: %s", out_dir)


@main.command("compare-sheet")
@click.option("--runs", "run_names", multiple=True, required=True, help="Прогоны рядом (столбцы).")
@click.option("--key", "keys", multiple=True, required=True, help="Кадры «группа/имя» (строки).")
@click.option("--name", required=True, help="Имя файла листа в compare/.")
@click.option("--width", default=430, show_default=True, help="Ширина оверлея на листе, px.")
def compare_sheet(run_names: tuple[str, ...], keys: tuple[str, ...], name: str, width: int) -> None:
    """Лист «движки рядом»: строка — кадр, столбец — прогон (оверлеи целиком, с шапкой)."""
    rows = []
    for key in keys:
        tiles = []
        for run_name in run_names:
            img = cv2.imread(str(RUNS_DIR / run_name / "overlays" / (key.replace("/", "__") + ".jpg")))
            img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
            tiles.append(img)
        height = max(t.shape[0] for t in tiles)
        tiles = [np.pad(t, ((0, height - t.shape[0]), (0, 6), (0, 0)), constant_values=255) for t in tiles]
        rows.append(np.pad(np.hstack(tiles), ((0, 6), (0, 0), (0, 0)), constant_values=255))
    out = OUT_ROOT / "compare"
    out.mkdir(parents=True, exist_ok=True)
    width_max = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width_max - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    cv2.imwrite(str(out / f"{name}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    log.info("лист сравнения: %s", out / f"{name}.jpg")


def _copy_one(frame: Frame, run_name: str, source: str, guard: bool) -> str:
    """Прямоугольники копирования одного кадра (воркер пула ``copy-out``).

    Args:
        frame: Кадр.
        run_name: Прогон, чьи зоны брать.
        source: ``preview`` или ``nofinger``.
        guard: Брать защиту печати из ``runs/<run>/guard/`` (четырёхугольник с охватом и зазор без дорисовки);
            нужен прогон ``guard-eval``.

    Returns:
        Строка лога (размеры прямоугольников) или пустая строка, если зоны нет.
    """
    from research.newspaper_crop.fill import CopyMode, copy_rectangle

    out = RUNS_DIR / run_name / "copy"
    base = RUNS_DIR / run_name / "zones" / frame.key
    info = json.loads(Path(str(base) + "_zone.json").read_text())
    if info["quad_norm"] is None:
        return ""
    rgb = read_source_rgb(frame, source)
    h, w = rgb.shape[:2]
    paper_small = cv2.imread(str(base) + "_paper.png", cv2.IMREAD_GRAYSCALE)
    paper = cv2.resize(paper_small, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    quad = np.array(info["quad_norm"]) * [w, h]
    keep = None
    if guard:
        g = RUNS_DIR / run_name / "guard" / frame.key
        quad = np.array(json.loads(Path(str(g) + "_quad.json").read_text())["quad_norm"]) * [w, h]
        keep_small = cv2.imread(str(g) + "_keep.png", cv2.IMREAD_GRAYSCALE)
        keep = cv2.resize(keep_small, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    pics = {mode: copy_rectangle(bgr, paper, quad, mode, keep) for mode in CopyMode}
    target = out / frame.key
    target.parent.mkdir(parents=True, exist_ok=True)
    for mode, pic in pics.items():
        cv2.imwrite(f"{target}_{mode.value}.jpg", pic, [cv2.IMWRITE_JPEG_QUALITY, 92])
    tiles = [
        cv2.resize(p, (int(p.shape[1] * 1400 / p.shape[0]), 1400), interpolation=cv2.INTER_AREA) for p in pics.values()
    ]
    pair = np.hstack([np.pad(tiles[0], ((0, 0), (0, 12), (0, 0)), constant_values=255), tiles[1]])
    title = [f"{frame.key} — слева «по месту», справа выпрямлено; не-бумага дорисована"]
    cv2.imwrite(f"{target}_pair.jpg", framed(pair, title, []), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return f"{frame.key}: по месту {pics[CopyMode.IN_PLACE].shape[:2]}, выпрямлено {pics[CopyMode.WARP].shape[:2]}"


@main.command("copy-out")
@click.option("--run", "run_name", required=True, help="Прогон, чьи зоны брать.")
@click.option("--key", "keys", multiple=True, help="Кадры «группа/имя»; без опции — весь набор.")
@click.option("--source", type=click.Choice(["preview", "nofinger"]), default="nofinger", show_default=True)
@click.option("--jobs", default=12, show_default=True, help="Воркеров; 12 — JPEG-кодирование и запись на SSD.")
@click.option("--guard/--no-guard", default=True, show_default=True, help="Защита печати (нужен прогон guard-eval).")
def copy_out(run_name: str, keys: tuple[str, ...], source: str, jobs: int, guard: bool) -> None:
    """Прямоугольники копирования по зонам прогона (оба способа) на масштабе превью + склейка «по месту | выпрямлено»."""
    frames = [f for f in read_sample() if not f.skip and (not keys or f.key in keys)]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        futures = {pool.submit(_copy_one, f, run_name, source, guard): f for f in frames}
        for fut in as_completed(futures):
            try:
                line = fut.result()
                if line:
                    log.info("%s", line)
            except Exception as exc:  # noqa: BLE001 — сбой кадра не должен ронять прогон
                log.warning("выход %s: %s", futures[fut].key, exc)


def _sharpness_one(frame: Frame, run_name: str, source: str, with_raw: bool) -> list[dict]:
    """Замер резкости выпрямления одного кадра (воркер пула ``sharpness``).

    Args:
        frame: Кадр.
        run_name: Прогон, чьи четырёхугольники брать.
        source: ``preview`` или ``nofinger``.
        with_raw: Мерить ли варианты на проявленном RAW.

    Returns:
        Строки замеров: кадр, вариант, участок, DOM и крутизна до/после, наклон.
    """
    from research.newspaper_crop.sharpness import measure_frame

    info = json.loads((RUNS_DIR / run_name / "zones" / (frame.key + "_zone.json")).read_text())
    if info["quad_norm"] is None:
        return []
    preview = OUT_ROOT / "nofinger" / (frame.key + ".jpg") if source == "nofinger" else frame.preview_path
    scores = measure_frame(preview, frame.path if with_raw else None, np.array(info["quad_norm"]))
    rows = []
    for variant, items in scores.items():
        for i, s in enumerate(items):
            rows.append(
                dict(
                    key=frame.key,
                    variant=variant.value,
                    tile=i,
                    dom_src=s.dom_src,
                    dom_out=s.dom_out,
                    steep_src=s.steep_src,
                    steep_out=s.steep_out,
                    tilt_deg=s.tilt_deg,
                )
            )
    return rows


@main.command()
@click.option("--run", "run_name", default="sam2_box_nofinger", show_default=True, help="Прогон с четырёхугольниками.")
@click.option("--source", type=click.Choice(["preview", "nofinger"]), default="nofinger", show_default=True)
@click.option("--raw/--no-raw", "with_raw", default=True, show_default=True, help="Мерить и на проявленном RAW.")
@click.option("--jobs", default=6, show_default=True, help="Воркеров; 6 — проявка RAW сама многопоточна и читает NTFS.")
@click.option("--limit", default=0, show_default=True, help="Только первые N кадров (0 — все).")
def sharpness(run_name: str, source: str, with_raw: bool, jobs: int, limit: int) -> None:
    """Цена выпрямления для резкости текста: участки печати до и после гомографии → ``sharpness/tiles.csv`` и сводка."""
    frames = [f for f in read_sample() if not f.skip]
    frames = frames[:limit] if limit else frames
    rows = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        futures = {pool.submit(_sharpness_one, f, run_name, source, with_raw): f for f in frames}
        for fut in as_completed(futures):
            try:
                rows += fut.result()
            except Exception as exc:  # noqa: BLE001 — сбой кадра не должен ронять замер
                log.warning("резкость %s: %s", futures[fut].key, exc)
    out = OUT_ROOT / "sharpness"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "tiles.csv", index=False)
    df["dom_ratio"] = df.dom_out / df.dom_src
    df["steep_ratio"] = df.steep_out / df.steep_src
    # Сводка: сначала медиана по участкам внутри кадра, потом по кадрам — кадр с большим числом участков не перевешивает.
    per_frame = df.groupby(["variant", "key"])[["dom_ratio", "steep_ratio"]].median().reset_index()
    summary = per_frame.groupby("variant").agg(
        frames=("key", "nunique"),
        dom_med=("dom_ratio", "median"),
        dom_p10=("dom_ratio", lambda s: s.quantile(0.1)),
        steep_med=("steep_ratio", "median"),
        steep_p10=("steep_ratio", lambda s: s.quantile(0.1)),
    )
    summary["tiles"] = df.groupby("variant").size()
    summary.to_csv(out / "summary.csv")
    log.info("резкость: %d участков\n%s", len(df), summary.round(4).to_string())


def _spine_cut_one(frame: Frame, run_name: str, name: str, source: str) -> tuple[dict, list[dict]]:
    """Отрезка за складкой одного кадра (воркер пула ``spine-cut``): зона, оверлей итога и оверлей «трапеция + зона».

    Args:
        frame: Кадр.
        run_name: Прогон с трапециями листа.
        name: Прогон-итог.
        source: ``preview`` или ``nofinger``.

    Returns:
        (строка сводки, строки признаков сторон).
    """
    from research.newspaper_crop.spine_cut import cut_zone

    out = RUNS_DIR / name
    base = RUNS_DIR / run_name / "zones" / frame.key
    paper = cv2.imread(str(base) + "_paper.png", cv2.IMREAD_GRAYSCALE) > 0
    raw = cv2.imread(str(base) + "_raw.png", cv2.IMREAD_GRAYSCALE) > 0
    info = json.loads(Path(str(base) + "_zone.json").read_text())
    quad = None if info["quad_norm"] is None else np.array(info["quad_norm"]) * [paper.shape[1], paper.shape[0]]
    rgb = read_source_rgb(frame, source)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    detected = zone_of(paper, quad)
    result = cut_zone(bgr, detected)
    cut = [f for f in result.folds if f.accepted]
    cut_text = ", ".join(f"{f.side.value} {f.depth_min:.0f}–{f.depth_max:.0f} px" for f in cut) or "нет"
    row = finish_frame(
        out, frame, rgb, raw, float("nan"), f"{run_name} + отрезка за складкой", result.zone, result.lines,
        f"складки: {cut_text}",
    )  # fmt: skip
    row["detected_keystone_deg"] = detected.keystone_deg
    row["cut_sides"] = ",".join(f.side.value for f in cut)
    save_zone(out, frame, result.zone, raw)
    # Оверлей «трапеция листа + зона копирования» — рядом с трапециями, в прогоне-источнике.
    header = [
        f"{frame.key} — {run_name}: трапеция листа и зона копирования",
        f"трапеция {detected.keystone_deg:.1f}°, дорисовка {result.zone.fill_frac:.3f}; отрезано за складкой: {cut_text}",
    ]
    picture = zone_overlay(
        bgr, result.zone, header, read_finger_mask(frame, paper.shape), folds=result.lines, detected_quad=quad
    )
    zone_dir = RUNS_DIR / run_name / "overlays_zone"
    zone_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(zone_dir / (frame.key.replace("/", "__") + ".jpg")), picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
    sides = [
        dict(key=frame.key, side=f.side.value, accepted=f.accepted, reason=f.reason, depth_min=f.depth_min,
             depth_max=f.depth_max, support=f.support, ridge=f.ridge, ab_shift=f.ab_shift,
             beyond_level=f.beyond_level, beyond_ink=f.beyond_ink)
        for f in result.folds
    ]  # fmt: skip
    return row, sides


@main.command("spine-cut")
@click.option("--run", "run_name", default="sam2_box_nofinger", show_default=True, help="Прогон с зонами.")
@click.option("--name", default="spine_cut", show_default=True, help="Имя прогона-итога.")
@click.option("--source", type=click.Choice(["preview", "nofinger"]), default="nofinger", show_default=True)
@click.option("--jobs", default=14, show_default=True, help="Воркеров (CPU: выпрямление, складки, оверлеи).")
def spine_cut(run_name: str, name: str, source: str, jobs: int) -> None:
    """Отрезка за складкой корешка у каждой стороны выпрямленной полосы; заодно оверлеи «трапеция + зона копирования»
    в ``runs/<run>/overlays_zone/``."""
    out = RUNS_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    rows, sides = [], []
    frames = [f for f in read_sample() if not f.skip]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        futures = {pool.submit(_spine_cut_one, f, run_name, name, source): f for f in frames}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                row, frame_sides = fut.result()
                rows.append(row)
                sides += frame_sides
            except Exception as exc:  # noqa: BLE001 — сбой кадра не должен ронять прогон
                log.warning("отрезка %s: %s", futures[fut].key, exc)
            if i % 50 == 0:
                log.info("отрезка: %d / %d", i, len(frames))
    pd.DataFrame(rows).sort_values("key").to_csv(out / "summary.csv", index=False)
    pd.DataFrame(sides).sort_values(["key", "side"]).to_csv(out / "sides.csv", index=False)
    log.info(
        "отрезка за складкой: %d кадров, отрезано сторон %d → %s", len(rows), sum(s["accepted"] for s in sides), out
    )


def _guard_one(frame: Frame, run_name: str, source: str, guard_px: int, cut_sides: str) -> dict:
    """Меры защиты печати одного кадра (воркер пула ``guard-eval``): сохраняет зазор и пишет оверлей зазора.

    Args:
        frame: Кадр.
        run_name: Прогон с зонами (после отрезки за складкой).
        source: ``preview`` или ``nofinger``.
        guard_px: Ширина зазора, px рабочего масштаба.
        cut_sides: Стороны, отрезанные по складке (``left,right`` из ``summary.csv`` прогона), — охват их не двигает.

    Returns:
        Строка сводки: доля своей печати ближе зазора к дорисовке (было), подозрение «свои строки в дорисовке»,
        сдвиг стороны охватом, площадь зазора вне листа.
    """
    from research.newspaper_crop.text_guard import protect, side_index

    base = RUNS_DIR / run_name / "zones" / frame.key
    paper = cv2.imread(str(base) + "_paper.png", cv2.IMREAD_GRAYSCALE) > 0
    info = json.loads(Path(str(base) + "_zone.json").read_text())
    if info["quad_norm"] is None:
        return dict(key=frame.key)
    quad = np.array(info["quad_norm"]) * [paper.shape[1], paper.shape[0]]
    rgb = read_source_rgb(frame, source)
    zone = zone_of(paper, quad)
    frozen = tuple(side_index(quad, name) for name in cut_sides.split(",") if name)
    guard = protect(rgb, zone, guard_px, frozen)
    target = RUNS_DIR / run_name / "guard" / frame.key
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target) + "_keep.png", guard.keep.astype(np.uint8) * 255)
    (Path(str(target) + "_quad.json")).write_text(
        json.dumps(dict(quad_norm=(guard.zone.quad / [paper.shape[1], paper.shape[0]]).tolist()))
    )
    return dict(
        key=frame.key,
        print_near_fill=guard.print_fill_before,
        suspect_fill=guard.suspect_fill,
        expand_px=guard.expand_px,
        keep_frac=float(guard.keep.mean()),
        fill_before=zone.fill_frac,
        fill_after=guard.zone.fill_frac,
    )


@main.command("guard-eval")
@click.option("--run", "run_name", default="spine_cut_all", show_default=True, help="Прогон с зонами.")
@click.option("--source", type=click.Choice(["preview", "nofinger"]), default="nofinger", show_default=True)
@click.option("--guard-px", default=10, show_default=True, help="Зазор у своей печати, px рабочего масштаба (≈2 мм).")
@click.option("--jobs", default=14, show_default=True, help="Воркеров (CPU).")
def guard_eval(run_name: str, source: str, guard_px: int, jobs: int) -> None:
    """Защита своей печати: зазор и охват по зонам прогона → ``runs/<run>/guard/`` и ``guard.csv``."""
    frames = [f for f in read_sample() if not f.skip]
    # Стороны, отрезанные по складке, — из сводки прогона отрезки.
    summary = pd.read_csv(RUNS_DIR / run_name / "summary.csv", keep_default_na=False)
    cut = dict(zip(summary.key, summary.cut_sides))
    rows = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        futures = {pool.submit(_guard_one, f, run_name, source, guard_px, cut.get(f.key, "")): f for f in frames}
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception as exc:  # noqa: BLE001 — сбой кадра не должен ронять прогон
                log.warning("защита %s: %s", futures[fut].key, exc)
    df = pd.DataFrame(rows).sort_values("key")
    df.to_csv(RUNS_DIR / run_name / "guard.csv", index=False)
    log.info("защита печати: %d кадров\n%s", len(df), df.describe().round(4).to_string())


def _tilt_one(frame: Frame, run_name: str, source: str) -> list[dict]:
    """Расхождение сторон четырёхугольника с содержимым одного кадра (воркер пула ``tilt-candidates``).

    Args:
        frame: Кадр.
        run_name: Прогон с зонами.
        source: ``preview`` или ``nofinger``.

    Returns:
        Четыре строки (по стороне): семейство, угол против точки схода и против ближайших признаков, разбросы,
        опоры, доля стороны на рамке кадра, длина стороны.
    """
    from research.newspaper_crop.content_tilt import (
        border_share,
        content_segments,
        fit_vanishing_points,
        local_direction,
        side_tilts,
        signed_angle,
    )
    from research.newspaper_crop.text_guard import side_index

    base = RUNS_DIR / run_name / "zones" / frame.key
    paper = cv2.imread(str(base) + "_paper.png", cv2.IMREAD_GRAYSCALE) > 0
    info = json.loads(Path(str(base) + "_zone.json").read_text())
    if info["quad_norm"] is None:
        return []
    quad = np.array(info["quad_norm"]) * [paper.shape[1], paper.shape[0]]
    model = fit_vanishing_points(content_segments(read_source_rgb(frame, source), paper, quad))
    names = {side_index(quad, n): n for n in ("left", "right", "top", "bottom")}
    rows = []
    for t in side_tilts(model, quad):
        local, spread, count = local_direction(model, quad, t.index)
        rows.append(
            dict(
                key=frame.key,
                side=names.get(t.index, str(t.index)),
                index=t.index,
                family=t.family.value,
                delta_vp=t.delta_deg,
                sigma_vp=t.sigma_deg,
                support_vp=t.support,
                delta_local=float("nan") if local is None else signed_angle(t.mask_dir, local),
                spread_local=spread,
                count_local=count,
                border=border_share(quad, t.index, paper.shape),
                length=float(np.linalg.norm(quad[(t.index + 1) % 4] - quad[t.index])),
            )
        )
    return rows


@main.command("tilt-candidates")
@click.option("--run", "run_name", default="spine_cut_all", show_default=True, help="Прогон с зонами.")
@click.option("--source", type=click.Choice(["preview", "nofinger"]), default="nofinger", show_default=True)
@click.option("--jobs", default=14, show_default=True, help="Воркеров (CPU).")
def tilt_candidates(run_name: str, source: str, jobs: int) -> None:
    """Наклон сторон против содержимого по всем кадрам → ``runs/<run>/tilt_sides.csv`` (сырьё для отбора кандидатов)."""
    frames = [f for f in read_sample() if not f.skip]
    rows = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        futures = {pool.submit(_tilt_one, f, run_name, source): f for f in frames}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rows += fut.result()
            except Exception as exc:  # noqa: BLE001 — сбой кадра не должен ронять прогон
                log.warning("наклон %s: %s", futures[fut].key, exc)
            if i % 100 == 0:
                log.info("наклон: %d / %d", i, len(frames))
    df = pd.DataFrame(rows).sort_values(["key", "index"])
    df.to_csv(RUNS_DIR / run_name / "tilt_sides.csv", index=False)
    log.info("наклон сторон: %d строк → %s", len(df), RUNS_DIR / run_name / "tilt_sides.csv")


@main.command("tilt-export")
@click.option("--run", "run_name", default="spine_cut_all", show_default=True, help="Прогон с зонами и tilt_sides.csv.")
@click.option("--extra", multiple=True, help="Кадры пользователя «группа/имя».")
@click.option("--annotator", default="user", show_default=True, help="Пользователь CVAT для джобов.")
def tilt_export(run_name: str, extra: tuple[str, ...], annotator: str) -> None:
    """Отбор кадров под эталон наклона сторон и задача CVAT с полилиниями текущих сторон."""
    from research.newspaper_crop.tilt_cvat import PICK_CSV, export_task, pick_frames

    picks = pick_frames(pd.read_csv(RUNS_DIR / run_name / "tilt_sides.csv"), list(extra))
    picks.to_csv(PICK_CSV, index=False)
    log.info("отобрано кадров %d: %s", len(picks), picks.reason.value_counts().to_dict())
    log.info("задача CVAT %s", export_task(picks, run_name, annotator))


@main.command("tilt-import")
@click.option("--run", "run_name", default="spine_cut_all", show_default=True, help="Прогон с зонами.")
def tilt_import(run_name: str) -> None:
    """Забор эталона наклона сторон из CVAT → ``sets/tilt_gt.csv``."""
    from research.newspaper_crop.tilt_cvat import GT_CSV, import_task

    df = import_task(run_name)
    log.info("эталон: %d сторон, поправлено %d → %s", len(df), int(df.changed.sum()), GT_CSV)


if __name__ == "__main__":
    main()
