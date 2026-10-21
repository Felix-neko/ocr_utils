"""Команды стенда брака разворотов: ``features`` — общий проход по паку (смаз и корешок), ранги, листы для отсмотра."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
import threadpoolctl

from research.spread_defects.frames import PACK4_ROOT, Frame, iter_pack, read_rgb
from research.spread_defects.motion import MotionTiles, motion_tiles

# Выход стенда на SSD.
OUT_ROOT = Path("/mnt/hotstore/scan_processing/ph_pack4_defects")
# Разметка отсмотра стенда (в git).
SETS_DIR = Path(__file__).resolve().parent / "sets"


def _init_worker() -> None:
    """Инициализатор воркера: без hugepage и потоков BLAS/OpenCV (см. ``.claude/rules/gpu_and_pools.md``)."""
    np._core.multiarray._set_madvise_hugepage(False)
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)


def tiles_path(out_dir: Path, frame: Frame) -> Path:
    """Путь npz с плиточной сеткой кадра: ``<out>/tiles/<выпуск>/<имя>.npz``.

    Args:
        out_dir: Папка прогона ``motion``.
        frame: Кадр.

    Returns:
        Путь файла (папка не создаётся).
    """
    return out_dir / "tiles" / frame.issue / (frame.path.stem + ".npz")


def _motion_one(frame: Frame, out_dir: Path, rgb: np.ndarray | None = None) -> dict:
    """Считает плиточные признаки смаза одного кадра и пишет их в npz (воркер пула).

    Args:
        frame: Кадр.
        out_dir: Папка прогона.
        rgb: Уже прочитанный кадр (общий проход ``features``); None — прочитать здесь.

    Returns:
        Строка сводки: ключ, выпуск, номер, размер кадра, число текстовых плиток, время; ``error`` при сбое.
    """
    started = time.time()
    row = {"key": frame.key, "issue": frame.issue, "index": frame.index, "path": str(frame.path)}
    try:
        rgb = read_rgb(frame.path) if rgb is None else rgb
        tiles = motion_tiles(rgb)
    except Exception as error:  # битый файл не должен ронять прогон по паку
        return {**row, "error": repr(error)}
    target = tiles_path(out_dir, frame)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **{name: getattr(tiles, name) for name in MotionTiles.__dataclass_fields__})
    return {
        **row,
        "height": rgb.shape[0],
        "width": rgb.shape[1],
        "text_tiles": int(tiles.text.sum()),
        "seconds": round(time.time() - started, 2),
        "error": "",
    }


@click.group()
def main() -> None:
    """Стенд брака разворотов фото-скана (смаз, корешок, пропуски и дубли)."""


@main.command()
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=PACK4_ROOT)
@click.option("--only", multiple=True, help="Префикс ключа выпуска (``1985/02``, ``1979``); можно несколько.")
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), default=OUT_ROOT / "motion")
@click.option(
    "--jobs",
    default=8,
    show_default=True,
    help="Воркеров. Умолчание ниже 16: чтение ~10 МБ JPEG с NTFS-3G упирается в диск, а не в счёт.",
)
@click.option("--force", is_flag=True, help="Пересчитать кадры, у которых npz уже есть.")
def motion(root: Path, only: tuple[str, ...], out_dir: Path, jobs: int, force: bool) -> None:
    """Плиточные признаки смаза (относительный кепстр) по всем кадрам пака; сводка в ``frames.csv``."""
    run_pool(iter_pack(root, only), _motion_one, out_dir, out_dir / "frames.csv", jobs, force, tiles_path)


def load_tiles(path: Path) -> MotionTiles:
    """Читает плиточную сетку кадра из npz.

    Args:
        path: Файл, записанный командой ``motion``.

    Returns:
        ``MotionTiles``.
    """
    with np.load(path) as data:
        return MotionTiles(**{name: data[name] for name in MotionTiles.__dataclass_fields__})


def gutter_path(out_dir: Path, frame: Frame) -> Path:
    """Путь npz с профилями шага штрихов кадра: ``<out>/profiles/<выпуск>/<имя>.npz``.

    Args:
        out_dir: Папка прогона ``gutter``.
        frame: Кадр.

    Returns:
        Путь файла (папка не создаётся).
    """
    return out_dir / "profiles" / frame.issue / (frame.path.stem + ".npz")


def _gutter_one(frame: Frame, out_dir: Path, rgb: np.ndarray | None = None) -> dict:
    """Профили шага штрихов обеих полос кадра у сгиба (воркер пула): npz и строка сводки.

    Args:
        frame: Кадр.
        out_dir: Папка прогона.
        rgb: Уже прочитанный кадр (общий проход ``features``); None — прочитать здесь.

    Returns:
        Строка сводки: ключ, выпуск, номер, x сгиба посередине высоты, по сторонам ``L``/``R`` —
        медиана и минимум по зонам сжатия у края текста и медианный отступ текста от сгиба; ``error`` при сбое.
    """
    from research.spread_defects.gutter import LEFT, RIGHT, edge_compression, find_fold, side_profile

    started = time.time()
    row = {"key": frame.key, "issue": frame.issue, "index": frame.index, "path": str(frame.path)}
    try:
        rgb = read_rgb(frame.path) if rgb is None else rgb
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        fold, fold_ink = find_fold(gray)
        profiles = {side: side_profile(gray, fold, side) for side in (LEFT, RIGHT)}
    except Exception as error:  # битый файл не должен ронять прогон по паку
        return {**row, "error": repr(error)}
    arrays = {"fold": fold, "height": gray.shape[0], "width": gray.shape[1]}
    for side, profile in profiles.items():
        arrays.update({f"{side}_offsets": profile.offsets, f"{side}_pitch": profile.pitch})
        arrays.update({f"{side}_rows": profile.rows, f"{side}_base": profile.base})
        edge = edge_compression(profile)
        with np.errstate(all="ignore"):
            row[f"{side}_ratio_med"] = float(np.nanmedian(edge.edge_ratio))
            row[f"{side}_ratio_min"] = (
                float(np.nanmin(edge.edge_ratio)) if np.isfinite(edge.edge_ratio).any() else np.nan
            )
            row[f"{side}_margin"] = float(np.nanmedian(edge.margin))
            row[f"{side}_base"] = float(np.nanmedian(profile.base))
    target = gutter_path(out_dir, frame)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
    row["fold_x"] = float(np.polyval(fold, gray.shape[0] / 2))
    row["fold_ink"] = round(fold_ink, 3)
    row["width"] = gray.shape[1]
    row["seconds"] = round(time.time() - started, 2)
    row["error"] = ""
    return row


def run_pool(frames: list[Frame], worker, out_dir: Path, summary_csv: Path, jobs: int, force: bool, done) -> None:
    """Прогон воркера по кадрам пулом процессов с досчётом и периодической записью сводки.

    Args:
        frames: Все кадры прогона.
        worker: Функция ``(frame, out_dir) -> dict`` (верхнего уровня модуля — её пиклит пул).
        out_dir: Папка прогона.
        summary_csv: Сводка по кадрам (дописывается, пишется каждые 200 кадров).
        jobs: Воркеров.
        force: Пересчитать всё.
        done: Функция ``(out_dir, frame) -> Path`` — файл, наличие которого значит «кадр посчитан».
    """
    todo = [f for f in frames if force or not done(out_dir, f).exists()]
    click.echo(f"кадров {len(frames)}, считать {len(todo)}, воркеров {jobs}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(summary_csv).to_dict("records") if summary_csv.exists() and not force else []
    done_keys = {r["key"] for r in rows}
    started = time.time()
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        futures = [pool.submit(worker, f, out_dir) for f in todo]
        for n, future in enumerate(as_completed(futures), 1):
            row = future.result()
            if row["key"] not in done_keys:
                rows.append(row)
            if row.get("error"):
                click.echo(f"сбой {row['key']}: {row['error']}")
            if n % 200 == 0 or n == len(futures):
                pd.DataFrame(rows).to_csv(summary_csv, index=False)
                rate = n / (time.time() - started)
                click.echo(f"{n}/{len(futures)}  {rate:.1f} кадр/с  осталось ~{(len(futures) - n) / rate / 60:.0f} мин")
    pd.DataFrame(rows).to_csv(summary_csv, index=False)


@main.command()
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=PACK4_ROOT)
@click.option("--only", multiple=True, help="Префикс ключа выпуска (``1985/02``, ``1979``); можно несколько.")
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), default=OUT_ROOT / "gutter")
@click.option("--jobs", default=8, show_default=True, help="Воркеров (диск NTFS-3G — узкое место).")
@click.option("--force", is_flag=True, help="Пересчитать кадры, у которых npz уже есть.")
def gutter(root: Path, only: tuple[str, ...], out_dir: Path, jobs: int, force: bool) -> None:
    """Профили шага штрихов у сгиба (сжатие глифов у корешка) по всем кадрам пака; сводка в ``frames.csv``."""
    run_pool(iter_pack(root, only), _gutter_one, out_dir, out_dir / "frames.csv", jobs, force, gutter_path)


@main.command("motion-rank")
@click.option("--run-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=OUT_ROOT / "motion")
def motion_rank(run_dir: Path) -> None:
    """Оценки смаза кадров из плиточных сеток прогона ``motion``: ``scores.csv`` по убыванию зональной оценки."""
    from research.spread_defects.motion import MidtoneTiles, frame_vs_issue, midtone_score, motion_score

    frames = pd.read_csv(run_dir / "frames.csv")
    midtone_dir = run_dir.parent / "midtone"
    stem_by_key = dict(zip(frames.key, frames.path.map(lambda p: Path(p).stem)))
    rows = []
    for issue, group in frames.groupby("issue"):
        tiles = {}
        for key, stem in zip(group.key, group.path.map(lambda p: Path(p).stem)):
            path = run_dir / "tiles" / issue / f"{stem}.npz"
            if path.exists():
                tiles[key] = load_tiles(path)
        refs = [t.ref for t in tiles.values() if t.text.sum() >= 4]
        issue_ref = np.median(np.stack(refs).astype(np.float32), axis=0) if len(refs) >= 3 else None
        for key, t in tiles.items():
            score = motion_score(t)
            row = {"key": key, "issue": issue, **score.__dict__}
            if issue_ref is not None and t.text.sum() >= 4:
                z, dx, dy, loss = frame_vs_issue(t.ref, issue_ref)
                row.update({"frame_z": z, "frame_dx": dx, "frame_dy": dy, "frame_hf_loss": loss})
            # Полутоновый ореол — из второго прохода (если он уже был).
            mid_path = midtone_dir / issue / f"{stem_by_key[key]}.npz"
            if mid_path.exists():
                with np.load(mid_path) as data:
                    mid = MidtoneTiles(**{n: data[n] for n in MidtoneTiles.__dataclass_fields__})
                row.update({f"mid_{k}": v for k, v in midtone_score(mid, score.row, score.col).__dict__.items()})
            rows.append(row)
        click.echo(f"{issue}: {len(tiles)}")
    scores = pd.DataFrame(rows).merge(frames[["key", "path", "index"]], on="key")
    scores = scores.sort_values("agree_peak", ascending=False)
    scores.to_csv(run_dir / "scores.csv", index=False)
    click.echo(f"записано {len(scores)} → {run_dir / 'scores.csv'}")


@main.command("motion-sheets")
@click.option("--run-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=OUT_ROOT / "motion")
@click.option("--start", default=0, show_default=True, help="С какого места рейтинга.")
@click.option("--count", default=48, show_default=True, help="Сколько кадров.")
@click.option("--per-sheet", default=8, show_default=True)
@click.option("--sort", "sort_by", default="agree_peak", show_default=True, help="Колонка ``scores.csv`` для порядка.")
@click.option(
    "--keys", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="CSV с колонкой key — взять эти кадры."
)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
def motion_sheets(
    run_dir: Path, start: int, count: int, per_sheet: int, sort_by: str, keys: Path | None, out_dir: Path | None
) -> None:
    """Листы крупов 3×3 плитки вокруг самой смазанной зоны кадра, по рейтингу или по списку ключей."""
    from research.spread_defects.motion import TILE
    from research.spread_defects.sheets import Crop, contact_sheet

    scores = pd.read_csv(run_dir / "scores.csv").sort_values(sort_by, ascending=False).reset_index(drop=True)
    scores["rank"] = scores.index + 1
    if keys:
        chosen = scores[scores.key.isin(pd.read_csv(keys).key)]
    else:
        chosen = scores.iloc[start : start + count]
    out_dir = out_dir or run_dir / "sheets" / f"{sort_by}_{start:05d}"
    crops = []
    for r in chosen.itertuples():
        # Круп 640 px вокруг центра пиковой плитки — почти 1:1 на листе: двоение в 4–5 px иначе не видно.
        cx, cy = int((r.col + 0.5) * TILE), int((r.row + 0.5) * TILE)
        box = (cx - 320, cy - 320, cx + 320, cy + 320)
        label = f"#{r.rank} {r.key[:7]} {Path(r.path).stem}  a{r.agree_peak:.1f} z{r.zone_share:.2f} ({r.shift_dx},{r.shift_dy})"
        crops.append(Crop(Path(r.path), box, label))
    for n in range(0, len(crops), per_sheet):
        path = contact_sheet(
            crops[n : n + per_sheet], out_dir / f"sheet_{n // per_sheet:02d}.jpg", columns=4, cell=(620, 620)
        )
        click.echo(path)


def load_midtone(path: Path) -> "MidtoneTiles":
    """Читает сетку полутонов кадра из npz второго прохода.

    Args:
        path: Файл ``midtone/<выпуск>/<кадр>.npz``.

    Returns:
        ``MidtoneTiles``.
    """
    from research.spread_defects.motion import MidtoneTiles

    with np.load(path) as data:
        return MidtoneTiles(**{name: data[name] for name in MidtoneTiles.__dataclass_fields__})


def _print_mask_one(path: str, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Плитки полиграфии кадра (растр, цвет) и высота букв среди плиток маски — воркер пула ``motion-zones``.

    Args:
        path: Файл кадра.
        mask: bool-сетка плиток, которые проверять.

    Returns:
        ``print_mask``: сетка плиток полиграфии и сетка высоты букв; при сбое чтения — пустая и NaN (кадр остаётся
        без отсева).
    """
    from research.spread_defects.blur_zones import print_mask

    try:
        return print_mask(read_rgb(Path(path)), mask)
    except Exception:  # битый файл не должен ронять прогон
        return np.zeros(mask.shape, bool), np.full(mask.shape, np.nan, np.float32)


@main.command("motion-zones")
@click.option("--out-root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=OUT_ROOT)
@click.option(
    "--jobs",
    default=8,
    show_default=True,
    help="Воркеров отсева полиграфии (читают полный кадр только у кадров с зонами; NTFS-3G — узкое место).",
)
def motion_zones(out_root: Path, jobs: int) -> None:
    """Карта смаза по всему кадру: зоны ореола и двоения — ``motion/zones.csv`` и ``motion/zone_frames.csv``.

    Читает сетки кепстра (``motion/tiles``), полутонов (``midtone/``) и x сгиба из ``glyphs.csv`` (второй проход).
    Кадры, где нашлись зоны, читаются целиком: плитки растра и цветных плашек убираются из карты, зоны
    пересчитываются; зоны-заголовки (кегль крупнее набора вокруг, ``heading_ratio``) отбрасываются. ``zones.csv`` — строка на зону; ``zone_frames.csv`` — строка на кадр (число зон, сильнейшая
    зона, ореол кадра к выпуску, убрано плиток полиграфии), по убыванию ``mid_peak``.
    """
    from research.spread_defects.blur_zones import (
        blur_zones,
        heading_ratio,
        is_heading,
        issue_midtone_level,
        zone_mask,
        zone_maps,
    )

    glyphs = pd.read_csv(out_root / "glyphs.csv", usecols=["key", "issue", "path", "fold_x"])
    # Проход 1: карты всех кадров без отсева; уровень полутонов — по выпуску.
    frames = []
    for issue, group in glyphs.groupby("issue"):
        loaded = []
        for r in group.itertuples():
            stem = Path(r.path).stem
            tiles_file = out_root / "motion" / "tiles" / issue / f"{stem}.npz"
            mid_file = out_root / "midtone" / issue / f"{stem}.npz"
            if tiles_file.exists() and mid_file.exists():
                loaded.append((r, load_tiles(tiles_file), load_midtone(mid_file)))
        masks = [zone_mask(mid, r.fold_x) for r, _, mid in loaded]
        level = issue_midtone_level([mid for _, _, mid in loaded], masks)
        frames.extend((r, tiles, mid, level) for r, tiles, mid in loaded)
        click.echo(f"{issue}: кадров {len(loaded)}, уровень полутонов {level:.3f}")
    first = [zone_maps(tiles, mid, r.fold_x, level) for r, tiles, mid, level in frames]
    candidates = [n for n, maps in enumerate(first) if blur_zones(maps, frames[n][0].fold_x)]
    click.echo(f"кадров с зонами до отсева полиграфии: {len(candidates)}")

    # Проход 2: у кандидатов — плитки полиграфии из полного кадра.
    excluded, heights = {}, {}
    ctx = get_context("forkserver")
    with ProcessPoolExecutor(jobs, mp_context=ctx, initializer=_init_worker) as pool:
        futures = {pool.submit(_print_mask_one, frames[n][0].path, first[n].mask): n for n in candidates}
        for future in as_completed(futures):
            excluded[futures[future]], heights[futures[future]] = future.result()

    zone_rows, frame_rows = [], []
    for n, (r, tiles, mid, level) in enumerate(frames):
        maps = zone_maps(tiles, mid, r.fold_x, level, excluded.get(n))
        zones = blur_zones(maps, r.fold_x)
        # Заголовки: зона с кеглем заметно крупнее набора вокруг на той же полосе — ореол от кегля, не смаз.
        flags = [n in heights and is_heading(heading_ratio(z, heights[n], r.fold_x)) for z in zones]
        headings = sum(flags)
        zones = [z for z, flag in zip(zones, flags) if not flag]
        base = {"key": r.key, "issue": r.issue, "path": r.path}
        for number, zone in enumerate(zones, start=1):
            x0, y0, x1, y1 = zone.box_px()
            fields = {k: v for k, v in zone.__dict__.items() if k not in ("side", "band")}
            zone_rows.append(
                {**base, "zone": number, **fields, "side": zone.side.value, "band": zone.band.value}
                | {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
            )
        peak = np.nanmax(maps.mid_rel) if np.isfinite(maps.mid_rel).any() else np.nan
        frame_rows.append(
            {
                **base,
                "zones": len(zones),
                "mid_peak": float(peak),
                "zone_tiles": int(sum(z.tiles for z in zones)),
                "map_tiles": int(maps.mask.sum()),
                "print_tiles": int(excluded[n].sum()) if n in excluded else 0,
                "heading_zones": headings,
                "frame_rel": maps.frame_rel,
                "issue_level": level,
            }
        )
    zones_df = pd.DataFrame(zone_rows)
    frames_df = pd.DataFrame(frame_rows).sort_values("mid_peak", ascending=False)
    zones_df.to_csv(out_root / "motion" / "zones.csv", index=False)
    frames_df.to_csv(out_root / "motion" / "zone_frames.csv", index=False)
    click.echo(f"кадров {len(frames_df)}, с зонами {int((frames_df.zones > 0).sum())}, зон {len(zones_df)}")


@main.command("motion-zone-sheets")
@click.option("--out-root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=OUT_ROOT)
@click.option(
    "--min-peak",
    default=0.0,
    show_default=True,
    help="Кадры с ``mid_peak`` не ниже (если без --keys); по умолчанию — все кадры с зонами.",
)
@click.option(
    "--max-frame-rel",
    default=1.8,
    show_default=True,
    help="Пропускать кадры с ореолом всего кадра выше " "(обложки и цветные вклейки: полутона там везде).",
)
@click.option(
    "--keys",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="CSV с колонкой key — взять эти кадры; необязательная колонка label (например, «тяжёлый смаз») идёт в шапку.",
)
@click.option("--max-zones", default=4, show_default=True, help="Крупов на кадр (сильнейшие зоны).")
@click.option("--jobs", default=8, show_default=True, help="Воркеров (чтение JPEG с NTFS-3G — узкое место).")
@click.option(
    "--plain-names",
    is_flag=True,
    help="Имена файлов как у папок симлинков (``1980_11_IMG_0847.jpg``), без места в рейтинге.",
)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
def motion_zone_sheets(
    out_root: Path,
    min_peak: float,
    max_frame_rel: float,
    keys: Path | None,
    max_zones: int,
    jobs: int,
    plain_names: bool,
    out_dir: Path | None,
) -> None:
    """Лист на кадр (оверлей): весь кадр с зонами смаза поверх и крупы 1:1 у максимума каждой зоны — для отсмотра."""
    from research.spread_defects.zone_sheets import render_zone_sheet

    frames = pd.read_csv(out_root / "motion" / "zone_frames.csv")
    zones = pd.read_csv(out_root / "motion" / "zones.csv")
    labels = {}
    if keys:
        wanted = pd.read_csv(keys)
        chosen = frames[frames.key.isin(wanted.key)]
        if "label" in wanted.columns:
            labels = dict(zip(wanted.key, wanted.label))
    else:
        chosen = frames[(frames.mid_peak >= min_peak) & (frames.frame_rel <= max_frame_rel) & (frames.zones > 0)]
    chosen = chosen.sort_values("mid_peak", ascending=False).reset_index(drop=True)
    out_dir = out_dir or out_root / "motion" / "zone_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    glyphs = pd.read_csv(out_root / "glyphs.csv", usecols=["key", "fold_x"]).set_index("key").fold_x
    tasks = []
    for rank, r in enumerate(chosen.itertuples(), start=1):
        frame_zones = zones[zones.key == r.key].sort_values("zone").head(max_zones)
        if plain_names:
            # Год и номер выпуска без хвоста папки («оплачено…»), как в именах симлинков.
            name = f"{r.issue[:7].replace('/', '_')}_{Path(r.path).stem}.jpg"
        else:
            name = f"{rank:03d}_{r.issue.replace('/', '_').replace(' ', '_')}_{Path(r.path).stem}.jpg"
        tasks.append(
            (
                r._asdict(),
                frame_zones.to_dict("records"),
                float(glyphs.get(r.key, np.nan)),
                out_dir / name,
                labels.get(r.key),
            )
        )
    ctx = get_context("forkserver")
    with ProcessPoolExecutor(jobs, mp_context=ctx, initializer=_init_worker) as pool:
        futures = [pool.submit(render_zone_sheet, *task) for task in tasks]
        for future in as_completed(futures):
            click.echo(future.result())
    click.echo(f"листов {len(tasks)} → {out_dir}")


def features_done(out_root: Path, frame: Frame) -> Path:
    """Файл-признак «кадр посчитан общим проходом»: npz корешка (пишется вторым, после смаза).

    Args:
        out_root: Корень выхода стенда.
        frame: Кадр.

    Returns:
        Путь npz профилей корешка.
    """
    return gutter_path(out_root / "gutter", frame)


def _features_one(frame: Frame, out_root: Path) -> dict:
    """Общий проход по кадру: одно чтение с медленного диска, признаки смаза и корешка (воркер пула).

    Args:
        frame: Кадр.
        out_root: Корень выхода стенда (``motion/``, ``gutter/``).

    Returns:
        Сводка кадра: поля смаза и поля корешка (у вторых — префикс ``g_`` для служебных полей).
    """
    try:
        rgb = read_rgb(frame.path)
    except Exception as error:  # битый файл не должен ронять прогон по паку
        return {
            "key": frame.key,
            "issue": frame.issue,
            "index": frame.index,
            "path": str(frame.path),
            "error": repr(error),
        }
    motion_row = _motion_one(frame, out_root / "motion", rgb)
    gutter_row = _gutter_one(frame, out_root / "gutter", rgb)
    shared = {"key", "issue", "index", "path", "width"}
    merged = dict(motion_row)
    merged.update(
        {(k if k not in {"seconds", "error"} else f"g_{k}"): v for k, v in gutter_row.items() if k not in shared}
    )
    return merged


@main.command()
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=PACK4_ROOT)
@click.option("--only", multiple=True, help="Префикс ключа выпуска (``1985/02``, ``1979``); можно несколько.")
@click.option("--out-root", type=click.Path(file_okay=False, path_type=Path), default=OUT_ROOT)
@click.option("--jobs", default=8, show_default=True, help="Воркеров (диск NTFS-3G — узкое место).")
@click.option("--force", is_flag=True, help="Пересчитать кадры, у которых npz уже есть.")
def features(root: Path, only: tuple[str, ...], out_root: Path, jobs: int, force: bool) -> None:
    """Общий проход: признаки смаза и корешка за одно чтение кадра; сводки ``motion/frames.csv``, ``gutter/frames.csv``."""
    summary = out_root / "features.csv"
    run_pool(iter_pack(root, only), _features_one, out_root, summary, jobs, force, features_done)
    merged = pd.read_csv(summary)
    for sub in ("motion", "gutter"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_root / sub / "frames.csv", index=False)


def pages_path(out_dir: Path, frame: Frame) -> Path:
    """Путь json с числовыми токенами колонтитула кадра: ``<out>/tokens/<выпуск>/<имя>.json``.

    Args:
        out_dir: Папка прогона ``pages``.
        frame: Кадр.

    Returns:
        Путь файла (папка не создаётся).
    """
    return out_dir / "tokens" / frame.issue / (frame.path.stem + ".json")


def _pages_one(frame: Frame, out_dir: Path, place: str = "header") -> dict:
    """Номера страниц обеих полос кадра из колонтитула (воркер пула): json токенов и строка сводки.

    Args:
        frame: Кадр.
        out_dir: Папка прогона.
        place: Где номер на полосе — значение ``NumberPlace`` (``header`` — верхний колонтитул, ``footer`` — нижний).

    Returns:
        Строка: ключ, выпуск, номер кадра, номера левой/правой полосы и их уверенность, x сгиба (полный масштаб).
    """
    import json

    from research.spread_defects.sequence import NumberPlace, read_numbers

    row = {"key": frame.key, "issue": frame.issue, "index": frame.index, "path": str(frame.path)}
    try:
        left, right, tokens, fold_x = read_numbers(read_rgb(frame.path, reduce=2), place=NumberPlace(place))
    except Exception as error:  # битый файл не должен ронять прогон по паку
        return {**row, "error": repr(error)}
    target = pages_path(out_dir, frame)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([t.__dict__ for t in tokens], ensure_ascii=False))
    return {
        **row,
        "left": left.value if left else None,
        "left_conf": left.confidence if left else None,
        "right": right.value if right else None,
        "right_conf": right.confidence if right else None,
        "fold_x": fold_x * 2,
        "error": "",
    }


@main.command()
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=PACK4_ROOT)
@click.option("--only", multiple=True, help="Префикс ключа выпуска (``1985/02``, ``1979``); можно несколько.")
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), default=OUT_ROOT / "pages")
@click.option(
    "--jobs", default=12, show_default=True, help="Воркеров (tesseract однопоточный, чтение — половина кадра)."
)
@click.option("--force", is_flag=True, help="Перечитать кадры, у которых json уже есть.")
@click.option(
    "--place",
    type=click.Choice(["header", "footer"]),
    default="header",
    show_default=True,
    help="Где номер страницы: в верхнем (header) или нижнем (footer) колонтитуле.",
)
def pages(root: Path, only: tuple[str, ...], out_dir: Path, jobs: int, force: bool, place: str) -> None:
    """Номера страниц из колонтитула по всем кадрам пака (tesseract, верхняя или нижняя полоса); сводка ``frames.csv``."""
    from functools import partial

    worker = partial(_pages_one, place=place)
    run_pool(iter_pack(root, only), worker, out_dir, out_dir / "frames.csv", jobs, force, pages_path)


@main.command()
@click.option("--run-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=OUT_ROOT / "pages")
@click.option("--confirm", default=2, show_default=True, help="Сколько следующих кадров подтверждают новое смещение.")
def sequence(run_dir: Path, confirm: int) -> None:
    """Разрывы ряда номеров по выпускам (пропуски, повторы): ``events.csv`` и ``issues.csv``."""
    from research.spread_defects.sequence import chosen_offsets, offset_candidates, offset_runs, sequence_events

    frames = pd.read_csv(run_dir / "frames.csv").sort_values(["issue", "index", "key"])
    events, issues = [], []
    for issue, group in frames.groupby("issue"):
        group = group.reset_index(drop=True)
        candidates = [
            offset_candidates(k, None if pd.isna(l) else int(l), None if pd.isna(r) else int(r))
            for k, (l, r) in enumerate(zip(group.left, group.right))
        ]
        found = sequence_events(candidates)
        for event in found:
            events.append(
                {
                    "issue": issue,
                    "kind": event.kind.value,
                    "frame_before": group.key[event.before],
                    "frame_after": group.key[event.after],
                    "number_before": event.number_before,
                    "number_after": event.number_after,
                    "spreads": event.missing_spreads,
                }
            )
        # Номера только из участков (не одиночек): обложка и мусорные числа на краях выпуска не в счёт.
        runs = offset_runs(chosen_offsets(candidates))
        numbered = [run.offset + 2 * k for run in runs for k in run.frames]
        issues.append(
            {
                "issue": issue,
                "frames": len(group),
                "numbered": len(numbered),
                "first_page": min(numbered) if numbered else None,
                "last_page": max(numbered) + 1 if numbered else None,
                "events": len(found),
            }
        )
    pd.DataFrame(events).to_csv(run_dir / "events.csv", index=False)
    pd.DataFrame(issues).to_csv(run_dir / "issues.csv", index=False)
    click.echo(f"выпусков {len(issues)}, событий {len(events)} → {run_dir / 'events.csv'}")


def gutter_scores(frames: pd.DataFrame) -> pd.DataFrame:
    """Оценка корешка по кадрам: худшая сторона по медиане зон сжатия у края текста.

    Args:
        frames: Сводка прогона (``gutter/frames.csv`` или ``features.csv``).

    Returns:
        Те же строки с колонками ``gutter_score`` (меньше — сильнее сжатие) и ``gutter_side``, по возрастанию оценки.
    """
    scored = frames.copy()
    scored["gutter_score"] = scored[["L_ratio_med", "R_ratio_med"]].min(axis=1)
    scored["gutter_side"] = np.where(scored.L_ratio_med <= scored.R_ratio_med, "L", "R")
    return scored.sort_values("gutter_score").reset_index(drop=True)


@main.command("gutter-sheets")
@click.option(
    "--frames-csv", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=OUT_ROOT / "features.csv"
)
@click.option("--start", default=0, show_default=True, help="С какого места рейтинга (по возрастанию оценки).")
@click.option("--count", default=24, show_default=True)
@click.option("--per-sheet", default=6, show_default=True)
@click.option(
    "--keys", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="CSV с колонкой key — взять эти кадры."
)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), default=None)
def gutter_sheets(
    frames_csv: Path, start: int, count: int, per_sheet: int, keys: Path | None, out_dir: Path | None
) -> None:
    """Листы крупов у сгиба (±450 px, середина высоты, сгиб красной линией) по рейтингу сжатия или списку ключей."""
    from research.spread_defects.sheets import Crop, contact_sheet

    scored = gutter_scores(pd.read_csv(frames_csv))
    scored["rank"] = scored.index + 1
    chosen = scored[scored.key.isin(pd.read_csv(keys).key)] if keys else scored.iloc[start : start + count]
    out_dir = out_dir or frames_csv.parent / "gutter" / "sheets" / f"rank_{start:05d}"
    crops = []
    for r in chosen.itertuples():
        y0 = int(r.height * 0.42)
        label = f"#{r.rank} {r.key[:7]} {Path(r.path).stem[:8]}  L {r.L_ratio_med:.2f} R {r.R_ratio_med:.2f}"
        crops.append(Crop(Path(r.path), (int(r.fold_x - 450), y0, int(r.fold_x + 450), y0 + 600), label, (r.fold_x,)))
    for n in range(0, len(crops), per_sheet):
        path = contact_sheet(
            crops[n : n + per_sheet], out_dir / f"sheet_{n // per_sheet:02d}.jpg", columns=2, cell=(900, 600)
        )
        click.echo(path)


def _pass2_one(frame: Frame, out_root: Path) -> dict:
    """Второй проход по кадру: буквы у сгиба (сжатие у корешка, мера по глифам) и номера страниц (воркер пула).

    Args:
        frame: Кадр.
        out_root: Корень выхода стенда (``pages/tokens/…json`` пишется здесь).

    Returns:
        Сводка: x сгиба и доля «сгиб режет текст»; по сторонам — сжатие у края, контроль, край текста, буквы;
        номера левой и правой полосы с уверенностью.
    """
    import json

    from research.spread_defects.gutter import LEFT, RIGHT, ZONES, find_fold, glyph_components, glyph_edge
    from research.spread_defects.motion import MidtoneTiles, midtone_tiles
    from research.spread_defects.sequence import read_numbers

    row = {"key": frame.key, "issue": frame.issue, "index": frame.index, "path": str(frame.path)}
    try:
        rgb = read_rgb(frame.path)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        fold, cut = find_fold(gray)
        cx, cy, w, h, glyph_h = glyph_components(gray)
        fold_x = float(np.polyval(fold, gray.shape[0] / 2))
        half = cv2.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2), interpolation=cv2.INTER_AREA)
        left, right, tokens, _ = read_numbers(half, fold_x / 2)
        midtones = midtone_tiles(rgb)
    except Exception as error:  # битый файл не должен ронять прогон по паку
        return {**row, "error": repr(error)}
    row.update({"height": gray.shape[0], "width": gray.shape[1], "fold_x": fold_x, "fold_ink": round(cut, 3)})
    row["glyph_h"] = glyph_h
    for side in (LEFT, RIGHT):
        edge = glyph_edge(cx, cy, w, h, glyph_h, fold, side, gray.shape[0])
        if edge is None:
            continue
        row.update(
            {
                f"{side}_near": edge.near_ratio,
                f"{side}_next": edge.next_ratio,
                f"{side}_edge": edge.edge,
                f"{side}_glyphs": edge.glyphs,
                f"{side}_near_glyphs": edge.near_glyphs,
            }
        )
        row.update({f"{side}_zone{z}": edge.zone_ratio[z] for z in range(ZONES)})
    target = pages_path(out_root / "pages", frame)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps([t.__dict__ for t in tokens], ensure_ascii=False))
    # Карта полутонов пишется последней — её наличие и есть «кадр прошёл второй проход».
    mid_target = pass2_done(out_root, frame)
    mid_target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(mid_target, **{n: getattr(midtones, n) for n in MidtoneTiles.__dataclass_fields__})
    row.update(
        {
            "left": left.value if left else None,
            "left_conf": left.confidence if left else None,
            "right": right.value if right else None,
            "right_conf": right.confidence if right else None,
            "error": "",
        }
    )
    return row


def pass2_done(out_root: Path, frame: Frame) -> Path:
    """Файл-признак «кадр прошёл второй проход»: npz карты полутонов (пишется последним).

    Args:
        out_root: Корень выхода стенда.
        frame: Кадр.

    Returns:
        Путь ``<out_root>/midtone/<выпуск>/<кадр>.npz``.
    """
    return out_root / "midtone" / frame.issue / (frame.path.stem + ".npz")


@main.command()
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=PACK4_ROOT)
@click.option("--only", multiple=True, help="Префикс ключа выпуска (``1985/02``, ``1979``); можно несколько.")
@click.option("--out-root", type=click.Path(file_okay=False, path_type=Path), default=OUT_ROOT)
@click.option("--jobs", default=12, show_default=True, help="Воркеров (диск NTFS-3G и однопоточный tesseract).")
@click.option("--force", is_flag=True, help="Пересчитать кадры, уже прошедшие проход.")
def pass2(root: Path, only: tuple[str, ...], out_root: Path, jobs: int, force: bool) -> None:
    """Второй проход: буквы у сгиба (``glyphs.csv``) и номера страниц (``pages/frames.csv``) за одно чтение кадра."""
    summary = out_root / "glyphs.csv"
    run_pool(iter_pack(root, only), _pass2_one, out_root, summary, jobs, force, pass2_done)
    (out_root / "pages").mkdir(parents=True, exist_ok=True)
    pd.read_csv(summary).to_csv(out_root / "pages" / "frames.csv", index=False)


def fold_midtone(mid_path: Path, fold_x: float) -> tuple[float, float]:
    """Расфокус у сгиба: доля полутонов в плитках основного набора у сгиба к медиане кадра, по сторонам.

    Args:
        mid_path: npz карты полутонов кадра (второй проход).
        fold_x: x сгиба посередине высоты, px полного кадра.

    Returns:
        (левая, правая) — отношение медианы доли полутонов плиток, центр которых не дальше ``FOLD_TILES`` плиток
        от сгиба, к медиане по кадру; NaN, если плиток мало.
    """
    from research.spread_defects.motion import BODY_INK, BODY_LINES, TILE, MidtoneTiles

    with np.load(mid_path) as data:
        mid = MidtoneTiles(**{n: data[n] for n in MidtoneTiles.__dataclass_fields__})
    # У сгиба плитки часто не «внутренние» (сосед — тень корешка), поэтому здесь без interior_mask.
    body = mid.text & (mid.ink >= BODY_INK) & (mid.lines <= BODY_LINES)
    if body.sum() < 10:
        return np.nan, np.nan
    frame = float(np.median(mid.mid[body]))
    centers = (np.arange(body.shape[1]) + 0.5) * TILE - fold_x
    out = []
    for side in (-1, 1):
        near_cols = (np.sign(centers) == side) & (np.abs(centers) <= FOLD_TILES * TILE)
        values = mid.mid[:, near_cols][body[:, near_cols]]
        out.append(float(np.median(values)) / frame if len(values) >= 4 and frame > 0 else np.nan)
    return out[0], out[1]


# Полоса «у сгиба» для расфокуса, плиток по 256 px (≈ 2.2 см при 450 dpi).
FOLD_TILES = 1.5


@main.command("gutter-rank")
@click.option("--out-root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=OUT_ROOT)
def gutter_rank(out_root: Path) -> None:
    """Сводка по корешку: сжатие букв у края текста (``glyphs.csv``) и расфокус у сгиба (полутона) — ``gutter/scores.csv``."""
    glyphs = pd.read_csv(out_root / "glyphs.csv")
    rows = []
    for r in glyphs.itertuples():
        mid_path = out_root / "midtone" / r.issue / f"{Path(r.path).stem}.npz"
        left, right = (
            fold_midtone(mid_path, r.fold_x) if mid_path.exists() and np.isfinite(r.fold_x) else (np.nan, np.nan)
        )
        rows.append({"key": r.key, "L_blur": left, "R_blur": right})
    scores = glyphs.merge(pd.DataFrame(rows), on="key")
    # Сжатие засчитывается, если следующая полоса за краем — как середина (контроль: иначе это не край текста,
    # а обложка, таблица или иная вёрстка).
    for side in ("L", "R"):
        valid = scores[f"{side}_next"].between(0.9, 1.1)
        scores[f"{side}_squeeze"] = np.where(valid, scores[f"{side}_near"], np.nan)
    scores["squeeze"] = scores[["L_squeeze", "R_squeeze"]].min(axis=1)
    scores["blur"] = scores[["L_blur", "R_blur"]].max(axis=1)
    scores.to_csv(out_root / "gutter" / "scores.csv", index=False)
    click.echo(f"записано {len(scores)} → {out_root / 'gutter' / 'scores.csv'}")


# Пороги находок (калибровка 2026-10-03 по распределению пака и отсмотру поясов, reports/spread_defects_pack4.md).
MOTION_PEAK, MOTION_MIDTONE = 6.0, 1.15  # пик кепстра (agree_peak) и полутоновый ореол у пика к медиане кадра
GUTTER_SQUEEZE = 0.75  # сжатие букв у края текста (ширина/высота к середине страницы)
GUTTER_BLUR = 1.5  # доля полутонов у сгиба к медиане кадра
# Сторона годится для меры сжатия: букв у края и всего на стороне не меньше (оглавления, обложки — меньше).
MIN_NEAR_GLYPHS, MIN_SIDE_GLYPHS = 150, 2000


@main.command()
@click.option("--out-root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=OUT_ROOT)
def export(out_root: Path) -> None:
    """Итоговые списки находок по порогам: ``findings.csv`` (смаз, сжатие и расфокус у корешка) и ``sequence.csv``."""
    motion = pd.read_csv(out_root / "motion" / "scores.csv")
    hit = motion[(motion.agree_peak >= MOTION_PEAK) & (motion.mid_at_peak_rel >= MOTION_MIDTONE)]
    rows = [
        {
            "key": r.key,
            "path": r.path,
            "defect": "motion",
            "score": round(r.agree_peak, 2),
            "detail": f"сдвиг ({r.shift_dx},{r.shift_dy}) px, плитка {r.row},{r.col}, полутона ×{r.mid_at_peak_rel:.2f}",
        }
        for r in hit.itertuples()
    ]
    gutter = pd.read_csv(out_root / "gutter" / "scores.csv")
    for r in gutter.to_dict("records"):
        for side in ("L", "R"):
            ok = r.get(f"{side}_near_glyphs", 0) >= MIN_NEAR_GLYPHS and r.get(f"{side}_glyphs", 0) >= MIN_SIDE_GLYPHS
            squeeze = r.get(f"{side}_squeeze")
            if ok and pd.notna(squeeze) and squeeze < GUTTER_SQUEEZE:
                rows.append(
                    {
                        "key": r["key"],
                        "path": r["path"],
                        "defect": "gutter_squeeze",
                        "score": round(squeeze, 3),
                        "detail": f"сторона {side}, край текста {r[f'{side}_edge']:.0f} px от сгиба",
                    }
                )
            blur = r.get(f"{side}_blur")
            if pd.notna(blur) and blur > GUTTER_BLUR:
                rows.append(
                    {
                        "key": r["key"],
                        "path": r["path"],
                        "defect": "gutter_blur",
                        "score": round(blur, 3),
                        "detail": f"сторона {side}",
                    }
                )
    findings = pd.DataFrame(rows).sort_values(["defect", "key"])
    findings.to_csv(out_root / "findings.csv", index=False)
    events = pd.read_csv(out_root / "pages" / "events.csv")
    events.to_csv(out_root / "sequence.csv", index=False)
    click.echo(findings.groupby("defect").key.nunique().to_string())
    click.echo(f"событий ряда номеров: {len(events)}")


def issue_page_numbers(pages_csv: Path) -> dict[str, int | None]:
    """Номер левой полосы каждого кадра пака по прочитанным номерам колонтитула.

    Args:
        pages_csv: ``pages/frames.csv`` второго прохода.

    Returns:
        Ключ кадра → номер левой полосы разворота (None — не определяется).
    """
    from research.spread_defects.sequence import frame_numbers, offset_candidates

    frames = pd.read_csv(pages_csv).sort_values(["issue", "index", "key"])
    out = {}
    for _, group in frames.groupby("issue"):
        candidates = [
            offset_candidates(k, None if pd.isna(l) else int(l), None if pd.isna(r) else int(r))
            for k, (l, r) in enumerate(zip(group.left, group.right))
        ]
        out.update(dict(zip(group.key, frame_numbers(candidates))))
    return out


def spread_pages(left: int | None) -> str:
    """Подпись разворота для списка пересъёмки: ``112–113`` или ``?``."""
    return "?" if left is None else f"{left}–{left + 1}"


@main.command()
@click.option("--out-root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=OUT_ROOT)
@click.option(
    "--gaps",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=SETS_DIR / "sequence.csv",
    help="Разметка событий ряда номеров: в список идут события с confirmed=yes и kind=gap.",
)
def reshoot(out_root: Path, gaps: Path) -> None:
    """Список на пересъёмку по разворотам (год, выпуск, страницы): смаз из ``findings.csv`` и подтверждённые пропуски."""
    numbers = issue_page_numbers(out_root / "pages" / "frames.csv")
    # Номера, проставленные глазами там, где колонтитула нет (содержание, первые полосы).
    overrides = pd.read_csv(SETS_DIR / "page_overrides.csv")
    numbers.update(dict(zip(overrides.key, overrides.left.astype(int))))
    findings = pd.read_csv(out_root / "findings.csv")
    rows = []
    for r in findings[findings.defect == "motion"].drop_duplicates("key").itertuples():
        issue, name = r.key.rsplit("/", 1)
        rows.append(
            {
                "year": issue[:4],
                "issue": issue[5:7],
                "pages": spread_pages(numbers.get(r.key)),
                "defect": "смаз",
                "frame": name,
                "folder": issue,
            }
        )
    marked = pd.read_csv(gaps)
    for r in marked[
        (marked.kind == "gap") & (marked.confirmed == "yes") & ~marked.issue.str.startswith("неразобранное")
    ].itertuples():
        for pages in str(r.pages).split(";"):
            rows.append(
                {
                    "year": r.issue[:4],
                    "issue": r.issue[5:7],
                    "pages": pages.strip(),
                    "defect": "пропуск разворота",
                    "frame": r.frame,
                    "folder": r.issue,
                }
            )
    table = pd.DataFrame(rows)
    table["first"] = table.pages.str.extract(r"^(\d+)").astype(float)
    table = table.sort_values(["year", "issue", "first"]).drop(columns="first")
    table.to_csv(out_root / "reshoot.csv", index=False)
    click.echo(table.to_string(index=False))


def pass3_done(out_root: Path, frame: Frame) -> Path:
    """npz строк у сгиба кадра (третий проход): ``<out_root>/lines/<выпуск>/<кадр>.npz``.

    Args:
        out_root: Корень выхода стенда.
        frame: Кадр.

    Returns:
        Путь npz.
    """
    return out_root / "lines" / frame.issue / (frame.path.stem + ".npz")


def _pass3_one(frame: Frame, out_root: Path) -> dict:
    """Третий проход по кадру: строки у сгиба и признаки концевых букв против опорных (воркер пула).

    Args:
        frame: Кадр.
        out_root: Корень выхода стенда.

    Returns:
        Строка сводки: ключ, выпуск, номер, путь, число строк по сторонам, время; ``error`` при сбое.
    """
    from ocr_utils.defocus_detection.metrics.dom import gradient_threshold
    from research.spread_defects.glyph_quality import lines_table, measure_line
    from research.spread_defects.gutter import find_fold
    from research.spread_defects.lines import LEFT, RIGHT, find_glyphs, side_lines

    started = time.time()
    row = {"key": frame.key, "issue": frame.issue, "index": frame.index, "path": str(frame.path)}
    try:
        gray = cv2.cvtColor(read_rgb(frame.path), cv2.COLOR_RGB2GRAY).astype(np.float32)
        fold, cut = find_fold(gray)
        glyphs = find_glyphs(gray)
        threshold = gradient_threshold(gray.astype(np.uint8))
        measures = []
        for side in (LEFT, RIGHT):
            if not np.isfinite(glyphs.glyph_h):
                break
            for line in side_lines(glyphs, fold, side):
                measure = measure_line(gray, glyphs, line, side, threshold)
                if measure is not None:
                    measures.append(measure)
    except Exception as error:  # битый файл не должен ронять прогон по паку
        return {**row, "error": repr(error)}
    target = pass3_done(out_root, frame)
    target.parent.mkdir(parents=True, exist_ok=True)
    table = lines_table(measures) if measures else {}
    np.savez_compressed(
        target,
        fold=fold,
        fold_ink=cut,
        height=gray.shape[0],
        width=gray.shape[1],
        frame_glyph_h=glyphs.glyph_h,
        **table,
    )
    sides = [m.side for m in measures]
    return {
        **row,
        "L_lines": sides.count(LEFT),
        "R_lines": sides.count(RIGHT),
        "seconds": round(time.time() - started, 2),
        "error": "",
    }


@main.command()
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=PACK4_ROOT)
@click.option("--only", multiple=True, help="Префикс ключа выпуска (``1985/02``, ``1979``); можно несколько.")
@click.option("--out-root", type=click.Path(file_okay=False, path_type=Path), default=OUT_ROOT)
@click.option("--jobs", default=12, show_default=True, help="Воркеров (диск NTFS-3G — узкое место).")
@click.option("--force", is_flag=True, help="Пересчитать кадры, уже прошедшие проход.")
def pass3(root: Path, only: tuple[str, ...], out_root: Path, jobs: int, force: bool) -> None:
    """Третий проход: строки у сгиба, признаки концевых букв против опорных той же строки (``lines/…npz``)."""
    run_pool(iter_pack(root, only), _pass3_one, out_root, out_root / "pass3.csv", jobs, force, pass3_done)


@main.command("pass3-rank")
@click.option("--out-root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=OUT_ROOT)
def pass3_rank_cmd(out_root: Path) -> None:
    """Признаки кадров третьего прохода (``pass3_scores.csv``): худшие четверти, пересвет, буквы под переплётом."""
    from research.spread_defects.pass3_rank import rank

    scores = rank(out_root)
    scores.to_csv(out_root / "pass3_scores.csv", index=False)
    click.echo(f"кадров {len(scores)} → {out_root / 'pass3_scores.csv'}")


def _surya_crops(frame: Frame) -> dict:
    """Подготовка кадра для surya (воркер пула, CPU): сгиб и вырезки обеих полос.

    Args:
        frame: Кадр.

    Returns:
        ``{"key", "fold_x", "crops": {side: (вырезка uint8, x0, y0)}}`` или ``{"key", "error"}``.
    """
    from research.spread_defects.gutter import find_fold
    from research.spread_defects.truncated_words import side_crop

    try:
        gray = cv2.cvtColor(read_rgb(frame.path), cv2.COLOR_RGB2GRAY)
        fold, _ = find_fold(gray.astype(np.float32))
        fold_x = float(np.polyval(fold, gray.shape[0] / 2))
        crops = {side: side_crop(gray, fold_x, side) for side in ("L", "R")}
    except Exception as error:  # битый файл не должен ронять прогон
        return {"key": frame.key, "error": repr(error)}
    return {"key": frame.key, "fold_x": fold_x, "crops": crops}


@main.command()
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=PACK4_ROOT)
@click.option("--only", multiple=True, help="Префикс ключа выпуска.")
@click.option("--keys", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="CSV с колонкой key.")
@click.option("--out-root", type=click.Path(file_okay=False, path_type=Path), default=OUT_ROOT)
@click.option("--jobs", default=6, show_default=True, help="Воркеров подготовки кадров (CPU); surya — один на GPU.")
def truncated(root: Path, only: tuple[str, ...], keys: Path | None, out_root: Path, jobs: int) -> None:
    """Слова-обрубки у сгиба: surya по полосам (кэш ``ocr_fold/…json``), словарь — ``truncated.csv``."""
    import json

    from research.spread_defects.truncated_words import check_surya_side, surya_lines

    frames = iter_pack(root, only)
    if keys:
        wanted = set(pd.read_csv(keys).key)
        frames = [f for f in frames if f.key in wanted]
    cache = {f.key: out_root / "ocr_fold" / f.issue / f"{f.path.stem}.json" for f in frames}
    todo = [f for f in frames if not cache[f.key].exists()]
    click.echo(f"кадров {len(frames)}, читать surya {len(todo)}")
    if todo:
        from PIL import Image
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        recognizer, detector = RecognitionPredictor(FoundationPredictor()), DetectionPredictor()
        started = time.time()
        with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
            for n, prepared in enumerate(pool.map(_surya_crops, todo, chunksize=1), 1):
                key = prepared["key"]
                if "error" in prepared:
                    click.echo(f"сбой {key}: {prepared['error']}")
                    continue
                sides = list(prepared["crops"])
                images = [Image.fromarray(prepared["crops"][s][0]).convert("RGB") for s in sides]
                # Партии ограничены: по умолчанию surya занимает всю видеопамять и падает с OOM на плотной полосе.
                results = recognizer(
                    images, det_predictor=detector, math_mode=False, recognition_batch_size=64, detection_batch_size=2
                )
                record = {"fold_x": prepared["fold_x"]}
                for side, result in zip(sides, results):
                    _, x0, y0 = prepared["crops"][side]
                    record[side] = surya_lines(result, x0, y0)
                target = cache[key]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(record, ensure_ascii=False))
                if n % 20 == 0:
                    rate = n / (time.time() - started)
                    click.echo(f"{n}/{len(todo)}  {rate:.2f} кадр/с  осталось ~{(len(todo) - n) / rate / 60:.0f} мин")
    rows = []
    for f in frames:
        if not cache[f.key].exists():
            continue
        record = json.loads(cache[f.key].read_text())
        row = {"key": f.key, "issue": f.issue, "path": str(f.path)}
        for side in ("L", "R"):
            check = check_surya_side(record.get(side, []), record["fold_x"], side)
            row.update({f"{side}_fold_n": check.fold_checked, f"{side}_fold_bad": check.fold_unknown,
                        f"{side}_outer_n": check.outer_checked, f"{side}_outer_bad": check.outer_unknown,
                        f"{side}_excess": check.excess, f"{side}_examples": " ".join(check.examples)})  # fmt: skip
        rows.append(row)
    table = pd.DataFrame(rows)
    table["excess"] = table[["L_excess", "R_excess"]].max(axis=1)
    table.sort_values("excess", ascending=False).to_csv(out_root / "truncated.csv", index=False)
    click.echo(f"записано {len(table)} → {out_root / 'truncated.csv'}")


def pass4_done(out_root: Path, frame: Frame) -> Path:
    """npz сетки пересвета кадра: ``<out_root>/exposure/<выпуск>/<кадр>.npz``.

    Args:
        out_root: Корень выхода стенда.
        frame: Кадр.

    Returns:
        Путь npz.
    """
    return out_root / "exposure" / frame.issue / (frame.path.stem + ".npz")


def _pass4_one(frame: Frame, out_root: Path) -> dict:
    """Пересвет по плиткам кадра (воркер пула): npz сетки и сводка по полосам.

    Args:
        frame: Кадр.
        out_root: Корень выхода стенда.

    Returns:
        Строка сводки (``exposure.side_summary``) с ключом, выпуском, путём и x сгиба; ``error`` при сбое.
    """
    from research.spread_defects.exposure import ExposureTiles, exposure_tiles, side_summary
    from research.spread_defects.gutter import find_fold

    row = {"key": frame.key, "issue": frame.issue, "index": frame.index, "path": str(frame.path)}
    try:
        rgb = read_rgb(frame.path)
        fold, _ = find_fold(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32))
        fold_x = float(np.polyval(fold, rgb.shape[0] / 2))
        tiles = exposure_tiles(rgb)
    except Exception as error:  # битый файл не должен ронять прогон по паку
        return {**row, "error": repr(error)}
    target = pass4_done(out_root, frame)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **{n: getattr(tiles, n) for n in ExposureTiles.__dataclass_fields__})
    return {**row, "fold_x": fold_x, **side_summary(tiles, fold_x), "error": ""}


@main.command()
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=PACK4_ROOT)
@click.option("--only", multiple=True, help="Префикс ключа выпуска.")
@click.option("--out-root", type=click.Path(file_okay=False, path_type=Path), default=OUT_ROOT)
@click.option("--jobs", default=14, show_default=True, help="Воркеров (диск NTFS-3G — узкое место).")
@click.option("--force", is_flag=True, help="Пересчитать кадры, уже прошедшие проход.")
def pass4(root: Path, only: tuple[str, ...], out_root: Path, jobs: int, force: bool) -> None:
    """Четвёртый проход: пересвет по плиткам (толщина штриха, светлота и доля краски) — ``exposure.csv``."""
    run_pool(iter_pack(root, only), _pass4_one, out_root, out_root / "exposure.csv", jobs, force, pass4_done)


def _deepseek_crops(frame: Frame, out_root: Path) -> list[dict]:
    """Вырезки колонки у сгиба для DeepSeek (воркер пула): задания с путями PNG; пусто при сбое."""
    from research.spread_defects.deepseek_fold import make_crops

    try:
        return make_crops(frame, out_root / "deepseek_fold" / "crops" / frame.issue / frame.path.stem)
    except Exception:  # битый кадр не роняет прогон
        return []


@main.command("hidden-ds")
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=PACK4_ROOT)
@click.option("--only", multiple=True, help="Префикс ключа выпуска.")
@click.option("--out-root", type=click.Path(file_okay=False, path_type=Path), default=OUT_ROOT)
@click.option("--jobs", default=12, show_default=True, help="Воркеров подготовки вырезок (CPU).")
@click.option("--batch-frames", default=400, show_default=True, help="Кадров на один запуск vLLM.")
def hidden_ds(root: Path, only: tuple[str, ...], out_root: Path, jobs: int, batch_frames: int) -> None:
    """Буквы под переплётом: DeepSeek-OCR-2 по колонке у сгиба (кэш), словарь — ``hidden_ds.csv``."""
    from research.spread_defects.deepseek_fold import frame_check, read_answers, run_deepseek

    frames = iter_pack(root, only)
    answers_dir = out_root / "deepseek_fold"
    for start in range(0, len(frames), batch_frames):
        part = frames[start : start + batch_frames]
        with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
            tasks = [t for chunk in pool.map(_deepseek_crops, part, [out_root] * len(part)) for t in chunk]
        tick = time.time()
        run_deepseek(tasks, answers_dir)
        click.echo(
            f"{start + len(part)}/{len(frames)} кадров, вырезок {len(tasks)}, DeepSeek {time.time() - tick:.0f} с"
        )
    answers = read_answers(answers_dir / "free.jsonl")
    rows = []
    for f in frames:
        row = {"key": f.key, "issue": f.issue, "path": str(f.path)}
        for side in ("L", "R"):
            c = frame_check(answers, f.key, side)
            row.update({f"{side}_fold_n": c.fold_checked, f"{side}_fold_bad": c.fold_unknown,
                        f"{side}_outer_n": c.outer_checked, f"{side}_outer_bad": c.outer_unknown,
                        f"{side}_examples": " ".join(c.examples)})  # fmt: skip
            row[f"{side}_bad_share"] = c.fold_unknown / c.fold_checked if c.fold_checked >= 8 else np.nan
            row[f"{side}_excess"] = c.excess
        rows.append(row)
    table = pd.DataFrame(rows)
    table["bad_share"] = table[["L_bad_share", "R_bad_share"]].max(axis=1)
    table.sort_values("bad_share", ascending=False).to_csv(out_root / "hidden_ds.csv", index=False)
    click.echo(f"записано {len(table)} → {out_root / 'hidden_ds.csv'}")


def copies_done(out_root: Path, frame: Frame) -> Path:
    """npz сетки признаков ксерокопии кадра: ``<out_root>/copies/<выпуск>/<кадр>.npz``.

    Args:
        out_root: Корень выхода стенда.
        frame: Кадр.

    Returns:
        Путь файла (папка не создаётся).
    """
    return out_root / "copies" / frame.issue / (frame.path.stem + ".npz")


def _copies_one(frame: Frame, out_root: Path) -> dict:
    """Признаки ксерокопии по плиткам кадра (воркер пула): npz сетки и сводка по полосам.

    Args:
        frame: Кадр.
        out_root: Корень выхода стенда.

    Returns:
        Строка сводки (``photocopy.side_summary``) с ключом, выпуском, путём и x сгиба; ``error`` при сбое.
    """
    from research.spread_defects.gutter import find_fold
    from research.spread_defects.photocopy import CopyTiles, copy_tiles, side_summary

    row = {"key": frame.key, "issue": frame.issue, "index": frame.index, "path": str(frame.path)}
    try:
        rgb = read_rgb(frame.path)
        fold, _ = find_fold(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32))
        fold_x = float(np.polyval(fold, rgb.shape[0] / 2))
        tiles = copy_tiles(rgb)
    except Exception as error:  # битый файл не должен ронять прогон по паку
        return {**row, "error": repr(error)}
    target = copies_done(out_root, frame)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **{n: getattr(tiles, n) for n in CopyTiles.__dataclass_fields__})
    return {**row, "fold_x": fold_x, **side_summary(tiles, fold_x), "error": ""}


@main.command()
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=PACK4_ROOT)
@click.option("--only", multiple=True, help="Префикс ключа выпуска.")
@click.option("--out-root", type=click.Path(file_okay=False, path_type=Path), default=OUT_ROOT)
@click.option("--jobs", default=14, show_default=True, help="Воркеров (диск NTFS-3G — узкое место).")
@click.option("--force", is_flag=True, help="Пересчитать кадры, уже прошедшие проход.")
def copies(root: Path, only: tuple[str, ...], out_root: Path, jobs: int, force: bool) -> None:
    """Признаки ксерокопии по полосам (цвет и зерно бумаги, крап, краска, неровность фона) — ``copies.csv``."""
    run_pool(iter_pack(root, only), _copies_one, out_root, out_root / "copies.csv", jobs, force, copies_done)


def _crops_one(frame: Frame, out_root: Path, fold_x: float) -> tuple[str, dict]:
    """Вырезки текста полос кадра для эмбеддингов (воркер пула).

    Args:
        frame: Кадр.
        out_root: Корень выхода стенда (там сетки ``copies``).
        fold_x: x сгиба посередине высоты, px.

    Returns:
        Пара (ключ кадра, словарь сторона → вырезки); пустой словарь при сбое.
    """
    from research.spread_defects.copy_embed import side_crops

    try:
        return frame.key, side_crops(frame, copies_done(out_root, frame), fold_x)
    except Exception:  # битый кадр не роняет прогон
        return frame.key, {}


@main.command("copies-embed")
@click.option("--out-root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=OUT_ROOT)
@click.option("--jobs", default=12, show_default=True, help="Воркеров чтения кадров (GPU — один процесс).")
def copies_embed(out_root: Path, jobs: int) -> None:
    """Эмбеддинги DINOv2 по вырезкам текста каждой полосы (после ``copies``) — ``copies_embed.npz``."""
    from research.spread_defects.copy_embed import Embedder

    frames = pd.read_csv(out_root / "copies.csv")
    frames = frames[frames.fold_x.notna()]
    embed = Embedder()
    keys, sides, vectors = [], [], []
    started = time.time()
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        futures = [
            pool.submit(_crops_one, Frame(Path(r.path), r.issue, int(r["index"])), out_root, float(r.fold_x))
            for _, r in frames.iterrows()
        ]
        for n, future in enumerate(as_completed(futures), 1):
            key, crops = future.result()
            for side, stack in crops.items():
                keys.append(key)
                sides.append(side)
                vectors.append(embed(stack))
            if n % 500 == 0:
                click.echo(f"{n}/{len(futures)}  {n / (time.time() - started):.1f} кадр/с")
    np.savez_compressed(
        out_root / "copies_embed.npz", key=np.array(keys), side=np.array(sides), vector=np.stack(vectors)
    )
    click.echo(f"полос {len(keys)} → {out_root / 'copies_embed.npz'}")


@main.command("copies-rank")
@click.option(
    "--out-root",
    "out_roots",
    multiple=True,
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Корень прогона пака (после ``copies`` и ``copies-embed``); можно несколько.",
)
@click.option(
    "--labels",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=SETS_DIR / "photocopy.csv",
    show_default=True,
)
@click.option("--threshold", default=0.5, show_default=True, help="Порог ядра участка копий.")
@click.option("--extend", default=0.05, show_default=True, help="Порог расширения участка на соседние полосы.")
@click.option("--result-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
def copies_rank(out_roots: tuple[Path, ...], labels: Path, threshold: float, extend: float, result_dir: Path) -> None:
    """Оценка «ксерокопия» по полосам (DINOv2 + ручные признаки, вне складки по выпускам) и участки копий."""
    from research.spread_defects.photocopy import copy_regions, labeled_pages, out_of_fold_scores, page_numbers

    pages = labeled_pages(list(out_roots), pd.read_csv(labels))
    pages["score"] = out_of_fold_scores(pages)
    numbers = pd.concat([page_numbers(root / "pages").assign(pack=root.name) for root in out_roots])
    pages = pages.merge(numbers, on=["pack", "key", "side"], how="left")
    keep = [c for c in pages.columns if not (c.startswith("e") and c[1:].isdigit())]
    result_dir.mkdir(parents=True, exist_ok=True)
    pages[keep].sort_values("score", ascending=False).to_csv(result_dir / "copies_scores.csv", index=False)
    regions = copy_regions(pages, threshold, extend)
    regions.to_csv(result_dir / "copies_regions.csv", index=False)
    click.echo(f"полос {len(pages)}, выше порога {(pages.score >= threshold).sum()}, участков {len(regions)}")


def _quality_one(task: tuple[str, str, str, float]) -> dict:
    """Читаемость одной полосы-копии (воркер пула): tesseract + словарь.

    Args:
        task: (ключ кадра, путь, сторона, x сгиба).

    Returns:
        Строка: key, side и поля ``TextQuality``; ``error`` при сбое.
    """
    from research.spread_defects.photocopy import text_quality

    key, path, side, fold_x = task
    try:
        gray = cv2.cvtColor(read_rgb(Path(path)), cv2.COLOR_RGB2GRAY)
        quality = text_quality(gray, fold_x, side)
    except Exception as error:  # битый кадр не роняет прогон
        return {"key": key, "side": side, "error": repr(error)}
    return {"key": key, "side": side, **quality.__dict__, "error": ""}


@main.command("copies-quality")
@click.option("--result-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option(
    "--out-root",
    "out_roots",
    multiple=True,
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Корни прогонов паков (оттуда x сгиба из ``copies.csv``).",
)
@click.option("--threshold", default=0.5, show_default=True, help="Брать полосы с оценкой не ниже.")
@click.option("--labeled", is_flag=True, help="Брать полосы, размеченные копиями (``y``), а не по порогу.")
@click.option("--jobs", default=14, show_default=True, help="Воркеров (tesseract однопоточный).")
def copies_quality(result_dir: Path, out_roots: tuple[Path, ...], threshold: float, labeled: bool, jobs: int) -> None:
    """Читаемость текста на полосах-копиях (tesseract, словарь) — ``copies_quality.csv``."""
    scores = pd.read_csv(result_dir / "copies_scores.csv")
    folds = pd.concat([pd.read_csv(r / "copies.csv")[["key", "fold_x"]].assign(pack=r.name) for r in out_roots])
    chosen = scores[scores.y] if labeled else scores[scores.score >= threshold]
    todo = chosen.merge(folds, on=["pack", "key"])
    tasks = [(r.key, r.path, r.side, float(r.fold_x)) for r in todo.itertuples()]
    click.echo(f"полос {len(tasks)}, воркеров {jobs}")
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        rows = list(pool.map(_quality_one, tasks))
    quality = todo[["pack", "issue", "key", "side", "page", "score", "speck_iz", "unevenness_iz", "paper_l_iz"]]
    quality = quality.merge(pd.DataFrame(rows), on=["key", "side"])
    quality.to_csv(result_dir / "copies_quality.csv", index=False)
    click.echo(f"записано {len(quality)} → {result_dir / 'copies_quality.csv'}")


def _page_glyphs_one(task: tuple[Path, Path, str]) -> pd.DataFrame:
    """Порча черт по всем буквам кадра (воркер пула): таблица букв, тепловая карта, сводка по полосам.

    Args:
        task: (путь кадра, папка выхода, подпись кадра для шапки).

    Returns:
        Сводка ``page_glyphs.side_summary`` с колонкой ``frame``; пустая при сбое.
    """
    from ocr_utils.defocus_detection.metrics.dom import gradient_threshold
    from research.spread_defects.gutter import find_fold
    from research.spread_defects.lines import find_glyphs
    from research.spread_defects.page_glyphs import cell_maps, glyph_table, heat_overlay, side_summary

    path, out_dir, label = task
    try:
        rgb = read_rgb(path)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        fold, _ = find_fold(gray)
        glyphs = find_glyphs(gray)
        table = glyph_table(gray, glyphs, fold, gradient_threshold(gray.astype(np.uint8)))
    except Exception as error:  # битый кадр не роняет прогон
        click.echo(f"сбой {label}: {error!r}")
        return pd.DataFrame()
    stem = label.replace("/", "_").replace(" ", "_")
    table.to_csv(out_dir / f"{stem}.csv", index=False)
    core, _, cell = cell_maps(table, gray.shape, glyphs.glyph_h)
    title = [f"{label} — порча черт по всем буквам: медиана доли ядра штриха в клетке {cell} px к медиане полосы"]
    cv2.imwrite(
        str(out_dir / f"{stem}_карта.jpg"), heat_overlay(rgb, core, cell, title), [cv2.IMWRITE_JPEG_QUALITY, 88]
    )
    summary = side_summary(table)
    summary.insert(0, "frame", label)
    return summary


@main.command("page-glyphs")
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--frame", "frames", multiple=True, required=True, help="Путь кадра от корня пака; можно несколько.")
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--jobs", default=8, show_default=True, help="Воркеров (кадр ~10 с, память ~1.5 ГБ на воркер).")
def page_glyphs(root: Path, frames: tuple[str, ...], out_dir: Path, jobs: int) -> None:
    """Порча черт по всем буквам выбранных кадров (не только у сгиба): таблицы букв, карты, ``summary.csv``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(root / rel, out_dir, rel) for rel in frames]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        summary = pd.concat(list(pool.map(_page_glyphs_one, tasks)), ignore_index=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    click.echo(summary.to_string(index=False))


# --- Постраничная съёмка газеты (ЭГ): уход текста под корешок ----------------------------------------------------


def _page_gutter_prepare(task: tuple[Path, Path, Path | None, Path | None]) -> dict:
    """Подготовка кадра для surya (воркер пула, CPU): линия сгиба, лента у корешка, превью ленты на диск.

    Линия — ``fold_line`` вариантом ``single_page_gutter.FOLD_VARIANT``; рамки строк surya для запрета берутся из
    кэша ``fold-layout`` (если его нет — линия без рамок).

    Args:
        task: (путь JPEG-превью кадра, папка для лент, папка кэша ``fold-layout`` или None, папка исходных RAF или
            None). С RAF лента режется из полного кадра (линия сгиба — по превью, в масштабе кадра), иначе — из превью.

    Returns:
        ``{"stem", "side", "fold", "fold_col", "image"}``; ``{"stem", "error"}``, если сгиб не найден или кадр
        не читается.
    """
    import json

    from research.spread_defects.fold_line import BoxSource, find_line, parse_variant, select_boxes
    from research.spread_defects.single_page_gutter import (
        FOLD_BOXES,
        FOLD_VARIANT,
        FOLD_KEEP,
        STRIP_DEPTH,
        gutter_strip,
        page_side,
        read_preview,
        read_raw,
        scale_fold,
    )

    path, strip_dir, layout_dir, raw_dir = task
    stem = path.stem
    try:
        rgb = read_preview(path)
        cache = layout_dir / f"{stem}.json" if layout_dir else None
        boxes = select_boxes(json.loads(cache.read_text()), BoxSource(FOLD_BOXES)) if cache and cache.exists() else None
        fold = find_line(rgb, parse_variant(FOLD_VARIANT), boxes=boxes).rows
        if not np.isfinite(fold).all():
            return {"stem": stem, "error": "сгиб не найден"}
        side = page_side(stem)
        if raw_dir is None:
            strip = gutter_strip(rgb, fold, side)
        else:
            full = read_raw(raw_dir / f"{stem}.RAF")
            k = full.shape[1] / rgb.shape[1]
            fold = scale_fold(fold, rgb.shape[:2], full.shape[:2])
            strip = gutter_strip(full, fold, side, depth=int(STRIP_DEPTH * k), keep=FOLD_KEEP * k)
    except Exception as error:  # битый файл не должен ронять прогон
        return {"stem": stem, "error": repr(error)}
    # Лента на диск — для листов отсмотра (половинный масштаб хватает глазу).
    small = cv2.resize(strip.image, None, fx=0.5 if raw_dir is None else 0.3, fy=0.5 if raw_dir is None else 0.3,
                       interpolation=cv2.INTER_AREA)  # fmt: skip
    cv2.imwrite(str(strip_dir / f"{stem}.jpg"), cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {
        "stem": stem,
        "side": side.value,
        "fold": np.round(fold, 1).tolist(),
        "fold_col": strip.fold_col,
        "image": strip.image,
    }


@main.command("page-gutter")
@click.option("--previews", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Папка JPEG-превью кадров (имя кадра.jpg, суффикс _1/_2 — сторона полосы).")  # fmt: skip
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Выход: ocr_surya/*.json (кэш), strips/*.jpg, features.csv.")  # fmt: skip
@click.option("--skip", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Файл со списком имён кадров (по одному в строке), которые не читать: строки параллельны сгибу.")  # fmt: skip
@click.option("--layout-dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Кэш fold-layout (layout/<кадр>.json): строки surya — запрет для линии сгиба.")  # fmt: skip
@click.option("--raw-dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Папка исходных RAF: лента у сгиба режется из полного кадра (строка ~58 px вместо ~33 на превью).")  # fmt: skip
@click.option("--rec-batch", default=64, show_default=True,
              help="Строк в пачке распознавания surya. На полном RAW — 16: GPU с подключённым экраном снимает долгие "
                   "вызовы ядра сторожевым таймером (cudaErrorLaunchTimeout на пачке 64, 2026-10-06).")  # fmt: skip
@click.option("--jobs", default=6, show_default=True, help="Воркеров подготовки кадров (CPU); surya — один на GPU.")
def page_gutter(
    previews: Path,
    out_dir: Path,
    skip: Path | None,
    layout_dir: Path | None,
    raw_dir: Path | None,
    rec_batch: int,
    jobs: int,
) -> None:
    """Уход текста под корешок на постраничной съёмке газеты: surya по ленте у сгиба (кэш), признаки — features.csv."""
    import json

    from research.spread_defects.single_page_gutter import Side, gutter_features

    skipped = set(skip.read_text().split()) if skip else set()
    paths = sorted(p for p in previews.glob("*.jpg") if p.stem not in skipped)
    cache_dir, strip_dir = out_dir / "ocr_surya", out_dir / "strips"
    cache_dir.mkdir(parents=True, exist_ok=True)
    strip_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in paths if not (cache_dir / f"{p.stem}.json").exists()]
    click.echo(f"кадров {len(paths)} (пропущено по списку {len(skipped)}), читать surya {len(todo)}")
    if todo:
        from PIL import Image
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        recognizer, detector = RecognitionPredictor(FoundationPredictor()), DetectionPredictor()
        started = time.time()
        tasks = [(p, strip_dir, layout_dir, raw_dir) for p in todo]
        with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
            for n, prepared in enumerate(pool.map(_page_gutter_prepare, tasks, chunksize=1), 1):
                stem = prepared["stem"]
                record: dict = {"stem": stem}
                if "error" in prepared:
                    click.echo(f"{stem}: {prepared['error']}")
                    record["error"] = prepared["error"]
                else:
                    # Партии ограничены: по умолчанию surya занимает всю видеопамять и падает с OOM.
                    result = recognizer(
                        [Image.fromarray(prepared["image"])],
                        det_predictor=detector,
                        math_mode=False,
                        recognition_batch_size=rec_batch,
                        detection_batch_size=2,
                    )[0]
                    record.update(
                        side=prepared["side"],
                        fold=prepared["fold"],
                        fold_col=np.round(prepared["fold_col"], 1).tolist(),
                        size=list(prepared["image"].shape[:2]),
                        lines=[
                            {"bbox": [round(float(v), 1) for v in line.bbox], "text": line.text,
                             "conf": round(float(line.confidence or 0), 3)}  # fmt: skip
                            for line in result.text_lines
                        ],
                    )
                (cache_dir / f"{stem}.json").write_text(json.dumps(record, ensure_ascii=False))
                if n % 10 == 0:
                    rate = n / (time.time() - started)
                    click.echo(f"{n}/{len(todo)}  {rate:.2f} кадр/с  осталось ~{(len(todo) - n) / rate / 60:.0f} мин")
    rows = []
    for p in paths:
        record = json.loads((cache_dir / f"{p.stem}.json").read_text())
        row = {"stem": p.stem, "error": record.get("error", "")}
        if not row["error"]:
            features = gutter_features(record["lines"], np.asarray(record["fold_col"]), Side(record["side"]))
            row.update(
                side=record["side"],
                lines=features.lines,
                line_height=round(features.line_height, 1),
                margin_p10=round(features.margin_p10, 2),
                margin_median=round(features.margin_median, 2),
                touch_share=round(features.touch_share, 3),
                fold_checked=features.fold_checked,
                fold_unknown=features.fold_unknown,
                outer_checked=features.outer_checked,
                outer_unknown=features.outer_unknown,
                fold_share=round(features.fold_share, 3),
                outer_share=round(features.outer_share, 3),
                inner_checked=features.inner_checked,
                inner_share=round(features.inner_share, 3),
                excess=round(features.excess, 3),
                examples=" ".join(features.examples),
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "features.csv", index=False)
    click.echo(f"записано {len(table)} → {out_dir / 'features.csv'}")


@main.command("page-gutter-sort")
@click.option("--out-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Выход ``page-gutter`` (features.csv); сюда же пишутся verdicts.csv и links/.")  # fmt: skip
@click.option("--raw-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Папка исходных RAF (на них ставятся симлинки; туда ничего не пишется).")  # fmt: skip
@click.option("--manual", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="CSV отсмотра глазами (stem, verdict, reason): перекрывает решение по признакам.")  # fmt: skip
@click.option("--loss-share", default=0.16, show_default=True, help="Доля обрубков у сгиба для «есть потеря».")
@click.option("--normal-share", default=0.05, show_default=True, help="Доля обрубков у сгиба ниже — «в норме».")
@click.option("--second-features", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="features.csv второго чтения (например, по превью при основном по RAW): доля обрубков, шум и число "
                   "слов берутся наименьшие из двух — обрубок засчитывается, только если его видят оба чтения, а "
                   "случайные ошибки чтения у двух прогонов разные.")  # fmt: skip
def page_gutter_sort(
    out_dir: Path,
    raw_dir: Path,
    manual: Path | None,
    loss_share: float,
    normal_share: float,
    second_features: Path | None,
) -> None:
    """Раскладка кадров по трём папкам симлинков: есть потеря / в норме / непонятно (verdicts.csv)."""
    import shutil

    from research.spread_defects.single_page_gutter import FrameSigns, Verdict, decide

    table = pd.read_csv(out_dir / "features.csv")
    if second_features is not None:
        # Согласие двух чтений: меньшая доля обрубков, меньший шум, меньшее число проверенных слов.
        second = pd.read_csv(second_features).set_index("stem")
        for column in ("fold_share", "inner_share", "fold_checked"):
            other = table.stem.map(second[column])
            table[column] = np.fmin(table[column], other).where(other.notna(), table[column])
    # Пиксельные признаки у складки (page-gutter-pixels), если посчитаны.
    pixels_path = out_dir / "pixels.csv"
    if pixels_path.exists():
        table = table.merge(pd.read_csv(pixels_path)[["stem", "touching", "gap_median"]], on="stem", how="left")
    overrides = pd.read_csv(manual).set_index("stem") if manual else pd.DataFrame(columns=["verdict", "reason"])
    rows = []
    for raw in sorted(raw_dir.glob("*.RAF")):
        stem = raw.stem
        found = table[table.stem == stem]
        share = float(found.fold_share.iloc[0]) if len(found) else float("nan")
        if stem in overrides.index:
            verdict, reason = Verdict(overrides.loc[stem, "verdict"]), f"глазами: {overrides.loc[stem, 'reason']}"
        elif not len(found):
            verdict, reason = Verdict.UNCLEAR, "нет признаков"
        elif isinstance(found.error.iloc[0], str) and found.error.iloc[0]:
            verdict, reason = Verdict.UNCLEAR, f"сбой разбора: {found.error.iloc[0]}"
        else:
            r = found.iloc[0]
            signs = FrameSigns(
                fold_checked=int(r.fold_checked),
                fold_share=float(r.fold_share),
                inner_share=float(r.inner_share),
                touch_share=float(r.get("touch_share", float("nan"))),
                margin_p10=float(r.get("margin_p10", float("nan"))),
                touching=int(r.get("touching", 0) if pd.notna(r.get("touching", 0)) else 0),
                gap_median=float(r.get("gap_median", float("nan"))),
            )
            verdict, reason = decide(signs, loss_share, normal_share)
        rows.append({"stem": stem, "verdict": verdict.value, "fold_share": share, "reason": reason, "raw": str(raw)})
    verdicts = pd.DataFrame(rows)
    verdicts.to_csv(out_dir / "verdicts.csv", index=False)
    # Папки пересобираются с нуля: в них только симлинки, исходники не трогаются.
    links = out_dir / "links"
    if links.exists():
        shutil.rmtree(links)
    for verdict in Verdict:
        (links / verdict.value).mkdir(parents=True)
    for row in rows:
        # Доля обрубков в начале имени — сортировка по силе потери (у отсмотренных без признаков — «na»).
        prefix = "na" if np.isnan(row["fold_share"]) else f"{row['fold_share']:.2f}"
        (links / row["verdict"] / f"{prefix}_{Path(row['raw']).name}").symlink_to(row["raw"])
    click.echo(verdicts.verdict.value_counts().to_string())
    click.echo(f"→ {links}")


def _page_gutter_ds_crops(task: tuple[Path, Path, Path]) -> list[dict]:
    """Вырезки колонки у сгиба для DeepSeek одного кадра (воркер пула, CPU).

    Args:
        task: (JPEG-превью кадра, JSON кэша surya кадра, папка вырезок).

    Returns:
        Задания воркера DeepSeek (пусто, если колонки нет или сгиб не найден).
    """
    import json

    from research.spread_defects.single_page_gutter import Side, deepseek_crops, gutter_strip, read_preview

    preview, cache, crop_root = task
    record = json.loads(cache.read_text())
    if record.get("error"):
        return []
    rgb = read_preview(preview)
    fold = np.asarray(record["fold"])
    if len(fold) != rgb.shape[1]:
        # Кэш surya снят с полного RAW (линия в масштабе кадра): вырезки для сверки делаются только по превью.
        return []
    strip = gutter_strip(rgb, fold, Side(record["side"]))
    return deepseek_crops(strip, record["lines"], record["stem"], crop_root / record["stem"])


@main.command("page-gutter-deepseek")
@click.option("--out-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Выход ``page-gutter`` (previews рядом — см. --previews).")  # fmt: skip
@click.option("--previews", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--low", default=0.06, show_default=True, help="Нижняя граница пояса доли обрубков surya.")
@click.option("--high", default=0.30, show_default=True, help="Верхняя граница пояса доли обрубков surya.")
@click.option("--jobs", default=8, show_default=True, help="Воркеров нарезки (CPU); DeepSeek — один процесс на GPU.")
def page_gutter_deepseek(out_dir: Path, previews: Path, low: float, high: float, jobs: int) -> None:
    """Второй читатель DeepSeek-OCR-2 для кадров спорного пояса surya: deepseek.csv (доля обрубков у сгиба)."""
    from research.spread_defects.deepseek_fold import read_answers, run_deepseek
    from research.spread_defects.single_page_gutter import Side, deepseek_check

    table = pd.read_csv(out_dir / "features.csv")
    band = table[(table.fold_share >= low) & (table.fold_share <= high)]
    click.echo(f"кадров в поясе {low}–{high}: {len(band)}")
    ds_dir = out_dir / "deepseek"
    tasks = [(previews / f"{s}.jpg", out_dir / "ocr_surya" / f"{s}.json", ds_dir / "crops") for s in band.stem]
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        all_jobs = [job for frame_jobs in pool.map(_page_gutter_ds_crops, tasks) for job in frame_jobs]
    click.echo(f"кусков {len(all_jobs)}")
    run_deepseek(all_jobs, ds_dir)
    answers = read_answers(ds_dir / "free.jsonl")
    rows = []
    for _, r in band.iterrows():
        check = deepseek_check(answers, r.stem, Side(r.side))
        share = check.fold_unknown / check.fold_checked if check.fold_checked >= 8 else float("nan")
        rows.append({"stem": r.stem, "ds_checked": check.fold_checked, "ds_unknown": check.fold_unknown,
                     "ds_share": round(share, 3), "ds_examples": " ".join(check.examples)})  # fmt: skip
    pd.DataFrame(rows).to_csv(out_dir / "deepseek.csv", index=False)
    click.echo(f"записано {len(rows)} → {out_dir / 'deepseek.csv'}")


# --- Линия сгиба: разметка ленты у корешка моделями и сравнение вариантов ----------------------------------------


@main.command("fold-layout")
@click.option("--previews", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--stems", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="CSV с колонкой stem (например, эталон линии сгиба).")  # fmt: skip
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True,
              help="Кэш: layout/<кадр>.json — рамки surya layout, строк surya и блоков DeepSeek в координатах кадра.")  # fmt: skip
def fold_layout(previews: Path, stems: Path, out_dir: Path) -> None:
    """Разметка верхней ленты кадров для линии сгиба: surya layout, строки surya, DeepSeek-OCR-2 grounding."""
    import json

    from PIL import Image

    from research.spread_defects.deepseek_fold import VLLM_ENV
    from research.spread_defects.fold_line import band_pieces, piece_box_to_frame
    from research.spread_defects.single_page_gutter import page_side, read_preview

    names = sorted(set(pd.read_csv(stems).stem))
    cache_dir, piece_dir = out_dir / "layout", out_dir / "layout_pieces"
    cache_dir.mkdir(parents=True, exist_ok=True)
    piece_dir.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not (cache_dir / f"{n}.json").exists()]
    click.echo(f"кадров {len(names)}, размечать {len(todo)}")
    if not todo:
        return
    # Куски лент (CPU, мало кадров — без пула).
    pieces = {n: band_pieces(read_preview(previews / f"{n}.jpg"), page_side(n).value) for n in todo}
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.layout import LayoutPredictor
    from surya.settings import settings

    layout = LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT))
    detector = DetectionPredictor()
    records: dict[str, dict] = {n: {"surya_layout": [], "surya_lines": [], "ds_blocks": []} for n in todo}
    jobs = []
    for name, frame_pieces in pieces.items():
        images = [Image.fromarray(p.image) for p in frame_pieces]
        for piece, result in zip(frame_pieces, layout(images)):
            for box in result.bboxes:
                records[name]["surya_layout"].append([*piece_box_to_frame(piece, *box.bbox), box.label])
        for piece, result in zip(frame_pieces, detector(images)):
            for box in result.bboxes:
                records[name]["surya_lines"].append([*piece_box_to_frame(piece, *box.bbox), "line"])
        for k, piece in enumerate(frame_pieces):
            path = piece_dir / f"{name}_{k}.png"
            Image.fromarray(piece.image).save(path)
            jobs.append({"id": f"{name}|{k}", "crop": str(path)})
    del layout, detector
    # DeepSeek-OCR-2 grounding (штатный промпт документа) — отдельным процессом в своём окружении vLLM.
    import subprocess

    import torch

    torch.cuda.empty_cache()
    jobs_file = out_dir / "layout_jobs.jsonl"
    jobs_file.write_text("\n".join(json.dumps(j, ensure_ascii=False) for j in jobs) + "\n")
    worker = (
        Path(__file__).resolve().parents[2]
        / "ocr_utils"
        / "page_layout"
        / "line_art"
        / "deepseek"
        / "vllm_env"
        / "worker.py"
    )
    subprocess.run(
        ["uv", "run", "--project", str(VLLM_ENV), "python", str(worker), "--jobs", str(jobs_file), "--out-dir",
         str(out_dir / "layout_ds"), "--prompts", "markdown", "--gpu-memory-utilization", "0.6"],
        check=True,
    )  # fmt: skip
    for line in (out_dir / "layout_ds" / "markdown.jsonl").read_text().splitlines():
        answer = json.loads(line)
        name, k = answer["id"].split("|")
        if name not in records:
            continue
        piece = pieces[name][int(k)]
        for element in answer.get("elements", []):
            box = (element["x0"], element["y0"], element["x1"], element["y1"])
            records[name]["ds_blocks"].append([*piece_box_to_frame(piece, *box), element["label"]])
    for name, record in records.items():
        (cache_dir / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False))
    click.echo(f"записано {len(records)} → {cache_dir}")


def _fold_variants_one(task: tuple) -> tuple[list[dict], dict]:
    """Линии вариантов одного кадра (воркер пула): ошибки против эталона, расхождение, оверлеи.

    Args:
        task: (имя кадра, папка превью, папка кэша разметки, [(вариант, источник рамок)], точки эталона
            ``[(x, y)]``, папка оверлеев или None).

    Returns:
        (строки ошибок по точкам эталона, сводка кадра: наибольшее расхождение вариантов и где).
    """
    import json

    from research.spread_defects.fold_line import column_levels, find_line, overlay_pieces, select_boxes
    from research.spread_defects.single_page_gutter import read_preview

    stem, previews, layout_dir, combos, etalon, overlay_dir = task
    rgb = read_preview(previews / f"{stem}.jpg")
    levels = column_levels(rgb)
    cache = layout_dir / f"{stem}.json"
    record = json.loads(cache.read_text()) if cache.exists() else None
    lines = []
    for variant, source in combos:
        boxes = select_boxes(record, source) if record else None
        lines.append(
            (f"{variant.name} | рамки: {source.value}", find_line(rgb, variant, boxes=boxes, levels=levels).rows)
        )
    errors = [
        {"stem": stem, "variant": name, "x": x, "err": float(rows[int(x)] - y)}
        for name, rows in lines
        for x, y in etalon
    ]
    # Расхождение вариантов: наибольшая по x разность самой нижней и самой верхней линии (в зоне без краёв кадра).
    stack = np.stack([rows for _, rows in lines])[:, 100:-100]
    spread = np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)
    summary = {"stem": stem, "spread_max": float(np.nanmax(spread)), "spread_x": int(np.nanargmax(spread)) + 100}
    if overlay_dir is not None:
        palette = [(0, 160, 0), (255, 90, 0), (0, 140, 255), (200, 0, 200), (0, 0, 220)]
        coloured = [(name, rows, palette[k % len(palette)]) for k, (name, rows) in enumerate(lines)]
        for k, image in enumerate(overlay_pieces(rgb, stem, coloured, etalon or None)):
            cv2.imwrite(str(overlay_dir / f"{stem}_{k}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return errors, summary


@main.command("fold-variants")
@click.option("--previews", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Папка вариантов линии сгиба (кэш layout/ от fold-layout; сюда пишутся CSV и overlays/).")  # fmt: skip
@click.option("--stems", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True,
              help="CSV с колонкой stem.")  # fmt: skip
@click.option("--combo", "combos", multiple=True, required=True,
              help="Вариант и источник рамок через «@»: «C+F+2@строки», «A+F+3@нет».")  # fmt: skip
@click.option("--gt", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Эталон линии (stem, x, y): ошибки вариантов в errors.csv.")  # fmt: skip
@click.option("--overlays/--no-overlays", default=False, help="Рисовать оверлеи (overlays/<кадр>_<кусок>.jpg).")
@click.option("--tag", default="run", show_default=True, help="Имя прогона: <tag>_errors.csv, <tag>_spread.csv.")
@click.option("--jobs", default=16, show_default=True)
def fold_variants(
    previews: Path,
    out_dir: Path,
    stems: Path,
    combos: tuple[str, ...],
    gt: Path | None,
    overlays: bool,
    tag: str,
    jobs: int,
) -> None:
    """Сравнение вариантов линии сгиба: ошибки против эталона, расхождение вариантов по кадрам, оверлеи."""
    from research.spread_defects.fold_line import BoxSource, parse_variant

    parsed = []
    for combo in combos:
        name, _, source = combo.partition("@")
        parsed.append((parse_variant(name), BoxSource(source or "нет")))
    names = sorted(set(pd.read_csv(stems).stem))
    etalon = pd.read_csv(gt) if gt else pd.DataFrame(columns=["stem", "x", "y"])
    overlay_dir = out_dir / "overlays" / tag if overlays else None
    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (n, previews, out_dir / "layout", parsed, etalon[etalon.stem == n][["x", "y"]].values.tolist(), overlay_dir)
        for n in names
    ]
    errors, summaries = [], []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        for frame_errors, summary in pool.map(_fold_variants_one, tasks, chunksize=1):
            errors += frame_errors
            summaries.append(summary)
    pd.DataFrame(summaries).sort_values("spread_max", ascending=False).to_csv(
        out_dir / f"{tag}_spread.csv", index=False
    )
    if errors:
        table = pd.DataFrame(errors)
        table.to_csv(out_dir / f"{tag}_errors.csv", index=False)
        table["abs"] = table.err.abs()
        frames = table.groupby(["variant", "stem"]).err
        report = table.groupby("variant").agg(
            median=("abs", "median"), p90=("abs", lambda a: a.quantile(0.9)), max=("abs", "max")
        )
        report["кадров_ниже_8"] = frames.agg(lambda e: (e > 8).any()).groupby("variant").mean()
        report["кадров_выше_15"] = frames.agg(lambda e: (e < -15).any()).groupby("variant").mean()
        click.echo(report.round(3).to_string())
    click.echo(f"кадров {len(names)} → {out_dir}")


# --- Пиксельные признаки у складки и оверлеи «линия разреза + буквы» ------------------------------------------------

# Цвета оверлея (BGR): линия сгиба, граница видимой страницы (маска загиба), строки surya, знаки, крайний знак
# без просвета и с просветом.
_FOLD_BGR, _VISIBLE_BGR, _LINE_BGR = (0, 0, 220), (0, 140, 255), (150, 150, 150)
_GLYPH_BGR, _CUT_BGR, _GAP_BGR = (200, 150, 60), (60, 20, 230), (60, 170, 30)
_SHADOW_BGR = (160, 0, 160)


def _cut_overlay(
    gray: np.ndarray, record: dict, edges: list, keep: float, header: list[str], out_prefix: Path, pieces: int = 4
) -> None:
    """Оверлеи ленты у складки: линия сгиба, граница видимой страницы, знаки, крайний знак строки (срезан / просвет).

    Args:
        gray: Лента (полутон, прямой текст).
        record: Запись кэша surya кадра (``fold_col``, ``side``).
        edges: ``LineEdge`` строк, касающихся складки.
        keep: Запас маски загиба, px.
        header: Строки шапки.
        out_prefix: Префикс файлов (``<prefix>_<кусок>.jpg``).
        pieces: На сколько кусков резать ленту вдоль складки.
    """
    from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed
    from research.spread_defects.single_page_gutter import NO_GAP, Side

    side = Side(record["side"])
    fold_col = np.asarray(record["fold_col"])
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    layer = canvas.copy()
    for edge in edges:
        x0, y0, x1, y1 = (int(v) for v in edge.box)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), _LINE_BGR, 1)
        for n, (gx0, gy0, gx1, gy1) in enumerate(edge.glyphs):
            if n == edge.edge:
                colour = _CUT_BGR if edge.gap < NO_GAP else _GAP_BGR
                cv2.rectangle(layer, (gx0, gy0), (gx1, gy1), colour, -1)
            else:
                cv2.rectangle(layer, (gx0, gy0), (gx1, gy1), _GLYPH_BGR, -1)
        if np.isfinite(edge.shadow):
            # Начало тени складки на строке — короткая вертикальная риска.
            cv2.line(canvas, (int(edge.shadow), y0), (int(edge.shadow), y1), _SHADOW_BGR, 2)
        if edge.edge >= 0:
            # Подпись просвета у крайнего знака — со стороны страницы, чтобы не закрывать складку.
            gx0, gy0, gx1, _ = edge.glyphs[edge.edge]
            tx = gx0 - 150 if side is Side.LEFT else gx1 + 8
            cv2.putText(
                canvas, f"{edge.gap:+.2f}", (max(0, tx), gy0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1
            )
    canvas = cv2.addWeighted(layer, 0.35, canvas, 0.65, 0)
    rows = np.arange(len(fold_col))
    fold_pts = np.stack([fold_col, rows], 1).astype(np.int32)
    cv2.polylines(canvas, [fold_pts], False, _FOLD_BGR, 2)
    shift = keep if side is Side.RIGHT else -keep
    for start in range(0, len(rows), 24):  # пунктир границы видимой страницы
        seg = fold_pts[start : start + 12].copy()
        seg[:, 0] = (fold_col[start : start + 12] + shift).astype(np.int32)
        cv2.polylines(canvas, [seg], False, _VISIBLE_BGR, 2)
    # Зона у складки: 900 px со стороны страницы.
    fx = int(np.median(fold_col))
    xa, xb = (max(0, fx - 900), min(canvas.shape[1], fx + 60)) if side is Side.LEFT else (max(0, fx - 60), fx + 900)
    legend = [
        LegendEntry("линия сгиба (разреза)", _FOLD_BGR),
        LegendEntry("граница видимой страницы (маска загиба)", _VISIBLE_BGR, style=SampleStyle.DASHED),
        LegendEntry("строка surya", _LINE_BGR),
        LegendEntry("начало тени складки на строке (граница видимой страницы)", _SHADOW_BGR),
        LegendEntry("знаки у складки", _GLYPH_BGR, 0.35, SampleStyle.BOX),
        LegendEntry(
            f"крайний знак без просвета до тени (< {NO_GAP} высоты строки) — срезан", _CUT_BGR, 0.35, SampleStyle.BOX
        ),
        LegendEntry("крайний знак с просветом", _GAP_BGR, 0.35, SampleStyle.BOX),
    ]
    step = int(np.ceil(canvas.shape[0] / pieces))
    for k in range(pieces):
        part = np.ascontiguousarray(canvas[k * step : (k + 1) * step, xa:xb])
        image = framed(
            part, [*header, f"кусок {k + 1}/{pieces} вдоль складки; подписи — просвет в высотах строки"], legend
        )
        cv2.imwrite(str(out_prefix.with_name(f"{out_prefix.name}_{k}.jpg")), image, [cv2.IMWRITE_JPEG_QUALITY, 85])


def _pixels_one(task: tuple) -> dict:
    """Пиксельные признаки кадра (воркер пула): лента из полного RAW, края строк у складки, оверлей по запросу.

    Args:
        task: (имя кадра, JSON кэша surya, папка RAF, ширина превью, папка оверлеев или None, строки шапки).

    Returns:
        Строка признаков (``stem`` + поля ``PixelFeatures``) или ``{"stem", "error"}``.
    """
    import json

    from research.spread_defects.single_page_gutter import (
        FOLD_KEEP,
        STRIP_DEPTH,
        TOUCH_LINE,
        Side,
        fold_column,
        gutter_strip,
        line_edge,
        line_margin,
        pixel_features,
        read_raw,
    )

    stem, cache, raw_dir, preview_width, overlay_dir, header = task
    record = json.loads(cache.read_text())
    if record.get("error"):
        return {"stem": stem, "error": record["error"]}
    try:
        full = read_raw(raw_dir / f"{stem}.RAF")
        k = full.shape[1] / preview_width
        side = Side(record["side"])
        keep = FOLD_KEEP * k
        strip = gutter_strip(full, np.asarray(record["fold"]), side, depth=int(STRIP_DEPTH * k), keep=keep)
    except Exception as error:  # битый файл не должен ронять прогон
        return {"stem": stem, "error": repr(error)}
    gray = cv2.cvtColor(strip.image, cv2.COLOR_RGB2GRAY)
    lines = [line for line in record["lines"] if len(line["text"].strip()) >= 3]
    if not lines:
        return {"stem": stem, "touching": 0}
    height = float(np.median([line["bbox"][3] - line["bbox"][1] for line in lines]))
    column = fold_column(lines, strip.fold_col, side, height)
    touching = [line for line in column if line_margin(line, strip.fold_col, side) < TOUCH_LINE * height]
    edges = [line_edge(gray, line, strip.fold_col, side, height, keep) for line in touching]
    features = pixel_features(edges)
    if overlay_dir is not None:
        summary = (f"касаются складки {features.touching}, медиана просвета до тени {features.gap_median:+.2f}, "
                   f"узкий крайний знак {features.narrow_share:.2f}, контраст у складки {features.contrast_median:.2f}")  # fmt: skip
        _cut_overlay(gray, {**record, "fold_col": strip.fold_col}, edges, keep, [*header, summary], overlay_dir / stem)
    return {"stem": stem, **{name: getattr(features, name) for name in features.__dataclass_fields__}}


@main.command("page-gutter-pixels")
@click.option("--out-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True,
              help="Выход page-gutter (ocr_surya/ — кэш чтения по RAW, features.csv); пишется pixels.csv.")  # fmt: skip
@click.option("--raw-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--preview-width", default=4416, show_default=True, help="Ширина превью (линия сгиба снята по нему).")
@click.option("--overlay-stems", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="CSV с колонкой stem (и необязательными label, note): по этим кадрам — оверлеи overlays_cut/.")  # fmt: skip
@click.option("--only-overlays", is_flag=True, help="Считать только кадры из --overlay-stems (pixels.csv не писать).")
@click.option("--jobs", default=12, show_default=True,
              help="Воркеров (CPU): декодирование RAW ~3 с и ~1 ГБ памяти на кадр.")  # fmt: skip
def page_gutter_pixels(
    out_dir: Path, raw_dir: Path, preview_width: int, overlay_stems: Path | None, only_overlays: bool, jobs: int
) -> None:
    """Пиксельные признаки у складки (просвет крайнего знака, ширина, затемнение) и оверлеи «линия разреза + буквы»."""
    cache_dir = out_dir / "ocr_surya"
    features = pd.read_csv(out_dir / "features.csv").set_index("stem")
    marked = pd.read_csv(overlay_stems).set_index("stem") if overlay_stems else pd.DataFrame()
    stems = list(marked.index) if only_overlays else [s for s in features.index if (cache_dir / f"{s}.json").exists()]
    overlay_dir = out_dir / "overlays_cut"
    overlay_dir.mkdir(exist_ok=True)
    tasks = []
    for stem in stems:
        header = None
        if stem in marked.index:
            label = marked.loc[stem].get("label", "") if "label" in marked.columns else ""
            share = features.loc[stem, "fold_share"] if stem in features.index else float("nan")
            header = [f"{stem}: разметка «{label}»; доля незнакомых у сгиба (RAW) {share:.2f}"]
        tasks.append(
            (stem, cache_dir / f"{stem}.json", raw_dir, preview_width, overlay_dir if header else None, header)
        )
    rows = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=_init_worker) as pool:
        for n, row in enumerate(pool.map(_pixels_one, tasks, chunksize=1), 1):
            rows.append(row)
            if n % 50 == 0:
                click.echo(f"{n}/{len(tasks)}")
    if not only_overlays:
        pd.DataFrame(rows).to_csv(out_dir / "pixels.csv", index=False)
    click.echo(f"кадров {len(rows)} → {out_dir / 'pixels.csv' if not only_overlays else overlay_dir}")
