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


def _pages_one(frame: Frame, out_dir: Path) -> dict:
    """Номера страниц обеих полос кадра из колонтитула (воркер пула): json токенов и строка сводки.

    Args:
        frame: Кадр.
        out_dir: Папка прогона.

    Returns:
        Строка: ключ, выпуск, номер кадра, номера левой/правой полосы и их уверенность, x сгиба (полный масштаб).
    """
    import json

    from research.spread_defects.sequence import read_numbers

    row = {"key": frame.key, "issue": frame.issue, "index": frame.index, "path": str(frame.path)}
    try:
        left, right, tokens, fold_x = read_numbers(read_rgb(frame.path, reduce=2))
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
def pages(root: Path, only: tuple[str, ...], out_dir: Path, jobs: int, force: bool) -> None:
    """Номера страниц из колонтитула по всем кадрам пака (tesseract, верхняя полоса); сводка ``frames.csv``."""
    run_pool(iter_pack(root, only), _pages_one, out_dir, out_dir / "frames.csv", jobs, force, pages_path)


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
