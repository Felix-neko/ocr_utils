"""Команды стенда: ``run`` — разбор кадров пака, ``select`` — отбор и папки симлинков, ``sheets`` — листы отсмотра."""

from __future__ import annotations

import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
import threadpoolctl

from research.edge_cut_pages.detect import FrameResult, Side, analyze, read_half
from research.edge_cut_pages.frames import Frame, iter_frames
from research.edge_cut_pages.overlay import SIDE_TEXT, frame_overlay, line_crop, marked, sheet

# Стороны, которые считаются «срезом краем листа» в отборе (корешок — отдельно, в links_by_side/spine).
EDGE_SIDES = (Side.BOTTOM, Side.TOP, Side.OUTER)
# Нижняя граница папки «у корешка, возможно» (links_by_side/spine_maybe), строк.
MIN_SPINE_MAYBE = 3
# Оверлей кадра пишется при срезе не у корешка или при стольких строках, срезанных у корешка.
SPINE_OVERLAY_MIN = 2
# Качество JPEG оверлеев и вырезок.
JPEG_QUALITY = 85


def init_worker() -> None:
    """Инициализатор воркера: без hugepage и потоков BLAS/OpenCV (см. ``.claude/rules/gpu_and_pools.md``)."""
    np._core.multiarray._set_madvise_hugepage(False)
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)


def frame_title(frame: Frame, result: FrameResult) -> str:
    """Шапка оверлея кадра.

    Args:
        frame: Кадр.
        result: Разбор.

    Returns:
        Строка.
    """
    counts = ", ".join(f"{SIDE_TEXT[side]} {result.count(side)}" for side in Side if result.count(side))
    return f"{frame.rel} — срезанных строк: {counts or 'нет'}; глифов {result.glyphs}"


def summary_row(frame: Frame, result: FrameResult) -> dict:
    """Строка сводки ``scores.csv`` по разбору кадра.

    Args:
        frame: Кадр.
        result: Разбор.

    Returns:
        Словарь колонок.
    """
    edge_lines = [line for line in result.lines if line.side in EDGE_SIDES]
    return {
        "path": str(frame.path),
        "rel": str(frame.rel),
        "journal": frame.journal,
        "year": frame.year,
        "issue": frame.issue,
        "frame": frame.name,
        "glyphs": result.glyphs,
        "colour_share": round(result.colour_share, 4),
        "is_cover": result.is_cover,
        **{f"n_{side.value}": result.count(side) for side in Side},
        "cut_lines": len(edge_lines),
        # Мера среза: сумма глифов, касающихся края, по строкам не у корешка; максимум по строке.
        "score": sum(line.touching for line in edge_lines),
        "max_touch": max((line.touching for line in edge_lines), default=0),
    }


def process_frame(frame: Frame, out_dir: Path) -> tuple[dict, list[dict]]:
    """Разбор кадра (воркер пула): сводка, строки, оверлей и вырезки строк, если есть срез.

    Args:
        frame: Кадр.
        out_dir: Папка прогона (``overlays/``, ``crops/`` внутри).

    Returns:
        Кортеж ``(строка сводки, срезанные строки)``; при сбое — строка с ``error``.
    """
    started = time.time()
    try:
        bgr = read_half(frame.path)
        result = analyze(bgr, keep_masks=True)
    except Exception as error:  # битый файл не должен ронять прогон по паку
        return {"path": str(frame.path), "rel": str(frame.rel), "error": repr(error)}, []
    row = summary_row(frame, result)
    lines = [
        {
            "rel": str(frame.rel),
            "side": line.side.value,
            "box": list(line.box),
            "glyphs": line.glyphs,
            "touching": line.touching,
        }
        for line in result.lines
    ]
    if result.lines:
        stem = Path(frame.journal, frame.year, frame.issue, frame.name)
        if row["cut_lines"] or row["n_spine"] >= SPINE_OVERLAY_MIN:
            target = out_dir / "overlays" / stem.with_suffix(".jpg")
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(target),
                frame_overlay(bgr, result, frame_title(frame, result)),
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
        # Вырезки всех срезанных строк (и у корешка) — материал листов отсмотра.
        picture = marked(bgr, result)
        crop_dir = out_dir / "crops" / stem
        crop_dir.mkdir(parents=True, exist_ok=True)
        for index, line in enumerate(result.lines):
            cv2.imwrite(
                str(crop_dir / f"{index:02d}_{line.side.value}.jpg"),
                line_crop(picture, line),
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
    row["seconds"] = round(time.time() - started, 2)
    row["error"] = ""
    return row, lines


@click.group()
def main() -> None:
    """Стенд «строки, срезанные краем листа» по фото-разворотам подшивок."""


@main.command()
@click.option(
    "--root", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True, help="Корень пака."
)
@click.option("--out", type=click.Path(file_okay=False, path_type=Path), required=True, help="Папка прогона.")
@click.option("--jobs", type=int, default=16, show_default=True, help="Число процессов.")
@click.option("--only", default="", help="Подстрока относительного пути: разобрать только такие кадры.")
@click.option("--limit", type=int, default=0, help="Не больше стольких кадров (0 — все).")
def run(root: Path, out: Path, jobs: int, only: str, limit: int) -> None:
    """Разбирает кадры пака: ``scores.csv`` (все кадры), ``lines.jsonl`` (срезанные строки), оверлеи и вырезки."""
    frames = [frame for frame in iter_frames(root) if only in str(frame.rel)]
    if limit:
        frames = frames[:limit]
    out.mkdir(parents=True, exist_ok=True)
    # Повторный прогон пишет всё заново: старые оверлеи и вырезки убираются, чтобы не смешивать версии.
    for sub in ("overlays", "crops"):
        shutil.rmtree(out / sub, ignore_errors=True)
    click.echo(f"кадров: {len(frames)}, процессов: {jobs}")
    rows, all_lines = [], []
    started = time.time()
    context = get_context("forkserver")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=context, initializer=init_worker) as pool:
        futures = [pool.submit(process_frame, frame, out) for frame in frames]
        for done, future in enumerate(as_completed(futures), 1):
            row, lines = future.result()
            rows.append(row)
            all_lines.extend(lines)
            if done % 100 == 0 or done == len(futures):
                rate = done / (time.time() - started)
                click.echo(f"{done}/{len(futures)} кадров, {rate:.1f} кадр/с")
    table = pd.DataFrame(rows).sort_values("rel")
    table.to_csv(out / "scores.csv", index=False)
    with open(out / "lines.jsonl", "w") as handle:
        for line in sorted(all_lines, key=lambda item: item["rel"]):
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    errors = table["error"].astype(str).str.len().gt(0).sum() if "error" in table else 0
    click.echo(
        f"готово: {len(table)} кадров, ошибок {errors}, срез не у корешка: {(table.get('cut_lines', 0) > 0).sum()}"
    )


def is_positive(row: pd.Series, min_score: int, min_outer: int, min_spine: int = 0) -> bool:
    """Решение по кадру: есть строки, срезанные краем листа (низ, верх, наружный край, корешок), и это не обложка.

    Args:
        row: Строка ``scores.csv``.
        min_score: Наименьшая сумма глифов у края по строкам, срезанным вдоль (низ, верх).
        min_outer: Наименьшее число строк, срезанных наружным краем.
        min_spine: Наименьшее число строк, срезанных у корешка (0 — корешок в отбор не ведёт).

    Returns:
        True — кадр в отбор.
    """
    if row["is_cover"]:
        return False
    along = row["n_bottom"] + row["n_top"]
    spine = min_spine > 0 and row["n_spine"] >= min_spine
    return bool((along > 0 and row["score"] >= min_score) or row["n_outer"] >= min_outer or spine)


def sides_of(row: pd.Series, min_spine: int) -> list[Side]:
    """Стороны среза кадра; корешок — только при ``min_spine`` строках и больше (одиночные касания у корешка часты
    и ненадёжны).

    Args:
        row: Строка ``scores.csv``.
        min_spine: Наименьшее число строк у корешка, чтобы корешок считался стороной среза (0 — не считать).

    Returns:
        Стороны по порядку :class:`Side`.
    """
    sides = []
    for side in Side:
        count = row[f"n_{side.value}"]
        if side is Side.SPINE:
            if min_spine > 0 and count >= min_spine:
                sides.append(side)
        elif count > 0:
            sides.append(side)
    return sides


def link(target: Path, source: Path) -> None:
    """Симлинк ``target`` → ``source`` (папки создаются, старый симлинк заменяется).

    Args:
        target: Путь симлинка.
        source: Исходный кадр.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    os.symlink(source, target)


@main.command()
@click.option(
    "--out", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True, help="Папка прогона."
)
@click.option("--min-score", type=int, default=3, show_default=True, help="Сумма глифов у края (низ/верх).")
@click.option("--min-outer", type=int, default=2, show_default=True, help="Строк, срезанных наружным краем.")
@click.option("--min-spine", type=int, default=0, show_default=True, help="Строк у корешка (0 — не отбирать).")
def select(out: Path, min_score: int, min_outer: int, min_spine: int) -> None:
    """Отбор: ``cut_pages.csv``, ``links/<журнал>/<год>/<выпуск>/<кадр>``, ``links_by_side/<сторона>/…``."""
    table = pd.read_csv(out / "scores.csv", keep_default_na=False)
    table = table[table["error"].astype(str) == ""]
    lines = pd.read_json(out / "lines.jsonl", lines=True) if (out / "lines.jsonl").stat().st_size else pd.DataFrame()
    for sub in ("links", "links_by_side"):
        shutil.rmtree(out / sub, ignore_errors=True)
    chosen = table[
        table.apply(is_positive, axis=1, min_score=min_score, min_outer=min_outer, min_spine=min_spine)
    ].copy()

    chosen["sides"] = chosen.apply(lambda row: ",".join(side.value for side in sides_of(row, min_spine)), axis=1)
    columns = [
        "path",
        "journal",
        "year",
        "issue",
        "frame",
        "sides",
        "score",
        "cut_lines",
        "n_bottom",
        "n_top",
        "n_outer",
        "n_spine",
    ]
    chosen[columns].to_csv(out / "cut_pages.csv", index=False)
    for _, row in chosen.iterrows():
        rel = Path(row["rel"])
        link(out / "links" / rel, Path(row["path"]))
        for side in sides_of(row, min_spine):
            link(out / "links_by_side" / side.value / rel, Path(row["path"]))
    # Кадры, где срез только у корешка (не обложка), — отдельная папка: в отбор не входят, но нужны для обзора.
    # Кадры с несколькими строками у корешка, не дотянувшие до отбора (MIN_SPINE_MAYBE ≤ n < min_spine), — отдельная
    # папка «возможно»: на поясе 3–7 строк примерно половина — настоящий срез (README).
    maybe = (table["n_spine"] >= MIN_SPINE_MAYBE) & (~table["is_cover"].astype(bool))
    spine_only = table[(~table.index.isin(chosen.index)) & maybe]
    for _, row in spine_only.iterrows():
        link(out / "links_by_side" / "spine_maybe" / Path(row["rel"]), Path(row["path"]))
    click.echo(
        f"отобрано {len(chosen)} из {len(table)} кадров (обложек {int(table['is_cover'].astype(bool).sum())}); "
        f"у корешка «возможно»: {len(spine_only)}; строк в lines.jsonl: {len(lines)}"
    )
    summary = chosen.groupby(["journal", "year", "issue"]).size()
    totals = table.groupby(["journal", "year", "issue"]).size()
    click.echo(pd.DataFrame({"отобрано": summary, "кадров": totals}).fillna(0).astype(int).to_string())


def crop_of(out: Path, rel: str, index: int, side: str) -> np.ndarray | None:
    """Вырезка срезанной строки из ``crops/``.

    Args:
        out: Папка прогона.
        rel: Относительный путь кадра.
        index: Номер строки в кадре (порядок ``lines.jsonl``).
        side: Сторона среза (значение :class:`Side`).

    Returns:
        Картинка BGR или None, если вырезки нет.
    """
    path = Path(rel)
    return cv2.imread(str(out / "crops" / path.parent / path.stem / f"{index:02d}_{side}.jpg"))


def short_name(rel: str) -> str:
    """Короткая подпись кадра: журнал/год/номер выпуска (первое слово папки)/кадр.

    Args:
        rel: Относительный путь кадра.

    Returns:
        Подпись.
    """
    journal, year, issue, name = Path(rel).parts
    return f"{journal}/{year}/{issue.split()[0]}/{Path(name).stem}"


def frame_tiles(out: Path, rels: list[str], best: pd.DataFrame) -> list[tuple[np.ndarray, str]]:
    """Вырезки «лучшей» строки каждого кадра с подписью для листа по кадрам.

    Args:
        out: Папка прогона.
        rels: Кадры (относительные пути).
        best: Лучшая строка кадра (индекс — ``rel``; колонки ``index``, ``side``, ``touching``).

    Returns:
        Пары (вырезка, подпись).
    """
    tiles = []
    for rel in rels:
        if rel not in best.index:
            continue
        item = best.loc[rel]
        crop = crop_of(out, rel, int(item["index"]), str(item["side"]))
        if crop is not None:
            tiles.append((crop, f"{short_name(rel)} · {SIDE_TEXT[Side(item['side'])]} {item['touching']}"))
    return tiles


@main.command()
@click.option(
    "--out", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True, help="Папка прогона."
)
@click.option("--per-sheet", type=int, default=40, show_default=True, help="Вырезок на лист.")
@click.option("--seed", type=int, default=0, show_default=True, help="Зерно случайной выборки.")
def sheets(out: Path, per_sheet: int, seed: int) -> None:
    """Листы отсмотра по поясам меры: случайные вырезки строк по стороне и поясу ``max_touch`` кадра."""
    lines = pd.read_json(out / "lines.jsonl", lines=True)
    lines["index"] = lines.groupby("rel").cumcount()
    target = out / "sheets"
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir()
    rng = np.random.default_rng(seed)
    # Пояса по числу касающихся глифов строки: 1–2, 3–5, 6–10, 11+.
    bands = [(1, 2), (3, 5), (6, 10), (11, 10_000)]
    for side in Side:
        for low, high in bands:
            part = lines[(lines["side"] == side.value) & lines["touching"].between(low, high)]
            if part.empty:
                continue
            take = part.iloc[rng.permutation(len(part))[:per_sheet]]
            tiles = []
            for _, item in take.iterrows():
                crop = crop_of(out, item["rel"], int(item["index"]), str(item["side"]))
                if crop is None:
                    continue
                tiles.append((crop, f"{short_name(item['rel'])} · {item['touching']}/{item['glyphs']}"))
            title = (
                f"{SIDE_TEXT[side]}, глифов у края {low}–{high if high < 10_000 else '∞'}: {len(take)} из {len(part)}"
            )
            picture = sheet(tiles, title=title)
            cv2.imwrite(str(target / f"{side.value}_{low:02d}.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            click.echo(f"{side.value} {low}–{high}: {len(part)} строк")
    # Листы по кадрам: на кадр одна вырезка — строка с наибольшим числом глифов у края; пояса по мере кадра.
    table = pd.read_csv(out / "scores.csv", keep_default_na=False)
    table = table[(table["error"].astype(str) == "") & (table["is_cover"].astype(str) != "True")]
    edge_lines = lines[lines["side"].isin([side.value for side in EDGE_SIDES])]
    best = edge_lines.sort_values("touching").groupby("rel").tail(1).set_index("rel")
    for low, high in [(1, 2), (3, 5), (6, 10), (11, 20), (21, 10_000)]:
        part = table[table["score"].between(low, high)]
        if part.empty:
            continue
        take = part.iloc[rng.permutation(len(part))[:per_sheet]]
        title = f"кадры, мера {low}–{high if high < 10_000 else '∞'}: {len(take)} из {len(part)} (лучшая строка кадра)"
        picture = sheet(frame_tiles(out, list(take["rel"]), best), title=title)
        cv2.imwrite(str(target / f"frames_score_{low:02d}.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        click.echo(f"кадры, мера {low}–{high}: {len(part)}")


__all__ = ["main", "process_frame", "summary_row", "is_positive"]
