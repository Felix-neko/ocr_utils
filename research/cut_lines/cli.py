"""Команды стенда дорисовки срезанных строк: пары кадров двух фотосетов и дорисовка по всем кадрам основы."""

from __future__ import annotations

import csv
import logging
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

import click

from research.cut_lines.frames import BASE_DIR, DONOR_DIR, OUT_ROOT, V2_WORK, frames_of, init_worker
from research.cut_lines.pairing import FramePair, pair_frames
from research.cut_lines.process import PageRow, process_frame
from research.cut_lines.fragments import LIBRARY_CAP, merge_samples, save_library
from research.cut_lines.process_extend import ExtendRow, PageVerdict, harvest_frame, process_frame_extend, read_set
from research.cut_lines.process_synth import LineRow, process_frame_synth

log = logging.getLogger(__name__)


def read_pairs(path: Path, base_dir: Path, donor_dir: Path) -> dict[str, FramePair]:
    """Пары из CSV ``pairs.csv``.

    Args:
        path: CSV.
        base_dir: Папка основы.
        donor_dir: Папка донора.

    Returns:
        Словарь «имя кадра основы → пара».
    """
    with path.open(encoding="utf-8") as fh:
        return {r["base"]: FramePair(base_dir / r["base"], donor_dir / r["donor"], float(r["similarity"]))
                for r in csv.DictReader(fh)}  # fmt: skip


def frame_task(args: tuple[Path, Path | None, Path, Path]) -> list[PageRow]:
    """Задача пула: дорисовка одного кадра (ошибка не роняет прогон).

    Args:
        args: ``(кадр основы, кадр донора или None, кэш v2, корень выхода)``.

    Returns:
        Строки сводки.
    """
    base, donor, v2_work, out_root = args
    try:
        return process_frame(base, donor, v2_work, out_root)
    except Exception as exc:  # кадр с ошибкой пишется как есть
        log.exception("%s: ошибка", base.name)
        shutil.copy2(base, out_root / "frames" / base.name)
        return [PageRow(base.name, donor.name if donor else "", 0, f"error: {exc!r}")]


@click.group()
@click.option("--log-level", default="INFO", show_default=True, help="Уровень логов.")
def main(log_level: str) -> None:
    """Стенд «дорисовка срезанных краем листа строк по второму фотосету»."""
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@main.command()
@click.option("--base-dir", type=click.Path(path_type=Path), default=BASE_DIR, show_default=True, help="Основа (1).")
@click.option("--donor-dir", type=click.Path(path_type=Path), default=DONOR_DIR, show_default=True, help="Донор (2).")
@click.option("--out-root", type=click.Path(path_type=Path), default=OUT_ROOT, show_default=True, help="Выход.")
@click.option("--jobs", default=16, show_default=True, help="Воркеров.")
def pairs(base_dir: Path, donor_dir: Path, out_root: Path, jobs: int) -> None:
    """Пары кадров основы и донора → ``<out-root>/pairs.csv``."""
    found = pair_frames(frames_of(base_dir), frames_of(donor_dir), jobs)
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "pairs.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["base", "donor", "similarity"])
        for p in found:
            writer.writerow([p.base.name, p.donor.name, f"{p.similarity:.4f}"])
    log.info("пар: %d", len(found))


@main.command("pairs-from-text")
@click.option("--compare-csv", type=click.Path(exists=True, path_type=Path), required=True,
              help="pairs_compare.csv стенда research/spread_ocr (пары разворотов по распознанному тексту).")  # fmt: skip
@click.option("--min-confidence", default=0.5, show_default=True, help="Ниже этой уверенности кадра пара не берётся.")
@click.option("--out-root", type=click.Path(path_type=Path), default=OUT_ROOT, show_default=True, help="Выход.")
def pairs_from_text(compare_csv: Path, min_confidence: float, out_root: Path) -> None:
    """Пары кадров по распознанному тексту (``research/spread_ocr``) → ``<out-root>/pairs.csv``.

    Сверка по тексту подтвердила все пары SIFT и нашла вторые снимки разворотов, которые монотонное выравнивание
    оставляло без пары (РИСК 1991/05: 17 кадров).
    """
    rows = []
    with compare_csv.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["text_pair"] and float(r["text_frame_confidence"] or 0) >= min_confidence:
                rows.append([f"{r['base_frame']}.jpg", f"{r['text_pair']}.jpg", r["text_frame_confidence"]])
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "pairs.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["base", "donor", "similarity"])
        writer.writerows(rows)
    log.info("пар по тексту: %d", len(rows))


@main.command()
@click.option("--base-dir", type=click.Path(path_type=Path), default=BASE_DIR, show_default=True, help="Основа (1).")
@click.option("--donor-dir", type=click.Path(path_type=Path), default=DONOR_DIR, show_default=True, help="Донор (2).")
@click.option("--v2-work", type=click.Path(path_type=Path), default=V2_WORK, show_default=True,
              help="Кэш scan_cropping v2 по основе.")  # fmt: skip
@click.option("--out-root", type=click.Path(path_type=Path), default=OUT_ROOT, show_default=True, help="Выход.")
@click.option("--keys", default="", help="Только эти кадры основы (имена без расширения через запятую).")
@click.option("--jobs", default=16, show_default=True, help="Воркеров.")
def run(base_dir: Path, donor_dir: Path, v2_work: Path, out_root: Path, keys: str, jobs: int) -> None:
    """Дорисовка срезанных строк по всем кадрам основы → ``frames/``, ``sheets/``, ``patches.csv``."""
    paired = read_pairs(out_root / "pairs.csv", base_dir, donor_dir)
    todo = frames_of(base_dir)
    if keys:
        todo = [p for p in todo if p.stem in set(keys.split(","))]
    tasks = [(p, paired[p.name].donor if p.name in paired else None, v2_work, out_root) for p in todo]
    t0 = time.perf_counter()
    rows: list[PageRow] = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for frame_rows in pool.map(frame_task, tasks):
            rows += frame_rows
            for r in frame_rows:
                log.info("%s стр.%d ← %s: %s, строк %d (%s)", r.frame, r.page, r.donor, r.status, r.lines, r.sides)
    with (out_root / "patches.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    patched = sorted({r.frame for r in rows if r.status == "patched"})
    log.info("готово за %.0f с: кадров %d, дорисовано кадров %d, строк %d", time.perf_counter() - t0, len(todo),
             len(patched), sum(r.lines for r in rows))  # fmt: skip


def synth_task(args: tuple[Path, Path | None, Path, Path]) -> list[LineRow]:
    """Задача пула: синтез нижних строк одного кадра (ошибка не роняет прогон).

    Args:
        args: ``(кадр основы, кадр донора или None, кэш v2, корень выхода)``.

    Returns:
        Строки сводки.
    """
    base, donor, v2_work, out_root = args
    try:
        return process_frame_synth(base, donor, v2_work, out_root)
    except Exception as exc:  # кадр с ошибкой пишется как есть
        log.exception("%s: ошибка", base.name)
        (out_root / "frames").mkdir(parents=True, exist_ok=True)
        shutil.copy2(base, out_root / "frames" / base.name)
        return [LineRow(base.name, donor.name if donor else "", 0, f"error: {exc!r}")]


@main.command()
@click.option("--base-dir", type=click.Path(path_type=Path), default=BASE_DIR, show_default=True, help="Основа (1).")
@click.option("--donor-dir", type=click.Path(path_type=Path), default=DONOR_DIR, show_default=True, help="Донор (2).")
@click.option("--v2-work", type=click.Path(path_type=Path), default=V2_WORK, show_default=True,
              help="Кэш scan_cropping v2 по основе.")  # fmt: skip
@click.option("--pairs-root", type=click.Path(path_type=Path), default=OUT_ROOT, show_default=True,
              help="Где лежит pairs.csv.")  # fmt: skip
@click.option("--out-root", type=click.Path(path_type=Path), required=True, help="Выход синтеза.")
@click.option("--keys", default="", help="Только эти кадры основы (имена без расширения через запятую).")
@click.option("--jobs", default=12, show_default=True, help="Воркеров.")
def synth(
    base_dir: Path, donor_dir: Path, v2_work: Path, pairs_root: Path, out_root: Path, keys: str, jobs: int
) -> None:
    """Синтез срезанных нижних строк из букв (донор — только геометрия и текст) → ``frames/``, ``sheets/``, ``lines.csv``."""
    paired = read_pairs(pairs_root / "pairs.csv", base_dir, donor_dir)
    todo = frames_of(base_dir)
    if keys:
        todo = [p for p in todo if p.stem in set(keys.split(","))]
    tasks = [(p, paired[p.name].donor if p.name in paired else None, v2_work, out_root) for p in todo]
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows: list[LineRow] = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for frame_rows in pool.map(synth_task, tasks):
            rows += frame_rows
            for r in frame_rows:
                log.info("%s стр.%d: %s %.0f | %s", r.frame, r.page, r.status, r.match, r.text)
    with (out_root / "lines.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    log.info("готово за %.0f с: кадров %d, набрано строк %d", time.perf_counter() - t0, len(todo),
             sum(r.status == "synth" for r in rows))  # fmt: skip


def extend_task(
    args: tuple[Path, Path | None, Path, Path, dict[tuple[str, str], PageVerdict], int, Path | None],
) -> list[ExtendRow]:
    """Задача пула: продление кромки и допечатка строк одного кадра (ошибка не роняет прогон).

    Args:
        args: ``(кадр основы, кадр донора или None, кэш v2, корень выхода, набор вердиктов, сдвиг кромки px,
            библиотека букв или None)``.

    Returns:
        Строки сводки.
    """
    base, donor, v2_work, out_root, verdicts, extend_px, library = args
    try:
        return process_frame_extend(base, donor, v2_work, out_root, verdicts, extend_px, library).rows
    except Exception as exc:  # кадр с ошибкой пишется как есть
        log.exception("%s: ошибка", base.name)
        (out_root / "frames").mkdir(parents=True, exist_ok=True)
        shutil.copy2(base, out_root / "frames" / base.name)
        return [ExtendRow(base.name, "", f"error: {exc!r}")]


@main.command()
@click.option("--base-dir", type=click.Path(path_type=Path), default=BASE_DIR, show_default=True, help="Основа (1).")
@click.option("--donor-dir", type=click.Path(path_type=Path), default=DONOR_DIR, show_default=True,
              help="Донор (2) — только для сверки текста нижних строк.")  # fmt: skip
@click.option("--v2-work", type=click.Path(path_type=Path), default=V2_WORK, show_default=True,
              help="Кэш scan_cropping v2 по основе (маски страниц, пальцы, LaMa-кадр).")  # fmt: skip
@click.option("--pairs-root", type=click.Path(path_type=Path), default=OUT_ROOT, show_default=True,
              help="Где лежит pairs.csv.")  # fmt: skip
@click.option("--set", "set_path", type=click.Path(exists=True, path_type=Path), required=True,
              help="Набор вердиктов по страницам: frame,page,status,texts.")  # fmt: skip
@click.option("--out-root", type=click.Path(path_type=Path), required=True, help="Выход.")
@click.option(
    "--keys", default="", help="Только эти кадры основы (имена без расширения через запятую); пусто — все кадры набора."
)
@click.option("--extend-mm", default=9.0, show_default=True, help="На сколько мм продлить кромку вниз.")
@click.option("--px-per-mm", default=16.0, show_default=True, help="Масштаб кадров основы, px/мм.")
@click.option("--library", type=click.Path(exists=True, path_type=Path), default=None,
              help="Общая библиотека букв подшивки (npz от команды harvest): буквы с соседних страниц.")  # fmt: skip
@click.option("--jobs", default=12, show_default=True, help="Воркеров.")
def extend(base_dir: Path, donor_dir: Path, v2_work: Path, pairs_root: Path, set_path: Path, out_root: Path, keys: str,extend_mm: float, px_per_mm: float, library: Path | None, jobs: int) -> None:  # fmt: skip
    """Продление нижней кромки листа с её изгибом и допечатка срезанных строк буквами той же подшивки по набору →
    ``frames/``, ``sheets/``, ``lines.csv``."""
    verdicts = read_set(set_path)
    paired = read_pairs(pairs_root / "pairs.csv", base_dir, donor_dir) if (pairs_root / "pairs.csv").exists() else {}
    wanted = set(keys.split(",")) if keys else {frame for frame, _ in verdicts}
    todo = [p for p in frames_of(base_dir) if p.stem in wanted]
    extend_px = int(round(extend_mm * px_per_mm))
    tasks = [(p, paired[p.name].donor if p.name in paired else None, v2_work, out_root, verdicts, extend_px, library)
             for p in todo]  # fmt: skip
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows: list[ExtendRow] = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for frame_rows in pool.map(extend_task, tasks):
            rows += frame_rows
            for r in frame_rows:
                log.info("%s %s: %s %.0f | %s", r.frame, r.page, r.status, r.match, r.text)
    with (out_root / "lines.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    log.info("готово за %.0f с: кадров %d, сдвиг %d px, набрано строк %d, под вопросом %d",
             time.perf_counter() - t0, len(todo), extend_px, sum(r.status == "synth" for r in rows),
             sum(r.status == "check" for r in rows))  # fmt: skip


def harvest_task(args: tuple[Path, Path]) -> dict:
    """Задача пула: буквы одного кадра (ошибка — пустой словарь)."""
    base, v2_work = args
    try:
        return harvest_frame(base, v2_work)
    except Exception:
        log.exception("%s: ошибка", base.name)
        return {}


@main.command()
@click.option("--base-dir", type=click.Path(path_type=Path), default=BASE_DIR, show_default=True, help="Основа (1).")
@click.option("--v2-work", type=click.Path(path_type=Path), default=V2_WORK, show_default=True,
              help="Кэш scan_cropping v2 по основе.")  # fmt: skip
@click.option("--out", "out_path", type=click.Path(path_type=Path), required=True, help="Файл библиотеки .npz.")
@click.option("--keys", default="", help="Только эти кадры (имена без расширения через запятую); пусто — все.")
@click.option("--jobs", default=12, show_default=True, help="Воркеров.")
def harvest(base_dir: Path, v2_work: Path, out_path: Path, keys: str, jobs: int) -> None:
    """Общая библиотека букв подшивки: буквы строк у нижней кромки всех кадров (по кругу из кадров, до
    LIBRARY_CAP образцов на букву) → npz для ``extend --library``."""
    todo = frames_of(base_dir)
    if keys:
        todo = [p for p in todo if p.stem in set(keys.split(","))]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        parts = list(pool.map(harvest_task, [(p, v2_work) for p in todo]))
    merged = merge_samples(parts, LIBRARY_CAP)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_library(merged, out_path)
    log.info("готово за %.0f с: кадров %d, букв %d, образцов %d", time.perf_counter() - t0, len(todo), len(merged),
             sum(len(v) for v in merged.values()))  # fmt: skip
