"""Команды стенда «разрез блока по смене выключки»: ``ocr`` (tesseract), ``candidates`` (концы статей на чтение), ``run`` (разбор способами), ``joints`` (стыки из черновиков), ``sheets`` (листы стыков), ``evaluate`` (оценка способов на множестве)."""

from __future__ import annotations

import csv
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import click
import cv2

from research.flush_split.bench import run_page
from research.flush_split.candidates import page_ends, render_end
from research.flush_split.labels import Expect, Joint, draft_joints, drop_conflicts, joint_cut, load_joints, save_joints
from research.flush_split.ocr import lines_of, load_lines, run_tesseract, save_lines
from research.flush_split.sheets import joint_sheet

logger = logging.getLogger(__name__)

# Вырезки холста Кунса ХиП 1991 (PNG + JSON с dpi), выпуски 01–07: `research/line_art_boxes canvas`.
CANVAS_DIR = Path("/mnt/hotstore/scan_processing/line_art_boxes/hp_1991/canvas")
OUT_DIR = Path("/mnt/hotstore/scan_processing/block_split/v7")
SETS_DIR = Path(__file__).parent / "sets"


def crops(canvas_dir: Path) -> list[Path]:
    """Вырезки страниц холста: ``<выпуск>/<кадр>_<L|R>.png``, по порядку.

    Args:
        canvas_dir: Папка вырезок.

    Returns:
        Пути PNG.
    """
    return sorted(canvas_dir.glob("*/*.png"))


def page_key(png: Path) -> str:
    """Ключ страницы ``<две цифры выпуска>/<кадр>_<L|R>`` (как в разметке прошлых стендов: ``01/IMG_1008_R``)."""
    return f"{png.parent.name[:2]}/{png.stem}"


def _ocr_one(args: tuple[Path, Path]) -> tuple[str, int, str]:
    """Воркер ``ocr``: tesseract одной вырезки → CSV строк.

    Args:
        args: ``(PNG вырезки, CSV итога)``.

    Returns:
        ``(ключ, число строк, ошибка или пусто)``.
    """
    png, out = args
    try:
        dpi = json.loads(png.with_suffix(".json").read_text())["dpi"]
        lines = lines_of(run_tesseract(png), dpi)
        save_lines(lines, out)
        return page_key(png), len(lines), ""
    except Exception as exc:  # noqa: BLE001 — одна плохая вырезка не роняет прогон
        return page_key(png), 0, repr(exc)


@click.group()
def main() -> None:
    """Стенд «разрез текстового блока по смене выключки»."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@main.command()
@click.option("--canvas-dir", type=click.Path(path_type=Path), default=CANVAS_DIR, show_default=True)
@click.option("--out-dir", type=click.Path(path_type=Path), default=OUT_DIR / "ocr", show_default=True)
@click.option("--jobs", type=int, default=12, show_default=True,
              help="Процессов tesseract (по потоку на процесс); 12 — 75 % ядер машины.")  # fmt: skip
def ocr(canvas_dir: Path, out_dir: Path, jobs: int) -> None:
    """Tesseract (rus, psm 3) по всем вырезкам: CSV строк с рамками в мм; готовые пропускаются."""
    tasks = []
    for png in crops(canvas_dir):
        out = out_dir / png.parent.name / f"{png.stem}.csv"
        if not out.exists():
            tasks.append((png, out))
    logger.info("вырезок к распознаванию: %d", len(tasks))
    errors = 0
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver")) as pool:
        for i, (key, n, err) in enumerate(pool.map(_ocr_one, tasks), 1):
            if err:
                errors += 1
                logger.warning("%s: %s", key, err)
            if i % 100 == 0:
                logger.info("готово %d / %d", i, len(tasks))
    logger.info("готово, ошибок %d", errors)


@main.command()
@click.option("--ocr-dir", type=click.Path(path_type=Path), default=OUT_DIR / "ocr", show_default=True)
@click.option("--out-dir", type=click.Path(path_type=Path), default=OUT_DIR / "candidates", show_default=True)
@click.option("--batch", type=int, default=40, show_default=True, help="Страниц в одном файле на чтение.")
def candidates(ocr_dir: Path, out_dir: Path, batch: int) -> None:
    """Концы статей и строки-подписи по строкам tesseract: хвосты текста на чтение (``candidates_NN.md``)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    reasons: dict[str, int] = {}
    for csv_path in sorted(ocr_dir.glob("*/*.csv")):
        key = f"{csv_path.parent.name[:2]}/{csv_path.stem}"
        ends = page_ends(load_lines(csv_path))
        if ends:
            pages.append((key, ends))
            for end in ends:
                reasons[end.reason] = reasons.get(end.reason, 0) + 1
    for start in range(0, len(pages), batch):
        chunk = pages[start : start + batch]
        text = "\n".join(render_end(key, end) for key, ends in chunk for end in ends)
        (out_dir / f"candidates_{start // batch:02d}.md").write_text(
            "Строка: `верх строки, мм | отступ слева, мм  отступ справа, мм |` текст\n\n" + text
        )
    logger.info("страниц с концами: %d, мест: %s", len(pages), reasons)


def _init_worker() -> None:
    """Воркер пула: без hugepage-пометки и без потоков BLAS/OpenCV (см. .claude/rules/gpu_and_pools.md)."""
    import cv2
    import numpy as np
    import threadpoolctl

    try:
        np._core.multiarray._set_madvise_hugepage(False)  # type: ignore[attr-defined]
    except AttributeError:
        pass
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)


def read_keys(path: Path | None) -> set[str] | None:
    """Ключи страниц из файла (по ключу в строке или CSV с колонкой ``page``); None — все вырезки."""
    if path is None:
        return None
    if path.suffix == ".csv":
        with path.open() as fh:
            return {row["page"] for row in csv.DictReader(fh)}
    return {line.split()[0] for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")}


@main.command()
@click.option("--canvas-dir", type=click.Path(path_type=Path), default=CANVAS_DIR, show_default=True)
@click.option("--pages", type=click.Path(path_type=Path), default=None,
              help="Ключи страниц (файл или CSV множества с колонкой page); без — все вырезки.")  # fmt: skip
@click.option("--modes", default="align_gap,align_right", show_default=True, help="Способы FlushSplit через запятую.")
@click.option("--out", type=click.Path(path_type=Path), required=True, help="JSONL сводок блоков.")
@click.option("--jobs", type=int, default=12, show_default=True, help="Процессов (12 — 75 % ядер).")
def run(canvas_dir: Path, pages: Path | None, modes: str, out: Path, jobs: int) -> None:
    """Разбор вырезок заданными способами; по строке JSON на страницу."""
    wanted = read_keys(pages)
    paths = [p for p in crops(canvas_dir) if wanted is None or page_key(p) in wanted]
    names = tuple(m.strip() for m in modes.split(",") if m.strip())
    logger.info("страниц: %d, способы: %s", len(paths), names)
    out.parent.mkdir(parents=True, exist_ok=True)
    pool = ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=_init_worker)
    with out.open("w") as fh, pool:
        for i, rec in enumerate(pool.map(run_page, [(p, names) for p in paths], chunksize=2), 1):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if "error" in rec:
                logger.warning("%s: %s", rec["page"], rec["error"])
            if i % 50 == 0:
                logger.info("готово %d / %d", i, len(paths))


def load_runs(path: Path) -> dict[str, dict]:
    """Записи прогона ``run`` по ключу страницы (с ошибкой — пропущены)."""
    out = {}
    with path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            if "error" not in rec:
                out[rec["page"]] = rec
    return out


@main.command()
@click.option("--drafts", type=click.Path(path_type=Path), multiple=True, required=True, help="CSV черновиков.")
@click.option("--ocr-dir", type=click.Path(path_type=Path), default=OUT_DIR / "ocr", show_default=True)
@click.option("--out", type=click.Path(path_type=Path), default=SETS_DIR / "hp1991_joints_draft.csv", show_default=True)
def joints(drafts: tuple[Path, ...], ocr_dir: Path, out: Path) -> None:
    """Стыки множества из черновиков разметки по тексту (разрез над блоком, один блок внутри и над ним)."""
    found, dropped = drop_conflicts([j for d in drafts for j in draft_joints(d, ocr_dir)])
    logger.info("противоречивых стыков «один блок» убрано: %d", dropped)
    save_joints(found, out)
    by = {}
    for j in found:
        by[(j.kind, j.expect.value)] = by.get((j.kind, j.expect.value), 0) + 1
    logger.info("стыков %d → %s: %s", len(found), out, by)


@main.command()
@click.option("--joints", "joints_path", type=click.Path(path_type=Path), required=True)
@click.option("--runs", type=click.Path(path_type=Path), required=True, help="JSONL прогона run.")
@click.option("--modes", default="align_gap,align_right", show_default=True)
@click.option("--out-dir", type=click.Path(path_type=Path), default=OUT_DIR / "sheets", show_default=True)
@click.option("--misses", type=click.Path(path_type=Path), default=None,
              help="CSV промахов evaluate: листы только по ним (любого ожидания) для способа --miss-mode.")  # fmt: skip
@click.option("--miss-mode", default="align_right", show_default=True)
def sheets(joints_path: Path, runs: Path, modes: str, out_dir: Path, misses: Path | None, miss_mode: str) -> None:
    """Листы стыков ``cut`` (или промахов): разборы способов в ряд вокруг стыка."""
    recs = load_runs(runs)
    names = tuple(m.strip() for m in modes.split(","))
    all_joints = load_joints(joints_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    wanted = None
    if misses is not None:
        with misses.open() as fh:
            wanted = {(r["page"], r["y_mm"]) for r in csv.DictReader(fh) if r["mode"] == miss_mode}
    for j in all_joints:
        if j.page not in recs:
            continue
        if wanted is not None:
            if (j.page, f"{j.y_mm:.1f}") not in wanted:
                continue
        elif j.expect is not Expect.CUT:
            continue
        page_joints = [k for k in all_joints if k.page == j.page]
        sheet = joint_sheet(recs[j.page], names, page_joints, j.y_mm)
        cv2.imwrite(
            str(out_dir / f"{j.page.replace('/', '_')}_{j.y_mm:05.1f}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]
        )
        n += 1
    logger.info("листов: %d → %s", n, out_dir)


@main.command()
@click.option("--joints", "joints_path", type=click.Path(path_type=Path), required=True)
@click.option("--runs", type=click.Path(path_type=Path), required=True, help="JSONL прогона run.")
@click.option("--modes", default="align_gap,align_right,align", show_default=True)
@click.option("--misses", type=click.Path(path_type=Path), default=None, help="CSV промахов по способам.")
def evaluate(joints_path: Path, runs: Path, modes: str, misses: Path | None) -> None:
    """Доля найденных разрезов и ложные разрезы по видам стыков, по способам."""
    recs = load_runs(runs)
    all_joints = [j for j in load_joints(joints_path) if j.page in recs]
    rows = []
    for mode in (m.strip() for m in modes.split(",")):
        stats: dict[tuple[str, str], list[int]] = {}
        for j in all_joints:
            cut = joint_cut(recs[j.page][mode], j)
            key = (j.expect.value, j.kind)
            s = stats.setdefault(key, [0, 0, 0])  # верно, неверно, ряд не найден
            if cut is None:
                s[2] += 1
            elif cut == (j.expect is Expect.CUT):
                s[0] += 1
            else:
                s[1] += 1
                rows.append([mode, j.page, f"{j.y_mm:.1f}", j.expect.value, j.kind, j.note])
        click.echo(f"== {mode}")
        for (expect, kind), (ok, bad, none) in sorted(stats.items()):
            click.echo(f"  {expect:4s} {kind:22s} верно {ok:3d}  неверно {bad:3d}  без ряда {none:3d}")
    if misses is not None:
        with misses.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["mode", "page", "y_mm", "expect", "kind", "note"])
            writer.writerows(rows)


def row_owner(blocks: list[dict]) -> dict[tuple[int, int], int]:
    """Номер блока каждого ряда по ключу ``(y, x0)`` (округлённые px рабочей копии)."""
    return {(round(r[0]), round(r[1])): bi for bi, b in enumerate(blocks) for r in b["rows"]}


def changed_joints(before: list[dict], after: list[dict]) -> list[tuple[float, float, str]]:
    """Стыки соседних рядов одного блока, разбитые или склеенные во втором разборе.

    Соседство — по рядам разбора «было»: ряд и ближайший ряд выше в том же блоке. Ряды без пары во втором разборе
    пропускаются.

    Args:
        before: Блоки первого разбора.
        after: Блоки второго разбора.

    Returns:
        ``(y стыка в мм, x нижнего ряда в мм, "cut" — новый разрез | "join" — новая склейка)``.
    """
    from research.flush_split.labels import MM_PER_INCH
    from ocr_utils.page_layout.text_blocks import WORK_DPI

    to_mm = MM_PER_INCH / WORK_DPI
    own_a, own_b = row_owner(before), row_owner(after)
    out = []
    for blocks, own_other, kind in ((before, own_b, "cut"), (after, own_a, "join")):
        for b in blocks:
            rows = sorted(b["rows"], key=lambda r: r[0])
            for upper, lower in zip(rows[:-1], rows[1:]):
                ka, kb = (round(upper[0]), round(upper[1])), (round(lower[0]), round(lower[1]))
                if ka in own_other and kb in own_other and own_other[ka] != own_other[kb]:
                    out.append(((lower[0] - lower[3] / 2) * to_mm, (lower[1] + lower[2]) / 2 * to_mm, kind))
    return out


@main.command()
@click.option("--before", type=click.Path(path_type=Path), required=True, help="JSONL прогона «было».")
@click.option("--before-mode", default="align_gap", show_default=True)
@click.option("--after", type=click.Path(path_type=Path), required=True, help="JSONL прогона «стало».")
@click.option("--after-mode", default="align_right", show_default=True)
@click.option("--joints", "joints_path", type=click.Path(path_type=Path), default=None,
              help="Множество: стыки страниц множества на листы не идут (они уже отсмотрены).")  # fmt: skip
@click.option("--out-dir", type=click.Path(path_type=Path), default=OUT_DIR / "compare", show_default=True)
def compare(before: Path, before_mode: str, after: Path, after_mode: str, joints_path: Path | None,
            out_dir: Path) -> None:  # fmt: skip
    """Изменённые стыки всех страниц: сводка и листы «было | стало» по каждому новому разрезу и склейке."""
    a, b = load_runs(before), load_runs(after)
    known = {j.page for j in load_joints(joints_path)} if joints_path else set()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, sheets_n = [], 0
    for page in sorted(set(a) & set(b)):
        events = changed_joints(a[page][before_mode], b[page][after_mode])
        for y, x, kind in events:
            rows.append([page, f"{y:.1f}", f"{x:.1f}", kind, int(page in known)])
            if page in known:
                continue
            rec = {**a[page], "before": a[page][before_mode], "after": b[page][after_mode]}
            fake = Joint(page, y, x, Expect.CUT if kind == "cut" else Expect.KEEP, kind)
            sheet = joint_sheet(rec, ("before", "after"), [fake], y)
            cv2.imwrite(str(out_dir / f"{kind}_{page.replace('/', '_')}_{y:05.1f}.jpg"), sheet,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])  # fmt: skip
            sheets_n += 1
    with (out_dir / "changes.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["page", "y_mm", "x_mm", "kind", "in_set"])
        writer.writerows(rows)
    cuts = sum(r[3] == "cut" for r in rows)
    logger.info("страниц: %d, новых разрезов %d, склеек %d, страниц с изменениями %d, листов %d",
                len(set(a) & set(b)), cuts, len(rows) - cuts, len({r[0] for r in rows}), sheets_n)  # fmt: skip
