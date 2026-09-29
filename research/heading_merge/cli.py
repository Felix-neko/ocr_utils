"""Команды стенда: ``run`` — пересчёт полос в варианте запретов, ``trace`` — виновники стыков, ``compare`` — «было | стало» по C, P, N."""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.page_layout.pack_analysis import final
from ocr_utils.page_layout.pack_analysis.run import list_tasks
from ocr_utils.page_layout.pack_analysis.stages import load_image, page_key
from ocr_utils.page_layout.text_blocks.columns import GutterMode
from ocr_utils.page_layout.text_blocks.page import AxisKind
from research.gutter_crossing.cli import _init_worker as _base_init
from research.heading_merge.gates import HeadingPatch, Variant
from research.heading_merge.measure import candidate_joints
from research.heading_merge.significant import SIGNIFICANT_PX, shift_of
from research.heading_merge.trace import TracePatch, culprits

logger = logging.getLogger(__name__)

SETS_DIR = Path(__file__).parent / "sets"
# Лист просмотра: ширина одной пары «было | стало» и доля высоты оверлея без легенды.
SHEET_PAIR_WIDTH = 1000
OVERLAY_BODY_SHARE = 0.86


class Group(str, Enum):
    """Выборка полосы в сводке: подтверждённый случай, прочий кандидат, нормальная полоса."""

    CONFIRMED = "C"
    PROBLEM = "P"
    NORMAL = "N"


def read_cases(path: Path) -> dict[str, str]:
    """Подтверждённые случаи: полоса → тип сращивания (первый встреченный)."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines()[1:]:
        if line.strip():
            page, kind = line.split("\t")[:2]
            out.setdefault(page, kind)
    return out


def read_pages(path: Path) -> list[str]:
    """Полосы списка по строке."""
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


# Последний разбор полосы в воркере: его забирает :func:`run_page` после ``reblock_page``.
_LAST: dict = {}
_TEXT_BLOCKS = final.text_blocks


def _capturing_text_blocks(*args, **kwargs):
    """``final.text_blocks`` с сохранением разбора для подсчёта стыков."""
    analysis, hints = _TEXT_BLOCKS(*args, **kwargs)
    _LAST["analysis"] = analysis
    return analysis, hints


def _init_worker(variant: str) -> None:
    """Воркер: hugepage и потоки BLAS выключены, подмены варианта установлены, разбор перехватывается."""
    _base_init()
    HeadingPatch(Variant(variant)).install()
    final.text_blocks = _capturing_text_blocks


def run_page(args: tuple) -> dict:
    """Полоса: пересчёт текстовых блоков (оверлей и JSON в ``out``) и сводка разбора.

    Args:
        args: ``(PageTask, корень прошлого разбора, корень выхода)``.

    Returns:
        Строка ``summary.csv``: оси, блоки, ряды, стыки-кандидаты (JSON) или ошибка.
    """
    task, source, out = args
    try:
        final.reblock_page(task, source, out, AxisKind.BODY, GutterMode.SHORT)
    except Exception as error:  # noqa: BLE001 — одна битая полоса не валит прогон
        logger.warning("%s: %s: %s", task.name, type(error).__name__, error)
        return {"page": task.name, "error": f"{type(error).__name__}: {error}"}
    analysis = _LAST.pop("analysis")
    joints = candidate_joints(analysis)
    return {
        "page": task.name,
        "axes": len(analysis.axes),
        "blocks": len(analysis.blocks),
        "rows": sum(len(b.rows) for b in analysis.blocks),
        "joints": len(joints),
        "joint_list": json.dumps(joints, ensure_ascii=False),
    }


@click.group()
def main() -> None:
    """Стенд «сращивание строк разного набора» (подпись + заголовок, шапка, оглавление)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@main.command()
@click.option("--sharpened-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option(
    "--source",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Разбор пака (pages/*.json).",
)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--pages", "pages_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--variant", type=click.Choice([v.value for v in Variant]), required=True)
@click.option("--jobs", default=16, show_default=True)
def run(sharpened_dir: Path, source: Path, out_dir: Path, pages_file: Path, variant: str, jobs: int) -> None:
    """Пересчитать полосы списка в варианте запретов → ``<out-dir>/<variant>/`` (pages, overlays, summary.csv)."""
    tasks = list_tasks(sharpened_dir, pages_file)
    out = out_dir / variant
    out.mkdir(parents=True, exist_ok=True)
    context = get_context("forkserver")
    with ProcessPoolExecutor(jobs, mp_context=context, initializer=_init_worker, initargs=(variant,)) as pool:
        results = list(pool.map(run_page, [(t, source, out) for t in tasks], chunksize=2))
    fields = ["page", "axes", "blocks", "rows", "joints", "joint_list", "error"]
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(results)
    logger.info("%s: полос %d, стыков %d", variant, len(results), sum(r.get("joints", 0) for r in results))


@main.command()
@click.option("--sharpened-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--source", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option(
    "--cases",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=SETS_DIR / "cases.tsv",
    show_default=True,
)
@click.option("--variant", type=click.Choice([v.value for v in Variant]), default=Variant.BASE.value, show_default=True)
@click.option("--kind", "kinds", multiple=True, help="Только случаи этих типов (как в cases.tsv).")
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True, help="TSV виновников.")
def trace(sharpened_dir: Path, source: Path, cases: Path, variant: str, kinds: tuple[str, ...], out: Path) -> None:
    """Под вариантом запретов: стыки-кандидаты подтверждённых полос и шаги, которые их склеили (последовательно)."""
    chosen = {page: kind for page, kind in read_cases(cases).items() if not kinds or kind in kinds}
    tasks = {t.name: t for t in list_tasks(sharpened_dir)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with HeadingPatch(Variant(variant)), TracePatch() as tracer, out.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            ["page", "type", "kind", "block", "row", "x_gap0", "x_gap1", "y", "h_left", "h_right", "culprits"]
        )
        for page, kind in sorted(chosen.items()):
            tracer.reset()
            record_final = json.loads((source / "pages" / f"{page_key(page)}.json").read_text())
            record = final.page_record(record_final)
            image = load_image(tasks[page], record["rotate_cw"])
            analysis, hints = final.text_blocks(image, record, record_final["objects"], AxisKind.BODY, GutterMode.SHORT)
            for joint in candidate_joints(analysis):
                found = culprits(tracer.log, joint, hints.barrier_lines)
                writer.writerow(
                    [page, kind, joint["kind"], joint["block"], joint["row"], round(joint["x_gap0"]),
                     round(joint["x_gap1"]), joint["y"], joint["h_left"], joint["h_right"], " | ".join(found) or "?"]
                )  # fmt: skip
                logger.info("%s %s %s → %s", page, kind, joint["kind"], " | ".join(found) or "?")


def _read_summary(path: Path) -> dict[str, dict]:
    """``summary.csv`` прогона: полоса → строка."""
    return {row["page"]: row for row in csv.DictReader(path.open())}


def _polygons(run_dir: Path, page: str) -> list:
    """Многоугольники блоков полосы из итогового JSON прогона."""
    data = json.loads((run_dir / "pages" / f"{page_key(page)}.json").read_text())
    return [block["polygon"] for block in data["text_blocks"]["blocks"]]


def _overlay(run_dir: Path, page: str) -> np.ndarray | None:
    """Оверлей полосы прогона (папка по классам — любая)."""
    found = list((run_dir / "overlays").rglob(f"{page_key(page)}.jpg"))
    return cv2.imread(str(found[0])) if found else None


def _pair(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Склейка «было | стало» одной высоты."""
    height = max(before.shape[0], after.shape[0])
    padded = [
        cv2.copyMakeBorder(im, 0, height - im.shape[0], 0, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        for im in (before, after)
    ]
    return np.hstack(padded)


def write_sheets(pairs: list[Path], out_dir: Path) -> None:
    """Листы 2×2 из склеек «было | стало» (без легенды оверлея), с именем полосы красным.

    Args:
        pairs: Файлы склеек.
        out_dir: Куда класть ``NN.jpg``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(pairs), 4):
        images = []
        for path in pairs[start : start + 4]:
            image = cv2.imread(str(path))
            image = image[: int(image.shape[0] * OVERLAY_BODY_SHARE)]
            scale = SHEET_PAIR_WIDTH / image.shape[1]
            image = cv2.resize(image, (SHEET_PAIR_WIDTH, int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            cv2.putText(image, path.stem, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            images.append(image)
        while len(images) < 4:
            images.append(np.full_like(images[0], 255))
        height = max(image.shape[0] for image in images)
        images = [
            cv2.copyMakeBorder(im, 0, height - im.shape[0], 0, 6, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            for im in images
        ]
        sheet = np.vstack([np.hstack(images[:2]), np.hstack(images[2:])])
        cv2.imwrite(str(out_dir / f"{start // 4:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


@main.command()
@click.option("--run-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--before", "before_name", default=Variant.BASE.value, show_default=True)
@click.option("--after", "after_name", required=True)
@click.option(
    "--cases",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=SETS_DIR / "cases.tsv",
    show_default=True,
)
@click.option(
    "--set-p",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=SETS_DIR / "set_p.txt",
    show_default=True,
)
def compare(run_dir: Path, before_name: str, after_name: str, cases: Path, set_p: Path) -> None:
    """Сводка «было → стало» по C (по типам), P, N; склейки и листы изменившихся полос в ``compare_<before>_<after>``."""
    before, after = _read_summary(run_dir / before_name / "summary.csv"), _read_summary(
        run_dir / after_name / "summary.csv"
    )
    confirmed, problem = read_cases(cases), set(read_pages(set_p))
    out = run_dir / f"compare_{before_name}_{after_name}"
    rows, pairs = [], defaultdict(list)
    for page in sorted(before):
        b, a = before[page], after.get(page)
        if a is None or b.get("error") or a.get("error"):
            continue
        group = Group.CONFIRMED if page in confirmed else (Group.PROBLEM if page in problem else Group.NORMAL)
        changed = _polygons(run_dir / before_name, page) != _polygons(run_dir / after_name, page)
        rows.append(
            {"page": page, "set": group.value, "type": confirmed.get(page, ""), "changed": int(changed),
             "joints_before": int(b["joints"]), "joints_after": int(a["joints"]), "blocks_before": int(b["blocks"]),
             "blocks_after": int(a["blocks"]), "axes_before": int(b["axes"]), "axes_after": int(a["axes"])}
        )  # fmt: skip
        if changed:
            first, second = _overlay(run_dir / before_name, page), _overlay(run_dir / after_name, page)
            if first is not None and second is not None:
                target = out / group.value / f"{page_key(page)}.jpg"
                target.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(target), _pair(first, second), [cv2.IMWRITE_JPEG_QUALITY, 80])
                pairs[group].append(target)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "compare.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for group, paths in pairs.items():
        write_sheets(sorted(paths), out / f"sheets_{group.value}")
    for group in Group:
        part = [r for r in rows if r["set"] == group.value]
        click.echo(
            f"{group.value}: полос {len(part)}, изменилось {sum(r['changed'] for r in part)}, "
            f"стыков {sum(r['joints_before'] for r in part)} → {sum(r['joints_after'] for r in part)}, "
            f"полос со стыками {sum(r['joints_before'] > 0 for r in part)} → {sum(r['joints_after'] > 0 for r in part)}, "
            f"блоков {sum(r['blocks_before'] for r in part)} → {sum(r['blocks_after'] for r in part)}, "
            f"осей {sum(r['axes_before'] for r in part)} → {sum(r['axes_after'] for r in part)}"
        )
    left = Counter(r["type"] for r in rows if r["set"] == Group.CONFIRMED.value and r["joints_after"] > 0)
    total = Counter(r["type"] for r in rows if r["set"] == Group.CONFIRMED.value)
    for kind in sorted(total):
        click.echo(f"  C/{kind}: полос {total[kind]}, со стыками после — {left[kind]}")


@main.command()
@click.option("--run-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--before", "before_name", default=Variant.BASE.value, show_default=True)
@click.option("--after", "after_name", required=True)
def significant(run_dir: Path, before_name: str, after_name: str) -> None:
    """Заметные изменения (граница блока сдвинулась больше ``SIGNIFICANT_PX`` или сменилось число блоков) по группам; листы ``sheets_<группа>_big``."""
    out = run_dir / f"compare_{before_name}_{after_name}"
    rows = [row for row in csv.DictReader((out / "compare.csv").open()) if row["changed"] == "1"]
    for group in Group:
        pages = [
            row["page"]
            for row in rows
            if row["set"] == group.value
            and shift_of(run_dir / before_name, run_dir / after_name, row["page"]) > SIGNIFICANT_PX
        ]
        write_sheets(sorted(out / group.value / f"{page_key(p)}.jpg" for p in pages), out / f"sheets_{group.value}_big")
        click.echo(f"{group.value}: изменилось {sum(r['set'] == group.value for r in rows)}, заметно {len(pages)}")
        if group is Group.NORMAL:
            click.echo("  " + " ".join(pages))


if __name__ == "__main__":
    main()
