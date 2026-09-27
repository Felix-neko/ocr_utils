"""Команды стенда: ``collect`` — признаки формул по паку, ``sample`` — выборка по слоям, ``view``/``snap``/``verdict`` — разметка глазами, ``evaluate`` — метрики, ``overlays`` — картинки по трём папкам."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
import threadpoolctl

from research.surya_equations.annotate import grid_view
from research.surya_equations.evaluate import bootstrap, estimate, match_page, per_page
from research.surya_equations.labels import (
    BLOCKS_FILE,
    SAMPLE_FILE,
    VERDICTS_FILE,
    Block,
    PageLabels,
    block_row,
    load_labels,
    snap_to_ink,
    parse_verdict,
    upsert_jsonl,
)
from research.surya_equations.overlay import Category, draw_view, write_page
from research.surya_equations.sources import Stratum, collect_page, load_gray, read_equations

logger = logging.getLogger(__name__)

PAGES_CSV = "pages.csv"
# Выборка для разметки (слой → число полос; None — слой целиком; семя 0). Отчёт: reports/surya_equations.md.
SAMPLE_PLAN = {
    Stratum.BOTH: 30,
    Stratum.DEEPSEEK_ONLY: None,
    Stratum.SURYA_ONLY: None,
    Stratum.INLINE_FRAC_ONLY: 20,
    Stratum.NO_SIGNAL: 24,
}
# Поле вокруг блоков и боксов на проверочной картинке ``snap`` (единиц кадра).
SNAP_VIEW_PAD = 15


def init_worker() -> None:
    """Инициализатор воркера: hugepage и потоки BLAS/OpenCV выключены (``.claude/rules/gpu_and_pools.md``)."""
    np._core.multiarray._set_madvise_hugepage(False)
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)


def _pool(jobs: int) -> ProcessPoolExecutor:
    """Пул процессов forkserver с инициализатором."""
    return ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker)


def _layout_dir(layout_cache: Path) -> Path:
    """Вариант кэша surya, с которым сверяется эталон: заострённые JPEG — та же картинка, что у DeepSeek."""
    return layout_cache / "sharpened"


def _collect_one(layout_dir: Path, pages_dir: Path, page: str) -> dict:
    """Обёртка для пула: строка ``pages.csv`` одной полосы."""
    signals = collect_page(layout_dir, pages_dir, page)
    return {
        "page": page,
        "stratum": signals.stratum.value,
        "equations": len(signals.equations),
        "display": signals.display,
        "inline": signals.inline,
        "inline_frac": signals.inline_frac,
        "confidences": json.dumps([round(e.confidence, 4) for e in signals.equations]),
    }


@click.group()
def main() -> None:
    """Стенд «как surya layout находит выносные формулы» (эталон глазами, подсказки DeepSeek)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@main.command("collect")
@click.option("--layout-cache", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--pages-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--jobs", default=16, show_default=True, type=int, help="Воркеров (чтение JSON и markdown).")
def collect_command(layout_cache: Path, pages_dir: Path, out_dir: Path, jobs: int) -> None:
    """Признаки формул по всем полосам кэша: боксы surya Equation и ``$$`` DeepSeek → ``pages.csv``."""
    layout_dir = _layout_dir(layout_cache)
    pages = sorted(str(p.relative_to(layout_dir).with_suffix("")) for p in layout_dir.glob("*/*/*.json"))
    with _pool(jobs) as pool:
        rows = list(pool.map(_collect_one, [layout_dir] * len(pages), [pages_dir] * len(pages), pages, chunksize=64))
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / PAGES_CSV, index=False)
    logger.info("полос %d; по слоям: %s", len(table), table.stratum.value_counts().to_dict())


@main.command("sample")
@click.option("--out-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--force", is_flag=True, help="Перезаписать labels/sample.csv (разметка по старой выборке потеряет смысл)."
)
def sample_command(out_dir: Path, force: bool) -> None:
    """Выборка полос для разметки по слоям (``SAMPLE_PLAN``) → ``labels/sample.csv``."""
    if SAMPLE_FILE.exists() and not force:
        raise click.ClickException(f"{SAMPLE_FILE} уже есть; разметка сделана по ней (--force — перезаписать)")
    table = pd.read_csv(out_dir / PAGES_CSV)
    parts = []
    for stratum, size in SAMPLE_PLAN.items():
        rows = table[table.stratum == stratum.value]
        parts.append(rows if size is None else rows.sample(size, random_state=0))
    pd.concat(parts)[["page", "stratum"]].sort_values("page").to_csv(SAMPLE_FILE, index=False)


@main.command("view")
@click.argument("page")
@click.argument("out", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--sharpened-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--layout-cache", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--region", nargs=4, type=float, default=None, help="Область x0 y0 x1 y1 в кадре кэша (сетка через 25).")
def view_command(page: str, out: Path, sharpened_dir: Path, layout_cache: Path, region) -> None:
    """Полоса или область с сеткой координат кадра — для разметки вслепую (боксов surya нет)."""
    frame, _ = read_equations(_layout_dir(layout_cache) / f"{page}.json")
    grid_view(load_gray(sharpened_dir, page), frame[0], tuple(region) if region else None).save(out)


@main.command("snap")
@click.argument("page")
@click.argument("rough_json")
@click.argument("out", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--sharpened-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--layout-cache", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
def snap_command(page: str, rough_json: str, out: Path, sharpened_dir: Path, layout_cache: Path) -> None:
    """Эталон полосы из грубых рамок: ужать по краске, записать в ``labels/blocks.jsonl``, нарисовать проверку.

    ROUGH_JSON — список ``[x0, y0, x1, y1]`` или ``[x0, y0, x1, y1, {"lines": n, "inline": false, "note": ""}]``;
    ``[]`` — формул на полосе нет.
    """
    frame, equations = read_equations(_layout_dir(layout_cache) / f"{page}.json")
    gray = load_gray(sharpened_dir, page)
    scale = gray.shape[1] / frame[0]
    blocks = []
    for item in json.loads(rough_json):
        meta = item[4] if len(item) > 4 else {}
        rough = tuple(float(v) for v in item[:4])
        blocks.append(
            Block(
                snap_to_ink(gray, rough, scale),
                rough,
                meta.get("lines"),
                bool(meta.get("inline")),
                meta.get("note", ""),
            )
        )
    upsert_jsonl(BLOCKS_FILE, block_row(page, blocks))
    click.echo(json.dumps([b.box for b in blocks]))
    # Проверочная картинка: вердиктов ещё нет — все боксы surya рисуются как «на формуле».
    labels = PageLabels(page, tuple(blocks), {j: parse_verdict("formula") for j in range(len(equations))})
    boxes = [b.box for b in blocks] + [e.box for e in equations]
    region = (
        (
            min(b[0] for b in boxes) - SNAP_VIEW_PAD,
            min(b[1] for b in boxes) - SNAP_VIEW_PAD,
            max(b[2] for b in boxes) + SNAP_VIEW_PAD,
            max(b[3] for b in boxes) + SNAP_VIEW_PAD,
        )
        if boxes
        else (0, 0, frame[0], frame[1])
    )
    zoom = min(1.0, 1300 / ((region[2] - region[0]) * scale), 1800 / ((region[3] - region[1]) * scale))
    cv2.imwrite(str(out), draw_view(gray, scale, region, zoom, labels, equations, page, 3))


@main.command("verdict")
@click.argument("page")
@click.argument("verdicts_json")
def verdict_command(page: str, verdicts_json: str) -> None:
    """Вердикты боксам surya полосы → ``labels/verdicts.jsonl``.

    VERDICTS_JSON — ``{"S0": "formula", "S1": "not_formula: таблица"}``; значения — ``labels.Verdict``,
    после двоеточия комментарий о качестве бокса. ``{}`` — боксов нет.
    """
    verdicts = json.loads(verdicts_json)
    for value in verdicts.values():
        parse_verdict(value)  # неизвестный вердикт — ошибка сразу, а не при evaluate
    upsert_jsonl(VERDICTS_FILE, {"page": page, "verdicts": verdicts})


@main.command("evaluate")
@click.option("--layout-cache", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
def evaluate_command(layout_cache: Path, out_dir: Path) -> None:
    """Метрики против эталона: ``eval_blocks.csv``, ``eval_boxes.csv``, ``summary.json`` и сводка в лог."""
    layout_dir = _layout_dir(layout_cache)
    sample = pd.read_csv(SAMPLE_FILE)
    labels = load_labels()
    missing = sorted(set(sample.page) - set(labels))
    if missing:
        raise click.ClickException(f"не размечено {len(missing)} полос: {missing[:5]}…")
    block_rows, box_rows = [], []
    for row in sample.itertuples():
        _, equations = read_equations(layout_dir / f"{row.page}.json")
        blocks, boxes = match_page(row.page, Stratum(row.stratum), labels[row.page], equations)
        block_rows += blocks
        box_rows += boxes
    blocks, boxes = pd.DataFrame(block_rows), pd.DataFrame(box_rows)
    blocks.to_csv(out_dir / "eval_blocks.csv", index=False)
    boxes.to_csv(out_dir / "eval_boxes.csv", index=False)
    sizes = pd.read_csv(out_dir / PAGES_CSV).stratum.value_counts().to_dict()
    pages = per_page(sample, blocks, boxes)
    pack = estimate(pages, sizes)
    (recall_lo, recall_hi), (precision_lo, precision_hi) = bootstrap(pages, sizes)
    display = blocks[~blocks.inline]
    found = display[display.detected]
    quality = found[["iou", "block_cover", "excess", "left_mm", "top_mm", "right_mm", "bottom_mm"]]
    summary = {
        "sample_pages": len(sample),
        "display_blocks": len(display),
        "display_detected": int(display.detected.sum()),
        "inline_blocks": int(blocks.inline.sum()),
        "inline_detected": int(blocks[blocks.inline].detected.sum()),
        "boxes": len(boxes),
        "boxes_by_verdict": boxes.verdict.value_counts().to_dict(),
        "by_stratum": display.groupby("stratum")
        .agg(blocks=("block", "size"), detected=("detected", "sum"))
        .to_dict("index"),
        "pack_recall": pack.recall,
        "pack_recall_90ci": [recall_lo, recall_hi],
        "pack_precision": pack.precision,
        "pack_precision_90ci": [precision_lo, precision_hi],
        "merged_boxes": int((boxes.n_blocks >= 2).sum()),
        "split_blocks": int((display.n_boxes >= 2).sum()),
        "quality_median": quality.median().round(3).to_dict(),
        "quality_p10": quality.quantile(0.1).round(3).to_dict(),
        "quality_p90": quality.quantile(0.9).round(3).to_dict(),
        "iou_ge_0_7": float((found.iou >= 0.7).mean()),
        "iou_lt_0_5": int((found.iou < 0.5).sum()),
        "bottom_cut_gt_0_3mm": int((found.bottom_mm < -0.3).sum()),
        "right_cut_gt_0_3mm": int((found.right_mm < -0.3).sum()),
        "recall_low_blocks": float(display[display.height_mm < 5.5].detected.mean()),
        "recall_tall_blocks": float(display[display.height_mm >= 5.5].detected.mean()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    click.echo(json.dumps(summary, ensure_ascii=False, indent=1, default=float))


def _overlay_one(sharpened_dir: Path, layout_dir: Path, out_dir: Path, page: str, labels: PageLabels) -> dict[str, int]:
    """Обёртка для пула: оверлеи одной полосы."""
    frame, equations = read_equations(layout_dir / f"{page}.json")
    return write_page(sharpened_dir, out_dir, page, frame, labels, equations)


@main.command("overlays")
@click.option("--sharpened-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--layout-cache", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--jobs", default=16, show_default=True, type=int, help="Воркеров (декодирование JPEG 600 dpi).")
def overlays_command(sharpened_dir: Path, layout_cache: Path, out_dir: Path, jobs: int) -> None:
    """Оверлеи размеченных полос по папкам «формула была / не было» × «surya нашла / не нашла»."""
    labels = load_labels()
    pages = sorted(labels)
    layout_dir = _layout_dir(layout_cache)
    with _pool(jobs) as pool:
        counts = list(
            pool.map(
                _overlay_one,
                [sharpened_dir] * len(pages),
                [layout_dir] * len(pages),
                [out_dir] * len(pages),
                pages,
                [labels[p] for p in pages],
            )
        )
    for category in Category:
        cases = sum(c.get(category.value, 0) for c in counts)
        page_count = sum(1 for c in counts if category.value in c)
        logger.info("«%s»: полос %d, случаев %d", category.value, page_count, cases)
