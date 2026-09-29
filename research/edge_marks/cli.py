"""Команды стенда краевых пометок: ``select`` — выбросы концов строк на выровненных сторонах и листы для разметки."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import click
import cv2
import pandas as pd

from research.text_block_specks.pages import NOGEO_DIR

# Прошлый стенд соринок: концы строк всего пака и его разметка.
SPECKS_DIR = Path("/mnt/system/raw/mts/text_block_specks")
OUT_DIR = Path("/mnt/system/raw/mts/pack1_edge_marks")
# Сторона выровнена: доля концов «на кривой» не меньше этой.
ALIGNED_SHARE = 0.8
# Выброс наружу, мм: пояс, где встречается сор (дальше — почти всегда законная вёрстка, прошлый стенд).
MIN_OUT_MM, MAX_OUT_MM = 0.8, 6.0
# Блок не короче стольких строк.
MIN_ROWS = 8
SEED = 0
# Разметка стенда в git.
SETS_DIR = Path(__file__).resolve().parent / "sets"


def aligned_outliers(ends_csv: Path) -> pd.DataFrame:
    """Концы строк, торчащие наружу на ``MIN_OUT_MM``–``MAX_OUT_MM`` мм на выровненной стороне блока от ``MIN_ROWS`` строк.

    Args:
        ends_csv: ``ends.csv`` прошлого стенда (все концы строк пака, nogeo).

    Returns:
        Строки ``ends.csv`` с долей выровненных концов стороны ``on_share``.
    """
    ends = pd.read_csv(ends_csv)
    share = ends.groupby(["key", "block", "side"]).status.apply(lambda s: float((s == "on").mean())).rename("on_share")
    ends = ends.join(share, on=["key", "block", "side"])
    mask = (
        (ends.rows >= MIN_ROWS)
        & (ends.on_share >= ALIGNED_SHARE)
        & (ends.resid_mm <= -MIN_OUT_MM)
        & (ends.resid_mm >= -MAX_OUT_MM)
    )
    return ends[mask].copy()


def prior_labels() -> pd.DataFrame:
    """Разметка концов строк прошлого стенда: ``key, block, side, row, label`` (после перепроверки)."""
    candidates = pd.read_csv(SPECKS_DIR / "label" / "candidates.csv")
    labels = pd.read_csv(SPECKS_DIR / "label" / "labels_v2.csv")
    merged = candidates.merge(labels, on="id")
    return merged[["key", "block", "side", "row", "label"]]


@click.group()
def main() -> None:
    """Стенд краевых пометок и сора."""


@main.command()
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), default=OUT_DIR / "set")
@click.option("--sample", default=240, show_default=True, help="Сколько неразмеченных концов вынести на листы.")
@click.option("--pdf-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=NOGEO_DIR)
def select(out_dir: Path, sample: int, pdf_dir: Path) -> None:
    """Выбросы на выровненных сторонах (``outliers.csv``) с прошлой разметкой и листы неразмеченных (``sheets/``)."""
    from research.text_block_specks.sheets import contact_sheets, crops_for

    out_dir.mkdir(parents=True, exist_ok=True)
    found = aligned_outliers(SPECKS_DIR / "scan" / "ends.csv")
    found = found.merge(prior_labels(), on=["key", "block", "side", "row"], how="left")
    found = found.sort_values(["key", "block", "side", "row"]).reset_index(drop=True)
    found.insert(0, "id", [f"e{n:04d}" for n in range(1, len(found) + 1)])
    found.to_csv(out_dir / "outliers.csv", index=False)
    todo = found[found.label.isna()].to_dict("records")
    random.Random(SEED).shuffle(todo)
    todo = sorted(todo[:sample], key=lambda r: r["id"])
    crops = crops_for(todo, pdf_dir)
    sheets_dir = out_dir / "sheets"
    sheets_dir.mkdir(exist_ok=True)
    for number, sheet in enumerate(contact_sheets([crops[r["id"]] for r in todo])):
        cv2.imwrite(str(sheets_dir / f"sheet_{number:03d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    with (out_dir / "to_label.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "key", "resid_mm"])
        writer.writerows([[r["id"], r["key"], r["resid_mm"]] for r in todo])
    click.echo(
        f"выбросов {len(found)} на {found.key.nunique()} полосах; размечено ранее {found.label.notna().sum()}; "
        f"на листы {len(todo)} → {sheets_dir}"
    )


@main.command()
@click.option("--labels", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=SETS_DIR / "ends_labelled.csv")
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), default=OUT_DIR / "cand")
@click.option("--pdf-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=NOGEO_DIR)
def candidates(labels: Path, out_dir: Path, pdf_dir: Path) -> None:
    """Кандидаты по размеченным концам: рамки, вырезки ``a/``, ``b/``, ``b_erased/``, полосы ``pages/``, ``candidates.json``."""
    from research.edge_marks.candidates import build

    built = build(labels, SPECKS_DIR / "scan" / "ends.csv", out_dir, pdf_dir)
    junk = sum(1 for c in built if c.truth == "junk")
    click.echo(f"кандидатов {len(built)}: сор {junk}, знаков {len(built) - junk} → {out_dir}")


# Как запускать судей: интерпретатор и модуль/скрипт. Судьи в чужих окружениях — отдельные скрипты-воркеры.
PROJECT = Path(__file__).resolve().parents[2]
JUDGES: dict[str, list[str]] = {
    "rule": ["uv", "run", "python", "-m", "research.edge_marks.judges.rule"],
    "tesseract": ["uv", "run", "python", "-m", "research.edge_marks.judges.tess"],
    "deepseek": ["uv", "run", "python", "-m", "research.edge_marks.judges.deepseek"],
    "craft": ["/home/felix/Projects/mts_markup/line_axis_engines/craft/bin/python", "research/edge_marks/workers/craft_judge.py", "--fp16"],
    "doctr": ["/home/felix/Projects/mts_markup/line_axis_engines/doctr/bin/python", "research/edge_marks/workers/doctr_judge.py"],
    "pero": ["/mnt/system/raw/mts/curved_layout_engines/pero/bin/python", "research/edge_marks/workers/pero_judge.py"],
    "surya": ["uv", "run", "python", "-m", "research.edge_marks.judges.surya_judge"],
    "paddle": ["/home/felix/Projects/mts_markup/line_axis_engines/paddle/bin/python", "research/edge_marks/workers/paddle_judge.py"],
}


@main.command()
@click.option("--judge", "names", multiple=True, required=True, help="Имя судьи (ключ JUDGES) или имя=команда.")
@click.option("--cand-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=OUT_DIR / "cand")
@click.option("--results-dir", type=click.Path(file_okay=False, path_type=Path), default=OUT_DIR / "results")
@click.option("--extra", default="", help="Дополнительные аргументы судье (строкой).")
def judge(names: tuple[str, ...], cand_dir: Path, results_dir: Path, extra: str) -> None:
    """Прогнать судей по кандидатам с замером времени, видеопамяти и ОЗУ: ``<results>/<судья>.jsonl`` и ``_usage.json``."""
    import json
    import shlex
    from dataclasses import asdict

    from research.edge_marks.vram import run_measured

    results_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        label, _, command = name.partition("=")
        cmd = shlex.split(command) if command else JUDGES[label]
        out = results_dir / f"{label}.jsonl"
        full = [*cmd, "--cand-dir", str(cand_dir), "--out", str(out), *shlex.split(extra)]
        with (results_dir / f"{label}.log").open("w") as log:
            usage = run_measured(full, cwd=str(PROJECT), log=log)
        (results_dir / f"{label}_usage.json").write_text(json.dumps(asdict(usage)))
        click.echo(f"{label}: код {usage.returncode}, {usage.seconds:.0f} с, VRAM {usage.vram_mb:.0f} МБ, ОЗУ {usage.rss_mb:.0f} МБ")


@main.command()
@click.option("--cand-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=OUT_DIR / "cand")
@click.option("--results-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=OUT_DIR / "results")
def score(cand_dir: Path, results_dir: Path) -> None:
    """Сводка судей → ``<results>/summary.csv`` и таблица в терминал."""
    from research.edge_marks.score import summary

    table = summary(cand_dir, results_dir)
    table.to_csv(results_dir / "summary.csv", index=False)
    click.echo(table.to_markdown(index=False))


@main.command()
@click.option("--cand-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=OUT_DIR / "cand")
@click.option("--results-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=OUT_DIR / "results")
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), default=OUT_DIR / "judges")
@click.option("--judge", "names", multiple=True, help="Только эти судьи (по умолчанию — все с результатами).")
def overlays(cand_dir: Path, results_dir: Path, out_dir: Path, names: tuple[str, ...]) -> None:
    """Оверлеи полос по судьям (``<out>/<судья>/overlays/``) и листы «вырезка — вердикты всех судей» (``<out>/sheets/``)."""
    from research.edge_marks.overlays import judge_pages, load_judges, sheets

    candidates, judges = load_judges(cand_dir, results_dir)
    for judge_frame in judges:
        if names and judge_frame.name not in names:
            continue
        count = judge_pages(cand_dir, results_dir, out_dir, judge_frame, candidates)
        click.echo(f"{judge_frame.name}: {count} полос")
    click.echo(f"листов: {sheets(cand_dir, out_dir, candidates, judges)}")
