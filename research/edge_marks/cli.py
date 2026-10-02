"""Команды стенда краевых пометок: ``select`` — выбросы концов строк на выровненных сторонах и листы для разметки."""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path

import click
import cv2
import pandas as pd

from research.text_block_specks.pages import NOGEO_DIR

# Прошлый стенд соринок: концы строк всего пака и его разметка.
SPECKS_DIR = Path("/mnt/hotstore/scan_processing/mts/text_block_specks")
OUT_DIR = Path("/mnt/hotstore/scan_processing/mts/pack1_edge_marks")
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
    "craft": ["/mnt/hotstore/scan_processing/mts_markup/line_axis_engines/craft/bin/python", "research/edge_marks/workers/craft_judge.py", "--fp16"],
    "doctr": ["/mnt/hotstore/scan_processing/mts_markup/line_axis_engines/doctr/bin/python", "research/edge_marks/workers/doctr_judge.py"],
    "pero": ["/mnt/hotstore/scan_processing/mts/curved_layout_engines/pero/bin/python", "research/edge_marks/workers/pero_judge.py"],
    "surya": ["uv", "run", "python", "-m", "research.edge_marks.judges.surya_judge"],
    "paddle": ["/mnt/hotstore/scan_processing/mts_markup/line_axis_engines/paddle/bin/python", "research/edge_marks/workers/paddle_judge.py"],
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


@main.command("pero-layout")
@click.option("--pages-dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=OUT_DIR / "test_pages_bumps" / "pages")  # fmt: skip
@click.option("--cand-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=OUT_DIR / "cand")
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), default=OUT_DIR / "test_pages_bumps_pero_native")
@click.option("--pero-python", type=click.Path(path_type=Path), default=None, help="python окружения pero")
@click.option("--pero-config", type=click.Path(path_type=Path), default=None, help="config.ini модели pero")
def pero_layout(pages_dir: Path, cand_dir: Path, out_dir: Path, pero_python: Path | None, pero_config: Path | None) -> None:
    """Собственный разбор вёрстки pero (регионы — блоки, строки): JSON и оверлеи по полосам ``--pages-dir``."""
    from ocr_utils.page_layout.text_blocks.engines.pero import PeroEngine
    from research.edge_marks.candidates import load
    from research.edge_marks.pero_layout import run

    count = run(pages_dir, load(cand_dir), out_dir, PeroEngine(python=pero_python, config=pero_config))
    click.echo(f"полос {count} → {out_dir}")


def guard_jobs(pages_set: str) -> list:
    """Полосы набора для защиты сторон: ``bumps`` (55 с выступами от сора), ``clean`` (30 чистых тестового
    множества), ``random200`` (200 случайных текстовых полос пака вне тестового множества, сид ``SEED``)."""
    import json

    from research.edge_marks.guard.guard import PageJob

    if pages_set == "bumps":
        rows = list(csv.DictReader((OUT_DIR / "test_pages_bumps" / "pages.csv").open()))
        return [PageJob(r["полоса"], NOGEO_DIR / r["pdf"], int(r["страница pdf"])) for r in rows]
    spec = json.loads((SETS_DIR / "pages.json").read_text())
    if pages_set == "clean":
        return [PageJob(p["key"], NOGEO_DIR / p["pdf"], int(p["page"])) for p in spec["clean"]]
    taken = {p["key"] for p in spec["defect"] + spec["clean"]}
    ends = pd.read_csv(SPECKS_DIR / "scan" / "ends.csv", usecols=["key", "pdf", "page"]).drop_duplicates("key")
    ends = ends[~ends.key.isin(taken)].sort_values("key")
    sample = ends.sample(200, random_state=SEED).sort_values("key")
    return [PageJob(r.key, NOGEO_DIR / r.pdf, int(r.page)) for r in sample.itertuples()]


# Воркеры карт детекторов второго прохода: интерпретатор окружения и скрипт.
MAP_WORKERS = {
    "craft": ["/mnt/hotstore/scan_processing/mts_markup/line_axis_engines/craft/bin/python", "research/edge_marks/workers/craft_maps.py", "--fp16"],
    "pero": ["/mnt/hotstore/scan_processing/mts/curved_layout_engines/pero/bin/python", "research/edge_marks/workers/pero_maps.py"],
    "doctr": ["/mnt/hotstore/scan_processing/mts_markup/line_axis_engines/doctr/bin/python", "research/edge_marks/workers/doctr_maps.py"],
}


def detector_maps(jobs: list, engines: set, maps_dir: Path, base: Path) -> None:
    """Карты детекторов по полосам: по одному GPU-процессу на детектор (строго по очереди), кэш по ключу, замер ресурсов.

    Args:
        jobs: Полосы (``PageJob``).
        engines: Детекторы (``glyph_vote.Engine``).
        maps_dir: Корень карт ``<maps_dir>/<детектор>/<ключ>.png``.
        base: Папка набора (вход воркеров, журналы, замеры).
    """
    import json

    from ocr_utils.page_layout.text_blocks.page import render_page
    from research.edge_marks.vram import run_measured

    pngs_dir = base / "maps_input"
    pngs_dir.mkdir(parents=True, exist_ok=True)
    for engine in sorted(engines, key=lambda e: e.value):
        todo = [job for job in jobs if not (maps_dir / engine.value / f"{job.key}.png").exists()]
        if not todo:
            continue
        for job in todo:
            target = pngs_dir / f"{job.key}.png"
            if not target.exists():
                cv2.imwrite(str(target), render_page(job.pdf, job.page))
        spec = base / f"maps_{engine.value}.json"
        spec.write_text(json.dumps([{"key": j.key, "png": str(pngs_dir / f"{j.key}.png")} for j in todo]))
        worker = MAP_WORKERS[engine.value]
        command = [worker[0], worker[1], "--jobs", str(spec), "--out-dir", str(maps_dir / engine.value), *worker[2:]]
        with (base / f"maps_{engine.value}.log").open("w") as log:
            usage = run_measured(command, cwd=str(PROJECT), log=log)
        (base / f"maps_{engine.value}_usage.json").write_text(json.dumps(usage.__dict__))
        click.echo(f"карты {engine.value}: {len(todo)} полос, код {usage.returncode}, {usage.seconds:.0f} с, "
                   f"VRAM {usage.vram_mb:.0f} МБ")  # fmt: skip


@main.command()
@click.option("--pages-set", type=click.Choice(["bumps", "clean", "random200"]), default="bumps", show_default=True)
@click.option("--second-pass", "passes", multiple=True, type=click.Choice(["craft", "craft_pero", "craft_pero_doctr"]),
              default=("craft", "craft_pero", "craft_pero_doctr"), show_default=True,
              help="варианты второго прохода на аномальных полосах (фильтр голосованием детекторов и повторный разбор); "
                   "выступы, оставшиеся по итогу, помечаются всегда")  # fmt: skip
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), default=OUT_DIR / "guard")
@click.option("--layout-cache", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=Path("/mnt/hotstore/scan_processing/mts/pack1_page_layout"))  # fmt: skip
@click.option("--bump-mm", type=float, default=None, help="порог выступа, мм (по умолчанию anomaly.BUMP_MM)")
@click.option("--jobs", type=int, default=16, show_default=True, help="процессов разбора (CPU)")
def guard(pages_set: str, passes: tuple[str, ...], out_dir: Path, layout_cache: Path, bump_mm: float | None,
          jobs: int) -> None:  # fmt: skip
    """Защита выровненных сторон блоков: разбор без второго прохода по всем полосам (выступы помечаются
    недостоверными), затем выбранные варианты второго прохода по аномальным полосам; склейки «без второго прохода |
    варианты» в ``compare/`` и ``summary_<вариант>.json``."""
    from research.edge_marks.guard.anomaly import BUMP_MM
    from research.edge_marks.guard.guard import GuardPass, GuardSettings, compare, run_pass

    base = out_dir / pages_set
    maps_dir = out_dir / "maps"
    settings = GuardSettings(layout_cache, BUMP_MM if bump_mm is None else bump_mm, maps_dir)
    all_jobs = guard_jobs(pages_set)
    started = time.monotonic()
    plain = run_pass(GuardPass.PLAIN, all_jobs, jobs, settings, base)
    anomalous = [job for job, item in zip(all_jobs, plain) if item["anomalous"]]
    click.echo(f"без второго прохода: полос {len(all_jobs)}, аномальных {len(anomalous)}, "
               f"{time.monotonic() - started:.0f} с")  # fmt: skip
    variants = [GuardPass(name) for name in passes]
    if anomalous and variants:
        detector_maps(anomalous, {e for v in variants for e in v.engines}, maps_dir, base)
    for variant in variants:
        tick = time.monotonic()
        run_pass(variant, anomalous, jobs, settings, base)
        click.echo(f"{variant.title}: полос {len(anomalous)}, {time.monotonic() - tick:.0f} с")
    shown = [GuardPass.PLAIN, *variants]
    for job in anomalous:
        compare(job.key, [base / v.value for v in shown], [v.title for v in shown], base / "compare" / f"{job.key}.jpg")
