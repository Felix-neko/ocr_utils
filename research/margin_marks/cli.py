"""CLI стенда «пометки на полях»: ``features`` (признаки кандидатов), ``contact`` (мозаики для разметки глазами)."""

from __future__ import annotations

import csv
import logging
import random
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import click

from ocr_utils.page_layout.pack_analysis.stages import init_worker
from research.margin_marks.sources import PdfVariant, load_candidates, load_deepseek, read_gray

logger = logging.getLogger(__name__)

# Итоги, у которых смотрим признаки: рисунок (объекты с классом «рисунок») и «неясно».
DRAWING = "рисунок"
UNCLEAR = "неясно"

# Состояние воркера: ответы DeepSeek варианта грузятся один раз на воркер.
_DEEPSEEK: dict = {}


def _features_one(args: tuple) -> dict | None:
    """Признаки одного кандидата в воркере."""
    from research.margin_marks.features import as_row, features_of

    candidate, root = args
    if candidate.variant not in _DEEPSEEK:
        _DEEPSEEK[candidate.variant] = load_deepseek(root)
    gray = read_gray(root, candidate.id, "pass1")
    if gray is None:
        return None
    return as_row(features_of(candidate, gray, read_gray(root, candidate.id, "pass2"), _DEEPSEEK[candidate.variant]))


@click.group()
@click.option("--log-level", default="INFO", show_default=True)
def main(log_level: str) -> None:
    """Стенд «пометки на полях, ложно принятые за рисунок»."""
    logging.basicConfig(level=log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@main.command()
@click.option("--layout-root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Корень разбора v6 (подпапки nogeo, geo).")  # fmt: skip
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--variants", default="nogeo,geo", show_default=True)
@click.option("--jobs", default=16, show_default=True, type=int)
def features(layout_root: Path, out_dir: Path, variants: str, jobs: int) -> None:
    """Признаки всех кандидатов, ставших рисунком или «неясно», по вариантам → ``features.csv``."""
    tasks = []
    for name in variants.split(","):
        variant = PdfVariant(name)
        root = layout_root / variant.value
        chosen = [c for c in load_candidates(root, variant) if DRAWING in c.classes or c.outcome == UNCLEAR]
        logger.info("%s: кандидатов-рисунков и «неясно» %d", variant.value, len(chosen))
        tasks += [(candidate, root) for candidate in chosen]
    rows = []
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for row in pool.map(_features_one, tasks, chunksize=8):
            if row is not None:
                rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "features.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Признаки: %d строк → %s", len(rows), out_dir / "features.csv")


def mark_score(row: dict) -> float:
    """Похожесть на пометку для порядка отбора: мало компонент, линейная краска, мало краски, тонко, бледно."""
    score = 0.0
    score += 2.0 if int(row["components"]) <= 3 else 0.0
    score += 2.0 * float(row["linear_share"])
    score += 1.5 if float(row["ink_share"]) < 0.04 else 0.0
    score += 1.0 if float(row["thickness_max_mm"]) <= 0.6 else 0.0
    score += 0.5 if row["edge_touch"] == "True" else 0.0
    score += 0.5 if not row["pass1_nontext"] else 0.0
    return score


@main.command()
@click.option("--layout-root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--features-csv", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--variant", default="nogeo", show_default=True)
@click.option("--top", default=150, show_default=True, type=int, help="Самых похожих на пометку.")
@click.option("--random-rest", default=150, show_default=True, type=int, help="Случайных из остальных.")
@click.option("--per-sheet", default=12, show_default=True, type=int)
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--max-thickness", default=None, type=float,
              help="Только кандидаты с толщиной штриха остатка не больше, мм (и с непустым остатком).")  # fmt: skip
@click.option("--ids", "ids_file", default=None, type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Только кандидаты из файла (id по строке).")  # fmt: skip
def contact(layout_root: Path, features_csv: Path, out_dir: Path, variant: str, top: int, random_rest: int,
            per_sheet: int, seed: int, max_thickness: float | None, ids_file: Path | None) -> None:  # fmt: skip
    """Мозаики для разметки: самые похожие на пометку плюс случайные остальные; порядок и номера — в ``contact.csv``."""
    from research.margin_marks.sheets import contact_sheet

    with features_csv.open(encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["variant"] == variant]
    if ids_file is not None:
        wanted = set(ids_file.read_text(encoding="utf-8").split())
        rows = [r for r in rows if r["id"] in wanted]
    if max_thickness is not None:
        rows = [r for r in rows if int(r["components"]) > 0 and float(r["thickness_max_mm"]) <= max_thickness]
    rows.sort(key=lambda r: -mark_score(r))
    chosen = rows[:top]
    rest = rows[top:]
    random.Random(seed).shuffle(rest)
    chosen += rest[:random_rest]
    root = layout_root / variant
    deepseek = load_deepseek(root)
    inners = {c.id: c.crop.inner for c in load_candidates(root, PdfVariant(variant))}
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "contact.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sheet", "cell", "id", "score"])
        for start in range(0, len(chosen), per_sheet):
            batch = chosen[start : start + per_sheet]
            sheet = start // per_sheet
            picture = contact_sheet(root, batch, f"лист {sheet:02d}", deepseek.pass1_words, inners)
            import cv2

            cv2.imwrite(str(out_dir / f"sheet_{sheet:02d}.jpg"), picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
            for cell, row in enumerate(batch):
                writer.writerow([sheet, cell, row["id"], f"{mark_score(row):.2f}"])
    logger.info("Мозаик: %d, кандидатов %d → %s", (len(chosen) + per_sheet - 1) // per_sheet, len(chosen), out_dir)


def _replay_one(args: tuple) -> list[dict]:
    """Пересчёт полосы в воркере; ошибка — строка с ``error``."""
    from research.margin_marks.replay import replay_page

    root, variant, page_key, pdf_dir, rule = args
    try:
        return replay_page(root, variant, page_key, pdf_dir, rule)
    except Exception as error:  # noqa: BLE001 — одна битая полоса не валит прогон
        return [{"variant": variant.value, "id": page_key, "error": f"{type(error).__name__}: {error}"}]


@main.command()
@click.option("--layout-root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--geo-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--nogeo-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--variants", default="nogeo,geo", show_default=True)
@click.option("--jobs", default=16, show_default=True, type=int)
def replay(layout_root: Path, geo_dir: Path, nogeo_dir: Path, out_dir: Path, variants: str, jobs: int) -> None:
    """Пересчёт решения line art всех кандидатов без GPU: воспроизводимость и смены от правила «пометка» → ``replay.csv``."""
    import json

    from research.margin_marks.rule import MarkRule

    rule = MarkRule()
    tasks = []
    for name in variants.split(","):
        variant = PdfVariant(name)
        root = layout_root / variant.value
        pdf_dir = geo_dir if variant is PdfVariant.GEO else nogeo_dir
        for path in sorted((root / "pages").glob("*.json")):
            if json.loads(path.read_text(encoding="utf-8")).get("candidates"):
                tasks.append((root, variant, path.stem, pdf_dir, rule))
    logger.info("Полос с кандидатами: %d", len(tasks))
    rows, errors = [], 0
    with ProcessPoolExecutor(jobs, mp_context=get_context("forkserver"), initializer=init_worker) as pool:
        for result in pool.map(_replay_one, tasks, chunksize=4):
            for row in result:
                if "error" in row:
                    errors += 1
                    logger.error("%s", row)
                else:
                    rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "replay.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    same = sum(1 for r in rows if r["reproduced"])
    changed = [r for r in rows if r["changed"]]
    logger.info("Кандидатов %d, воспроизведено %d, ошибок полос %d, смен от правила %d", len(rows), same, errors, len(changed))


__all__ = ["main", "mark_score"]
