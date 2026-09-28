"""Команды стенда соринок у края строки: ``scan`` — концы строк по паку (бинаризованные PDF)."""

from __future__ import annotations

import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click

from ocr_utils.page_layout.pack_analysis.stages import init_worker
from research.text_block_specks.pages import NOGEO_DIR, only_text_keys, pdf_pages
from research.text_block_specks.scan import append_records, scan_page, write_header


# Зерно порядка полос в ``scan``.
SEED = 20260928


@click.group()
def main() -> None:
    """Стенд соринок и отчёркиваний у края строки (``reports/text_block_specks.md``)."""


@main.command()
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--pdf-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=NOGEO_DIR)
@click.option("--jobs", default=16, show_default=True, help="воркеров пула (разбор полосы — CPU)")
@click.option("--limit", default=0, show_default=True, help="только первые N полос (0 — все)")
def scan(out_dir: Path, pdf_dir: Path, jobs: int, limit: int) -> None:
    """Концы строк у сторон блоков по всем полосам «только текст» → ``<out-dir>/ends.csv``.

    Идемпотентно: полосы, уже записанные в ``done.txt``, пропускаются; ошибки — в ``errors.txt``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ends, done_path, errors = out_dir / "ends.csv", out_dir / "done.txt", out_dir / "errors.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    pages = [page for page in pdf_pages(only_text_keys()) if page.key not in done]
    # Случайный (но воспроизводимый) порядок: прерванный прогон уже равномерно покрывает годы и выпуски.
    random.Random(SEED).shuffle(pages)
    if limit:
        pages = pages[:limit]
    click.echo(f"полос к разбору: {len(pages)} (уже готово {len(done)})")
    write_header(ends)
    context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(jobs, mp_context=context, initializer=init_worker) as pool:
        futures = [pool.submit(scan_page, (pdf_dir, page)) for page in pages]
        for number, future in enumerate(as_completed(futures), 1):
            page, records, seconds, error = future.result()
            # Записи и отметка о готовности пишет только главный процесс — без гонок за файлы.
            if error:
                with errors.open("a") as handle:
                    handle.write(f"{page.key}\t{error}\n")
            else:
                append_records(ends, records)
            with done_path.open("a") as handle:
                handle.write(page.key + "\n")
            if number % 200 == 0 or number == len(pages):
                click.echo(f"{number}/{len(pages)}  последняя {page.key} {seconds:.1f} с")


@main.command()
@click.option("--scan-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--pdf-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=NOGEO_DIR)
@click.option("--max-resid", default=-0.8, show_default=True, help="выступ наружу больше стольких мм (отрицательное)")
@click.option("--min-resid", default=-6.0, show_default=True, help="…но не больше стольких мм (дальше — вёрстка)")
@click.option("--far-sample", default=48, show_default=True, help="сколько выступов за --min-resid взять выборкой")
def candidates(scan_dir: Path, out_dir: Path, pdf_dir: Path, max_resid: float, min_resid: float, far_sample: int) -> None:
    """Кандидаты для разметки глазами: выступы концов строк наружу → ``candidates.csv`` и контакт-листы ``sheets/``.

    Берутся все концы с выступом в поясе ``[min_resid, max_resid]`` мм и выборка ``far_sample`` более
    дальних. Номер кандидата (``c0001``…) подписан на вырезке; разметка — в ``labels.csv`` рядом.
    """
    import csv

    import cv2

    from research.text_block_specks.sheets import contact_sheets, crops_for

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((scan_dir / "ends.csv").open()))
    near = [r for r in rows if min_resid <= float(r["resid_mm"]) < max_resid]
    far = [r for r in rows if float(r["resid_mm"]) < min_resid]
    random.Random(SEED).shuffle(far)
    chosen = sorted(near, key=lambda r: (r["key"], r["block"], r["side"], int(r["row"]))) + far[:far_sample]
    for number, record in enumerate(chosen, 1):
        record["id"] = f"c{number:04d}"
    with (out_dir / "candidates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", *rows[0].keys()])
        writer.writeheader()
        writer.writerows(chosen)
    crops = crops_for(chosen, pdf_dir)
    sheets_dir = out_dir / "sheets"
    sheets_dir.mkdir(exist_ok=True)
    for number, sheet in enumerate(contact_sheets([crops[r["id"]] for r in chosen])):
        cv2.imwrite(str(sheets_dir / f"sheet_{number:03d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    click.echo(f"кандидатов {len(chosen)} (в поясе {len(near)}, дальних {min(far_sample, len(far))})")


@main.command()
@click.option("--pages-file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--rules", default="none,templates", show_default=True, help="варианты правил через запятую (evaluate.RULES)")
@click.option("--pdf-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), default=NOGEO_DIR)
@click.option("--jobs", default=16, show_default=True)
def evaluate(pages_file: Path, out_dir: Path, rules: str, pdf_dir: Path, jobs: int) -> None:
    """Прогон набора (``eval_pages.json``: полосы с дефектом и без) с вариантами правил → ``<out-dir>/<правило>/*.json``.

    Готовые пары «полоса × правило» пропускаются.
    """
    import json

    from research.text_block_specks.evaluate import run_task
    from research.text_block_specks.pages import PdfPage

    spec = json.loads(pages_file.read_text())
    pages = [PdfPage(**item) for group in ("defect", "clean") for item in spec[group]]
    tasks = [(pdf_dir, page, rule, out_dir) for rule in rules.split(",") for page in pages]
    context = multiprocessing.get_context("forkserver")
    with ProcessPoolExecutor(jobs, mp_context=context, initializer=init_worker) as pool:
        for number, (key, rule, error) in enumerate(pool.map(run_task, tasks), 1):
            if error:
                click.echo(f"ошибка {key} {rule}: {error}")
            if number % 50 == 0 or number == len(tasks):
                click.echo(f"{number}/{len(tasks)}")


@main.command()
@click.option("--pages-file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--labels", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--candidates", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--rules", default="templates", show_default=True)
def score(pages_file: Path, labels: Path, candidates: Path, out_dir: Path, rules: str) -> None:
    """Сводка «до — после» по вариантам правил и ``changes_<правило>.csv`` со всеми сдвинутыми концами."""
    import csv
    import json
    from dataclasses import asdict

    from research.text_block_specks.evaluate import compare, summary

    spec = json.loads(pages_file.read_text())
    defect = {item["key"] for item in spec["defect"]}
    keys = [item["key"] for group in ("defect", "clean") for item in spec[group]]
    marks = {r["id"]: r["label"] for r in csv.DictReader(labels.open())}
    by_end = {
        (r["key"], int(r["block"]), r["side"], int(r["row"])): marks[r["id"]]
        for r in csv.DictReader(candidates.open())
        if r["id"] in marks
    }
    for rule in rules.split(","):
        changes = []
        for key in keys:
            before, after = out_dir / "none" / f"{key}.json", out_dir / rule / f"{key}.json"
            if before.exists() and after.exists():
                changes.extend(compare(json.loads(before.read_text()), json.loads(after.read_text()), by_end))
        click.echo(f"{rule}: {summary(changes, defect)}")
        moved = [c for c in changes if c.label in ("S", "M", "P") or (c.moved_mm == c.moved_mm and abs(c.moved_mm) > 0.3)]
        with (out_dir / f"changes_{rule}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(moved[0]).keys()) if moved else ["key"])
            writer.writeheader()
            writer.writerows(asdict(c) for c in moved)
