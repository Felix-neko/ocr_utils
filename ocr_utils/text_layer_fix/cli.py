"""Команды пакета: ``uv run python -m ocr_utils.text_layer_fix <команда>``.

``survey`` — обзор текстового слоя по всем PDF пака (только чтение): по странице число
слов, формы спанов, повёрнутые матрицы, растяжения, привязка. ``run`` — зоны, вердикты,
OCR по выборке страниц с кэшем JSON. ``fix`` — исправленные копии PDF из кэша. ``overlay``
и ``report`` — только читают кэш и CSV. ``eval-lineart`` — точность и полнота детекторов
line art против ручной разметки.
"""

from __future__ import annotations

import csv
import logging
import multiprocessing
import os
import statistics
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import click

from ocr_utils.text_layer_fix import VERSION
from ocr_utils.text_layer_fix.cache import cache_path as core_cache_path
from ocr_utils.text_layer_fix.cache import load_page, save_page

logger = logging.getLogger("ocr_utils.text_layer_fix")

# Страниц одного PDF на задачу пула: PDF открывается в воркере один раз на пачку.
CHUNK_PAGES = 12


def _init_worker() -> None:
    """Инициализатор воркера: без hugepage и потоков BLAS (``.claude/rules/gpu_and_pools.md``)."""
    import cv2
    import numpy as np
    import threadpoolctl

    np._core.multiarray._set_madvise_hugepage(False)
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)


def physical_cpu_count() -> int:
    """Число физических ядер машины (без SMT-потоков); без psutil — логических."""
    try:
        import psutil

        return psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except ImportError:
        return os.cpu_count() or 1


def effective_jobs(jobs: int, reserve_cpu_cores: int) -> int:
    """Сколько воркеров запускать, оставив машине ``reserve_cpu_cores`` физических ядер.

    Args:
        jobs: Запрошенное число воркеров.
        reserve_cpu_cores: Сколько физических ядер не трогать; 0 — как просили.

    Returns:
        ``min(jobs, ядер − резерв)``, но не меньше одного.
    """
    if reserve_cpu_cores <= 0:
        return max(1, jobs)
    return max(1, min(jobs, physical_cpu_count() - reserve_cpu_cores))


def parse_pages(text: "str | None", total: int) -> list[int]:
    """Список номеров страниц с нуля из строки вида ``1,5-9`` (нумерация с единицы).

    Args:
        text: Строка опции ``--pages``; пусто — все страницы.
        total: Число страниц в документе.

    Returns:
        Номера страниц с нуля в пределах документа.
    """
    if not text:
        return list(range(total))
    pages: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            low, high = part.split("-", 1)
            pages.extend(range(int(low) - 1, int(high)))
        elif part:
            pages.append(int(part) - 1)
    return [p for p in pages if 0 <= p < total]


def collect_pdfs(pdf_dir: Path, only: tuple[str, ...], limit: "int | None") -> list[Path]:
    """PDF папки по имени, с фильтром по подстроке и ограничением числа."""
    pdfs = sorted(p for p in pdf_dir.glob("*.pdf") if p.is_file())
    if only:
        pdfs = [p for p in pdfs if any(sub in p.name for sub in only)]
    if limit:
        pdfs = pdfs[:limit]
    return pdfs


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO), format="%(levelname)s %(name)s: %(message)s"
    )


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--log-level", default="INFO", show_default=True)
def main(log_level: str) -> None:
    """Чистка и дополнение текстового слоя FineReader для повёрнутого текста."""
    _setup_logging(log_level)


# --- survey ----------------------------------------------------------------------------------


@dataclass
class SurveyRow:
    """Статистика текстового слоя одной страницы."""

    pdf: str
    page: int
    width_pt: float = 0.0
    height_pt: float = 0.0
    image_w: int = 0
    image_h: int = 0
    streams: int = 0
    figures: int = 0  # дополнительные образы — куски, которые FineReader счёл картинками
    spans: int = 0
    words: int = 0  # спаны с непробельным текстом
    plain: int = 0
    td_only: int = 0
    rotated: int = 0  # спаны с повёрнутой матрицей
    rotated_words: int = 0  # из них с непробельным текстом
    empty: int = 0
    no_span: int = 0  # текстовые объекты вне спанов
    chars: int = 0
    short_words: int = 0  # слова в 1–2 знака без цифр — кандидаты в россыпь
    unmatched: int = 0
    orphans: int = 0
    offpage: int = 0
    struct_lines: int = 0
    stretch_median: float = 0.0
    stretch_p90: float = 0.0
    stretch_outliers: int = 0  # растяжение вне [0.7, 1.5]
    render_mode3: int = 0
    seconds: float = 0.0
    error: str = ""


SURVEY_FIELDS = list(SurveyRow("", 0).__dict__.keys())


def _survey_chunk(args: tuple) -> list[dict]:
    """Обзор пачки страниц одного PDF (в пуле)."""
    import time

    import fitz
    import pikepdf

    from ocr_utils.text_layer_fix.raster import page_raster
    from ocr_utils.text_layer_fix.text_layer import (
        SpanShape,
        attach_chars,
        count_offpage,
        page_content,
        page_fonts,
        page_matrix,
        parse_content,
        struct_lines,
    )

    path, pages = args
    rows: list[dict] = []
    with fitz.open(path) as doc, pikepdf.open(path) as pdf:
        for index in pages:
            row = SurveyRow(Path(path).name, index)
            started = time.time()
            try:
                page = doc[index]
                row.width_pt, row.height_pt = round(page.rect.width, 2), round(page.rect.height, 2)
                raster = page_raster(page)
                row.image_w, row.image_h = raster.width, raster.height
                row.figures = len(raster.figures)
                row.streams = len(page.get_contents())
                words, no_span = parse_content(page_content(page), page_matrix(page), page_fonts(pdf.pages[index]))
                row.unmatched, row.orphans = attach_chars(words, page.get_text("rawdict", flags=0))
                row.offpage = count_offpage(words, page.rect)
                row.unmatched -= row.offpage
                row.no_span = no_span
                lines = struct_lines(pdf, index)
                row.struct_lines = len(set(lines.values()))
                stretches: list[float] = []
                for word in words:
                    if word.mcid is not None:
                        row.spans += 1
                    text = word.text.strip()
                    if word.shape == SpanShape.PLAIN:
                        row.plain += 1
                    elif word.shape == SpanShape.TD_ONLY:
                        row.td_only += 1
                    elif word.shape == SpanShape.ROTATED:
                        row.rotated += 1
                        if text:
                            row.rotated_words += 1
                    else:
                        row.empty += 1
                    if text:
                        row.words += 1
                        row.chars += len(text)
                        if len(text) <= 2 and not any(ch.isdigit() for ch in text):
                            row.short_words += 1
                        if word.shape == SpanShape.PLAIN:
                            stretches.append(word.stretch)
                    if word.render_mode == 3 and word.glyphs:
                        row.render_mode3 += 1
                if stretches:
                    stretches.sort()
                    row.stretch_median = round(statistics.median(stretches), 3)
                    row.stretch_p90 = round(stretches[int(0.9 * (len(stretches) - 1))], 3)
                    row.stretch_outliers = sum(1 for s in stretches if s < 0.7 or s > 1.5)
            except Exception as error:  # noqa: BLE001 — страница не должна валить прогон
                row.error = f"{type(error).__name__}: {error}"
            row.seconds = round(time.time() - started, 3)
            rows.append(asdict(row))
    return rows


@main.command()
@click.option("--pdf-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--jobs", default=16, show_default=True, type=int, help="воркеров; разбор потока упирается в CPU")
@click.option(
    "--reserve-cpu-cores", default=0, show_default=True, type=int, help="столько физических ядер оставить машине"
)
@click.option("--only", multiple=True, help="подстрока имени PDF (можно несколько)")
@click.option("--pages", help="страницы: 1,5-9 (с единицы)")
@click.option("--limit", type=int, help="не больше стольких PDF")
def survey(pdf_dir, out_dir, jobs, reserve_cpu_cores, only, pages, limit) -> None:
    """Обзор текстового слоя всех страниц → survey.csv (только чтение PDF)."""
    import fitz
    from tqdm import tqdm

    jobs = effective_jobs(jobs, reserve_cpu_cores)
    pdfs = collect_pdfs(pdf_dir, only, limit)
    tasks = []
    for path in pdfs:
        with fitz.open(path) as doc:
            wanted = parse_pages(pages, len(doc))
        for start in range(0, len(wanted), CHUNK_PAGES):
            tasks.append((str(path), wanted[start : start + CHUNK_PAGES]))
    total = sum(len(t[1]) for t in tasks)
    click.echo(f"PDF: {len(pdfs)}, страниц: {total}, задач: {len(tasks)}, воркеров: {jobs}, версия {VERSION}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if jobs <= 1:
        outcomes = map(_survey_chunk, tasks)
    else:
        pool = ProcessPoolExecutor(
            max_workers=jobs, mp_context=multiprocessing.get_context("forkserver"), initializer=_init_worker
        )
        outcomes = pool.map(_survey_chunk, tasks)
    rows: list[dict] = []
    with tqdm(total=total, desc="страницы", unit="стр") as bar:
        for chunk in outcomes:
            rows.extend(chunk)
            bar.update(len(chunk))
    csv_path = out_dir / "survey.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SURVEY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    errors = [r for r in rows if r["error"]]
    rotated = sum(1 for r in rows if r["rotated_words"])
    unmatched = sum(r["unmatched"] for r in rows)
    click.echo(
        f"Готово: {len(rows)} страниц, с повёрнутыми спанами FineReader: {rotated}, "
        f"непривязанных глифов всего: {unmatched}, ошибок: {len(errors)}. CSV: {csv_path}"
    )
    for row in errors[:10]:
        click.echo(f"  {row['pdf']} с.{row['page'] + 1}: {row['error']}")


# --- run ------------------------------------------------------------------------------------


def cache_path(out_dir: Path, pdf: str, page: int) -> Path:
    """JSON страницы в каталоге прогона (``<out_dir>/cache/<pdf>/pNNNN.json``)."""
    return core_cache_path(out_dir / "cache", pdf, page)


def _run_chunk(args: tuple) -> list[dict]:
    """Разбор пачки страниц одного PDF (в пуле): JSON в кэш, краткая запись на страницу."""

    import fitz
    import pikepdf

    from ocr_utils.text_layer_fix.pipeline import Options, process_page

    from ocr_utils.external_ocr_services.hyphen_join import default_morph

    path, pages, out_dir, options_dict, skip_done = args
    out_dir = Path(out_dir)
    options = Options(**options_dict, known=default_morph().known)
    rows: list[dict] = []
    with fitz.open(path) as doc, pikepdf.open(path) as pdf:
        for index in pages:
            target = cache_path(out_dir, Path(path).name, index)
            cached = load_page(target) if skip_done else None
            if cached is not None:
                rows.append(_summary_row(cached))
                continue
            payload = process_page(doc, pdf, index, options).to_json()
            save_page(target, payload)
            rows.append(_summary_row(payload))
    return rows


def _summary_row(payload: dict) -> dict:
    """Строка pages.csv по JSON страницы."""
    counts: dict[str, int] = {}
    for word in payload.get("words", []):
        counts[word["verdict"]] = counts.get(word["verdict"], 0) + 1
    zones = payload.get("zones", [])
    kinds: dict[str, int] = {}
    for zone in zones:
        kinds[zone["kind"]] = kinds.get(zone["kind"], 0) + 1
    readings = payload.get("readings", {})
    return {
        "pdf": payload["pdf"],
        "page": payload["page"],
        "tables": sum(1 for t in payload.get("tables", []) if t["kind"] == "таблица"),
        "art": sum(1 for t in payload.get("tables", []) if t["kind"] != "таблица"),
        "figures": len(payload.get("figures", [])),
        "zones": len(zones),
        "zone_table_cell": kinds.get("table_cell", 0),
        "zone_mixed": kinds.get("table_cell_mixed", 0),
        "zone_upright_missing": kinds.get("table_cell_upright", 0),
        "zone_line_art": kinds.get("line_art_label", 0),
        "zone_standalone": kinds.get("standalone", 0),
        "zone_line_art_upright": kinds.get("line_art_upright", 0),
        "zone_standalone_upright": kinds.get("standalone_upright", 0),
        "zones_no_side": sum(1 for z in zones if z.get("rotate_cw") is None and not z["kind"].endswith("upright")),
        "read_accepted": sum(1 for r in readings.values() if r.get("accepted")),
        "read_rejected": sum(1 for r in readings.values() if not r.get("accepted")),
        "layer_words": payload.get("layer_words", 0),
        "delete": counts.get("delete", 0),
        "sanitize": counts.get("sanitize", 0),
        "suspect": counts.get("suspect", 0),
        "keep_rotated": counts.get("keep_rotated", 0),
        "unmatched": payload.get("unmatched", 0),
        "offpage": payload.get("offpage", 0),
        "seconds": payload.get("seconds", 0),
        "error": payload.get("error", ""),
    }


PAGE_FIELDS = list(_summary_row({"pdf": "", "page": 0}).keys())


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_sample(out_dir: Path) -> list[dict]:
    with open(out_dir / "sample.csv", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@main.command()
@click.option("--pdf-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--probe-db",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="база-зонд с full_pdf_page_idx",
)
@click.option(
    "--markup-db",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="база разметки (только чтение)",
)
@click.option(
    "--rotated-summary",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="summary.csv прогона rotated_text",
)
@click.option(
    "--line-art-csv", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="CSV line_art_detection"
)
@click.option(
    "--min-coverage",
    default=0.05,
    show_default=True,
    type=float,
    help="порог покрытия line art для категории high_coverage",
)
@click.option("--controls", default=200, show_default=True, type=int, help="случайных контрольных страниц")
@click.option("--seed", default=7, show_default=True, type=int)
@click.option("--only", multiple=True, help="подстрока имени PDF (можно несколько)")
@click.option("--pages", help="страницы явно: 1,5-9 (с единицы) — тогда выборка не строится, нужен --only")
@click.option("--limit-pages", type=int, help="не больше стольких страниц выборки (для пробы)")
@click.option("--jobs", default=16, show_default=True, type=int)
@click.option("--reserve-cpu-cores", default=0, show_default=True, type=int)
@click.option("--allowed", default="0,90,180,270", show_default=True, help="допустимые повороты текста ячеек")
@click.option("--lang", default="rus", show_default=True)
@click.option("--no-free-text", is_flag=True, help="не искать боковой текст вне таблиц")
@click.option("--skip-done/--redo", default=True, show_default=True)
def run(
    pdf_dir,
    out_dir,
    probe_db,
    markup_db,
    rotated_summary,
    line_art_csv,
    min_coverage,
    controls,
    seed,
    only,
    pages,
    limit_pages,
    jobs,
    reserve_cpu_cores,
    allowed,
    lang,
    no_free_text,
    skip_done,
) -> None:
    """Зоны, вердикты и чтение по выборке страниц → cache/ и pages.csv, words.csv, zones.csv."""
    import fitz
    from tqdm import tqdm

    from ocr_utils.text_layer_fix.db_models import KIND_LINE_ART_SCHEMA
    from ocr_utils.text_layer_fix.pages import (
        PageRef,
        Sample,
        SampleKind,
        page_index_map,
        page_rotations,
        pages_with_coverage,
        sample_pages,
        sheets_with_regions,
        sheets_with_rotated_tables,
    )

    jobs = effective_jobs(jobs, reserve_cpu_cores)
    out_dir.mkdir(parents=True, exist_ok=True)
    if pages:
        pdfs = collect_pdfs(pdf_dir, only, None)
        samples: list[Sample] = []
        for path in pdfs:
            with fitz.open(path) as doc:
                total = len(doc)
            samples += [Sample(PageRef(path.name, i), SampleKind.CONTROL) for i in parse_pages(pages, total)]
    else:
        index = page_index_map(probe_db)
        rotations = page_rotations(markup_db)
        rotated = sheets_with_rotated_tables(rotated_summary) if rotated_summary else set()
        art = sheets_with_regions(markup_db, (KIND_LINE_ART_SCHEMA,))
        coverage = pages_with_coverage(line_art_csv, min_coverage) if line_art_csv else set()
        samples = sample_pages(index, rotations, rotated, art, coverage, controls, seed)
        if only:
            samples = [s for s in samples if any(sub in s.ref.pdf for sub in only)]
    if limit_pages:
        samples = samples[:limit_pages]
    _write_csv(
        out_dir / "sample.csv",
        [
            {
                "pdf": s.ref.pdf,
                "page": s.ref.index,
                "kind": str(s.kind),
                "year": s.ref.year,
                "issue": s.ref.issue,
                "sheet": s.ref.sheet,
                "rotate_cw": s.ref.rotate_cw,
            }
            for s in samples
        ],
        ["pdf", "page", "kind", "year", "issue", "sheet", "rotate_cw"],
    )
    by_pdf: dict[str, list[int]] = {}
    for sample in samples:
        by_pdf.setdefault(sample.ref.pdf, []).append(sample.ref.index)
    options = {
        "allowed": tuple(int(a) for a in allowed.split(",")),
        "lang": lang,
        "read_zones": True,
        "free_text": not no_free_text,
    }
    tasks = []
    for pdf_name, indices in sorted(by_pdf.items()):
        path = pdf_dir / pdf_name
        if not path.is_file():
            logger.warning("нет файла %s", path)
            continue
        indices = sorted(set(indices))
        for start in range(0, len(indices), CHUNK_PAGES):
            tasks.append((str(path), indices[start : start + CHUNK_PAGES], str(out_dir), options, skip_done))
    total = sum(len(t[1]) for t in tasks)
    kinds = {}
    for s in samples:
        kinds[str(s.kind)] = kinds.get(str(s.kind), 0) + 1
    click.echo(f"страниц: {total} {kinds}, задач: {len(tasks)}, воркеров: {jobs}, версия {VERSION}")
    if jobs <= 1:
        outcomes = map(_run_chunk, tasks)
    else:
        pool = ProcessPoolExecutor(
            max_workers=jobs, mp_context=multiprocessing.get_context("forkserver"), initializer=_init_worker
        )
        outcomes = pool.map(_run_chunk, tasks)
    rows: list[dict] = []
    with tqdm(total=total, desc="страницы", unit="стр") as bar:
        for chunk in outcomes:
            rows.extend(chunk)
            bar.update(len(chunk))
    _write_csv(out_dir / "pages.csv", rows, PAGE_FIELDS)
    _collect_details(out_dir)
    errors = [r for r in rows if r["error"]]
    click.echo(
        f"Готово: {len(rows)} страниц; зон {sum(r['zones'] for r in rows)}, delete {sum(r['delete'] for r in rows)}, "
        f"sanitize {sum(r['sanitize'] for r in rows)}, suspect {sum(r['suspect'] for r in rows)}, ошибок {len(errors)}. {out_dir / 'pages.csv'}"
    )
    for row in errors[:10]:
        click.echo(f"  {row['pdf']} с.{row['page'] + 1}: {row['error']}")


WORD_FIELDS = [
    "pdf",
    "page",
    "kind",
    "mcid",
    "verdict",
    "text",
    "x0",
    "y0",
    "x1",
    "y1",
    "zone_index",
    "zone_kind",
    "overlap",
    "dist_mm",
    "keep_glyphs",
    "reason",
    "struct_line",
    "shape",
    "stretch",
]
ZONE_FIELDS = [
    "pdf",
    "page",
    "kind",
    "zone_index",
    "zone_kind",
    "x0",
    "y0",
    "x1",
    "y1",
    "rotate_cw",
    "confidence",
    "table_index",
    "cell_key",
    "letters_90",
    "letters_270",
    "note",
    "read_text",
    "read_conf",
    "read_rotate",
    "read_accepted",
    "read_reason",
    "words_delete",
    "words_sanitize",
    "words_keep_rotated",
]


def _collect_details(out_dir: Path) -> None:
    """words.csv (все слова с вердиктом не KEEP) и zones.csv по кэшу выборки."""
    import json

    sample = {(r["pdf"], int(r["page"])): r["kind"] for r in _read_sample(out_dir)}
    words: list[dict] = []
    zones: list[dict] = []
    for (pdf_name, page), kind in sorted(sample.items()):
        path = cache_path(out_dir, pdf_name, page)
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("error"):
            continue
        per_zone: dict[int, dict[str, int]] = {}
        for word in payload["words"]:
            if word["verdict"] == "keep":
                continue
            zone = payload["zones"][word["zone_index"]] if word.get("zone_index") is not None else None
            if zone is not None:
                counter = per_zone.setdefault(word["zone_index"], {})
                counter[word["verdict"]] = counter.get(word["verdict"], 0) + 1
            x0, y0, x1, y1 = word["bbox_px"]
            words.append(
                {
                    "pdf": pdf_name,
                    "page": page,
                    "kind": kind,
                    "mcid": word["mcid"],
                    "verdict": word["verdict"],
                    "text": word["text"],
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "zone_index": word.get("zone_index"),
                    "zone_kind": zone["kind"] if zone else "",
                    "overlap": word["overlap"],
                    "dist_mm": word["dist_mm"],
                    "keep_glyphs": " ".join(map(str, word["keep_glyphs"])),
                    "reason": word["reason"],
                    "struct_line": word.get("struct_line"),
                    "shape": word.get("shape"),
                    "stretch": word.get("stretch"),
                }
            )
        for index, zone in enumerate(payload["zones"]):
            reading = payload["readings"].get(str(index), {})
            counter = per_zone.get(index, {})
            x0, y0, x1, y1 = zone["box"]
            zones.append(
                {
                    "pdf": pdf_name,
                    "page": page,
                    "kind": kind,
                    "zone_index": index,
                    "zone_kind": zone["kind"],
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "rotate_cw": zone.get("rotate_cw"),
                    "confidence": zone.get("confidence"),
                    "table_index": zone.get("table_index"),
                    "cell_key": " ".join(map(str, zone["cell_key"])) if zone.get("cell_key") else "",
                    "letters_90": zone.get("letters", {}).get("90"),
                    "letters_270": zone.get("letters", {}).get("270"),
                    "note": zone.get("note", ""),
                    "read_text": reading.get("text", ""),
                    "read_conf": reading.get("confidence"),
                    "read_rotate": reading.get("rotate_cw"),
                    "read_accepted": reading.get("accepted"),
                    "read_reason": reading.get("reason", ""),
                    "words_delete": counter.get("delete", 0),
                    "words_sanitize": counter.get("sanitize", 0),
                    "words_keep_rotated": counter.get("keep_rotated", 0),
                }
            )
    _write_csv(out_dir / "words.csv", words, WORD_FIELDS)
    _write_csv(out_dir / "zones.csv", zones, ZONE_FIELDS)


# --- fix ------------------------------------------------------------------------------------


def _fix_one(args: tuple) -> list[dict]:
    """Исправленная копия одного PDF (в пуле)."""
    from ocr_utils.text_layer_fix.fixer import fix_pdf, load_cache

    src, out_dir, out_pdf = Path(args[0]), Path(args[1]), Path(args[2])
    cache = load_cache(out_dir / "cache", src.name)
    if not cache:
        return []
    try:
        fixes = fix_pdf(src, cache, out_pdf)
    except Exception as error:  # noqa: BLE001
        return [{"pdf": src.name, "page": -1, "error": f"{type(error).__name__}: {error}"}]
    return [
        {
            "pdf": src.name,
            "page": f.page,
            "blanked": f.blanked,
            "trimmed": f.trimmed,
            "inserted": f.inserted,
            "skipped_duplicates": f.skipped_duplicates,
            "kept_missing": f.verify.kept_missing,
            "deleted_remaining": f.verify.deleted_remaining,
            "inserts_missing": f.verify.inserts_missing,
            "image_changed": f.verify.image_changed,
            "ok": f.verify.ok and not f.error,
            "notes": "; ".join(f.verify.notes),
            "error": f.error,
        }
        for f in fixes
    ]


FIX_FIELDS = [
    "pdf",
    "page",
    "blanked",
    "trimmed",
    "inserted",
    "skipped_duplicates",
    "kept_missing",
    "deleted_remaining",
    "inserts_missing",
    "image_changed",
    "ok",
    "notes",
    "error",
]


@main.command()
@click.option("--pdf-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="каталог прогона run (с cache/)",
)
@click.option("--only", multiple=True)
@click.option("--jobs", default=8, show_default=True, type=int, help="по PDF на воркер; копия выпуска пишется целиком")
def fix(pdf_dir, out_dir, only, jobs) -> None:
    """Исправленные копии PDF по кэшу → <out-dir>/pdf/ и fix.csv (исходники не трогаются)."""
    from tqdm import tqdm

    cache_root = out_dir / "cache"
    names = sorted(p.name for p in cache_root.iterdir() if p.is_dir())
    if only:
        names = [n for n in names if any(sub in n for sub in only)]
    tasks = [
        (str(pdf_dir / name), str(out_dir), str(out_dir / "pdf" / name)) for name in names if (pdf_dir / name).is_file()
    ]
    click.echo(f"PDF: {len(tasks)}, воркеров: {jobs}")
    if jobs <= 1:
        outcomes = map(_fix_one, tasks)
    else:
        pool = ProcessPoolExecutor(
            max_workers=jobs, mp_context=multiprocessing.get_context("forkserver"), initializer=_init_worker
        )
        outcomes = pool.map(_fix_one, tasks)
    rows: list[dict] = []
    for chunk in tqdm(outcomes, total=len(tasks), desc="PDF"):
        rows.extend(chunk)
    _write_csv(out_dir / "fix.csv", rows, FIX_FIELDS)
    bad = [r for r in rows if not r.get("ok")]
    click.echo(
        f"Готово: страниц {len(rows)}, вставок {sum(r.get('inserted', 0) for r in rows)}, удалено слов {sum(r.get('blanked', 0) for r in rows)}, не прошли сверку {len(bad)}. {out_dir / 'fix.csv'}"
    )
    for row in bad[:10]:
        click.echo(f"  {row['pdf']} с.{row['page'] + 1}: {row.get('error') or row.get('notes') or row}")


# --- overlay -------------------------------------------------------------------------------


def _overlay_chunk(args: tuple) -> int:
    """Оверлеи пачки страниц одного PDF (в пуле)."""
    import json

    import fitz

    from ocr_utils.text_layer_fix.overlay import draw_overlay
    from ocr_utils.text_layer_fix.raster import page_raster, render_gray

    path, pages, out_dir, kinds = Path(args[0]), args[1], Path(args[2]), args[3]
    done = 0
    with fitz.open(str(path)) as doc:
        for index in pages:
            cache = cache_path(out_dir, path.name, index)
            if not cache.is_file():
                continue
            payload = json.loads(cache.read_text(encoding="utf-8"))
            if payload.get("error"):
                continue
            page = doc[index]
            raster = page_raster(page)
            gray = render_gray(page, raster)
            kind = kinds.get(index, "page")
            target = out_dir / "overlays" / kind / f"{path.stem}_p{index + 1:04d}.png"
            draw_overlay(gray, payload, target, raster.dpi, f"{path.stem} с.{index + 1} [{kind}]")
            done += 1
    return done


@main.command()
@click.option("--pdf-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--only", multiple=True)
@click.option("--jobs", default=8, show_default=True, type=int)
def overlay(pdf_dir, out_dir, only, jobs) -> None:
    """Оверлеи страниц выборки по кэшу → <out-dir>/overlays/<категория>/."""
    from tqdm import tqdm

    sample = _read_sample(out_dir)
    by_pdf: dict[str, dict[int, str]] = {}
    for row in sample:
        if only and not any(sub in row["pdf"] for sub in only):
            continue
        by_pdf.setdefault(row["pdf"], {})[int(row["page"])] = row["kind"]
    tasks = [
        (str(pdf_dir / name), sorted(kinds), str(out_dir), kinds)
        for name, kinds in sorted(by_pdf.items())
        if (pdf_dir / name).is_file()
    ]
    if jobs <= 1:
        outcomes = map(_overlay_chunk, tasks)
    else:
        pool = ProcessPoolExecutor(
            max_workers=jobs, mp_context=multiprocessing.get_context("forkserver"), initializer=_init_worker
        )
        outcomes = pool.map(_overlay_chunk, tasks)
    total = sum(tqdm(outcomes, total=len(tasks), desc="PDF"))
    click.echo(f"Оверлеев: {total} → {out_dir / 'overlays'}")


# --- second-opinion (surya) ------------------------------------------------------------------


@main.command("second-opinion")
@click.option("--pdf-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--only", multiple=True)
@click.option("--batch", default=16, show_default=True, type=int, help="партия surya")
def second_opinion(pdf_dir, out_dir, only, batch) -> None:
    """Второе мнение surya (GPU, в этом процессе) по зонам с ненадёжным чтением; JSON кэша обновляется."""
    from ocr_utils.rotated_text.tables import second_opinion as surya_module

    from ocr_utils.text_layer_fix.second_opinion import revise

    if not surya_module.surya_available():
        raise click.ClickException("surya недоступна")
    pages = [
        (pdf_dir / r["pdf"], cache_path(out_dir, r["pdf"], int(r["page"])))
        for r in _read_sample(out_dir)
        if not only or any(sub in r["pdf"] for sub in only)
    ]
    stats = revise(pages, batch)
    _collect_details(out_dir)
    click.echo(
        f"surya: страниц {stats.pages}, зон спрошено {stats.asked}, ответов принято {stats.accepted}, стали пригодными {stats.newly_accepted}"
    )


# --- eval-lineart ------------------------------------------------------------------------------


def _eval_chunk(args: tuple) -> tuple[list[dict], dict[str, dict]]:
    """Оценка пачки страниц одного no-geo PDF (в пуле)."""
    import fitz

    from ocr_utils.text_layer_fix.lineart_eval import SOURCES, SourceScore, evaluate_page, score_page

    path, pages, truths, layout_dir, rel_paths = args
    scores = {name: SourceScore() for name in SOURCES}
    rows: list[dict] = []
    with fitz.open(path) as doc:
        for index in pages:
            truth = truths.get(index)
            try:
                predictions, boxes = evaluate_page(
                    doc, index, truth, Path(layout_dir) if layout_dir else None, rel_paths.get(index)
                )
            except Exception as error:  # noqa: BLE001
                rows.append(
                    {
                        "pdf": Path(path).name,
                        "page": index,
                        "kind": "error",
                        "source": "",
                        "box": "",
                        "iou": "",
                        "covered": "",
                        "error": str(error),
                    }
                )
                continue
            for row in score_page(boxes, predictions, scores):
                rows.append({"pdf": Path(path).name, "page": index, **row, "error": ""})
    return rows, {name: score.to_json() for name, score in scores.items()}


EVAL_FIELDS = ["pdf", "page", "kind", "source", "box", "iou", "covered", "error"]


@main.command("eval-lineart")
@click.option(
    "--nogeo-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="PDF без коррекции геометрии",
)
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="каталог прогона run (sample.csv)",
)
@click.option("--probe-db", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--markup-db", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--layout-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), help="кэш разметки surya по сканам"
)
@click.option(
    "--controls", default=200, show_default=True, type=int, help="контрольных страниц без эталона (ложные срабатывания)"
)
@click.option("--jobs", default=16, show_default=True, type=int)
@click.option("--reserve-cpu-cores", default=0, show_default=True, type=int)
def eval_lineart(nogeo_dir, out_dir, probe_db, markup_db, layout_dir, controls, jobs, reserve_cpu_cores) -> None:
    """Точность и полнота источников line art против ручной разметки → lineart_eval.csv, lineart_eval.json."""
    import json
    import sqlite3

    from tqdm import tqdm

    from ocr_utils.text_layer_fix.lineart_eval import SOURCES, truth_pages
    from ocr_utils.text_layer_fix.pages import page_index_map

    jobs = effective_jobs(jobs, reserve_cpu_cores)
    index = page_index_map(probe_db)
    truths = truth_pages(markup_db, index)
    connection = sqlite3.connect(f"file:{markup_db}?mode=ro", uri=True)
    rel_by_key = {
        (str(y), str(i), Path(f).stem): rel
        for y, i, f, rel in connection.execute(
            "select y.year, i.name, p.source_file_name, p.source_rel_path from pages p join issues i on i.id=p.issue_id join year_packages y on y.id=i.year_package_id"
        )
    }
    connection.close()
    rel_by_page = {ref.key: rel_by_key.get(key, "") for key, ref in index.items()}
    sample = _read_sample(out_dir)
    control_pages = [(r["pdf"], int(r["page"])) for r in sample if r["kind"] == "control"][:controls]
    wanted: dict[str, set[int]] = {}
    for pdf_name, page in list(truths) + control_pages:
        wanted.setdefault(pdf_name, set()).add(page)
    tasks = []
    for pdf_name, pages in sorted(wanted.items()):
        path = nogeo_dir / pdf_name
        if not path.is_file():
            continue
        page_truths = {p: truths[(pdf_name, p)] for p in pages if (pdf_name, p) in truths}
        rels = {p: rel_by_page.get((pdf_name, p), "") for p in pages}
        ordered = sorted(pages)
        for start in range(0, len(ordered), CHUNK_PAGES):
            chunk = ordered[start : start + CHUNK_PAGES]
            tasks.append(
                (
                    str(path),
                    chunk,
                    {p: page_truths[p] for p in chunk if p in page_truths},
                    str(layout_dir) if layout_dir else "",
                    {p: rels[p] for p in chunk},
                )
            )
    total = sum(len(t[1]) for t in tasks)
    click.echo(
        f"страниц с эталоном: {len(truths)}, областей: {sum(len(t.boxes) for t in truths.values())}, контрольных: {len(control_pages)}, задач: {len(tasks)}, воркеров: {jobs}"
    )
    if jobs <= 1:
        outcomes = map(_eval_chunk, tasks)
    else:
        pool = ProcessPoolExecutor(
            max_workers=jobs, mp_context=multiprocessing.get_context("forkserver"), initializer=_init_worker
        )
        outcomes = pool.map(_eval_chunk, tasks)
    rows: list[dict] = []
    totals: dict[str, dict] = {name: {} for name in SOURCES}
    with tqdm(total=total, desc="страницы", unit="стр") as bar:
        for chunk_rows, scores in outcomes:
            rows.extend(chunk_rows)
            for name, score in scores.items():
                for key, value in score.items():
                    if isinstance(value, (int, float)) and key not in (
                        "precision",
                        "recall",
                        "coverage",
                        "f1",
                        "iou_median",
                    ):
                        totals[name][key] = totals[name].get(key, 0) + value
            bar.update(CHUNK_PAGES)
    summary: dict[str, dict] = {}
    for name, score in totals.items():
        predicted, matched, truth = score.get("predicted", 0), score.get("matched", 0), score.get("truth", 0)
        precision = matched / predicted if predicted else 0.0
        recall = score.get("truth_hit", 0) / truth if truth else 0.0
        summary[name] = {
            **score,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "coverage": round(score.get("truth_covered", 0) / truth, 3) if truth else 0.0,
            "f1": round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0,
        }
    _write_csv(out_dir / "lineart_eval.csv", rows, EVAL_FIELDS)
    (out_dir / "lineart_eval.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    click.echo(
        f"{'источник':10} {'предск.':>8} {'совп.':>6} {'точн.':>6} {'полн.':>6} {'накр.':>6} {'F1':>6} {'ложн.на контр.':>15}"
    )
    for name, score in summary.items():
        click.echo(
            f"{name:10} {score.get('predicted', 0):>8} {score.get('matched', 0):>6} {score['precision']:>6.2f} {score['recall']:>6.2f} {score['coverage']:>6.2f} {score['f1']:>6.2f} {score.get('false_on_pages_without_truth', 0):>15}"
        )


# --- llm-compare ---------------------------------------------------------------------------


LLM_FIELDS = [
    "pdf",
    "page",
    "struct_line",
    "token",
    "verdict",
    "geometry_garbage",
    "llm_garbage",
    "cost_usd",
    "latency_s",
    "error",
]


@main.command("llm-compare")
@click.option("--out-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--model", default="deepseek-v41-flash", show_default=True)
@click.option("--limit", default=150, show_default=True, type=int, help="строк на сравнение")
@click.option("--seed", default=7, show_default=True, type=int)
@click.option("--api-key", default=None, help="ключ OpenRouter; по умолчанию OPENROUTER_API_KEY")
def llm_compare(out_dir, model, limit, seed, api_key) -> None:
    """Сравнить геометрический вердикт с вычёркиванием мусора языковой моделью → llm_compare.csv/json."""
    import json
    import random

    from tqdm import tqdm

    from ocr_utils.external_ocr_services.client import OpenRouterClient, api_key_from
    from ocr_utils.external_ocr_services.models import resolve

    from ocr_utils.text_layer_fix.llm_sanitize import Agreement, ask, lines_from_payload

    cases = []
    for row in _read_sample(out_dir):
        path = cache_path(out_dir, row["pdf"], int(row["page"]))
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not payload.get("error"):
                cases.extend(lines_from_payload(payload))
    random.Random(seed).shuffle(cases)
    cases = cases[:limit]
    click.echo(f"строк на сравнение: {len(cases)}, модель {model}")
    client = OpenRouterClient(api_key_from(api_key))
    spec = resolve(model)
    agreement = Agreement()
    rows: list[dict] = []
    for case in tqdm(cases, desc="строки"):
        ask(client, spec, case)
        agreement.add(case)
        for token, verdict, geometric, llm in zip(case.tokens, case.verdicts, case.geometric_garbage, case.llm_garbage):
            rows.append(
                {
                    "pdf": case.pdf,
                    "page": case.page,
                    "struct_line": case.struct_line,
                    "token": token,
                    "verdict": verdict,
                    "geometry_garbage": geometric,
                    "llm_garbage": llm,
                    "cost_usd": round(case.cost_usd, 6),
                    "latency_s": round(case.latency_s, 2),
                    "error": case.error,
                }
            )
    _write_csv(out_dir / "llm_compare.csv", rows, LLM_FIELDS)
    summary = agreement.to_json()
    (out_dir / "llm_compare.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    click.echo(json.dumps(summary, ensure_ascii=False))


# --- reclassify -------------------------------------------------------------------------------


def _reclassify_chunk(args: tuple) -> int:
    """Пересчитать вердикты слов по кэшу (зоны и чтения — из JSON, слой — заново из PDF)."""
    import json

    import fitz
    import pikepdf

    from ocr_utils.external_ocr_services.hyphen_join import default_morph

    from ocr_utils.text_layer_fix.classify import classify_words
    from ocr_utils.text_layer_fix.raster import page_raster
    from ocr_utils.text_layer_fix.text_layer import load_layer
    from ocr_utils.text_layer_fix.zones import RotatedZone

    path, pages, out_dir = Path(args[0]), args[1], Path(args[2])
    known = default_morph().known
    done = 0
    with fitz.open(str(path)) as doc, pikepdf.open(str(path)) as pdf:
        for index in pages:
            cache = cache_path(out_dir, path.name, index)
            if not cache.is_file():
                continue
            payload = json.loads(cache.read_text(encoding="utf-8"))
            if payload.get("error"):
                continue
            page = doc[index]
            layer = load_layer(page, pdf)
            raster = page_raster(page)
            zones = [RotatedZone.from_json(z) for z in payload["zones"]]
            readable = {int(k) for k, r in payload["readings"].items() if r.get("accepted")}
            verdicts = classify_words(layer, zones, raster.to_px(), raster.dpi, readable, known)
            payload["words"] = [v.to_json() for v in verdicts]
            cache.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            done += 1
    return done


@main.command()
@click.option("--pdf-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--out-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--only", multiple=True)
@click.option("--jobs", default=16, show_default=True, type=int)
def reclassify(pdf_dir, out_dir, only, jobs) -> None:
    """Пересчитать вердикты слов по кэшу (после второго мнения или смены порогов) → words.csv, pages.csv."""
    from tqdm import tqdm

    by_pdf: dict[str, list[int]] = {}
    for row in _read_sample(out_dir):
        if only and not any(sub in row["pdf"] for sub in only):
            continue
        by_pdf.setdefault(row["pdf"], []).append(int(row["page"]))
    tasks = []
    for name, pages in sorted(by_pdf.items()):
        if not (pdf_dir / name).is_file():
            continue
        pages = sorted(set(pages))
        for start in range(0, len(pages), CHUNK_PAGES):
            tasks.append((str(pdf_dir / name), pages[start : start + CHUNK_PAGES], str(out_dir)))
    if jobs <= 1:
        outcomes = map(_reclassify_chunk, tasks)
    else:
        pool = ProcessPoolExecutor(
            max_workers=jobs, mp_context=multiprocessing.get_context("forkserver"), initializer=_init_worker
        )
        outcomes = pool.map(_reclassify_chunk, tasks)
    done = sum(tqdm(outcomes, total=len(tasks), desc="задачи"))
    import json

    rows = []
    for row in _read_sample(out_dir):
        path = cache_path(out_dir, row["pdf"], int(row["page"]))
        if path.is_file():
            rows.append(_summary_row(json.loads(path.read_text(encoding="utf-8"))))
    _write_csv(out_dir / "pages.csv", rows, PAGE_FIELDS)
    _collect_details(out_dir)
    click.echo(
        f"пересчитано страниц: {done}; delete {sum(r['delete'] for r in rows)}, sanitize {sum(r['sanitize'] for r in rows)}, suspect {sum(r['suspect'] for r in rows)}"
    )


@main.command()
@click.option("--out-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--md-out", type=click.Path(dir_okay=False, path_type=Path), help="куда записать markdown-сводку")
def report(out_dir, md_out) -> None:
    """Сводка прогона по CSV/JSON каталога → markdown (в stdout или файл)."""
    from ocr_utils.text_layer_fix.report import full_report

    text = full_report(out_dir)
    if md_out:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(text, encoding="utf-8")
        click.echo(f"записано: {md_out}")
    else:
        click.echo(text)
