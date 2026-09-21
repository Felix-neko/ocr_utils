"""Командная строка детектора: тяжёлый ``scan`` и дешёвый ``export``.

Команд две, потому что вопрос «какой порог» решается глазами и не с первого раза.
``scan`` один раз считает признаки всех страниц и кладёт их в CSV; ``export`` сколько
угодно раз применяет к этому CSV разный порог и раскладывает картинки по папкам. Тяжёлый
проход по девяти тысячам страниц при этом делается ровно один.
"""

import logging
from pathlib import Path

import click

from ocr_utils.line_art_detection import export as export_module
from ocr_utils.line_art_detection.analysis import (
    DEFAULT_JOBS,
    DEFAULT_MIN_AGE_MINUTES,
    analyse_folder,
    collect_pdfs,
    detect_layout,
)
from ocr_utils.page_layout.line_art.features import params_for_dpi
from ocr_utils.line_art_detection.markup import load_pdf_markup
from ocr_utils.line_art_detection.render import DEFAULT_DPI
from ocr_utils.line_art_detection.report import console_table, markdown_report, read_csv, summary, write_csv

CONTEXT = dict(help_option_names=["-h", "--help"])


@click.group(context_settings=CONTEXT)
def cli() -> None:
    """Поиск страниц с крупным штриховым рисунком в бинаризованных PDF."""


@cli.command()
@click.option(
    "--input-dir", required=True, type=click.Path(exists=True, path_type=Path), help="Папка с PDF (или один файл PDF)."
)
@click.option("--recursive/--no-recursive", default=False, show_default=True, help="Обходить вложенные папки.")
@click.option("--csv", "csv_path", required=True, type=click.Path(path_type=Path), help="Куда писать полный CSV.")
@click.option("--md-report", type=click.Path(path_type=Path), default=None, help="Куда писать markdown-отчёт.")
@click.option("--db", type=click.Path(exists=True, path_type=Path), default=None, help="База разметки (отсев растра).")
@click.option("--pack-name", default="пак-1", show_default=True, help="Имя пака в базе разметки.")
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Число процессов счёта.")
@click.option("--dpi", type=int, default=DEFAULT_DPI, show_default=True, help="Разрешение рендера страниц.")
@click.option("--threshold", type=float, default=0.05, show_default=True, help="Порог покрытия для markdown-отчёта.")
@click.option("--ink-threshold", type=int, default=None, help="Порог краски (страница битональная, обычно не нужен).")
@click.option("--pitch-px", type=int, default=None, help="Шаг текстовых строк при 600 dpi (замер пака-1 — 89).")
@click.option("--stroke-px", type=int, default=None, help="Толщина штриха при 600 dpi (замер пака-1 — 7).")
@click.option("--min-area-px", type=int, default=None, help="Минимальная площадь пятна-кандидата при 600 dpi.")
@click.option("--max-aspect", type=float, default=None, help="Предел вытянутости рамки пятна (выше — линейка).")
@click.option("--no-tables", is_flag=True, help="Не искать разлинованные таблицы по скоплениям линеек.")
@click.option("--reject-halftone", is_flag=True, help="Отсеивать растр по пикселям (на паке-1 вредно, см. features).")
@click.option("--use-surya-layout", is_flag=True, help="Добавить предложения Surya: таблицы без линеек и формулы.")
@click.option("--surya-cache", type=click.Path(path_type=Path), default=None, help="Папка кэша разметки Surya.")
@click.option(
    "--min-age-minutes",
    type=float,
    default=DEFAULT_MIN_AGE_MINUTES,
    show_default=True,
    help="PDF моложе этого возраста считается недописанным и пропускается: из идущей "
    "выгрузки FineReader читать нельзя. 0 — брать все файлы.",
)
@click.option("--limit", type=int, default=None, help="Взять не больше стольких PDF (для пробы).")
def scan(
    input_dir,
    recursive,
    csv_path,
    md_report,
    db,
    pack_name,
    jobs,
    dpi,
    threshold,
    ink_threshold,
    pitch_px,
    stroke_px,
    min_area_px,
    max_aspect,
    no_tables,
    reject_halftone,
    use_surya_layout,
    surya_cache,
    min_age_minutes,
    limit,
) -> None:
    """Считает признаки по всем страницам всех PDF и пишет CSV."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    overrides = {}
    if ink_threshold is not None:
        overrides["ink_threshold"] = ink_threshold
    if pitch_px is not None:
        overrides["pitch_px"] = pitch_px
    if stroke_px is not None:
        overrides["stroke_px"] = stroke_px
    if min_area_px is not None:
        overrides["min_area_px"] = min_area_px
    if max_aspect is not None:
        overrides["max_aspect"] = max_aspect
    if no_tables:
        overrides["detect_tables"] = False
    if reject_halftone:
        overrides["reject_halftone"] = True
    params = params_for_dpi(dpi, **overrides)

    pdfs, fresh = collect_pdfs(input_dir, recursive=recursive, min_age_minutes=min_age_minutes)
    if fresh:
        # Молчать об этом нельзя: отчёт по неполной папке внешне неотличим от полного.
        click.echo(f"ПРОПУЩЕНО как ещё пишущиеся (моложе {min_age_minutes:g} мин): {len(fresh)}", err=True)
        for path in fresh[:10]:
            click.echo(f"    {path.name}", err=True)
        if len(fresh) > 10:
            click.echo(f"    ... ещё {len(fresh) - 10}", err=True)
    if not pdfs:
        raise click.ClickException(f"в {input_dir} не найдено готовых PDF")
    if limit:
        pdfs = pdfs[:limit]
    click.echo(f"PDF: {len(pdfs)}")

    markup = None
    if db:
        markup = load_pdf_markup(db, pack_name)
        known = sum(1 for pdf, _ in markup if pdf in {p.name for p in pdfs})
        click.echo(f"Разметка: {len(markup)} страниц в базе, из них {known} относятся к этим PDF")
    else:
        click.echo("Разметка не задана (--db): полутоновые фотографии отсеиваться не будут")

    surya_by_pdf = None
    if use_surya_layout:
        from ocr_utils.line_art_detection.layout import LayoutProposals

        surya_by_pdf = detect_layout(pdfs, params, LayoutProposals(surya_cache))

    results = analyse_folder(pdfs, params, markup=markup, jobs=jobs, surya_by_pdf=surya_by_pdf)
    write_csv(csv_path, results)
    click.echo(f"\nCSV: {csv_path}")
    click.echo(summary(results))
    click.echo("")
    click.echo(console_table(results))

    if md_report:
        Path(md_report).parent.mkdir(parents=True, exist_ok=True)
        Path(md_report).write_text(markdown_report(results, threshold), encoding="utf-8")
        click.echo(f"\nОтчёт: {md_report}")


@cli.command(name="export")
@click.option("--csv", "csv_path", required=True, type=click.Path(exists=True, path_type=Path), help="CSV от scan.")
@click.option("--out-dir", required=True, type=click.Path(path_type=Path), help="Куда складывать картинки.")
@click.option("--min-coverage", type=float, default=0.05, show_default=True, help="Порог покрытия.")
@click.option("--long-side", type=int, default=1600, show_default=True, help="Длинная сторона JPEG.")
@click.option("--jpeg-quality", type=int, default=85, show_default=True, help="Качество JPEG.")
@click.option("--jobs", type=int, default=DEFAULT_JOBS, show_default=True, help="Число процессов.")
@click.option(
    "--source",
    "sources",
    multiple=True,
    help="Брать только страницы, где находка пришла из такого источника: ink, rules, "
    "surya:Figure, surya:Table, surya:Equation, surya:Form. Можно повторять. "
    "Нужно прежде всего для формул: многострочная формула занимает 0.4-1.3% полосы "
    "(замер по эталону), и порогом покрытия, настроенным на рисунки, её не достать.",
)
@click.option("--limit", type=int, default=None, help="Выгрузить не больше стольких страниц.")
def export_cmd(csv_path, out_dir, min_coverage, long_side, jpeg_quality, jobs, sources, limit) -> None:
    """Выгружает страницы выше порога картинками, по папке на выпуск."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results = read_csv(csv_path)
    chosen = [r for r in results if r.status == "ok" and r.coverage >= min_coverage]
    if sources:
        wanted = set(sources)
        chosen = [r for r in chosen if wanted & set(r.sources.split(","))]
    chosen.sort(key=lambda r: (r.issue, r.page_no))
    if limit:
        chosen = chosen[:limit]
    if not chosen:
        raise click.ClickException(f"ни одной страницы с покрытием >= {min_coverage} и нужным источником")

    filtered = f", источники {'/'.join(sources)}" if sources else ""
    click.echo(f"Страниц к выгрузке: {len(chosen)} (порог {min_coverage:.1%}{filtered})")
    written = export_module.export_pages(chosen, Path(out_dir), long_side, jpeg_quality, jobs)
    click.echo(f"Записано файлов: {written}")
    click.echo(f"Папка: {out_dir}")
