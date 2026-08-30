"""CLI: поиск разворотов, у которых текст ушёл под переплёт."""

import math
from pathlib import Path

import click
import numpy as np

from ocr_utils.defocus_detection.image_io import SUPPORTED_SUFFIXES, collect_images
from ocr_utils.gutter_loss_detection.analysis import analyze_folder, sort_worst_first
from ocr_utils.gutter_loss_detection.metrics import THRESHOLD
from ocr_utils.gutter_loss_detection.report import (
    console_table,
    contact_sheet,
    markdown_report,
    write_csv,
    write_link_dir,
)

EPILOG = f"""\b
Балл: 0 — внутреннее поле в норме, 1 — строки упираются в сгиб.
Вердикт «таблица» — у корешка съедены цифры, восстановить нельзя, только пересканировать.
Поддерживаемые файлы: {', '.join(sorted(SUPPORTED_SUFFIXES))}.
"""


@click.command(context_settings=dict(help_option_names=["-h", "--help"]), epilog=EPILOG)
@click.argument("input_dir", type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path))
@click.option("--recursive/--no-recursive", default=False, show_default=True, help="Обходить вложенные папки.")
@click.option("--jobs", type=int, default=None, help="Процессов счёта (по умолчанию ядра минус запас).")
@click.option("--threshold", type=float, default=THRESHOLD, show_default=True, help="Порог по баллу.")
@click.option("--count", type=int, default=None, help="Показать столько худших кадров.")
@click.option("--csv", "csv_path", type=click.Path(path_type=Path), default=None, help="Куда писать полный CSV.")
@click.option("--md-report", type=click.Path(path_type=Path), default=None, help="Куда писать markdown-отчёт.")
@click.option("--link-dir", type=click.Path(path_type=Path), default=None, help="Папка симлинков на худшие кадры.")
@click.option("--sheet", type=click.Path(path_type=Path), default=None, help="Куда писать лист врезок для проверки.")
def main(input_dir, recursive, jobs, threshold, count, csv_path, md_report, link_dir, sheet):
    """Ранжирует развороты по тому, насколько текст ушёл под переплёт."""
    files = collect_images(input_dir, recursive=recursive)
    if not files:
        raise click.ClickException(f"в {input_dir} нет поддерживаемых изображений")
    click.echo(f"кадров: {len(files)}")

    results = sort_worst_first(analyze_folder(files, jobs=jobs, threshold=threshold))
    hit = [r for r in results if math.isfinite(r.score) and r.score >= threshold]
    limit = count if count is not None else len(hit)

    click.echo(console_table(results, limit))
    measured = sum(1 for r in results if math.isfinite(r.score))
    tables = sum(1 for r in hit if r.code == "таблица")
    click.echo(
        f"\nизмерено {measured} из {len(results)}; порог перешли {len(hit)}, "
        f"из них таблиц {tables} (только пересканировать), текста {len(hit) - tables}"
    )

    base = input_dir if input_dir.is_dir() else input_dir.parent
    if csv_path:
        write_csv(csv_path, results)
        click.echo(f"CSV: {csv_path}")
    if md_report:
        md_report.parent.mkdir(parents=True, exist_ok=True)
        md_report.write_text(markdown_report(results, limit, threshold, base), encoding="utf-8")
        click.echo(f"markdown: {md_report}")
    if link_dir:
        root, made = write_link_dir(link_dir, results[:limit])
        click.echo(f"симлинков: {made} в {root}")
    if sheet:
        click.echo(f"лист врезок: {contact_sheet(sheet, results[:limit])}")


if __name__ == "__main__":
    main()
