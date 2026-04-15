"""CLI-точка входа: python -m ocr_utils или ocr-utils."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from ocr_utils.pipeline import process_directory, process_single_pdf


def _parse_pages(value: str) -> list[int] | slice | None:
    """Разобрать строку с номерами страниц.

    Форматы:
        "all"         → None (все страницы)
        "0,2,5"       → [0, 2, 5]
        "1:10"        → slice(1, 10)
        "1:10:2"      → slice(1, 10, 2)
    """
    value = value.strip()
    if value.lower() == "all":
        return None
    if ":" in value:
        parts = value.split(":")
        args = [int(p) if p else None for p in parts]
        return slice(*args)
    return [int(x.strip()) for x in value.split(",")]


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Подробный вывод (DEBUG)")
@click.option("-q", "--quiet", is_flag=True, help="Минимальный вывод (только ошибки)")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool) -> None:
    """Обработка PDF-сканов журналов: разбивка разворотов на страницы + OCR."""
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.ERROR
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@cli.command()
@click.argument("src", type=click.Path(exists=True, path_type=Path))
@click.argument("dst", type=click.Path(path_type=Path))
@click.option(
    "--pages",
    default="all",
    help="Какие страницы обрабатывать: 'all', '0,2,5' или '1:10' (0-based, Python slice). По умолчанию: all",
)
@click.option("--tmp-dir", type=click.Path(path_type=Path), default=None, help="Директория для временных файлов")
@click.option("--language", default="rus", help="Язык OCR (по умолчанию: rus)")
@click.option("--upscale-ratio", default=2.0, type=float, help="Коэффициент увеличения изображений (по умолчанию: 2.0)")
@click.option("--deskew/--no-deskew", default=True, help="Выравнивание страниц при OCR (по умолчанию: включено)")
@click.option("--clean/--no-clean", default=True, help="Очистка шума при OCR (по умолчанию: включено)")
@click.option("--rotate/--no-rotate", default=True, help="Автоповорот страниц при OCR (по умолчанию: включено)")
def single(
    src: Path,
    dst: Path,
    pages: str,
    tmp_dir: Path | None,
    language: str,
    upscale_ratio: float,
    deskew: bool,
    clean: bool,
    rotate: bool,
) -> None:
    """Обработать один PDF-файл."""
    pages_parsed = _parse_pages(pages)
    process_single_pdf(
        src_pdf=src,
        dst_pdf=dst,
        pages=pages_parsed,
        tmp_dir=tmp_dir,
        language=language,
        upscale_ratio=upscale_ratio,
        deskew=deskew,
        clean=clean,
        rotate_pages=rotate,
    )


@cli.command()
@click.argument("src", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("dst", type=click.Path(path_type=Path))
@click.option("--language", default="rus", help="Язык OCR (по умолчанию: rus)")
@click.option("--upscale-ratio", default=2.0, type=float, help="Коэффициент увеличения изображений (по умолчанию: 2.0)")
@click.option("--deskew/--no-deskew", default=True, help="Выравнивание страниц при OCR (по умолчанию: включено)")
@click.option("--clean/--no-clean", default=True, help="Очистка шума при OCR (по умолчанию: включено)")
@click.option("--rotate/--no-rotate", default=True, help="Автоповорот страниц при OCR (по умолчанию: включено)")
def dir(
    src: Path,
    dst: Path,
    language: str,
    upscale_ratio: float,
    deskew: bool,
    clean: bool,
    rotate: bool,
) -> None:
    """Рекурсивно обработать все PDF в директории."""
    results = process_directory(
        src_dir=src,
        dst_dir=dst,
        language=language,
        upscale_ratio=upscale_ratio,
        deskew=deskew,
        clean=clean,
        rotate_pages=rotate,
    )
    errors = {k: v for k, v in results.items() if v is not None}
    if errors:
        click.echo(f"\nОшибки ({len(errors)}):", err=True)
        for path, err in errors.items():
            click.echo(f"  {path}: {err}", err=True)
        sys.exit(1)


def main() -> None:
    """Точка входа для CLI."""
    cli()


if __name__ == "__main__":
    main()
