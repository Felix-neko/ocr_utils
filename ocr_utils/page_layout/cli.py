"""CLI разбора структуры страницы: кэш surya (импорт старого, проверка) и разбор одной страницы."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from ocr_utils.page_layout.image import Variant
from ocr_utils.page_layout.surya.cache import SuryaCache, import_legacy

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
VARIANTS = tuple(v.value for v in Variant)


def _set_log_level(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@click.group()
def main() -> None:
    """Разбор структуры страницы: растр, таблицы, line art, повёрнутый текст, ориентация, surya."""


@main.command("import-legacy-cache")
@click.option(
    "--src", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path), help="Старый кэш pickle."
)
@click.option("--dst", required=True, type=click.Path(file_okay=False, path_type=Path), help="Корень нового кэша.")
@click.option("--variant", required=True, type=click.Choice(VARIANTS), help="Под каким вариантом класть.")
@click.option(
    "--overwrite/--no-overwrite", default=False, show_default=True, help="Перезаписывать существующие записи."
)
@click.option("--log-level", default="INFO", show_default=True, type=click.Choice(LOG_LEVELS, case_sensitive=False))
def import_legacy_command(src: Path, dst: Path, variant: str, overwrite: bool, log_level: str) -> None:
    """Перенести старый кэш surya (pickle по rel_path) в новый формат записями legacy."""
    _set_log_level(log_level)
    done, skipped, broken = import_legacy(src, Variant(variant), SuryaCache(dst), overwrite=overwrite)
    click.echo(f"Перенесено {done}, уже было {skipped}, битых {broken} -> {dst / variant}")


@main.command("verify-cache")
@click.option("--cache", "cache_root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--variant", required=True, type=click.Choice(VARIANTS))
@click.option("--limit", default=None, type=int, help="Проверить только первые N записей.")
@click.option("--log-level", default="INFO", show_default=True, type=click.Choice(LOG_LEVELS, case_sensitive=False))
def verify_cache_command(cache_root: Path, variant: str, limit: int | None, log_level: str) -> None:
    """Прочитать записи кэша и посчитать битые, legacy и без источника — без обращения к картинкам."""
    _set_log_level(log_level)
    cache = SuryaCache(cache_root, readonly=True)
    total = broken = legacy = no_source = 0
    for name in cache.iter_names(Variant(variant)):
        if limit is not None and total >= limit:
            break
        total += 1
        entry = cache.read(Variant(variant), name)
        if entry is None:
            broken += 1
            continue
        legacy += entry.legacy
        no_source += entry.source is None and not entry.legacy
    click.echo(f"Записей {total}: битых {broken}, legacy {legacy}, без отпечатка источника {no_source}.")
