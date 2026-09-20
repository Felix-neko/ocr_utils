"""Разовая утилита: перевести сырые ответы модели в ``--debug-dir`` из ``.raw.txt`` в ``.raw.json`` для чтения.

ЗАЧЕМ. Раньше ``external_ocr_services`` писал ответ модели в ``имя.raw.txt`` как есть — одной строкой
JSON, иногда в ограждении ```json и с хвостом. Теперь ``ocr.write_raw_response`` пишет ``имя.raw.json``
с отступами в 4 пробела и кириллицей без escape, а ``.raw.txt`` остаётся только там, где JSON в ответе
не нашлось. Скрипт приводит уже накопленные файлы к тому же виду: каждый ``*.raw.txt`` (в том числе
``*.pass2.raw.txt``) разбирается тем же терпимым способом, что и при прогоне; разобрался — рядом
появляется ``.raw.json``, а ``.raw.txt`` удаляется; не разобрался — файл не трогается.

Запуск:
    uv run python scripts/prettify_raw_responses.py /mnt/system/raw/mts/pack1_external_ocr_services/debug
    uv run python scripts/prettify_raw_responses.py <папка> --dry-run   # только посчитать, ничего не писать
"""

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ocr_utils.external_ocr_services.ocr import write_raw_response
from ocr_utils.external_ocr_services.schema import pretty_json

RAW_TEXT_SUFFIX = ".raw.txt"
RAW_JSON_SUFFIX = ".raw.json"


@click.command()
@click.argument("debug_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--dry-run", is_flag=True, help="Только посчитать, сколько файлов разберётся, ничего не писать.")
def main(debug_dir: Path, dry_run: bool) -> None:
    """Переписать все ``*.raw.txt`` под DEBUG_DIR в ``*.raw.json`` с отступами; нечитаемые оставить как есть.

    Args:
        debug_dir: Корень отладочного выхода прогона (``--debug-dir``), обходится рекурсивно.
        dry_run: Не писать и не удалять файлы — только вывести, что было бы сделано.
    """
    converted, kept = 0, []
    for text_path in sorted(debug_dir.rglob(f"*{RAW_TEXT_SUFFIX}")):
        # Суффикс составной (``.pass2.raw.txt``), поэтому режется по имени, а не через with_suffix.
        json_path = text_path.with_name(text_path.name[: -len(RAW_TEXT_SUFFIX)] + RAW_JSON_SUFFIX)
        text = text_path.read_text(encoding="utf-8")
        if dry_run:
            written = json_path if pretty_json(text) is not None else text_path
        else:
            # Разобрался — записан .raw.json, а .raw.txt удалён; нет — .raw.txt перезаписан тем же текстом.
            written = write_raw_response(json_path, text_path, text)
        if written == json_path:
            converted += 1
        else:
            kept.append(text_path)
    click.echo(f"{'[dry-run] ' if dry_run else ''}переведено в .raw.json: {converted}, оставлено .raw.txt: {len(kept)}")
    for path in kept:
        click.echo(f"  не JSON: {path.relative_to(debug_dir)}")


if __name__ == "__main__":
    main()
