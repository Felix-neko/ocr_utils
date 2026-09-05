"""CLI: восстановление текста, срезанного корешком, по папке со сканами."""

import shutil
from collections import Counter
from pathlib import Path

import click
import cv2

from ocr_utils.defocus_detection.image_io import collect_images
from ocr_utils.gutter_loss_detection.analysis import analyze_folder, sort_worst_first
from ocr_utils.gutter_loss_restoration import lexicon as lexicon_module
from ocr_utils.gutter_loss_restoration import library as library_module
from ocr_utils.gutter_loss_restoration.ocrcache import ensure, load_json
from ocr_utils.gutter_loss_restoration.pipeline import build_shared, restore_many


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.argument("out_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--work",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Где держать кэш распознавания, словарь и библиотеку литер.",
)
@click.option(
    "--threshold", type=float, default=0.35, show_default=True, help="Порог детектора: какие кадры вообще чинить."
)
@click.option("--limit", type=int, default=None, help="Взять столько худших кадров.")
@click.option(
    "--copy-source/--no-copy-source",
    default=True,
    show_default=True,
    help="Класть рядом исходный кадр для сравнения «было-стало».",
)
def main(input_dir, out_dir, work, threshold, limit, copy_source):
    """Дописывает съеденные корешком хвосты строк на кадрах папки."""
    work = work or out_dir / "работа"
    files = collect_images(input_dir)
    if not files:
        raise click.ClickException(f"в {input_dir} нет изображений")

    click.echo(f"кадров в папке: {len(files)}; ищу пострадавшие…")
    found = sort_worst_first(analyze_folder(files, threshold=threshold))
    targets = [r for r in found if r.code == "текст" and r.score == r.score and r.score >= threshold]
    if limit:
        targets = targets[:limit]
    click.echo(
        f"к восстановлению: {len(targets)} (табличных пропущено: " f"{sum(1 for r in found if r.code == 'таблица')})"
    )

    click.echo("распознаю полосы (surya, GPU)…")
    for i, result in enumerate(files, 1):
        ensure(work / "ocr", result)
        if i % 25 == 0:
            click.echo(f"  {i}/{len(files)}")

    counter = lexicon_module.build(work / "ocr", load_json)
    lexicon_module.save(work / "словарь.json", counter)
    words = lexicon_module.load(work / "словарь.json")
    click.echo(f"словарь выпуска: {len(words)} слов")

    shared = build_shared(files, work)
    click.echo(f"библиотека литер: {len(shared)} знаков")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows, stats = restore_many([r.path for r in targets], work, words, shared, out_dir, copy_source)
    click.echo(f"\nвосстановлено кадров: {stats['кадров']}, строк: {stats['строк']}")
    click.echo("причины пропуска строк: " + ", ".join(f"{k} — {v}" for k, v in Counter(stats["отказы"]).most_common(6)))
    click.echo(f"результат: {out_dir}")


if __name__ == "__main__":
    main()
