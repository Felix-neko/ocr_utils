"""CLI: ранжирование сканов папки по качеству фокуса."""

import math
import os
from pathlib import Path

import click

from ocr_utils.defocus_detection.analysis import analyze_folder, sort_by_zonal, sort_worst_first
from ocr_utils.defocus_detection.image_io import SUPPORTED_SUFFIXES, collect_images
from ocr_utils.defocus_detection.metrics import ALGORITHMS, CHOICES, COMBO_MEMBERS, COMBO_NAME, DEFAULT_ALGORITHM
from ocr_utils.defocus_detection.report import console_table, markdown_report, write_csv, zonal_table
from ocr_utils.defocus_detection.scoring import AGGREGATIONS, DEFAULT_AGGREGATION, DEFAULT_QUANTILE
from ocr_utils.defocus_detection.tiles import DEFAULT_TILE_SIZE
from ocr_utils.defocus_detection.zonal import AXES

# Сколько ядер не занимать при автоматическом выборе числа процессов.
DEFAULT_RESERVED_CORES = 2


def _algorithm_help() -> str:
    """Собирает справку по доступным алгоритмам.

    Returns:
        Многострочный текст для эпилога --help.
    """
    # "\b" в начале абзаца просит click не переносить строки — иначе список слипнется.
    lines = ["Алгоритмы (--algorithm):", "", "\b"]
    for name, spec in ALGORITHMS.items():
        lines.append(f"  {name:<12} {spec.summary}")
    lines.append(f"  {COMBO_NAME:<12} средний ранг по {' + '.join(COMBO_MEMBERS)} — осторожный режим")
    lines.append("")
    lines.append("Поддерживаемые файлы: " + ", ".join(sorted(SUPPORTED_SUFFIXES)) + " (из RAF берётся JPEG-превью).")
    return "\n".join(lines)


def _select(total: int, count: int | None, percent: float | None) -> tuple[int, str]:
    """Определяет, сколько строк показать в отчёте, и как это описать словами.

    Args:
        total: Сколько файлов доступно для отчёта.
        count: Запрошенное число худших файлов или None.
        percent: Запрошенная доля худших файлов или None.

    Returns:
        Кортеж (сколько строк показать, человекочитаемое описание отбора).
    """
    if count is not None:
        limit = min(count, total)
        return limit, f"худшие {limit}"
    if percent is not None:
        limit = max(1, math.ceil(total * percent / 100.0)) if total else 0
        return limit, f"худшие {percent:g}% ({limit} файлов)"
    return total, "все файлы"


@click.command(context_settings=dict(help_option_names=["-h", "--help"]), epilog=_algorithm_help())
@click.argument("input_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--algorithm",
    "-a",
    type=click.Choice(CHOICES),
    default=DEFAULT_ALGORITHM,
    show_default=True,
    help="Алгоритм оценки фокуса (список — в конце справки).",
)
@click.option(
    "--worst-percent",
    "-p",
    type=click.FloatRange(0, 100, min_open=True),
    default=None,
    help="Показать только худшие N%% файлов (например, 5).",
)
@click.option("--worst-count", "-n", type=click.IntRange(min=1), default=None, help="Показать только N худших файлов.")
@click.option(
    "--zonal-percent",
    type=click.FloatRange(0, 100, min_open=True),
    default=None,
    help="Во второй отчёт (зональный расфокус) — худшие N%% файлов.",
)
@click.option("--zonal-count", type=click.IntRange(min=1), default=None, help="Во второй отчёт — N худших файлов.")
@click.option(
    "--zonal-axis",
    type=click.Choice(AXES),
    default="rows",
    show_default=True,
    help="Поперёк чего искать зону: rows — горизонтальные полосы (верно для полос "
    "с вертикальными колонками), cols — вертикальные.",
)
@click.option("--no-zonal", is_flag=True, help="Не считать и не печатать второй отчёт.")
@click.option("--recursive", "-r", is_flag=True, help="Обходить вложенные папки.")
@click.option(
    "--aggregate",
    "aggregation",
    type=click.Choice(AGGREGATIONS),
    default=DEFAULT_AGGREGATION,
    show_default=True,
    help="Как свести тайлы в балл: worst — квантиль самых мягких тайлов, "
    "median — медиана, best — квантиль самых резких («есть ли на полосе чёткий текст»).",
)
@click.option(
    "--quantile",
    type=click.FloatRange(0.5, 1.0),
    default=DEFAULT_QUANTILE,
    show_default=True,
    help="Какую долю тайлов брать в режимах worst/best.",
)
@click.option(
    "--tile-size",
    type=click.IntRange(min=0),
    default=DEFAULT_TILE_SIZE,
    show_default=True,
    help="Сторона тайла в пикселях; 0 — авто (девять тайлов по ширине кадра).",
)
@click.option(
    "--workers",
    "-j",
    type=click.IntRange(min=0),
    default=0,
    help="Число процессов; 0 — по числу ядер минус --reserve-cores.  [default: 0]",
)
@click.option(
    "--reserve-cores",
    type=click.IntRange(min=0),
    default=DEFAULT_RESERVED_CORES,
    show_default=True,
    help="Сколько ядер оставить системе при автоматическом выборе числа процессов. "
    "Воркеры вдобавок работают с пониженным приоритетом и в один поток каждый.",
)
@click.option(
    "--md-report", type=click.Path(path_type=Path), default=None, help="Записать markdown-отчёт по этому пути."
)
@click.option(
    "--csv", "csv_path", type=click.Path(path_type=Path), default=None, help="Записать полные результаты в CSV."
)
@click.option("--quiet", "-q", is_flag=True, help="Не показывать полосу прогресса.")
def main(
    input_dir: Path,
    algorithm: str,
    worst_percent: float | None,
    worst_count: int | None,
    zonal_percent: float | None,
    zonal_count: int | None,
    zonal_axis: str,
    no_zonal: bool,
    recursive: bool,
    aggregation: str,
    quantile: float,
    tile_size: int,
    workers: int,
    reserve_cores: int,
    md_report: Path | None,
    csv_path: Path | None,
    quiet: bool,
) -> None:
    """Ранжирует сканы в папке INPUT_DIR по качеству фокуса, худшие — сверху.

    Печатает два отчёта: по общему качеству фокуса и по зональному расфокусу
    (когда мягкая только часть кадра — такой файл может быть неплох в среднем).
    Если отбор не задан, в отчёт идут все файлы с числовой метрикой.
    """
    if worst_percent is not None and worst_count is not None:
        raise click.UsageError("--worst-percent и --worst-count взаимоисключающи: задайте что-то одно.")
    if zonal_percent is not None and zonal_count is not None:
        raise click.UsageError("--zonal-percent и --zonal-count взаимоисключающи: задайте что-то одно.")
    if no_zonal and (zonal_percent is not None or zonal_count is not None):
        raise click.UsageError("--no-zonal несовместим с --zonal-percent/--zonal-count.")

    files = collect_images(input_dir, recursive=recursive)
    if not files:
        raise click.ClickException(f"В {input_dir} не найдено поддерживаемых изображений.")

    if workers == 0:
        # Часть ядер оставляем системе: прогон по выпуску идёт минутами, и всё это время
        # машиной надо продолжать пользоваться. Одного лишь ограничения числа процессов
        # для этого мало — сами воркеры ещё и понижают себе приоритет (см. analysis.py).
        workers = min(len(files), max(1, (os.cpu_count() or 1) - reserve_cores))

    click.echo(f"Файлов: {len(files)}, алгоритм: {algorithm}, агрегация: {aggregation}, процессов: {workers}", err=True)

    results = analyze_folder(
        files,
        algorithm=algorithm,
        tile_size=tile_size,
        aggregation=aggregation,
        quantile=quantile,
        zonal_axis=None if no_zonal else zonal_axis,
        workers=workers,
        progress=not quiet,
    )
    results = sort_worst_first(results)
    limit, shown = _select(len(results), worst_count, worst_percent)
    selected = results[:limit]

    click.echo("== 1. ОБЩЕЕ КАЧЕСТВО ФОКУСА " + "=" * 40)
    click.echo(console_table(selected, algorithm, total=len(results)))

    zonal_selected = None
    zonal_shown = ""
    if not no_zonal:
        ranked = sort_by_zonal([r for r in results if r.zonal is not None])
        zonal_limit, zonal_shown = _select(len(ranked), zonal_count, zonal_percent)
        zonal_selected = ranked[:zonal_limit]
        click.echo("\n== 2. ЗОНАЛЬНЫЙ РАСФОКУС (мягкая часть кадра) " + "=" * 22)
        click.echo(zonal_table(zonal_selected, total=len(ranked)))
        skipped = len(results) - len(ranked)
        if skipped:
            click.echo(f"Без зональной оценки (мало текста): {skipped} файлов.", err=True)

    failed = [r for r in results if r.error]
    if failed:
        click.echo(f"\nНе прочитано файлов: {len(failed)} (перечислены в начале первой таблицы).", err=True)

    if md_report is not None:
        md_report.parent.mkdir(parents=True, exist_ok=True)
        md_report.write_text(
            markdown_report(
                selected, zonal_selected, algorithm, len(results), input_dir.resolve(), shown, zonal_shown, aggregation
            ),
            encoding="utf-8",
        )
        click.echo(f"Markdown-отчёт: {md_report}", err=True)

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        # В CSV всегда пишем все файлы: он нужен для калибровки порогов, а не для чтения глазами.
        write_csv(csv_path, results, algorithm)
        click.echo(f"CSV: {csv_path}", err=True)


if __name__ == "__main__":
    main()
