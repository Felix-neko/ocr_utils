"""CLI: поиск полос с просвечивающей бумагой в папке со сканами."""

import math
import os
from pathlib import Path

import click

from ocr_utils.defocus_detection.image_io import SUPPORTED_SUFFIXES, collect_images
from ocr_utils.show_through_detection.analysis import (
    all_halves,
    analyze_folder,
    sort_files_worst_first,
    sort_worst_first,
)
from ocr_utils.show_through_detection.metrics import (
    ALGORITHMS,
    CHOICES,
    COMBO_MEMBERS,
    COMBO_NAME,
    DEFAULT_ALGORITHM,
    FALLBACK,
)
from ocr_utils.show_through_detection.report import (
    LinkDirError,
    files_table,
    halves_table,
    markdown_report,
    write_csv,
    write_link_dir,
)

# Сколько ядер не занимать при автоматическом выборе числа процессов.
DEFAULT_RESERVED_CORES = 2

# Порог по умолчанию задан в долях от калибровочного порога метрики, а не в её единицах:
# так одно и то же число работает и для основной метрики, и для запасной, у которых
# шкалы разные. 1.0 — ровно тот порог, что откалиброван в модуле метрики.
DEFAULT_THRESHOLD = 1.0


def _algorithm_help() -> str:
    """Собирает справку по доступным метрикам.

    Returns:
        Многострочный текст для эпилога --help.
    """
    # "\b" в начале абзаца просит click не переносить строки — иначе список слипнется.
    lines = ["Метрики (--algorithm), у всех шкала «больше = сильнее просвет»:", "", "\b"]
    for name, spec in ALGORITHMS.items():
        threshold = f"порог {spec.threshold:g}" if spec.threshold is not None else "без порога"
        lines.append(f"  {name:<14} {spec.summary} ({threshold})")
    lines.append(f"  {COMBO_NAME:<14} средний ранг по {' + '.join(COMBO_MEMBERS)} — только ранжирование, без порога")
    lines.append("")
    lines.append(
        f"Полосам без чистых полей балл считает запасная метрика {FALLBACK} — в отчёте они "
        "помечены оговоркой «нет опорных полей»."
    )
    lines.append("Поддерживаемые файлы: " + ", ".join(sorted(SUPPORTED_SUFFIXES)) + ".")
    return "\n".join(lines)


def _select(total: int, count: int | None, percent: float | None, threshold_hits: int | None) -> tuple[int, str]:
    """Определяет, сколько строк показать в отчёте, и как это описать словами.

    Args:
        total: Сколько записей доступно.
        count: Запрошенное число худших или None.
        percent: Запрошенная доля худших в процентах или None.
        threshold_hits: Сколько записей превысило порог; используется, если отбор
            не задан явно.

    Returns:
        Пара (сколько показать, описание словами).
    """
    if count is not None:
        limit = min(count, total)
        return limit, f"худшие {limit}"
    if percent is not None:
        limit = max(1, math.ceil(total * percent / 100.0)) if total else 0
        return limit, f"худшие {percent:g}% ({limit} записей)"
    if threshold_hits is not None:
        return threshold_hits, f"превысившие порог ({threshold_hits} из {total})"
    return total, "все"


@click.command(context_settings=dict(help_option_names=["-h", "--help"]), epilog=_algorithm_help())
@click.argument("input_dir", type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path))
@click.option("--recursive/--no-recursive", default=False, show_default=True, help="Обходить вложенные папки.")
@click.option(
    "--algorithm",
    "-a",
    type=click.Choice(CHOICES),
    default=DEFAULT_ALGORITHM,
    show_default=True,
    help="Метрика просвечивания (описания — в конце справки).",
)
@click.option(
    "--threshold",
    type=float,
    default=DEFAULT_THRESHOLD,
    show_default=True,
    help="Порог в долях от калибровочного порога метрики: 1.0 — как откалибровано, 1.5 — строже.",
)
@click.option("--worst-count", type=int, default=None, help="Показать ровно столько худших полос.")
@click.option("--worst-percent", type=float, default=None, help="Показать столько процентов худших полос.")
@click.option("--workers", type=int, default=0, show_default=True, help="Число процессов; 0 — по числу ядер.")
@click.option(
    "--reserve-cores",
    type=int,
    default=DEFAULT_RESERVED_CORES,
    show_default=True,
    help="Сколько ядер оставить системе при автоматическом выборе числа процессов.",
)
@click.option(
    "--txt-report",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Куда сохранить текстовый отчёт.",
)
@click.option(
    "--md-report", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Куда сохранить markdown-отчёт."
)
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Куда сохранить CSV со ВСЕМИ полосами.",
)
@click.option(
    "--link-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Куда разложить симлинки на кадры из отчёта.",
)
@click.option("--quiet", is_flag=True, help="Не показывать полосу прогресса.")
def main(
    input_dir: Path,
    recursive: bool,
    algorithm: str,
    threshold: float,
    worst_count: int | None,
    worst_percent: float | None,
    workers: int,
    reserve_cores: int,
    txt_report: Path | None,
    md_report: Path | None,
    csv_path: Path | None,
    link_dir: Path | None,
    quiet: bool,
) -> None:
    """Ищет в INPUT_DIR полосы, на которых просвечивает текст с оборота листа.

    Печатает два отчёта: по полосам (какая именно страница плоха) и по кадрам
    (что пересканировать — переснимают разворот целиком). Если отбор не задан,
    в отчёт идут записи, превысившие порог.
    """
    if worst_percent is not None and worst_count is not None:
        raise click.UsageError("--worst-percent и --worst-count взаимоисключающи: задайте что-то одно.")
    if algorithm == COMBO_NAME and (worst_percent is None and worst_count is None):
        # У среднего ранга нет калибровочного порога: он зависит от состава выборки,
        # и «превысившие порог» для него означало бы просто «верхняя половина списка».
        raise click.UsageError(
            f"--algorithm {COMBO_NAME} даёт только ранжирование без порога: "
            "задайте --worst-count или --worst-percent."
        )

    files = collect_images(input_dir, recursive=recursive)
    if not files:
        raise click.ClickException(f"В {input_dir} не найдено поддерживаемых изображений.")

    if workers == 0:
        # Часть ядер оставляем системе: прогон по паку идёт часами, и всё это время
        # машиной надо продолжать пользоваться. Сами воркеры ещё и понижают приоритет.
        workers = min(len(files), max(1, (os.cpu_count() or 1) - reserve_cores))

    click.echo(f"Кадров: {len(files)}, метрика: {algorithm}, порог: {threshold:g}×, процессов: {workers}", err=True)

    results = analyze_folder(files, algorithm=algorithm, workers=workers, progress=not quiet)

    halves = sort_worst_first(all_halves(results))
    measured = [h for h in halves if math.isfinite(h.severity)]
    over = [h for h in measured if h.severity >= threshold]
    limit, shown = _select(len(halves), worst_count, worst_percent, len(over))
    selected = halves[:limit]

    ranked_files = sort_files_worst_first(results)
    flagged = [r for r in ranked_files if math.isfinite(r.severity) and r.severity >= threshold]

    text_blocks = [
        "== 1. ПОЛОСЫ " + "=" * 60,
        halves_table(selected, algorithm, threshold, total=len(halves)),
        "\n== 2. КАДРЫ НА ПЕРЕСКАНИРОВАНИЕ " + "=" * 41,
        files_table(flagged, threshold, total=len(results)),
    ]
    for block in text_blocks:
        click.echo(block)

    skipped = [h for h in halves if h.problem]
    if skipped:
        click.echo(f"Без балла (нет текста на полосе): {len(skipped)} полос.", err=True)
    no_margin = [h for h in halves if h.note]
    if no_margin:
        click.echo(f"Считаны запасной метрикой (нет опорных полей): {len(no_margin)} полос.", err=True)
    failed = [r for r in results if r.error]
    if failed:
        click.echo(f"Не прочитано кадров: {len(failed)}.", err=True)

    if txt_report is not None:
        txt_report.parent.mkdir(parents=True, exist_ok=True)
        # Пишем те же блоки, что напечатаны выше, но напрямую в файл — минуя stdout,
        # который при параллельном прогоне делят с полосой прогресса.
        txt_report.write_text("\n".join(text_blocks) + "\n", encoding="utf-8")
        click.echo(f"Текстовый отчёт: {txt_report}", err=True)

    if md_report is not None:
        md_report.parent.mkdir(parents=True, exist_ok=True)
        md_report.write_text(
            markdown_report(
                selected, flagged, algorithm, threshold, len(halves), len(results), input_dir.resolve(), shown
            ),
            encoding="utf-8",
        )
        click.echo(f"Markdown-отчёт: {md_report}", err=True)

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        # В CSV всегда пишем все полосы: он нужен для калибровки порога, а не для чтения.
        write_csv(csv_path, halves, results, algorithm)
        click.echo(f"CSV: {csv_path}", err=True)

    if link_dir is not None:
        try:
            root, made = write_link_dir(link_dir, flagged, base=input_dir, threshold=threshold)
        except LinkDirError as error:
            raise click.ClickException(str(error)) from error
        click.echo(f"Симлинки: {root}/ ({made} шт.)", err=True)


if __name__ == "__main__":
    main()
