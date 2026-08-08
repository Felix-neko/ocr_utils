"""Печать отчёта: консольная таблица, markdown и CSV."""

import csv
import math
from pathlib import Path
from urllib.parse import quote

from ocr_utils.defocus_detection.analysis import FileResult
from ocr_utils.defocus_detection.metrics import ALGORITHMS, COMBO_NAME, resolve


def _score_cell(value: float) -> str:
    """Форматирует балл, подбирая число знаков под порядок величины.

    Args:
        value: Числовое значение балла.

    Returns:
        Строковое представление или "—" для NaN.
    """
    if value is None or not math.isfinite(value):
        return "—"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def _extra_columns(algorithm: str) -> list[tuple[str, str]]:
    """Определяет дополнительные колонки отчёта (метрики combo или расшифровка балла).

    Args:
        algorithm: Имя выбранного алгоритма.

    Returns:
        Список пар (ключ, заголовок колонки).
    """
    if algorithm == COMBO_NAME:
        return [(name, name) for name in resolve(algorithm)]
    spec = ALGORITHMS[algorithm]
    return [("display", spec.display_unit)] if spec.display is not None else []


def _row_values(result: FileResult, algorithm: str) -> list[str]:
    """Собирает значения дополнительных колонок для одной строки.

    Args:
        result: Результат анализа файла.
        algorithm: Имя выбранного алгоритма.

    Returns:
        Список строковых значений в порядке ``_extra_columns``.
    """
    if algorithm == COMBO_NAME:
        return [_score_cell(result.per_metric.get(name, float("nan"))) for name in resolve(algorithm)]
    spec = ALGORITHMS[algorithm]
    if spec.display is None:
        return []
    if not math.isfinite(result.score):
        return ["—"]
    return [_score_cell(spec.display(result.score))]


def console_table(results: list[FileResult], algorithm: str, total: int) -> str:
    """Строит текстовую таблицу отчёта (худшие сверху).

    Args:
        results: Уже отсортированные и, при необходимости, урезанные результаты.
        algorithm: Имя выбранного алгоритма.
        total: Сколько файлов было проанализировано всего.

    Returns:
        Готовый к печати многострочный текст.
    """
    spec_unit = "средний ранг" if algorithm == COMBO_NAME else ALGORITHMS[algorithm].unit
    extras = _extra_columns(algorithm)

    headers = ["#", f"балл ({spec_unit})", *(title for _, title in extras), "файл"]
    rows: list[list[str]] = []
    for position, result in enumerate(results, 1):
        name = result.path.name if not result.error else f"{result.path.name}  [{result.error}]"
        rows.append([str(position), _score_cell(result.score), *_row_values(result, algorithm), name])

    return _render(
        headers, rows, f"Показано {len(results)} из {total} файлов; сортировка — от худшего фокуса к лучшему."
    )


def zonal_table(results: list[FileResult], total: int) -> str:
    """Строит текстовую таблицу отчёта по зональному расфокусу.

    Args:
        results: Отсортированные по перепаду и урезанные результаты.
        total: Сколько файлов имеет зональную оценку.

    Returns:
        Готовый к печати многострочный текст.
    """
    headers = ["#", "перепад", "σ резч.", "σ мягч.", "где мягче", "файл"]
    rows: list[list[str]] = []
    for position, result in enumerate(results, 1):
        zone = result.zonal
        if zone is None:
            rows.append([str(position), "—", "—", "—", "нет текста", result.path.name])
            continue
        sharp, soft = zone.profile[zone.best], zone.profile[zone.worst]
        rows.append(
            [str(position), f"+{zone.drop * 100:.0f}%", f"{sharp:.3f}", f"{soft:.3f}", zone.where(), result.path.name]
        )
    return _render(headers, rows, f"Показано {len(results)} из {total} файлов с зональной оценкой.")


def _render(headers: list[str], rows: list[list[str]], footer: str) -> str:
    """Выравнивает таблицу в моноширинный текст.

    Args:
        headers: Заголовки колонок.
        rows: Строки таблицы (той же длины, что и headers).
        footer: Строка-итог под таблицей.

    Returns:
        Многострочный текст таблицы.
    """
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i]) for i in range(len(headers))
    ]

    def fmt(cells: list[str]) -> str:
        # Последняя колонка (имя файла) выравнивается по левому краю и не дополняется.
        parts = [cells[i].rjust(widths[i]) for i in range(len(cells) - 1)]
        return "  ".join([*parts, cells[-1]])

    lines = [fmt(headers), "  ".join("-" * w for w in widths[:-1]) + "  " + "-" * max(widths[-1], 4)]
    lines.extend(fmt(row) for row in rows)
    lines.append("")
    lines.append(footer)
    return "\n".join(lines)


def _link(result: FileResult) -> str:
    """Кликабельная ссылка на файл для markdown-отчёта.

    Args:
        result: Результат анализа файла.

    Returns:
        Markdown-ссылка с абсолютным URL-кодированным путём.
    """
    link = f"[{result.path.name}]({quote(str(result.path.resolve()))})"
    return link + (f" — _{result.error}_" if result.error else "")


def markdown_report(
    overall: list[FileResult],
    zonal: list[FileResult] | None,
    algorithm: str,
    total: int,
    folder: Path,
    shown: str,
    zonal_shown: str,
    aggregation: str,
) -> str:
    """Строит markdown-отчёт из двух таблиц с кликабельными ссылками на файлы.

    Args:
        overall: Отсортированные и урезанные результаты по общему качеству фокуса.
        zonal: То же по зональному расфокусу либо None, если он не считался.
        algorithm: Имя выбранного алгоритма.
        total: Сколько файлов проанализировано всего.
        folder: Исходная папка.
        shown: Человекочитаемое описание отбора для первой таблицы.
        zonal_shown: То же для второй таблицы.
        aggregation: Режим агрегации тайлов.

    Returns:
        Текст markdown-документа.
    """
    spec_unit = "средний ранг" if algorithm == COMBO_NAME else ALGORITHMS[algorithm].unit
    extras = _extra_columns(algorithm)

    lines = [
        "# Подозрение на расфокус",
        "",
        f"Папка: `{folder}`",
        "",
        f"Алгоритм: **{algorithm}**, агрегация по тайлам: **{aggregation}**. " f"Проанализировано файлов: {total}.",
        "",
        "## 1. Общее качество фокуса",
        "",
        f"В таблице: {shown}. Сортировка — от худшего фокуса к лучшему.",
        "",
        "| # | балл (" + spec_unit + ") | " + " | ".join(t for _, t in extras) + (" | " if extras else "") + "файл |",
        "|--:|--:|" + "--:|" * len(extras) + "---|",
    ]
    for position, result in enumerate(overall, 1):
        cells = [str(position), _score_cell(result.score), *_row_values(result, algorithm), _link(result)]
        lines.append("| " + " | ".join(cells) + " |")

    if zonal is not None:
        lines += [
            "",
            "## 2. Зональный расфокус",
            "",
            f"В таблице: {zonal_shown}. Сортировка — по убыванию перепада резкости внутри кадра.",
            "",
            "Перепад — на сколько процентов шире штрих в самой мягкой полосе кадра по сравнению",
            "с самой резкой. Кадр может быть хорош в среднем и всё равно попасть сюда: значит,",
            "поплыла его часть.",
            "",
            "| # | перепад | σ резч. | σ мягч. | где мягче | файл |",
            "|--:|--:|--:|--:|---|---|",
        ]
        for position, result in enumerate(zonal, 1):
            zone = result.zonal
            if zone is None:
                lines.append(f"| {position} | — | — | — | нет текста | {_link(result)} |")
                continue
            sharp, soft = zone.profile[zone.best], zone.profile[zone.worst]
            lines.append(
                f"| {position} | +{zone.drop * 100:.0f}% | {sharp:.3f} | {soft:.3f} | "
                f"{zone.where()} | {_link(result)} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, results: list[FileResult], algorithm: str) -> None:
    """Сохраняет полные результаты в CSV (для собственного анализа и калибровки).

    Args:
        path: Куда записать файл.
        results: Отсортированные результаты.
        algorithm: Имя выбранного алгоритма.
    """
    metric_names = resolve(algorithm)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "file",
                "score",
                *metric_names,
                "zonal_drop",
                "zonal_axis",
                "zonal_sharp_band",
                "zonal_soft_band",
                "zonal_bands",
                "printed_tiles",
                "tiles",
                "width",
                "height",
                "error",
            ]
        )
        for position, result in enumerate(results, 1):
            height, width = result.shape if result.shape else ("", "")
            zone = result.zonal
            writer.writerow(
                [
                    position,
                    str(result.path),
                    result.score,
                    *(result.per_metric.get(name, "") for name in metric_names),
                    zone.drop if zone else "",
                    zone.axis if zone else "",
                    zone.best if zone else "",
                    zone.worst if zone else "",
                    zone.n_bands if zone else "",
                    result.n_printed,
                    result.n_tiles,
                    width,
                    height,
                    result.error,
                ]
            )
