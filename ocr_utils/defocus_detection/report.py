"""Печать отчёта: консольная таблица, markdown и CSV."""

import csv
import math
from pathlib import Path
from urllib.parse import quote

from ocr_utils.defocus_detection.analysis import FileResult
from ocr_utils.defocus_detection.metrics import ALGORITHMS, COMBO_NAME, resolve
from ocr_utils.defocus_detection.zonal import DIRECTIONS


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


def by_lines(results: list[FileResult]) -> bool:
    """Считался ли прогон по областям строк.

    Определяется по самим результатам, а не по флагу: так отчёту не нужно ничего знать
    о том, как его вызвали.

    Args:
        results: Результаты анализа.

    Returns:
        True, если хотя бы у одного файла есть измеренные строки.
    """
    return any(r.n_lines_detected for r in results)


def _readability(result: FileResult) -> str:
    """Ячейка колонки читаемости: размытие в долях высоты строки.

    Показывается именно σ/высота, а не обратная величина: «0.05» читается как «край
    размазан на двадцатую часть буквы» и сравнивается с порогом глазом, тогда как
    «20» требует помнить, что это такое.

    Args:
        result: Результат анализа файла.

    Returns:
        Строка для таблицы.
    """
    if not math.isfinite(result.score_norm) or result.score_norm <= 0:
        return "—"
    return f"{1.0 / result.score_norm:.4f}"


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
    lines_mode = by_lines(results)

    headers = ["#", f"балл ({spec_unit})", *(title for _, title in extras)]
    if lines_mode:
        headers += ["σ/высота", "строк"]
    headers.append("файл")

    rows: list[list[str]] = []
    for position, result in enumerate(results, 1):
        name = result.path.name if not result.error else f"{result.path.name}  [{result.error}]"
        row = [str(position), _score_cell(result.score), *_row_values(result, algorithm)]
        if lines_mode:
            row += [_readability(result), str(result.n_lines)]
        row.append(name)
        rows.append(row)

    return _render(
        headers, rows, f"Показано {len(results)} из {total} файлов; сортировка — от худшего фокуса к лучшему."
    )


def zonal_table(results: list[FileResult], total: int) -> str:
    """Строит текстовую таблицу отчёта по зональному расфокусу.

    В режиме по строкам печатаются ОБЕ оценки: тайловая (по сетке 3x3, куда строки
    привязаны центром тяжести) и старая направленная (профили по четырём направлениям).
    Они считаются независимо и по-разному ошибаются, так что видеть их рядом полезно —
    расхождение колонок само по себе повод посмотреть кадр глазами.

    Args:
        results: Отсортированные по перепаду и урезанные результаты.
        total: Сколько файлов имеет зональную оценку.

    Returns:
        Готовый к печати многострочный текст.
    """
    lines_mode = by_lines(results)
    headers = ["#"]
    if lines_mode:
        headers += ["перепад (тайлы)", "где мягче (тайлы)"]
    headers += ["перепад", "σ резч.", "σ мягч.", "где мягче", "файл"]

    rows: list[list[str]] = []
    for position, result in enumerate(results, 1):
        row = [str(position)]
        if lines_mode:
            tile = result.tile_zonal
            row += ["—", "нет строк"] if tile is None else [f"+{tile.drop * 100:.0f}%", tile.where()]
        zone = result.zonal
        if zone is None:
            row += ["—", "—", "—", "нет текста"]
        else:
            row += [
                f"+{zone.drop * 100:.0f}%",
                f"{zone.profile[zone.best]:.3f}",
                f"{zone.profile[zone.worst]:.3f}",
                zone.where(),
            ]
        row.append(result.path.name)
        rows.append(row)
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

    Схема ``file://`` обязательна. Без неё в ссылке остаётся голый абсолютный путь
    (``/mnt/dump3/…``), а просмотрщик markdown трактует ведущий слэш как корень ТЕКУЩЕГО
    документа, а не файловой системы: ссылка молча ведёт в никуда. Путь приводится к
    абсолютному ещё и потому, что отчёт кладётся в корень репозитория, а снимки лежат на
    другом разделе — относительный путь тут не построить.

    Кодирование оставлено на ``quote``: у него safe="/" по умолчанию, то есть разделители
    пути сохраняются, а пробелы и кириллица в именах папок уезжают в проценты — ровно то,
    что нужно для file-URI.

    Args:
        result: Результат анализа файла.

    Returns:
        Markdown-ссылка с абсолютным URL-кодированным путём и схемой file://.
    """
    link = f"[{result.path.name}](file://{quote(str(result.path.resolve()))})"
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
    lines_mode = by_lines(overall)

    how = "по областям строк текста (surya-ocr)" if lines_mode else f"агрегация по тайлам: **{aggregation}**"
    lines = [
        "# Подозрение на расфокус",
        "",
        f"Папка: `{folder}`",
        "",
        f"Алгоритм: **{algorithm}**, {how}. Проанализировано файлов: {total}.",
        "",
        "## 1. Общее качество фокуса",
        "",
        f"В таблице: {shown}. Сортировка — от худшего фокуса к лучшему.",
        "",
    ]
    if lines_mode:
        lines += [
            "Колонка **σ/высота** — размытие в долях высоты строки: именно она отвечает на вопрос",
            "«читается ли мелкий текст», тогда как сырая σ говорит только о том, попала ли оптика",
            "в фокус. Одно и то же размытие не мешает заголовку и убивает петит.",
            "",
        ]

    head = ["#", f"балл ({spec_unit})", *(t for _, t in extras)]
    align = ["--:", "--:", *("--:" for _ in extras)]
    if lines_mode:
        head += ["σ/высота", "строк"]
        align += ["--:", "--:"]
    head.append("файл")
    align.append("---")
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join(align) + "|")

    for position, result in enumerate(overall, 1):
        cells = [str(position), _score_cell(result.score), *_row_values(result, algorithm)]
        if lines_mode:
            cells += [_readability(result), str(result.n_lines)]
        cells.append(_link(result))
        lines.append("| " + " | ".join(cells) + " |")

    if zonal is not None:
        lines += [
            "",
            "## 2. Зональный расфокус",
            "",
            f"В таблице: {zonal_shown}. Сортировка — по убыванию перепада резкости внутри кадра.",
            "",
            "Перепад — на сколько процентов шире штрих в самой мягкой части кадра по сравнению",
            "с самой резкой. Кадр может быть хорош в среднем и всё равно попасть сюда: значит,",
            "поплыла его часть.",
            "",
        ]
        if lines_mode:
            lines += [
                "Колонок с перепадом две, и считаются они независимо: **тайлы** — по сетке 3x3, куда",
                "строки привязаны центром тяжести; **полосы** — старый профиль по четырём направлениям",
                "на равномерной сетке. Расхождение колонок само по себе повод посмотреть кадр глазами.",
                "",
                "| # | перепад (тайлы) | где мягче (тайлы) | перепад (полосы) | σ резч. | σ мягч. | где мягче | файл |",
                "|--:|--:|---|--:|--:|--:|---|---|",
            ]
        else:
            lines += ["| # | перепад | σ резч. | σ мягч. | где мягче | файл |", "|--:|--:|--:|--:|---|---|"]
        for position, result in enumerate(zonal, 1):
            cells = [str(position)]
            if lines_mode:
                tile = result.tile_zonal
                cells += ["—", "нет строк"] if tile is None else [f"+{tile.drop * 100:.0f}%", tile.where()]
            zone = result.zonal
            if zone is None:
                cells += ["—", "—", "—", "нет текста"]
            else:
                cells += [
                    f"+{zone.drop * 100:.0f}%",
                    f"{zone.profile[zone.best]:.3f}",
                    f"{zone.profile[zone.worst]:.3f}",
                    zone.where(),
                ]
            cells.append(_link(result))
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


class LinkDirError(Exception):
    """В целевой папке для симлинков лежит что-то, кроме симлинков."""


def _link_metric(result: FileResult, zonal: bool) -> str:
    """Значение метрики для имени симлинка — ровно то же, что напечатано в отчёте.

    Args:
        result: Результат анализа файла.
        zonal: True — брать перепад зонального отчёта, False — общий балл.

    Returns:
        Короткая строка, годная в имя файла.
    """
    if zonal:
        # В режиме по строкам список отсортирован по ТАЙЛОВОМУ перепаду, поэтому и в
        # имени должен стоять он же: иначе номер по порядку и число в имени разошлись бы.
        zone = result.tile_zonal or result.zonal
        return f"+{zone.drop * 100:.0f}%" if zone else "нет"
    cell = _score_cell(result.score)
    # "—" из _score_cell в имени файла смотрится дико и плохо ищется грепом.
    return "нет" if cell == "—" else cell


def write_link_dir(root: Path, overall: list[FileResult], zonal: list[FileResult] | None) -> tuple[Path, int]:
    """Раскладывает попавшие в отчёт кадры симлинками, пронумерованными по рейтингу.

    ЗАЧЕМ ЭТО ВООБЩЕ ЕСТЬ. Кликабельные ссылки в markdown-отчёте работают не везде:
    PyCharm и Chrome рендерят превью в Chromium, а тот запрещает переход из документа
    на ``file://`` — тихо, без сообщения. Обойти это со стороны разметки нельзя никак,
    поэтому список выдаётся ещё и в виде, который не зависит от просмотрщика: папку с
    симлинками достаточно открыть в XnView (или любом просмотрщике) и листать стрелками.

    Имя симлинка — ``<позиция>_<метрика>_<исходное имя>``, например
    ``03_0.7219_1220_1.RAF`` или ``01_+41%_1240_1.RAF``. Позиция идёт первой и с
    ведущими нулями, чтобы алфавитный порядок в просмотрщике совпал с порядком отчёта
    «от худшего к лучшему»; ширина берётся по длине списка, чтобы не городить лишние
    нули на коротких выборках. Метрика печатается той же функцией, что и в таблице
    отчёта, — число в имени файла и число в отчёте всегда совпадают.

    Args:
        root: Куда складывать; создаётся, если нет. Внутри — подпапки ``overall``
            и ``zonal`` по числу отчётов: списки почти не пересекаются.
        overall: Отобранные результаты первого отчёта, в порядке отчёта.
        zonal: То же для второго отчёта либо None, если он не считался.

    Returns:
        Кортеж (путь к папке, сколько симлинков создано).

    Raises:
        LinkDirError: Если в подпапке уже лежит не симлинк. Обычные файлы не трогаем
            никогда: папку могли указать по ошибке, и удалять оригиналы недопустимо.
    """
    made = 0
    for name, selected in (("overall", overall), ("zonal", zonal)):
        if selected is None:
            continue
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)

        # Чистим только свои прошлые симлинки, чтобы повторный прогон не копил мусор.
        for existing in directory.iterdir():
            if not existing.is_symlink():
                raise LinkDirError(f"в {directory} лежит не симлинк ({existing.name}) — папка не очищена")
            existing.unlink()

        # Минимум два знака: с одним знаком список из десяти и больше кадров
        # рассыпается в просмотрщике (1, 10, 2, 3…).
        width = max(2, len(str(len(selected))))
        for position, result in enumerate(selected, 1):
            if result.error:
                continue
            target = result.path.resolve()
            metric = _link_metric(result, zonal=name == "zonal")
            (directory / f"{position:0{width}d}_{metric}_{target.name}").symlink_to(target)
            made += 1
    return root, made


def write_csv(path: Path, results: list[FileResult], algorithm: str) -> None:
    """Сохраняет полные результаты в CSV (для собственного анализа и калибровки).

    Args:
        path: Куда записать файл.
        results: Отсортированные результаты.
        algorithm: Имя выбранного алгоритма.
    """
    metric_names = resolve(algorithm)
    lines_mode = by_lines(results)
    # Сторона зональной сетки берётся из первого же посчитанного кадра: в пределах одного
    # прогона она общая, а разворачивать карту в колонки надо заранее, до первой строки.
    grid_side = next((r.tile_zonal.n for r in results if r.tile_zonal is not None), 0)
    # Карта разворачивается в отдельные колонки (а не в одну строку с числами), чтобы её
    # можно было прямо в таблице отсортировать и построить по ней сводку по выпуску.
    tile_columns = [f"tile_sharp_r{iy + 1}c{ix + 1}" for iy in range(grid_side) for ix in range(grid_side)]

    line_headers = ["score_norm", "lines_measured", "lines_detected", "chunks", "tile_drop", "tile_worst", "tile_zone"]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "file",
                "score",
                *metric_names,
                *(line_headers if lines_mode else []),
                *tile_columns,
                "zonal_drop",
                "zonal_axis",
                # Перепад по каждому направлению отдельно: по нему видно, завал это
                # плоскости съёмки (одно направление стабильно хуже прочих во всём
                # выпуске) или разовый шум вёрстки, и по нему же подбираются пороги.
                *(f"zonal_drop_{name}" for name in DIRECTIONS),
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
            tile = result.tile_zonal
            line_values = [
                result.score_norm,
                result.n_lines,
                result.n_lines_detected,
                result.n_chunks,
                tile.drop if tile else "",
                f"r{tile.worst[0] + 1}c{tile.worst[1] + 1}" if tile else "",
                tile.where() if tile else "",
            ]
            tile_values = (
                list(tile.sharpness.reshape(-1))
                if tile is not None and tile.n == grid_side
                else [""] * len(tile_columns)
            )
            writer.writerow(
                [
                    position,
                    str(result.path),
                    result.score,
                    *(result.per_metric.get(name, "") for name in metric_names),
                    *(line_values if lines_mode else []),
                    *tile_values,
                    zone.drop if zone else "",
                    zone.axis if zone else "",
                    *((zone.drops.get(name, "") if zone else "") for name in DIRECTIONS),
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
