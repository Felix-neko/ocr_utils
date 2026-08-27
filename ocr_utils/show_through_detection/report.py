"""Отчёты о просвечивании: консольные таблицы, markdown, CSV, папка симлинков.

Отчётов два, потому что вопросов у прогона тоже два. Первый — «какие ПОЛОСЫ плохи»:
дефект принадлежит странице, и знать, левая она или правая, нужно, чтобы понимать,
что именно смотреть в бумажном экземпляре. Второй — «какие КАДРЫ пересканировать»:
переснимают всё равно разворот целиком, и список для работы должен быть по файлам.
"""

import csv
import math
from pathlib import Path
from urllib.parse import quote

from ocr_utils.show_through_detection.analysis import FileResult, HalfResult
from ocr_utils.show_through_detection.metrics import ALGORITHMS, COMBO_NAME, resolve


def _cell(value: float) -> str:
    """Форматирует число, подбирая знаки под порядок величины.

    Args:
        value: Числовое значение.

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


def _metric_columns(algorithm: str) -> tuple[str, ...]:
    """Какие сырые баллы показывать колонками.

    Args:
        algorithm: Имя выбранного алгоритма.

    Returns:
        Кортеж имён метрик.
    """
    return resolve(algorithm)


def _mark(half: HalfResult, threshold: float) -> str:
    """Пометка полосы в отчёте: превышен ли порог и с какой оговоркой.

    Args:
        half: Результат по полосе.
        threshold: Порог по ``severity``.

    Returns:
        Короткая строка для колонки «вердикт».
    """
    if half.problem:
        return half.problem
    if not math.isfinite(half.severity):
        return "не измерена"
    verdict = "ПЕРЕСКАН" if half.severity >= threshold else "ok"
    return f"{verdict} ({half.note})" if half.note else verdict


def _render(headers: list[str], rows: list[list[str]], footer: str) -> str:
    """Собирает текстовую таблицу с выравниванием по ширине колонок.

    Args:
        headers: Заголовки колонок.
        rows: Строки таблицы.
        footer: Строка-подпись под таблицей.

    Returns:
        Готовый к печати многострочный текст.
    """
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    last = len(headers) - 1

    def line(cells: list[str]) -> str:
        parts = [c.ljust(widths[i]) if i == last else c.rjust(widths[i]) for i, c in enumerate(cells)]
        return "  ".join(parts).rstrip()

    out = [line(headers), "-" * min(160, sum(widths) + 2 * last)]
    out += [line(r) for r in rows]
    out += ["", footer]
    return "\n".join(out)


def halves_table(halves: list[HalfResult], algorithm: str, threshold: float, total: int) -> str:
    """Таблица по полосам: сильнее всего просвечивающие — сверху.

    Args:
        halves: Уже отсортированные и урезанные результаты по полосам.
        algorithm: Имя выбранного алгоритма.
        threshold: Порог по ``severity``.
        total: Сколько полос измерено всего.

    Returns:
        Готовый к печати текст.
    """
    metrics = _metric_columns(algorithm)
    headers = ["#", "×порог", *metrics, "вердикт", "полоса"]
    rows = [
        [
            str(position),
            _cell(half.severity),
            *(_cell(half.per_metric.get(name, float("nan"))) for name in metrics),
            _mark(half, threshold),
            half.name,
        ]
        for position, half in enumerate(halves, 1)
    ]
    return _render(
        headers, rows, f"Показано {len(halves)} из {total} полос; сортировка — от сильного просвета к слабому."
    )


def files_table(results: list[FileResult], threshold: float, total: int) -> str:
    """Таблица по кадрам: список на пересканирование.

    Args:
        results: Отсортированные и урезанные результаты по кадрам.
        threshold: Порог по ``severity``.
        total: Сколько кадров проанализировано всего.

    Returns:
        Готовый к печати текст.
    """
    headers = ["#", "×порог", "полосы", "кадр"]
    rows = []
    for position, result in enumerate(results, 1):
        sides = result.worst_sides(threshold) or "—"
        name = result.path.name if not result.error else f"{result.path.name}  [{result.error}]"
        rows.append([str(position), _cell(result.severity), sides, name])
    return _render(
        headers, rows, f"Показано {len(results)} из {total} кадров; порог — {threshold:g}× от калибровочного."
    )


def _link(path: Path, title: str) -> str:
    """Кликабельная ссылка на файл для markdown.

    Схема ``file://`` обязательна, иначе относительная ссылка ведёт в никуда;
    ``quote`` с ``safe="/"`` нужен из-за пробелов и кириллицы в путях пака.

    Args:
        path: Путь к файлу.
        title: Текст ссылки.

    Returns:
        Разметка ссылки.
    """
    return f"[{title}](file://{quote(str(path.resolve()))})"


def markdown_report(
    halves: list[HalfResult],
    files: list[FileResult],
    algorithm: str,
    threshold: float,
    total_halves: int,
    total_files: int,
    folder: Path,
    shown: str,
) -> str:
    """Строит markdown-отчёт из двух таблиц с кликабельными ссылками.

    Args:
        halves: Отобранные полосы, в порядке отчёта.
        files: Отобранные кадры на пересканирование, в порядке отчёта.
        algorithm: Имя выбранного алгоритма.
        threshold: Порог по ``severity``.
        total_halves: Сколько полос измерено всего.
        total_files: Сколько кадров проанализировано всего.
        folder: Исходная папка.
        shown: Человекочитаемое описание отбора для первой таблицы.

    Returns:
        Текст markdown-документа.
    """
    metrics = _metric_columns(algorithm)
    unit = "средний ранг" if algorithm == COMBO_NAME else ALGORITHMS[algorithm].unit
    lines = [
        "# Подозрение на просвечивающую бумагу",
        "",
        f"Папка: `{folder}`",
        "",
        f"Метрика: **{algorithm}** ({unit}). Полос измерено: {total_halves}, кадров: {total_files}.",
        "",
        "Колонка **×порог** — во сколько раз балл превышает калибровочный порог метрики.",
        "Она сравнима между основной метрикой и запасной (у них разные шкалы) и читается",
        "напрямую: 1.0 — ровно порог, 2.0 — вдвое сильнее того, что решено считать браком.",
        "",
        "## 1. Полосы",
        "",
        f"В таблице: {shown}. Сортировка — от сильного просвета к слабому.",
        "",
        "| # | ×порог | " + " | ".join(metrics) + " | вердикт | полоса |",
        "|--:|--:|" + "".join("--:|" for _ in metrics) + "---|---|",
    ]
    for position, half in enumerate(halves, 1):
        cells = [
            str(position),
            _cell(half.severity),
            *(_cell(half.per_metric.get(name, float("nan"))) for name in metrics),
            _mark(half, threshold),
            _link(half.path, half.name),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 2. Кадры на пересканирование",
        "",
        f"Кадров выше порога: {len(files)} из {total_files}. Пересканировать надо разворот",
        "целиком, но в колонке «полосы» указано, какая из страниц собственно просвечивает.",
        "",
        "| # | ×порог | полосы | кадр |",
        "|--:|--:|---|---|",
    ]
    for position, result in enumerate(files, 1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(position),
                    _cell(result.severity),
                    result.worst_sides(threshold) or "—",
                    _link(result.path, result.path.name),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


class LinkDirError(Exception):
    """В целевой папке для симлинков лежит что-то, кроме симлинков."""


def _origin(path: Path, base: Path | None) -> str:
    """Приставка имени симлинка, по которой видно, откуда кадр.

    Имя файла внутри пака само по себе о происхождении не говорит: ``09_0005.jpg``
    — это девятый номер какого года? Поэтому в имя вносятся папки, через которые
    кадр лежит относительно корня прогона: год и выпуск.

    Args:
        path: Путь к исходному кадру.
        base: Корень прогона (папка, переданная в CLI); None — приставки не будет.

    Returns:
        Строка вида ``1938_09_`` либо пустая, если кадр лежит прямо в корне прогона
        (плоская папка — тогда приставка ничего не добавила бы).
    """
    if base is None:
        return ""
    try:
        parents = path.resolve().relative_to(base.resolve()).parts[:-1]
    except ValueError:
        # Кадр вне корня прогона — такого быть не должно, но имя портить незачем.
        return ""
    return "".join(f"{part}_" for part in parents)


def write_link_dir(
    root: Path, results: list[FileResult], base: Path | None = None, threshold: float = 1.0
) -> tuple[Path, int]:
    """Раскладывает попавшие в отчёт кадры симлинками, пронумерованными по рейтингу.

    ЗАЧЕМ ЭТО ЕСТЬ. Кликабельные ссылки в markdown работают не везде: PyCharm и Chrome
    рендерят превью в Chromium, а тот молча запрещает переход на ``file://``. Поэтому
    список выдаётся ещё и в виде, не зависящем от просмотрщика: папку с симлинками
    достаточно открыть в любом просмотрщике и листать стрелками.

    Имя симлинка — ``<позиция>_<×порог>_<стороны>_<год>_<выпуск>_<исходное имя>``:
    позиция первой и с ведущими нулями, чтобы алфавитный порядок совпал с порядком
    отчёта, а год с выпуском — потому что имя файла внутри пака о происхождении молчит
    (``09_0005.jpg`` — девятый номер какого года?). Год и выпуск берутся из папок, через
    которые кадр лежит относительно корня прогона; на плоской папке приставки не будет.

    Args:
        root: Куда складывать; создаётся, если нет.
        results: Отобранные кадры, в порядке отчёта.
        base: Корень прогона — от него отсчитываются год и выпуск в имени.
        threshold: Порог по ``severity``; по нему определяется, какие стороны назвать
            в имени симлинка.

    Returns:
        Кортеж (путь к папке, сколько симлинков создано).

    Raises:
        LinkDirError: Если в папке уже лежит не симлинк. Обычные файлы не трогаем
            никогда: папку могли указать по ошибке, и удалять оригиналы недопустимо.
    """
    root.mkdir(parents=True, exist_ok=True)
    for existing in root.iterdir():
        if not existing.is_symlink():
            raise LinkDirError(f"в {root} лежит не симлинк ({existing.name}) — папка не очищена")
        existing.unlink()

    # Минимум два знака: с одним список из десяти и больше кадров рассыпается
    # в просмотрщике (1, 10, 2, 3…).
    width = max(2, len(str(len(results))))
    made = 0
    for position, result in enumerate(results, 1):
        if result.error:
            continue
        target = result.path.resolve()
        cell = _cell(result.severity)
        sides = result.worst_sides(threshold).replace(", ", "") or "нет"
        name = f"{position:0{width}d}_{cell}_{sides}_{_origin(result.path, base)}{target.name}"
        (root / name).symlink_to(target)
        made += 1
    return root, made


def write_csv(path: Path, halves: list[HalfResult], results: list[FileResult], algorithm: str) -> None:
    """Сохраняет полные результаты по полосам в CSV — для калибровки порогов.

    Пишутся ВСЕ полосы, а не только попавшие в отчёт, и ВСЕ посчитанные метрики,
    а не только запрошенная: файл нужен, чтобы смотреть распределение баллов и двигать
    порог. Запасная метрика считается всё равно, и не выложить её в калибровочный файл
    было бы обидно — именно по ней видно, из-за чего балл полосы такой, какой он есть.

    Args:
        path: Куда записать файл.
        halves: Отсортированные результаты по полосам.
        results: Результаты по кадрам (для геометрии кадра и ошибок чтения).
        algorithm: Имя выбранного алгоритма.
    """
    # Порядок колонок: сначала запрошенные метрики, затем всё остальное, что посчиталось.
    requested = list(resolve(algorithm))
    extra = sorted({name for half in halves for name in half.per_metric} - set(requested))
    metric_names = [*requested, *extra]
    geometry = {r.path: r for r in results}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "file",
                "side",
                "severity",
                "score",
                "metric",
                *metric_names,
                "note",
                "problem",
                "gutter",
                "gutter_confident",
                "width",
                "height",
            ]
        )
        for position, half in enumerate(halves, 1):
            frame = geometry.get(half.path)
            height, width = frame.shape if frame and frame.shape else ("", "")
            writer.writerow(
                [
                    position,
                    str(half.path),
                    half.side,
                    half.severity,
                    half.score,
                    half.metric,
                    *(half.per_metric.get(name, "") for name in metric_names),
                    half.note,
                    half.problem,
                    frame.gutter if frame else "",
                    frame.gutter_confident if frame else "",
                    width,
                    height,
                ]
            )
        for frame in results:
            if frame.error:
                writer.writerow(
                    ["", str(frame.path), "", "", "", "", *("" for _ in metric_names), "", frame.error, "", "", "", ""]
                )
