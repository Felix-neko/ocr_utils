"""CLI: ранжирование сканов папки по качеству фокуса."""

import math
import os
from pathlib import Path

import click
from click.core import ParameterSource

from ocr_utils.defocus_detection.analysis import (
    FileResult,
    analyze_folder,
    sort_by_tile_zonal,
    sort_by_zonal,
    sort_worst_first,
)
from ocr_utils.defocus_detection.image_io import SUPPORTED_SUFFIXES, collect_images, unsupported_files
from ocr_utils.defocus_detection.lines.detect import (
    DEFAULT_MIN_CONF,
    DEFAULT_MODE,
    DEFAULT_TILE_OVERLAP,
    DEFAULT_TILE_SIDE,
    DETECT_MODES,
    DetectCache,
    DetectParams,
    LineDetector,
)
from ocr_utils.defocus_detection.lines.measure import DEFAULT_HEIGHT_CORRIDOR
from ocr_utils.defocus_detection.lines.options import LineOptions
from ocr_utils.defocus_detection.lines.zonal_tiles import DEFAULT_MIN_LINES
from ocr_utils.defocus_detection.lines.zonal_tiles import DEFAULT_TILE_SIDE as DEFAULT_ZONAL_TILES
from ocr_utils.defocus_detection.metrics import (
    ALGORITHMS,
    CHOICES,
    COMBO_MEMBERS,
    COMBO_NAME,
    DEFAULT_ALGORITHM,
    resolve,
)
from ocr_utils.defocus_detection.report import (
    LinkDirError,
    console_table,
    markdown_report,
    write_csv,
    write_link_dir,
    write_retake_list,
    zonal_table,
)
from ocr_utils.defocus_detection.scoring import AGGREGATIONS, DEFAULT_AGGREGATION, DEFAULT_QUANTILE
from ocr_utils.defocus_detection.thresholds import (
    TAG_ORDER,
    TAG_TITLES,
    PRESETS,
    Preset,
    parse_thresholds,
    registry_text,
)
from ocr_utils.defocus_detection.tiles import DEFAULT_TILE_SIZE
from ocr_utils.defocus_detection.zonal import ALL_AXES, AXES

# Сколько ядер не занимать при автоматическом выборе числа процессов.
DEFAULT_RESERVED_CORES = 2


def _default_cache_dir() -> Path:
    """Куда класть кэш детекции строк, если путь не задан явно.

    Returns:
        Путь внутри пользовательского XDG-кэша.
    """
    root = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(root) / "ocr_utils" / "defocus_lines"


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


def _print_presets(ctx: click.Context, param: click.Parameter, value: str | None) -> str | None:
    """Печатает реестр пресетов и выходит, если запрошено ``--tag-preset list``.

    Обработчик нужен «жадный» (``is_eager``): справку по порогам спрашивают до того, как
    выбрана папка, и требовать ради неё существующий INPUT_DIR было бы издевательством.

    Args:
        ctx: Контекст click.
        param: Сам параметр (не используется, требуется интерфейсом click).
        value: Значение ``--tag-preset``.

    Returns:
        Значение без изменений, если это не запрос списка.
    """
    if value == "list" and not ctx.resilient_parsing:
        click.echo(registry_text())
        ctx.exit()
    return value


def _preset_settings(ctx: click.Context, preset: Preset, algorithm: str, aggregation: str, quantile: float):
    """Согласует выбранные настройки с теми, под которые пресет откалиброван.

    Пороги — это числа на шкале КОНКРЕТНОЙ метрики при КОНКРЕТНОЙ агрегации: балл ``dom``
    по самым резким тайлам и он же по самым мягким различаются на треть, и порог 3.00
    во втором случае помечал бы всю папку. Поэтому настройки, которых пользователь явно
    не задавал, берутся из пресета, а явно заданное и расходящееся — это ошибка, а не
    повод молча посчитать не то.

    Args:
        ctx: Контекст click — по нему видно, задан ли параметр пользователем или взят
            из значения по умолчанию.
        preset: Выбранный набор порогов.
        algorithm: Значение ``--algorithm``.
        aggregation: Значение ``--aggregate``.
        quantile: Значение ``--quantile``.

    Returns:
        Кортеж (алгоритм, агрегация, квантиль) с подставленными значениями пресета.

    Raises:
        click.UsageError: Если пользователь явно задал настройку, не совпадающую с той,
            под которую пороги откалиброваны.
    """
    options = {"algorithm": "--algorithm", "aggregation": "--aggregate", "quantile": "--quantile"}
    given = {"algorithm": algorithm, "aggregation": aggregation, "quantile": quantile}
    wanted = {"algorithm": preset.algorithm, "aggregation": preset.aggregation, "quantile": preset.quantile}
    for name, want in wanted.items():
        if ctx.get_parameter_source(name) is ParameterSource.DEFAULT:
            given[name] = want
        elif given[name] != want:
            raise click.UsageError(
                f"пресет {preset.name} откалиброван на {options[name]} {want}, а задано {given[name]}: "
                f"порог на другой шкале ничего не значит. Уберите {options[name]} или задайте свои "
                f"пороги через --tag-thresholds."
            )
    return given["algorithm"], given["aggregation"], given["quantile"]


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
    "--tag-preset",
    default=None,
    is_eager=True,
    callback=_print_presets,
    help="Ставить кадрам АБСОЛЮТНЫЕ теги тяжести по готовому набору порогов "
    f"(есть: {', '.join(PRESETS)}; «list» — напечатать реестр с порогами и происхождением). "
    "В отличие от --worst-percent порог не зависит от содержимого папки: в удачной съёмке "
    "список сам окажется коротким, в неудачной — длинным. Настройки алгоритма и агрегации "
    "берутся из пресета, если не заданы явно.",
)
@click.option(
    "--tag-thresholds",
    default=None,
    help="Свои пороги вместо пресета: heavy=2.9,medium=2.95,light=3.0. Считаются по шкале "
    "выбранного алгоритма и агрегации, «ниже порога» значит «хуже».",
)
@click.option(
    "--tagged-only",
    is_flag=True,
    help="В первый отчёт брать только кадры, превысившие пороги, вместо худших N%%. "
    "Без этого флага пороги не отменяют обычный отчёт, а добавляют к нему отдельный "
    "список — оба взгляда на одну папку сразу.",
)
@click.option(
    "--retake-list",
    type=click.Path(path_type=Path),
    default=None,
    help="Записать плоский список кадров на пересъёмку (.csv — таблица, иначе простой "
    "текст). Пишется всегда, в том числе пустым: пустой файл значит «переснимать нечего», "
    "а отсутствие файла — «прогон не доехал».",
)
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
    default=ALL_AXES,
    show_default=True,
    help="Вдоль чего искать провал резкости: rows — мягкий верх/низ, cols — мягкий "
    "бок, diag и anti — мягкий угол (полосы идут перпендикулярно диагонали), "
    "all — все четыре, в отчёт идёт худшее. Одно направление имеет смысл задавать, "
    "когда известно, что завал в выпуске всегда одинаковый: список тогда чище.",
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
    "--txt-report",
    type=click.Path(path_type=Path),
    default=None,
    help="Записать текстовый отчёт (те же таблицы, что печатаются в консоль) по этому пути, "
    "минуя stdout — полоса прогресса пишется в stderr, но так надёжнее.",
)
@click.option(
    "--md-report", type=click.Path(path_type=Path), default=None, help="Записать markdown-отчёт по этому пути."
)
@click.option(
    "--csv", "csv_path", type=click.Path(path_type=Path), default=None, help="Записать полные результаты в CSV."
)
@click.option(
    "--link-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Разложить попавшие в отчёты кадры симлинками (подпапки overall/ и zonal/), "
    "пронумерованными по рейтингу: папку можно открыть просмотрщиком и листать по порядку. "
    "Нужно потому, что ссылки в markdown работают не во всех просмотрщиках — PyCharm и "
    "Chrome запрещают переход на file:// из превью.",
)
@click.option(
    "--debug-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Складывать сюда JPEG с отладочными наложениями: границы тайлов, рамки найденных "
    "областей строк с их метрикой, балл и число строк в углу каждого тайла, общий балл "
    "в углу кадра. Работает только вместе с --use-surya-lines: по сетке тайлов "
    "показывать нечего.",
)
@click.option(
    "--use-surya-lines",
    is_flag=True,
    help="Мерить фокус ТОЛЬКО по областям строк текста, найденным surya-ocr, а не по "
    "равномерной сетке тайлов. Отвечает на вопрос «читается ли мелкий текст», а не "
    "«резкий ли кадр в среднем»; зональный расфокус при этом ищется ещё и по сетке "
    "3x3 с привязкой строк по центру тяжести. Требует GPU и заметно медленнее.",
)
@click.option(
    "--surya-detect-mode",
    type=click.Choice(DETECT_MODES),
    default=DEFAULT_MODE,
    show_default=True,
    help="Как подавать кадр детектору. tiles — перекрывающимися тайлами в нативном "
    "разрешении; page — кадром целиком. На замеренных полосах page находит втрое меньше "
    "строк и смещён к заголовкам (p90 высоты 46-49 px против 30-31), потому что сеть "
    "ужимает ширину кадра втрое и корпусный набор до неё не доживает.",
)
@click.option(
    "--surya-tile-side",
    type=click.IntRange(min=256),
    default=DEFAULT_TILE_SIDE,
    show_default=True,
    help="Сторона тайла детекции. Мельчить вредно: при 800 px найденных строк стало "
    "МЕНЬШЕ — строка чаще перерезается границей тайла.",
)
@click.option(
    "--surya-tile-overlap",
    type=click.IntRange(min=0),
    default=DEFAULT_TILE_OVERLAP,
    show_default=True,
    help="Перекрытие тайлов детекции: строка на стыке должна целиком уместиться хотя бы "
    "в одном тайле. Дубли снимаются геометрически, по центру строки.",
)
@click.option(
    "--surya-conf",
    type=click.FloatRange(0.0, 1.0),
    default=DEFAULT_MIN_CONF,
    show_default=True,
    help="Порог уверенности блока строки.",
)
@click.option(
    "--surya-batch-size",
    type=click.IntRange(min=0),
    default=0,
    help="Сколько тайлов отдавать сети за раз; 0 — выбор surya (36 для cuda). "
    "Уменьшите, если не хватает видеопамяти.  [default: 0]",
)
@click.option(
    "--zonal-tiles",
    type=click.IntRange(min=2, max=8),
    default=DEFAULT_ZONAL_TILES,
    show_default=True,
    help="Сторона зональной сетки в режиме по строкам: 3 — это 3x3, 4 — 4x4.",
)
@click.option(
    "--line-height-percent",
    nargs=2,
    type=float,
    default=DEFAULT_HEIGHT_CORRIDOR,
    show_default=True,
    help="Коридор перцентилей высоты строки — что считать мелким текстом. Считается "
    "ВНУТРИ каждого тайла, а не по кадру: при трапеции ближний край снят крупнее, и "
    "общий коридор выкосил бы из зональной карты целую сторону.",
)
@click.option(
    "--min-lines-per-tile",
    type=click.IntRange(min=1),
    default=DEFAULT_MIN_LINES,
    show_default=True,
    help="Тайл с меньшим числом измеренных строк не участвует в зональной карте.",
)
@click.option(
    "--detect-cache",
    type=click.Path(path_type=Path),
    default=None,
    help="Дисковый кэш детекции строк (по умолчанию ~/.cache/ocr_utils/defocus_lines). "
    "Нужен не для красоты: детекция — самая дорогая часть прогона, а пороги метрик "
    "подбираются итеративно, и повторный прогон той же папки не должен трогать GPU.",
)
@click.option("--no-detect-cache", is_flag=True, help="Не пользоваться кэшем детекции.")
@click.option("--quiet", "-q", is_flag=True, help="Не показывать полосу прогресса.")
@click.pass_context
def main(
    ctx: click.Context,
    input_dir: Path,
    algorithm: str,
    worst_percent: float | None,
    worst_count: int | None,
    tag_preset: str | None,
    tag_thresholds: str | None,
    tagged_only: bool,
    retake_list: Path | None,
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
    txt_report: Path | None,
    md_report: Path | None,
    csv_path: Path | None,
    link_dir: Path | None,
    debug_dir: Path | None,
    use_surya_lines: bool,
    surya_detect_mode: str,
    surya_tile_side: int,
    surya_tile_overlap: int,
    surya_conf: float,
    surya_batch_size: int,
    zonal_tiles: int,
    line_height_percent: tuple[float, float],
    min_lines_per_tile: int,
    detect_cache: Path | None,
    no_detect_cache: bool,
    quiet: bool,
) -> None:
    """Ранжирует сканы в папке INPUT_DIR по качеству фокуса, худшие — сверху.

    Печатает два отчёта: по общему качеству фокуса и по зональному расфокусу
    (когда мягкая только часть кадра — такой файл может быть неплох в среднем).
    Если отбор не задан, в отчёт идут все файлы с числовой метрикой.
    """
    if tag_preset is not None and tag_thresholds is not None:
        raise click.UsageError("--tag-preset и --tag-thresholds взаимоисключающи: задайте что-то одно.")
    if tag_preset is not None and tag_preset not in PRESETS:
        raise click.UsageError(
            f"неизвестный пресет порогов «{tag_preset}»; есть: {', '.join(PRESETS)}. "
            "Полный список с порогами — --tag-preset list."
        )

    preset: Preset | None = None
    if tag_preset is not None:
        preset = PRESETS[tag_preset]
        algorithm, aggregation, quantile = _preset_settings(ctx, preset, algorithm, aggregation, quantile)
    elif tag_thresholds is not None:
        try:
            levels = parse_thresholds(tag_thresholds)
        except ValueError as error:
            raise click.UsageError(f"--tag-thresholds: {error}") from error
        try:
            preset = Preset(
                name="свои пороги",
                algorithm=algorithm,
                aggregation=aggregation,
                quantile=quantile,
                basis="raw",
                source="заданы вручную через --tag-thresholds",
                **levels,
            )
        except ValueError as error:
            raise click.UsageError(f"--tag-thresholds: {error}") from error

    if tagged_only and preset is None:
        raise click.UsageError("--tagged-only без --tag-preset/--tag-thresholds бессмыслен: тегов не будет.")
    if tagged_only and (worst_percent is not None or worst_count is not None):
        raise click.UsageError("--tagged-only отменяет отбор по доле: --worst-percent/--worst-count тут лишние.")
    if retake_list is not None and preset is None:
        raise click.UsageError("--retake-list требует порогов: задайте --tag-preset или --tag-thresholds.")
    if preset is not None and preset.basis == "norm" and not use_surya_lines:
        raise click.UsageError(
            f"пресет {preset.name} считает порог по нормированному на кегль баллу, "
            "а тот есть только в режиме --use-surya-lines."
        )

    if worst_percent is not None and worst_count is not None:
        raise click.UsageError("--worst-percent и --worst-count взаимоисключающи: задайте что-то одно.")
    if zonal_percent is not None and zonal_count is not None:
        raise click.UsageError("--zonal-percent и --zonal-count взаимоисключающи: задайте что-то одно.")
    if no_zonal and (zonal_percent is not None or zonal_count is not None):
        raise click.UsageError("--no-zonal несовместим с --zonal-percent/--zonal-count.")
    if debug_dir is not None and not use_surya_lines:
        raise click.UsageError("--debug-dir работает только вместе с --use-surya-lines.")
    if use_surya_lines:
        # Спектральным метрикам нужен кусок кадра приличного размера, а кусок строки —
        # это десятки пикселей. Молча вернуть NaN по всей папке было бы хуже отказа.
        unsupported = [name for name in resolve(algorithm) if not ALGORITHMS[name].supports_regions]
        if unsupported:
            raise click.UsageError(
                f"--use-surya-lines несовместим с алгоритмом {', '.join(unsupported)}: "
                "он меряет по спектру и требует тайла, а не куска строки."
            )

    files = collect_images(input_dir, recursive=recursive)
    if not files:
        raise click.ClickException(f"В {input_dir} не найдено поддерживаемых изображений.")

    # Про пропущенные файлы говорим ВСЕГДА и до счёта. Обычно это чужой мусор в папке,
    # но ровно так же выглядит кадр с опечаткой в расширении, и отличить их может только
    # человек — а для этого он должен о них узнать.
    skipped = unsupported_files(input_dir, recursive=recursive)
    if skipped:
        shown = ", ".join(p.name for p in skipped[:5])
        rest = f" и ещё {len(skipped) - 5}" if len(skipped) > 5 else ""
        click.echo(
            f"Пропущено файлов: {len(skipped)} — не распознаны как изображения ({shown}{rest}). "
            f"Поддерживаются: {', '.join(sorted(SUPPORTED_SUFFIXES))}.",
            err=True,
        )

    detector, line_options = None, None
    if use_surya_lines:
        cache = None if no_detect_cache else DetectCache(detect_cache or _default_cache_dir())
        detector = LineDetector(
            DetectParams(
                mode=surya_detect_mode,
                tile_side=surya_tile_side,
                tile_overlap=surya_tile_overlap,
                min_conf=surya_conf,
                batch_size=surya_batch_size or None,
            ),
            cache=cache,
        )
        line_options = LineOptions(
            n_tiles=zonal_tiles,
            height_corridor=tuple(line_height_percent),
            min_lines=min_lines_per_tile,
            debug_dir=debug_dir,
        )

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
        detector=detector,
        line_options=line_options,
    )
    results = sort_worst_first(results)

    tagged: list[FileResult] | None = None
    if preset is not None:
        for result in results:
            value = result.score if preset.basis == "raw" else result.score_norm
            result.tag = preset.tag(value)
        # results уже отсортирован от худшего, поэтому и список тегированных выходит
        # в том же порядке — отдельно сортировать нечего.
        tagged = [result for result in results if result.tag]

    if tagged_only:
        selected, shown = tagged, f"кадры выше порога ({len(tagged)} из {len(results)})"
    else:
        limit, shown = _select(len(results), worst_count, worst_percent)
        selected = results[:limit]

    tagging = preset is not None
    text_blocks = [
        "== 1. ОБЩЕЕ КАЧЕСТВО ФОКУСА " + "=" * 40,
        console_table(selected, algorithm, total=len(results), tagging=tagging),
    ]
    click.echo(text_blocks[0])
    click.echo(text_blocks[1])

    zonal_selected = None
    zonal_shown = ""
    if not no_zonal:
        if use_surya_lines:
            # В режиме по строкам главная зональная оценка — тайловая: она построена на
            # том же тексте, что и общий балл. Направленная остаётся в отчёте колонкой
            # рядом, но порядок списка задаёт не она.
            ranked = sort_by_tile_zonal([r for r in results if r.tile_zonal is not None])
        else:
            ranked = sort_by_zonal([r for r in results if r.zonal is not None])
        zonal_limit, zonal_shown = _select(len(ranked), zonal_count, zonal_percent)
        zonal_selected = ranked[:zonal_limit]
        text_blocks.append("\n== 2. ЗОНАЛЬНЫЙ РАСФОКУС (мягкая часть кадра) " + "=" * 22)
        text_blocks.append(zonal_table(zonal_selected, total=len(ranked)))
        click.echo(text_blocks[-2])
        click.echo(text_blocks[-1])
        skipped = len(results) - len(ranked)
        if skipped:
            click.echo(f"Без зональной оценки (мало текста): {skipped} файлов.", err=True)

    if tagging and not tagged_only:
        counts = ", ".join(f"{TAG_TITLES[tag]} — {sum(1 for r in tagged if r.tag == tag)}" for tag in TAG_ORDER)
        share = len(tagged) / len(results) * 100
        text_blocks.append("\n== 3. ПРЕВЫСИЛИ АБСОЛЮТНЫЙ ПОРОГ " + "=" * 35)
        text_blocks.append(
            f"Пороги: {preset.name} — {preset.describe()}.\n"
            f"Кадров выше порога: {len(tagged)} из {len(results)} ({share:.1f} %); {counts}.\n"
        )
        text_blocks.append(
            console_table(tagged, algorithm, total=len(results), tagging=True)
            if tagged
            else "Ни один кадр порогов не превысил — переснимать нечего."
        )
        for block in text_blocks[-3:]:
            click.echo(block)

    failed = [r for r in results if r.error]
    if failed:
        click.echo(f"\nНе прочитано файлов: {len(failed)} (перечислены в начале первой таблицы).", err=True)

    if txt_report is not None:
        txt_report.parent.mkdir(parents=True, exist_ok=True)
        # Пишем те же блоки, что напечатаны выше, но напрямую в файл — минуя stdout,
        # который при параллельном прогоне делят с полосой прогресса (та пишется в stderr,
        # но перехват стримов в обёртках всё равно ненадёжен).
        txt_report.write_text("\n".join(text_blocks) + "\n", encoding="utf-8")
        click.echo(f"Текстовый отчёт: {txt_report}", err=True)

    if md_report is not None:
        md_report.parent.mkdir(parents=True, exist_ok=True)
        md_report.write_text(
            markdown_report(
                selected,
                zonal_selected,
                algorithm,
                len(results),
                input_dir.resolve(),
                shown,
                zonal_shown,
                aggregation,
                preset=preset,
                tagged=None if tagged_only else tagged,
            ),
            encoding="utf-8",
        )
        click.echo(f"Markdown-отчёт: {md_report}", err=True)

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        # В CSV всегда пишем все файлы: он нужен для калибровки порогов, а не для чтения глазами.
        write_csv(csv_path, results, algorithm)
        click.echo(f"CSV: {csv_path}", err=True)

    if retake_list is not None:
        write_retake_list(retake_list, tagged)
        click.echo(f"Список на пересъёмку: {retake_list} — кадров в нём: {len(tagged)}", err=True)

    if link_dir is not None:
        try:
            root, made = write_link_dir(link_dir, selected, zonal_selected, tagged=None if tagged_only else tagged)
        except LinkDirError as error:
            raise click.ClickException(str(error)) from error
        folders = ["overall/"] + ([] if no_zonal else ["zonal/"]) + ([f"{t}/" for t in TAG_ORDER] if tagging else [])
        click.echo(f"Симлинки: {root}/ ({made} шт., подпапки {', '.join(folders)})", err=True)


if __name__ == "__main__":
    main()
