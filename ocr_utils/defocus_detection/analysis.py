"""Прогон метрик по файлам папки: чтение, тайлы, баллы, сводный ранг."""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import cv2

from ocr_utils.defocus_detection.image_io import read_gray
from ocr_utils.defocus_detection.lines.measure import LineMeasurements, measure_lines
from ocr_utils.defocus_detection.lines.options import LineOptions
from ocr_utils.defocus_detection.lines.regions import LineRegion
from ocr_utils.defocus_detection.lines.zonal_tiles import TileZonalResult, tile_zonal
from ocr_utils.defocus_detection.metrics import ALGORITHMS, COMBO_NAME, resolve
from ocr_utils.defocus_detection.scoring import DEFAULT_AGGREGATION, DEFAULT_QUANTILE, aggregate, rank_combine
from ocr_utils.defocus_detection.tiles import DEFAULT_TILE_SIZE, detail_rms_map, make_grid, printed_mask
from ocr_utils.defocus_detection.zonal import ZonalResult, zonal_defocus


# На сколько понижается приоритет процессов-воркеров. Одного лишь ограничения их числа
# для отзывчивости интерфейса мало: полтора десятка равноприоритетных счётчиков всё равно
# конкурируют с рабочим столом за планировщик. С nice рабочий стол вытесняет их сразу,
# а «зарезервированные» ядра становятся зарезервированными на деле, а не на бумаге.
WORKER_NICE = 10

# Переменные, которыми numpy/OpenCV/LibRaw ограничивают свои внутренние пулы потоков.
# Без них каждый воркер вправе развернуть по потоку на ядро, и полтора десятка процессов
# устраивают многократную перезапись — машина занята вся, считая при этом не быстрее.
THREAD_LIMIT_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _init_worker() -> None:
    """Готовит процесс-воркер: один поток счёта и пониженный приоритет."""
    cv2.setNumThreads(1)
    try:
        os.nice(WORKER_NICE)
    except OSError:
        # Понижать приоритет можно всегда, но на всякий случай не падаем из-за политики.
        pass


@dataclass
class FileResult:
    """Результат анализа одного файла.

    Attributes:
        path: Путь к файлу.
        score: Итоговый балл резкости (больше = резче); NaN, если оценить не удалось.
        per_metric: Баллы отдельных метрик (в режиме ``combo`` их несколько).
        shape: Размер прочитанного кадра (height, width) или None.
        n_printed: Сколько тайлов признано печатными.
        n_tiles: Всего тайлов в сетке.
        zonal: Оценка зонального расфокуса или None, если её не считали
            (выключена) либо посчитать не удалось (кадр без текста).
        score_norm: Балл ЧИТАЕМОСТИ — резкость, нормированная на высоту строки
            («сколько ширин размытого края укладывается в букву»). Считается только в
            режиме по строкам и только для метрик с размерностью длины; иначе NaN.
        n_lines: Сколько строк удалось измерить (режим по строкам).
        n_lines_detected: Сколько строк нашёл детектор до отбора по кеглю.
        n_chunks: На сколько кусков эти строки были нарезаны.
        tile_zonal: Зональная карта по сетке тайлов или None.
        tag: Тег тяжести расфокуса из ``thresholds`` (пустая строка — порогов не
            превысил либо тегирование не включено). Проставляется после прогона, в
            CLI: пороги — свойство выбранного пресета, а не отдельного файла.
        error: Текст ошибки, если файл не прочитался.
    """

    path: Path
    score: float = float("nan")
    per_metric: dict[str, float] = field(default_factory=dict)
    shape: tuple[int, int] | None = None
    n_printed: int = 0
    n_tiles: int = 0
    zonal: ZonalResult | None = None
    score_norm: float = float("nan")
    n_lines: int = 0
    n_lines_detected: int = 0
    n_chunks: int = 0
    tile_zonal: TileZonalResult | None = None
    tag: str = ""
    error: str = ""


def analyze_file(
    path: Path,
    metric_names: tuple[str, ...],
    tile_size: int,
    aggregation: str,
    quantile: float,
    zonal_axis: str | None = "rows",
    lines: list[LineRegion] | None = None,
    line_options: LineOptions | None = None,
) -> FileResult:
    """Считает баллы всех запрошенных метрик по одному файлу.

    Args:
        path: Путь к изображению.
        metric_names: Имена метрик из реестра ``ALGORITHMS``.
        tile_size: Желаемая сторона тайла в пикселях.
        aggregation: Режим агрегации тайлов ("best" / "median" / "worst").
        quantile: Квантиль для режимов "best"/"worst".
        zonal_axis: Ось профиля для поиска зонального расфокуса ("rows"/"cols")
            либо None, чтобы его не считать.
        lines: Области строк от детектора. None — считать по сетке тайлов, как раньше.
        line_options: Настройки режима по строкам; нужны только вместе с ``lines``.

    Returns:
        Заполненный ``FileResult``; при ошибке чтения — с непустым ``error``.
    """
    gray = read_gray(path)
    if gray is None:
        return FileResult(path=path, error="не прочитан (нет превью или битый файл)")
    if gray.ndim != 2 or min(gray.shape) < 64:
        return FileResult(path=path, error=f"слишком маленький кадр {gray.shape}")

    grid = make_grid(gray.shape, tile_size)
    printed = printed_mask(detail_rms_map(gray, grid))

    per_metric: dict[str, float] = {}
    for name in metric_names:
        tile_map = ALGORITHMS[name].tile_sharpness(gray, grid)
        per_metric[name] = aggregate(tile_map, printed, mode=aggregation, quantile=quantile)

    result = FileResult(
        path=path,
        score=float("nan"),  # проставляется в analyze_folder: в режиме combo нужен весь список
        per_metric=per_metric,
        shape=(int(gray.shape[0]), int(gray.shape[1])),
        n_printed=int(printed.sum()),
        n_tiles=grid.ny * grid.nx,
        zonal=zonal_defocus(gray, grid, axis=zonal_axis) if zonal_axis else None,
    )
    if lines is not None:
        _measure_by_lines(result, gray, metric_names, lines, line_options or LineOptions(), aggregation, quantile)
    return result


def _measure_by_lines(
    result: FileResult,
    gray: np.ndarray,
    metric_names: tuple[str, ...],
    lines: list[LineRegion],
    options: LineOptions,
    aggregation: str,
    quantile: float,
) -> None:
    """Пересчитывает баллы файла по областям строк и заполняет ``result`` на месте.

    Баллы по сетке тайлов, уже лежащие в ``result``, ЗАМЕЩАЮТСЯ: если детектор строк
    включён, именно он определяет, где текст, и мерить заодно по тайлам значило бы
    печатать в одной колонке две разные величины. А вот направленный зональный отчёт
    (``result.zonal``) остаётся нетронутым — он считается параллельно тайловому, чтобы
    их можно было сравнить на одной папке.

    Единица выборки при агрегации — строка, а не тайл, но сама агрегация та же самая
    (``scoring.aggregate``) и с тем же квантилем: причина брать именно мягкий край
    распределения, а не медиану, от смены единицы не меняется — промахи фокуса при
    съёмке с рук почти всегда неравномерны по кадру (разбор в докстринге ``scoring``).

    Args:
        result: Результат, который заполняется на месте.
        gray: Полутоновый кадр.
        metric_names: Имена метрик.
        lines: Области строк.
        options: Настройки режима по строкам.
        aggregation: Режим агрегации.
        quantile: Квантиль агрегации.
    """
    zonal_source: LineMeasurements | None = None
    zonal_algorithm = None
    for name in metric_names:
        algorithm = ALGORITHMS[name]
        measurements = measure_lines(
            gray,
            lines,
            algorithm,
            n_tiles=options.n_tiles,
            corridor=options.height_corridor,
            aspect=options.chunk_aspect,
        )
        result.n_lines = int(measurements.valid.sum())
        result.n_lines_detected = measurements.n_lines_detected
        result.n_chunks = measurements.n_chunks

        everything = np.ones(measurements.sharpness.shape, dtype=bool)
        result.per_metric[name] = aggregate(measurements.sharpness, everything, mode=aggregation, quantile=quantile)
        if algorithm.length_scaled:
            result.score_norm = aggregate(measurements.normalized(), everything, mode=aggregation, quantile=quantile)
            # Зональная карта строится на СЫРОЙ резкости — она про оптику. Нормированная
            # мерила бы читаемость, а та по кадру меняется ещё и от кегля, который при
            # трапеции сам по себе неоднороден.
            zonal_source, zonal_algorithm = measurements, algorithm
        elif zonal_source is None:
            zonal_source, zonal_algorithm = measurements, algorithm

    if zonal_source is None:
        return
    result.tile_zonal = tile_zonal(zonal_source, n=options.n_tiles, min_lines=options.min_lines)

    if options.debug_dir is not None:
        from ocr_utils.defocus_detection.lines.overlay import draw_overlay, write_overlay

        canvas = draw_overlay(
            gray,
            zonal_source,
            result.tile_zonal,
            zonal_algorithm,
            score=result.per_metric.get(zonal_algorithm.name, float("nan")),
            score_norm=result.score_norm,
            n_tiles=options.n_tiles,
        )
        write_overlay(options.debug_dir / f"{result.path.stem}.jpg", canvas)


def _worker(args: tuple) -> FileResult:
    """Обёртка для ProcessPoolExecutor (нужен модульного уровня callable).

    Args:
        args: Кортеж аргументов ``analyze_file``.

    Returns:
        Результат анализа файла.
    """
    return analyze_file(*args)


def detect_lines(files: list[Path], detector, progress: bool = True) -> list[list[LineRegion] | None]:
    """Прогоняет детектор строк по всем файлам — ЭТАП ДО пула процессов.

    Детекция живёт в главном процессе намеренно. Счётная часть пакета работает
    в полутора десятках воркеров, а surya требует GPU и загруженных весов: держать по
    копии модели в каждом воркере нельзя ни по видеопамяти, ни по здравому смыслу.
    Поэтому кадры сначала размечаются здесь, а в воркеры уезжают уже готовые полигоны —
    несколько сотен четвёрок чисел на кадр, которые пиклятся мгновенно. Побочная выгода:
    torch в дочерние процессы не попадает вовсе.

    Args:
        files: Список путей к изображениям.
        detector: ``lines.LineDetector``.
        progress: Показывать ли полосу прогресса.

    Returns:
        Список областей строк по каждому файлу; None там, где файл не прочитался
        (ошибку сформулирует уже ``analyze_file``).
    """
    detected: list[list[LineRegion] | None] = []
    for path in _with_progress(iter(files), len(files), progress, desc="Детекция строк"):
        gray = read_gray(path)
        detected.append(None if gray is None else detector.detect(path, gray))
    return detected


def analyze_folder(
    files: list[Path],
    algorithm: str,
    tile_size: int = DEFAULT_TILE_SIZE,
    aggregation: str = DEFAULT_AGGREGATION,
    quantile: float = DEFAULT_QUANTILE,
    zonal_axis: str | None = "rows",
    workers: int = 1,
    progress: bool = True,
    detector=None,
    line_options: LineOptions | None = None,
) -> list[FileResult]:
    """Анализирует список файлов и проставляет итоговый балл каждому.

    Args:
        files: Список путей к изображениям.
        algorithm: Имя алгоритма или ``"combo"``.
        tile_size: Желаемая сторона тайла в пикселях.
        aggregation: Режим агрегации тайлов.
        quantile: Квантиль для режимов "best"/"worst".
        zonal_axis: Ось профиля зонального расфокуса или None, чтобы его не считать.
        workers: Число параллельных процессов.
        progress: Показывать ли полосу прогресса.
        detector: ``lines.LineDetector`` для режима по строкам; None — считать по сетке
            тайлов, как раньше.
        line_options: Настройки режима по строкам.

    Returns:
        Список результатов в том же порядке, что и ``files``.
    """
    metric_names = resolve(algorithm)
    lines_per_file: list[list[LineRegion] | None] = [None] * len(files)
    if detector is not None:
        lines_per_file = detect_lines(files, detector, progress)

    tasks = [
        (f, metric_names, tile_size, aggregation, quantile, zonal_axis, lines, line_options)
        for f, lines in zip(files, lines_per_file)
    ]

    if workers > 1 and len(tasks) > 1:
        # Ограничения потоков ставятся ДО создания пула: forkserver поднимает своих детей
        # заново, и переменные окружения они читают в момент импорта библиотек.
        # setdefault, чтобы явно заданное пользователем значение осталось в силе.
        for name in THREAD_LIMIT_VARS:
            os.environ.setdefault(name, "1")
        # forkserver, а не fork: rawpy собран с OpenMP и в форкнутом процессе способен
        # намертво заклиниться (он сам об этом предупреждает при импорте).
        context = multiprocessing.get_context("forkserver")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker) as pool:
            iterator = pool.map(_worker, tasks, chunksize=1)
            results = list(_with_progress(iterator, len(tasks), progress))
    else:
        results = list(_with_progress((_worker(t) for t in tasks), len(tasks), progress))

    if algorithm == COMBO_NAME:
        scores_by_metric = {name: [r.per_metric.get(name, float("nan")) for r in results] for name in metric_names}
        for result, combined in zip(results, rank_combine(scores_by_metric)):
            result.score = combined
    else:
        for result in results:
            result.score = result.per_metric.get(algorithm, float("nan"))
    return results


def _with_progress(iterator, total: int, enabled: bool, desc: str = "Анализ"):
    """Оборачивает итератор в tqdm, если прогресс включён.

    Args:
        iterator: Итератор результатов.
        total: Ожидаемое число элементов.
        enabled: Показывать ли полосу.
        desc: Подпись полосы.

    Yields:
        Элементы исходного итератора.
    """
    if not enabled:
        yield from iterator
        return
    from tqdm import tqdm

    yield from tqdm(iterator, total=total, desc=desc, unit="файл")


def sort_by_zonal(results: list[FileResult]) -> list[FileResult]:
    """Сортирует результаты по убыванию перепада резкости внутри кадра.

    Файлы без зональной оценки (обложки, кадры без текста) уходят в конец: судить
    о зоне там не по чему.

    Args:
        results: Результаты анализа.

    Returns:
        Новый отсортированный список — самые «перекошенные» кадры сверху.
    """
    return sorted(results, key=lambda r: -r.zonal.drop if r.zonal else float("inf"))


def sort_by_tile_zonal(results: list[FileResult]) -> list[FileResult]:
    """Сортирует результаты по убыванию перепада между тайлами зональной сетки.

    Args:
        results: Результаты анализа.

    Returns:
        Новый отсортированный список — самые «перекошенные» кадры сверху; файлы без
        тайловой карты уходят в конец.
    """
    return sorted(results, key=lambda r: -r.tile_zonal.drop if r.tile_zonal else float("inf"))


def sort_worst_first(results: list[FileResult]) -> list[FileResult]:
    """Сортирует результаты от худшего фокуса к лучшему.

    Нечитаемые файлы и файлы без балла ставятся в самое начало: их всё равно надо
    посмотреть глазами.

    Args:
        results: Результаты анализа.

    Returns:
        Новый отсортированный список.
    """
    return sorted(results, key=lambda r: (np.isfinite(r.score), r.score if np.isfinite(r.score) else 0.0))
