"""Прогон метрик по файлам папки: чтение, тайлы, баллы, сводный ранг."""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import cv2

from ocr_utils.defocus_detection.image_io import read_gray
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
        error: Текст ошибки, если файл не прочитался.
    """

    path: Path
    score: float = float("nan")
    per_metric: dict[str, float] = field(default_factory=dict)
    shape: tuple[int, int] | None = None
    n_printed: int = 0
    n_tiles: int = 0
    zonal: ZonalResult | None = None
    error: str = ""


def analyze_file(
    path: Path,
    metric_names: tuple[str, ...],
    tile_size: int,
    aggregation: str,
    quantile: float,
    zonal_axis: str | None = "rows",
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

    return FileResult(
        path=path,
        score=float("nan"),  # проставляется в analyze_folder: в режиме combo нужен весь список
        per_metric=per_metric,
        shape=(int(gray.shape[0]), int(gray.shape[1])),
        n_printed=int(printed.sum()),
        n_tiles=grid.ny * grid.nx,
        zonal=zonal_defocus(gray, grid, axis=zonal_axis) if zonal_axis else None,
    )


def _worker(args: tuple) -> FileResult:
    """Обёртка для ProcessPoolExecutor (нужен модульного уровня callable).

    Args:
        args: Кортеж аргументов ``analyze_file``.

    Returns:
        Результат анализа файла.
    """
    return analyze_file(*args)


def analyze_folder(
    files: list[Path],
    algorithm: str,
    tile_size: int = DEFAULT_TILE_SIZE,
    aggregation: str = DEFAULT_AGGREGATION,
    quantile: float = DEFAULT_QUANTILE,
    zonal_axis: str | None = "rows",
    workers: int = 1,
    progress: bool = True,
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

    Returns:
        Список результатов в том же порядке, что и ``files``.
    """
    metric_names = resolve(algorithm)
    tasks = [(f, metric_names, tile_size, aggregation, quantile, zonal_axis) for f in files]

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


def _with_progress(iterator, total: int, enabled: bool):
    """Оборачивает итератор в tqdm, если прогресс включён.

    Args:
        iterator: Итератор результатов.
        total: Ожидаемое число элементов.
        enabled: Показывать ли полосу.

    Yields:
        Элементы исходного итератора.
    """
    if not enabled:
        yield from iterator
        return
    from tqdm import tqdm

    yield from tqdm(iterator, total=total, desc="Анализ", unit="файл")


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
