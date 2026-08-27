"""Прогон метрик просвечивания по файлам: чтение, деление на полосы, баллы, ранги."""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import cv2

from ocr_utils.defocus_detection.image_io import read_gray
from ocr_utils.defocus_detection.scoring import rank_combine
from ocr_utils.show_through_detection.metrics import ALGORITHMS, COMBO_NAME, FALLBACK, needed, resolve
from ocr_utils.show_through_detection.page_split import LEFT, RIGHT, WHOLE, Spread, find_gutter, split_spread
from ocr_utils.show_through_detection.zones import build_zones

# Насколько понижается приоритет воркеров: прогон по паку идёт часами, и всё это время
# машиной надо продолжать пользоваться.
WORKER_NICE = 10

# Переменные, которыми numpy/OpenCV ограничивают внутренние пулы потоков. Без них
# каждый воркер разворачивает по потоку на ядро, и полтора десятка процессов устраивают
# многократную перезапись — машина занята вся, считая при этом не быстрее.
THREAD_LIMIT_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")

# Порядок сторон в отчёте — порядок страниц на развороте.
SIDE_ORDER = (WHOLE, LEFT, RIGHT)


def _init_worker() -> None:
    """Готовит процесс-воркер: один поток счёта и пониженный приоритет."""
    cv2.setNumThreads(1)
    try:
        os.nice(WORKER_NICE)
    except OSError:
        # Понижать приоритет можно всегда, но на всякий случай не падаем из-за политики.
        pass


@dataclass
class HalfResult:
    """Результат замера одной полосы (половины разворота).

    Attributes:
        path: Путь к исходному кадру.
        side: Сторона разворота: "L", "R" либо "-" для одиночной страницы.
        score: Сырой балл метрики, которой полоса в итоге измерена.
        severity: Балл, делённый на порог этой метрики. Именно по нему идёт сортировка
            и отбор: он сравним между основной метрикой и запасной, у которых шкалы
            разные, и читается как «во сколько раз превышен порог». 1.0 — ровно порог.
        per_metric: Сырые баллы всех посчитанных метрик.
        metric: Имя метрики, давшей ``score``; отличается от запрошенной, когда сработала
            запасная.
        note: Оговорка к замеру («нет опорных полей») либо пустая строка.
        problem: Почему полосу измерить нельзя вовсе («нет текста»); пустая — измерена.
    """

    path: Path
    side: str
    score: float = float("nan")
    severity: float = float("nan")
    per_metric: dict[str, float] = field(default_factory=dict)
    metric: str = ""
    note: str = ""
    problem: str = ""

    @property
    def name(self) -> str:
        """Человекочитаемое имя полосы: имя файла и сторона."""
        return self.path.name if self.side == "-" else f"{self.path.name} [{self.side}]"


@dataclass
class FileResult:
    """Результат анализа одного кадра.

    Attributes:
        path: Путь к файлу.
        halves: Результаты по полосам кадра.
        shape: Размер прочитанного кадра (height, width) или None.
        gutter: Найденный столбец корешка.
        gutter_confident: Нашёлся ли корешок уверенно.
        error: Текст ошибки, если файл не прочитался.
    """

    path: Path
    halves: list[HalfResult] = field(default_factory=list)
    shape: tuple[int, int] | None = None
    gutter: int = 0
    gutter_confident: bool = False
    error: str = ""

    @property
    def severity(self) -> float:
        """Худшая (наибольшая) превышенность порога среди полос кадра."""
        values = [h.severity for h in self.halves if np.isfinite(h.severity)]
        return max(values) if values else float("nan")

    def worst_sides(self, threshold: float = 1.0) -> str:
        """Стороны, превысившие порог, через запятую.

        Порог передаётся, а не берётся равным единице: его можно двигать ключом
        ``--threshold``, и при сдвинутом пороге жёсткая единица врала бы — кадр попадал
        бы в список, а колонка «полосы» показывала бы прочерк.

        Args:
            threshold: Порог по ``severity`` в долях от калибровочного.

        Returns:
            Строка вида ``"L, R"``; пустая, если ни одна полоса порога не достигла.
        """
        order = {side: index for index, side in enumerate(SIDE_ORDER)}
        sides = [h.side for h in self.halves if np.isfinite(h.severity) and h.severity >= threshold]
        # Порядок страниц, а не порядок, в котором полосы легли в список: он зависит от
        # того, как результат собран, и «R, L» в отчёте выглядело бы опечаткой.
        return ", ".join(sorted(sides, key=lambda side: order.get(side, len(order))))


def _severity(name: str, value: float) -> float:
    """Переводит сырой балл метрики в «во сколько раз превышен её порог».

    Args:
        name: Имя метрики.
        value: Сырой балл.

    Returns:
        Отношение к порогу либо сам балл, если порога у метрики нет (режим ``combo``).
    """
    threshold = ALGORITHMS[name].threshold if name in ALGORITHMS else None
    if threshold is None or not np.isfinite(value):
        return value
    return value / threshold


def analyze_half(path: Path, side: str, gray: np.ndarray, metric_names: tuple[str, ...], algorithm: str) -> HalfResult:
    """Считает баллы всех запрошенных метрик по одной полосе.

    Args:
        path: Путь к исходному кадру.
        side: Сторона разворота.
        gray: Полутоновая полоса.
        metric_names: Какие метрики считать (уже с запасной, см. ``metrics.needed``).
        algorithm: Запрошенный алгоритм (или ``combo``).

    Returns:
        Заполненный ``HalfResult``.
    """
    zones = build_zones(gray)
    result = HalfResult(path=path, side=side, note=zones.note, problem=zones.problem)
    if not zones.usable:
        return result

    result.per_metric = {name: ALGORITHMS[name].score_of(zones) for name in metric_names}
    if algorithm == COMBO_NAME:
        # Балл combo — средний ранг по всей выборке, он проставляется после пула.
        return result

    value = result.per_metric.get(algorithm, float("nan"))
    used = algorithm
    if not np.isfinite(value) and FALLBACK in result.per_metric:
        value, used = result.per_metric[FALLBACK], FALLBACK
    result.score, result.metric = value, used
    result.severity = _severity(used, value)
    return result


def analyze_file(path: Path, algorithm: str) -> FileResult:
    """Читает кадр, делит на полосы и считает по каждой баллы просвечивания.

    Args:
        path: Путь к изображению.
        algorithm: Имя алгоритма или ``"combo"``.

    Returns:
        Заполненный ``FileResult``; при ошибке чтения — с непустым ``error``.
    """
    gray = read_gray(path)
    if gray is None:
        return FileResult(path=path, error="не прочитан (нет превью или битый файл)")
    if gray.ndim != 2 or min(gray.shape) < 256:
        return FileResult(path=path, error=f"слишком маленький кадр {gray.shape}")

    metric_names = needed(algorithm)
    spread = find_gutter(gray)
    halves = [analyze_half(path, side, half, metric_names, algorithm) for side, half in split_spread(gray, spread)]
    return FileResult(
        path=path, halves=halves, shape=gray.shape, gutter=spread.gutter, gutter_confident=spread.confident
    )


def _worker(args: tuple) -> FileResult:
    """Обёртка для ProcessPoolExecutor (нужен модульного уровня callable).

    Args:
        args: Кортеж аргументов ``analyze_file``.

    Returns:
        Результат анализа файла.
    """
    return analyze_file(*args)


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

    yield from tqdm(iterator, total=total, desc=desc, unit="кадр")


def analyze_folder(files: list[Path], algorithm: str, workers: int = 1, progress: bool = True) -> list[FileResult]:
    """Анализирует список файлов и проставляет балл каждой полосе.

    Args:
        files: Список путей к изображениям.
        algorithm: Имя алгоритма или ``"combo"``.
        workers: Число параллельных процессов.
        progress: Показывать ли полосу прогресса.

    Returns:
        Список результатов в том же порядке, что и ``files``.
    """
    tasks = [(f, algorithm) for f in files]

    if workers > 1 and len(tasks) > 1:
        # Ограничения потоков ставятся ДО создания пула: forkserver поднимает своих детей
        # заново и переменные окружения читает в момент импорта библиотек.
        for name in THREAD_LIMIT_VARS:
            os.environ.setdefault(name, "1")
        # forkserver, а не fork: rawpy собран с OpenMP и в форкнутом процессе способен
        # намертво заклиниться (он сам об этом предупреждает при импорте).
        context = multiprocessing.get_context("forkserver")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_init_worker) as pool:
            results = list(_with_progress(pool.map(_worker, tasks, chunksize=1), len(tasks), progress))
    else:
        results = list(_with_progress((_worker(t) for t in tasks), len(tasks), progress))

    if algorithm == COMBO_NAME:
        halves = all_halves(results)
        scores_by_metric = {n: [h.per_metric.get(n, float("nan")) for h in halves] for n in resolve(algorithm)}
        for half, combined in zip(halves, rank_combine(scores_by_metric)):
            half.score, half.severity, half.metric = combined, combined, COMBO_NAME
    return results


def all_halves(results: list[FileResult]) -> list[HalfResult]:
    """Разворачивает результаты по кадрам в плоский список полос.

    Args:
        results: Результаты анализа кадров.

    Returns:
        Список полос в порядке кадров.
    """
    return [half for result in results for half in result.halves]


def sort_worst_first(halves: list[HalfResult]) -> list[HalfResult]:
    """Сортирует полосы от самого сильного просвета к самому слабому.

    Полосы без балла (нет текста, нечитаемый кадр) уходят в конец: судить о них
    не по чему, и наверху отчёта они только мешали бы.

    Args:
        halves: Результаты по полосам.

    Returns:
        Новый отсортированный список.
    """
    return sorted(halves, key=lambda h: -h.severity if np.isfinite(h.severity) else float("inf"))


def sort_files_worst_first(results: list[FileResult]) -> list[FileResult]:
    """Сортирует кадры по худшей из их полос.

    Args:
        results: Результаты анализа кадров.

    Returns:
        Новый отсортированный список.
    """
    return sorted(results, key=lambda r: -r.severity if np.isfinite(r.severity) else float("inf"))
