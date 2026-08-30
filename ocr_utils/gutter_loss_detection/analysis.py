"""Прогон детектора по папке: чтение, разбор разворота, баллы, сортировка."""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import cv2

from ocr_utils.gutter_loss_detection.geometry import LEFT, RIGHT, SpreadGeometry, analyze_spread, read_work_gray
from ocr_utils.gutter_loss_detection.metrics import THRESHOLD, Verdict, side_bite, spread_bite, verdict

# Насколько понижается приоритет воркеров: прогон по паку идёт долго, машиной всё это
# время надо продолжать пользоваться.
WORKER_NICE = 10

# Переменные, которыми numpy/OpenCV ограничивают внутренние пулы потоков: без них
# каждый воркер разворачивает поток на ядро и процессы дерутся за машину.
THREAD_LIMIT_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")

DEFAULT_RESERVED_CORES = 2


def _init_worker() -> None:
    """Готовит воркер: один поток счёта и пониженный приоритет."""
    cv2.setNumThreads(1)
    for name in THREAD_LIMIT_VARS:
        os.environ.setdefault(name, "1")
    try:
        os.nice(WORKER_NICE)
    except OSError:
        pass


@dataclass
class SideResult:
    """Результат по одной полосе разворота.

    Attributes:
        side: "L" или "R".
        bite: Балл полосы, [0, 1].
        margin: Внутреннее поле по самым тесным строкам, в шагах строк.
        tabular: Похожа ли приосевая зона на таблицу.
    """

    side: str
    bite: float
    margin: float
    tabular: bool


@dataclass
class FileResult:
    """Результат по кадру.

    Attributes:
        path: Путь к файлу.
        score: Балл кадра, [0, 1]; больше — сильнее ушёл текст.
        code: Вердикт: "ок", "текст" либо "таблица".
        why: Пояснение к вердикту.
        sides: Результаты по полосам.
        pitch: Шаг строк в пикселях исходного кадра.
        lines: Сколько строк найдено.
        tilt: Наклон сгиба, пикселей на высоту исходного кадра.
        problem: Почему кадр не измерен; пустая строка — измерен.
        error: Текст ошибки чтения.
    """

    path: Path
    score: float = float("nan")
    code: str = "ок"
    why: str = ""
    sides: list[SideResult] = field(default_factory=list)
    pitch: float = float("nan")
    lines: int = 0
    tilt: float = 0.0
    problem: str = ""
    error: str = ""

    @property
    def bitten_sides(self) -> str:
        """Полосы, где поле съедено, через запятую."""
        order = {LEFT: 0, RIGHT: 1}
        names = [s.side for s in self.sides if np.isfinite(s.bite) and s.bite >= THRESHOLD]
        return ", ".join(sorted(names, key=lambda s: order.get(s, 9)))


def _to_result(path: Path, geometry: SpreadGeometry, threshold: float) -> FileResult:
    """Собирает результат кадра из разбора геометрии."""
    decision: Verdict = verdict(geometry, threshold)
    sides = [
        SideResult(side=s.side, bite=side_bite(s), margin=s.tight, tabular=s.rules_v >= 2 or s.rules_h >= 2)
        for s in geometry.sides
    ]
    return FileResult(
        path=path,
        score=spread_bite(geometry),
        code=decision.code,
        why=decision.why,
        sides=sides,
        pitch=geometry.pitch * geometry.scale,
        lines=geometry.lines,
        tilt=geometry.tilt * geometry.scale,
        problem=geometry.problem,
    )


def analyze_file(path: Path, threshold: float = THRESHOLD) -> FileResult:
    """Меряет один кадр.

    Args:
        path: Путь к файлу.
        threshold: Порог по баллу.

    Returns:
        ``FileResult``; при ошибке чтения заполнено поле ``error``.
    """
    try:
        gray, scale = read_work_gray(path)
    except Exception as exc:  # noqa: BLE001 — на паке важнее продолжить, чем упасть
        return FileResult(path=path, error=f"{type(exc).__name__}: {exc}")
    return _to_result(path, analyze_spread(gray, scale), threshold)


def _worker(args) -> FileResult:
    path, threshold = args
    return analyze_file(path, threshold)


def analyze_folder(paths, jobs: int | None = None, threshold: float = THRESHOLD) -> list[FileResult]:
    """Меряет папку в пуле процессов.

    Args:
        paths: Пути к кадрам.
        jobs: Сколько процессов; None — по числу ядер минус запас.
        threshold: Порог по баллу.

    Returns:
        Список результатов в порядке входных путей.
    """
    paths = list(paths)
    if jobs is None:
        jobs = max(1, (multiprocessing.cpu_count() or 4) - DEFAULT_RESERVED_CORES)
    if jobs <= 1 or len(paths) <= 1:
        return [analyze_file(p, threshold) for p in paths]
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker) as pool:
        return list(pool.map(_worker, [(p, threshold) for p in paths], chunksize=2))


def sort_worst_first(results: list[FileResult]) -> list[FileResult]:
    """Сортирует кадры по убыванию балла; неизмеренные — в конец.

    Args:
        results: Результаты прогона.

    Returns:
        Новый отсортированный список.
    """
    return sorted(results, key=lambda r: (-r.score if np.isfinite(r.score) else 1.0, r.path.name))
