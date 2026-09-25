"""Общая часть адаптеров чужих сегментаторов: запуск воркера в своём окружении через файлы.

У kraken, pero и eynollah свои пины torch/numpy (у eynollah ещё и TensorFlow), поэтому в основном
окружении проекта они жить не могут. Каждый ставится в отдельный ``uv venv``, а вызывается
подпроцессом: страница пишется в PNG, воркер отвечает JSON.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Корень окружений по умолчанию: рядом с данными пака, а не в репозитории.
ENGINES_ROOT = Path("/mnt/system/raw/mts/curved_layout_engines")
WORKERS = Path(__file__).parent / "workers"


def run_worker(python: Path, worker: str, gray: np.ndarray, extra: list[str] | None = None, timeout: int = 900) -> dict:
    """Прогнать воркер чужого окружения по изображению и вернуть разобранный JSON.

    Args:
        python: Интерпретатор окружения движка.
        worker: Имя файла воркера в ``workers/``.
        gray: Серое изображение страницы.
        extra: Дополнительные аргументы воркера.
        timeout: Предел времени, секунды.

    Returns:
        Разобранный JSON воркера.

    Raises:
        RuntimeError: Воркер завершился с ошибкой или не отдал JSON.
    """
    with tempfile.TemporaryDirectory(prefix="text_blocks_") as folder:
        page = Path(folder) / "page.png"
        out = Path(folder) / "out.json"
        cv2.imwrite(str(page), gray)
        command = [str(python), str(WORKERS / worker), str(page), str(out), *(extra or [])]
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        if not out.is_file():
            tail = (result.stderr or result.stdout or "").strip().splitlines()[-5:]
            raise RuntimeError(f"{worker}: код {result.returncode}; " + " | ".join(tail))
        return json.loads(out.read_text(encoding="utf-8"))


def polyline(points: list[list[float]], scale: float) -> np.ndarray:
    """Ломаная из JSON в пиксели рабочей копии."""
    return np.asarray(points, dtype=np.float64) * scale


def _column_extents(boundary: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Верх и низ залитого полигона в столбцах через каждые ``step`` пикселей.

    Полигон растеризуется, и в каждом выбранном столбце берётся первый и последний залитый пиксель.
    Раньше толщина бралась по ВЕРШИНАМ полигона, попавшим в полосу по x: в узкую полосу часто
    попадают вершины только одной кромки, и «центр» прыгал на кромку — ось eynollah шла пилой
    с размахом ~22 px при 300 dpi (аудит 2026-09-25, ``reports/line_axis_models.md``).

    Args:
        boundary: Полигон строки ``(M, 2)`` в пикселях.
        step: Шаг выборки столбцов в пикселях.

    Returns:
        Кортеж ``(xs, tops, bottoms)``: абсциссы выбранных столбцов и ординаты верхней и нижней
        кромки залитого полигона в них; столбцы без заливки пропущены.
    """
    # Рамка полигона с полем в пиксель, чтобы заливка не касалась края маски.
    origin = np.floor(boundary.min(axis=0)) - 1.0
    size = np.ceil(boundary.max(axis=0) - origin).astype(int) + 2
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(boundary - origin).astype(np.int32)], 1)
    xs, tops, bottoms = [], [], []
    # Столбцы берутся с шагом step, но не реже одного на пиксель.
    for column in np.arange(0, size[0], max(1.0, step)).astype(int):
        filled = np.flatnonzero(mask[:, column])
        if filled.size:
            xs.append(column + origin[0])
            tops.append(filled[0] + origin[1])
            bottoms.append(filled[-1] + origin[1])
    return np.asarray(xs, dtype=np.float64), np.asarray(tops, dtype=np.float64), np.asarray(bottoms, dtype=np.float64)


def polygon_height(boundary: np.ndarray) -> float:
    """Высота строки по её полигону: медиана вертикальной толщины залитого полигона по столбцам.

    Args:
        boundary: Полигон строки ``(M, 2)`` в пикселях или ``None``.

    Returns:
        Высота в тех же пикселях; 0 для вырожденного полигона.
    """
    if boundary is None or len(boundary) < 3:
        return 0.0
    _, tops, bottoms = _column_extents(boundary, step=2.0)
    return float(np.median(bottoms - tops + 1.0)) if tops.size else 0.0


def centre_from_polygon(boundary: np.ndarray, step: float = 4.0) -> np.ndarray:
    """Центр-линия по полигону строки: середины вертикальной толщины залитого полигона по столбцам.

    Нужна движкам, которые отдают контур строки без базовой линии (eynollah — всегда, PaddleOCR).

    Args:
        boundary: Полигон строки ``(M, 2)``.
        step: Шаг выборки по x в пикселях изображения.

    Returns:
        Ломаная ``(N, 2)`` слева направо; пустой массив, если полигон вырожден.
    """
    if boundary is None or len(boundary) < 3:
        return np.zeros((0, 2), dtype=np.float64)
    xs, tops, bottoms = _column_extents(boundary, step)
    if xs.size < 2:
        return np.zeros((0, 2), dtype=np.float64)
    return np.column_stack([xs, (tops + bottoms) / 2.0])


__all__ = ["ENGINES_ROOT", "WORKERS", "centre_from_polygon", "polygon_height", "polyline", "run_worker"]
