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
    with tempfile.TemporaryDirectory(prefix="curved_layout_") as folder:
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


def polygon_height(boundary: np.ndarray) -> float:
    """Высота строки по её полигону: медиана вертикальной толщины по столбцам."""
    if boundary is None or len(boundary) < 3:
        return 0.0
    xs, ys = boundary[:, 0], boundary[:, 1]
    edges = np.linspace(xs.min(), xs.max(), num=min(24, max(3, int(xs.max() - xs.min()) // 10 + 3)))
    thickness = []
    for left, right in zip(edges[:-1], edges[1:]):
        own = ys[(xs >= left) & (xs <= right)]
        if own.size >= 2:
            thickness.append(own.max() - own.min())
    return float(np.median(thickness)) if thickness else float(ys.max() - ys.min())


def centre_from_polygon(boundary: np.ndarray, step: float = 4.0) -> np.ndarray:
    """Центр-линия по полигону строки: середины вертикальной толщины по столбцам.

    Нужна движкам, которые отдают контур строки без базовой линии (eynollah в части строк).

    Args:
        boundary: Полигон строки ``(M, 2)``.
        step: Шаг выборки по x в пикселях изображения.

    Returns:
        Ломаная ``(N, 2)`` слева направо; пустой массив, если полигон вырожден.
    """
    if boundary is None or len(boundary) < 3:
        return np.zeros((0, 2), dtype=np.float64)
    xs, ys = boundary[:, 0], boundary[:, 1]
    grid = np.arange(xs.min(), xs.max() + step, step)
    points = []
    for left, right in zip(grid[:-1], grid[1:]):
        own = ys[(xs >= left) & (xs <= right)]
        if own.size >= 2:
            points.append(((left + right) / 2.0, (own.min() + own.max()) / 2.0))
    return np.asarray(points, dtype=np.float64) if len(points) >= 2 else np.zeros((0, 2), dtype=np.float64)


__all__ = ["ENGINES_ROOT", "WORKERS", "centre_from_polygon", "polygon_height", "polyline", "run_worker"]
