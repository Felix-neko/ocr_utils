"""Заметные изменения границ блоков «было | стало»: хаусдорфово расстояние многоугольников и смена числа блоков."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ocr_utils.page_layout.pack_analysis.stages import page_key

# Изменение границы блока меньше стольких пикселей рабочей копии — шум кромки (1–3 px), а не перестройка.
SIGNIFICANT_PX = 10.0


def polygons(run_dir: Path, page: str) -> list[np.ndarray]:
    """Многоугольники блоков полосы из итогового JSON прогона."""
    data = json.loads((run_dir / "pages" / f"{page_key(page)}.json").read_text())
    return [np.asarray(block["polygon"], dtype=float) for block in data["text_blocks"]["blocks"]]


def hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    """Хаусдорфово расстояние между вершинами двух многоугольников (пиксели)."""
    distances = np.linalg.norm(a[:, None] - b[None], axis=2)
    return float(max(distances.min(1).max(), distances.min(0).max()))


def shift_of(before: Path, after: Path, page: str) -> float:
    """Наибольший сдвиг границы блока полосы между прогонами; ``inf`` — поменялось число блоков."""
    first, second = polygons(before, page), polygons(after, page)
    if len(first) != len(second):
        return float("inf")
    return max((hausdorff(a, b) for a, b in zip(first, second)), default=0.0)
