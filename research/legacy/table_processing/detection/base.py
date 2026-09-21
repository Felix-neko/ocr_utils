"""Общий контракт детекторов таблиц.

Тот же приём, что в ``ocr_utils.page_layout.orientation.detectors.base``: детекторы
взаимозаменяемы, отчёт знает только имя и найденные рамки, а ``stage`` разводит CPU
(едет в пул процессов) и GPU (обязан остаться в родителе — видеопамять одна на всех).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

from research.legacy.table_processing.geometry import TableBox


class BatchTableDetector(Protocol):
    """GPU-детектор: модель грузится лениво, на вход пачка серых полос."""

    def __call__(self, images: Sequence[np.ndarray]) -> list[list[TableBox]]: ...


def _always() -> bool:
    return True


@dataclass(frozen=True)
class TableDetector:
    name: str
    summary: str
    stage: str  # "cpu" | "gpu"
    run: Callable[[np.ndarray, int], list[TableBox]] | None = None  # (серая полоса, dpi)
    make_batch: Callable[[], BatchTableDetector] | None = None
    available: Callable[[], bool] = _always

    def __post_init__(self) -> None:
        if self.stage not in ("cpu", "gpu"):
            raise ValueError(f"stage={self.stage!r}, а бывает 'cpu' или 'gpu'")
        if self.stage == "cpu" and self.run is None:
            raise ValueError(f"{self.name}: CPU-детектору нужен run")
        if self.stage == "gpu" and self.make_batch is None:
            raise ValueError(f"{self.name}: GPU-детектору нужен make_batch")
