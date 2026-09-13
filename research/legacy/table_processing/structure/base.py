"""Общий контракт разборщиков структуры: вырезанная таблица → сетка ячеек."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np

from research.legacy.table_processing.geometry import Grid


class BatchCellExtractor(Protocol):
    def __call__(self, images: Sequence[np.ndarray]) -> list[Grid]: ...


def _always() -> bool:
    return True


@dataclass(frozen=True)
class CellExtractor:
    name: str
    summary: str
    stage: str  # "cpu" | "gpu"
    run: Callable[[np.ndarray, int], Grid] | None = None  # (вырезанная таблица, dpi)
    make_batch: Callable[[], BatchCellExtractor] | None = None
    available: Callable[[], bool] = _always

    def __post_init__(self) -> None:
        if self.stage not in ("cpu", "gpu"):
            raise ValueError(f"stage={self.stage!r}, а бывает 'cpu' или 'gpu'")
        if self.stage == "cpu" and self.run is None:
            raise ValueError(f"{self.name}: CPU-разборщику нужен run")
        if self.stage == "gpu" and self.make_batch is None:
            raise ValueError(f"{self.name}: GPU-разборщику нужен make_batch")
