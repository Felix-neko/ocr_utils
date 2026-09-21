"""Общий контракт детекторов поворота ЯЧЕЙКИ.

ВАЛЮТА ТА ЖЕ, что у ориентации полос: ``Verdict.rotate_cw`` — на сколько повернуть ПО
ЧАСОВОЙ, чтобы стало прямо. Классы ``Verdict``, ``Detector`` и функция ``unknown``
переиспользуются из ``ocr_utils.page_layout.orientation.detectors.base`` намеренно: второе
соглашение о знаке угла по соседству рано или поздно повернуло бы текст не в ту сторону,
а отчёты обоих пакетов читает один человек.

ЧТО ОТЛИЧАЕТСЯ ОТ ПОЛОСЫ. Ячейка — это несколько слов, а не полторы сотни строк. Всё, что
на полосе считалось статистикой (асимметрия выносных элементов, периодичность профиля),
здесь считается по горстке букв, и потому детекторы обязаны честно говорить «не знаю».
Зато есть то, чего нет у полосы: соседи. Все боковые ячейки одной таблицы повёрнуты в одну
сторону, и это даёт приор, которым разрешаются сомнительные случаи (см. ``combine``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np

# Единая валюта углов и общий контракт вердикта — из пакета ориентации полос.
from ocr_utils.page_layout.orientation.detectors.base import Verdict, unknown  # noqa: F401
from ocr_utils.scan_markup.rotation import rotate_cw  # noqa: F401

from research.legacy.table_processing.geometry import Cell

# Допустимые повороты ячейки. 180 не рассматриваем: перевёрнутая вверх ногами ячейка в
# таблице не встречается, а лишний класс — лишние ошибки.
DEFAULT_ALLOWED: tuple[int, ...] = (0, 90, 270)


@dataclass
class CellCrop:
    """Вырезанная ячейка со всем, что нужно детектору."""

    crop_id: str
    cell: Cell
    gray: np.ndarray  # внутренность ячейки без линеек
    dpi: int
    allowed: tuple[int, ...] = DEFAULT_ALLOWED
    metrics: dict[str, float] = field(default_factory=dict)
    # Вся таблица целиком, если детектору нужен контекст. Не копия, а ссылка: детектор
    # строк Surya на одной ячейке 100x230 px работает заметно хуже, чем на таблице, где
    # видны соседние строки, — а рамки строк потом раскладываются по ячейкам сами.
    table: "np.ndarray | None" = None

    @property
    def key(self) -> str:
        return f"{self.crop_id}:r{self.cell.row}c{self.cell.col}"


class BatchCellDetector(Protocol):
    """GPU-детектор: модель грузится лениво, на вход пачка вырезанных ячеек."""

    def __call__(self, crops: Sequence[CellCrop]) -> list[Verdict]: ...


def _always() -> bool:
    return True


@dataclass(frozen=True)
class CellDetector:
    name: str
    summary: str
    stage: str  # "cpu" | "gpu"
    gives_sign: bool  # различает ли 90 и 270 или только ось
    run: Callable[[CellCrop], Verdict] | None = None
    make_batch: Callable[[], BatchCellDetector] | None = None
    available: Callable[[], bool] = _always

    def __post_init__(self) -> None:
        if self.stage not in ("cpu", "gpu"):
            raise ValueError(f"stage={self.stage!r}, а бывает 'cpu' или 'gpu'")
        if self.stage == "cpu" and self.run is None:
            raise ValueError(f"{self.name}: CPU-детектору нужен run")
        if self.stage == "gpu" and self.make_batch is None:
            raise ValueError(f"{self.name}: GPU-детектору нужен make_batch")
