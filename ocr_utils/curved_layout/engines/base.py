"""Единый вид результата любого поставщика строк: ломаная оси (или базовой линии) и полигон строки."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class EngineLine:
    """Одна строка от поставщика: ломаная вдоль строки и, если движок его даёт, полигон вокруг неё.

    Args:
        points: Ломаная ``(N, 2)`` в пикселях рабочей копии — центр-линия (наш ход) или базовая
            линия (нейросетевые движки, у них ось ниже центра на половину x-height).
        polygon: Замкнутый полигон строки ``(M, 2)`` в тех же пикселях или ``None``.
        height: Высота строки в пикселях рабочей копии (у нейродвижков — из полигона).
        baseline: ``True``, если ``points`` — базовая линия, а не центр строки: ось тогда
            поднимается на половину высоты.
        confidence: Уверенность движка, если он её сообщает.
    """

    points: np.ndarray
    polygon: np.ndarray | None
    height: float
    baseline: bool = False
    confidence: float = float("nan")


@dataclass(frozen=True)
class EngineResult:
    """Что поставщик нашёл на странице: строки, его регионы и локальные межколонники."""

    lines: list[EngineLine]
    regions: list[np.ndarray] = field(default_factory=list)
    gutters: list = field(default_factory=list)  # локальные межколонники (``columns.Gutter``)
    engine: str = ""
    seconds: float = 0.0
    note: str = ""


class Engine(Protocol):
    """Поставщик строк страницы: по серому рендеру 300 dpi отдаёт :class:`EngineResult`."""

    name: str

    def segment(self, gray300: np.ndarray, dpi: float) -> EngineResult:
        """Найти строки на рендере ``gray300`` и отдать их в пикселях рабочей копии ``dpi``."""
        ...


__all__ = ["Engine", "EngineLine", "EngineResult"]
