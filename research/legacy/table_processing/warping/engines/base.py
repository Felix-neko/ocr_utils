"""Общий контракт выпрямителей таблицы."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Warped:
    """Результат выпрямления: картинка, поле смещений и слово о том, что вышло."""

    image: np.ndarray
    field: "np.ndarray | None" = None  # (2, H, W): смещения по x и по y, в пикселях
    note: str = ""
    # Отдельным флагом, а не по наличию поля: движок по строкам текста поле наружу не
    # отдаёт, но картинку меняет, и без флага сравнение считало его бездействующим.
    changed: bool = False


def _always() -> bool:
    return True


@dataclass(frozen=True)
class Warper:
    """Запись в реестре выпрямителей.

    ``run`` обязан вернуть картинку ВСЕГДА: не хватило опор — вернуть исходную и объяснить
    словами. Прогон по паку не должен падать из-за одной таблицы без линеек.
    """

    name: str
    summary: str
    run: Callable[[np.ndarray, int], Warped]
    available: Callable[[], bool] = _always
