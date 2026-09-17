"""Единый интерфейс движка выпрямления страниц."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class DewarpEngine(ABC):
    """Базовый класс движка dewarp.

    Жизненный цикл: ``load(device)`` один раз (скачать веса/репозиторий, поднять модель),
    затем ``dewarp(bgr)`` на каждый кадр. ``dewarp`` возвращает выпрямленный BGR-кадр либо
    ``None``, если кадр обработать не удалось (оркестратор его пропускает, не прерывая
    остальную пачку).

    ``dpi`` — разрешение подаваемого кадра; нужно движкам, которые меряют строки в
    физических единицах (``textline``). Оркестратор выставляет его перед ``dewarp``.
    """

    name: str = "base"
    dpi: int = 600

    @abstractmethod
    def load(self, device: str) -> None:
        """Подготавливает модель к инференсу (скачивание + загрузка в память)."""

    @abstractmethod
    def dewarp(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Возвращает выпрямленный BGR-кадр или None (если кадр не обработан)."""

    def unload(self) -> None:
        """Освобождает модель (видеопамять); по умолчанию ничего не делает."""
