"""Единый интерфейс движка выпрямления страниц."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class DewarpEngine(ABC):
    """Базовый класс движка dewarp.

    Жизненный цикл: ``load(device)`` один раз (скачать веса/репозиторий, поднять
    модель), затем ``dewarp(bgr)`` на каждый кадр. ``dewarp`` возвращает выпрямленный
    BGR-кадр либо ``None``, если данный кадр обработать не удалось (тогда оркестратор
    его пропускает, не прерывая остальную пачку).
    """

    name: str = "base"

    @abstractmethod
    def load(self, device: str) -> None:
        """Подготавливает модель к инференсу (скачивание + загрузка в память)."""

    @abstractmethod
    def dewarp(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Возвращает выпрямленный BGR-кадр или None (если кадр не обработан)."""
