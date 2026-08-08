"""Общий интерфейс алгоритмов оценки резкости.

Каждый алгоритм — это функция, которая по полутоновому кадру и сетке тайлов возвращает
карту РЕЗКОСТИ по тайлам: «больше значение — резче тайл». Единое направление шкалы нужно,
чтобы агрегация (``scoring.py``), сортировка и отчёт не зависели от конкретного алгоритма.
Если у метрики естественная шкала обратная (например, ширина края в пикселях — «меньше
значит резче»), алгоритм сам её переворачивает, а поле ``display`` описывает, как показать
итоговый балл человеку.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ocr_utils.defocus_detection.tiles import Grid


@dataclass(frozen=True)
class Algorithm:
    """Описание одного алгоритма оценки резкости.

    Attributes:
        name: Имя для CLI (``--algorithm``).
        summary: Однострочное описание для справки.
        tile_sharpness: Функция (gray, grid) -> карта резкости (ny, nx), больше = резче.
            NaN в ячейке означает «в этом тайле метрику посчитать не удалось».
        unit: Подпись колонки с баллом в отчёте.
        display: Как превратить итоговый балл в человекочитаемое число второй колонки
            (например, обратно в ширину края в пикселях). None — второй колонки нет.
        display_unit: Подпись второй колонки.
    """

    name: str
    summary: str
    tile_sharpness: Callable[[np.ndarray, Grid], np.ndarray]
    unit: str = "балл"
    display: Callable[[float], float] | None = None
    display_unit: str = ""
    params: dict = field(default_factory=dict)
