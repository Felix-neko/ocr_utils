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
        length_scaled: Величина метрики обратно пропорциональна ДЛИНЕ (как 1/σ у ширины
            края). Только такие метрики можно нормировать на кегль, получая «размытие в
            долях высоты строки» — меру читаемости мелкого текста. У остальных (доли
            энергии, безразмерные отношения) деление на высоту строки смысла не имеет,
            и нормированная колонка у них не считается.
        region_sharpness: Резкость ОДНОГО маленького кропа (кусок строки): пара
            (балл, вес), где вес — объём статистики, на которой балл получен (для
            ширины края это число измеренных перепадов). Вес нужен, чтобы куски одной
            строки складывались в общий замер строки пропорционально тому, сколько
            каждый из них намерил. None — использовать ``tile_sharpness`` с сеткой 1x1
            и площадь кропа как вес. Второй аргумент — контекст кадра из ``frame_context``.
        frame_context: Что посчитать по всему кадру ОДИН раз и передать в каждый вызов
            ``region_sharpness``. Нужно метрикам, у которых порог отбора привязан к
            контрасту кадра целиком: в кропе одной строки такой порог посчитался бы по
            разбросу внутри букв и у каждого куска вышел бы свой.
    """

    name: str
    summary: str
    tile_sharpness: Callable[[np.ndarray, Grid], np.ndarray]
    unit: str = "балл"
    display: Callable[[float], float] | None = None
    display_unit: str = ""
    params: dict = field(default_factory=dict)
    length_scaled: bool = False
    region_sharpness: Callable[[np.ndarray, object], tuple[float, float]] | None = None
    frame_context: Callable[[np.ndarray], object] | None = None

    def context(self, gray: np.ndarray) -> object:
        """Контекст кадра для замеров по областям.

        Args:
            gray: Полутоновый кадр полного разрешения.

        Returns:
            Значение, которое будет передано в каждый вызов ``sharpness_of``; None,
            если метрике контекст не нужен.
        """
        return self.frame_context(gray) if self.frame_context is not None else None

    def sharpness_of(self, crop: np.ndarray, context: object = None) -> tuple[float, float]:
        """Резкость одного маленького кропа: пара (балл, вес).

        Универсальный запасной путь — вызвать ``tile_sharpness`` на сетке 1x1 и взять
        весом площадь кропа. Сетка конструируется здесь напрямую, минуя
        ``tiles.make_grid``: тот принудительно разбивает кадр минимум на 2x2 тайла, а
        нам нужен ровно один замер на весь кроп.

        Args:
            crop: Полутоновый кусок кадра.
            context: Контекст кадра из ``context()``.

        Returns:
            Пара (балл резкости, вес). Нулевой вес означает «не измерено».
        """
        if self.region_sharpness is not None:
            return self.region_sharpness(crop, context)
        height, width = crop.shape[:2]
        grid = Grid(ny=1, nx=1, height=height, width=width)
        value = float(self.tile_sharpness(crop, grid)[0, 0])
        return (value, float(crop.size)) if np.isfinite(value) else (float("nan"), 0.0)

    def pool(self, values: np.ndarray, weights: np.ndarray) -> float:
        """Сводит замеры нескольких кусков одной строки в один балл строки.

        Для метрик с размерностью длины берётся ВЗВЕШЕННОЕ ГАРМОНИЧЕСКОЕ среднее, и это
        не вкусовщина. Балл там — величина, обратная длине (резкость = 1/σ), а усреднять
        физически осмысленно саму σ: она есть среднее по измеренным перепадам, поэтому
        куски складываются пропорционально числу перепадов в каждом. Среднее σ, выраженное
        через баллы, — это ровно ``Σw / Σ(w/s)``. Арифметическое среднее баллов дало бы
        систематически завышенную резкость (неравенство средних) тем сильнее, чем
        неоднороднее строка.

        Для безразмерных метрик обратной связи с длиной нет, и берётся обычное
        взвешенное среднее.

        Args:
            values: Баллы кусков.
            weights: Их веса.

        Returns:
            Балл строки либо NaN, если суммарный вес нулевой.
        """
        good = np.isfinite(values) & (weights > 0)
        if not good.any():
            return float("nan")
        v, w = values[good], weights[good]
        total = float(w.sum())
        if self.length_scaled:
            return float(total / np.sum(w / v))
        return float(np.sum(w * v) / total)
