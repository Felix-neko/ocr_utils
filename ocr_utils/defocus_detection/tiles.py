"""Сетка тайлов и маска «печатных» (текстовых) тайлов.

ЗАЧЕМ ТАЙЛЫ. Считать резкость по всему кадру нельзя: балл тогда зависит от того, сколько
на полосе текста, а не от фокуса. Полоса с крупным заголовком и большими пустотами даст
низкий балл при идеальном фокусе. Поэтому метрика считается по тайлам, пустые тайлы
(поля, межколонники, светлый фон) выбрасываются, а балл файла собирается только по тайлам
с краской — см. ``scoring.py``.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# 0 — размер тайла подобрать автоматически, по ширине кадра (см. make_grid).
DEFAULT_TILE_SIZE = 0
# Сколько тайлов укладывать по ширине в автоматическом режиме. Девять — это ~490 px на
# превью 4416×2944, то есть примерно газетная колонка на треть высоты полосы. Значение
# подобрано на размеченной папке 1979 года: мельче тайл — меньше краёв в нём и больше
# шума в оценке (AUC падает с 0.90 до 0.86–0.87 при тайле 128–256 px).
# Задавать сетку, а не пиксели, важно для кадров разного разрешения: тайл должен покрывать
# одну и ту же ДОЛЮ полосы, иначе «худшие 20 % тайлов» означают у них разное.
AUTO_GRID_COLUMNS = 9

# Порог «печатности» тайла: доля от 75-го процентиля детальности по кадру.
# Относительный порог (а не абсолютный) — потому что уровни гуляют от ISO и освещения.
DEFAULT_PRINTED_REL = 0.40
# Страховка от совсем тёмных/шумных кадров: ниже этого RMS деталей (в уровнях 8 бит)
# тайл считается пустым в любом случае.
DEFAULT_PRINTED_MIN_RMS = 1.5


@dataclass(frozen=True)
class Grid:
    """Равномерная сетка тайлов по кадру.

    Attributes:
        ny: Число тайлов по вертикали.
        nx: Число тайлов по горизонтали.
        height: Высота кадра в пикселях.
        width: Ширина кадра в пикселях.
    """

    ny: int
    nx: int
    height: int
    width: int

    def bounds(self, iy: int, ix: int) -> tuple[int, int, int, int]:
        """Границы тайла (y1, y2, x1, x2) в пикселях.

        Args:
            iy: Индекс тайла по вертикали.
            ix: Индекс тайла по горизонтали.

        Returns:
            Кортеж (y1, y2, x1, x2); из-за целочисленного деления крайние тайлы
            могут отличаться по размеру на пиксель.
        """
        y1, y2 = iy * self.height // self.ny, (iy + 1) * self.height // self.ny
        x1, x2 = ix * self.width // self.nx, (ix + 1) * self.width // self.nx
        return y1, y2, x1, x2


def make_grid(shape: tuple[int, int], tile_size: int = DEFAULT_TILE_SIZE) -> Grid:
    """Строит сетку тайлов по кадру.

    Args:
        shape: Размер кадра (height, width).
        tile_size: Желаемая сторона тайла в пикселях; 0 — подобрать так, чтобы по
            ширине уложилось ``AUTO_GRID_COLUMNS`` тайлов (кадры разного разрешения
            получат тайлы одинаковой доли полосы, а не одинакового числа пикселей).

    Returns:
        Сетка Grid (минимум 2×2 тайла).
    """
    height, width = shape
    if tile_size <= 0:
        tile_size = max(32, round(width / AUTO_GRID_COLUMNS))
    ny = max(2, round(height / tile_size))
    nx = max(2, round(width / tile_size))
    return Grid(ny=ny, nx=nx, height=height, width=width)


def tile_reduce(values: np.ndarray, grid: Grid, kind: str = "mean") -> np.ndarray:
    """Агрегирует попиксельную карту по тайлам сетки.

    Args:
        values: Попиксельный массив размера (height, width).
        grid: Сетка тайлов.
        kind: "mean" — среднее по тайлу, "sum" — сумма, "rms" — среднеквадратичное.

    Returns:
        Массив (ny, nx) со значением на тайл.
    """
    out = np.zeros((grid.ny, grid.nx), dtype=np.float64)
    for iy in range(grid.ny):
        for ix in range(grid.nx):
            y1, y2, x1, x2 = grid.bounds(iy, ix)
            block = values[y1:y2, x1:x2]
            if kind == "sum":
                out[iy, ix] = float(block.sum())
            elif kind == "rms":
                out[iy, ix] = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
            else:
                out[iy, ix] = float(block.mean())
    return out


def detail_rms_map(gray: np.ndarray, grid: Grid, illum_sigma: float = 8.0) -> np.ndarray:
    """Карта «детальности» тайлов — RMS высокочастотной составляющей.

    Из кадра вычитается сильно размытая копия: так уходит неравномерность освещения
    и общий уровень яркости, остаётся только рисунок краски. Величина по-прежнему
    зависит от экспозиции/контраста, поэтому использовать её надо только относительно
    (см. ``printed_mask``), а не как абсолютный порог.

    Args:
        gray: Полутоновый кадр.
        grid: Сетка тайлов.
        illum_sigma: Сигма гауссова размытия для оценки фона (пиксели).

    Returns:
        Массив (ny, nx) с RMS деталей по тайлам.
    """
    g = gray.astype(np.float32)
    background = cv2.GaussianBlur(g, (0, 0), illum_sigma)
    return tile_reduce(g - background, grid, kind="rms")


def printed_mask(
    detail_rms: np.ndarray, rel: float = DEFAULT_PRINTED_REL, min_rms: float = DEFAULT_PRINTED_MIN_RMS
) -> np.ndarray:
    """Маска тайлов с краской (текст, растр, иллюстрации).

    Порог берётся относительно 75-го процентиля детальности самого кадра: так маска
    не зависит ни от экспозиции/ISO, ни от того, насколько плотно набрана полоса.
    75-й процентиль, а не медиана, — потому что на малотекстовых полосах пустых тайлов
    больше половины и медиана уехала бы в фон.

    Args:
        detail_rms: Карта детальности из ``detail_rms_map``.
        rel: Доля от 75-го процентиля, ниже которой тайл считается пустым.
        min_rms: Абсолютный нижний предел RMS деталей (уровни 8 бит).

    Returns:
        Булев массив (ny, nx): True — в тайле есть краска.
    """
    threshold = max(min_rms, rel * float(np.percentile(detail_rms, 75)))
    return detail_rms >= threshold
