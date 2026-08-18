"""Замер резкости по областям строк и отбор «мелкого текста».

ЕДИНИЦА ЗАМЕРА — СТРОКА, А НЕ КУСОК. Куски (``regions.line_chunks``) существуют только
как способ вырезать наклонную строку без единого пересчёта пикселей; сами по себе они
слишком мелки, чтобы судить по каждому отдельно. Поэтому замеры кусков одной строки
складываются в один балл строки пропорционально объёму статистики в каждом
(``Algorithm.pool``), и уже строка — со своим центром тяжести — относится к тайлу
зональной сетки.

Порога «достаточно ли краёв» на уровне куска нет намеренно: наклон укорачивает куски, и
пороговый отсев выкашивал бы замеры ровно там, где искажения сильнее — то есть создавал
бы ложный зональный сигнал из ничего. Решение принимается на уровне строки, где статистики
на порядок больше.

ДВА БАЛЛА, И ОНИ ОТВЕЧАЮТ НА РАЗНЫЕ ВОПРОСЫ.

**Сырая σ края (в пикселях)** — свойство оптики. Ширина, до которой объектив размазал
резкий перепад, от кегля почти не зависит: у большой буквы переходов меньше, но каждый
такой же крутой, как у мелкой. Поэтому именно сырая σ — честный ответ на вопрос «в этом
месте кадра объектив попал в фокус или нет», и зональная карта строится на ней.

**σ, делённая на высоту строки** — читаемость. Одно и то же размытие в полпикселя не
мешает заголовку и убивает петит: важно не сколько пикселей размазано, а какую долю
буквы они составляют. Это и есть ответ на вопрос «распознается ли здесь мелкий текст»,
и общий балл файла считается по ней. Тот же приём — раздельная оценка размытия и кегля —
лежит в основе Rodin et al., «Document Image Quality Assessment via Explicit Blur and
Text Size Estimation» (ICDAR 2021).

Нормировка осмысленна только для метрик с размерностью длины (``Algorithm.length_scaled``);
у безразмерных отношений вроде ``reblur`` деление на высоту строки не значит ничего, и
нормированный балл им не считается.

ОТБОР МЕЛКОГО ТЕКСТА ИДЁТ ПО ТАЙЛАМ, А НЕ ПО КАДРУ. Коридор перцентилей высоты строки
считается внутри каждого тайла зональной сетки отдельно. Причина — трапеция: плоскость
съёмки завалена, ближний край кадра снят крупнее, и один и тот же петит даёт там строки
заметно выше, чем на дальнем краю. Глобальный коридор «оставить строки ниже медианы»
выкосил бы тогда весь ближний край — то есть выбросил бы из зональной карты целую
сторону кадра, причём ровно ту, где искажения сильнее всего.
"""

from dataclasses import dataclass, field

import numpy as np

from ocr_utils.defocus_detection.lines.regions import DEFAULT_CHUNK_ASPECT, Chunk, LineRegion, line_chunks
from ocr_utils.defocus_detection.metrics.base import Algorithm

# Коридор перцентилей высоты строки внутри тайла: что считаем «мелким текстом».
# Верхняя граница 60 отсекает заголовки, подзаголовки и врезки, оставляя корпусный набор
# (на замеренных полосах «Социалистической индустрии» медиана высоты 16-20 px при p90 в
# 28-31 px — то есть выше 60-го перцентиля начинается уже не корпус). Нижняя граница 0:
# снизу отсекать нечего, самый мелкий набор нам и нужен, а обрывки отсеются по весу.
DEFAULT_HEIGHT_CORRIDOR = (0.0, 60.0)
# Минимум строк в тайле, чтобы вообще считать по нему перцентили. Если строк меньше,
# коридор вырождается в случайный отбор — берём тайл целиком.
MIN_LINES_FOR_CORRIDOR = 8
# Минимальный суммарный вес строки (для ширины края — число измеренных перепадов), ниже
# которого балл строки считается ненадёжным. Строка корпусного набора длиной в колонку
# даёт сотни перепадов, так что порог отсекает обрывки в два-три знака, а не нормальный
# набор — даже сильно наклонный.
DEFAULT_MIN_WEIGHT = 60.0


@dataclass
class LineMeasurements:
    """Замеры по строкам одного кадра.

    Attributes:
        regions: Строки, прошедшие отбор по кеглю.
        chunks: Куски каждой строки — нужны для отладочных наложений.
        sharpness: Балл резкости строки (больше = резче); NaN — не измерена.
        weights: Объём статистики, на которой получен балл строки.
        heights: Высота каждой строки в пикселях.
        tile_index: Плоский индекс тайла зональной сетки для каждой строки.
        n_lines_detected: Сколько строк нашёл детектор до отбора по кеглю.
        n_chunks: Сколько кусков всего вырезано.
    """

    regions: list[LineRegion] = field(default_factory=list)
    chunks: list[list[Chunk]] = field(default_factory=list)
    sharpness: np.ndarray = field(default_factory=lambda: np.zeros(0))
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    heights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tile_index: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    n_lines_detected: int = 0
    n_chunks: int = 0

    @property
    def valid(self) -> np.ndarray:
        """Маска строк, для которых балл посчитался.

        Returns:
            Булев массив по строкам.
        """
        return np.isfinite(self.sharpness)

    def normalized(self) -> np.ndarray:
        """Балл читаемости каждой строки: её высота, делённая на σ.

        Больше = читаемее. Для метрики ``edge_width`` (резкость = 1/σ) это ровно
        ``высота / σ`` — «сколько ширин размытого края укладывается в букву».

        Returns:
            Массив по строкам.
        """
        return self.sharpness * self.heights


def tile_of(centre: tuple[float, float], shape: tuple[int, int], n: int) -> int:
    """Плоский индекс тайла сетки n x n, которому принадлежит точка.

    К тайлу относят по ЦЕНТРУ ТЯЖЕСТИ строки — так строка не режется границей тайла и
    целиком достаётся одному из них.

    Args:
        centre: Координаты точки (x, y) в пикселях кадра.
        shape: Размер кадра (height, width).
        n: Сторона сетки.

    Returns:
        Индекс ``iy * n + ix``.
    """
    height, width = shape
    cx, cy = centre
    ix = min(n - 1, max(0, int(cx * n / max(width, 1))))
    iy = min(n - 1, max(0, int(cy * n / max(height, 1))))
    return iy * n + ix


def small_text_mask(
    heights: np.ndarray, tile_index: np.ndarray, corridor: tuple[float, float], n_tiles: int
) -> np.ndarray:
    """Маска строк, попадающих в коридор «мелкого текста» ВНУТРИ своего тайла.

    Перцентили считаются по каждому тайлу отдельно — см. докстринг модуля: при
    трапециевидных искажениях кегль систематически меняется по кадру, и общий на весь
    кадр коридор выбросил бы целый край.

    Args:
        heights: Высоты строк.
        tile_index: Индекс тайла каждой строки.
        corridor: Границы коридора в перцентилях (нижняя, верхняя).
        n_tiles: Всего тайлов в сетке.

    Returns:
        Булев массив по строкам.
    """
    keep = np.zeros(heights.shape, dtype=bool)
    low, high = corridor
    for tile in range(n_tiles):
        here = tile_index == tile
        count = int(here.sum())
        if count == 0:
            continue
        if count < MIN_LINES_FOR_CORRIDOR:
            # Перцентили по горстке строк — это не отбор, а лотерея. Берём тайл целиком:
            # пусть в нём останется заголовок, чем тайл вовсе выпадет из карты.
            keep[here] = True
            continue
        lo, hi = np.percentile(heights[here], [low, high])
        keep[here] = (heights[here] >= lo) & (heights[here] <= hi)
    return keep


def measure_lines(
    gray: np.ndarray,
    regions: list[LineRegion],
    algorithm: Algorithm,
    n_tiles: int,
    corridor: tuple[float, float] = DEFAULT_HEIGHT_CORRIDOR,
    aspect: float = DEFAULT_CHUNK_ASPECT,
    min_weight: float = DEFAULT_MIN_WEIGHT,
) -> LineMeasurements:
    """Считает резкость по областям строк кадра.

    Порядок шагов выбран ради скорости: строки сначала отсеиваются по кеглю и только
    потом режутся на куски и измеряются — заголовки, которых нам не надо, не стоят
    ни одного замера.

    Args:
        gray: Полутоновый кадр полного разрешения.
        regions: Области строк от детектора, в координатах этого кадра.
        algorithm: Алгоритм оценки резкости.
        n_tiles: Сторона зональной сетки (коридор кегля считается по её тайлам).
        corridor: Коридор перцентилей высоты строки.
        aspect: Ширина куска в высотах строки.
        min_weight: Минимальный суммарный вес строки, иначе её балл — NaN.

    Returns:
        Замеры по строкам; при отсутствии строк — пустой набор.
    """
    if not regions:
        return LineMeasurements()

    heights = np.array([r.height for r in regions], dtype=np.float64)
    tiles = np.array([tile_of(r.centre, gray.shape, n_tiles) for r in regions], dtype=np.int64)
    keep = small_text_mask(heights, tiles, corridor, n_tiles * n_tiles)

    selected = [r for r, take in zip(regions, keep) if take]
    if not selected:
        return LineMeasurements(n_lines_detected=len(regions))

    context = algorithm.context(gray)
    chunks_per_line: list[list[Chunk]] = []
    sharpness, weights = [], []
    for region in selected:
        chunks = line_chunks(region, gray.shape, aspect=aspect)
        chunks_per_line.append(chunks)
        if not chunks:
            sharpness.append(float("nan"))
            weights.append(0.0)
            continue
        measured = [algorithm.sharpness_of(chunk.crop(gray), context) for chunk in chunks]
        values = np.array([v for v, _ in measured], dtype=np.float64)
        chunk_weights = np.array([w for _, w in measured], dtype=np.float64)
        total = float(chunk_weights[np.isfinite(values)].sum())
        sharpness.append(algorithm.pool(values, chunk_weights) if total >= min_weight else float("nan"))
        weights.append(total)

    return LineMeasurements(
        regions=selected,
        chunks=chunks_per_line,
        sharpness=np.array(sharpness, dtype=np.float64),
        weights=np.array(weights, dtype=np.float64),
        heights=np.array([r.height for r in selected], dtype=np.float64),
        tile_index=np.array([tile_of(r.centre, gray.shape, n_tiles) for r in selected], dtype=np.int64),
        n_lines_detected=len(regions),
        n_chunks=sum(len(c) for c in chunks_per_line),
    )
