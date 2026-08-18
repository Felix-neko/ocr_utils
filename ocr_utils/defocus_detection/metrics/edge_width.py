"""Ширина края штриха (Marziliano / JNB) — насколько «размазан» отдельный переход.

ИДЕЯ. Расфокус превращает резкий перепад яркости в пологий: край, который в фокусе
занимает пиксель-полтора, растягивается на три-четыре. Меряем это напрямую. Каждая строка
и каждый столбец кадра режутся на монотонные участки между локальными экстремумами
яркости — один такой участок и есть один край в терминах Marziliano.

Ширину края считаем не как длину участка в целых пикселях (слишком грубо для лёгких
расфокусов), а как СКО профиля градиента внутри него:

    σ² = Σ (k − центр)² · |ΔI_k| / Σ |ΔI_k| − 1/12

Для ступеньки, размытой гауссианой σ, это в точности даёт σ, причём с субпиксельной
точностью; вычитаемая 1/12 — дисперсия пиксельной апертуры (интегрирование по площади
пикселя), без неё идеально резкий край давал бы 0.29 вместо нуля. Итоговая метрика —
средняя σ по тайлу; резкость = 1/σ_эфф.

ПОЧЕМУ ЭТО ПОДХОДИТ ПОД ЗАДАЧУ (главное отличие от Лапласиана и энергетических метрик):
- **Не зависит от количества текста.** Плотность текста меняет ЧИСЛО замеров, а не их
  среднее. Полоса с одной колонкой и полоса-простыня в одинаковом фокусе дают одну и ту же
  среднюю ширину края.
- **Не боится крупных заголовков.** У большой буквы переходов мало, но каждый переход
  такой же крутой, как у мелкой, — ширина края та же. Энергетические метрики (variance of
  Laplacian, Tenengrad) на заголовках проваливаются, потому что считают суммарную энергию.
- **Устойчиво к ISO и освещению.** Ширина края — геометрия профиля, а не его амплитуда:
  экспозиция масштабирует перепад, но не растягивает его. Порог отбора перепадов берётся
  относительно контраста самого кадра, поэтому уезжает вместе с экспозицией.

Литература: Marziliano et al., «A no-reference perceptual blur metric» (ICIP 2002);
Ferzli & Karam, JNB (IEEE TIP 2009); Narvekar & Karam, CPBD (IEEE TIP 2011). Оценка σ по
второму моменту профиля градиента — то же, что делает slanted-edge метод измерения MTF
(ISO 12233), только по «случайным» краям текста вместо специальной миры.
"""

import numpy as np

from ocr_utils.defocus_detection.metrics.base import Algorithm
from ocr_utils.defocus_detection.tiles import Grid

# Перепады с амплитудой ниже этой доли от «чернил кадра» (p95 амплитуд) считаем шумом:
# у них экстремумы расставляет не типографика, а зерно матрицы.
DEFAULT_AMP_REL = 0.25
# Абсолютный нижний предел амплитуды в уровнях 8 бит — страховка на почти пустых кадрах,
# где p95 сам по себе шум.
DEFAULT_AMP_MIN = 8.0
# Перепады длиннее этого — не штрихи текста, а плавные растяжки (виньетка, тень от сгиба,
# градиент фона). Измерять их бессмысленно, они бы утащили среднее вверх.
DEFAULT_MAX_RUN = 12
# Минимум замеров в тайле: статистика по десятку краёв — шум.
DEFAULT_MIN_EDGES = 200
# Через строку (столбец) — обсчитывается вдвое меньше пикселей при том же результате:
# в тайле 256×256 газетного текста даже с шагом 2 набирается несколько тысяч краёв,
# так что на среднее прореживание не влияет, а счёт ускоряется вдвое.
DEFAULT_STEP = 2
# Дисперсия пиксельной апертуры: пиксель интегрирует свет по своей площади, что само по
# себе размывает профиль градиента на прямоугольник шириной 1 px (дисперсия 1/12).
PIXEL_APERTURE_VAR = 1.0 / 12.0


def _row_runs(img: np.ndarray) -> dict[str, np.ndarray]:
    """Режет каждую строку на монотонные участки и меряет их геометрию.

    Реализация полностью векторная: позиции экстремумов ищутся по смене знака первой
    разности, «следующий экстремум справа» — кумулятивным минимумом справа налево,
    а моменты профиля градиента внутри участка — разностями кумулятивных сумм.

    Args:
        img: Полутоновый кадр (float64), участки ищутся вдоль оси X.

    Returns:
        Словарь массивов размера кадра, значимых там, где ``start`` = True:
        ``start`` — в пикселе начинается монотонный участок;
        ``run`` — длина участка в целых пикселях;
        ``amplitude`` — перепад яркости на его концах;
        ``sigma`` — СКО профиля градиента внутри участка (субпиксельная ширина края).
    """
    h, w = img.shape
    diff = np.diff(img, axis=1)

    # Экстремум — там, где знак наклона меняется (уход в ноль тоже завершает участок:
    # плоскость — это конец края). Границы кадра всегда считаем экстремумами.
    sign = np.sign(diff).astype(np.int8)
    extremum = np.zeros((h, w), dtype=bool)
    extremum[:, 1:-1] = sign[:, :-1] != sign[:, 1:]
    extremum[:, 0] = True
    extremum[:, -1] = True
    del sign

    # Ближайший экстремум СТРОГО правее текущей позиции.
    cols = np.arange(w, dtype=np.int32)
    pos = np.where(extremum, cols[None, :], np.int32(w))
    next_incl = np.minimum.accumulate(pos[:, ::-1], axis=1)[:, ::-1]
    del pos
    next_strict = np.full((h, w), w, dtype=np.int32)
    next_strict[:, :-1] = next_incl[:, 1:]
    del next_incl

    start = extremum & (next_strict < w)
    del extremum
    right = np.minimum(next_strict, w - 1)
    run = (next_strict - cols[None, :]).astype(np.float32)
    del next_strict
    amplitude = np.abs(np.take_along_axis(img, right, axis=1) - img)

    # Моменты профиля градиента. Веса — модули первых разностей; координата k берётся
    # в центре между отсчётами (k + 0.5), поэтому центроид попадает ровно на середину
    # односэмплового края. Координата отсчитывается от середины строки, чтобы моменты
    # второго порядка не разрастались до величин, где разность префиксных сумм теряет
    # значащие разряды.
    weight = np.abs(diff)
    centre = cols[:-1].astype(np.float64) + (0.5 - w / 2.0)
    zero = np.zeros((h, 1), dtype=np.float64)
    # Префиксные суммы имеют ровно w столбцов: s[j] = сумма по k < j. Поэтому левый край
    # участка — это сам массив без индексации, а правый берётся из обрезанного `right`
    # (у невалидных позиций значения мусорные, но их отсекает маска `start`).
    moments = []
    for power in (0, 1, 2):
        prefix = np.concatenate([zero, np.cumsum(weight * centre[None, :] ** power, axis=1)], axis=1)
        moments.append(np.take_along_axis(prefix, right, axis=1) - prefix)
    m0, m1, m2 = moments

    with np.errstate(invalid="ignore", divide="ignore"):
        safe = np.maximum(m0, 1e-9)
        variance = m2 / safe - (m1 / safe) ** 2
    sigma = np.sqrt(np.maximum(variance - PIXEL_APERTURE_VAR, 0.0))

    return dict(start=start, run=run, amplitude=amplitude, sigma=sigma)


def _segment_sums(values: np.ndarray, bounds: list[tuple[int, int]]) -> np.ndarray:
    """Суммы значений полосы по отрезкам вдоль её оси X.

    Args:
        values: Массив полосы (высота полосы, длина).
        bounds: Границы отрезков [(a, b), …] вдоль оси X.

    Returns:
        Массив длины ``len(bounds)`` с суммой по каждому отрезку.
    """
    return np.array([values[:, a:b].sum() for a, b in bounds], dtype=np.float64)


def _accumulate(
    img: np.ndarray, grid: Grid, amp_threshold: float, max_run: int, step: int = DEFAULT_STEP
) -> dict[str, np.ndarray]:
    """Суммирует замеры принятых краёв по тайлам — по строкам и по столбцам сразу.

    Кадр обрабатывается ПОЛОСАМИ в один ряд (столбец) тайлов, а не целиком: промежуточных
    массивов в ``_row_runs`` полтора десятка, и на превью 4416×2944 они дают под два
    гигабайта на файл — при прогоне в дюжину процессов машина уходит в своп. Полоса режет
    пик памяти пропорционально числу рядов тайлов, а результат не меняет: монотонные
    участки ищутся внутри строки (столбца), и полоса всегда захватывает её целиком.

    Args:
        img: Полутоновый кадр (float64).
        grid: Сетка тайлов.
        amp_threshold: Минимальная амплитуда перепада, чтобы его измерять.
        max_run: Максимальная длина участка, чтобы считать его штрихом.
        step: Через сколько строк (столбцов) брать замеры.

    Returns:
        Словарь с массивами (ny, nx): ``count`` — число принятых краёв в тайле,
        ``sum_run`` — сумма их длин, ``sum_sigma`` — сумма их σ.
    """
    totals = {k: np.zeros((grid.ny, grid.nx), dtype=np.float64) for k in ("count", "sum_run", "sum_sigma")}
    rows = [grid.bounds(iy, 0)[:2] for iy in range(grid.ny)]
    cols = [grid.bounds(0, ix)[2:] for ix in range(grid.nx)]

    def band_totals(band: np.ndarray, bounds: list[tuple[int, int]]) -> dict[str, np.ndarray]:
        """Считает края в полосе и раскладывает суммы по отрезкам вдоль неё."""
        runs = _row_runs(band)
        accepted = runs["start"] & (runs["amplitude"] >= amp_threshold) & (runs["run"] <= max_run)
        return {
            "count": _segment_sums(accepted, bounds),
            "sum_run": _segment_sums(np.where(accepted, runs["run"], 0.0), bounds),
            "sum_sigma": _segment_sums(np.where(accepted, runs["sigma"], 0.0), bounds),
        }

    # Вертикальные штрихи ищутся по строкам кадра: полоса — ряд тайлов во всю ширину.
    for iy, (y1, y2) in enumerate(rows):
        for key, values in band_totals(img[y1:y2:step], cols).items():
            totals[key][iy] += values

    # Горизонтальные — по столбцам: полоса — столбец тайлов во всю высоту, транспонированный,
    # так что «строкой» для ``_row_runs`` становится полный столбец кадра.
    for ix, (x1, x2) in enumerate(cols):
        for key, values in band_totals(np.ascontiguousarray(img[:, x1:x2:step].T), rows).items():
            totals[key][:, ix] += values
    return totals


def amplitude_threshold(gray: np.ndarray, amp_rel: float = DEFAULT_AMP_REL, amp_min: float = DEFAULT_AMP_MIN) -> float:
    """Минимальная амплитуда перепада, ниже которой он считается шумом.

    Порог привязан к контрасту самого кадра: p95 амплитуд определяют крупные контрастные
    детали (заголовки, рамки, границы иллюстраций), которые лёгкий расфокус почти не
    трогает, — поэтому порог не «уезжает» вслед за размытием, но подстраивается под
    экспозицию и ISO. Считаем его по каждой восьмой строке: счёт вчетверо дешевле, а на
    сотнях тысяч замеров p95 от прореживания не двигается.

    Вынесено в отдельную функцию, потому что при замере по областям строк порог обязан
    считаться по ВСЕМУ кадру и передаваться в каждый кусок готовым: в кропе одной строки
    p95 амплитуд — это уже не «чернила кадра», а разброс внутри самих букв, и порог
    получился бы у каждого куска свой.

    Args:
        gray: Полутоновый кадр.
        amp_rel: Доля от p95 амплитуд кадра.
        amp_min: Абсолютный минимум амплитуды в уровнях 8 бит.

    Returns:
        Порог амплитуды в уровнях 8 бит.
    """
    img = gray.astype(np.float64)
    probe = img[:: max(1, img.shape[0] // 400)]
    return max(amp_min, amp_rel * float(np.percentile(_row_runs(probe)["amplitude"], 95)))


def edge_stats(
    gray: np.ndarray,
    grid: Grid,
    amp_rel: float = DEFAULT_AMP_REL,
    amp_min: float = DEFAULT_AMP_MIN,
    max_run: int = DEFAULT_MAX_RUN,
    min_edges: int = DEFAULT_MIN_EDGES,
    amp_threshold: float | None = None,
) -> dict[str, np.ndarray]:
    """Карты замеров краёв по тайлам: субпиксельная σ, целая длина участка, число краёв.

    Число краёв нужно не только как признак «достаточно ли статистики»: по нему
    отбираются однородные тайлы тела-текста при поиске зонального расфокуса
    (``ocr_utils.defocus_detection.zonal``).

    Args:
        gray: Полутоновый кадр.
        grid: Сетка тайлов.
        amp_rel: Доля от p95 амплитуд кадра, ниже которой перепад игнорируется.
        amp_min: Абсолютный минимум амплитуды в уровнях 8 бит.
        max_run: Максимальная длина участка, ещё считающаяся штрихом.
        min_edges: Минимум замеров в тайле, иначе σ и длина участка = NaN.
        amp_threshold: Готовый порог амплитуды; None — посчитать по поданному кадру
            (см. ``amplitude_threshold``).

    Returns:
        Словарь с массивами (ny, nx): ``sigma`` — средняя σ края в пикселях,
        ``run`` — средняя длина монотонного участка, ``count`` — число принятых краёв.
    """
    img = gray.astype(np.float64)
    if amp_threshold is None:
        amp_threshold = amplitude_threshold(gray, amp_rel, amp_min)

    totals = _accumulate(img, grid, amp_threshold, max_run)
    enough = totals["count"] >= min_edges
    with np.errstate(invalid="ignore", divide="ignore"):
        count = np.maximum(totals["count"], 1.0)
        sigma = np.where(enough, totals["sum_sigma"] / count, np.nan)
        run = np.where(enough, totals["sum_run"] / count, np.nan)
    return dict(sigma=sigma, run=run, count=totals["count"])


def edge_maps(gray: np.ndarray, grid: Grid, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Карты средней ширины края по тайлам: субпиксельная σ и целая длина участка.

    Args:
        gray: Полутоновый кадр.
        grid: Сетка тайлов.
        **kwargs: Параметры отбора краёв, см. ``edge_stats``.

    Returns:
        Кортеж (sigma, run); NaN там, где замеров не хватило.
    """
    stats = edge_stats(gray, grid, **kwargs)
    return stats["sigma"], stats["run"]


def _tile_sharpness(gray: np.ndarray, grid: Grid) -> np.ndarray:
    """Карта резкости: обратная субпиксельная ширина края (больше = резче).

    Args:
        gray: Полутоновый кадр.
        grid: Сетка тайлов.

    Returns:
        Массив (ny, nx) со значениями 1/σ; NaN там, где краёв не хватило.
    """
    sigma, _ = edge_maps(gray, grid)
    with np.errstate(invalid="ignore", divide="ignore"):
        # Нижняя отсечка σ страхует от деления на ноль на синтетике с идеальными краями.
        return 1.0 / np.maximum(sigma, 1e-3)


def _region_sharpness(crop: np.ndarray, context: object) -> tuple[float, float]:
    """Резкость одного куска строки: обратная субпиксельная ширина края и число краёв.

    Порога «достаточно ли краёв» здесь НЕТ намеренно. Кусок строки в сотню раз меньше
    тайла по площади, и требовать от него тайловых ``DEFAULT_MIN_EDGES`` значило бы
    выбрасывать куски целиком — причём тем чаще, чем сильнее наклон строки (наклон
    укорачивает кусок). Получился бы обрыв замеров ровно там, где искажения сильнее.
    Вместо порога кусок возвращает свой вес — число измеренных перепадов, — а решение
    «хватило ли статистики» принимается уже на уровне СТРОКИ, где куски сложены вместе
    (см. ``Algorithm.pool`` и ``lines.measure``).

    Args:
        crop: Полутоновый кусок строки.
        context: Порог амплитуды, посчитанный по всему кадру (``amplitude_threshold``).

    Returns:
        Пара (1/σ, число краёв); вес 0 означает «в куске не нашлось ни одного перепада».
    """
    height, width = crop.shape[:2]
    grid = Grid(ny=1, nx=1, height=height, width=width)
    stats = edge_stats(crop, grid, min_edges=1, amp_threshold=context)
    sigma = float(stats["sigma"][0, 0])
    count = float(stats["count"][0, 0])
    if not np.isfinite(sigma) or count <= 0:
        return float("nan"), 0.0
    return 1.0 / max(sigma, 1e-3), count


ALGORITHM = Algorithm(
    name="edge_width",
    summary="субпиксельная ширина края штриха (Marziliano/JNB): не зависит ни от количества текста, ни от кегля",
    tile_sharpness=_tile_sharpness,
    unit="1/σ",
    display=lambda score: 1.0 / score if score > 0 else float("nan"),
    display_unit="σ края, px",
    # σ измеряется в пикселях, поэтому её осмысленно делить на высоту строки и получать
    # «размытие в долях кегля» — то, что определяет читаемость мелкого текста.
    length_scaled=True,
    region_sharpness=_region_sharpness,
    frame_context=amplitude_threshold,
)
