"""Поиск ЗОНАЛЬНОГО расфокуса — когда часть кадра мягче остального.

ЗАЧЕМ ОТДЕЛЬНО ОТ ОБЩЕГО БАЛЛА. Общий балл отвечает на вопрос «насколько плох кадр
целиком» и по построению усредняет. Кадр, у которого верхние две трети резче среднего
по выпуску, а нижняя треть заметно поплыла, по общему баллу окажется в середине списка —
и это не ошибка метрики, а другой вопрос. Переснимать такой кадр всё равно надо.
Эталонный пример — DSCF0196 из папки 1979 года: 129-е место из 317 по общему баллу и
2-е по перепаду (разбор в `defocus_detection_validation_report.md`, § 8).

КАК МЕРЯЕТСЯ. Кадр режется на сетку вдвое мельче основной, по каждому тайлу считается
ширина края σ, затем строится ПРОФИЛЬ по горизонтальным полосам (полоса = ряд тайлов),
и метрика — относительный перепад между самой резкой и самой мягкой полосой.

Три решения, без которых это не работает (проверено на размеченной папке):

1. **Только однородные тайлы тела-текста.** σ зависит от того, чем набран кусок полосы:
   узкая колонка мелким шрифтом или жирный курсивный лозунг дают большую σ и при
   идеальном фокусе. Поэтому в профиль берутся только тайлы, у которых И число краёв,
   И средняя длина штриха попадают в коридоры перцентилей этого же кадра.

2. **Полосы поперёк колонок, а не вдоль.** Горизонтальная полоса пересекает все колонки
   разворота и усредняет вёрстку; вертикальная целиком лежит внутри одной колонки и
   наследует её кегль. На реальной выборке столбцовый профиль оказался бесполезен:
   в топ стабильно лезли кадры из-за узкой колонки другим шрифтом у правого поля,
   одной и той же у всех выпусков. Ось выбирается параметром ``axis``, по умолчанию
   горизонтальные полосы — они верны для газет и книг с вертикальными колонками.

3. **Сглаживание профиля и разнос полос.** Без них метрика срабатывает на паре соседних
   полос, то есть на шуме. Профиль сглаживается по три полосы, а самая резкая и самая
   мягкая обязаны отстоять друг от друга минимум на ``min_separation`` полос: оптический
   завал — это плавный градиент через полкадра, а не скачок между соседями.
"""

from dataclasses import dataclass

import numpy as np

from ocr_utils.defocus_detection.metrics.edge_width import edge_stats
from ocr_utils.defocus_detection.tiles import Grid, tile_reduce

# Во сколько раз сетка для поиска зон мельче основной (по каждой оси).
FINE_FACTOR = 2
# Коридоры перцентилей внутри кадра, отбирающие однородно набранные тайлы.
# Число краёв в тайле — прокси «сколько тут текста»; средняя длина монотонного участка —
# прокси кегля и начертания. Второй коридор обязателен: без него в профиль попадают
# крупные жирные КУРСИВНЫЕ лозунги, у которых σ завышена не расфокусом, а наклоном
# штриха относительно строк развёртки, и кадр уходит в ложные срабатывания.
DEFAULT_COUNT_CORRIDOR = (35.0, 90.0)
DEFAULT_RUN_CORRIDOR = (25.0, 75.0)
# Минимальная доля «бумаги» в тайле. Отсекает полутоновые фотографии: у них белого фона
# нет вовсе (доля ~0), тогда как даже у самой плотно набранной полосы её пятая часть.
# Порог абсолютный, а не перцентильный, именно потому, что плотность набора гуляет от
# полосы к полосе, а «в тайле нет бумаги» — признак не вёрстки, а другого типа контента.
DEFAULT_MIN_PAPER = 0.18
# Сколько крайних рядов и столбцов тайлов выбросить: там поля, обрез страницы и стол.
DEFAULT_MARGIN = 1
# Минимум однородных тайлов в полосе, иначе полоса не участвует.
DEFAULT_MIN_TILES = 4
# Минимальный разнос самой резкой и самой мягкой полосы (в полосах).
DEFAULT_MIN_SEPARATION = 3

AXES = ("rows", "cols")


@dataclass
class ZonalResult:
    """Оценка зонального расфокуса одного кадра.

    Attributes:
        drop: Относительный перепад ширины края между самой резкой и самой мягкой
            полосой: 0.0 — кадр ровный, 0.3 — в мягкой полосе штрих на 30 % шире.
        best: Индекс самой резкой полосы.
        worst: Индекс самой мягкой полосы.
        n_bands: Всего полос в кадре.
        axis: По какой оси строился профиль ("rows" — горизонтальные полосы).
        profile: Сглаженный профиль σ по полосам (NaN там, где тайлов не хватило).
    """

    drop: float
    best: int
    worst: int
    n_bands: int
    axis: str
    profile: list[float]

    def where(self) -> str:
        """Человекочитаемое описание, какая часть кадра поплыла.

        Returns:
            Строка вида «низ (полоса 10 из 12, резче всего 4)».
        """
        if self.axis == "rows":
            names = ("верх", "середина", "низ")
        else:
            names = ("лево", "центр", "право")
        position = self.worst / max(self.n_bands - 1, 1)
        label = names[0] if position < 0.34 else (names[1] if position < 0.67 else names[2])
        return f"{label} ({self.worst + 1} из {self.n_bands}, резче всего {self.best + 1})"


def fine_grid(shape: tuple[int, int], grid: Grid) -> Grid:
    """Строит сетку для поиска зон — вдвое мельче основной по каждой оси.

    Args:
        shape: Размер кадра (height, width).
        grid: Основная сетка, по которой считается общий балл.

    Returns:
        Сетка с удвоенным числом тайлов по обеим осям.
    """
    return Grid(ny=grid.ny * FINE_FACTOR, nx=grid.nx * FINE_FACTOR, height=shape[0], width=shape[1])


def paper_map(gray: np.ndarray, grid: Grid) -> np.ndarray:
    """Доля пикселей бумаги (светлого фона) в каждом тайле.

    Уровень бумаги берётся как p90 самого кадра, уровень краски — как p5, порог лежит
    на четверти пути от бумаги к краске. Всё относительно кадра, поэтому величина
    не зависит от экспозиции.

    Args:
        gray: Полутоновый кадр.
        grid: Сетка тайлов.

    Returns:
        Массив (ny, nx) с долей пикселей бумаги в [0, 1].
    """
    white, dark = np.percentile(gray, 90), np.percentile(gray, 5)
    return tile_reduce((gray > white - 0.25 * (white - dark)).astype(np.float32), grid, kind="mean")


def _smooth(profile: np.ndarray) -> np.ndarray:
    """Сглаживает профиль по три полосы, не размазывая его на пустые полосы.

    Args:
        profile: Профиль по полосам, NaN там, где полоса не измерена.

    Returns:
        Сглаженный профиль; NaN остаются NaN (иначе край кадра «подтянулся» бы
        к соседу и попал в отчёт как самая мягкая полоса).
    """
    out = np.full_like(profile, np.nan)
    for k in range(profile.size):
        if not np.isfinite(profile[k]):
            continue
        window = profile[max(0, k - 1) : k + 2]
        window = window[np.isfinite(window)]
        out[k] = window.mean()
    return out


def band_profile(
    sigma: np.ndarray,
    count: np.ndarray,
    axis: str = "rows",
    run: np.ndarray | None = None,
    paper: np.ndarray | None = None,
    count_corridor: tuple[float, float] = DEFAULT_COUNT_CORRIDOR,
    run_corridor: tuple[float, float] = DEFAULT_RUN_CORRIDOR,
    min_paper: float = DEFAULT_MIN_PAPER,
    margin: int = DEFAULT_MARGIN,
    min_tiles: int = DEFAULT_MIN_TILES,
) -> np.ndarray:
    """Строит профиль ширины края по полосам, по однородным тайлам тела-текста.

    Args:
        sigma: Карта σ по тайлам мелкой сетки.
        count: Карта числа измеренных краёв по тем же тайлам.
        axis: "rows" — горизонтальные полосы, "cols" — вертикальные.
        run: Карта средней длины монотонного участка (прокси кегля и начертания);
            None — не фильтровать по ней.
        paper: Карта доли пикселей бумаги в тайле; None — не фильтровать по ней.
        count_corridor: Коридор перцентилей числа краёв.
        run_corridor: Коридор перцентилей длины штриха.
        min_paper: Минимальная доля бумаги в тайле (отсев полутоновых фотографий).
        margin: Сколько крайних рядов/столбцов тайлов выбросить.
        min_tiles: Минимум однородных тайлов в полосе.

    Returns:
        Профиль средней σ по полосам; NaN там, где тайлов не хватило.
    """
    if not np.isfinite(sigma).any():
        # Кадр без текста (обложка, чистый лист): мерить нечего, а перцентили по массиву
        # из одних NaN только сыплют предупреждениями.
        return np.full(sigma.shape[0] if axis == "rows" else sigma.shape[1], np.nan)

    low, high = np.nanpercentile(count, count_corridor)
    homogeneous = (count >= low) & (count <= high) & np.isfinite(sigma)
    if run is not None:
        run_low, run_high = np.nanpercentile(run, run_corridor)
        homogeneous &= (run >= run_low) & (run <= run_high)
    if paper is not None:
        homogeneous &= paper >= min_paper
    if margin > 0:
        homogeneous[:margin, :] = homogeneous[-margin:, :] = False
        homogeneous[:, :margin] = homogeneous[:, -margin:] = False

    if axis == "cols":
        sigma, homogeneous = sigma.T, homogeneous.T
    profile = np.full(sigma.shape[0], np.nan)
    for k in range(sigma.shape[0]):
        selected = homogeneous[k]
        if selected.sum() >= min_tiles:
            profile[k] = sigma[k][selected].mean()
    return profile


def profile_drop(profile: np.ndarray, min_separation: int = DEFAULT_MIN_SEPARATION) -> tuple[float, int, int] | None:
    """Находит наибольший перепад между разнесёнными полосами профиля.

    Args:
        profile: Сглаженный профиль σ по полосам.
        min_separation: Минимальное расстояние между полосами (в полосах).

    Returns:
        Кортеж (перепад, индекс резкой полосы, индекс мягкой) либо None, если
        измеренных полос слишком мало.
    """
    # Нулевая σ бывает только на синтетике с идеальными ступеньками (на реальном снимке
    # край всегда шире пикселя), но делить на неё нельзя, поэтому такие полосы отбрасываем.
    known = np.where(np.isfinite(profile) & (profile > 0))[0]
    if known.size < min_separation + 1:
        return None
    best: tuple[float, int, int] | None = None
    for sharp in known:
        for soft in known:
            if abs(int(sharp) - int(soft)) < min_separation:
                continue
            drop = profile[soft] / profile[sharp] - 1.0
            if best is None or drop > best[0]:
                best = (float(drop), int(sharp), int(soft))
    return best


def zonal_defocus(gray: np.ndarray, grid: Grid, axis: str = "rows", **kwargs) -> ZonalResult | None:
    """Оценивает зональный расфокус кадра.

    Args:
        gray: Полутоновый кадр.
        grid: Основная сетка тайлов (внутри используется вдвое более мелкая).
        axis: Ось профиля: "rows" (по умолчанию) или "cols".
        **kwargs: Параметры отбора полос, см. ``band_profile``.

    Returns:
        ``ZonalResult`` либо None, если измеримых полос не набралось (обложка,
        пустой лист, кадр без текста).
    """
    fine = fine_grid(gray.shape, grid)
    stats = edge_stats(gray, fine)
    profile = _smooth(
        band_profile(stats["sigma"], stats["count"], axis=axis, run=stats["run"], paper=paper_map(gray, fine), **kwargs)
    )
    found = profile_drop(profile)
    if found is None:
        return None
    drop, best, worst = found
    return ZonalResult(
        drop=drop, best=best, worst=worst, n_bands=profile.size, axis=axis, profile=[float(v) for v in profile]
    )
