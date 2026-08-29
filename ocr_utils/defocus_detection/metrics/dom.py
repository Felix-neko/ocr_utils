"""DOM (Difference of Median) — резкость края, нормированная на контраст.

Kumar, Chen, Doermann, «Sharpness Estimation for Document and Scene Images», ICPR 2012.

ПОЧЕМУ ИМЕННО ЭТА МЕТРИКА В НАБОРЕ. Она единственная из общепринятых спроектирована не
под фотографии, а под ДОКУМЕНТЫ, и устроена ровно под нашу главную беду: числитель
(насколько «ступенькой» выглядит переход) делится на знаменатель (полная вариация яркости
на том же участке). Контраст входит в обе части и сокращается, поэтому балл почти не
зависит ни от экспозиции, ни от того, насколько густо набрана полоса, — а освещение у нас
прыгает даже внутри одной подшивки.

КАК УСТРОЕНО. Резкий переход тёмное→светлое на медианно отфильтрованном изображении даёт
большую вторую разность через пиксель (``I[i+2] - 2·I[i] + I[i-2]``): яркость меняется
скачком. Размытый переход ту же разницу яркостей растягивает на несколько пикселей, вторая
разность падает, а сумма модулей первых разностей — нет, она равна полному перепаду в
любом случае. Отношение и есть мера «скачкообразности».

ПРО МЕДИАННЫЙ ФИЛЬТР. Он тут не косметика: вторая разность — операция, усиливающая шум
вчетверо, и на зерне высокого ISO она даёт отклик, неотличимый от края. Медиана 3×3 зерно
убирает, а ступеньку сохраняет, поскольку не смешивает значения по разные стороны от неё.
"""

import cv2
import numpy as np

from ocr_utils.defocus_detection.metrics.base import Algorithm
from ocr_utils.defocus_detection.tiles import Grid

# Полуширина окна, по которому суммируются числитель и знаменатель вокруг края.
# Два пикселя — как в статье: столько занимает переход у резкого штриха с учётом
# пиксельной апертуры.
DEFAULT_WINDOW = 2
# Край засчитывается, если модуль первой разности превышает эту долю от p95 первых
# разностей кадра. Порог относительный по той же причине, что и у edge_width: уровни
# гуляют от экспозиции, а типографский контраст относительно них стабилен.
DEFAULT_EDGE_REL = 0.25
DEFAULT_EDGE_MIN = 4.0
# Минимум краёв в тайле, иначе балл не считается: на десятке переходов отношение сумм — шум.
DEFAULT_MIN_EDGES = 200

# Доля кадра по каждой стороне, по которой берутся опорные уровни при выравнивании.
# Центральные три четверти: края полосы — это поля, тень от держателя и фон стола, их
# яркость к типографике отношения не имеет и опорные точки бы утащила.
EQUALIZE_CROP = 0.75
# Перцентили, которые растягиваются на весь диапазон: бумага и краска. Не min/max —
# те определяются единичными выбросами (блик, чёрная плашка).
EQUALIZE_LO, EQUALIZE_HI = 2.0, 98.0


def equalize_levels(gray: np.ndarray, crop: float = EQUALIZE_CROP) -> np.ndarray:
    """Приводит уровни кадра к общей шкале по центральной части.

    ЗАЧЕМ. Формула DOM — отношение двух сумм, линейных по контрасту, поэтому сам контраст
    в ней сокращается. А вот ОТБОР краёв — нет: у него есть абсолютный нижний порог в
    уровнях 8 бит, и на тёмном или вялом кадре в замер попадает другая популяция
    переходов. Отсюда и берётся замеренная связь балла с яркостью (+0.32), которой по
    построению быть не должно.

    Опорные уровни считаются по центральным 75 % кадра: по краям полосы лежат поля, тень
    от держателя и стол, и их яркость к типографике отношения не имеет.

    Args:
        gray: Полутоновый кадр.
        crop: Доля стороны кадра, по которой берутся опорные уровни.

    Returns:
        Кадр uint8 с растянутыми уровнями; при вырожденном диапазоне — исходный.
    """
    height, width = gray.shape[:2]
    dy, dx = int(height * (1.0 - crop) / 2), int(width * (1.0 - crop) / 2)
    centre = gray[dy : height - dy, dx : width - dx]
    lo, hi = np.percentile(centre, [EQUALIZE_LO, EQUALIZE_HI])
    if hi - lo < 8.0:
        return gray
    scaled = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def _axis_maps(img: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Считает по одной оси числитель, знаменатель и модуль первой разности.

    Обе суммы берутся скользящим окном через свёртку прямоугольным ядром — так они
    считаются за один проход на весь кадр, без питоновских циклов по краям.

    Args:
        img: Полутоновый кадр (float32), ось — X.
        window: Полуширина окна суммирования в пикселях.

    Returns:
        Кортеж (сумма |DoM| в окне, сумма |первых разностей| в окне, |первая разность|).
        Все массивы имеют форму кадра.
    """
    # Вторая разность ЧЕРЕЗ пиксель: сравниваются точки, отстоящие на 2, поэтому одиночный
    # выброс в соседнем пикселе на неё не влияет, а настоящая ступенька — влияет.
    dom = np.zeros_like(img)
    dom[:, 2:-2] = np.abs(img[:, 4:] - 2.0 * img[:, 2:-2] + img[:, :-4])

    first = np.zeros_like(img)
    first[:, 1:] = np.abs(np.diff(img, axis=1))

    kernel = np.ones((1, 2 * window + 1), dtype=np.float32)
    dom_sum = cv2.filter2D(dom, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    first_sum = cv2.filter2D(first, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return dom_sum, first_sum, first


def sharpness_maps(
    gray: np.ndarray,
    window: int = DEFAULT_WINDOW,
    edge_rel: float = DEFAULT_EDGE_REL,
    edge_min: float = DEFAULT_EDGE_MIN,
) -> tuple[np.ndarray, np.ndarray]:
    """Попиксельная карта резкости DOM и маска краевых пикселей.

    Args:
        gray: Полутоновый кадр.
        window: Полуширина окна суммирования.
        edge_rel: Доля от p95 первых разностей кадра, ниже которой пиксель не край.
        edge_min: Абсолютный минимум первой разности в уровнях 8 бит.

    Returns:
        Кортеж (карта резкости, маска краёв). Вне маски значения карты бессмысленны.
    """
    img = cv2.medianBlur(gray, 3).astype(np.float32)

    dom_x, first_x, grad_x = _axis_maps(img, window)
    img_t = np.ascontiguousarray(img.T)
    dom_y, first_y, grad_y = (a.T for a in _axis_maps(img_t, window))

    gradient = np.maximum(grad_x, grad_y)
    threshold = max(edge_min, edge_rel * float(np.percentile(gradient[::4, ::4], 95)))
    edges = gradient >= threshold

    with np.errstate(invalid="ignore", divide="ignore"):
        # Числитель и знаменатель складываются по обеим осям до деления, а не после:
        # так вклад оси пропорционален тому, сколько перепада на ней реально есть, и
        # горизонтальный штрих не портит замер вертикального.
        sharpness = (dom_x + dom_y) / np.maximum(first_x + first_y, 1e-6)
    return sharpness, edges


def _tile_sharpness_equalized(gray: np.ndarray, grid: Grid) -> np.ndarray:
    """Карта DOM по тайлам после выравнивания уровней.

    Args:
        gray: Полутоновый кадр.
        grid: Сетка тайлов.

    Returns:
        Массив (ny, nx); NaN там, где краёв не хватило.
    """
    return _tile_sharpness(equalize_levels(gray), grid)


def _tile_sharpness(gray: np.ndarray, grid: Grid) -> np.ndarray:
    """Карта DOM по тайлам (больше = резче).

    Args:
        gray: Полутоновый кадр.
        grid: Сетка тайлов.

    Returns:
        Массив (ny, nx); NaN там, где краёв не хватило.
    """
    sharpness, edges = sharpness_maps(gray)
    out = np.full((grid.ny, grid.nx), np.nan)
    for iy in range(grid.ny):
        for ix in range(grid.nx):
            y1, y2, x1, x2 = grid.bounds(iy, ix)
            mask = edges[y1:y2, x1:x2]
            count = int(mask.sum())
            if count < DEFAULT_MIN_EDGES:
                continue
            out[iy, ix] = float(sharpness[y1:y2, x1:x2][mask].mean())
    return out


def _region_sharpness(crop: np.ndarray, context: object) -> tuple[float, float]:
    """Резкость одного куска строки: средний DOM по краям и число краёв как вес.

    Порог краёв берётся из контекста кадра: в кропе одной строки p95 градиента — это
    разброс внутри букв, а не контраст типографики, и порог получился бы у каждого
    куска свой.

    Args:
        crop: Полутоновый кусок строки.
        context: Готовый порог градиента (float) либо None.

    Returns:
        Пара (балл, число краевых пикселей); вес 0 означает «не измерено».
    """
    if min(crop.shape[:2]) < 5:
        return float("nan"), 0.0
    threshold = float(context) if context is not None else DEFAULT_EDGE_MIN
    sharpness, _ = sharpness_maps(crop, edge_rel=0.0, edge_min=threshold)
    img = cv2.medianBlur(crop, 3).astype(np.float32)
    grad = np.zeros_like(img)
    grad[:, 1:] = np.abs(np.diff(img, axis=1))
    grad[1:, :] = np.maximum(grad[1:, :], np.abs(np.diff(img, axis=0)))
    mask = grad >= threshold
    count = int(mask.sum())
    if count == 0:
        return float("nan"), 0.0
    return float(sharpness[mask].mean()), float(count)


def gradient_threshold(gray: np.ndarray) -> float:
    """Порог «это край» по всему кадру — контекст для замеров по кускам строк.

    Args:
        gray: Полутоновый кадр.

    Returns:
        Порог первой разности в уровнях 8 бит.
    """
    img = cv2.medianBlur(gray, 3).astype(np.float32)
    grad = np.zeros_like(img)
    grad[:, 1:] = np.abs(np.diff(img, axis=1))
    grad[1:, :] = np.maximum(grad[1:, :], np.abs(np.diff(img, axis=0)))
    return max(DEFAULT_EDGE_MIN, DEFAULT_EDGE_REL * float(np.percentile(grad[::4, ::4], 95)))


ALGORITHM_EQUALIZED = Algorithm(
    name="dom_eq",
    summary="DOM с выравниванием уровней по центру кадра — попытка снять остаточную связь с освещением",
    tile_sharpness=_tile_sharpness_equalized,
    unit="DOM",
    region_sharpness=lambda crop, ctx: _region_sharpness(equalize_levels(crop), ctx),
    frame_context=lambda gray: gradient_threshold(equalize_levels(gray)),
)


ALGORITHM = Algorithm(
    name="dom",
    summary="DOM (Kumar-Chen-Doermann): резкость края, делённая на контраст — сделана под документы",
    tile_sharpness=_tile_sharpness,
    unit="DOM",
    region_sharpness=_region_sharpness,
    frame_context=gradient_threshold,
)
