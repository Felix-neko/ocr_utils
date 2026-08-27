"""Нормировка полосы и зоны замера: где межстрочья, где чистые поля, где краска.

ЗАЧЕМ ЭТОТ МОДУЛЬ ОТДЕЛЬНО. Все метрики просвета меряют одно и то же место — бумагу
МЕЖДУ строками, — и отличаются только тем, что именно они там считают. Разбор полосы
на зоны стоит примерно столько же, сколько сам замер, поэтому он делается один раз,
складывается в ``Zones`` и раздаётся всем метрикам прогона.

ИДЕЯ НОРМИРОВКИ. Работаем не с яркостью, а с ОТРАЖЕНИЕМ ``R = g / bg``, где ``bg`` —
гладкая оценка уровня бумаги (светлое раздувается, потом сильно размывается — та же
«raise above background», что в ScanTailor Equalize Illumination и в
``defocus_detection.scale.ink_map``). У чистой бумаги R ≈ 1 независимо от того, жёлтая
она или серая и как лёг свет. Это прямо отвечает на заявленное требование: цвет бумаги
и освещение гуляют от года к году, и абсолютный уровень серого признаком быть не может.

ПОЧЕМУ В ПОЛНОМ РАЗРЕШЕНИИ. При 300 dpi межстрочный промежуток корпусного набора —
около 20 px, а маску краски приходится раздувать на ~5 px, чтобы в замер не попала
кайма самих букв. На копии, уменьшенной вдвое, раздутие съедает промежуток целиком,
и метрика начинает мерить поля и зерно бумаги: замерено, AUC падает с 0.98 до 0.76.
"""

from dataclasses import dataclass

import numpy as np

import cv2

# Высота кадра, для которой названы все размеры ниже: разворот пака «Плановое хозяйство»
# в 300 dpi. Свой эталон, а не ``scale.REFERENCE_HEIGHT`` (там 2944 — высота JPEG-превью
# из RAF совсем другого пака): числа ниже замерены именно на этих сканах.
REFERENCE_HEIGHT = 2800.0

PAPER_DILATE_PX = 7  # радиус раздутия светлого при оценке уровня бумаги
PAPER_BLUR_PX = 75  # размытие оценки бумаги: крупнее буквы, мельче неровности света

INK_LEVEL = 0.65  # темнее этой доли от бумаги — настоящая краска, а не призрак

DENSITY_BLUR_PX = 75  # окно, в котором считается «сколько тут краски»
BLOCK_DENSITY = 0.03  # плотность краски, с которой начинается наборная полоса
MARGIN_DENSITY = 0.002  # плотность, ниже которой это чистое поле
BLOCK_ERODE_PX = 30  # отступ внутрь наборной полосы: не хватать её край
MARGIN_ERODE_PX = 40  # отступ внутрь поля: не хватать кайму крайних букв

INK_DILATE_PX = 5  # раздутие маски краски: столько занимает кайма буквы при 300 dpi

BAND_SMALL_PX = 1.0  # полосовой фильтр масштаба штриха: нижняя сигма
BAND_LARGE_PX = 4.0  # ...и верхняя; между ними живёт штрих призрака

# Растр (фото, вклейка) даёт в межстрочьях полутон, неотличимый от призрака.
# Константы — из ``background_smoothing.processing``, где тем же способом ищут растр
# для защиты от сглаживания: КРУПНЫЕ сплошные пятна средних тонов, у текста серое
# сидит тонкой каймой и размыкание её убирает.
HALFTONE_DOWNSCALE = 4
HALFTONE_LO, HALFTONE_HI = 0.35, 0.90  # границы «средних тонов» в долях бумаги
HALFTONE_OPEN_PX = 15  # сторона ядра размыкания в пикселях УМЕНЬШЕННОЙ копии
HALFTONE_DILATE_PX = 25  # запас вокруг найденного растра, тоже на уменьшенной копии

# Минимальная доля площади полосы, при которой зона считается пригодной для статистики.
# Меньше — и перцентиль начинает прыгать от кадра к кадру сильнее, чем сам эффект.
MIN_ZONE_FRACTION = 0.004

NO_TEXT = "нет текста"
NO_MARGIN = "нет опорных полей"
NO_GAP = "нет межстрочий"


def scaled(height: int, px_at_reference: float) -> int:
    """Пересчитывает размер, названный для ``REFERENCE_HEIGHT``, под текущую полосу.

    Args:
        height: Высота обрабатываемой полосы в пикселях.
        px_at_reference: Размер в пикселях при эталонной высоте.

    Returns:
        Размер в пикселях текущей полосы, не меньше единицы.
    """
    return max(1, int(round(scaled_float(height, px_at_reference))))


def scaled_float(height: int, px_at_reference: float) -> float:
    """То же, что ``scaled``, но без округления — для сигм гауссова размытия.

    Args:
        height: Высота обрабатываемой полосы в пикселях.
        px_at_reference: Размер в пикселях при эталонной высоте.

    Returns:
        Размер в пикселях текущей полосы.
    """
    return px_at_reference * height / REFERENCE_HEIGHT


def _odd(value: int) -> int:
    """Ближайшее нечётное число не меньше единицы (ядра OpenCV любят нечётную сторону).

    Args:
        value: Желаемая сторона ядра.

    Returns:
        Нечётная сторона.
    """
    return value if value % 2 else value + 1


def _disk(radius: int) -> np.ndarray:
    """Круглый структурный элемент заданного радиуса.

    Args:
        radius: Радиус в пикселях.

    Returns:
        Ядро для морфологии.
    """
    side = _odd(2 * max(1, radius) + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))


def paper_level(gray: np.ndarray) -> np.ndarray:
    """Гладкая оценка уровня бумаги: «сколько было бы яркости, не будь тут краски».

    Args:
        gray: Полутоновая полоса.

    Returns:
        Массив float32 той же формы.
    """
    img = gray.astype(np.float32)
    height = img.shape[0]
    dilated = cv2.dilate(img, _disk(scaled(height, PAPER_DILATE_PX)))
    side = _odd(scaled(height, PAPER_BLUR_PX))
    return cv2.blur(dilated, (side, side))


def reflectance(gray: np.ndarray) -> np.ndarray:
    """Отражение относительно бумаги: бумага ≈ 1.0, краска — заметно ниже.

    Args:
        gray: Полутоновая полоса.

    Returns:
        Массив float32 в диапазоне [0, 1.2].
    """
    img = gray.astype(np.float32)
    return np.clip(img / np.maximum(paper_level(gray), 1.0), 0.0, 1.2)


def halftone_mask(refl: np.ndarray) -> np.ndarray:
    """Маска крупных растровых областей (фото, вклейки), которые надо исключить.

    Растр даёт в межстрочьях ровно тот же полутон, что и просвет с оборота, и без
    этой маски любая иллюстрированная полоса уезжает в топ рейтинга.

    Args:
        refl: Карта отражения.

    Returns:
        Булева маска той же формы: True там, где растр.
    """
    height, width = refl.shape
    size = (max(1, width // HALFTONE_DOWNSCALE), max(1, height // HALFTONE_DOWNSCALE))
    small = cv2.resize(refl, size, interpolation=cv2.INTER_AREA)
    mid = ((small > HALFTONE_LO) & (small < HALFTONE_HI)).astype(np.uint8)
    opened = cv2.morphologyEx(mid, cv2.MORPH_OPEN, np.ones((HALFTONE_OPEN_PX, HALFTONE_OPEN_PX), np.uint8))
    grown = cv2.dilate(opened, np.ones((HALFTONE_DILATE_PX, HALFTONE_DILATE_PX), np.uint8))
    return cv2.resize(grown, (width, height), interpolation=cv2.INTER_NEAREST) > 0


@dataclass
class Zones:
    """Разобранная полоса: карты и маски, общие для всех метрик просвета.

    Attributes:
        reflectance: Отражение относительно бумаги (бумага ≈ 1.0).
        block: Наборная полоса целиком — зона замера ВМЕСТЕ с краской и каймой вокруг
            неё. Нужна метрикам, которым важна связность: отличить призрак, свободно
            лежащий в межстрочье, от краски, дотянувшейся туда своим краем, можно только
            на маске, где сама краска ещё присутствует.
        otsu: Порог Оцу по наборной полосе, в долях уровня бумаги. Это модель
            предварительной бинаризации: именно ею решается, станет призрак краской
            или уйдёт в фон.
        dark: Полосовой отклик на тонких ТЁМНЫХ структурах масштаба штриха.
            Гладкие пятна (лисьи, тень у корешка, неровность света) в него не попадают —
            именно поэтому метрика не путает грязную бумагу с просвечивающей.
        ink: Ядро настоящей краски лицевой стороны.
        ink_grown: Она же, раздутая на кайму: краска на грубой бумаге расплывается,
            и без запаса её край попадал бы в замер как призрак.
        gap: Маска замера — бумага внутри наборной полосы: межстрочья и просветы
            между словами, за вычетом раздутой краски и растра.
        margin: Опорная маска — чистые поля полосы. Наборные полосы лицевой и
            оборотной сторон в журнале совпадают, поэтому на полях призрака нет,
            и они дают уровень зерна бумаги и шума JPEG именно этого кадра.
        paper: Медиана отражения по маске замера.
        problem: Почему полосу нельзя измерить ВООБЩЕ; пустая строка — всё в порядке.
        note: Оговорка к замеру, не отменяющая его: сейчас единственная — «нет опорных
            полей» (полоса под обрез, таблица во всю ширину). Метрика, которой поля
            нужны, на такой полосе не считается, а балл берётся у запасной; в отчёте
            оговорка печатается рядом с баллом.
    """

    reflectance: np.ndarray
    dark: np.ndarray
    ink: np.ndarray
    ink_grown: np.ndarray
    block: np.ndarray
    gap: np.ndarray
    margin: np.ndarray
    paper: float
    otsu: float = float("nan")
    problem: str = ""
    note: str = ""

    @property
    def usable(self) -> bool:
        """Годится ли полоса для ранжирования."""
        return not self.problem

    @property
    def has_margin(self) -> bool:
        """Есть ли на полосе чистые поля, годные в опору."""
        return bool(self.margin.size) and bool(self.margin.any())


def otsu_level(refl: np.ndarray, block: np.ndarray) -> float:
    """Порог Оцу по наборной полосе — модель предварительной бинаризации.

    Считается по ОТРАЖЕНИЮ, а не по яркости, поэтому не зависит ни от цвета бумаги,
    ни от экспозиции; и считается только по наборной полосе, чтобы пустые поля не
    перетягивали гистограмму в сторону бумаги.

    Args:
        refl: Карта отражения.
        block: Маска наборной полосы.

    Returns:
        Порог в долях уровня бумаги; NaN, если считать не по чему.
    """
    values = refl[block]
    if values.size < 256:
        return float("nan")
    level, _ = cv2.threshold(
        np.clip(values * 255.0, 0, 255).astype(np.uint8).reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return float(level) / 255.0


def build_zones(gray: np.ndarray) -> Zones:
    """Разбирает полосу на зоны замера.

    Args:
        gray: Полутоновая полоса (одна страница, не разворот).

    Returns:
        Заполненный ``Zones``; при вырожденной полосе — с непустым ``problem``
        и пустыми масками.
    """
    height, width = gray.shape[:2]
    area = float(height * width)
    refl = reflectance(gray)

    ink = (refl < INK_LEVEL).astype(np.uint8)
    density_side = _odd(scaled(height, DENSITY_BLUR_PX))
    density = cv2.blur(ink.astype(np.float32), (density_side, density_side))

    block = cv2.erode((density > BLOCK_DENSITY).astype(np.uint8), _disk(scaled(height, BLOCK_ERODE_PX))) > 0
    margin = cv2.erode((density < MARGIN_DENSITY).astype(np.uint8), _disk(scaled(height, MARGIN_ERODE_PX))) > 0

    grown_ink = cv2.dilate(ink, _disk(scaled(height, INK_DILATE_PX))) > 0
    raster = halftone_mask(refl)

    gap = block & ~grown_ink & ~raster
    margin = margin & ~grown_ink & ~raster

    small = cv2.GaussianBlur(refl, (0, 0), max(0.6, scaled_float(height, BAND_SMALL_PX)))
    large = cv2.GaussianBlur(refl, (0, 0), max(1.2, scaled_float(height, BAND_LARGE_PX)))
    # Знак: у тёмного тонкого штриха слабое размытие сохраняет провал глубже сильного,
    # поэтому «темно и мелко» — это large - small, а не наоборот.
    dark = np.clip(large - small, 0.0, None)

    core_ink = ink > 0
    empty = np.zeros((0,), dtype=bool)
    if not block.any():
        return Zones(refl, dark, core_ink, grown_ink, block, empty, empty, float("nan"), problem=NO_TEXT)
    if gap.sum() < MIN_ZONE_FRACTION * area:
        return Zones(refl, dark, core_ink, grown_ink, block, empty, empty, float("nan"), problem=NO_GAP)

    paper = float(np.median(refl[gap]))
    level = otsu_level(refl, block)
    if margin.sum() < MIN_ZONE_FRACTION * area:
        # Не фатально: полоса измерима, просто без опоры на собственное зерно бумаги.
        margin = np.zeros_like(margin)
        return Zones(refl, dark, core_ink, grown_ink, block, gap, margin, paper, level, note=NO_MARGIN)
    return Zones(refl, dark, core_ink, grown_ink, block, gap, margin, paper, level)
