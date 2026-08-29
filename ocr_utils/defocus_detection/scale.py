"""Масштаб и геометрия набора: шаг строк, локальный наклон, толщина штриха.

ЗАЧЕМ ЭТО НУЖНО. Сырая ширина размытого края в пикселях — величина, на которую нельзя
повесить абсолютный порог: она зависит от разрешения превью, от расстояния до полосы, от
кегля и от формата газеты (А2 против А3 при том же кадре — это вдвое разный кегль в
пикселях). Порог переносится между съёмками только у БЕЗРАЗМЕРНОЙ величины — размытия,
выраженного в долях самого текста. Поэтому нужен измеритель масштаба набора.

ПОЧЕМУ ШАГ СТРОК, А НЕ ВЫСОТА БУКВЫ. Шаг строк — это ПЕРИОД, а свёртка не двигает частоты:
размытие меняет амплитуду спектрального пика, но не его положение. Значит шаг строк можно
мерить прямо на том кадре, резкость которого и оценивается, не боясь, что нормировщик сам
поедет от расфокуса. Высота буквы таким свойством не обладает — размытие её раздувает.
Проверено на выборке СИ: у резкого кадра и у его же расфокусной версии шаг совпадает.

ПОЧЕМУ ПО УЗКИМ ПОЛОСАМ, А НЕ ПО ТАЙЛУ ЦЕЛИКОМ. Профиль краски, усреднённый по всей
ширине тайла, смешивает несколько колонок, заголовки и иллюстрации; периодичность в такой
смеси тонет. По полосе шириной примерно в одну колонку строки лежат ровно, и пик в спектре
получается чистым. Балл кадра — медиана по полосам с самым выраженным пиком.

ПРО НАКЛОН. Камера смотрит на полосу то прямо, то под углом, поэтому строки в кадре
повёрнуты на градус-другой, а сама полоса уходит в трапецию. Наклон приходится оценивать
локально и по двум причинам сразу: без него смазывается профиль проекции (и пик по шагу
строк тупеет), и без него измерение направленной резкости принимает наклон текста за
смаз от движения.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Ширина полосы, по которой считается профиль краски, в долях ширины кадра. 224 px на
# превью 4416 px — это примерно газетная колонка: строки внутри лежат ровно, а
# соседние колонки в замер не подмешиваются.
STRIP_WIDTH_FRACTION = 224.0 / 4416.0
# Высота полосы в долях высоты кадра: должно уложиться заведомо больше десятка строк,
# иначе частоту не на чем оценивать.
STRIP_HEIGHT_FRACTION = 480.0 / 2944.0

# Границы поиска шага строк, в долях высоты кадра. Нижняя отсекает растр и внутрибуквенные
# частоты, верхняя — крупные заголовки и межколонные пустоты. Заданы отношениями, а не
# пикселями, чтобы кадры другого разрешения искали в том же месте страницы.
PITCH_MIN_FRACTION = 1.0 / 210.0
PITCH_MAX_FRACTION = 1.0 / 33.0

# Сигма гауссианы, которой оценивается фон при выделении краски. Больше кегля, но меньше
# межколонника: снимает неравномерность освещения, не трогая сам текст.
ILLUM_SIGMA = 8.0
# Сигма, которой из профиля убирается низкочастотный тренд (плавный уход яркости по
# высоте полосы). Порядка половины ожидаемого шага строк.
DETREND_SIGMA = 12.0

# Обе сигмы выше названы в пикселях превью 4416×2944, но применяться обязаны В ДОЛЯХ
# КАДРА, иначе оценка ломается при смене разрешения. Это не теория: на кадре,
# уменьшенном вдвое, фиксированная ILLUM_SIGMA=8 оказывается соизмерима с самим шагом
# строк, высокочастотный фильтр съедает измеряемую структуру, и вместо шага 12.7 px
# оценка находит 21.8 — то есть промахивается почти вдвое. Ровно этот случай и есть
# «другая газета, формат А3 вместо А2».
REFERENCE_HEIGHT = 2944.0

# Доля лучших полос (по выраженности пика), которые идут в медиану.
TOP_STRIP_FRACTION = 0.2
MIN_STRIPS = 8

# Диапазон и шаг перебора наклона, градусы.
ANGLE_LIMIT = 5.0
ANGLE_STEP = 0.5


@dataclass(frozen=True)
class TextScale:
    """Оценка масштаба набора по кадру.

    Attributes:
        pitch: Шаг строк основного текста в пикселях; NaN, если периодичности не нашлось.
        confidence: Насколько уверенно виден пик: доля энергии профиля, собранная в нём,
            усреднённая по отобранным полосам. Ниже ~0.05 доверять величине не стоит.
        n_strips: Сколько полос участвовало в оценке.
        spread: Межквартильный разброс шага по отобранным полосам, в пикселях. Большой
            разброс означает разнокегельную полосу либо сильную трапецию.
    """

    pitch: float
    confidence: float
    n_strips: int
    spread: float

    @property
    def usable(self) -> bool:
        """Годится ли оценка для нормировки."""
        return bool(np.isfinite(self.pitch)) and self.confidence >= 0.04 and self.n_strips >= MIN_STRIPS


def frame_sigma(height: int, sigma_at_reference: float) -> float:
    """Пересчитывает сигму, названную для превью 4416×2944, под фактический кадр.

    Args:
        height: Высота обрабатываемого кадра в пикселях.
        sigma_at_reference: Значение сигмы для эталонной высоты ``REFERENCE_HEIGHT``.

    Returns:
        Сигма в пикселях текущего кадра, не меньше единицы.
    """
    return max(1.0, sigma_at_reference * height / REFERENCE_HEIGHT)


def ink_map(gray: np.ndarray, illum_sigma: float | None = None) -> np.ndarray:
    """Карта краски: насколько пиксель темнее локального фона.

    Вычитание сильно размытой копии убирает и общий уровень яркости, и её неравномерность
    по кадру — то самое «освещение прыгает», из-за которого нельзя работать с сырой
    яркостью. Остаётся только рисунок краски.

    Args:
        gray: Полутоновый кадр.
        illum_sigma: Сигма гауссова размытия для оценки фона в пикселях; None — взять
            из ``ILLUM_SIGMA`` с пересчётом под высоту кадра.

    Returns:
        Массив float32 той же формы: положительные значения там, где краска.
    """
    img = gray.astype(np.float32)
    if illum_sigma is None:
        illum_sigma = frame_sigma(img.shape[0], ILLUM_SIGMA)
    background = cv2.GaussianBlur(img, (0, 0), illum_sigma)
    return np.clip(background - img, 0.0, None)


def _strip_pitch(strip: np.ndarray, lo: float, hi: float, detrend_sigma: float) -> tuple[float, float]:
    """Шаг строк в одной узкой полосе через спектральный пик профиля краски.

    Args:
        strip: Кусок карты краски (высота, ширина).
        lo: Минимальный правдоподобный шаг, пиксели.
        hi: Максимальный правдоподобный шаг, пиксели.
        detrend_sigma: Сигма снятия низкочастотного тренда, пиксели.

    Returns:
        Пара (шаг в пикселях, доля энергии в пике). NaN и 0.0, если оценить нельзя.
    """
    profile = strip.mean(axis=1)
    if profile.size < 32:
        return float("nan"), 0.0
    # Снимаем плавный тренд: без этого низкие частоты доминируют и пик строк тонет.
    trend = cv2.GaussianBlur(profile.reshape(-1, 1), (0, 0), detrend_sigma).ravel()
    profile = profile - trend
    spectrum = np.abs(np.fft.rfft(profile * np.hanning(profile.size))) ** 2
    freq = np.fft.rfftfreq(profile.size)
    band = (freq > 1.0 / hi) & (freq < 1.0 / lo)
    if not band.any():
        return float("nan"), 0.0
    index = int(np.argmax(np.where(band, spectrum, 0.0)))
    total = float(spectrum[band].sum())
    if total <= 0 or freq[index] <= 0:
        return float("nan"), 0.0
    return float(1.0 / freq[index]), float(spectrum[index] / total)


def text_line_pitch(gray: np.ndarray) -> TextScale:
    """Оценивает шаг строк основного текста по всему кадру.

    Args:
        gray: Полутоновый кадр.

    Returns:
        Оценка масштаба; ``TextScale.usable`` говорит, можно ли ей пользоваться.
    """
    height, width = gray.shape[:2]
    lo = max(6.0, height * PITCH_MIN_FRACTION)
    hi = max(lo + 6.0, height * PITCH_MAX_FRACTION)
    strip_w = max(64, int(round(width * STRIP_WIDTH_FRACTION)))
    strip_h = max(128, int(round(height * STRIP_HEIGHT_FRACTION)))

    ink = ink_map(gray)
    detrend = frame_sigma(height, DETREND_SIGMA)
    found: list[tuple[float, float]] = []
    for y in range(0, height - strip_h + 1, strip_h):
        for x in range(0, width - strip_w + 1, strip_w):
            pitch, confidence = _strip_pitch(ink[y : y + strip_h, x : x + strip_w], lo, hi, detrend)
            if np.isfinite(pitch):
                found.append((confidence, pitch))
    if not found:
        return TextScale(float("nan"), 0.0, 0, float("nan"))

    # Берём только полосы с самым выраженным пиком: там, где текста нет (поля, фото),
    # аргмаксимум спектра — это шум, и усреднять его с настоящими замерами нельзя.
    found.sort(reverse=True)
    keep = max(MIN_STRIPS, int(round(len(found) * TOP_STRIP_FRACTION)))
    top = found[:keep]
    pitches = np.array([p for _, p in top], dtype=np.float64)
    confidences = np.array([c for c, _ in top], dtype=np.float64)
    spread = float(np.percentile(pitches, 75) - np.percentile(pitches, 25))
    return TextScale(
        pitch=float(np.median(pitches)), confidence=float(np.mean(confidences)), n_strips=len(top), spread=spread
    )


def text_angle(tile: np.ndarray, limit: float = ANGLE_LIMIT, step: float = ANGLE_STEP) -> float:
    """Локальный наклон строк — угол, при котором профиль проекции самый контрастный.

    Классический приём оценки перекоса документа: когда строки лежат горизонтально,
    профиль «краска по строкам кадра» состоит из резких пиков и провалов, и его дисперсия
    максимальна; при перекосе строки размазываются друг по другу и профиль сглаживается.

    Args:
        tile: Полутоновый кусок кадра.
        limit: Полуширина диапазона перебора, градусы.
        step: Шаг перебора, градусы.

    Returns:
        Угол в градусах (положительный — против часовой стрелки); 0.0, если оценить
        не удалось.
    """
    if min(tile.shape[:2]) < 64:
        return 0.0
    ink = ink_map(tile)
    height, width = ink.shape
    centre = (width / 2.0, height / 2.0)
    # Поля после поворота обрезаем: там появляются пустые треугольники, которые сами по
    # себе дают дисперсию и смещают максимум.
    margin = max(8, int(round(min(height, width) * 0.05)))
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-limit, limit + 1e-9, step):
        matrix = cv2.getRotationMatrix2D(centre, float(angle), 1.0)
        rotated = cv2.warpAffine(ink, matrix, (width, height), flags=cv2.INTER_LINEAR)
        profile = rotated[margin:-margin, margin:-margin].mean(axis=1)
        score = float(profile.var())
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def stroke_width(gray: np.ndarray, max_run: int = 64) -> float:
    """Толщина штриха — медиана длин тёмных прогонов на уровне половины амплитуды.

    Запасной измеритель масштаба на случай, когда периодичности строк нет: сплошной
    заголовок, таблица, полоса с одной иллюстрацией. Устойчивость к размытию здесь хуже,
    чем у шага строк (симметричная свёртка сохраняет ширину на полувысоте лишь пока
    размытие меньше самого штриха), поэтому это именно запасной вариант.

    Args:
        gray: Полутоновый кадр или его кусок.
        max_run: Прогоны длиннее этого не считаются штрихом (это уже плашка или фон).

    Returns:
        Медианная толщина штриха в пикселях; NaN, если тёмных прогонов не нашлось.
    """
    ink = ink_map(gray)
    peak = float(np.percentile(ink, 99.5))
    if peak <= 0:
        return float("nan")
    mask = ink >= peak * 0.5

    lengths: list[np.ndarray] = []
    for row in mask:
        if not row.any():
            continue
        # Длины серий True: границы находим по изменению значения.
        edges = np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.int8), [0]))))
        runs = edges[1::2] - edges[::2]
        lengths.append(runs)
    if not lengths:
        return float("nan")
    runs = np.concatenate(lengths)
    runs = runs[(runs >= 1) & (runs <= max_run)]
    return float(np.median(runs)) if runs.size else float("nan")
