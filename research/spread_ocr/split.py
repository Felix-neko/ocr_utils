"""Разрезание фото-разворота на левую и правую полосы: маска бумаги по цвету, сгиб по медиане яркости столбцов, выравнивание наклона."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Масштаб, на котором ищутся бумага и сгиб (кадры 4600–5700 px по ширине → ~1100–1400 px).
SEARCH_SCALE = 0.25
# Бумага: яркая и не красная насыщенная; фон — чёрный мат (тёмный) или красная ткань (насыщенный красный).
# Синяя обложка выпуска — тоже «бумага»: по насыщенности её не отсечь от фона, только по оттенку.
PAPER_MIN_VALUE = 110
RED_MAX_SATURATION = 95
# Красный оттенок в HSV OpenCV (0..180): у нуля и у 180.
RED_HUE_LOW, RED_HUE_HIGH = 12, 165
# Сгиб ищется в средней полосе кадра, доли ширины бумаги.
FOLD_SEARCH_FROM, FOLD_SEARCH_TO = 0.35, 0.65
# Окно сглаживания профиля столбцов, px масштаба поиска.
PROFILE_SMOOTH = 5
# Поле, оставляемое у сгиба при разрезании, px полного кадра: часть строк уходит в сгиб, их лучше не терять.
FOLD_MARGIN = 40
# Запас за сгибом при дочитывании номера левой полосы, px полного кадра: номер стоит у сгиба и уходит в него.
NUMBER_FOLD_REACH = 250
# Предельный угол выравнивания наклона полосы, градусы.
MAX_SKEW_DEG = 8.0


@dataclass(frozen=True)
class SpreadSplit:
    """Геометрия разворота в px полного кадра.

    Attributes:
        fold_x: Абсцисса сгиба.
        top, bottom: Верх и низ бумаги по вертикали.
        left, right: Левый и правый край бумаги.
    """

    fold_x: int
    top: int
    bottom: int
    left: int
    right: int


def paper_mask(bgr: np.ndarray) -> np.ndarray:
    """Маска бумаги на уменьшенном кадре.

    Args:
        bgr: Кадр BGR uint8 любого масштаба.

    Returns:
        Булева маска той же высоты и ширины: True — пиксель похож на бумагу (яркий, не насыщенно-красный),
        сглаженная морфологическим закрытием, чтобы строки текста не дырявили маску.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0]
    red = ((hue <= RED_HUE_LOW) | (hue >= RED_HUE_HIGH)) & (hsv[..., 1] > RED_MAX_SATURATION)
    mask = (hsv[..., 2] >= PAPER_MIN_VALUE) & ~red
    # Закрытие: текст (тёмные буквы) внутри полосы не должен выпадать из бумаги.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    # Открытие убирает мелкие блики фона.
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened.astype(bool)


def find_split(bgr: np.ndarray) -> SpreadSplit:
    """Найти сгиб и рамку бумаги разворота.

    Сгиб — столбец средней полосы бумаги с наименьшей медианой яркости по строкам бумаги: медиана
    не чувствительна к буквам (их меньше половины пикселей столбца), а тень сгиба темнит весь столбец.

    Args:
        bgr: Полный кадр BGR uint8.

    Returns:
        ``SpreadSplit`` в px полного кадра.
    """
    small = cv2.resize(bgr, None, fx=SEARCH_SCALE, fy=SEARCH_SCALE, interpolation=cv2.INTER_AREA)
    mask = paper_mask(small)
    height, width = mask.shape
    # Рамка бумаги — по столбцам и строкам, где бумаги заметная доля (отсекает пальцы и блики).
    columns = np.flatnonzero(mask.mean(axis=0) > 0.25)
    rows = np.flatnonzero(mask.mean(axis=1) > 0.25)
    left, right = (int(columns[0]), int(columns[-1])) if columns.size else (0, width - 1)
    top, bottom = (int(rows[0]), int(rows[-1])) if rows.size else (0, height - 1)
    # Профиль столбцов: медиана серого по средней части высоты бумаги (без краёв, где загиб и тень).
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    span = bottom - top
    band = gray[top + span // 8 : bottom - span // 8, :]
    profile = np.median(band, axis=0).astype(np.float32)
    profile = cv2.blur(profile.reshape(1, -1), (PROFILE_SMOOTH, 1)).ravel()
    lo = left + int(FOLD_SEARCH_FROM * (right - left))
    hi = left + int(FOLD_SEARCH_TO * (right - left))
    fold = lo + int(np.argmin(profile[lo:hi])) if hi > lo else (left + right) // 2
    scale = 1.0 / SEARCH_SCALE
    return SpreadSplit(
        fold_x=int(fold * scale),
        top=int(top * scale),
        bottom=int(bottom * scale),
        left=int(left * scale),
        right=int(right * scale),
    )


def deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Выровнять небольшой наклон полосы по горизонтальной проекции краски.

    Перебирает углы от ``-MAX_SKEW_DEG`` до ``+MAX_SKEW_DEG`` (на уменьшенной копии) и берёт тот, при котором
    дисперсия построчной суммы краски максимальна — строки текста ложатся горизонтально.

    Args:
        gray: Полутоновая полоса uint8.

    Returns:
        Кортеж (повёрнутая полоса той же формы, угол в градусах).
    """
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    # Краска — адаптивная бинаризация, чтобы неровное освещение не влияло на проекцию.
    ink = cv2.adaptiveThreshold(small, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    center = (small.shape[1] / 2, small.shape[0] / 2)
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-MAX_SKEW_DEG, MAX_SKEW_DEG + 1e-6, 0.25):
        matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(ink, matrix, (small.shape[1], small.shape[0]), flags=cv2.INTER_NEAREST)
        score = float(np.var(rotated.sum(axis=1)))
        if score > best_score:
            best_angle, best_score = float(angle), score
    if abs(best_angle) < 0.1:
        return gray, 0.0
    matrix = cv2.getRotationMatrix2D((gray.shape[1] / 2, gray.shape[0] / 2), best_angle, 1.0)
    rotated = cv2.warpAffine(
        gray, matrix, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, best_angle


def page_crops(bgr: np.ndarray, split: SpreadSplit) -> tuple[np.ndarray, np.ndarray]:
    """Вырезать левую и правую полосы разворота в полутоне.

    Args:
        bgr: Полный кадр BGR uint8.
        split: Геометрия разворота из ``find_split``.

    Returns:
        Кортеж (левая полоса, правая полоса) — полутон uint8; у сгиба оставлено поле ``FOLD_MARGIN``.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    top, bottom = split.top, split.bottom + 1
    left_page = gray[top:bottom, split.left : min(split.fold_x + FOLD_MARGIN, gray.shape[1])]
    right_page = gray[top:bottom, max(split.fold_x - FOLD_MARGIN, 0) : split.right + 1]
    return left_page, right_page
