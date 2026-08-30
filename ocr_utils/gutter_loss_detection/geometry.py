"""Разбор разворота: линия сгиба, строки, внутренние поля обеих полос.

Считается один раз на кадр и служит всем метрикам сразу.

ПОЧЕМУ ТАК, А НЕ ПРОЩЕ. Три вещи здесь выглядят избыточными, но без каждой из них
метрика на реальном паке рассыпается — проверено на размеченной папке «1926/08»:

1. **Маска краски — полосовой отклик, а не порог по уровню бумаги.** У корешка бумага
   уходит в тень (у «Планового хозяйства» 253 → 200 на полусотне пикселей), и любой
   порог вида «темнее доли локального фона» записывает саму тень в краску. Внутреннее
   поле тогда всюду выходит нулевым, и помеченные кадры от чистых не отличаются вовсе.
   Полосовой отклик (DoG на масштабе штриха) на гладкую тень не реагирует.

2. **Из маски вычитаются длинные вертикали и горизонтали.** Это края страниц (тёмный
   след сгиба) и линейки таблиц. Обе структуры лежат ровно там, где мы меряем поле, и
   обе не текст: с ними коридор у корешка «зарастает» и снова всё сливается.

3. **Сгиб — НАКЛОННАЯ ПРЯМАЯ, и поля меряются построчно.** Съёмка с рук даёт поворот
   разворота порядка 15 px по высоте кадра. Профиль краски, усреднённый по всем строкам,
   размазывает этим поворотом коридор шириной в те же десятки пикселей — измеренное
   качество ранжирования падает с AUC 0.88 до 0.5, то есть до подбрасывания монеты.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import cv2

Image.MAX_IMAGE_PIXELS = None

# Рабочая ширина кадра. Признак крупный (десятки пикселей на полном разрешении), и
# JPEG умеет отдавать уменьшённое изображение прямо из декодера — на папке в полторы
# сотни кадров это разница между секундами и минутами.
WORK_WIDTH = 2000

# Масштабы DoG для маски краски, в пикселях РАБОЧЕГО кадра: примерно полуштрих и
# примерно расстояние между штрихами корпусного набора.
DOG_FINE, DOG_COARSE = 0.8, 2.6

# Насколько отклик должен быть глубже уровня бумаги, в её долях.
INK_LEVEL = 0.055

# Длина, с которой вертикаль/горизонталь считается не текстом, а краем или линейкой.
RULE_HEIGHT, RULE_WIDTH = 41, 121

# Полоса поиска сгиба вокруг центра кадра и вокруг грубой оценки, в долях ширины и в px.
SEARCH_BAND = 0.32
FOLD_BAND = 130

# Отбраковка точек при подгонке прямой сгиба, в пикселях рабочего кадра.
FOLD_OUTLIER_COARSE, FOLD_OUTLIER_FINE = 60, 25

# Полоса кадра по вертикали, внутри которой лежит наборная полоса (без колонтитулов
# и полей), в долях высоты.
BODY_TOP, BODY_BOTTOM = 0.12, 0.90

# Порог строки в долях 95-го процентиля профиля краски по строкам.
LINE_LEVEL = 0.22

# Насколько далеко от сгиба вообще искать конец строки, в долях ширины кадра. Дальше
# начинается наружное поле, и «конец строки» там означал бы пустую полосу.
REACH = 0.21

# Ниже этого отношения ширины к высоте кадр считается одиночной страницей и не мерится.
SPREAD_MIN_ASPECT = 1.15

# Минимум строк, при котором замер вообще осмыслен.
MIN_LINES = 10

LEFT, RIGHT = "L", "R"


@dataclass(frozen=True)
class SideGeometry:
    """Внутреннее поле одной полосы разворота.

    Attributes:
        side: "L" или "R".
        margins: Расстояния от сгиба до конца (начала) каждой строки, в шагах строк.
        rules_v: Сколько длинных вертикальных линеек найдено у корешка.
        rules_h: Сколько длинных горизонтальных линеек найдено у корешка.
    """

    side: str
    margins: np.ndarray
    rules_v: int
    rules_h: int

    @property
    def tight(self) -> float:
        """Поле по самым тесным строкам (10-й процентиль), в шагах строк."""
        return float(np.percentile(self.margins, 10)) if self.margins.size else float("nan")

    @property
    def median(self) -> float:
        """Поле по медианной строке, в шагах строк."""
        return float(np.median(self.margins)) if self.margins.size else float("nan")


@dataclass(frozen=True)
class SpreadGeometry:
    """Разбор кадра-разворота.

    Attributes:
        pitch: Шаг строк в пикселях рабочего кадра — единица длины для всех полей.
        lines: Сколько строк найдено.
        tilt: Наклон сгиба, пикселей на всю высоту кадра.
        fold_at_middle: Столбец сгиба на середине высоты, в рабочем масштабе.
        scale: Во сколько раз рабочий кадр меньше исходного.
        sides: Разбор полос.
        problem: Почему кадр измерить нельзя; пустая строка — измерен.
    """

    pitch: float = float("nan")
    lines: int = 0
    tilt: float = 0.0
    fold_at_middle: float = 0.0
    scale: float = 1.0
    sides: tuple[SideGeometry, ...] = ()
    problem: str = ""


def read_work_gray(path: Path) -> tuple[np.ndarray, float]:
    """Читает кадр в полутон рабочего масштаба.

    Args:
        path: Путь к файлу.

    Returns:
        Пара (полутоновый кадр float32, во сколько раз он меньше исходного).
    """
    with Image.open(path) as im:
        full_w = im.width
        im.draft("L", (WORK_WIDTH, max(1, WORK_WIDTH * im.height // max(1, im.width))))
        gray = np.asarray(im.convert("L")).astype(np.float32)
    return gray, full_w / max(1, gray.shape[1])


def build_masks(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Маска краски и отдельно длинные вертикали и горизонтали.

    Args:
        gray: Полутоновый кадр рабочего масштаба.

    Returns:
        Тройка (краска без линеек и краёв, длинные вертикали, длинные горизонтали).
    """
    paper = cv2.GaussianBlur(gray, (0, 0), 6)
    response = cv2.GaussianBlur(gray, (0, 0), DOG_FINE) - cv2.GaussianBlur(gray, (0, 0), DOG_COARSE)
    raw = (response < -INK_LEVEL * np.maximum(paper, 1.0)).astype(np.uint8)
    vertical = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((RULE_HEIGHT, 1), np.uint8))
    horizontal = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((1, RULE_WIDTH), np.uint8))
    linear = cv2.dilate(vertical | horizontal, np.ones((3, 5), np.uint8))
    return (raw & (1 - linear)).astype(np.uint8), vertical, horizontal


def fit_fold(gray: np.ndarray, ink: np.ndarray) -> tuple[np.ndarray, float]:
    """Подгоняет прямую сгиба по тёмному следу между полосами.

    Args:
        gray: Полутоновый кадр рабочего масштаба.
        ink: Маска краски.

    Returns:
        Пара (коэффициенты прямой x = k*y + b, наклон в пикселях на высоту кадра).
    """
    height, width = gray.shape
    profile = ink[int(height * 0.15) : int(height * 0.85)].mean(axis=0)
    profile = cv2.blur(profile.reshape(1, -1), (31, 1)).ravel()
    lo = int(width * (0.5 - SEARCH_BAND / 2))
    hi = int(width * (0.5 + SEARCH_BAND / 2))
    coarse = lo + int(np.argmin(profile[lo:hi]))

    blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)
    step = max(8, height // 200)
    ys, xs = [], []
    left, right = max(0, coarse - FOLD_BAND), min(width, coarse + FOLD_BAND)
    for y in range(int(height * BODY_TOP), int(height * BODY_BOTTOM), step):
        ys.append(y + step / 2)
        xs.append(left + int(np.argmin(blurred[y : y + step, left:right].mean(axis=0))))
    ys, xs = np.array(ys, float), np.array(xs, float)

    keep = np.abs(xs - np.median(xs)) < FOLD_OUTLIER_COARSE
    if keep.sum() < 8:
        return np.array([0.0, float(coarse)]), 0.0
    line = np.polyfit(ys[keep], xs[keep], 1)
    keep &= np.abs(xs - np.polyval(line, ys)) < FOLD_OUTLIER_FINE
    if keep.sum() >= 8:
        line = np.polyfit(ys[keep], xs[keep], 1)
    return line, float(line[0] * height)


def find_lines(ink: np.ndarray, x0: int, x1: int) -> list[tuple[int, int]]:
    """Полосы строк по профилю краски в столбцах [x0, x1).

    Args:
        ink: Маска краски.
        x0: Левая граница окна.
        x1: Правая граница окна.

    Returns:
        Список пар (верх, низ) в пикселях рабочего кадра.
    """
    height = ink.shape[0]
    rows = ink[:, x0:x1].sum(axis=1).reshape(-1, 1).astype(np.float32)
    rows = cv2.blur(rows, (1, 3)).ravel()
    level = LINE_LEVEL * float(np.percentile(rows, 95))
    bands, inside, start = [], False, 0
    for i, value in enumerate(rows):
        if not inside and value > level:
            inside, start = True, i
        elif inside and value <= level:
            inside = False
            if i - start >= 5:
                bands.append((start, i))
    return [b for b in bands if int(height * BODY_TOP) < b[0] < int(height * BODY_BOTTOM)]


def _side_margins(ink, line, bands, pitch, side, width) -> np.ndarray:
    """Расстояния от сгиба до внутреннего конца каждой строки, в шагах строк."""
    reach = int(width * REACH)
    out = []
    for top, bottom in bands:
        fold = int(round(np.polyval(line, (top + bottom) / 2)))
        if side == LEFT:
            x0, x1 = max(0, fold - reach), max(1, fold - 2)
            columns = np.flatnonzero(ink[top:bottom, x0:x1].sum(axis=0) > 0)
            if columns.size:
                out.append((x1 - (x0 + int(columns[-1]))) / pitch)
        else:
            x0, x1 = min(width - 2, fold + 3), min(width - 1, fold + 3 + reach)
            columns = np.flatnonzero(ink[top:bottom, x0:x1].sum(axis=0) > 0)
            if columns.size:
                out.append(int(columns[0]) / pitch)
    return np.array(out, float)


def _side_rules(vertical, horizontal, line, side, shape, pitch) -> tuple[int, int]:
    """Сколько длинных линеек лежит в приосевой зоне полосы (признак таблицы)."""
    height, width = shape
    fold = int(round(np.polyval(line, height / 2)))
    reach = int(width * REACH)
    if side == LEFT:
        x0, x1 = max(0, fold - reach), max(1, fold - 15)
    else:
        x0, x1 = min(width - 2, fold + 15), min(width - 1, fold + reach)
    y0, y1 = int(height * BODY_TOP), int(height * BODY_BOTTOM)
    columns = vertical[y0:y1, x0:x1].sum(axis=0)
    rows = horizontal[y0:y1, x0:x1].sum(axis=1)
    return int((columns > 6 * pitch).sum()), int((rows > 0.4 * (x1 - x0)).sum())


def analyze_spread(gray: np.ndarray, scale: float = 1.0) -> SpreadGeometry:
    """Разбирает кадр-разворот: сгиб, строки, внутренние поля обеих полос.

    Args:
        gray: Полутоновый кадр рабочего масштаба.
        scale: Во сколько раз рабочий кадр меньше исходного (для отчёта).

    Returns:
        ``SpreadGeometry``; при непригодном кадре заполнено поле ``problem``.
    """
    height, width = gray.shape[:2]
    if width < SPREAD_MIN_ASPECT * height:
        return SpreadGeometry(scale=scale, problem="одиночная страница")

    ink, vertical, horizontal = build_masks(gray)
    line, tilt = fit_fold(gray, ink)
    fold = float(np.polyval(line, height / 2))

    # Строки ищутся по той полосе, где их больше: на развороте одна страница бывает
    # почти пустой (конец главы, вклейка), и шаг строк по ней не определить.
    candidates = [
        find_lines(ink, int(width * 0.10), max(1, int(fold - width * 0.06))),
        find_lines(ink, min(width - 2, int(fold + width * 0.06)), int(width * 0.90)),
    ]
    bands = max(candidates, key=len)
    if len(bands) < MIN_LINES:
        return SpreadGeometry(scale=scale, problem="мало строк")
    centres = np.array([(a + b) / 2 for a, b in bands])
    pitch = float(np.median(np.diff(centres)))
    if not np.isfinite(pitch) or pitch <= 1:
        return SpreadGeometry(scale=scale, problem="не найден шаг строк")

    sides = []
    for side in (LEFT, RIGHT):
        margins = _side_margins(ink, line, bands, pitch, side, width)
        rules_v, rules_h = _side_rules(vertical, horizontal, line, side, (height, width), pitch)
        sides.append(SideGeometry(side=side, margins=margins, rules_v=rules_v, rules_h=rules_h))
    return SpreadGeometry(
        pitch=pitch, lines=len(bands), tilt=tilt, fold_at_middle=fold, scale=scale, sides=tuple(sides)
    )
