"""Допечатка срезанной нижней строки по видимым верхушкам букв: колонки и оси строк у кромки, библиотека литер со страницы, раскладка текста по фрагментам с выключкой, вклейка низа букв ниже старой кромки."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz
from scipy import ndimage

from research.cut_lines import synth as S
from research.cut_lines.extend import EDGE_ABOVE, BottomEdge

log = logging.getLogger(__name__)

# Перебор наклона строк колонки, градусы: предел и шаг.
SKEW_MAX = 20.0
SKEW_STEP = 0.5
# Строка считается срезанной (низы на кромке), если низов не выше кромки − EDGE_ABOVE − CUT_TOUCH больше этой доли.
CUT_TOUCH = 4
CUT_SHARE = 0.3
# Глубина полосы под осью при этой проверке (выносные элементы целой строки до кромки не достают), в высотах буквы.
CUT_BAND_DOWN = 0.7
# Срезы колонки для трекинга базовых линий: ширина и шаг, px; допуск подхвата в долях шага строк; сколько срезов
# подряд без линии обрывают строку; минимум точек строки.
SLICE_W = 40
SLICE_STEP = 20
TRACK_TOL = 0.35
TRACK_MISS = 3
MIN_TRACKED = 6
# Столбец полосы строки с краской больше этой доли высоты — не буква (тень корешка), в точки оси не идёт.
SOLID_SHARE = 0.8
# Тень корешка: средняя яркость столбца полосы строки (медиана по SHADOW_MEDIAN столбцам) ниже этого.
SHADOW_GRAY = 110
SHADOW_MEDIAN = 9
# Итераций совместной подгонки формы строк и их сдвигов.
SHAPE_ITERATIONS = 3
# Высота нижнего окна над кромкой (оси строк у кромки) и окон выше (библиотека букв), в высотах буквы.
LOW_UP = 8.0
TALL_UP = 26.0
# Сколько окон высотой TALL_UP над нижним брать для библиотеки букв (выше — своя геометрия строк).
LIBRARY_WINDOWS = 8
# Полоса у кромки, по которой режутся колонки (от кромки вверх), в высотах буквы.
COLUMN_BAND = 1.6
# Разрыв между колонками, в высотах буквы; минимальная ширина колонки.
COLUMN_GAP = 2.5
COLUMN_MIN = 4.0
# Базовая линия ближе этого к кромке — ложная (спад краски на самой кромке), в высотах буквы.
EDGE_BASELINE = 0.3
# Краска ниже выносных последней строки считается верхушками следующей (срезанной) строки, если её больше
# этой доли столбцов отрезка и этого числа пикселей.
NEXT_ROW_INK = 0.02
NEXT_ROW_PX = 30
# Окно поиска буквы по фрагменту: по x (в высотах буквы) и по y (px); минимум видимых строк буквы для поиска.
SEARCH_X = 0.6
SEARCH_Y = 3
MIN_VISIBLE_ROWS = 4
# Порог корреляции, при котором найденное место буквы принимается.
MIN_CORR = 0.35
# Строка — последняя в абзаце (без выключки), если видимая краска кончается раньше правого края колонки
# на столько высот буквы.
RAGGED_GAP = 1.0
# Сходство перечитанной строки с текстом (0..100), ниже — статус ``check``.
MIN_REREAD = 70.0
# Отступ по x от отрезка колонки при выпрямлении полосы, в высотах буквы; запас колонки для видимой краски и стирания.
X_PAD = 1.0
COLUMN_MARGIN = 0.6
# Вклейка низа букв начинается выше старой кромки на это перекрытие, px (над кромкой лист уже перезалит бумагой
# на EDGE_ABOVE, перекрытие прячет шов между видимой частью буквы и допечатанной).
PASTE_OVERLAP = 3
# Верх окна стирания: диапазон поиска низа выносных строки выше, в высотах буквы над базовой линией.
TOP_RANGE = (1.6, 1.0)
# Поверхность бумаги под окном: радиус размытия и перо окна, px; потолок зерна, уровни.
PAPER_SIGMA = 12.0
# Высота полосы для оценки бумаги над базовой линией, в высотах буквы (строка выше и промежутки).
PAPER_UP = 2.8
PAPER_FEATHER = 2.0
GRAIN_MAX = 6.0
# Окно притягивания границы между словами к просвету старой краски, в высотах буквы.
BOUND_SNAP = 0.7
# Если срезано больше этой доли слов строки, набирается вся строка.
WHOLE_LINE_SHARE = 0.5
# Просвет старой краски считается пробелом между словами, если он не уже этого, в высотах буквы.
SPACE_GAP_MIN = 0.2
# Штраф динамики раскладки за отклонение просвета от номинального (внутри слова и между словами), корреляция
# на высоту буквы; штраф за сдвиг начала строки от левого края.
GAP_PENALTY = 1.5
SPACE_PENALTY = 0.3
# Предел общего сдвига номинальной раскладки по x, в высотах буквы.
GLOBAL_SHIFT = 3.0
LEFT_PENALTY = 0.15
# Допуск просвета: сжатие (px) и растяжение (в высотах буквы) относительно номинального — внутри слова и между словами.
GAP_SQUEEZE = 3
GAP_STRETCH = 0.35
SPACE_STRETCH = 1.0
# Пробел между словами не уже этого, в высотах буквы.
SPACE_MIN = 0.3
# Предел сжатия букв по ширине, когда строка не помещается в отрезок.
MIN_SQUEEZE = 0.7
# Латинские буквы, для которых при отсутствии образца берётся кириллический двойник.
LOOKALIKE = {"a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у", "A": "А", "B": "В", "C": "С",
             "E": "Е", "H": "Н", "K": "К", "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х"}  # fmt: skip
# Классы высоты букв для отбора образцов: строчные без надстрочных элементов и высокие знаки.
LOW_LETTERS = set("авгежзиклмнопрстуфхцчшщъыьэюяaceimnorsuvwxz")
TALL_LETTERS = set("АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯЁбйёABCDEFGHIJKLMNOPQRSTUVWXYZbdfhklt0123456789()")
# Буквы с подстрочными элементами (низ краски ниже базовой линии).
DESCENDERS = set("друфцщДЦЩpqgyjQ()")
# Шрифт для латиницы без образца и высота прописной от очка, если прописных образцов нет.
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
)
FONT_CAP = 1.45
# Своя библиотека (колонка, кадр) предпочитается общей, если годных образцов в ней не меньше стольких.
MIN_OWN_SAMPLES = 3
# Сколько лучших образцов латинской буквы проверять tesseract.
VERIFY_CANDIDATES = 5
# Потолок образцов на букву в общей библиотеке подшивки.
LIBRARY_CAP = 60
# Минимум пикселей краски в годном образце буквы.
GLYPH_MIN_INK = 12
# Размер буквы для оценки высоты очка: компоненты краски высотой в этих долях ожидаемой высоты.
XH_RANGE = (0.6, 1.4)


@dataclass(frozen=True)
class Column:
    """Колонка у кромки: отрезок по x в кадре и её текст."""

    x0: int
    x1: int
    text: str


@dataclass
class LineReport:
    """Итог по строке колонки.

    Поля:
        status: ``synth`` / ``check`` (набрано, но перечитывается плохо) / ``visible`` (ни одно слово не доходит
            до кромки — не тронуто) / ``no_axis`` / ``no_text``.
        text: Заданный текст.
        reread: Что прочёл tesseract в набранной строке.
        match: Сходство (0..100).
        missing: Буквы без образца.
        box: Окно изменений ``(x0, y0, x1, y1)``.
    """

    status: str
    text: str = ""
    reread: str = ""
    match: float = 0.0
    missing: str = ""
    box: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class PageCompletion:
    """Итог допечатки страницы: кадр и отчёты по колонкам."""

    image: np.ndarray
    reports: list[LineReport] = field(default_factory=list)


def ink_above_edge(gray: np.ndarray, page: np.ndarray, edge: BottomEdge) -> np.ndarray:
    """Краска страницы выше старой кромки (с отступом :data:`EDGE_ABOVE`).

    Args:
        gray: Серое кадра.
        page: Маска страницы (bool).
        edge: Кромка.

    Returns:
        Маска краски (bool).
    """
    ink = S.ink_of(gray) & page
    rows = np.arange(gray.shape[0])[:, None]
    sub = ink[:, edge.x0 : edge.x1]
    sub &= rows < (edge.y - EDGE_ABOVE)[None, :]
    out = np.zeros_like(ink)
    out[:, edge.x0 : edge.x1] = sub
    return out


def x_height_of(ink: np.ndarray, hint: float) -> float:
    """Высота очка: медиана высот компонент краски размером с букву.

    Args:
        ink: Краска (bool).
        hint: Ожидаемая высота, px.

    Returns:
        Высота, px (``hint``, если компонент мало).
    """
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    hs = stats[1:, cv2.CC_STAT_HEIGHT]
    sel = (hs >= XH_RANGE[0] * hint) & (hs <= XH_RANGE[1] * hint) & (stats[1:, cv2.CC_STAT_AREA] >= 8)
    return float(np.median(hs[sel])) if sel.sum() >= 30 else hint


def bottom_columns(ink: np.ndarray, edge: BottomEdge, x_height: float) -> list[tuple[int, int]]:
    """Колонки нижнего ряда: отрезки по x с краской в полосе :data:`COLUMN_BAND` над кромкой.

    Args:
        ink: Краска выше кромки (bool).
        edge: Кромка.
        x_height: Высота буквы, px.

    Returns:
        Отрезки ``(x0, x1)`` слева направо.
    """
    rows = np.arange(ink.shape[0])[:, None]
    sub = ink[:, edge.x0 : edge.x1] & (rows >= (edge.y - COLUMN_BAND * x_height)[None, :])
    cols = ndimage.binary_closing(sub.any(axis=0), structure=np.ones(int(COLUMN_GAP * x_height)))
    labels, count = ndimage.label(cols)
    out = []
    for i in range(1, count + 1):
        idx = np.nonzero(labels == i)[0]
        if len(idx) >= COLUMN_MIN * x_height:
            out.append((edge.x0 + int(idx[0]), edge.x0 + int(idx[-1]) + 1))
    return out


def assign_columns(found: list[tuple[int, int]], texts: list[str]) -> list[Column]:
    """Сопоставить найденные отрезки колонок текстам набора (слева направо; при расхождении числа —
    самые широкие отрезки).

    Args:
        found: Найденные отрезки.
        texts: Тексты колонок из набора (пустой — колонка не срезана).

    Returns:
        Колонки с текстом (только те, где текст непустой).
    """
    if len(found) != len(texts):
        found = sorted(sorted(found, key=lambda c: c[0] - c[1])[: len(texts)])
    return [Column(a, b, t.strip()) for (a, b), t in zip(found, texts) if t.strip()]


def skew_angle(ink: np.ndarray) -> float:
    """Наклон строк колонки: угол, при котором дисперсия построчной суммы краски максимальна.

    Args:
        ink: Краска колонки (bool).

    Returns:
        Угол, градусы (положительный — строки идут вниз слева направо).
    """
    small = cv2.resize(ink.astype(np.uint8), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    h, w = small.shape
    best, best_var = 0.0, -1.0
    for deg in np.arange(-SKEW_MAX, SKEW_MAX + 1e-6, SKEW_STEP):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
        rot = cv2.warpAffine(small, m, (w, h), flags=cv2.INTER_NEAREST)
        var = float(rot.sum(axis=1).astype(float).var())
        if var > best_var:
            best, best_var = float(deg), var
    return best


def rotate_points(xs: np.ndarray, ys: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Точки через аффинную матрицу 2×3."""
    return m[0, 0] * xs + m[0, 1] * ys + m[0, 2], m[1, 0] * xs + m[1, 1] * ys + m[1, 2]


@dataclass(frozen=True)
class RowGeometry:
    """Строки колонки у кромки.

    Поля:
        axes: Оси полностью видимых строк (сверху вниз), в кадре.
        cut_axis: Ось срезанной строки.
        x_height: Высота буквы, px.
        pitch: Шаг строк, px.
        baseline_visible: Базовая линия срезанной строки видна (срезаны только выносные).
    """

    axes: list[S.Axis]
    cut_axis: S.Axis
    x_height: float
    pitch: float
    baseline_visible: bool


def shadow_columns(gray_band: np.ndarray) -> np.ndarray:
    """Столбцы полосы строки в тени корешка: средняя яркость столбца, сглаженная медианой по соседним столбцам
    (штрихи букв узкие и не проходят), ниже :data:`SHADOW_GRAY`.

    Args:
        gray_band: Серое полосы строки (h, w).

    Returns:
        Маска столбцов (bool).
    """
    means = gray_band.astype(np.float32).mean(axis=0)
    return ndimage.median_filter(means, SHADOW_MEDIAN, mode="nearest") < SHADOW_GRAY


def row_bottom_points(
    ink: np.ndarray, gray: np.ndarray, axis: S.Axis, x_height: float, x0: int, x1: int, down: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """Точки низа краски строки в полосе вдоль оси (0.9 очка над базовой линией — ``down`` очка под), в координатах
    кадра.

    Args:
        ink: Краска кадра (bool).
        gray: Серое кадра uint8 (отсев столбцов в тени корешка).
        axis: Ось строки.
        x_height: Высота буквы, px.
        x0: Левый край отрезка.
        x1: Правый край.
        down: Глубина полосы под базовой линией, в высотах буквы.

    Returns:
        ``(xs, ys)`` — по столбцу с краской одна точка (низ краски).
    """
    up, down = 0.9 * x_height, down * x_height
    maps = S.strip_maps(axis, up, down, x0, x1)
    band = cv2.remap(ink.astype(np.uint8), maps[0], maps[1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=0) > 0  # fmt: skip
    bottom = np.where(band.any(axis=0), band.shape[0] - 1 - np.argmax(band[::-1], axis=0), -1)
    # Столбцы, залитые краской почти целиком, и столбцы в тени корешка — не буквы: низ там упирается в край
    # полосы и тянет ось вниз.
    gray_band = cv2.remap(gray, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    solid = (band.mean(axis=0) > SOLID_SHARE) | shadow_columns(gray_band)
    us = np.nonzero((bottom >= 0) & ~solid)[0]
    xs = maps[0][0, us]
    ys = axis.y(xs) + bottom[us] - int(up)
    return xs.astype(float), ys.astype(float)


def shared_shape(
    rows: list[tuple[np.ndarray, np.ndarray]], x_height: float, x0: int, x1: int
) -> tuple[tuple[np.ndarray, np.ndarray], list[float]] | None:
    """Общая форма строк: одна гладкая кривая ``s(x)`` по точкам всех строк (после вычитания сдвига строки) и
    сдвиг каждой строки (медиана остатков); несколько итераций.

    Args:
        rows: Точки ``(xs, ys)`` каждой строки сверху вниз.
        x_height: Высота буквы, px.
        x0: Левый край выборки кривой.
        x1: Правый край.

    Returns:
        ``(кривая (curve_x, curve_y), сдвиги строк)`` или None, если точек мало.
    """
    rows = [(xs, ys) for xs, ys in rows if len(xs) >= 10]
    if not rows:
        return None
    offsets = [float(np.median(ys)) for _, ys in rows]
    curve = None
    for _ in range(SHAPE_ITERATIONS):
        xs_all = np.concatenate([xs for xs, _ in rows])
        ys_all = np.concatenate([ys - o for (_, ys), o in zip(rows, offsets)])
        curve = S.smooth_curve(xs_all, ys_all, x_height, x0, x1)
        if curve is None:
            return None
        base = S.Axis(x0, x1, np.array([0.0, 0.0]), x_height, curve[0], curve[1])
        offsets = [float(np.median(ys - base.y(xs))) for xs, ys in rows]
    return curve, offsets


def slice_baselines(
    ink: np.ndarray, gray: np.ndarray, x0: int, x1: int, ya: int, yb: int, x_height: float
) -> list[tuple[int, list[int]]]:
    """Базовые линии по вертикальным срезам колонки: в каждом срезе шириной :data:`SLICE_W` (шаг
    :data:`SLICE_STEP`) — спады плотности краски (:func:`synth.all_baselines`); срезы в тени корешка пропускаются.
    Срезы локальны, поэтому подъём строк у корешка не теряется, как у прямой полосы во всю колонку.

    Args:
        ink: Краска кадра (bool).
        gray: Серое кадра uint8.
        x0: Левый край колонки.
        x1: Правый край.
        ya: Верх окна.
        yb: Низ окна.
        x_height: Высота буквы, px.

    Returns:
        Список ``(середина среза по x, базовые линии в кадре сверху вниз)``.
    """
    out = []
    for a in range(x0, max(x0 + 1, x1 - SLICE_W + 1), SLICE_STEP):
        b = min(x1, a + SLICE_W)
        win = ink[ya:yb, a:b]
        if win.sum() < 20 or np.percentile(gray[ya:yb, a:b], 75) < SHADOW_GRAY:
            continue
        out.append((a + (b - a) // 2, [ya + v for v in S.all_baselines(win, x_height)]))
    return out


def track_rows(slices: list[tuple[int, list[int]]], pitch: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Строки по срезам: от среза в середине колонки с наибольшим числом базовых линий — вправо и влево, каждая
    строка подхватывает в соседнем срезе ближайшую базовую линию не дальше :data:`TRACK_TOL` шага; после
    :data:`TRACK_MISS` срезов без линии подряд строка в эту сторону обрывается.

    Args:
        slices: Базовые линии по срезам (:func:`slice_baselines`).
        pitch: Шаг строк, px.

    Returns:
        Точки ``(xs, ys)`` каждой строки сверху вниз.
    """
    if not slices:
        return []
    mid = len(slices) // 2
    seed = max(range(len(slices)), key=lambda i: (len(slices[i][1]), -abs(i - mid)))
    seed_x, seed_ys = slices[seed]
    rows: list[list[tuple[int, int]]] = [[(seed_x, y)] for y in seed_ys]
    tol = TRACK_TOL * pitch
    for order in (range(seed + 1, len(slices)), range(seed - 1, -1, -1)):
        cur = [float(y) for y in seed_ys]
        miss = [0] * len(rows)
        for i in order:
            x, ys = slices[i]
            for k in range(len(rows)):
                if miss[k] > TRACK_MISS or not ys:
                    miss[k] += 1
                    continue
                j = int(np.argmin([abs(y - cur[k]) for y in ys]))
                if abs(ys[j] - cur[k]) <= tol:
                    cur[k] = float(ys[j])
                    rows[k].append((x, ys[j]))
                    miss[k] = 0
                else:
                    miss[k] += 1
    out = []
    for pts in rows:
        pts.sort()
        out.append((np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)))
    return out


@dataclass(frozen=True)
class WindowRows:
    """Строки одного окна колонки: оси в кадре, базовые линии и краска в повёрнутых координатах, шаг."""

    axes: list[S.Axis]
    baselines: list[int]
    rot: np.ndarray
    edge_rot: np.ndarray
    pitch: float


def window_rows(
    ink: np.ndarray, gray: np.ndarray, column: tuple[int, int], edge: BottomEdge, x_height: float, ya: int, yb: int
):
    """Строки колонки в окне по y: наклон, базовые линии в повёрнутых координатах (ложные у кромки — отсев),
    ось каждой строки по низам её краски, шаг строк.

    Args:
        ink: Краска страницы выше кромки (bool), координаты кадра.
        gray: Серое кадра uint8.
        column: Отрезок x колонки.
        edge: Кромка.
        x_height: Высота буквы, px.
        ya: Верх окна, px кадра.
        yb: Низ окна (не включая).

    Returns:
        ``WindowRows`` или None, если строк мало.
    """
    c0, c1 = column
    pad = int(X_PAD * x_height)
    xa, xb = max(0, c0 - pad), min(ink.shape[1], c1 + pad)
    win = ink[ya:yb, xa:xb]
    if win.sum() < 100:
        return None
    deg = skew_angle(win)
    h, w = win.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    rot = cv2.warpAffine(win.astype(np.uint8), m, (w, h), flags=cv2.INTER_NEAREST) > 0
    rot_gray = cv2.warpAffine(gray[ya:yb, xa:xb], m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ex = np.arange(xa, xb, dtype=float) - xa
    ey = np.interp(np.arange(xa, xb), np.arange(edge.x0, edge.x1), edge.y) - ya
    rx, ry = rotate_points(ex, ey, m)
    edge_rot = np.interp(np.arange(w), rx, ry) if rx[-1] > rx[0] else np.full(w, ry.min())
    baselines = []
    for b in S.all_baselines(rot, x_height):
        # Спад краски на самой кромке — не базовая линия: отсев по кромке в столбцах с краской этой строки.
        band = rot[max(0, int(b - 0.9 * x_height)) : int(b + 0.15 * x_height)]
        us = np.nonzero(band.any(axis=0))[0] if band.size else np.array([], int)
        if len(us) and np.mean(b > edge_rot[us] - EDGE_BASELINE * x_height) > 0.5:
            continue
        baselines.append(b)
    if len(baselines) < 2:
        return None
    pitch = float(np.median(np.diff(baselines))) if len(baselines) >= 3 else float(baselines[-1] - baselines[-2])
    inv = cv2.invertAffineTransform(m)
    first: list[S.Axis] = []
    for b in baselines:
        band = np.zeros_like(rot)
        band[max(0, int(b - 0.9 * x_height)) : int(b + 0.15 * x_height)] = True
        sel = rot & band
        bottom = np.where(sel.any(axis=0), sel.shape[0] - 1 - np.argmax(sel[::-1], axis=0), -1)
        ra, rb = max(0, int(b - 0.9 * x_height)), int(b + 0.15 * x_height)
        solid = (sel.sum(axis=0) > SOLID_SHARE * (rb - ra)) | shadow_columns(rot_gray[ra:rb])
        us = np.nonzero((bottom >= 0) & ~solid)[0]
        fx, fy = rotate_points(us.astype(float), bottom[us].astype(float), inv)
        axis = S.axis_from_points(fx + xa, fy + ya, x_height, c0, c1)
        if axis is not None:
            first.append(axis)
    # Второй проход: точки низов букв в полосе вдоль кривой первого прохода (а не вдоль повёрнутой прямой —
    # у корешка строка поднимается, и прямая полоса её теряет), затем общая форма всех строк окна со своими
    # сдвигами: одна строка у кромки обрезана и шумна, а изгиб у всех строк окна один.
    tracked = [r for r in track_rows(slice_baselines(ink, gray, xa, xb, ya, yb, x_height), pitch)
               if len(r[0]) >= MIN_TRACKED]  # fmt: skip
    shape = shared_shape(tracked, x_height, c0, c1) if len(tracked) >= 2 else None
    if shape is None:
        # Срезы не дали строк (узкая колонка): точки низов букв вдоль осей первого прохода.
        shape = shared_shape([row_bottom_points(ink, gray, a, x_height, xa, xb) for a in first], x_height, c0, c1)
    axes = first
    if shape is not None:
        curve, offsets = shape
        axes = [S.Axis(c0, c1, np.polyfit(curve[0], curve[1] + o, 1), x_height, curve[0], curve[1] + o)
                for o in offsets]  # fmt: skip
        # Уточнение по низам букв вдоль найденных осей (спады плотности дают базовую линию с точностью ±2 px).
        shape = shared_shape([row_bottom_points(ink, gray, a, x_height, xa, xb) for a in axes], x_height, c0, c1)
        if shape is not None:
            curve, offsets = shape
            axes = [S.Axis(c0, c1, np.polyfit(curve[0], curve[1] + o, 1), x_height, curve[0], curve[1] + o)
                    for o in offsets]  # fmt: skip
        if len(axes) >= 2:
            pitch = float(np.median(np.diff([float(np.median(a.y(np.arange(c0, c1, 10.0)))) for a in axes])))
    return WindowRows(axes, baselines, rot, edge_rot, pitch)


def row_geometry(
    ink: np.ndarray, gray: np.ndarray, column: tuple[int, int], edge: BottomEdge, x_height: float
) -> RowGeometry | None:
    """Оси строк колонки и ось срезанной строки: нижнее окно высотой :data:`LOW_UP` над кромкой; срезанная
    строка — на шаг ниже последней видимой, если под ней видны верхушки, иначе сама последняя (срезаны выносные).
    Для библиотеки букв добавляются оси строк из :data:`LIBRARY_WINDOWS` окон выше (каждое со своим наклоном).

    Args:
        ink: Краска страницы выше кромки (bool), координаты кадра.
        gray: Серое кадра uint8.
        column: Отрезок x колонки.
        edge: Кромка.
        x_height: Высота буквы, px.

    Returns:
        Геометрия или None, если строк мало.
    """
    c0, c1 = column
    e = edge.y[max(0, c0 - edge.x0) : max(1, c1 - edge.x0)]
    tall = int(TALL_UP * x_height)
    yb = int(min(ink.shape[0], e.max() + 2))
    # Нижнее окно невысокое: базовые линии ищутся по спадам плотности краски относительно максимума в окне, и
    # крупный заголовок выше по колонке заглушил бы строки у кромки.
    ya = int(max(0, e.min() - LOW_UP * x_height))
    low = window_rows(ink, gray, column, edge, x_height, ya, yb)
    if low is None or not low.axes:
        return None
    axes = list(low.axes)
    prev = axes[-1]
    # Низы последней найденной строки лежат на кромке — это сама срезанная строка (видны только верхушки), её
    # ось по низам завышена: берётся строка выше, сдвинутая на шаг.
    pxs, pys = row_bottom_points(ink, gray, prev, x_height, c0, c1, down=CUT_BAND_DOWN)
    if len(pxs) >= 10 and len(axes) >= 2:
        edge_at = np.interp(pxs, np.arange(edge.x0, edge.x1), edge.y)
        if float(np.mean(pys >= edge_at - EDGE_ABOVE - CUT_TOUCH)) > CUT_SHARE:
            axes = axes[:-1]
            prev = axes[-1]
    # Под последней строкой (на шаг ниже) есть краска — верхушки срезанной строки; иначе срезаны только выносные
    # последней видимой строки.
    share = row_ink_share(ink, prev.shifted(low.pitch), x_height, c0, c1)
    baseline_visible = not (share >= NEXT_ROW_INK and share * (c1 - c0) >= NEXT_ROW_PX)
    if baseline_visible:
        cut_axis = prev
        axes = axes[:-1]
    else:
        cut_axis = prev.shifted(low.pitch)
    # Проверка по краске вдоль осей: срезанная строка — самая нижняя с краской в зоне строчных (базовые линии у
    # самой кромки сходят за спад краски на кромке и теряются), под ней краски уже нет.
    while True:
        lower = cut_axis.shifted(low.pitch)
        if row_ink_share(ink, lower, x_height, c0, c1) < NEXT_ROW_INK:
            break
        axes.append(cut_axis)
        cut_axis = lower
        baseline_visible = False
    for k in range(LIBRARY_WINDOWS):
        wb, wa = ya - k * tall, ya - (k + 1) * tall
        if wb <= 0:
            break
        upper = window_rows(ink, gray, column, edge, x_height, max(0, wa), wb)
        if upper is not None:
            axes = upper.axes + axes
    return RowGeometry(axes, cut_axis, x_height, low.pitch, baseline_visible)


def row_ink_share(ink: np.ndarray, axis: S.Axis, x_height: float, x0: int, x1: int) -> float:
    """Доля столбцов отрезка с краской в зоне строчных букв вдоль оси (0.9 очка над базовой линией — 0.15 под).

    Args:
        ink: Краска кадра (bool).
        axis: Ось строки.
        x_height: Высота буквы, px.
        x0: Левый край отрезка.
        x1: Правый край.

    Returns:
        Доля (0..1).
    """
    maps = S.strip_maps(axis, 0.9 * x_height, 0.15 * x_height, x0, x1)
    band = cv2.remap(ink.astype(np.uint8), maps[0], maps[1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=0) > 0  # fmt: skip
    return float(band.any(axis=0).mean()) if band.size else 0.0


def harvest_rows(frame: np.ndarray, axes: list[S.Axis], library: S.Library, x_height: float) -> int:
    """Буквы строк в библиотеку: каждая строка выпрямляется вдоль своей оси, буквы — по рамкам tesseract.

    Args:
        frame: Кадр BGR.
        axes: Оси строк.
        library: Библиотека (пополняется).
        x_height: Высота буквы, px.

    Returns:
        Сколько букв добавлено.
    """
    g_up, g_down = int(S.GLYPH_UP * x_height), int(S.GLYPH_DOWN * x_height)
    added = 0
    for axis in axes:
        maps = S.strip_maps(axis, g_up, g_down)
        sub = cv2.remap(frame, maps[0], maps[1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        added += S.harvest_chars(sub, g_up, library, "base")
    return added


def column_edges(frame_ink: np.ndarray, axes: list[S.Axis], column: tuple[int, int], x_height: float):
    """Левый и правый края набора колонки по строкам выше: медиана начал и концов краски строк.

    Args:
        frame_ink: Краска кадра (bool).
        axes: Оси строк колонки.
        column: Отрезок x колонки (запасной ответ).
        x_height: Высота буквы, px.

    Returns:
        ``(левый край, правый край)`` в кадре.
    """
    lefts, rights = [], []
    for axis in axes:
        maps = S.strip_maps(axis, 0.9 * x_height, 0.15 * x_height, column[0], column[1])
        band = cv2.remap(frame_ink.astype(np.uint8), maps[0], maps[1], cv2.INTER_NEAREST) > 0
        cols = np.nonzero(band.any(axis=0))[0]
        if len(cols):
            lefts.append(column[0] + int(cols[0]))
            rights.append(column[0] + int(cols[-1]) + 1)
    if not lefts:
        return column
    return int(np.median(lefts)), int(np.max(rights))


@dataclass
class Strip:
    """Выпрямленная полоса срезанной строки.

    Поля:
        axis: Ось.
        u0: Левый край полосы в кадре.
        maps: Карты выпрямления.
        image: Полоса кадра BGR (после продления листа).
        base_v: Строка полосы с базовой линией.
        edge_v: Старая кромка по столбцам полосы (строка полосы, float).
    """

    axis: S.Axis
    u0: int
    maps: tuple[np.ndarray, np.ndarray]
    image: np.ndarray
    base_v: int
    edge_v: np.ndarray


def make_strip(frame: np.ndarray, axis: S.Axis, edge: BottomEdge, span: tuple[int, int], x_height: float) -> Strip:
    """Полоса срезанной строки вдоль её оси с окном одной строки (:data:`synth.GLYPH_UP` / ``GLYPH_DOWN``).

    Args:
        frame: Кадр BGR (после продления).
        axis: Ось строки.
        edge: Старая кромка.
        span: Отрезок x полосы.
        x_height: Высота буквы, px.

    Returns:
        Полоса.
    """
    g_up, g_down = int(S.GLYPH_UP * x_height), int(S.GLYPH_DOWN * x_height)
    maps = S.strip_maps(axis, g_up, g_down, span[0], span[1])
    img = cv2.remap(frame, maps[0], maps[1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    xs = np.arange(span[0], span[1])
    ey = np.interp(xs, np.arange(edge.x0, edge.x1), edge.y)
    edge_v = ey - axis.y(xs.astype(float)) + g_up
    return Strip(axis, span[0], maps, img, g_up, edge_v)


def pick_glyph(library: S.Library, char: str, x_height: float) -> S.Glyph | None:
    """Образец буквы: из первой библиотеки цепочки (своя колонка → общая библиотека кадра → подшивки), где годных
    по классу высоты и форме образцов не меньше :data:`MIN_OWN_SAMPLES` (иначе — все годные по цепочке), берётся
    ближайший к медианному отпечатку; латиница проверяется tesseract (рамки tesseract на латинице часто врут);
    для латиницы без образца — кириллический двойник.

    Args:
        library: Библиотека колонки (с общей библиотекой в ``fallback``).
        char: Буква.
        x_height: Высота очка, px.

    Returns:
        Образец или None.
    """
    for ch in (char, LOOKALIKE.get(char)):
        if ch is None:
            continue
        pools: list[list[S.Glyph]] = []
        lib: S.Library | None = library
        while lib is not None:
            pools.append([g for g in lib.samples.get(ch, []) if glyph_has_ink(g) and height_class_ok(ch, g, x_height)
                          and shape_ok(ch, g)])  # fmt: skip
            lib = lib.fallback
        pool = next((p for p in pools if len(p) >= MIN_OWN_SAMPLES), [g for p in pools for g in p])
        if not pool:
            continue
        widths = np.array([g.width for g in pool], float)
        med = float(np.median(widths))
        keep = [g for g, wd in zip(pool, widths) if abs(wd - med) <= max(2.0, 0.25 * med)] or pool
        stack = np.stack([cv2.resize(g.ratio.mean(axis=2), (20, 28)) for g in keep])
        dist = ((stack - np.median(stack, axis=0)[None]) ** 2).mean(axis=(1, 2))
        order = [keep[i] for i in np.argsort(dist)]
        if ch.isascii() and ch.isalpha():
            # Латиница: метки tesseract на ней в русском тексте часто чужие (корзина «F» — сплошь «Г», «р», «п»),
            # берётся только образец, который tesseract перечитывает как эту букву; иначе — двойник, потом шрифт.
            order = [g for g in order[:VERIFY_CANDIDATES] if verified_by_tesseract(ch, g)]
        if order:
            return order[0]
    if char.isascii() and char.isalpha():
        return font_glyph(char, library, x_height)
    return None


def font_glyph(char: str, library: S.Library, x_height: float) -> S.Glyph | None:
    """Буква из шрифта (DejaVu Serif) как последнее средство для латиницы без образца: высота по прописным
    образцам библиотеки (или :data:`FONT_CAP` очка), сила краски — по медианной глубине образцов.

    Args:
        char: Буква.
        library: Библиотека (для высоты прописных и глубины краски).
        x_height: Высота очка, px.

    Returns:
        Образец или None, если шрифта нет.
    """
    from PIL import Image, ImageDraw, ImageFont

    path = next((p for p in FONT_CANDIDATES if Path(p).is_file()), None)
    if path is None:
        return None
    tops, depths = [], []
    lib: S.Library | None = library
    while lib is not None:
        for ch, gs in lib.samples.items():
            for g in gs:
                if ch in TALL_LETTERS and ch.isupper() and glyph_has_ink(g):
                    tops.append(ink_span(g)[0])
                    depths.append(float(np.median(1 - g.ratio.mean(axis=2)[glyph_darkness(g) > 0.3])))
        lib = lib.fallback
    cap = float(np.median(tops)) if len(tops) >= 5 else FONT_CAP * x_height
    depth = float(np.median(depths)) if depths else 0.85
    font = ImageFont.truetype(path, int(round(cap * 4)))
    canvas = Image.new("L", (int(cap * 6), int(cap * 6)), 0)
    ImageDraw.Draw(canvas).text((cap, cap), char, fill=255, font=font)
    arr = np.asarray(canvas, np.float32) / 255.0
    ys, xs = np.nonzero(arr > 0.2)
    if not len(ys):
        return None
    arr = arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    scale = cap / arr.shape[0]
    arr = cv2.resize(arr, (max(2, int(round(arr.shape[1] * scale))), int(round(cap))), interpolation=cv2.INTER_AREA)
    arr = cv2.GaussianBlur(arr, (0, 0), 0.6)
    dark = np.clip(arr, 0, 1) * depth
    ratio = np.ones((arr.shape[0] + 4, arr.shape[1] + 2 * S.GLYPH_PAD, 3), np.float32)
    ratio[: arr.shape[0], S.GLYPH_PAD : S.GLYPH_PAD + arr.shape[1]] = (1 - dark)[..., None]
    log.info("буква «%s» из шрифта (высота %.0f px)", char, cap)
    return S.Glyph(ratio, arr.shape[0], arr.shape[1], "font")


def shape_ok(char: str, g: S.Glyph) -> bool:
    """Форма образца соответствует букве: у «ы» краска распадается по столбцам на две части (рамка tesseract часто
    отрезает «ь»); остальные не проверяются."""
    if char not in "ыЫ":
        return True
    cols = (glyph_darkness(g) > 0.3).any(axis=0)
    idx = np.nonzero(cols)[0]
    return len(idx) > 0 and not cols[idx[0] : idx[-1] + 1].all()


_VERIFIED: dict[tuple[str, int], bool] = {}


def verified_by_tesseract(char: str, g: S.Glyph) -> bool:
    """Образец латинской буквы читается tesseract (psm 10) как эта буква или её двойник (регистр не важен).

    Args:
        char: Буква.
        g: Образец.

    Returns:
        Подтверждён ли образец (ответы кэшируются по объекту образца).
    """
    key = (char, id(g))
    if key in _VERIFIED:
        return _VERIFIED[key]
    im = np.clip(g.ratio.mean(axis=2) * 230, 0, 255).astype(np.uint8)
    im = cv2.resize(im, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    im = cv2.copyMakeBorder(im, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=230)
    with tempfile.TemporaryDirectory(prefix="cutlines_") as work:
        path = Path(work) / "glyph.png"
        cv2.imwrite(str(path), im)
        env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
        done = subprocess.run(["tesseract", str(path), "stdout", "--psm", "10", "-l", "rus+eng"],
                              capture_output=True, text=True, env=env, timeout=30)  # fmt: skip
    answer = done.stdout.strip().lower()
    accept = {char.lower(), LOOKALIKE.get(char, char).lower(), LOOKALIKE.get(char.upper(), char).lower()}
    ok = answer in accept
    _VERIFIED[key] = ok
    return ok


def glyph_has_ink(g: S.Glyph | None) -> bool:
    """Образец годен: в нём есть краска (не пустая вырезка от съехавшей рамки tesseract)."""
    if g is None:
        return False
    dark = glyph_darkness(g) > 0.3
    return dark.sum() >= GLYPH_MIN_INK and dark.any(axis=0).sum() >= 0.5 * g.width


def ink_span(g: S.Glyph) -> tuple[float, float]:
    """Верх и низ краски буквы относительно базовой линии, px (верх — вверх положительно, низ — вниз положительно);
    ``(0, 0)``, если краски нет."""
    rows = np.nonzero((glyph_darkness(g) > 0.3).any(axis=1))[0]
    if not len(rows):
        return 0.0, 0.0
    return float(g.top - rows[0]), float(rows[-1] + 1 - g.top)


def height_class_ok(char: str, g: S.Glyph, x_height: float) -> bool:
    """Краска образца лежит там, где положено букве: строчные без надстрочных — верх 0.7–1.3 очка, высокие
    (прописные, цифры, «б», «й», скобки) — 1.2–2.0; низ — у базовой линии, у букв с подстрочными элементами —
    ниже; дефис и тире — посередине очка. Прочие знаки не проверяются.

    Args:
        char: Буква.
        g: Образец.
        x_height: Высота очка, px.

    Returns:
        Годен ли образец.
    """
    top, bottom = (v / max(x_height, 1.0) for v in ink_span(g))
    if char in "-—":
        return 0.25 <= top <= 0.85 and bottom <= -0.1
    if char in LOW_LETTERS:
        top_ok = 0.7 <= top <= 1.3
    elif char in TALL_LETTERS:
        top_ok = 1.2 <= top <= 2.0
    else:
        return True
    if char in DESCENDERS:
        return top_ok and 0.1 <= bottom <= 0.7
    return top_ok and -0.2 <= bottom <= 0.2


def nominal_layout(
    text: str, library: S.Library, x_height: float, left: int, right: int, ragged: bool
) -> tuple[list[S.Placed], str]:
    """Номинальная раскладка текста: буквы подряд с обычными просветами; с выключкой по формату пробелы
    растягиваются до правого края, без неё строка начинается слева с обычными пробелами.

    Args:
        text: Текст строки.
        library: Библиотека букв.
        x_height: Высота буквы, px.
        left: Левый край, координаты полосы.
        right: Правый край.
        ragged: Без выключки (последняя строка абзаца).

    Returns:
        ``(буквы, буквы без образца, номера слов с буквами без образца)``.
    """
    gap = S.LETTER_GAP * x_height
    tokens = text.split()
    words = [[(c, pick_glyph(library, c, x_height)) for c in tok] for tok in tokens]
    missing = "".join(sorted({c for tok in words for c, g in tok if g is None}))
    missing_words = {wi for wi, tok in enumerate(words) if any(g is None for _, g in tok)}
    widths = []
    for tok in words:
        wd = 0.0
        for k, (c, g) in enumerate(tok):
            if k:
                wd += gap * (S.PUNCT_GAP if c in S.PUNCT else 1.0)
            wd += g.width if g is not None else 0.5 * x_height
        widths.append(wd)
    space = S.SPACE * x_height
    min_space = SPACE_MIN * x_height
    # Строка не помещается между краями (у корешка лист в ракурсе, буквы уже образцов): буквы сжимаются по
    # ширине до нужной, не сильнее :data:`MIN_SQUEEZE`.
    need = sum(widths) + min_space * max(0, len(words) - 1)
    if need > right - left and len(words):
        f = max(MIN_SQUEEZE, (right - left - min_space * max(0, len(words) - 1)) / max(sum(widths), 1.0))
        words = [[(c, squeeze_glyph(g, f) if g is not None else None) for c, g in tok] for tok in words]
        gap *= f
        widths = [wd * f for wd in widths]
        log.info("раскладка: строка шире отрезка на %.0f px, буквы сжаты в %.2f", need - (right - left), f)
    if not ragged and len(words) > 1:
        space = max(min_space, (right - left - sum(widths)) / (len(words) - 1))
    placed: list[S.Placed] = []
    u = float(left)
    for wi, tok in enumerate(words):
        for k, (c, g) in enumerate(tok):
            if k:
                u += gap * (S.PUNCT_GAP if c in S.PUNCT else 1.0)
            if g is None:
                u += 0.5 * x_height
                continue
            placed.append(S.Placed(int(round(u - S.GLYPH_PAD)), g, wi))
            u += g.width
        u += space
    return placed, missing, missing_words


def squeeze_glyph(g: S.Glyph, f: float) -> S.Glyph:
    """Буква, сжатая по ширине в ``f`` раз (ракурс листа у корешка)."""
    if f >= 0.999:
        return g
    w = max(3, int(round(g.ratio.shape[1] * f)))
    ratio = cv2.resize(g.ratio, (w, g.ratio.shape[0]), interpolation=cv2.INTER_AREA)
    return S.Glyph(ratio, g.top, max(1, int(round(g.width * f))), g.source)


def glyph_darkness(g: S.Glyph) -> np.ndarray:
    """Темнота буквы (0 — бумага, 1 — краска)."""
    return np.clip(1.0 - g.ratio.mean(axis=2), 0.0, 1.0).astype(np.float32)


def score_profile(
    dark: np.ndarray, g: S.Glyph, v0: int, edge_v: np.ndarray, u_hint: int, x_height: float
) -> np.ndarray | None:
    """Профиль корреляции буквы с темнотой полосы по всем сдвигам ``u``: по видимым строкам вырезки
    (над старой кромкой), нормированная кросс-корреляция.

    Args:
        dark: Темнота полосы (h, w) float32.
        g: Буква.
        v0: Верхняя строка вырезки в полосе.
        edge_v: Старая кромка по столбцам полосы.
        u_hint: Номинальное место буквы (окно, где берётся кромка).
        x_height: Высота буквы, px.

    Returns:
        Профиль длины ``w`` (нули там, где буква не помещается) или None, если видимых строк мало.
    """
    gd = glyph_darkness(g)
    gh, gw = gd.shape
    h, w = dark.shape
    # Видимые строки — по кромке около буквы (медиана в окне поиска), а не по всей полосе: у корешка кромка
    # уходит далеко вверх, и минимум по полосе обнулил бы профили всех букв.
    reach = int((GLOBAL_SHIFT + SEARCH_X) * x_height)
    local = edge_v[max(0, u_hint - reach) : min(len(edge_v), u_hint + gw + reach)]
    limit = int(np.median(local) - EDGE_ABOVE) if len(local) else 0
    ya, yb = max(0, v0), min(h, v0 + gh, limit)
    if yb - ya < MIN_VISIBLE_ROWS or gw >= w or (gd[ya - v0 : yb - v0] > 0.3).sum() < 3:
        return None
    templ = gd[ya - v0 : yb - v0]
    band = dark[ya:yb]
    prof = cv2.matchTemplate(band, templ, cv2.TM_CCORR_NORMED).ravel()
    out = np.zeros(w, np.float32)
    out[: len(prof)] = np.nan_to_num(prof)
    return out


def refine_layout(placed: list[S.Placed], strip: Strip, x_height: float) -> tuple[list[S.Placed], int]:
    """Уточнение мест букв по видимым верхушкам: сдвиг базовой линии — по лучшим корреляциям букв, места по x —
    динамикой (Витерби) по профилям корреляции с штрафом за отклонение просветов от номинальных; буквы без
    видимой части идут за соседями по номинальным просветам.

    Args:
        placed: Номинальная раскладка (слева направо).
        strip: Полоса.
        x_height: Высота буквы, px.

    Returns:
        ``(уточнённые буквы, сдвиг базовой линии)``.
    """
    if not placed:
        return placed, 0
    gray = cv2.cvtColor(strip.image, cv2.COLOR_BGR2GRAY)
    ink = S.ink_of(gray)
    level = float(S.paper_level(strip.image, ink).mean())
    dark = np.clip(1.0 - gray.astype(np.float32) / max(level, 1.0), 0.0, 1.0)
    w = dark.shape[1]
    # Проход 1: вертикальный сдвиг — медиана dv лучших корреляций по буквам с видимой частью.
    dvs = []
    for p in placed:
        best = (-1.0, 0)
        for dv in range(-SEARCH_Y, SEARCH_Y + 1):
            prof = score_profile(dark, p.glyph, strip.base_v - p.glyph.top + dv, strip.edge_v, p.u, x_height)
            if prof is not None:
                lo, hi = max(0, p.u - int(2 * x_height)), min(w, p.u + int(2 * x_height))
                if hi > lo and prof[lo:hi].max() > best[0]:
                    best = (float(prof[lo:hi].max()), dv)
        if best[0] >= MIN_CORR:
            dvs.append(best[1])
    dv = int(np.median(dvs)) if dvs else 0
    # Профили по x при найденном dv.
    profiles = []
    for p in placed:
        prof = score_profile(dark, p.glyph, strip.base_v - p.glyph.top + dv, strip.edge_v, p.u, x_height)
        profiles.append(prof if prof is not None else np.zeros(w, np.float32))
    # Общий сдвиг строки: номинальная раскладка целиком двигается по x туда, где сумма корреляций букв наибольшая
    # (строка в номинале привязана к краям колонки, а в кадре может стоять иначе: начало ушло под корешок и т.п.).
    reach = int(GLOBAL_SHIFT * x_height)
    # Сдвиг не выводит строку за полосу.
    lo_shift = -placed[0].u
    hi_shift = w - (placed[-1].u + placed[-1].glyph.ratio.shape[1])
    best_shift, best_total = 0, -1e9
    for shift in range(max(-reach, lo_shift), min(reach, hi_shift) + 1):
        total = 0.0
        for p, prof in zip(placed, profiles):
            u = p.u + shift
            if 0 <= u < w:
                total += float(prof[u])
        if total > best_total:
            best_shift, best_total = shift, total
    placed = [S.Placed(p.u + best_shift, p.glyph, p.word) for p in placed]
    log.debug("общий сдвиг %d px (сумма корреляций %.2f, букв с профилем %d из %d)", best_shift, best_total,
              sum(prof.any() for prof in profiles), len(placed))  # fmt: skip
    # Динамика: u_i ≥ u_{i−1} + w_{i−1} + gap_i − GAP_SQUEEZE, ≤ … + gap_i + GAP_STRETCH·xh.
    widths = [p.glyph.ratio.shape[1] for p in placed]
    gaps = [0] + [placed[i].u - (placed[i - 1].u + widths[i - 1]) for i in range(1, len(placed))]
    n = len(placed)
    neg = -1e9
    score = np.full((n, w), neg, np.float32)
    back = np.zeros((n, w), np.int32)
    us = np.arange(w)
    score[0] = profiles[0] - LEFT_PENALTY * np.abs(us - placed[0].u) / x_height
    score[0][us + widths[0] > w] = neg
    for i in range(1, n):
        g = gaps[i]
        best = np.full(w, neg, np.float32)
        arg = np.zeros(w, np.int32)
        same_word = placed[i].word == placed[i - 1].word
        squeeze = GAP_SQUEEZE if same_word else max(0, g - int(SPACE_MIN * x_height))
        penalty = GAP_PENALTY if same_word else SPACE_PENALTY
        stretch = int((GAP_STRETCH if same_word else SPACE_STRETCH) * x_height)
        for off in range(widths[i - 1] + g - squeeze, widths[i - 1] + g + stretch + 1):
            if off < 1 or off >= w:
                continue
            cand = np.full(w, neg, np.float32)
            cand[off:] = score[i - 1][: w - off] - penalty * abs(off - widths[i - 1] - g) / x_height
            better = cand > best
            best[better] = cand[better]
            arg[better] = us[better] - off
        score[i] = best + profiles[i]
        score[i][us + widths[i] > w] = neg
        back[i] = arg
    if float(score[-1].max()) <= neg / 2:
        # Допустимого пути нет (строка не помещается в полосу при заданных просветах) — номинальная раскладка.
        log.warning("раскладка: допустимого пути нет, остаётся номинальная")
        return [S.Placed(p.u, S.Glyph(p.glyph.ratio, p.glyph.top - dv, p.glyph.width, p.glyph.source), p.word)
                for p in placed], dv  # fmt: skip
    u = int(np.argmax(score[-1]))
    chosen = [0] * n
    for i in range(n - 1, -1, -1):
        chosen[i] = u
        u = int(back[i][u])
    out = [S.Placed(chosen[i], S.Glyph(p.glyph.ratio, p.glyph.top - dv, p.glyph.width, p.glyph.source), p.word)
           for i, p in enumerate(placed)]  # fmt: skip
    return out, dv


def paper_surface(img: np.ndarray, valid: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Поверхность бумаги под окном: нормированное размытие годных пикселей (вне краски и вне окна) — тон с
    градиентами освещения и тенью корешка; зерно — σ отклонений годных пикселей от поверхности.

    Args:
        img: Полоса BGR.
        valid: Годные пиксели (bool).
        sigma: Радиус размытия, px.

    Returns:
        ``(поверхность (h, w, 3) float32, σ зерна (3,))``.
    """
    v = valid.astype(np.float32)
    num = cv2.GaussianBlur(img.astype(np.float32) * v[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(v, (0, 0), sigma)[..., None]
    fallback = np.median(img[valid], axis=0) if valid.any() else np.full(3, 200.0)
    surface = np.where(den > 1e-3, num / np.maximum(den, 1e-3), fallback[None, None, :]).astype(np.float32)
    resid = img[valid].astype(np.float32) - surface[valid]
    grain = np.clip(resid.std(axis=0), 0, GRAIN_MAX).astype(np.float32) if valid.sum() > 50 else np.zeros(3, np.float32)
    return surface, grain


def paste_strip(frame: np.ndarray, strip: Strip, out_strip: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Вклейка полосы обратно в кадр вдоль оси с альфой; ремап премультиплицированный, чтобы край окна не
    подмешивал чёрный бордер.

    Args:
        frame: Кадр BGR.
        strip: Полоса (ось, левый край, базовая линия).
        out_strip: Полоса после правки, float32.
        alpha: Альфа правки (h, w) float32 0..1.

    Returns:
        Кадр с вклейкой.
    """
    h, w = out_strip.shape[:2]
    axis = strip.axis
    xs = np.arange(strip.u0, strip.u0 + w)
    ys = axis.y(xs.astype(float))
    by0 = int(max(0, ys.min() - strip.base_v - 2))
    by1 = int(min(frame.shape[0], ys.max() + (h - strip.base_v) + 2))
    yy, xx = np.mgrid[by0:by1, strip.u0 : strip.u0 + w].astype(np.float32)
    vmap = (yy - axis.y(xx) + strip.base_v).astype(np.float32)
    umap = (xx - strip.u0).astype(np.float32)
    pm = cv2.remap(out_strip * alpha[..., None], umap, vmap, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                   borderValue=0)  # fmt: skip
    a_back = cv2.remap(alpha, umap, vmap, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None]
    out = frame.copy()
    roi = out[by0:by1, strip.u0 : strip.u0 + w].astype(np.float32)
    out[by0:by1, strip.u0 : strip.u0 + w] = np.clip(roi * (1 - a_back) + pm, 0, 255).astype(np.uint8)
    return out


def window_top(ink: np.ndarray, base_v: int, x_height: float) -> np.ndarray:
    """Верх окна стирания по столбцам: первая строка под краской строки выше (выносные элементы) в диапазоне
    ``base_v − TOP_RANGE[0]·xh … base_v − TOP_RANGE[1]·xh``; без краски — верх диапазона.

    Args:
        ink: Краска полосы (bool).
        base_v: Строка базовой линии.
        x_height: Высота буквы, px.

    Returns:
        Верх по столбцам (int).
    """
    r0 = max(0, int(base_v - TOP_RANGE[0] * x_height))
    r1 = max(r0 + 1, int(base_v - TOP_RANGE[1] * x_height))
    sub = ink[r0:r1]
    last = np.where(sub.any(axis=0), r1 - 1 - np.argmax(sub[::-1], axis=0), r0 - 1)
    return np.minimum(last + 2, r1).astype(int)


def snap_to_gap(ink_cols: np.ndarray, x: int, reach: int, min_gap: int) -> int:
    """Граница между словами: центр ближайшего к ``x`` просвета (столбцов без краски) шириной не меньше
    ``min_gap`` (пробел, а не просвет между буквами) в окне ``±reach``; без просвета — ``x``.

    Args:
        ink_cols: Есть ли краска в столбце (bool).
        x: Исходная граница.
        reach: Полуширина окна поиска, px.
        min_gap: Минимальная ширина просвета, px.

    Returns:
        Столбец границы.
    """
    lo, hi = max(0, x - reach), min(len(ink_cols), x + reach + 1)
    gaps = ndimage.label(~ink_cols[lo:hi])[0]
    best, best_d = x, reach + 1
    for g in range(1, gaps.max() + 1):
        idx = np.nonzero(gaps == g)[0]
        if len(idx) < min_gap:
            continue
        center = lo + (idx[0] + idx[-1]) // 2
        if abs(center - x) < best_d:
            best, best_d = int(center), abs(center - x)
    return best


def paste_words(
    frame: np.ndarray, strip: Strip, placed: list[S.Placed], skip_words: set[int], limits: tuple[int, int]
) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    """Замена срезанных слов целиком в два прохода: сначала ячейки заменяемых слов заливаются бумагой (поверхность
    по окружению с зерном), потом на эту бумагу наносятся буквы. Слова, целиком видимые над кромкой, и слова с
    буквами без образца не трогаются.

    Args:
        frame: Кадр BGR (после продления).
        strip: Полоса.
        placed: Буквы с номерами слов.
        skip_words: Слова, которые не заменяются (нет образца буквы).
        limits: Отрезок колонки в координатах полосы — стирание не выходит за него (тень корешка рядом).

    Returns:
        ``(кадр, окно изменений, сколько слов заменено)``.
    """
    h, w = strip.image.shape[:2]
    rows = np.arange(h)[:, None]
    cut_zone = rows >= (strip.edge_v - EDGE_ABOVE - PASTE_OVERLAP)[None, :]
    cut_words: set[int] = set()
    for p in placed:
        g = p.glyph
        gh, gw = g.ratio.shape[:2]
        y0 = strip.base_v - g.top
        ya, yb = max(0, y0), min(h, y0 + gh)
        xa, xb = max(0, p.u + S.GLYPH_PAD), min(w, p.u + gw - S.GLYPH_PAD)
        if ya >= yb or xa >= xb or p.word in skip_words:
            continue
        dark = glyph_darkness(g)[ya - y0 : yb - y0, xa - p.u : xb - p.u] > 0.3
        if (dark & cut_zone[ya:yb, xa:xb]).any():
            cut_words.add(p.word)
    if not cut_words:
        return frame, (strip.u0, 0, strip.u0, 0), 0
    # Срезана большая часть строки — набирается вся строка: у сильно срезанной строки видны только верхушки, и
    # границы между заменяемыми и нетронутыми словами по ним ненадёжны.
    all_words = {p.word for p in placed} - skip_words
    if len(cut_words) > WHOLE_LINE_SHARE * len(all_words):
        cut_words = all_words
    xh = strip.axis.x_height
    boxes = {}
    for wi in set(p.word for p in placed):
        us = [p.u + S.GLYPH_PAD for p in placed if p.word == wi]
        ue = [p.u + p.glyph.ratio.shape[1] - S.GLYPH_PAD for p in placed if p.word == wi]
        boxes[wi] = (max(0, min(us)), min(w, max(ue)))
    # Полоса делится на ячейки слов: границы — между соседними рамками раскладки, притянутые к ближайшему просвету
    # старой краски (рамки у стыка слов могут уехать на несколько px, просвет между словами — надёжнее). Ячейки
    # заменяемых слов стираются целиком (вместе с остатками старых букв), ячейки нетронутых слов остаются.
    old_ink = S.ink_of(cv2.cvtColor(strip.image, cv2.COLOR_BGR2GRAY))
    top = window_top(old_ink, strip.base_v, xh)
    ink_cols = (old_ink & (rows >= top[None, :])).any(axis=0)
    order = sorted(boxes, key=lambda wi: boxes[wi][0])
    bounds = [max(0, limits[0])]
    for prev_w, next_w in zip(order, order[1:]):
        mid = (boxes[prev_w][1] + boxes[next_w][0]) // 2
        bounds.append(snap_to_gap(ink_cols, mid, int(BOUND_SNAP * xh), max(3, int(SPACE_GAP_MIN * xh))))
    bounds.append(min(w, limits[1]))
    cells = {wi: (bounds[k], bounds[k + 1]) for k, wi in enumerate(order)}
    erase = np.zeros((h, w), bool)
    for wi in cut_words:
        a, b = cells[wi]
        if b > a:
            erase[:, a:b] = rows >= top[None, a:b]
    # Проход 1: бумага. Поверхность — по пикселям вне краски и вне окна в более высокой полосе (межстрочные
    # промежутки и поля вокруг строки выше), зерно — как у окружения.
    extra = int(PAPER_UP * xh) - strip.base_v
    tall_maps = S.strip_maps(strip.axis, strip.base_v + max(0, extra), h - strip.base_v, strip.u0, strip.u0 + w)
    tall = cv2.remap(frame, tall_maps[0], tall_maps[1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    tall_ink = S.ink_of(cv2.cvtColor(tall, cv2.COLOR_BGR2GRAY))
    off = max(0, extra)
    tall_erase = np.zeros(tall.shape[:2], bool)
    tall_erase[off:] = erase
    near_ink = cv2.dilate(tall_ink.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    valid = ~near_ink & ~(cv2.dilate(tall_erase.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    surface, grain = paper_surface(tall, valid, PAPER_SIGMA)
    paper = np.clip(surface[off:] + S.paper_grain((h, w), grain, seed=int(strip.u0)), 0, 255)
    a_paper = cv2.GaussianBlur(erase.astype(np.float32), (0, 0), PAPER_FEATHER)
    a_paper[erase] = 1.0
    papered = strip.image.astype(np.float32) * (1 - a_paper[..., None]) + paper * a_paper[..., None]
    frame1 = paste_strip(frame, strip, papered, a_paper)
    # Проход 2: буквы на бумагу — полоса заново по кадру с бумагой, буквы умножаются на неё.
    image2 = cv2.remap(frame1, strip.maps[0], strip.maps[1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    canvas, new_ink = S.render(image2.astype(np.float32), [p for p in placed if p.word in cut_words], strip.base_v)
    glyph_zone = cv2.dilate(new_ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    a_glyph = cv2.GaussianBlur(glyph_zone.astype(np.float32), (0, 0), 0.8)
    a_glyph[glyph_zone] = 1.0
    out = paste_strip(frame1, strip, np.clip(canvas, 0, 255), a_glyph)
    axis = strip.axis
    xs = np.arange(strip.u0, strip.u0 + w)
    ys = axis.y(xs.astype(float))
    by0 = int(max(0, ys.min() - strip.base_v - 2))
    by1 = int(min(frame.shape[0], ys.max() + (h - strip.base_v) + 2))
    return out, (strip.u0, by0, strip.u0 + w, by1), len(cut_words)


def complete_column(
    frame: np.ndarray, ink: np.ndarray, column: Column, edge: BottomEdge, x_height: float, library: S.Library
) -> tuple[np.ndarray, LineReport]:
    """Допечатка срезанной строки одной колонки.

    Args:
        frame: Кадр BGR после продления листа.
        ink: Краска страницы выше кромки (bool).
        column: Колонка с текстом.
        edge: Старая кромка.
        x_height: Высота буквы страницы, px.
        library: Библиотека букв страницы (пополняется строками колонки).

    Returns:
        ``(кадр, отчёт)``.
    """
    geo = row_geometry(ink, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (column.x0, column.x1), edge, x_height)
    if geo is None:
        return frame, LineReport("no_axis", column.text)
    harvest_rows(frame, geo.axes, library, geo.x_height)
    left, right = column_edges(ink, geo.axes, (column.x0, column.x1), geo.x_height)
    pad = int(X_PAD * geo.x_height)
    # Полоса накрывает и края набора по строкам выше, и найденный отрезок самой срезанной строки (у корешка
    # строки начинаются неровно, и остаток старой строки за краем полосы остался бы нестёртым).
    span = (max(0, min(left, column.x0) - pad), min(frame.shape[1], max(right, column.x1) + pad))
    strip = make_strip(frame, geo.cut_axis, edge, span, geo.x_height)
    # Видимая краска строки: выключка по формату, если она доходит до правого края колонки.
    s_gray = cv2.cvtColor(strip.image, cv2.COLOR_BGR2GRAY)
    rows = np.arange(s_gray.shape[0])[:, None]
    vis_ink = S.ink_of(s_gray) & (rows < (strip.edge_v - EDGE_ABOVE)[None, :])
    # Только внутри колонки (с малым запасом): тень корешка рядом считается краской и утащила бы раскладку.
    limits = (
        int(column.x0 - COLUMN_MARGIN * geo.x_height - span[0]),
        int(column.x1 + COLUMN_MARGIN * geo.x_height - span[0]),
    )
    vis_ink[:, : max(0, limits[0])] = False
    vis_ink[:, max(0, limits[1]) :] = False
    cols = np.nonzero(vis_ink.any(axis=0))[0]
    ink_left = left - span[0] if not len(cols) else int(cols[0])
    ink_right = right - span[0] if not len(cols) else int(cols[-1]) + 1
    ragged = ink_right < right - span[0] - RAGGED_GAP * geo.x_height
    lay_left = min(left - span[0], ink_left) if len(cols) else left - span[0]
    lay_right = ink_right if ragged else right - span[0]
    placed, missing, missing_words = nominal_layout(column.text, library, geo.x_height, lay_left, lay_right, ragged)
    if not placed:
        return frame, LineReport("no_text", column.text, missing=missing)
    placed, dv = refine_layout(placed, strip, geo.x_height)
    log.info("колонка %d–%d: строк %d, шаг %.1f, очко %.1f, база видна %s, выключка %s, dv %d, нет букв «%s»",
             column.x0, column.x1, len(geo.axes), geo.pitch, geo.x_height, geo.baseline_visible, not ragged, dv,
             missing)  # fmt: skip
    out, box, replaced = paste_words(frame, strip, placed, missing_words, limits)
    # Проверка: набранная строка читается как заданный текст.
    check = make_strip(out, geo.cut_axis, edge, span, geo.x_height)
    reread = " ".join(wd[0] for wd in S.tesseract_words(cv2.cvtColor(check.image, cv2.COLOR_BGR2GRAY)))
    match = float(fuzz.ratio(reread.lower(), column.text.lower()))
    status = "synth" if match >= MIN_REREAD else "check"
    if replaced == 0:
        status = "visible"
    log.info("колонка %d–%d: заменено слов %d из %d, перечитано «%s» (%.0f)", column.x0, column.x1, replaced,
             len(column.text.split()), reread, match)  # fmt: skip
    return out, LineReport(status, column.text, reread, match, missing, box)


def complete_page(
    frame: np.ndarray,
    page: np.ndarray,
    edge: BottomEdge,
    texts: list[str],
    x_height_hint: float,
    library: S.Library | None = None,
) -> PageCompletion:
    """Допечатка срезанных нижних строк страницы по текстам колонок.

    Args:
        frame: Кадр BGR после продления листа (старая кромка — ``edge``).
        page: Маска страницы на полном кадре (bool).
        edge: Старая кромка.
        texts: Тексты колонок слева направо (пустой — колонка не срезана).
        x_height_hint: Ожидаемая высота буквы, px.
        library: Общая библиотека букв кадра или None.

    Returns:
        Итог.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ink = ink_above_edge(gray, page, edge)
    x_height = x_height_of(ink, x_height_hint)
    found = bottom_columns(ink, edge, x_height)
    columns = assign_columns(found, texts)
    lib = S.Library(fallback=library)
    result = PageCompletion(frame)
    for col in columns:
        image, report = complete_column(result.image, ink, col, edge, x_height, lib)
        result.image = image
        result.reports.append(report)
    if library is not None:
        lib.merge_into(library)
    return result


def save_library(samples: dict[str, list[S.Glyph]], path) -> None:
    """Библиотека букв в npz: образцы по буквам (отношение к бумаге, верх, ширина, источник).

    Args:
        samples: Образцы по буквам.
        path: Файл ``.npz``.
    """
    arrays = {}
    index = []
    for ch, glyphs in samples.items():
        for g in glyphs:
            arrays[f"g{len(index)}"] = g.ratio.astype(np.float16)
            index.append((ch, int(g.top), int(g.width), g.source))
    arrays["index"] = np.array(index, dtype=object)
    np.savez_compressed(path, **arrays)


def load_library(path) -> S.Library:
    """Библиотека букв из npz (см. :func:`save_library`).

    Args:
        path: Файл ``.npz``.

    Returns:
        Библиотека (лимит образцов на букву — :data:`LIBRARY_CAP`).
    """
    data = np.load(path, allow_pickle=True)
    lib = S.Library()
    for i, (ch, top, width, source) in enumerate(data["index"]):
        bucket = lib.samples.setdefault(str(ch), [])
        if len(bucket) < LIBRARY_CAP:
            bucket.append(S.Glyph(data[f"g{i}"].astype(np.float32), int(top), int(width), str(source)))
    return lib


def merge_samples(parts: list[dict[str, list[S.Glyph]]], cap: int) -> dict[str, list[S.Glyph]]:
    """Слить библиотеки кадров: по кругу из каждого кадра, не больше ``cap`` образцов на букву.

    Args:
        parts: Образцы по буквам от каждого кадра.
        cap: Потолок образцов на букву.

    Returns:
        Слитые образцы.
    """
    chars = sorted({ch for part in parts for ch in part})
    out: dict[str, list[S.Glyph]] = {}
    for ch in chars:
        queues = [list(part.get(ch, [])) for part in parts]
        bucket: list[S.Glyph] = []
        while len(bucket) < cap and any(queues):
            for q in queues:
                if q and len(bucket) < cap:
                    bucket.append(q.pop(0))
        out[ch] = bucket
    return out
