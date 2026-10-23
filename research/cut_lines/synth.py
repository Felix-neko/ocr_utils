"""Синтез срезанной нижней строки из букв: ось строки, выпрямленные полосы основы и донора, текст строки (tesseract + DeepSeek), библиотека букв, набор вдоль оси и чёткая кромка листа."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage, signal

log = logging.getLogger(__name__)

# Полоса строки относительно базовой линии: вверх и вниз, в высотах буквы (x-height).
STRIP_UP = 2.2
STRIP_DOWN = 1.1
# Окно одной строки (буквы с надстрочными и подстрочными элементами), в высотах буквы: вверх и вниз от базовой
# линии. Соседние строки в окно не попадают — иначе tesseract строку не читает.
GLYPH_UP = 1.5
GLYPH_DOWN = 0.55
# Высокая полоса над нижней строкой для поиска строк выше и сбора букв, в высотах буквы.
TALL_UP = 14.0
# Порог краски: на сколько уровней темнее локального фона бумаги (закрытие окном больше штриха).
INK_DROP = 28
INK_BG_PX = 31
# Разрыв между колонками в строке, в высотах буквы.
COLUMN_GAP = 3.0
# Шаг сетки по x при подгонке базовой линии, px; степень полинома (``fit_axis``).
BASELINE_BIN = 40
BASELINE_DEGREE = 2
# Гладкая кривая оси: ширина корзины точек, допуск остатка (в высотах буквы), шаг выборки, px; ширина ядра
# локальной прямой, в высотах буквы.
CURVE_BIN = 8
CURVE_TOL = 0.08
CURVE_STEP = 4
CURVE_KERNEL = 1.5
# Усиление шума зерна бумаги после размытия.
GRAIN_GAIN = 1.6
# Буква в библиотеке: ширина очка не меньше, высота не больше этой доли x-height; запас поля по бокам, px.
GLYPH_MIN_W = 3
GLYPH_PAD = 2
# Сколько образцов на букву хранить.
GLYPH_SAMPLES = 24
# Межбуквенный просвет при наборе по умолчанию, в высотах буквы; пробел.
LETTER_GAP = 0.12
SPACE = 0.55
# Ширина перехода (альфа) в межстрочном промежутке над строкой, в высотах буквы.
SEAM_RAMP = 0.35
# Отступ кромки нового листа от базовой линии не меньше этого, в высотах буквы (поле под строкой).
EDGE_MIN = 0.9
# Буквы, которые можно набирать (латиница — из формул вроде «(Ver-»).
USEFUL = set(
    "абвгдежзийклмнопрстуфхцчшщъыьэюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯёЁ"
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.,;:()«»—/%№"
)
# Знаки препинания у базовой линии: в их вырезке не бывает высоких штрихов; просвет перед ними — эта доля обычного.
PUNCT = set(".,;:")
PUNCT_GAP = 0.25
# Строчные буквы без выносных элементов: по ним считается высота очка и базовая линия.
X_LETTERS = set("авгеёжзикмнопстхчшьыэяcaeomnosuvxz")


@dataclass(frozen=True)
class Axis:
    """Ось строки в координатах кадра: базовая линия на отрезке ``[x0, x1)`` — кривая по выборке ``(curve_x,
    curve_y)`` (линейная интерполяция, за концами — продолжение по крайнему наклону), а без выборки —
    полином ``y = poly(x)``.

    Поля:
        x0: Левый край, px.
        x1: Правый край, px.
        poly: Коэффициенты полинома (``np.polyval``); при кривой — прямая, приближающая её.
        x_height: Высота строчной буквы, px.
        curve_x: Абсциссы выборки кривой (возрастающие) или None.
        curve_y: Ординаты выборки кривой или None.
    """

    x0: int
    x1: int
    poly: np.ndarray
    x_height: float
    curve_x: np.ndarray | None = None
    curve_y: np.ndarray | None = None

    def y(self, x: np.ndarray) -> np.ndarray:
        """Базовая линия в точках ``x``."""
        if self.curve_x is None:
            return np.polyval(self.poly, x)
        x = np.asarray(x, float)
        cx, cy = self.curve_x, self.curve_y
        out = np.interp(x, cx, cy)
        if len(cx) >= 2:
            k0 = (cy[1] - cy[0]) / max(cx[1] - cx[0], 1e-6)
            k1 = (cy[-1] - cy[-2]) / max(cx[-1] - cx[-2], 1e-6)
            out = np.where(x < cx[0], cy[0] + k0 * (x - cx[0]), out)
            out = np.where(x > cx[-1], cy[-1] + k1 * (x - cx[-1]), out)
        return out

    def shifted(self, dy: float) -> "Axis":
        """Та же ось, сдвинутая по вертикали на ``dy`` px."""
        poly = self.poly.copy()
        poly[-1] += dy
        cy = None if self.curve_y is None else self.curve_y + dy
        return Axis(self.x0, self.x1, poly, self.x_height, self.curve_x, cy)


@dataclass
class Glyph:
    """Буква-донор в выпрямленной полосе.

    Поля:
        ratio: Отношение пикселей к уровню бумаги (h, w, 3) float32; строки — от ``top`` над базовой линией.
        top: Сколько px над базовой линией верх вырезки.
        width: Ширина очка, px.
        source: ``base`` (основа) или ``donor``.
    """

    ratio: np.ndarray
    top: int
    width: int
    source: str


@dataclass
class Library:
    """Библиотека букв: по букве список образцов; ``fallback`` — общая библиотека кадра для недостающих букв."""

    samples: dict[str, list[Glyph]] = field(default_factory=dict)
    fallback: "Library | None" = None
    _chosen: dict[str, Glyph | None] = field(default_factory=dict)

    def add(self, char: str, glyph: Glyph) -> None:
        """Добавить образец, если букв этого знака ещё мало."""
        bucket = self.samples.setdefault(char, [])
        if len(bucket) < GLYPH_SAMPLES:
            bucket.append(glyph)

    def pick(self, char: str) -> Glyph | None:
        """Образец буквы по согласию: ближайший к медианному отпечатку образцов основы (если есть), иначе донора.

        Метка буквы от tesseract изредка съезжает на соседнюю, и в копилку попадает чужая литера; медианный
        отпечаток её отбраковывает (как ``gutter_loss_restoration.library.choose``).
        """
        if char in self._chosen:
            return self._chosen[char]
        found = self.samples.get(char) or []
        own = [g for g in found if g.source == "base"]
        pool = own if len(own) >= 2 else found
        pick = None
        if pool:
            widths = np.array([g.width for g in pool], float)
            med = float(np.median(widths))
            keep = [g for g, wd in zip(pool, widths) if abs(wd - med) <= max(2.0, 0.25 * med)] or pool
            stack = np.stack([cv2.resize(g.ratio.mean(axis=2), (20, 28)) for g in keep])
            dist = ((stack - np.median(stack, axis=0)[None]) ** 2).mean(axis=(1, 2))
            pick = keep[int(np.argmin(dist))]
        if pick is None and self.fallback is not None:
            # Буквы нет в своей колонке — из общей библиотеки кадра.
            pick = self.fallback.pick(char)
        self._chosen[char] = pick
        return pick

    def merge_into(self, other: "Library") -> None:
        """Переложить свои образцы в другую библиотеку (общую библиотеку кадра)."""
        for char, found in self.samples.items():
            for g in found:
                other.add(char, g)
        other._chosen.clear()


def ink_of(gray: np.ndarray) -> np.ndarray:
    """Краска: темнее локального фона бумаги (закрытие окном больше штриха и медиана) на :data:`INK_DROP`.

    Args:
        gray: Серое uint8.

    Returns:
        Маска (bool).
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (INK_BG_PX, INK_BG_PX))
    paper = cv2.medianBlur(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k), 5).astype(np.int16)
    return (paper - gray.astype(np.int16)) > INK_DROP


def paper_level(img: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Уровень бумаги по каналам: медиана вне краски (с отступом от неё).

    Args:
        img: BGR.
        ink: Краска (bool).

    Returns:
        Вектор (3,) float32.
    """
    far = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 3) > 3
    pix = img[far] if far.sum() > 30 else img.reshape(-1, 3)
    return np.median(pix, axis=0).astype(np.float32)


def fit_axis(ink: np.ndarray, x_height_hint: float) -> Axis | None:
    """Ось строки по её краске: низы букв без выносных элементов, по корзинам вдоль x, полином.

    Args:
        ink: Краска одной строки в координатах кадра (bool), остальное — нули.
        x_height_hint: Ожидаемая высота буквы, px (отсев выносных и сора).

    Returns:
        Ось или None, если краски мало.
    """
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    h_ = stats[1:, cv2.CC_STAT_HEIGHT]
    sel = (h_ >= 0.6 * x_height_hint) & (h_ <= 1.4 * x_height_hint) & (stats[1:, cv2.CC_STAT_AREA] >= 6)
    if sel.sum() < 3:
        # Короткая строка (колонтитул): берём и прописные/цифры — низы всех знаков строки высотой с букву и выше.
        sel = (h_ >= 0.6 * x_height_hint) & (h_ <= 2.2 * x_height_hint) & (stats[1:, cv2.CC_STAT_AREA] >= 6)
        if sel.sum() < 3:
            return None
    xs = stats[1:, cv2.CC_STAT_LEFT][sel] + stats[1:, cv2.CC_STAT_WIDTH][sel] / 2
    bottoms = (stats[1:, cv2.CC_STAT_TOP] + h_)[sel].astype(float)
    # Робастная подгонка: два прохода с отсевом далёких точек (хвосты «р», «у», запятые). Мало точек — прямая.
    keep = np.ones(len(xs), bool)
    poly = np.polyfit(xs, bottoms, 1)
    for tol in (0.5, 0.25):
        deg = BASELINE_DEGREE if keep.sum() > 15 else 1
        poly = np.polyfit(xs[keep], bottoms[keep], deg)
        new = np.abs(np.polyval(poly, xs) - bottoms) <= tol * x_height_hint
        if new.sum() < 3:
            break
        keep = new
    small = sel.copy()
    small[sel] = h_[sel] <= 1.4 * x_height_hint
    x_height = float(np.median(h_[small])) if small.sum() >= 3 else x_height_hint
    cols = np.nonzero(ink.any(axis=0))[0]
    return Axis(int(cols.min()), int(cols.max()) + 1, poly, x_height)


def smooth_curve(
    xs: np.ndarray, ys: np.ndarray, x_height: float, x0: int, x1: int, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Гладкая кривая по точкам: локально-линейное сглаживание (ядро Гаусса шириной :data:`CURVE_KERNEL` высот
    буквы) с робастным перевзвешиванием — далёкие точки (выносные элементы, сор) теряют вес, а не отсекаются
    жёстко, так что подъём строки у корешка сохраняется; за крайними точками кривая продолжается по наклону
    локальной прямой на краю (без переплёска сплайна).

    Args:
        xs: Абсциссы точек.
        ys: Ординаты.
        x_height: Высота буквы, px (масштаб ядра и допуска остатка).
        x0: Левый край выборки кривой.
        x1: Правый край.
        weights: Начальные веса точек или None.

    Returns:
        ``(curve_x, curve_y)`` — выборка кривой с шагом :data:`CURVE_STEP` на ``[x0, x1)``, или None, если
        точек мало.
    """
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    w = np.ones(len(xs)) if weights is None else np.asarray(weights, float)
    # Точки сливаются в корзины шириной CURVE_BIN (взвешенное среднее) — меньше точек, устойчивее к скоплениям.
    bins = np.floor((xs - xs.min()) / CURVE_BIN).astype(int)
    uniq, inv = np.unique(bins, return_inverse=True)
    if len(uniq) < 6 or xs.max() - xs.min() < 4 * CURVE_BIN:
        return None
    bw = np.bincount(inv, weights=w)
    bx = np.bincount(inv, weights=w * xs) / np.maximum(bw, 1e-6)
    by = np.bincount(inv, weights=w * ys) / np.maximum(bw, 1e-6)
    sigma = CURVE_TOL * x_height
    kernel = CURVE_KERNEL * x_height
    grid = np.arange(x0, x1, CURVE_STEP, dtype=float)
    robust = np.ones(len(bx))
    cy = np.zeros(len(grid))
    for _ in range(3):
        cy = local_linear(bx, by, bw * robust, grid, kernel)
        fit_at = local_linear(bx, by, bw * robust, bx, kernel)
        robust = 1.0 / np.maximum(1.0, np.abs(by - fit_at) / sigma)
    return grid, cy


def local_linear(bx: np.ndarray, by: np.ndarray, bw: np.ndarray, at: np.ndarray, kernel: float) -> np.ndarray:
    """Локально-линейная регрессия: в каждой точке ``at`` взвешенная прямая по точкам с ядром Гаусса.

    Args:
        bx: Абсциссы точек.
        by: Ординаты.
        bw: Веса точек.
        at: Где вычислять.
        kernel: Ширина ядра (σ), px.

    Returns:
        Значения в точках ``at``.
    """
    dx = bx[None, :] - at[:, None]
    k = bw[None, :] * np.exp(-0.5 * (dx / kernel) ** 2)
    # Вдали от данных ядро гаснет: вес не меньше крошечного, чтобы прямая строилась по ближайшим точкам.
    k = np.maximum(k, 1e-9 * bw[None, :])
    s0 = k.sum(axis=1)
    s1 = (k * dx).sum(axis=1)
    s2 = (k * dx * dx).sum(axis=1)
    t0 = (k * by[None, :]).sum(axis=1)
    t1 = (k * dx * by[None, :]).sum(axis=1)
    det = s0 * s2 - s1 * s1
    slope_ok = det > 1e-12 * np.maximum(s0 * s2, 1e-30)
    a = np.where(slope_ok, (s2 * t0 - s1 * t1) / np.where(slope_ok, det, 1.0), t0 / np.maximum(s0, 1e-30))
    return a


def axis_from_points(xs: np.ndarray, ys: np.ndarray, x_height: float, x0: int, x1: int) -> Axis | None:
    """Ось строки по точкам низа её краски: гладкая кривая (:func:`smooth_curve`) на длинной строке, робастная
    прямая на короткой.

    Args:
        xs: Абсциссы точек в кадре.
        ys: Ординаты.
        x_height: Высота буквы, px.
        x0: Левый край оси.
        x1: Правый край.

    Returns:
        Ось или None, если точек мало.
    """
    if len(xs) < 20:
        return None
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    keep = np.ones(len(xs), bool)
    poly = np.polyfit(xs, ys, 1)
    for tol in (0.6, 0.3):
        poly = np.polyfit(xs[keep], ys[keep], 1)
        new = np.abs(np.polyval(poly, xs) - ys) <= tol * x_height
        if new.sum() < 20:
            break
        keep = new
    if np.ptp(xs) <= 15 * x_height:
        return Axis(int(x0), int(x1), poly, x_height)
    curve = smooth_curve(xs, ys, x_height, x0, x1)
    if curve is None:
        return Axis(int(x0), int(x1), poly, x_height)
    return Axis(int(x0), int(x1), poly, x_height, curve[0], curve[1])


def paper_grain(shape: tuple[int, int], sigma: np.ndarray, seed: int = 0) -> np.ndarray:
    """Зерно бумаги: слегка размытый гауссов шум с заданным σ по каналам (размытие съедает часть σ, поэтому
    шум усиливается в :data:`GRAIN_GAIN`).

    Args:
        shape: ``(h, w)``.
        sigma: σ по каналам (3,).
        seed: Зерно генератора.

    Returns:
        Приращения (h, w, 3) float32.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((shape[0], shape[1], 3)).astype(np.float32)
    return cv2.GaussianBlur(noise, (0, 0), 0.7) * (np.asarray(sigma, np.float32) * GRAIN_GAIN)


def strip_maps(axis: Axis, up: float, down: float, x0: int | None = None, x1: int | None = None):
    """Карты выпрямления: пиксель полосы ``(u, v)`` → точка кадра ``(x0 + u, base(x) + v − up)``.

    Args:
        axis: Ось строки.
        up: Высота полосы над базовой линией, px.
        down: Высота полосы под базовой линией, px.
        x0: Левый край полосы (по умолчанию — оси).
        x1: Правый край полосы.

    Returns:
        ``(map_x, map_y)`` float32 размера полосы.
    """
    x0 = axis.x0 if x0 is None else x0
    x1 = axis.x1 if x1 is None else x1
    u = np.arange(x0, x1, dtype=np.float32)
    v = np.arange(-int(up), int(down), dtype=np.float32)
    map_x = np.repeat(u[None, :], len(v), axis=0)
    map_y = (axis.y(u)[None, :] + v[:, None]).astype(np.float32)
    return map_x, map_y


def tesseract_words(gray: np.ndarray, psm: int = 7) -> list[tuple[str, int, int, int, int, float]]:
    """Слова строки по tesseract (rus+eng): текст, рамка, уверенность.

    Args:
        gray: Серое полосы uint8.
        psm: Режим разбиения (7 — одна строка).

    Returns:
        Список ``(текст, left, top, width, height, conf)``.
    """
    scale = 2.0
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    big = cv2.copyMakeBorder(big, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=int(np.percentile(big, 90)))
    with tempfile.TemporaryDirectory(prefix="cutlines_") as work:
        path = Path(work) / "strip.png"
        cv2.imwrite(str(path), big)
        env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
        done = subprocess.run(["tesseract", str(path), "-", "--psm", str(psm), "-l", "rus+eng", "tsv"],
                              capture_output=True, text=True, env=env, timeout=60)  # fmt: skip
    out = []
    for row in done.stdout.splitlines()[1:]:
        parts = row.split("\t")
        if len(parts) < 12 or parts[0] != "5" or not parts[11].strip():
            continue
        left, top, wid, hgt = (int(p) for p in parts[6:10])
        out.append((parts[11].strip(), int((left - 20) / scale), int((top - 20) / scale), int(wid / scale),
                    int(hgt / scale), float(parts[10])))  # fmt: skip
    return out


def tesseract_chars(gray: np.ndarray) -> list[tuple[str, int, int]]:
    """Рамки букв строки по tesseract (``makebox``, rus+eng): буква и её отрезок по x в координатах полосы.

    Args:
        gray: Серое окна одной строки uint8.

    Returns:
        Список ``(буква, x0, x1)``.
    """
    scale, pad = 2.0, 20
    big = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    big = cv2.copyMakeBorder(big, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=int(np.percentile(big, 90)))
    with tempfile.TemporaryDirectory(prefix="cutlines_") as work:
        path = Path(work) / "line.png"
        cv2.imwrite(str(path), big)
        env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
        done = subprocess.run(["tesseract", str(path), "stdout", "--psm", "7", "-l", "rus+eng", "makebox"],
                              capture_output=True, text=True, env=env, timeout=60)  # fmt: skip
    out = []
    for row in done.stdout.splitlines():
        parts = row.split()
        if len(parts) != 6:
            continue
        x0, x1 = (int(parts[1]) - pad) / scale, (int(parts[3]) - pad) / scale
        out.append((parts[0], int(round(x0)), int(round(x1))))
    return out


def line_glyphs(line: np.ndarray, top: int, source: str) -> list[tuple[str, int, Glyph]]:
    """Буквы строки по рамкам tesseract с их местами: буква, левый край очка в окне, вырезка.

    Рамка tesseract уточняется по столбцам краски внутри неё; чужая краска в вырезке убирается.

    Args:
        line: Окно одной строки BGR (базовая линия — строка ``top``).
        top: Строка окна с базовой линией.
        source: ``base`` или ``donor``.

    Returns:
        Список ``(буква, левый край очка, буква-вырезка)`` слева направо.
    """
    gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
    ink = ink_of(gray)
    level = paper_level(line, ink)
    out = []
    for char, x0, x1 in tesseract_chars(gray):
        if char not in USEFUL or x1 - x0 < 1:
            continue
        cols = np.nonzero(ink[:, max(0, x0) : min(line.shape[1], max(x0 + 1, x1))].any(axis=0))[0]
        if len(cols) == 0:
            continue
        a, b = max(0, x0) + int(cols[0]), max(0, x0) + int(cols[-1]) + 1
        xa, xb = max(0, a - GLYPH_PAD), min(line.shape[1], b + GLYPH_PAD)
        patch = line[:, xa:xb].astype(np.float32) / np.maximum(level[None, None, :], 1.0)
        patch = clean_patch(patch, ink[:, xa:xb], a - xa, b - xa, top, char)
        out.append((char, a, Glyph(np.clip(patch, 0, 1.2), top, b - a, source)))
    return out


def harvest_chars(line: np.ndarray, top: int, library: Library, source: str) -> int:
    """Буквы строки в библиотеку (``line_glyphs``), кроме слишком узких.

    Args:
        line: Окно одной строки BGR (базовая линия — строка ``top``).
        top: Строка окна с базовой линией.
        library: Библиотека (пополняется).
        source: ``base`` или ``donor``.

    Returns:
        Сколько букв добавлено.
    """
    added = 0
    for char, _, glyph in line_glyphs(line, top, source):
        if glyph.width >= GLYPH_MIN_W or char in PUNCT:
            library.add(char, glyph)
            added += 1
    return added


def layout_by_donor(
    text: str, donor: list[tuple[str, int, Glyph]], library: Library, x_height: float, offset: int
) -> list["Placed"]:
    """Набор строки по буквам донора на их местах; буквы, которых в доноре нет или прочитанные не так, — из
    библиотеки, между соседними поставленными буквами.

    Буквы донора уже стоят в геометрии строки (масштаб, наклон, изгиб), поэтому строка из них ровнее, чем из
    образцов со строк выше; текст DeepSeek задаёт, какие буквы донора верны.

    Args:
        text: Текст строки.
        donor: Буквы донора (``line_glyphs``) в координатах окна строки.
        library: Библиотека для недостающих букв.
        x_height: Высота буквы, px.
        offset: Сдвиг координат окна донора в координаты полосы набора.

    Returns:
        Поставленные буквы с номерами слов.
    """
    import difflib

    chars = [(c, wi) for wi, tok in enumerate(text.split()) for c in tok]
    target = "".join(c for c, _ in chars)
    seen = "".join(c for c, _, _ in donor)
    match = difflib.SequenceMatcher(None, target, seen, autojunk=False)
    # Место каждой буквы текста: своя буква донора, если совпала.
    pos: list[tuple[int, Glyph] | None] = [None] * len(target)
    for a, b, size in match.get_matching_blocks():
        for k in range(size):
            _, x, g = donor[b + k]
            pos[a + k] = (x + offset, g)
    placed: list[Placed] = []
    i = 0
    while i < len(target):
        if pos[i] is not None:
            x, g = pos[i]
            placed.append(Placed(int(round(x - GLYPH_PAD)), g, chars[i][1]))
            i += 1
            continue
        # Пробел из недостающих букв: от правого края предыдущей поставленной до левого следующей.
        j = i
        while j < len(target) and pos[j] is None:
            j += 1
        prev = next((pos[k] for k in range(i - 1, -1, -1) if pos[k] is not None), None)
        nxt = pos[j] if j < len(target) else None
        glyphs = [library.pick(target[k]) for k in range(i, j)]
        need = sum(g.width for g in glyphs if g is not None) + LETTER_GAP * x_height * max(0, j - i - 1)
        if prev is not None:
            start = prev[0] + prev[1].width + LETTER_GAP * x_height
        elif nxt is not None:
            start = nxt[0] - LETTER_GAP * x_height - need
        else:
            start = offset
        if prev is not None and chars[i][1] != chars[i - 1][1]:
            start += SPACE * x_height
        u = start
        for k, g in zip(range(i, j), glyphs):
            if g is None:
                u += 0.5 * x_height
                continue
            placed.append(Placed(int(round(u - GLYPH_PAD)), g, chars[k][1]))
            u += g.width + LETTER_GAP * x_height
        i = j
    return placed


def clean_patch(patch: np.ndarray, ink: np.ndarray, a: int, b: int, top: int, char: str) -> np.ndarray:
    """Вырезка буквы без чужой краски: компоненты, лежащие в основном вне рамки буквы ``[a, b)``, — в бумагу; у
    знаков препинания — ещё и всё выше половины высоты буквы (хвост соседней буквы в рамке точки или запятой).

    Args:
        patch: Отношение к бумаге (h, w, 3).
        ink: Краска вырезки (bool).
        a: Левый край буквы в вырезке.
        b: Правый край.
        top: Строка базовой линии.
        char: Буква.

    Returns:
        Вычищенная вырезка.
    """
    out = patch.copy()
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    x_height = top / GLYPH_UP
    for i in range(1, count):
        left, width = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_WIDTH]
        inside = max(0, min(b, left + width) - max(a, left))
        foreign = inside < 0.5 * width
        if char in PUNCT:
            foreign |= stats[i, cv2.CC_STAT_TOP] < top - 0.5 * x_height
        if foreign:
            region = cv2.dilate((labels == i).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            out[region] = 1.0
    return out


def letter_runs(ink: np.ndarray) -> list[tuple[int, int]]:
    """Буквы слова по вертикальным просветам краски: отрезки столбцов с краской, слитые по перекрытию компонент.

    Args:
        ink: Краска слова (bool) в его рамке.

    Returns:
        Отрезки ``(x0, x1)``.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    spans = sorted((stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH])
                   for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 4)  # fmt: skip
    merged: list[list[int]] = []
    for a, b in spans:
        # Точки «й», «ё», «ы» и перекрывающиеся компоненты — одна буква.
        if merged and a < merged[-1][1] - 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def harvest(strip: np.ndarray, base_v: int, library: Library, source: str, words) -> int:
    """Буквы строки в библиотеку: слова, у которых число букв совпало с числом отрезков краски.

    Args:
        strip: Выпрямленная полоса BGR (базовая линия — строка ``base_v``).
        base_v: Номер строки полосы с базовой линией.
        library: Библиотека (пополняется).
        source: ``base`` или ``donor``.
        words: Слова tesseract этой полосы.

    Returns:
        Сколько букв добавлено.
    """
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    ink = ink_of(gray)
    level = paper_level(strip, ink)
    added = 0
    for text, left, top, wid, hgt, conf in words:
        letters = re.sub(r"\s", "", text)
        if conf < 60 or not letters:
            continue
        a, b = max(0, left - 1), min(strip.shape[1], left + wid + 1)
        runs = letter_runs(ink[:, a:b])
        if len(runs) != len(letters):
            continue
        for char, (r0, r1) in zip(letters, runs):
            if char not in USEFUL or r1 - r0 < GLYPH_MIN_W:
                continue
            x0, x1 = max(0, a + r0 - GLYPH_PAD), min(strip.shape[1], a + r1 + GLYPH_PAD)
            patch = strip[:, x0:x1].astype(np.float32) / np.maximum(level[None, None, :], 1.0)
            library.add(char, Glyph(np.clip(patch, 0, 1.2), base_v, r1 - r0, source))
            added += 1
    return added


def row_baselines(tall_ink: np.ndarray, x_height: float, base_v: int) -> list[int]:
    """Базовые линии строк в высокой выпрямленной полосе: резкие спады плотности краски по строкам полосы.

    Args:
        tall_ink: Краска высокой полосы (bool), нижняя строка — на ``base_v``.
        x_height: Высота буквы, px.
        base_v: Строка полосы с базовой линией нижней строки.

    Returns:
        Номера строк полосы с базовыми линиями (выше нижней строки), сверху вниз.
    """
    profile = ndimage.gaussian_filter1d(tall_ink.mean(axis=1).astype(float), 1.5)
    drop = -np.gradient(profile)
    peaks, _ = signal.find_peaks(drop, distance=max(3, int(1.3 * x_height)), height=0.25 * max(drop.max(), 1e-6))
    return [int(p) for p in peaks if p < base_v - 1.2 * x_height]


def all_baselines(ink: np.ndarray, x_height: float) -> list[int]:
    """Все базовые линии строк в выпрямленной полосе: резкие спады плотности краски по строкам полосы.

    Args:
        ink: Краска полосы (bool).
        x_height: Высота буквы, px.

    Returns:
        Номера строк полосы с базовыми линиями, сверху вниз.
    """
    profile = ndimage.gaussian_filter1d(ink.mean(axis=1).astype(float), 1.5)
    drop = -np.gradient(profile)
    peaks, _ = signal.find_peaks(drop, distance=max(3, int(1.3 * x_height)), height=0.25 * max(drop.max(), 1e-6))
    return [int(p) for p in peaks]


def column_segments(ink_row: np.ndarray, x_height: float) -> list[tuple[int, int]]:
    """Колонки нижней строки: отрезки по x между широкими просветами краски.

    Args:
        ink_row: Краска полосы нижней строки (bool).
        x_height: Высота буквы, px.

    Returns:
        Отрезки ``(u0, u1)`` полосы.
    """
    cols = ndimage.binary_closing(ink_row.any(axis=0), structure=np.ones(int(COLUMN_GAP * x_height)))
    labels, count = ndimage.label(cols)
    out = []
    for i in range(1, count + 1):
        idx = np.nonzero(labels == i)[0]
        if len(idx) > 3 * x_height:
            out.append((int(idx[0]), int(idx[-1]) + 1))
    return out


@dataclass(frozen=True)
class Placed:
    """Буква, поставленная в полосу.

    Поля:
        u: Левый край вырезки в полосе, px.
        glyph: Буква.
        word: Номер слова в строке.
    """

    u: int
    glyph: Glyph
    word: int = 0


def layout(text: str, words, segment: tuple[int, int], library: Library, x_height: float) -> list[Placed]:
    """Набор строки: слова текста по рамкам слов донора (если число слов совпало), иначе подряд от начала колонки.

    Args:
        text: Текст строки (колонки).
        words: Слова tesseract донорской полосы в этой колонке (рамки — позиции слов).
        segment: Колонка ``(u0, u1)``.
        library: Библиотека букв.
        x_height: Высота буквы, px.

    Returns:
        Поставленные буквы.
    """
    tokens = text.split()
    gap = LETTER_GAP * x_height
    placed: list[Placed] = []
    # Буквы слов вместе со знаками: перед знаком препинания просвет меньше (точка не отходит от слова).
    chars = [[(c, library.pick(c)) for c in tok] for tok in tokens]
    if words and len(words) == len(tokens):
        # Слово в рамку донора: просветы между буквами растягиваются так, чтобы слово заняло рамку.
        for wi, (glyphs, (_, left, _, wid, _, _)) in enumerate(zip(chars, words)):
            have = [(c, g) for c, g in glyphs if g is not None]
            ink_w = sum(g.width for _, g in have)
            weights = [PUNCT_GAP if c in PUNCT else 1.0 for c, _ in have[1:]]
            g_gap = float(np.clip((wid - ink_w) / max(sum(weights), 1.0), 0.0, 3 * gap)) if weights else 0.0
            u = float(left)
            for k, (c, g) in enumerate(have):
                if k:
                    u += g_gap * (PUNCT_GAP if c in PUNCT else 1.0)
                placed.append(Placed(int(round(u - GLYPH_PAD)), g, wi))
                u += g.width
        return placed
    u = float(segment[0])
    for wi, glyphs in enumerate(chars):
        for k, (c, g) in enumerate(glyphs):
            if k:
                u += gap * (PUNCT_GAP if c in PUNCT else 1.0)
            if g is None:
                u += 0.5 * x_height
                continue
            placed.append(Placed(int(round(u - GLYPH_PAD)), g, wi))
            u += g.width
        u += SPACE * x_height
    return placed


def render(paper: np.ndarray, placed: list[Placed], base_v: int) -> tuple[np.ndarray, np.ndarray]:
    """Набранная строка на бумаге: буквы умножаются на тон бумаги (минимум, где буквы перекрываются).

    Args:
        paper: Тон бумаги полосы (h, w, 3) float32.
        placed: Буквы.
        base_v: Строка полосы с базовой линией.

    Returns:
        ``(картинка float32, маска букв bool)``.
    """
    h, w = paper.shape[:2]
    ratio = np.ones((h, w, 3), np.float32)
    mask = np.zeros((h, w), bool)
    for p in placed:
        g = p.glyph
        gh, gw = g.ratio.shape[:2]
        # Вертикаль: базовая линия буквы совпадает с базовой линией полосы.
        y0 = base_v - g.top
        ya, yb = max(0, y0), min(h, y0 + gh)
        xa, xb = max(0, p.u), min(w, p.u + gw)
        if ya >= yb or xa >= xb:
            continue
        sub = g.ratio[ya - y0 : yb - y0, xa - p.u : xb - p.u]
        ratio[ya:yb, xa:xb] = np.minimum(ratio[ya:yb, xa:xb], np.minimum(sub, 1.0))
        mask[ya:yb, xa:xb] |= sub.mean(axis=2) < 0.85
    return paper * ratio, mask
