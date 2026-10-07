"""Линия сгиба у верхнего края кадра постраничной съёмки: варианты кандидатов и сборки в линию, эталон глазами.

Сгиб — граница нашей полосы под загибом соседней: сверху вниз идёт загиб соседней полосы (или тень), затем
складка (тонкая тёмная линия или перепад тени к бумаге), затем наша бумага. Прежний способ (самая тёмная строка
бумаги в куске + парабола, ``single_page_gutter.find_fold``) путает складку с верхним краем фото и рамок, лежащих
ниже неё. Здесь собраны варианты для сравнения на эталоне (``sets/eg_1985_10_12_fold_gt.csv``):

Кандидаты по столбцам (``Candidate``): A — минимум уровня бумаги; B — перепад «темно → светло» сверху вниз;
C — хребет (тонкая тёмная линия); B+C — сумма откликов. Фильтры: D — самый верхний сильный кандидат вместо
сильнейшего; E — под кандидатом бумага; F — прищепки (зелёный цвет) исключаются.

Сборка (``Assembly``): 1 — итеративная парабола МНК; 2 — RANSAC-парабола; 3 — путь минимальной стоимости по
карте отклика (динамическое программирование); 4 — путь со сглаживанием.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

# Зона уровня бумаги от верхнего края кадра, px превью, и строки, где может лежать складка (по эталону ЭГ 1985
# 10-12 складка на 54–201 строке; запас вниз — на сильный наклон).
SEARCH_ROWS, FOLD_ROWS = 700, 320
# Ширина столбца-полосы для уровня бумаги, px превью (уровень — 80-й перцентиль по столбцам полосы).
COLUMN_STEP = 16
# Сглаживание профиля уровня бумаги по y (сигма, px).
PROFILE_SIGMA = 2.0
# Перепад B: разность уровней бумаги ниже и выше строки на таком плече, px.
EDGE_ARM = 6
# Хребет C: полуширина тёмной линии складки, px (отклик — превышение среднего по плечам над центром).
RIDGE_HALF, RIDGE_ARM = 3, 10
# Фильтр E: полоса под кандидатом (от, до), px, и её уровень бумаги не ниже такой доли бумаги полосы.
PAPER_BELOW, PAPER_SHARE = (10, 60), 0.70
# Фильтр D: самый верхний кандидат с откликом не ниже такой доли максимума столбца.
TOP_SHARE = 0.5
# Фильтр F: прищепки — зелёный оттенок (HSV, OpenCV 0–180) и насыщенность.
CLIP_HUE, CLIP_SAT = (40, 95), 80
# Путь (сборка 3): штраф за скачок по y между соседними столбцами (на px отклика) и наибольший скачок, px.
PATH_JUMP_COST, PATH_MAX_JUMP = 0.6, 12


class Candidate(str, Enum):
    """Отклик складки по столбцу."""

    MINIMUM = "A"
    EDGE = "B"
    RIDGE = "C"
    EDGE_RIDGE = "BC"


class Assembly(str, Enum):
    """Сборка кандидатов в линию."""

    PARABOLA = "1"
    RANSAC = "2"
    PATH = "3"
    PATH_SMOOTH = "4"


@dataclass(frozen=True)
class Variant:
    """Вариант поиска линии: отклик, фильтры и сборка.

    Attributes:
        candidate: Отклик складки.
        topmost: D — брать самый верхний сильный кандидат столбца.
        paper_below: E — кандидат годен, только если под ним бумага.
        clips: F — исключать столбцы с прищепками.
        assembly: Сборка в линию.
    """

    candidate: Candidate
    topmost: bool
    paper_below: bool
    clips: bool
    assembly: Assembly

    @property
    def name(self) -> str:
        """Короткое имя варианта, например ``BC+EF+3``."""
        flags = ("D" if self.topmost else "") + ("E" if self.paper_below else "") + ("F" if self.clips else "")
        return f"{self.candidate.value}{'+' + flags if flags else ''}+{self.assembly.value}"


def ruler_crop(rgb: np.ndarray, x_center: int, half_width: int = 160, rows: int = 420, scale: float = 2.0):
    """Вырезка зоны сгиба 1:1 (с увеличением) с линейкой по y — для разметки складки глазами.

    Линейка: тонкие риски каждые 10 px, подписи каждые 50 px (в координатах превью), вертикальная метка — столбец
    ``x_center``, по которому читается y складки.

    Args:
        rgb: Кадр RGB uint8, корешок сверху.
        x_center: Столбец превью, где ставится точка эталона.
        half_width: Полуширина вырезки, px превью.
        rows: Высота вырезки от верхнего края кадра, px превью.
        scale: Увеличение вырезки.

    Returns:
        RGB uint8: вырезка с полем слева под подписи линейки.
    """
    x0, x1 = max(0, x_center - half_width), min(rgb.shape[1], x_center + half_width)
    crop = cv2.resize(rgb[:rows, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    margin = 70
    canvas = np.full((crop.shape[0], crop.shape[1] + margin, 3), 255, np.uint8)
    canvas[:, margin:] = crop
    for y in range(0, rows, 10):
        yy = int(y * scale)
        major = y % 50 == 0
        # Риски линейки: полупрозрачные поперёк вырезки, короткие — в поле.
        cv2.line(canvas, (margin - (25 if major else 10), yy), (margin, yy), (0, 0, 0), 1)
        if major:
            overlay = canvas.copy()
            cv2.line(overlay, (margin, yy), (canvas.shape[1], yy), (0, 160, 255), 1)
            canvas = cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0)
            cv2.putText(canvas, str(y), (2, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # Столбец разметки.
    xc = margin + int((x_center - x0) * scale)
    cv2.line(canvas, (xc, 0), (xc, canvas.shape[0]), (255, 0, 0), 1)
    return canvas


# --- Разметка полосы (surya layout, строки surya, DeepSeek grounding) в координатах кадра -----------------------

# Лента у верхнего края кадра для разметки: столько строк кадра (складка — на 50–210, ниже — начало полосы).
LAYOUT_BAND = 600
# Длина куска ленты вдоль сгиба (после поворота — высота картинки) и перекрытие кусков, px превью.
LAYOUT_PIECE, LAYOUT_OVERLAP = 1500, 150
# Штраф отклика внутри рамки блока (в единицах нормированного отклика) и отступ рамки, px.
BOX_PENALTY, BOX_PAD = 3.0, 2


@dataclass(frozen=True)
class BandPiece:
    """Кусок ленты у сгиба, повёрнутый к прямому чтению, и его привязка к кадру.

    Attributes:
        image: RGB uint8, текст полосы прямо.
        side: ``"L"`` (кадр ``_2``, повёрнут по часовой) или ``"R"`` (``_1``, против часовой).
        start: Первая строка куска в повёрнутой ленте (вдоль сгиба), px.
        band: Высота ленты в кадре (строк), px.
        width: Ширина кадра, px.
    """

    image: np.ndarray
    side: str
    start: int
    band: int
    width: int


def band_pieces(rgb: np.ndarray, side: str, band: int = LAYOUT_BAND) -> list[BandPiece]:
    """Нарезка верхней ленты кадра на прямые куски для моделей разметки.

    Args:
        rgb: Кадр RGB uint8, корешок сверху.
        side: ``"L"``/``"R"`` (см. ``single_page_gutter.page_side``).
        band: Высота ленты от верхнего края кадра, px.

    Returns:
        Куски вдоль сгиба длиной ``LAYOUT_PIECE`` с перекрытием ``LAYOUT_OVERLAP``.
    """
    strip = rgb[:band]
    # Как в ``gutter_strip``: правая полоса — против часовой (сгиб слева), левая — по часовой (сгиб справа).
    upright = np.rot90(strip, 1 if side == "R" else -1)
    length = upright.shape[0]
    pieces = []
    for start in range(0, max(1, length - LAYOUT_OVERLAP), LAYOUT_PIECE - LAYOUT_OVERLAP):
        stop = min(length, start + LAYOUT_PIECE)
        piece = np.ascontiguousarray(upright[start:stop])
        pieces.append(BandPiece(piece, side, start, band, rgb.shape[1]))
        if stop == length:
            break
    return pieces


def piece_box_to_frame(
    piece: BandPiece, x0: float, y0: float, x1: float, y1: float
) -> tuple[float, float, float, float]:
    """Рамка в координатах прямого куска → рамка в координатах кадра (``x0, y0, x1, y1``).

    Args:
        piece: Кусок ленты.
        x0, y0, x1, y1: Рамка в пикселях куска.

    Returns:
        Рамка в пикселях кадра.
    """
    i0, i1 = y0 + piece.start, y1 + piece.start  # строки повёрнутой ленты
    if piece.side == "R":
        # rot90 k=1: строка ленты i ↔ столбец кадра W-1-i, столбец ленты j ↔ строка кадра j.
        return piece.width - 1 - i1, x0, piece.width - 1 - i0, x1
    # rot90 k=-1: строка ленты i ↔ столбец кадра i, столбец ленты j ↔ строка кадра band-1-j.
    return i0, piece.band - 1 - x1, i1, piece.band - 1 - x0


# --- Отклики складки по столбцам ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Levels:
    """Уровень бумаги по столбцам-полосам верхней зоны кадра.

    Attributes:
        level: (строки ``SEARCH_ROWS``, столбцы) — 80-й перцентиль яркости полосы шириной ``COLUMN_STEP``,
            сглаженный по y.
        xs: Центры столбцов, px кадра.
        paper: Уровень бумаги кадра (90-й перцентиль уровня на строках 300–700).
        clips: Столбцы, где сверху лежит прищепка (доля зелёных пикселей в верхних 350 строках > 5 %).
    """

    level: np.ndarray
    xs: np.ndarray
    paper: float
    clips: np.ndarray


def column_levels(rgb: np.ndarray) -> Levels:
    """Уровень бумаги по столбцам и маска прищепок.

    Args:
        rgb: Кадр RGB uint8, корешок сверху.

    Returns:
        ``Levels``.
    """
    gray = cv2.cvtColor(rgb[:SEARCH_ROWS], cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(rgb[:350], cv2.COLOR_RGB2HSV)
    green = (hsv[..., 0] >= CLIP_HUE[0]) & (hsv[..., 0] <= CLIP_HUE[1]) & (hsv[..., 1] >= CLIP_SAT)
    n = gray.shape[1] // COLUMN_STEP
    cut = gray[:, : n * COLUMN_STEP].reshape(gray.shape[0], n, COLUMN_STEP)
    level = np.percentile(cut, 80, axis=2).astype(np.float32)
    level = cv2.GaussianBlur(level, (1, 0), sigmaX=0.1, sigmaY=PROFILE_SIGMA)
    clips = green[:, : n * COLUMN_STEP].reshape(green.shape[0], n, COLUMN_STEP).mean(axis=(0, 2)) > 0.05
    # Прищепка шире полосы — расширяем маску на соседей.
    clips = np.convolve(clips.astype(float), np.ones(5), mode="same") > 0
    xs = np.arange(n) * COLUMN_STEP + COLUMN_STEP / 2
    paper = float(np.percentile(level[300:], 90))
    return Levels(level=level, xs=xs, paper=paper, clips=clips)


def _shifted(level: np.ndarray, k: int) -> np.ndarray:
    """Уровень бумаги на ``k`` строк ниже (циклический сдвиг; края потом обнуляются).

    Args:
        level: Уровень бумаги (строки × столбцы).
        k: Сдвиг, строк (отрицательный — выше).

    Returns:
        Сдвинутый массив.
    """
    return np.roll(level, -k, axis=0)


def response_map(levels: Levels, candidate: Candidate, paper_below: bool) -> np.ndarray:
    """Отклик складки на каждой строке каждого столбца (больше — вероятнее складка), нормированный по кадру.

    Args:
        levels: Уровень бумаги.
        candidate: Вид отклика (A/B/C/BC).
        paper_below: E — обнулять отклик, если под строкой не бумага.

    Returns:
        Массив той же формы, что ``levels.level``; ≥ 0.
    """
    level = levels.level
    rows = level.shape[0]
    if candidate is Candidate.MINIMUM:
        out = levels.paper - level
    else:
        edge = _shifted(level, EDGE_ARM) - _shifted(level, -EDGE_ARM)
        ridge = (_shifted(level, RIDGE_ARM) + _shifted(level, -RIDGE_ARM)) / 2 - level
        ridge = np.maximum(ridge, 0)
        if candidate is Candidate.EDGE:
            out = edge
        elif candidate is Candidate.RIDGE:
            out = ridge
        else:
            out = np.maximum(edge, 0) / (np.percentile(np.maximum(edge, 0), 99) + 1e-6) + ridge / (
                np.percentile(ridge, 99) + 1e-6
            )
    out = np.maximum(out, 0)
    # Края по y — артефакт сдвига.
    out[: RIDGE_ARM + 1] = 0
    out[rows - RIDGE_ARM - 1 :] = 0
    # Складка не бывает ниже ``FOLD_ROWS``: там уже содержимое полосы (фото, рамки).
    out[FOLD_ROWS:] = 0
    if paper_below:
        lo, hi = PAPER_BELOW
        below = np.stack([np.median(level[min(rows - 1, y + lo) : min(rows, y + hi)], axis=0) for y in range(rows)])
        out = np.where(below >= PAPER_SHARE * levels.paper, out, 0)
    return out / (np.percentile(out, 99.5) + 1e-6)


def apply_boxes(response: np.ndarray, xs: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> np.ndarray:
    """Штраф отклика внутри рамок блоков разметки: складка не проходит сквозь блок текста или картинки.

    Args:
        response: Карта отклика (строки × столбцы).
        xs: Центры столбцов, px кадра.
        boxes: Рамки ``(x0, y0, x1, y1)`` в пикселях кадра.

    Returns:
        Новая карта (вход не меняется).
    """
    out = response.copy()
    for x0, y0, x1, y1 in boxes:
        cols = (xs >= x0) & (xs <= x1)
        r0, r1 = int(max(0, y0 + BOX_PAD)), int(min(out.shape[0], y1 - BOX_PAD))
        if r1 > r0 and cols.any():
            out[r0:r1, cols] -= BOX_PENALTY
    return out


# --- Сборка в линию ----------------------------------------------------------------------------------------------


def column_points(response: np.ndarray, levels: Levels, topmost: bool, clips: bool) -> tuple[np.ndarray, np.ndarray]:
    """Точка складки в каждом столбце: сильнейший отклик или (D) самый верхний сильный.

    Args:
        response: Карта отклика.
        levels: Уровень бумаги (центры столбцов, прищепки).
        topmost: D — самый верхний локальный максимум с откликом ≥ ``TOP_SHARE`` максимума столбца.
        clips: F — без столбцов с прищепками.

    Returns:
        (x точек, y точек) — только столбцы с откликом больше 0.15.
    """
    xs, ys = [], []
    for c in range(response.shape[1]):
        if clips and levels.clips[c]:
            continue
        col = response[:, c]
        peak = float(col.max())
        if peak < 0.15:
            continue
        if topmost:
            maxima = np.where((col[1:-1] >= col[:-2]) & (col[1:-1] >= col[2:]) & (col[1:-1] >= TOP_SHARE * peak))[0] + 1
            y = int(maxima[0]) if len(maxima) else int(col.argmax())
        else:
            y = int(col.argmax())
        xs.append(levels.xs[c])
        ys.append(y)
    return np.asarray(xs, float), np.asarray(ys, float)


def fit_parabola(xs: np.ndarray, ys: np.ndarray) -> np.ndarray | None:
    """Итеративная парабола МНК с отбросом выбросов (как в ``single_page_gutter.find_fold``).

    Args:
        xs, ys: Точки.

    Returns:
        Коэффициенты ``np.polyval`` или None (мало точек).
    """
    if len(xs) < 8:
        return None
    ok = np.ones(len(xs), bool)
    coef = np.polyfit(xs, ys, 2)
    for _ in range(3):
        residual = np.abs(np.polyval(coef, xs) - ys)
        ok = residual < max(25.0, 3.0 * float(np.median(residual[ok])))
        if ok.sum() < 8:
            return None
        coef = np.polyfit(xs[ok], ys[ok], 2)
    return coef


def fit_ransac(xs: np.ndarray, ys: np.ndarray, tol: float = 6.0, trials: int = 400, seed: int = 0) -> np.ndarray | None:
    """RANSAC-парабола: тройки точек, самое большое согласие в пределах ``tol``, затем МНК по согласным.

    Args:
        xs, ys: Точки.
        tol: Порог согласия, px.
        trials: Число проб.
        seed: Зерно генератора (воспроизводимость).

    Returns:
        Коэффициенты ``np.polyval`` или None.
    """
    if len(xs) < 8:
        return None
    rng = np.random.default_rng(seed)
    best, best_count = None, 0
    span = xs.max() - xs.min()
    for _ in range(trials):
        pick = rng.choice(len(xs), 3, replace=False)
        if np.ptp(xs[pick]) < span / 4:
            continue
        coef = np.polyfit(xs[pick], ys[pick], 2)
        # Складка — почти прямая: кривизна больше 50 px прогиба на ширину кадра отбрасывается.
        if abs(coef[0]) * (span / 2) ** 2 > 50:
            continue
        count = int((np.abs(np.polyval(coef, xs) - ys) < tol).sum())
        if count > best_count:
            best, best_count = coef, count
    if best is None:
        return None
    inliers = np.abs(np.polyval(best, xs) - ys) < tol
    return np.polyfit(xs[inliers], ys[inliers], 2)


def best_path(response: np.ndarray, skip: np.ndarray | None = None) -> np.ndarray:
    """Путь минимальной стоимости слева направо: по строке на столбец (динамическое программирование).

    Стоимость клетки — минус отклик; переход между соседними столбцами — штраф ``PATH_JUMP_COST`` за пиксель
    скачка, скачок не больше ``PATH_MAX_JUMP``. Столбцы из ``skip`` (прищепки) идут с нулевым откликом — путь
    проходит их по продолжению соседей.

    Args:
        response: Карта отклика (строки × столбцы).
        skip: Маска столбцов без отклика.

    Returns:
        Строка пути в каждом столбце.
    """
    rows, cols = response.shape
    cost = -response.astype(np.float64)
    if skip is not None:
        cost[:, skip] = 0.0
    jumps = np.arange(-PATH_MAX_JUMP, PATH_MAX_JUMP + 1)
    total = cost[:, 0].copy()
    back = np.zeros((rows, cols), np.int32)
    for c in range(1, cols):
        best = np.full(rows, np.inf)
        arg = np.zeros(rows, np.int32)
        for d in jumps:
            # Из строки y-d прошлого столбца в строку y текущего.
            prev = np.full(rows, np.inf)
            if d >= 0:
                prev[d:] = total[: rows - d] if d else total
            else:
                prev[:d] = total[-d:]
            cand = prev + PATH_JUMP_COST * abs(d) / COLUMN_STEP * 4
            better = cand < best
            best[better] = cand[better]
            arg[better] = d
        total = best + cost[:, c]
        back[:, c] = arg
    path = np.zeros(cols, np.int32)
    path[-1] = int(np.argmin(total))
    for c in range(cols - 1, 0, -1):
        path[c - 1] = path[c] - back[path[c], c]
    return path


def smooth_path(xs: np.ndarray, path: np.ndarray, window: int = 9) -> np.ndarray:
    """Сглаживание пути скользящей медианой и средним (окно в столбцах).

    Args:
        xs: Центры столбцов.
        path: Строки пути.
        window: Окно, столбцов.

    Returns:
        Сглаженные строки (float).
    """
    half = window // 2
    padded = np.pad(path.astype(float), half, mode="edge")
    median = np.array([np.median(padded[i : i + window]) for i in range(len(path))])
    return np.convolve(np.pad(median, half, mode="edge"), np.ones(window) / window, mode="valid")


@dataclass(frozen=True)
class FoldLine:
    """Найденная линия: строка складки для каждого столбца кадра.

    Attributes:
        rows: Строка складки для каждого x кадра (float), NaN — линия не найдена.
        points_x, points_y: Точки-кандидаты, по которым строилась линия (для оверлея).
    """

    rows: np.ndarray
    points_x: np.ndarray
    points_y: np.ndarray


def find_line(
    rgb: np.ndarray, variant: Variant, boxes: list[tuple[float, float, float, float]] | None = None, levels=None
) -> FoldLine:
    """Линия сгиба кадра вариантом ``variant`` (с рамками разметки, если даны).

    Args:
        rgb: Кадр RGB uint8, корешок сверху.
        variant: Вариант.
        boxes: Рамки блоков разметки в пикселях кадра (штраф отклика внутри них) или None.
        levels: Готовый ``column_levels`` (чтобы не считать заново для каждого варианта) или None.

    Returns:
        ``FoldLine``.
    """
    levels = levels if levels is not None else column_levels(rgb)
    response = response_map(levels, variant.candidate, variant.paper_below)
    if boxes:
        response = apply_boxes(response, levels.xs, boxes)
    width = rgb.shape[1]
    grid = np.arange(width)
    if variant.assembly in (Assembly.PATH, Assembly.PATH_SMOOTH):
        path = best_path(response, levels.clips if variant.clips else None).astype(float)
        if variant.assembly is Assembly.PATH_SMOOTH:
            path = smooth_path(levels.xs, path)
        return FoldLine(np.interp(grid, levels.xs, path), levels.xs, path)
    px, py = column_points(response, levels, variant.topmost, variant.clips)
    coef = fit_parabola(px, py) if variant.assembly is Assembly.PARABOLA else fit_ransac(px, py)
    rows = np.polyval(coef, grid) if coef is not None else np.full(width, np.nan)
    return FoldLine(rows, px, py)


# --- Ориентация: сторона корешка на правильно повёрнутом (по EXIF) кадре -----------------------------------------


class SpineSide(str, Enum):
    """Сторона корешка на кадре, повёрнутом по EXIF (как полосу видит человек).

    Ядро детектора (``column_levels`` … ``find_line``) работает в канонической системе «корешок сверху»; кадр
    поворачивается туда :func:`to_spine_top`, линия возвращается обратно :func:`points_from_spine_top`.
    ЭГ 1985 10-12: корешок всегда у верхнего края сенсора, а камеру поворачивали между полосами — у ``_1``
    EXIF Orientation 8 (корешок слева), у ``_2`` — 6 (справа); сторона берётся из EXIF (:func:`spine_side_from_exif`).
    """

    TOP = "top"
    RIGHT = "right"
    BOTTOM = "bottom"
    LEFT = "left"


# Поворот OpenCV, приводящий сторону корешка наверх (None — уже сверху).
_TO_TOP = {
    SpineSide.TOP: None,
    SpineSide.RIGHT: cv2.ROTATE_90_COUNTERCLOCKWISE,
    SpineSide.BOTTOM: cv2.ROTATE_180,
    SpineSide.LEFT: cv2.ROTATE_90_CLOCKWISE,
}


# Куда попадает верхний край снимка «как снято» после поворота по EXIF Orientation (1 — как есть, 3 — 180°,
# 6 — повернуть на 90° по часовой стрелке, 8 — на 90° против). Зеркальные теги 2/4/5/7 у камер не встречаются.
_RAW_TOP_AFTER_EXIF = {1: SpineSide.TOP, 3: SpineSide.BOTTOM, 6: SpineSide.RIGHT, 8: SpineSide.LEFT}
# Код ``flip`` LibRaw (rawpy ``raw.sizes.flip``) → тег EXIF Orientation.
LIBRAW_FLIP_TO_EXIF = {0: 1, 3: 3, 5: 8, 6: 6}


def spine_side_from_exif(orientation: int, raw_side: SpineSide = SpineSide.TOP) -> SpineSide:
    """Сторона корешка на кадре, повёрнутом по EXIF, если на снимке «как снято» корешок у стороны ``raw_side``.

    Args:
        orientation: Тег EXIF Orientation (1, 3, 6, 8).
        raw_side: Сторона корешка на снимке без поворота (ЭГ 1985 — верх сенсора).

    Returns:
        Сторона корешка на кадре по EXIF.

    Raises:
        KeyError: Тег зеркальный или неизвестный.
    """
    order = [SpineSide.TOP, SpineSide.RIGHT, SpineSide.BOTTOM, SpineSide.LEFT]
    # Поворот по EXIF в четвертях оборота по часовой стрелке: куда уходит верх снимка.
    quarter = order.index(_RAW_TOP_AFTER_EXIF[orientation])
    return order[(order.index(raw_side) + quarter) % 4]


def to_spine_top(img: np.ndarray, side: SpineSide) -> np.ndarray:
    """Поворачивает кадр (повёрнутый по EXIF) так, чтобы корешок оказался сверху — в каноническую систему ядра.

    Args:
        img: Кадр (H, W[, C]) в ориентации по EXIF.
        side: Где на нём корешок.

    Returns:
        Повёрнутый кадр (без копии, если поворот не нужен).
    """
    code = _TO_TOP[side]
    return img if code is None else cv2.rotate(img, code)


def points_from_spine_top(
    xs: np.ndarray, ys: np.ndarray, side: SpineSide, upright_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Точки канонической системы («корешок сверху») → координаты кадра, повёрнутого по EXIF.

    Args:
        xs: x в канонической системе.
        ys: y в канонической системе.
        side: Сторона корешка на кадре по EXIF.
        upright_shape: (H, W) кадра по EXIF.

    Returns:
        (x, y) в кадре по EXIF.
    """
    h, w = upright_shape
    if side is SpineSide.TOP:
        return xs, ys
    if side is SpineSide.LEFT:
        return ys, h - 1 - xs
    if side is SpineSide.RIGHT:
        return w - 1 - ys, xs
    return w - 1 - xs, h - 1 - ys


def points_to_spine_top(
    xs: np.ndarray, ys: np.ndarray, side: SpineSide, upright_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Точки кадра по EXIF → каноническая система «корешок сверху» (обратное к :func:`points_from_spine_top`).

    Args:
        xs: x в кадре по EXIF.
        ys: y в кадре по EXIF.
        side: Сторона корешка на кадре по EXIF.
        upright_shape: (H, W) кадра по EXIF.

    Returns:
        (x, y) в канонической системе.
    """
    h, w = upright_shape
    if side is SpineSide.TOP:
        return xs, ys
    if side is SpineSide.LEFT:
        return h - 1 - ys, xs
    if side is SpineSide.RIGHT:
        return ys, w - 1 - xs
    return w - 1 - xs, h - 1 - ys


def boxes_to_spine_top(
    boxes: list[tuple[float, float, float, float]], side: SpineSide, upright_shape: tuple[int, int]
) -> list[tuple[float, float, float, float]]:
    """Рамки ``(x0, y0, x1, y1)`` кадра по EXIF → каноническая система.

    Args:
        boxes: Рамки в кадре по EXIF.
        side: Сторона корешка.
        upright_shape: (H, W) кадра по EXIF.

    Returns:
        Рамки в канонической системе (снова упорядоченные min/max).
    """
    out = []
    for x0, y0, x1, y1 in boxes:
        xs, ys = points_to_spine_top(np.array([x0, x1], float), np.array([y0, y1], float), side, upright_shape)
        out.append((float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())))
    return out


@dataclass(frozen=True)
class OrientedFold:
    """Линия сгиба на кадре по EXIF: каноническая ``FoldLine`` и та же линия точками в координатах кадра.

    Attributes:
        side: Сторона корешка.
        line: Линия в канонической системе (``rows`` — по столбцу канонического кадра).
        points: (N, 2) точки линии (x, y) в кадре по EXIF; без NaN.
    """

    side: SpineSide
    line: FoldLine
    points: np.ndarray


def find_line_oriented(
    rgb: np.ndarray, side: SpineSide, variant: Variant, boxes: list[tuple[float, float, float, float]] | None = None
) -> OrientedFold:
    """Линия сгиба на кадре, повёрнутом по EXIF, с корешком на стороне ``side``.

    Args:
        rgb: Кадр RGB uint8 в ориентации по EXIF.
        side: Сторона корешка.
        variant: Вариант детектора (боевой — ``C+F+2`` со строками-запретом).
        boxes: Рамки строк-запрета в координатах кадра по EXIF или None.

    Returns:
        ``OrientedFold``.
    """
    canon = to_spine_top(rgb, side)
    canon_boxes = boxes_to_spine_top(boxes, side, rgb.shape[:2]) if boxes else None
    line = find_line(canon, variant, canon_boxes)
    xs = np.arange(len(line.rows), dtype=float)
    ok = np.isfinite(line.rows)
    ux, uy = points_from_spine_top(xs[ok], line.rows[ok], side, rgb.shape[:2])
    return OrientedFold(side, line, np.stack([ux, uy], axis=1))


# --- Оверлей для отсмотра ----------------------------------------------------------------------------------------

# Куски кадра по ширине для оверлея 1:1 (с перекрытием), px превью, и высота зоны сгиба.
OVERLAY_PIECES, OVERLAY_ROWS = ((0, 1550), (1430, 2980), (2860, 4416)), 330


def overlay_pieces(
    rgb: np.ndarray, stem: str, lines: list[tuple[str, np.ndarray, tuple[int, int, int]]], etalon=None
) -> list[np.ndarray]:
    """Оверлеи зоны сгиба 1:1 тремя кусками: линии вариантов, крестики эталона; шапка и легенда — в полях.

    Args:
        rgb: Кадр RGB uint8, корешок сверху.
        stem: Имя кадра (в шапку).
        lines: (подпись, строка линии для каждого x кадра, цвет BGR) по варианту.
        etalon: Точки эталона ``[(x, y), …]`` или None.

    Returns:
        Картинки BGR по кускам ``OVERLAY_PIECES``.
    """
    from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed

    canvas = cv2.cvtColor(rgb[:OVERLAY_ROWS], cv2.COLOR_RGB2BGR)
    xs = np.arange(rgb.shape[1])
    for _, rows, colour in lines:
        ok = np.isfinite(rows)
        points = np.stack([xs[ok], rows[ok]], 1)[::4].astype(np.int32)
        cv2.polylines(canvas, [points], False, colour, 2)
    for x, y in etalon or []:
        # Крестик эталона — белый с тёмной обводкой, виден и на бумаге, и на фото.
        cv2.drawMarker(canvas, (int(x), int(y)), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 22, 4)
        cv2.drawMarker(canvas, (int(x), int(y)), (0, 0, 0), cv2.MARKER_TILTED_CROSS, 18, 1)
    legend = [LegendEntry(name, colour) for name, _, colour in lines]
    if etalon:
        legend.append(LegendEntry("эталон глазами (крестик)", (0, 0, 0), style=SampleStyle.BOX))
    out = []
    for x0, x1 in OVERLAY_PIECES:
        header = [f"{stem}: линия сгиба, x {x0}–{x1} (1:1 превью)"]
        out.append(framed(np.ascontiguousarray(canvas[:, x0:x1]), header, legend))
    return out


class BoxSource(str, Enum):
    """Какие рамки разметки (``fold-layout``) штрафуют отклик складки."""

    NONE = "нет"
    LINES = "строки"
    LINES_SMALL_LAYOUT = "строки+layout_мал"
    LINES_SMALL_DS = "строки+ds_мал"
    SMALL_LAYOUT = "layout_мал"
    SMALL_DS = "ds_мал"
    LAYOUT = "layout"
    DS = "ds"


# Блок разметки «малый»: не выше стольких строк кадра (большие блоки накрывают и загиб, и полосу — ложный запрет).
SMALL_BLOCK = 250


def select_boxes(record: dict, source: BoxSource) -> list[tuple[float, float, float, float]] | None:
    """Рамки из кэша разметки кадра для источника ``source``.

    Args:
        record: Запись ``layout/<кадр>.json`` (``surya_layout``, ``surya_lines``, ``ds_blocks``: ``[x0, y0, x1, y1,
            метка]`` в пикселях кадра).
        source: Источник рамок.

    Returns:
        Рамки ``(x0, y0, x1, y1)`` или None (без рамок).
    """
    if source is BoxSource.NONE:
        return None
    lines = [tuple(b[:4]) for b in record["surya_lines"]]
    layout = [tuple(b[:4]) for b in record["surya_layout"]]
    ds = [tuple(b[:4]) for b in record["ds_blocks"]]
    return {
        BoxSource.LINES: lines,
        BoxSource.LINES_SMALL_LAYOUT: lines + [b for b in layout if b[3] - b[1] <= SMALL_BLOCK],
        BoxSource.LINES_SMALL_DS: lines + [b for b in ds if b[3] - b[1] <= SMALL_BLOCK],
        BoxSource.SMALL_LAYOUT: [b for b in layout if b[3] - b[1] <= SMALL_BLOCK],
        BoxSource.SMALL_DS: [b for b in ds if b[3] - b[1] <= SMALL_BLOCK],
        BoxSource.LAYOUT: layout,
        BoxSource.DS: ds,
    }[source]


def parse_variant(name: str) -> Variant:
    """Вариант по короткому имени (``Variant.name``), например ``C+F+2`` или ``A+DEF+1``.

    Args:
        name: Имя: отклик (A/B/C/BC), необязательные флаги D/E/F, номер сборки 1–4 через «+».

    Returns:
        ``Variant``.

    Raises:
        ValueError: Имя не разбирается.
    """
    parts = name.split("+")
    if len(parts) not in (2, 3):
        raise ValueError(f"не разобрать вариант {name!r}")
    flags = parts[1] if len(parts) == 3 else ""
    return Variant(Candidate(parts[0]), "D" in flags, "E" in flags, "F" in flags, Assembly(parts[-1]))
