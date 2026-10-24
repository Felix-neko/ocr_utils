"""Местный горизонтальный масштаб строк у корешка на выпрямленной странице: строки по краске, окна вдоль строки от края текста у корешка наружу и пять мер в окне (шаг штрихов по автокорреляции, шаг вертикальных штрихов, ширина компонент, толщина вертикального штриха, высота строчных) против той же строки вдали от корешка."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from ocr_utils.units import mm_to_px, mm_to_px_float, px_to_mm


class Side(StrEnum):
    """Страница разворота."""

    LEFT = "L"
    RIGHT = "R"


class Method(StrEnum):
    """Мера местного масштаба в окне строки (отношение к той же мере вдали от корешка)."""

    ACF = "acf"  # шаг штрихов: первый пик автокорреляции профиля краски по столбцам в поясе строчных
    STEM = "stem"  # шаг вертикальных штрихов: медиана расстояний между соседними «полными» столбцами внутри слова
    COMP = "comp"  # ширина связных компонент высотой со строчную
    STROKE = "stroke"  # толщина вертикального штриха: длина горизонтальных отрезков краски в середине пояса
    HEIGHT = "height"  # высота строчных (пояс строки): вертикальный масштаб, для поправки на перспективу


# Меры горизонтального масштаба (высота строчных — вертикальная, она не сжимается сгибом).
HORIZONTAL = (Method.ACF, Method.STEM, Method.COMP, Method.STROKE)


@dataclass(frozen=True)
class MeasureParams:
    """Параметры меры.

    Поля:
        bg_kernel_mm: Окно фона (серое закрытие), мм: шире буквы, у́же просвета между строками. 2.5 мм = 43 px при
            17.3 px/мм (ХиП 1991).
        ink_ratio: Пиксель — краска, если темнее фона в эту долю.
        glyph_mm: Высоты компонент-букв для оценки высоты строчных, мм.
        window_mm: Ширина окна вдоль строки, мм (5 мм — 3 строчные ХиП, 7–8 штрихов).
        near_step_mm: Шаг окон у корешка, мм.
        far_step_mm: Шаг окон дальше ``far_mm``, мм.
        far_mm: Опора строки — окна дальше этого от края текста у корешка, мм.
        reach_mm: Окна не дальше этого от края текста, мм.
        stem_fill: Столбец — вертикальный штрих, если краска покрывает эту долю пояса строчных.
        min_fill: Окно в счёт, если средняя доля краски в поясе строчных не меньше этой (не пробел).
    """

    bg_kernel_mm: float = 2.5
    ink_ratio: float = 0.62
    glyph_mm: tuple[float, float] = (0.6, 4.0)
    window_mm: float = 5.0
    near_step_mm: float = 1.0
    far_step_mm: float = 2.5
    far_mm: float = 25.0
    reach_mm: float = 60.0
    stem_fill: float = 0.7
    min_fill: float = 0.1


@dataclass(frozen=True)
class TextLine:
    """Строка страницы на выпрямленном холсте (строки после второго преобразования горизонтальны).

    Поля:
        side: Страница.
        core_top: Верх пояса строчных, px холста.
        core_bottom: Низ пояса строчных (включительно), px.
        x_lo: Левый край краски строки, px.
        x_hi: Правый край краски строки (включительно), px.
    """

    side: Side
    core_top: int
    core_bottom: int
    x_lo: int
    x_hi: int

    @property
    def core_height(self) -> int:
        """Высота пояса строчных, px."""
        return self.core_bottom - self.core_top + 1


def ink_mask(gray: np.ndarray, px_per_mm: float, params: MeasureParams) -> np.ndarray:
    """Краска: пиксели темнее фона (серое закрытие с окном шире буквы) в долю ``ink_ratio``.

    Фон считается на ¼ копии: закрытие окном 2.5 мм на полном кадре — секунды.

    Args:
        gray: Полутоновый холст uint8.
        px_per_mm: Плотность, px/мм.
        params: Параметры.

    Returns:
        Маска краски bool.
    """
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    side = max(3, int(round(params.bg_kernel_mm * px_per_mm * 0.25)) | 1)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (side, side)))
    bg = cv2.GaussianBlur(bg, (0, 0), side / 2)
    bg = cv2.resize(bg, gray.shape[::-1], interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return gray.astype(np.float32) < params.ink_ratio * bg


def glyph_mask(ink: np.ndarray, px_per_mm: float, params: MeasureParams) -> tuple[np.ndarray, float]:
    """Компоненты краски высотой с букву и медианная высота буквы.

    Args:
        ink: Маска краски страницы.
        px_per_mm: Плотность, px/мм.
        params: Параметры.

    Returns:
        (маска букв bool, медианная высота буквы в px или NaN — букв нет).
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    w, h = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    lo, hi = params.glyph_mm[0] * px_per_mm, params.glyph_mm[1] * px_per_mm
    keep = (h >= lo) & (h <= hi) & (w >= 2) & (w <= 1.5 * hi)
    keep[0] = False
    if keep.sum() < 50:
        return np.zeros_like(ink), float("nan")
    return keep[labels], float(np.median(h[keep]))


def page_lines(ink: np.ndarray, x_lo: int, x_hi: int, px_per_mm: float, side: Side,
               params: MeasureParams) -> tuple[list[TextLine], float]:  # fmt: skip
    """Строки страницы: пики профиля букв по высоте, пояс строчных — ряды вокруг пика с профилем от 0.45 пика.

    Строки после второго преобразования горизонтальны (размах базовой линии 0.1–0.2 мм), поэтому строка — ряд холста.

    Args:
        ink: Маска краски холста.
        x_lo: Левый край страницы на холсте, px.
        x_hi: Правый край страницы (исключительно), px.
        px_per_mm: Плотность, px/мм.
        side: Страница.
        params: Параметры.

    Returns:
        (строки, медианная высота буквы страницы в px или NaN).
    """
    page = ink[:, x_lo:x_hi]
    glyphs, xh = glyph_mask(page, px_per_mm, params)
    if not np.isfinite(xh):
        return [], xh
    profile = gaussian_filter1d(glyphs.sum(axis=1).astype(np.float64), 0.15 * xh)
    floor = 0.15 * np.percentile(profile[profile > 0], 95) if (profile > 0).any() else 1.0
    peaks, _ = find_peaks(profile, height=floor, distance=max(1, int(1.4 * xh)))
    lines = []
    for peak in peaks:
        level = 0.45 * profile[peak]
        top, bottom = peak, peak
        while top > 0 and profile[top - 1] >= level:
            top -= 1
        while bottom < len(profile) - 1 and profile[bottom + 1] >= level:
            bottom += 1
        height = bottom - top + 1
        if not 0.5 * xh <= height <= 1.6 * xh:
            continue
        cols = np.nonzero(glyphs[top : bottom + 1].any(axis=0))[0]
        if len(cols) < xh * 5:
            continue
        lines.append(TextLine(side, int(top), int(bottom), int(cols[0] + x_lo), int(cols[-1] + x_lo)))
    return lines, xh


def acf_pitch(fill: np.ndarray, lo: float, hi: float) -> float:
    """Шаг штрихов: первый пик автокорреляции профиля краски после первого спада.

    Args:
        fill: Доля краски по столбцам окна.
        lo: Наименьший допустимый шаг, px.
        hi: Наибольший допустимый шаг, px.

    Returns:
        Шаг, px (с параболическим уточнением вершины), или NaN.
    """
    centred = fill - fill.mean()
    if not np.any(centred):
        return float("nan")
    acf = np.correlate(centred, centred, mode="full")[len(centred) - 1 :]
    acf = acf / acf[0]
    top = min(len(acf) - 1, int(hi) + 1)
    lag = 1
    while lag < top and acf[lag] <= acf[lag - 1]:
        lag += 1
    while lag < top and acf[lag] >= acf[lag - 1]:
        lag += 1
    peak = lag - 1
    if peak < lo or peak >= top - 1 or acf[peak] < 0.05:
        return float("nan")
    a, b, c = acf[peak - 1], acf[peak], acf[peak + 1]
    denom = a - 2 * b + c
    return float(peak + (0.5 * (a - c) / denom if abs(denom) > 1e-9 else 0.0))


def runs_of(flags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Отрезки истинных значений одномерного массива.

    Args:
        flags: bool[n].

    Returns:
        (начала, концы исключительно).
    """
    padded = np.concatenate([[False], flags, [False]]).astype(np.int8)
    diff = np.diff(padded)
    return np.nonzero(diff == 1)[0], np.nonzero(diff == -1)[0]


def stem_pitch(fill: np.ndarray, stem_fill: float, max_gap: float) -> float:
    """Шаг вертикальных штрихов внутри слова: медиана расстояний между центрами соседних штрихов.

    Args:
        fill: Доля краски по столбцам окна в поясе строчных.
        stem_fill: Столбец — штрих, если доля не меньше этой.
        max_gap: Наибольший просвет между штрихами одного слова, px.

    Returns:
        Шаг, px, или NaN (меньше трёх пар штрихов).
    """
    starts, ends = runs_of(fill >= stem_fill)
    if len(starts) < 4:
        return float("nan")
    centres = (starts + ends - 1) / 2
    gaps = starts[1:] - ends[:-1]
    steps = np.diff(centres)[gaps < max_gap]
    return float(np.median(steps)) if len(steps) >= 3 else float("nan")


def stroke_width(core: np.ndarray, max_run: float) -> float:
    """Толщина вертикального штриха: медианная длина горизонтальных отрезков краски в средней трети пояса строчных.

    Args:
        core: Краска окна в поясе строчных (ряды × столбцы).
        max_run: Отрезки длиннее (горизонтальные черты, слипшиеся буквы) не в счёт, px.

    Returns:
        Толщина, px, или NaN.
    """
    third = max(1, core.shape[0] // 3)
    lengths = []
    for row in core[third : core.shape[0] - third]:
        starts, ends = runs_of(row)
        inner = (starts > 0) & (ends < len(row))  # отрезки, обрезанные краем окна, не в счёт
        runs = (ends - starts)[inner]
        lengths.append(runs[runs <= max_run])
    lengths = np.concatenate(lengths) if lengths else np.zeros(0)
    # Среднее, а не медиана: длины целые (3–5 px), медиана квантуется и не видит сжатия на 10–20 %.
    return float(np.mean(lengths)) if len(lengths) >= 6 else float("nan")


def local_core(band: np.ndarray) -> tuple[int, int, float] | None:
    """Пояс строчных в окне: ряды вокруг максимума профиля краски с профилем от 0.45 максимума.

    Args:
        band: Краска окна в расширенном поясе строки.

    Returns:
        (верхний ряд, нижний ряд включительно — в рядах ``band``; высота пояса с долями пикселя по линейной
        интерполяции пересечения уровня 0.45 на обоих краях) или None — краски нет.
    """
    profile = gaussian_filter1d(band.sum(axis=1).astype(np.float64), 1.0)
    if profile.max() <= 0:
        return None
    peak = int(np.argmax(profile))
    level = 0.45 * profile[peak]
    top, bottom = peak, peak
    while top > 0 and profile[top - 1] >= level:
        top -= 1
    while bottom < len(profile) - 1 and profile[bottom + 1] >= level:
        bottom += 1
    # Доли пикселя: где профиль пересекает уровень между крайним рядом пояса и соседним снаружи.
    top_f = top - (
        (profile[top] - level) / (profile[top] - profile[top - 1])
        if top > 0 and profile[top] > profile[top - 1]
        else 0.5
    )
    bottom_f = bottom + (
        (profile[bottom] - level) / (profile[bottom] - profile[bottom + 1])
        if bottom < len(profile) - 1 and profile[bottom] > profile[bottom + 1]
        else 0.5
    )
    return top, bottom, float(bottom_f - top_f)


def comp_widths(band: np.ndarray, core_h: float) -> tuple[np.ndarray, np.ndarray]:
    """Связные компоненты пояса строки высотой со строчную: центры и ширины.

    Args:
        band: Краска строки в расширенном поясе (ряды × столбцы холста строки).
        core_h: Высота строчных строки, px.

    Returns:
        (x центров в столбцах ``band``, ширины), px.
    """
    _, _, stats, centres = cv2.connectedComponentsWithStats(band.astype(np.uint8), connectivity=8)
    w, h = stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]
    keep = (h >= 0.7 * core_h) & (h <= 1.25 * core_h) & (w >= 2) & (w <= 2.0 * core_h)
    return centres[1:, 0][keep], w[keep].astype(np.float64)


def line_windows(ink: np.ndarray, line: TextLine, spine_x: float, px_per_mm: float,
                 params: MeasureParams) -> list[dict]:  # fmt: skip
    """Окна строки от края текста у корешка наружу и меры в каждом.

    Args:
        ink: Маска краски холста.
        line: Строка.
        spine_x: X линии корешка на холсте, px.
        px_per_mm: Плотность, px/мм.
        params: Параметры.

    Returns:
        Строки таблицы: ``d_mm`` (центр окна от корешка), ``from_edge_mm`` (от края текста у корешка), значения мер
        (px) по ``Method`` (ширина компонент и толщина штриха — средние: целые длины медиана квантует).
    """
    xh = float(line.core_height)
    pad = int(round(0.5 * xh))
    top, bottom = max(0, line.core_top - pad), min(ink.shape[0], line.core_bottom + pad + 1)
    band = ink[top:bottom, line.x_lo : line.x_hi + 1]
    comp_x, comp_w = comp_widths(band, xh)
    half = mm_to_px_float(params.window_mm, px_per_mm * 25.4) / 2
    # Край текста у корешка — ближний к корешку конец строки; окна идут от него наружу.
    toward = line.side is Side.LEFT
    edge = line.x_hi if toward else line.x_lo
    edge_mm = px_to_mm(abs(spine_x - edge), px_per_mm * 25.4)
    # Внешний (дальний от корешка) конец строки — для внешнего края текста стороны.
    outer_mm = px_to_mm(abs(spine_x - (line.x_lo if toward else line.x_hi)), px_per_mm * 25.4)
    offsets, offset = [], 0.0
    while offset <= params.reach_mm:
        offsets.append(offset)
        offset += params.near_step_mm if offset < params.far_mm else params.far_step_mm
    rows = []
    for offset in offsets:
        shift = offset * px_per_mm + half
        centre = edge - shift if toward else edge + shift
        a, b = int(round(centre - half)) - line.x_lo, int(round(centre + half)) - line.x_lo
        if a < 0 or b > band.shape[1]:
            break
        window = band[:, a:b]
        core = local_core(window)
        if core is None:
            continue
        c0, c1, core_h = core
        core_rows = window[c0 : c1 + 1]
        fill = core_rows.mean(axis=0)
        if fill.mean() < params.min_fill:
            continue
        inside = (comp_x >= a) & (comp_x < b)
        rows.append(
            {
                "y": (line.core_top + line.core_bottom) / 2,
                "x_centre": centre,
                "d_mm": px_to_mm(abs(centre - spine_x), px_per_mm * 25.4),
                "from_edge_mm": offset + params.window_mm / 2,
                "edge_mm": edge_mm,
                "outer_mm": outer_mm,
                Method.ACF.value: acf_pitch(fill, 0.15 * xh, 1.2 * xh),
                Method.STEM.value: stem_pitch(fill, params.stem_fill, 0.8 * xh),
                Method.COMP.value: float(np.mean(comp_w[inside])) if inside.sum() >= 2 else float("nan"),
                Method.STROKE.value: stroke_width(core_rows, 0.45 * xh),
                Method.HEIGHT.value: core_h,
            }
        )
    return rows


def page_windows(ink: np.ndarray, x_lo: int, x_hi: int, spine_x: float, side: Side, px_per_mm: float,
                 params: MeasureParams) -> pd.DataFrame:  # fmt: skip
    """Окна всех строк страницы с отношениями мер к опоре своей строки.

    Опора строки — медиана меры по окнам дальше ``far_mm`` от края текста (кегль, гарнитура, печать и свет у строки
    общие); строка без трёх опорных окон меры для этой меры не даёт.

    Args:
        ink: Маска краски холста.
        x_lo: Левый край страницы на холсте, px.
        x_hi: Правый край страницы (исключительно), px.
        spine_x: X линии корешка, px.
        side: Страница.
        px_per_mm: Плотность, px/мм.
        params: Параметры.

    Returns:
        Таблица окон: side, line, y, d_mm, from_edge_mm, edge_mm, значения мер и ``<мера>_ratio``.
    """
    lines, _ = page_lines(ink, x_lo, x_hi, px_per_mm, side, params)
    frames = []
    for index, line in enumerate(lines):
        rows = line_windows(ink, line, spine_x, px_per_mm, params)
        if not rows:
            continue
        table = pd.DataFrame(rows)
        table.insert(0, "line", index)
        table.insert(0, "side", side.value)
        far = table.from_edge_mm >= params.far_mm
        for method in Method:
            ref = table.loc[far, method.value].dropna()
            ratio = table[method.value] / float(ref.median()) if len(ref) >= 3 else np.nan
            table[f"{method.value}_ratio"] = ratio
        frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def spread_windows(gray: np.ndarray, rects: tuple, spine_x: float, px_per_mm: float,
                   params: MeasureParams) -> pd.DataFrame:  # fmt: skip
    """Окна строк обеих страниц выпрямленного разворота.

    Args:
        gray: Полутоновый холст uint8.
        rects: Прямоугольники левой и правой страницы (``coons.PageRect``).
        spine_x: X линии корешка, px.
        px_per_mm: Плотность, px/мм.
        params: Параметры.

    Returns:
        Таблица окон обеих страниц (пустая — строк нет).
    """
    ink = ink_mask(gray, px_per_mm, params)
    width = gray.shape[1]
    margin = mm_to_px(1.0, px_per_mm * 25.4)
    left = page_windows(ink, max(0, int(rects[0].x0) + margin), int(round(spine_x)), spine_x, Side.LEFT, px_per_mm,
                        params)  # fmt: skip
    right = page_windows(ink, int(round(spine_x)), min(width, int(rects[1].x1) - margin), spine_x, Side.RIGHT,
                         px_per_mm, params)  # fmt: skip
    return pd.concat([left, right], ignore_index=True)
