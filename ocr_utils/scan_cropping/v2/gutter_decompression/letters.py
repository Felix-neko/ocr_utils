"""Буквы строк выпрямленного разворота для меры сжатия у корешка: краска, строки (ряды холста после поля по тексту), компоненты высотой со строчную, ширина каждой буквы к медиане букв той же строки вдали от корешка."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass(frozen=True)
class LetterParams:
    """Параметры меры букв.

    Поля:
        bg_kernel_mm: Окно фона (серое закрытие), мм: шире буквы, у́же просвета между строками. 2.5 мм = 43 px при
            17.3 px/мм (ХиП 1991).
        ink_ratio: Пиксель — краска, если темнее фона в эту долю.
        glyph_mm: Высоты компонент-букв для оценки высоты строчных, мм.
        far_mm: Опора строки — буквы дальше этого от края текста у корешка, мм: дальше спадов сжатия пака (длина спада
            по буквам на ХиП 1991 — 1–8 мм).
        min_far_letters: Опорных букв в строке не меньше этого, иначе строка не в счёт (опора — медиана, кегль и
            гарнитура строки сокращаются).
    """

    bg_kernel_mm: float = 2.5
    ink_ratio: float = 0.62
    glyph_mm: tuple[float, float] = (0.6, 4.0)
    far_mm: float = 25.0
    min_far_letters: int = 12


@dataclass(frozen=True)
class TextLine:
    """Строка страницы на выпрямленном холсте (после поля по тексту строки горизонтальны).

    Поля:
        page: Страница (0 — левая, 1 — правая).
        core_top: Верх пояса строчных, px холста.
        core_bottom: Низ пояса строчных (включительно), px.
        x_lo: Левый край краски строки, px.
        x_hi: Правый край краски строки (включительно), px.
    """

    page: int
    core_top: int
    core_bottom: int
    x_lo: int
    x_hi: int

    @property
    def core_height(self) -> int:
        """Высота пояса строчных, px."""
        return self.core_bottom - self.core_top + 1


def ink_mask(gray: np.ndarray, px_per_mm: float, params: LetterParams) -> np.ndarray:
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


def glyph_mask(ink: np.ndarray, px_per_mm: float, params: LetterParams) -> tuple[np.ndarray, float]:
    """Компоненты краски высотой с букву и медианная высота буквы.

    Args:
        ink: Маска краски страницы.
        px_per_mm: Плотность, px/мм.
        params: Параметры.

    Returns:
        (маска букв bool, медианная высота буквы в px или NaN — букв меньше 50).
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    w, h = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    lo, hi = params.glyph_mm[0] * px_per_mm, params.glyph_mm[1] * px_per_mm
    keep = (h >= lo) & (h <= hi) & (w >= 2) & (w <= 1.5 * hi)
    keep[0] = False
    if keep.sum() < 50:
        return np.zeros_like(ink), float("nan")
    return keep[labels], float(np.median(h[keep]))


def page_lines(ink: np.ndarray, x_lo: int, x_hi: int, px_per_mm: float, page: int,
               params: LetterParams) -> list[TextLine]:  # fmt: skip
    """Строки страницы: пики профиля букв по высоте, пояс строчных — ряды вокруг пика с профилем от 0.45 пика.

    Строки после поля по тексту горизонтальны (размах базовой линии 0.1–0.2 мм), поэтому строка — ряд холста.

    Args:
        ink: Маска краски холста.
        x_lo: Левый край страницы на холсте, px.
        x_hi: Правый край страницы (исключительно), px.
        px_per_mm: Плотность, px/мм.
        page: Страница (0 — левая, 1 — правая).
        params: Параметры.

    Returns:
        Строки сверху вниз.
    """
    glyphs, xh = glyph_mask(ink[:, x_lo:x_hi], px_per_mm, params)
    if not np.isfinite(xh):
        return []
    # Профиль букв по рядам, сглаженный на долю высоты строчной; пики не ближе 1.4 строчной — разные строки.
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
        # Пояс не со строчную (заголовок крупным кеглем, слипшиеся строки) и короткие обрывки — не строки.
        if not 0.5 * xh <= bottom - top + 1 <= 1.6 * xh:
            continue
        cols = np.nonzero(glyphs[top : bottom + 1].any(axis=0))[0]
        cols = trim_stray(cols, xh, near_right=page == 0)
        if len(cols) < xh * 5:
            continue
        lines.append(TextLine(page, int(top), int(bottom), int(cols[0] + x_lo), int(cols[-1] + x_lo)))
    return lines


# Пятно у края строки со стороны корешка — сор, если оно у́же STRAY_WIDTH_XH строчной и отделено от остальной строки
# промежутком шире STRAY_GAP_XH строчной. Обрывки линии складки высотой с букву (ХиП 1991/01 IMG_0987 R: край строк
# в 0.15 мм от корешка вместо 4.7 мм, «буквы» шириной 2 px с отношением 0.08) уводили начало спада к корешку, и
# подгонка видела у края сжатие, которого нет. Межсловный пробел — до ~1 строчной, однобуквенное слово шире 0.5.
STRAY_WIDTH_XH = 0.5
STRAY_GAP_XH = 1.5


def trim_stray(cols: np.ndarray, xh: float, near_right: bool) -> np.ndarray:
    """Столбцы строки без отдельных узких пятен на конце со стороны корешка (обрывки линии складки, соринки).

    Args:
        cols: Возрастающие номера столбцов с краской букв в поясе строки.
        xh: Высота строчной страницы, px.
        near_right: Корешок справа (левая страница) — чистится правый конец, иначе левый.

    Returns:
        Столбцы без отброшенных пятен (те же, если отбрасывать нечего).
    """
    if len(cols) == 0:
        return cols
    # Пятна — серии столбцов без промежутков; конец со стороны корешка — первое пятно (с отражением для левой
    # страницы, чтобы чистить всегда с начала массива).
    work = -cols[::-1] if near_right else cols
    starts = np.r_[0, np.nonzero(np.diff(work) > 1)[0] + 1]
    ends = np.r_[starts[1:], len(work)]
    first = 0
    while first < len(starts) - 1:
        width = work[ends[first] - 1] - work[starts[first]] + 1
        gap = work[starts[first + 1]] - work[ends[first] - 1] - 1
        if width >= STRAY_WIDTH_XH * xh or gap <= STRAY_GAP_XH * xh:
            break
        first += 1
    kept = work[starts[first] :]
    return -kept[::-1] if near_right else kept


def comp_widths(band: np.ndarray, core_h: float) -> tuple[np.ndarray, np.ndarray]:
    """Связные компоненты пояса строки высотой со строчную: центры и ширины.

    Знаки препинания и дефисы не проходят по высоте, прописные с выносными — по верхнему пределу.

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


def line_letters(ink: np.ndarray, line: TextLine, spine_x: float, px_per_mm: float,
                 params: LetterParams) -> pd.DataFrame:  # fmt: skip
    """Буквы одной строки: положение от края строки у корешка и ширина к опоре строки.

    Опора — медиана ширины букв этой строки дальше ``far_mm`` от её края у корешка: кегль, гарнитура и печать строки
    сокращаются.

    Args:
        ink: Маска краски холста.
        line: Строка.
        spine_x: X линии корешка на холсте, px.
        px_per_mm: Плотность, px/мм.
        params: Параметры.

    Returns:
        Таблица букв: ``from_edge_mm`` (центр буквы от края строки у корешка), ``width_px``, ``ratio`` (к опоре
        строки), ``edge_mm`` (край строки у корешка от линии корешка), ``outer_mm`` (внешний край строки от линии
        корешка); пустая — опорных букв мало.
    """
    xh = float(line.core_height)
    pad = int(round(0.5 * xh))
    top, bottom = max(0, line.core_top - pad), min(ink.shape[0], line.core_bottom + pad + 1)
    centres, widths = comp_widths(ink[top:bottom, line.x_lo : line.x_hi + 1], xh)
    if len(widths) == 0:
        return pd.DataFrame()
    xs = centres + line.x_lo
    # Край строки у корешка — конец краски, ближний к корешку (у левой страницы правый), внешний — дальний.
    near_end, far_end = (line.x_hi, line.x_lo) if line.page == 0 else (line.x_lo, line.x_hi)
    from_edge = np.abs(xs - near_end) / px_per_mm
    far = from_edge >= params.far_mm
    if far.sum() < params.min_far_letters:
        return pd.DataFrame()
    reference = float(np.median(widths[far]))
    return pd.DataFrame({"from_edge_mm": from_edge, "width_px": widths, "ratio": widths / reference,
                         "edge_mm": abs(spine_x - near_end) / px_per_mm,
                         "outer_mm": abs(spine_x - far_end) / px_per_mm})  # fmt: skip


def page_letters(ink: np.ndarray, x_lo: int, x_hi: int, page: int, spine_x: float,
                 px_per_mm: float, params: LetterParams) -> pd.DataFrame:  # fmt: skip
    """Буквы всех строк одной страницы выпрямленного холста.

    Args:
        ink: Маска краски холста (:func:`ink_mask`).
        x_lo: Левый край страницы на холсте, px.
        x_hi: Правый край страницы (исключительно), px.
        page: Страница (0 — левая, 1 — правая).
        spine_x: X линии корешка, px.
        px_per_mm: Плотность, px/мм.
        params: Параметры.

    Returns:
        Таблица букв (:func:`line_letters`) с ``line`` (номер строки страницы) и ``y`` (центр пояса строчных, px);
        пустая — строк с опорой нет.
    """
    x_lo, x_hi = max(0, x_lo), min(ink.shape[1], x_hi)
    if x_hi - x_lo < 10:
        return pd.DataFrame()
    parts = []
    for index, line in enumerate(page_lines(ink, x_lo, x_hi, px_per_mm, page, params)):
        table = line_letters(ink, line, spine_x, px_per_mm, params)
        if len(table):
            table.insert(0, "line", index)
            table.insert(1, "y", (line.core_top + line.core_bottom) / 2)
            parts.append(table)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


__all__ = ["LetterParams", "TextLine", "ink_mask", "page_letters", "page_lines"]
