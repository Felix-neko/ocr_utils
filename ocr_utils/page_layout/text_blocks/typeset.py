"""Набор строки по боксам глифов — кегль, низ, одна базовая линия, разный набор у просвета: запреты сращивания подписи с заголовком, шапки с «Год издания», даты с оглавлением."""

from __future__ import annotations

import numpy as np

# Глифы ниже этой доли медианы (точки, запятые, тире) в кегль и низ не входят.
MARK_SHARE = 0.45
# Низы двух групп глифов ближе стольких меньших кеглей — одна базовая линия: логотип «50 лет» (1967/10)
# стоит на одной линии (2 px при кегле 16), подпись рядом с заголовком — нет (6–20 px).
BASELINE_SHIFT_KEGL = 0.25

# --- Разный набор у просвета строки (:func:`different_sets`) ---
# Кегль стороны просвета — по стольким ближайшим к нему глифам без мелочи.
SIDE_GLYPHS = 6
# Резать можно только крупный набор: крупная сторона от стольких пикселей рабочей копии (заголовки пака
# от 18 px, шапка — от 34, корпус ≈ 11). У корпуса кегль по медиане глифов — строчная, пробел при выключке
# по формату шире 1.5 строчной, низы через пробел расходятся от наклона и старинных цифр, уходящих под
# строку: резка строк корпуса давала +373 оси на 147 полосах из 300 случайных и резала «0,4% и т. д.»
# (стенд research/heading_merge, reports/text_blocks_heading_merge.md).
LARGE_KEGL_PX = 18.0
# Меньшая сторона — не мельче стольких пикселей: пыль и точки не кегль (1966/01 IMG_0039_1L).
MIN_KEGL_PX = 6.0
# Дальний просвет: шире стольких кеглей БОЛЬШЕЙ стороны при низах, разошедшихся больше чем на
# ``FAR_SHIFT_KEGL`` меньшего кегля, — разные строки. Месяц у «ТЕХНИЧЕСКОЕ» в шапке журнала: 2 кегля и 8.5 px
# низов при кегле 34; межсловный пробел заголовка — до кегля, шум низов — 1–2 px.
FAR_GAP_KEGL = 1.5
FAR_SHIFT_KEGL = 0.15
# Низы разошлись больше чем на полкегля — строки заведомо разные, и хватает кегля от 1.5 раза, самых
# высоких букв от 1.4 и просвета от кегля: «Г» (17 px) у «СНАБЖЕНИЕ» (54 px) через 30 px, 1967/02
# IMG_0054_2R; «август» (22) у «ТЕХНИЧЕСКОЕ» (43) через 56 px, 1968/08 IMG_0054_2R.
BIG_SHIFT_KEGL = 0.5
BIG_SHIFT_RATIO = 1.5
BIG_SHIFT_TALL_RATIO = 1.4
BIG_SHIFT_GAP_KEGL = 1.0
# Сильная разница: кегль от 1.8 раза, самые высокие буквы от 1.6, просвет шире двух меньших кеглей (низы
# при этом разошлись). Цифры корпуса («1968», «10») в 1.8 раза выше строчных, но не выше букв с выносными
# (1969/09 IMG_0105_2R): без условия на высокие буквы ось строки уходила на соседний ряд.
STRONG_RATIO = 1.8
STRONG_TALL_RATIO = 1.6
STRONG_GAP_KEGL = 2.0

# --- Линейка между группами осей (:func:`barrier_between`) ---
# Глиф выше стольких кеглей — не буква, а краска линейки, вошедшая в строку: линейка оглавления (1973/01
# IMG_0004_2R) стала «глифом» левой строки, ось кончалась на самой линейке, и отрезок между концами осей её
# не пересекал. Отрезок проверки идёт от последнего глифа нормальной высоты до первого.
RULE_GLYPH_KEGL = 2.0
# Разделяет только линейка длиннее стольких кеглей большей стороны: линейка оглавления — 956 px при кегле
# 27, а ствол «М» в «МТС», сросшийся с буквой строки ниже и принятый детектором таблиц за линейку, — 85 px
# при кегле 21 (1972/12 IMG_0130_1L).
BARRIER_MIN_SPAN_KEGL = 6.0


def _kept(heights: np.ndarray) -> np.ndarray:
    """Маска глифов без мелочи: не ниже ``MARK_SHARE`` медианы."""
    return heights >= MARK_SHARE * np.median(heights) if heights.size else np.zeros(0, dtype=bool)


def glyphs_kegl(glyphs: np.ndarray | None) -> float:
    """Кегль по боксам глифов: медиана высот без мелочи.

    Args:
        glyphs: Боксы ``(n, 4)`` — ``x0, y0, x1, y1`` (пиксели рабочей копии) или ``None``.

    Returns:
        Кегль; 0 — глифов нет.
    """
    if glyphs is None or len(glyphs) == 0:
        return 0.0
    heights = glyphs[:, 3] - glyphs[:, 1]
    heights = heights[_kept(heights)]
    return float(np.median(heights)) if heights.size else 0.0


def glyphs_bottom(glyphs: np.ndarray | None) -> float | None:
    """Низ набора: медиана низов глифов без мелочи.

    Args:
        glyphs: Боксы ``(n, 4)`` или ``None``.

    Returns:
        Ордината низа; ``None`` — глифов нет.
    """
    if glyphs is None or len(glyphs) == 0:
        return None
    heights = glyphs[:, 3] - glyphs[:, 1]
    return float(np.median(glyphs[_kept(heights), 3]))


def axes_glyphs(axes) -> np.ndarray:
    """Боксы глифов всех осей подряд ``(n, 4)``; пустой массив — глифов нет.

    Args:
        axes: Оси строк (``lines.LineAxis``), у которых есть ``glyphs``.

    Returns:
        Боксы глифов.
    """
    boxes = [np.asarray(axis.glyphs, dtype=float) for axis in axes if axis.glyphs is not None and len(axis.glyphs)]
    return np.concatenate(boxes) if boxes else np.zeros((0, 4))


def axis_kegl(axis) -> float:
    """Кегль оси по боксам глифов — без потолка ``blocks.GLYPH_MAX_MM``: шапка журнала тоже меряется.

    Args:
        axis: Ось строки.

    Returns:
        Кегль; 0 — глифов нет.
    """
    return glyphs_kegl(axes_glyphs([axis]))


def axes_kegl(axes) -> float:
    """Кегль группы осей (ряда): медиана кеглей осей; 0 — ни одна ось не меряется.

    Args:
        axes: Оси группы.

    Returns:
        Кегль.
    """
    values = [value for value in (axis_kegl(axis) for axis in axes) if value > 0]
    return float(np.median(values)) if values else 0.0


def same_baseline(first: np.ndarray, second: np.ndarray, low: float) -> bool:
    """Стоят ли две группы глифов на одной базовой линии: низы ближе ``BASELINE_SHIFT_KEGL`` кегля ``low``.

    Args:
        first: Глифы первой группы ``(n, 4)``.
        second: Второй.
        low: Меньший кегль групп.

    Returns:
        ``True`` — одна линия, или сравнить нечем (у группы нет глифов).
    """
    bottoms = (glyphs_bottom(first), glyphs_bottom(second))
    return None in bottoms or abs(bottoms[0] - bottoms[1]) <= BASELINE_SHIFT_KEGL * low


def side_glyphs(glyphs: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Глифы стороны просвета без мелочи; мелочь меряется по медиане САМОЙ стороны, а не всей строки.

    По медиане всей строки мелкий набор рядом с крупным («Год издания» 18 px у «СНАБЖЕНИЕ» 54 px, 1966/03
    IMG_0105_2R) целиком уходил в мелочь, и у просвета не оставалось правой стороны.

    Args:
        glyphs: Боксы глифов строки ``(n, 4)``.
        indices: Индексы глифов стороны.

    Returns:
        Боксы глифов стороны без мелочи.
    """
    side = glyphs[indices]
    return side[_kept(side[:, 3] - side[:, 1])]


def different_sets(left: np.ndarray, right: np.ndarray, gap: float) -> bool:
    """Разный ли набор по обе стороны просвета строки — только у крупного набора и только не на одной линии.

    Разный — если крупная сторона от ``LARGE_KEGL_PX`` и либо просвет шире ``FAR_GAP_KEGL`` кеглей большей
    стороны при разошедшихся низах, либо низы разошлись больше чем на полкегля при заметной разнице кегля и
    высоких букв, либо кегль и высокие буквы различаются сильно (``STRONG_*``).

    Args:
        left: Глифы слева от просвета (ближайшие, без мелочи), ``(n, 4)``.
        right: Глифы справа.
        gap: Ширина просвета, пиксели рабочей копии.

    Returns:
        ``True`` — резать строку в этом просвете.
    """
    if len(left) == 0 or len(right) == 0:
        return False
    low, high = sorted((glyphs_kegl(left), glyphs_kegl(right)))
    if low < MIN_KEGL_PX or high < LARGE_KEGL_PX:
        return False
    shift = abs(glyphs_bottom(left) - glyphs_bottom(right))
    if gap > FAR_GAP_KEGL * high and shift > FAR_SHIFT_KEGL * low:
        return True
    if shift <= BASELINE_SHIFT_KEGL * low:
        return False
    tall = sorted(((left[:, 3] - left[:, 1]).max(), (right[:, 3] - right[:, 1]).max()))
    if (
        shift > BIG_SHIFT_KEGL * low
        and high >= BIG_SHIFT_RATIO * low
        and tall[1] >= BIG_SHIFT_TALL_RATIO * tall[0]
        and gap > BIG_SHIFT_GAP_KEGL * low
    ):
        return True
    return high >= STRONG_RATIO * low and tall[1] >= STRONG_TALL_RATIO * tall[0] and gap > STRONG_GAP_KEGL * low


def set_cuts(glyphs: np.ndarray) -> list[int]:
    """Где резать строку по разному набору: индексы глифов (по x), с которых начинаются новые части.

    Args:
        glyphs: Боксы глифов строки ``(n, 4)``, отсортированные по ``x0``.

    Returns:
        Индексы начала частей (без нуля), по возрастанию.
    """
    cuts = []
    reach = np.maximum.accumulate(glyphs[:, 2])
    for index in range(1, len(glyphs)):
        gap = glyphs[index, 0] - reach[index - 1]
        if gap <= 0:
            continue
        left = side_glyphs(glyphs, np.arange(max(0, index - SIDE_GLYPHS), index))
        right = side_glyphs(glyphs, np.arange(index, min(len(glyphs), index + SIDE_GLYPHS)))
        if different_sets(left, right, gap):
            cuts.append(index)
    return cuts


def normal_edge(axes, right_side: bool) -> float | None:
    """Край глифов нормальной высоты (не выше ``RULE_GLYPH_KEGL`` кеглей) у группы осей.

    Args:
        axes: Оси группы.
        right_side: ``True`` — группа стоит справа от просвета, нужен её левый край; ``False`` — правый.

    Returns:
        Абсцисса края; ``None`` — глифов нет.
    """
    glyphs = axes_glyphs(axes)
    if len(glyphs) == 0:
        return None
    kegl = glyphs_kegl(glyphs)
    normal = glyphs[(glyphs[:, 3] - glyphs[:, 1]) <= RULE_GLYPH_KEGL * kegl] if kegl else glyphs
    if len(normal) == 0:
        return None
    return float(normal[:, 0].min()) if right_side else float(normal[:, 2].max())


def barrier_between(left_axes, right_axes, y0: float, y1: float, barriers) -> bool:
    """Проходит ли линейка-барьер длиннее ``BARRIER_MIN_SPAN_KEGL`` кеглей между глифами нормальной высоты двух групп.

    Args:
        left_axes: Оси левой группы.
        right_axes: Оси правой группы.
        y0: Ордината отрезка у левой группы.
        y1: У правой.
        barriers: Линейки-барьеры (``barriers.BarrierLines``) или ``None``.

    Returns:
        ``True`` — между группами линейка.
    """
    if barriers is None or barriers.empty:
        return False
    x0, x1 = normal_edge(left_axes, right_side=False), normal_edge(right_axes, right_side=True)
    if x0 is None or x1 is None or x1 <= x0:
        return False
    kegl = max(glyphs_kegl(axes_glyphs(left_axes)), glyphs_kegl(axes_glyphs(right_axes)))
    lines = [line for line in barriers.lines if np.ptp(line[:, 1]) + np.ptp(line[:, 0]) > BARRIER_MIN_SPAN_KEGL * kegl]
    return bool(lines) and type(barriers).of(lines).crosses((x0, y0), (x1, y1))


__all__ = [
    "axes_glyphs",
    "axes_kegl",
    "axis_kegl",
    "barrier_between",
    "different_sets",
    "glyphs_bottom",
    "glyphs_kegl",
    "normal_edge",
    "same_baseline",
    "set_cuts",
    "side_glyphs",
]
