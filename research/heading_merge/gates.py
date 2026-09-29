"""Запреты сращивания строк разного набора подменой функций детектора на время разбора (боевой код не меняется)."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum

import numpy as np

from ocr_utils.page_layout import mm_to_px
from ocr_utils.page_layout.text_blocks import WORK_DPI, blocks, regroup, zones
from ocr_utils.page_layout.text_blocks import page as page_module
from ocr_utils.page_layout.text_blocks.baseline_axis import line_x_height
from ocr_utils.page_layout.text_blocks.engines import ink as ink_module
from ocr_utils.page_layout.text_blocks.pieces import END_KEGL_LETTERS, END_KEGL_MIN_XH
from ocr_utils.page_layout.text_blocks.segment import Segment

# --- Сцепка кусков строки (``zones._verdict_of``) ---
# Разница кегля у стыка от стольких раз запрещает сцепку через просвет шире ``LINK_STRONG_GAP_KEGL``
# меньших кеглей. Ниже 1.8 порог не опускать: на C почти ничего не прибавляет, на N растут изменения
# (замер сессии 2026-09-29: пороги 1.5–1.6 — 6 и 12 изменённых полос N против 3).
LINK_STRONG_RATIO = 1.8
LINK_STRONG_GAP_KEGL = 2.0
# Самые высокие крайние буквы тоже различаются не меньше чем во столько раз: цифры корпуса («1968»,
# «10») в 1.8 раза выше строчных, но не выше букв с выносными (1969/09 IMG_0105_2R — без этого условия
# ось строки уходила на соседний ряд).
LINK_TALL_RATIO = 1.6
# Низы кусков (рядов, частей строки) расходятся больше чем на столько меньших кеглей: логотип «50 лет»
# (1967/10) стоит на одной базовой линии (2 px при кегле 16), подпись рядом с заголовком — нет (6–20 px).
BASELINE_SHIFT_KEGL = 0.25
# Ряд: защита «одна базовая линия» — только для просвета уже стольких меньших кеглей. У «50 лет» просвет
# 1.4 кегля, а у месяца и «Год издания», случайно стоящих на базовой линии шапки, — от 2 (1970/01
# IMG_0004_2R: «5-й» у «СНАБЖЕНИЕ» через 74 px при кегле 21).
ROW_SAME_BASELINE_MAX_GAP_KEGL = 1.8

# --- Ряды блока (``regroup._divided``): те же пороги, что у ``blocks._split_by_glyph`` ---
ROW_RATIO, ROW_GAP_MM = blocks.ROW_GLYPH_RATIO, blocks.ROW_GLYPH_GAP_MM
ROW_STRONG_RATIO, ROW_STRONG_GAP_MM = blocks.ROW_GLYPH_STRONG_RATIO, blocks.ROW_GLYPH_STRONG_GAP_MM
# Глифы ниже этой доли медианы (точки, запятые, тире) в кегль не входят.
MARK_SHARE = 0.45

# --- Резка готовой строки (``Variant.SEGMENT``) ---
# Кегль стороны просвета — по стольким ближайшим к нему глифам без мелочи.
SPLIT_SIDE_GLYPHS = 6
# Дальний просвет: шире стольких кеглей БОЛЬШЕЙ стороны при разных базовых линиях — разные строки
# (месяц и «Год издания» у шапки журнала: 2 кегля; межсловный пробел заголовка — до 1 кегля).
SPLIT_FAR_GAP_KEGL = 1.5
# ...и базовые линии там расходятся хотя бы на столько меньших кеглей: месяц у «ТЕХНИЧЕСКОЕ» стоит на
# 8.5 px выше при кегле 34 (0.25 — впритык), а шум низов у слов одной строки — 1–2 px.
SPLIT_FAR_SHIFT_KEGL = 0.15
# ...и все правила резки — только у крупного набора (крупная сторона от стольких px): у корпуса кегль по медиане глифов — это строчная (≈ 10 px), пробел при
# выключке по формату шире 1.5 строчной, а низы через пробел расходятся на 2–4 px от наклона строки —
# правило резало строки корпуса (N: +373 оси на 147 полосах, 1967/12 IMG_0143_2R). Порог — как у отбора
# кандидатов (``measure.CANDIDATE_MIN_HEIGHT``): заголовки пака от 18 px, шапка — от 34.
SPLIT_FAR_MIN_KEGL_PX = 18.0
# Базовые линии разошлись больше чем на полкегля — строки заведомо разные, и хватает кегля от 1.5
# раза при просвете от кегля: «Г» (17 px) у «СНАБЖЕНИЕ» (54 px) через 30 px, 1967/02 IMG_0054_2R;
# «август» (22) у «ТЕХНИЧЕСКОЕ» (43) через 56 px с низами на 23 px выше, 1968/08 IMG_0054_2R.
SPLIT_BIG_SHIFT_KEGL = 0.5
SPLIT_BIG_SHIFT_RATIO = 1.5
SPLIT_BIG_SHIFT_GAP_KEGL = 1.0
# ...и самые высокие буквы сторон различаются хотя бы во столько раз: у корпуса цифры (17 px) против
# строчных (кегль-медиана 7 px) дают и кегль 2.4×, и низы на 5 px, но высокие буквы — 17 против 14
# (1972/07 IMG_0023_1L, «0,4% и т. д.»); у шапки — 3.2 и 1.52.
SPLIT_BIG_SHIFT_TALL_RATIO = 1.4

# --- Барьер с заходом (``Variant.BARRIER``) ---
# Краска вертикальной линейки оглавления входит в строку, ось кончается на самой линейке (1973/01
# IMG_0004_2R: x1 = 355 при линейке 353–355), и отрезок между концами осей её не пересекал.
# Глиф выше стольких кеглей ряда — не буква, а краска линейки, вошедшая в строку: отрезок проверки
# линейки идёт от последнего глифа нормальной высоты одной стороны до первого — другой. Так линейка
# оглавления (глиф 956 px при кегле 16) попадает в отрезок, а штрих буквы «МТС», принятый детектором
# таблиц за линейку (1972/12 IMG_0130_1L), — нет: отрезок кончается на краю буквы.
RULE_GLYPH_KEGL = 2.0
# Разделяет только линейка длиннее стольких кеглей большей стороны: линейка оглавления — 956 px при
# кегле 27, а ствол «М» в «МТС», сросшийся с буквой строки ниже и принятый за линейку, — 85 px при
# кегле 21 (1972/12 IMG_0130_1L).
BARRIER_MIN_SPAN_KEGL = 6.0


# --- Продление второй оси до краски (``baseline_axis.extend_to_ink``) ---
# Полоса строки, в которой ищутся чужие глифы, — столько строчных вверх и вниз от конца оси (как у
# самого продления).
EXTEND_BAND_XH = 0.6


# Меньшая сторона просвета — не мельче стольких пикселей рабочей копии: пыль и точки не кегль (у
# заголовка 1966/01 IMG_0039_1L сторона из пылинок в 1 px давала «разные базовые линии»).
SPLIT_MIN_KEGL_PX = 6.0


class Variant(str, Enum):
    """Набор запретов стенда: каждый следующий включает предыдущие, кроме ``BARRIER`` (без резки строк)."""

    BASE = "base"  # детектор как есть
    V5 = "v5"  # запрет сцепки, слияния рядов и резка ряда по кеглю с проверкой базовой линии
    SEGMENT = "segment"  # V5 + резка готовой строки по разному набору
    BARRIER = "barrier"  # V5 + линейки-барьеры с заходом внутрь осей и рядов
    FULL = "full"  # всё вместе


# --- Меры набора по боксам глифов ---


def _kept(heights: np.ndarray) -> np.ndarray:
    """Маска глифов без мелочи: не ниже ``MARK_SHARE`` медианы."""
    return heights >= MARK_SHARE * np.median(heights) if heights.size else np.zeros(0, dtype=bool)


def glyphs_kegl(glyphs: np.ndarray) -> float:
    """Кегль по боксам глифов ``(n, 4)``: медиана высот без мелочи; 0 — глифов нет."""
    if glyphs is None or len(glyphs) == 0:
        return 0.0
    heights = glyphs[:, 3] - glyphs[:, 1]
    heights = heights[_kept(heights)]
    return float(np.median(heights)) if heights.size else 0.0


def glyphs_bottom(glyphs: np.ndarray) -> float | None:
    """Низ набора по боксам глифов ``(n, 4)``: медиана низов без мелочи; ``None`` — глифов нет."""
    if glyphs is None or len(glyphs) == 0:
        return None
    heights = glyphs[:, 3] - glyphs[:, 1]
    return float(np.median(glyphs[_kept(heights), 3]))


def axes_glyphs(axes) -> np.ndarray:
    """Боксы глифов всех осей подряд ``(n, 4)``; пустой массив — глифов нет."""
    boxes = [np.asarray(axis.glyphs, dtype=float) for axis in axes if axis.glyphs is not None and len(axis.glyphs)]
    return np.concatenate(boxes) if boxes else np.zeros((0, 4))


def axis_kegl(axis) -> float:
    """Кегль оси по боксам глифов (без потолка ``blocks.GLYPH_MAX_MM`` — шапка журнала тоже меряется)."""
    return glyphs_kegl(axes_glyphs([axis]))


def row_kegl(row) -> float:
    """Кегль ряда: медиана кеглей его осей; 0 — ни одна ось не меряется."""
    values = [v for v in (axis_kegl(axis) for axis in row.axes) if v > 0]
    return float(np.median(values)) if values else 0.0


def same_baseline(first: np.ndarray, second: np.ndarray, low: float) -> bool:
    """Стоят ли две группы глифов на одной базовой линии: низы ближе ``BASELINE_SHIFT_KEGL`` кегля ``low``."""
    bottoms = (glyphs_bottom(first), glyphs_bottom(second))
    return None in bottoms or abs(bottoms[0] - bottoms[1]) <= BASELINE_SHIFT_KEGL * low


# --- Сцепка кусков ---


def _end_letters(piece, at_start: bool) -> np.ndarray:
    """Высоты крайних букв куска (без низких меток и тире); пусто — букв нет."""
    letter = ~piece.marks & (piece.sizes[:, 1] >= END_KEGL_MIN_XH * piece.x_h)
    heights = piece.sizes[letter, 1]
    return heights[:END_KEGL_LETTERS] if at_start else heights[-END_KEGL_LETTERS:]


def end_kegl_any(piece, at_start: bool) -> float | None:
    """Кегль у конца куска: ``pieces.end_kegl``, а если букв мало — медиана имеющихся крайних букв.

    Args:
        piece: Кусок строки (``pieces.Piece``).
        at_start: ``True`` — левый конец, ``False`` — правый.

    Returns:
        Высота в пикселях рабочей копии; ``None`` — у куска нет ни одной буквы.
    """
    kegl = zones.end_kegl(piece, at_start=at_start)
    if kegl is not None:
        return kegl
    heights = _end_letters(piece, at_start)
    return float(np.median(heights)) if heights.size else None


def end_tall(piece, at_start: bool) -> float | None:
    """Высота самой высокой из крайних букв куска; ``None`` — букв нет."""
    heights = _end_letters(piece, at_start)
    return float(heights.max()) if heights.size else None


# --- Резка готовой строки ---


def _side(glyphs: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Глифы стороны просвета без мелочи; мелочь меряется по медиане САМОЙ стороны, а не всей строки.

    По медиане всей строки мелкий набор рядом с крупным («Год издания» 18 px у «СНАБЖЕНИЕ» 54 px,
    1966/03 IMG_0105_2R) целиком уходил в мелочь, и у просвета не оставалось правой стороны.
    """
    side = glyphs[indices]
    return side[_kept(side[:, 3] - side[:, 1])]


def different_sets(left: np.ndarray, right: np.ndarray, gap: float) -> bool:
    """Разный ли набор по обе стороны просвета строки.

    Разный — если стороны не на одной базовой линии и при этом либо просвет шире ``SPLIT_FAR_GAP_KEGL``
    кеглей большей стороны, либо низы разошлись больше чем на полкегля при разнице кегля от 1.5 раза, либо
    кегль и самые высокие буквы различаются сильно (как у сцепки кусков).

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
    if low < SPLIT_MIN_KEGL_PX:
        return False
    shift = abs(glyphs_bottom(left) - glyphs_bottom(right))
    # Строки корпуса не режутся вовсе: кегль по медиане глифов там — строчная, старинные цифры уходят
    # под строку, и «разный набор» находился у «0,4% и т. д.» (1972/07 IMG_0023_1L).
    if high < SPLIT_FAR_MIN_KEGL_PX:
        return False
    if gap > SPLIT_FAR_GAP_KEGL * high and shift > SPLIT_FAR_SHIFT_KEGL * low:
        return True
    if shift <= BASELINE_SHIFT_KEGL * low:
        return False
    tall = sorted(((left[:, 3] - left[:, 1]).max(), (right[:, 3] - right[:, 1]).max()))
    if (
        shift > SPLIT_BIG_SHIFT_KEGL * low
        and high >= SPLIT_BIG_SHIFT_RATIO * low
        and tall[1] >= SPLIT_BIG_SHIFT_TALL_RATIO * tall[0]
        and gap > SPLIT_BIG_SHIFT_GAP_KEGL * low
    ):
        return True
    return high >= LINK_STRONG_RATIO * low and tall[1] >= LINK_TALL_RATIO * tall[0] and gap > LINK_STRONG_GAP_KEGL * low


def cuts_of(glyphs: np.ndarray) -> list[int]:
    """Где резать строку по разному набору: индексы глифов (по x), с которых начинаются новые части.

    Резка по линейкам-барьерам здесь пробовалась и отвергнута: штрих буквы заголовка, принятый детектором
    таблиц за линейку, резал заголовок (1966/01 IMG_0039_1L, 1969/07 IMG_0004_2R), а оглавление и так
    чинится на уровне рядов (:func:`barrier_between`).

    Args:
        glyphs: Боксы глифов строки ``(n, 4)``, отсортированные по ``x0``.

    Returns:
        Индексы начала частей (без нуля), по возрастанию.
    """
    cuts = []
    reach = np.maximum.accumulate(glyphs[:, 2])
    for i in range(1, len(glyphs)):
        gap = glyphs[i, 0] - reach[i - 1]
        if gap <= 0:
            continue
        left = _side(glyphs, np.arange(max(0, i - SPLIT_SIDE_GLYPHS), i))
        right = _side(glyphs, np.arange(i, min(len(glyphs), i + SPLIT_SIDE_GLYPHS)))
        if different_sets(left, right, gap):
            cuts.append(i)
    return cuts


def split_segment(segment: Segment) -> list[Segment]:
    """Разрезать строку на части разного набора; без разрезов — ``[segment]``.

    Части — те же ``Segment`` с глифами, точками центр-линии и метками своего отрезка по x; высота
    пересчитывается пропорционально кеглю части.

    Args:
        segment: Строка после сегментации.

    Returns:
        Части слева направо.
    """
    glyphs = segment.glyphs
    if glyphs is None or len(glyphs) < 2 or segment.xs.size < 2:
        return [segment]
    glyphs = np.asarray(glyphs, dtype=float)
    glyphs = glyphs[np.argsort(glyphs[:, 0])]
    cuts = cuts_of(glyphs)
    if not cuts:
        return [segment]
    whole = glyphs_kegl(glyphs) or 1.0
    parts = []
    for start, end in zip([0, *cuts], [*cuts, len(glyphs)]):
        part = glyphs[start:end]
        x0, x1 = part[:, 0].min(), part[:, 2].max()
        inside = (segment.xs >= x0) & (segment.xs <= x1)
        if int(inside.sum()) < 2:
            continue
        parts.append(
            replace(
                segment,
                x0=int(np.floor(x0)),
                y0=int(np.floor(part[:, 1].min())),
                x1=int(np.ceil(x1)),
                y1=int(np.ceil(part[:, 3].max())),
                height=float(segment.height * (glyphs_kegl(part) or whole) / whole),
                xs=segment.xs[inside],
                ys=segment.ys[inside],
                weights=segment.weights[inside],
                mark_spans=tuple(span for span in segment.mark_spans if x0 <= (span[0] + span[1]) / 2.0 <= x1),
                glyphs=part,
            )
        )
    return parts if parts else [segment]


def clip_extension(axis, others: np.ndarray, end_x: float) -> float:
    """Докуда можно продлить конец второй оси, не заходя на глифы других осей.

    ``baseline_axis.extend_to_ink`` ведёт ось по краске полосы строки с пустотами до полстрочной и
    на три строчных: у шапки журнала (строчная ≈ 43 px) это 130 px, и ось «ТЕХНИЧЕСКОЕ» уходила в
    «июль», а «СНАБЖЕНИЕ» — в «Год издания» (1969/07 IMG_0004_2R). Дальше такие оси налезали по x, и
    ``blocks._split_by_glyph`` их не резал. Дефис и точка своей строки — не глифы чужой оси, их продление
    по-прежнему берёт.

    Args:
        axis: Ось до продления (``LineAxis`` со своими ``glyphs`` и ``body_points``).
        others: Глифы всех ДРУГИХ осей страницы ``(n, 4)``.
        end_x: Абсцисса конца после продления.

    Returns:
        Абсцисса конца продления; прежний конец оси, если продление задевает чужой глиф в полосе строки.
    """
    points = axis.body_points
    x_h = line_x_height(axis.glyphs)
    if len(others) == 0 or not x_h:
        return end_x
    right = end_x > points[-1, 0]
    x, y = points[-1] if right else points[0]
    band = EXTEND_BAND_XH * x_h
    # Чужие глифы между концом оси и концом продления, задевающие полосу строки.
    lo, hi = (x, end_x) if right else (end_x, x)
    # Чужой глиф — тот, чья СЕРЕДИНА в полосе строки: выносные соседних строк полосу задевают, но
    # продление дефиса своей строки из-за них не отменяется.
    middle = (others[:, 1] + others[:, 3]) / 2.0
    hit = (others[:, 2] > lo) & (others[:, 0] < hi) & (np.abs(middle - y) < band)
    if not hit.any():
        return end_x
    # Продление, упёршееся в чужой глиф, отменяется целиком: обрезанное вплотную к нему, оно оставляло
    # между осями просвет в пиксель, и ряд по кеглю не резался (1968/04 IMG_0004_2R: 682 → 683).
    return float(x)


def extend_to_ink_clipped(axes: list, ink: np.ndarray, k: float, original) -> list:
    """``baseline_axis.extend_to_ink``, у которого продление не заходит на глифы других осей.

    Args:
        axes: Оси страницы со вторыми осями.
        ink: Краска текста рендера.
        k: Во сколько раз рендер крупнее рабочей копии.
        original: Исходная ``extend_to_ink``.

    Returns:
        Оси с продлёнными (и обрезанными по чужим глифам) ``body_points``.
    """
    extended = original(axes, ink, k)
    boxes = [axes_glyphs([axis]) for axis in axes]
    out = []
    for index, (before, after) in enumerate(zip(axes, extended)):
        if before.body_points is None or after.body_points is before.body_points:
            out.append(after)
            continue
        own = boxes[index]
        kegl = glyphs_kegl(own)
        # Чужие — глифы осей НА ДРУГОЙ базовой линии: ось той же строки (половинка заголовка, разобранного
        # двумя налезающими осями, 1966/01 IMG_0039_1L) — законное продолжение, по ней продление идёт.
        foreign = [
            b
            for j, b in enumerate(boxes)
            if j != index and len(b) and not same_baseline(own, b, min(kegl, glyphs_kegl(b)) or kegl)
        ]
        others = np.concatenate(foreign) if foreign else np.zeros((0, 4))
        points = after.body_points
        start, end = before.body_points[0, 0], before.body_points[-1, 0]
        # Продлённые узлы — те, что вышли за концы прежней оси; узел, чьё продление задело чужой глиф,
        # выбрасывается (``clip_extension`` вернула прежний конец).
        if points[0, 0] < start and clip_extension(before, others, points[0, 0]) != points[0, 0]:
            points = points[1:]
        if points[-1, 0] > end and clip_extension(before, others, points[-1, 0]) != points[-1, 0]:
            points = points[:-1]
        out.append(replace(after, body_points=points))
    return out


def normal_edge(axes, right_side: bool) -> float | None:
    """Край глифов нормальной высоты у группы осей: правый край (``right_side=False``) или левый.

    Args:
        axes: Оси ряда (или группы ряда).
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
    """Проходит ли линейка-барьер между глифами нормальной высоты двух групп осей.

    Args:
        left_axes: Оси левой группы.
        right_axes: Оси правой группы.
        y0: Ордината отрезка у левой группы.
        y1: У правой.
        barriers: Линейки-барьеры (``barriers.BarrierLines``).

    Returns:
        ``True`` — между группами линейка.
    """
    x0, x1 = normal_edge(left_axes, right_side=False), normal_edge(right_axes, right_side=True)
    if x0 is None or x1 is None or x1 <= x0:
        return False
    kegl = max(glyphs_kegl(axes_glyphs(left_axes)), glyphs_kegl(axes_glyphs(right_axes)))
    lines = [line for line in barriers.lines if np.ptp(line[:, 1]) + np.ptp(line[:, 0]) > BARRIER_MIN_SPAN_KEGL * kegl]
    return bool(lines) and type(barriers).of(lines).crosses((x0, y0), (x1, y1))


class HeadingPatch:
    """Подмена функций детектора текстовых блоков на время разбора (контекстный менеджер).

    Подмены — методы объекта: исходные функции он запоминает при установке и зовёт из методов. В
    воркере пула достаточно ``install()`` без восстановления.

    Attributes:
        variant: Набор запретов.
    """

    def __init__(self, variant: Variant) -> None:
        self.variant = variant
        self.barrier_reach = variant in (Variant.BARRIER, Variant.FULL)
        self.segment_split = variant in (Variant.SEGMENT, Variant.FULL)
        self._saved: list[tuple[object, str, object]] = []
        # Исходные функции детектора — берутся при установке, чтобы подмены не наслаивались.
        self._verdict = zones._verdict_of
        self._divided = regroup._divided
        self._axis_glyph = blocks.axis_glyph_height
        self._split_by_glyph = blocks._split_by_glyph
        self._split_by_barrier = blocks._split_by_barrier
        self._segments_of = ink_module.segments_of
        self._extend_to_ink = page_module.extend_to_ink

    def verdict_gate(self, left, right, scale, *args, **kwargs):
        """Вердикт сцепки плюс запрет: сильная разница кегля и высоких букв через просвет в два меньших кегля не на одной базовой линии.

        Args:
            left, right, scale, args, kwargs: Как у ``zones._verdict_of``.

        Returns:
            ``LinkVerdict`` исходной функции или ``MIXED_GAP``.
        """
        result = self._verdict(left, right, scale, *args, **kwargs)
        if result is not zones.LinkVerdict.ACCEPTED:
            return result
        ends = (end_kegl_any(left, at_start=False), end_kegl_any(right, at_start=True))
        tall = (end_tall(left, at_start=False), end_tall(right, at_start=True))
        if None in ends or None in tall:
            return result
        low, high = sorted(ends)
        short, long_ = sorted(tall)
        if (
            high >= LINK_STRONG_RATIO * low
            and long_ >= LINK_TALL_RATIO * short
            and right.x0 - left.x1 > LINK_STRONG_GAP_KEGL * low
            and abs(left.y1 - right.y1) > BASELINE_SHIFT_KEGL * low
        ):
            return zones.LinkVerdict.MIXED_GAP
        return result

    def divided_gate(self, a, b, barriers, gutters) -> bool:
        """Разделены ли два ряда: прежний запрет, линейка с заходом (``BARRIER``), разный кегль через пустоту.

        Args:
            a, b, barriers, gutters: Как у ``regroup._divided``.

        Returns:
            ``True`` — ряды не сливать.
        """
        if self._divided(a, b, barriers, gutters):
            return True
        left, right = (a, b) if a.x0 <= b.x0 else (b, a)
        if self.barrier_reach and barriers is not None and not barriers.empty:
            # Края — по ГЛИФАМ осей рядов, а не по краске: край ряда по краске уходит за линейку в соседнюю
            # графу (1973/01 IMG_0004_2R: «1973» — оси до 358, край по краске 392 при линейке 354 и
            # оглавлении с 376), ряды «налезают», и отрезок между краями по краске линейку не видит.
            y = (a.y + b.y) / 2.0
            if barrier_between(left.axes, right.axes, y, y, barriers):
                return True
        # Просвет — по ОСЯМ рядов: края по краске у подписи рядом с заголовком налезают друг на друга
        # (1971/02 IMG_0071_2R: оси 70–270 и 293–760, а ряды по краске сомкнуты), и запрет молчал.
        # Половинки одной строки налезают и осями — у них просвет отрицательный, запрет не сработает.
        gap = min(axis.x0 for axis in right.axes) - max(axis.x1 for axis in left.axes)
        low, high = sorted((row_kegl(a), row_kegl(b)))
        if low <= 0:
            return False
        if gap <= ROW_SAME_BASELINE_MAX_GAP_KEGL * low and same_baseline(axes_glyphs(a.axes), axes_glyphs(b.axes), low):
            return False
        if high >= ROW_RATIO * low and gap > mm_to_px(ROW_GAP_MM, WORK_DPI):
            return True
        return high >= ROW_STRONG_RATIO * low and gap > mm_to_px(ROW_STRONG_GAP_MM, WORK_DPI)

    def axis_glyph_gate(self, axis, ink, k, dpi) -> float:
        """Кегль оси для второго прохода резки ряда — по боксам глифов у ВСЕХ осей.

        Прежняя мера (``blocks.axis_glyph_height``) молчит у букв выше 7 мм (шапка журнала). Подставлять
        запасную меру только там, где прежняя молчит, нельзя: две меры в одном ряду дают ложную разницу
        кегля — «АСУ МТС» (1972/12 IMG_0130_1L): 21 по одной мере против 15 по другой, а по одной — 1.4.

        Args:
            axis, ink, k, dpi: Как у ``blocks.axis_glyph_height`` (краска не нужна).

        Returns:
            Кегль оси по боксам её глифов; 0 — глифов нет.
        """
        return axis_kegl(axis)

    def split_by_glyph_gate(self, group, ink, k, dpi):
        """Резка ряда по кеглю: прежние разрезы как есть, новые (второй проход, мера по боксам глифов) — только при разных низах или широком просвете.

        Запасная мера нужна шапке журнала, но она же резала логотип «50 лет» (1967/10 IMG_0032_1L),
        где цифры и слово стоят на одной базовой линии.

        Args:
            group, ink, k, dpi: Как у ``blocks._split_by_glyph``.

        Returns:
            Части ряда слева направо.
        """
        blocks.axis_glyph_height = self._axis_glyph
        try:
            old = self._split_by_glyph(group, ink, k, dpi)
        finally:
            blocks.axis_glyph_height = self.axis_glyph_gate
        new = self._split_by_glyph(group, ink, k, dpi)
        if len(new) == len(old):
            return new
        old_starts = {id(part[0]) for part in old}
        out = [list(new[0])]
        for part in new[1:]:
            if id(part[0]) not in old_starts:
                kegls = [v for v in (axis_kegl(axis) for axis in (*out[-1], *part)) if v > 0]
                low = min(kegls) if kegls else 0.0
                gap = min(axis.x0 for axis in part) - max(axis.x1 for axis in out[-1])
                if gap <= ROW_SAME_BASELINE_MAX_GAP_KEGL * low and same_baseline(
                    axes_glyphs(out[-1]), axes_glyphs(part), low
                ):
                    out[-1].extend(part)
                    continue
            out.append(list(part))
        return out if len(out) > 1 else [group]

    def split_by_barrier_gate(self, group, axis, barriers) -> bool:
        """Прежняя проверка линейки между осями ряда плюс отрезок, зашедший внутрь обеих осей.

        Args:
            group, axis, barriers: Как у ``blocks._split_by_barrier``.

        Returns:
            ``True`` — ось в этот ряд не идёт.
        """
        if self._split_by_barrier(group, axis, barriers):
            return True
        if barriers.empty:
            return False
        nearest = min(group, key=lambda item: blocks._x_distance(item, axis))
        left, right = (nearest, axis) if nearest.x0 <= axis.x0 else (axis, nearest)
        return barrier_between([left], [right], left.y_at(left.x1), right.y_at(right.x0), barriers)

    def segments_gate(self, gray300, separators, dpi=WORK_DPI, leaders=None, linking=None, barriers=None, **kwargs):
        """Сегментация (``segment.segments_of``) плюс резка готовых строк по разному набору.

        Args:
            gray300, separators, dpi, leaders, linking, barriers, kwargs: Как у ``segment.segments_of``.

        Returns:
            Строки сверху вниз и сплошные черты.
        """
        args = (gray300, separators, dpi, leaders) + (() if linking is None else (linking,))
        segments, rules = self._segments_of(*args, barriers, **kwargs)
        out = [part for segment in segments for part in split_segment(segment)]
        return sorted(out, key=lambda item: item.cy), rules

    def extend_gate(self, axes: list, ink: np.ndarray, k: float) -> list:
        """Продление вторых осей до краски без захода на глифы других осей (см. :func:`clip_extension`)."""
        return extend_to_ink_clipped(axes, ink, k, self._extend_to_ink)

    def _set(self, module, name: str, value) -> None:
        """Подменить ``module.name`` и запомнить исходное."""
        self._saved.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    def install(self) -> "HeadingPatch":
        """Установить подмены варианта."""
        if self.variant is Variant.BASE:
            return self
        self._set(zones, "_verdict_of", self.verdict_gate)
        self._set(regroup, "_divided", self.divided_gate)
        self._set(blocks, "axis_glyph_height", self.axis_glyph_gate)
        self._set(blocks, "_split_by_glyph", self.split_by_glyph_gate)
        if self.barrier_reach:
            self._set(blocks, "_split_by_barrier", self.split_by_barrier_gate)
        if self.segment_split:
            self._set(ink_module, "segments_of", self.segments_gate)
            self._set(page_module, "extend_to_ink", self.extend_gate)
        return self

    def restore(self) -> None:
        """Вернуть исходные функции."""
        for module, name, value in reversed(self._saved):
            setattr(module, name, value)
        self._saved.clear()

    def __enter__(self) -> "HeadingPatch":
        return self.install()

    def __exit__(self, *exc) -> None:
        self.restore()


__all__ = [
    "HeadingPatch",
    "Variant",
    "cuts_of",
    "different_sets",
    "end_kegl_any",
    "end_tall",
    "glyphs_bottom",
    "glyphs_kegl",
    "same_baseline",
    "split_segment",
]
