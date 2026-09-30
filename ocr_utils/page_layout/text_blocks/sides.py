"""Стороны границы текстового блока (верх, право, низ, лево, углы) и выравнивание текста по вертикальным сторонам.

Граница блока (``BlockEnvelope.polygon``) — замкнутый контур без разметки. Здесь каждое его звено
получает сторону (:class:`SideKind`) и флаг «угол», причём тремя способами (:class:`SidesMethod`),
чтобы их можно было сравнить на оверлеях:

* ``construct`` — по построению: контур собран обходом четырёх кривых огибающей, и звено берёт
  метку своей кривой;
* ``rays`` — модель блока из четырёх границ: верх и низ — первая и последняя строки, лево и право —
  продления этих строк; стороны отсекаются лучами из концов крайних строк (по нормали к оси и по её
  касательной), а в каждом углу ровно один разрез делит угол на горизонтальную и вертикальную части;
* ``frame`` — локальная система координат строк: внешняя нормаль звена сравнивается с местным
  направлением строк, и контур делится ровно на четыре дуги динамическим программированием.

Угловые звенья в меры наклона и изгиба сторон не входят (:func:`side_measures`).

Выравнивание (:class:`AlignMethod`) считается по концам строк у вертикальных сторон тремя способами:
по x от тренда тела блока (как ``alignment.py``), вдоль касательной к строке и устойчивой подгонкой
кривой к концам строк в системе координат блока (как векторы табуляции Tesseract). Ряды собираются
в серии, и вдоль стороны помечаются выровненные и невыровненные участки.

Дополнительная линия вертикальной стороны (:func:`filled_side`) — сторона без невыровненных концов,
с гладкой заплаткой PCHIP вместо невыровненной середины; по ней меряются наклон и изгиб стороны.
"""

from __future__ import annotations

import math

from dataclasses import dataclass, replace
from enum import Enum

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

from ocr_utils.page_layout import mm_to_px, px_to_mm
from ocr_utils.page_layout.text_blocks.alignment import ALIGN_TOL_MM, INDENT_MIN_MM, block_frame
from ocr_utils.page_layout.text_blocks.blocks import GRID_STEP_MM, TAIL_BLEND_GLYPHS, TRIM_MM, BlockEnvelope, Row
from ocr_utils.page_layout.text_blocks.blocks import TextBlock, _row_axis, _smoothed_axis

# Шаг, с которым контур блока пересобирается перед разметкой (мм бумаги): все методы размечают
# ОДИН и тот же пересобранный контур, поэтому их метки можно сравнивать звено в звено, а разрез угла
# ищется с точностью до полумиллиметра.
DENSIFY_MM = 0.5
# По скольким последним мм сглаженной оси берётся касательная к её концу (лучи метода ``rays``,
# направление выравнивания ``tangent``). Меньше слова: изгиб бумаги у конца строки не усредняется.
END_TANGENT_MM = 5.0
# Звено метода ``frame`` считается угловым, если его нормаль ни к одной из четырёх ожидаемых не
# ближе этого угла: 45° ± 15° между «горизонтально» и «вертикально».
FRAME_AMBIGUOUS_DEG = 30.0
# Серия рядов у вертикальной стороны считается выровненной, если в ней столько рядов на кривой.
MIN_RUN_ROWS = 3
# Серия терпит одиночные ряды «мимо» между рядами на кривой — повреждённая строка (краска потеряна
# при бинаризации: 1973/07 с.88, «…монтажных р» вместо «ра-»), непойманный знак у края, — но только
# пока рядов на кривой в ней не меньше этой доли. Иначе рваный край, где «на кривой» и «мимо»
# чередуются, склеился бы в одну выровненную серию.
RUN_MIN_ON_SHARE = 0.75
# RANSAC параболы для метода ``robust``: число проб и зерно (воспроизводимость прогонов).
RANSAC_TRIALS = 200
RANSAC_SEED = 0
# Парабола берётся вместо прямой, только если вписывает хотя бы на столько рядов больше: на
# коротком блоке лишняя степень свободы иначе подгоняется под шум.
PARABOLA_GAIN_ROWS = 2
# Местная кривая стороны (:func:`_fit_local`): у каждого конца строки своя прямая Тейла–Сена по
# ±``LOCAL_FIT_ROWS`` соседним концам (концы, ушедшие внутрь блока, отсеиваются). Нужна краю, у которого наклон меняется по высоте (бумага
# изогнута: край по формату идёт под одним углом вверху, под другим внизу, или S-образно) — одна прямая
# или парабола на весь блок такой край не описывает, и половина выровненных строк объявлялась
# невыровненной (синтетика: S-образный край ±40 px — доля выровненных 0.60 вместо 1.00). Берётся,
# только если вписывает хотя бы на ``PARABOLA_GAIN_ROWS`` концов больше, чем общая кривая.
LOCAL_FIT_ROWS = 5
# Наклон местной прямой не дальше этого от наклона общей (градусы). Изгиб края бумаги пологий
# (S-образный край ±40 px на высоту полосы — до 5°), а зигзаг по серии абзацных отступов списка — от
# 30° (1966/05 0260_2R: без ограничения кривая заходила в отступы, и они считались выровненными;
# 1971/01 IMG_0032_1L — в края рисунка внутри блока).
LOCAL_MAX_TURN_DEG = 6.0
# Раундов одностороннего отсева местной кривой (концы, ушедшие внутрь, выбрасываются и кривая
# строится заново).
LOCAL_FIT_ROUNDS = 3
# Продление короткой крайней строки (:func:`edge_axis`): опора ищется среди стольких рядов от края
# блока; межстрочный интервал до опоры должен лежать в этих пределах, в шагах строк блока (подпись
# автора стоит через пустую строку — два шага; дальше — уже не соседняя строка).
EXTEND_SEARCH_ROWS = 5
EXTEND_MIN_PITCHES = 0.3
EXTEND_MAX_PITCHES = 4.0
# Неуверенные концы верхней и нижней сторон (:func:`_uncertain_ends`): от каждого конца стороны
# столько размеров буквы крайнего ряда (первого у верха, последнего у низа). Там кромка загибается
# в угол по краске крайних букв и продлению ``_extend_ends``, и в меры наклона и изгиба эти участки
# не берутся (просьба пользователя: загибы у концов верхней границы видны на 1975/05 с.97).
UNCERTAIN_GLYPHS = 1.0


class SideKind(str, Enum):
    """Сторона границы блока."""

    LEFT = "left"
    BOTTOM = "bottom"
    RIGHT = "right"
    TOP = "top"

    @property
    def vertical(self) -> bool:
        """Вертикальная ли сторона (левый или правый край текста)."""
        return self in (SideKind.LEFT, SideKind.RIGHT)


# Порядок сторон при обходе контура: так собирает контур ``blocks.envelope_of`` (лево сверху вниз,
# низ слева направо, право снизу вверх, верх справа налево) — в пикселях кадра это обход с
# ОТРИЦАТЕЛЬНОЙ ориентированной площадью; контур другой ориентации разворачивается.
CYCLE = (SideKind.LEFT, SideKind.BOTTOM, SideKind.RIGHT, SideKind.TOP)


class SidesMethod(str, Enum):
    """Способ разметки сторон."""

    CONSTRUCT = "construct"  # по построению огибающей
    RAYS = "rays"  # лучи из концов крайних строк, один разрез в каждом углу
    FRAME = "frame"  # локальная система строк + разбиение на четыре дуги


# Разметка сторон по умолчанию — по построению огибающей (решение пользователя после сравнения на
# оверлеях); ``rays`` и ``frame`` остаются и включаются явно (CLI ``--sides-method``).
DEFAULT_SIDES_METHOD = SidesMethod.CONSTRUCT


class AlignMethod(str, Enum):
    """Способ проверки выравнивания по вертикальной стороне."""

    TREND = "trend"  # по x кадра от тренда тела блока (``alignment.py``)
    TANGENT = "tangent"  # вдоль касательной к строке от того же тренда
    ROBUST = "robust"  # устойчивая подгонка к концам строк в системе координат блока


class RowStatus(str, Enum):
    """Положение конца строки относительно вертикальной стороны."""

    ON = "on"  # на кривой стороны, в допуске
    INDENT = "indent"  # ушёл внутрь блока (абзацный отступ, конец абзаца)
    OFF = "off"  # мимо кривой: рваный край или выступ


@dataclass(frozen=True)
class BlockSides:
    """Разметка контура блока одним методом.

    ``polygon`` — пересобранный контур ``(N, 2)``; звено ``i`` идёт из ``polygon[i]`` в
    ``polygon[(i + 1) % N]``, и ему соответствуют ``labels[i]`` и ``corner[i]``.
    """

    method: SidesMethod
    polygon: np.ndarray
    labels: tuple[SideKind, ...]
    corner: np.ndarray
    # Опорные точки метода (у ``rays`` — восемь точек пересечения лучей с контуром) — для оверлея.
    anchors: np.ndarray | None = None
    # Лучи ``(K, 2, 2)`` — отрезки «начало — точка на контуре», только для оверлея.
    rays: np.ndarray | None = None
    # Неуверенные звенья ``(N,)``: концы верхней и нижней сторон длиной в букву крайнего ряда
    # (:func:`_uncertain_ends`). Рисуются пунктиром, в меры сторон не входят, как и углы.
    uncertain: np.ndarray | None = None

    @property
    def excluded(self) -> np.ndarray:
        """Звенья, которые не берутся в меры: углы и неуверенные концы."""
        return self.corner if self.uncertain is None else self.corner | self.uncertain


@dataclass(frozen=True)
class SideMeasure:
    """Меры одной стороны по её НЕугловым звеньям."""

    side: SideKind
    length_mm: float  # длина стороны целиком, с углами
    corner_share: float  # доля угловой длины
    tilt_deg: float  # наклон: у вертикальной — от вертикали (dx/dy), у горизонтальной — от горизонтали
    bend_mm: float  # размах остатка от хорды


@dataclass(frozen=True)
class RowEnd:
    """Конец (или начало) строки у вертикальной стороны."""

    row: int  # номер ряда в блоке
    point: np.ndarray  # точка конца строки (x края краски, y оси)
    resid_mm: float  # отклонение от кривой стороны, «внутрь блока — плюс»
    status: RowStatus
    aligned: bool  # ряд лежит в выровненной серии


@dataclass(frozen=True)
class SideAlignment:
    """Выравнивание по одной вертикальной стороне одним методом."""

    method: AlignMethod
    side: SideKind
    ends: tuple[RowEnd, ...]
    on_share: float  # доля рядов на кривой среди НЕ-отступов
    aligned_share: float  # доля рядов в выровненных сериях среди рядов без отступов
    # Кривая, от которой мерились отклонения (кадр), — для оверлея.
    curve: np.ndarray | None = None


@dataclass(frozen=True)
class FilledSide:
    """Дополнительная линия вертикальной стороны: без невыровненных концов, с заплатками в середине.

    ``points`` идут сверху вниз в кадре (пиксели рабочей копии); ``filled[i]`` — точка лежит на
    заплатке (гладкой интерполяции поверх невыровненного участка), а не на самой стороне. Меры
    ``tilt_deg``/``bend_mm`` — по этой линии, ``raw_*`` — по всей стороне без углов, той же формулой.
    ``length_mm`` — длина линии поперёк строк: на коротком отрезке наклон шумит (сдвиг в 2 px на 9 мм
    — уже 5°), и меры таких линий надо читать с оглядкой на длину.
    """

    side: SideKind
    method: AlignMethod
    points: np.ndarray
    filled: np.ndarray
    length_mm: float
    tilt_deg: float
    bend_mm: float
    raw_tilt_deg: float
    raw_bend_mm: float


@dataclass(frozen=True)
class EdgeAxis:
    """Ось крайней (первой или последней) строки блока, продлённая до ширины опорной строки.

    ``points`` — вся кривая слева направо: продление влево, настоящая ось, продление вправо.
    Настоящая ось лежит между ``real_x0`` и ``real_x1``; ``reference`` — номер опорного ряда в
    блоке (``None`` — продления нет), ``gap`` — межстрочный интервал до опоры (пиксели, со знаком:
    у первой строки опора ниже, интервал отрицательный).
    """

    points: np.ndarray
    real_x0: float
    real_x1: float
    reference: int | None = None
    gap: float = 0.0

    @property
    def extended(self) -> bool:
        """Продлена ли ось хоть с одной стороны."""
        return self.reference is not None and (
            self.points[0, 0] < self.real_x0 - 1e-6 or self.points[-1, 0] > self.real_x1 + 1e-6
        )


def densify(polygon: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Пересобрать замкнутый контур с шагом не длиннее ``step``.

    Args:
        polygon: Контур ``(N, 2)``, замыкающее звено подразумевается.
        step: Наибольшая длина звена, пиксели.

    Returns:
        ``(points, source)``: новые вершины и для каждой — номер исходного звена, на котором она
        лежит (по нему метод ``construct`` переносит метки).
    """
    points = np.asarray(polygon, dtype=np.float64)
    nxt = np.roll(points, -1, axis=0)
    out, source = [], []
    for index, (start, end) in enumerate(zip(points, nxt)):
        length = float(np.hypot(*(end - start)))
        if length < 1e-9:
            continue  # повтор вершины: звена нет
        parts = max(1, int(math.ceil(length / step)))
        for part in range(parts):
            out.append(start + (end - start) * (part / parts))
            source.append(index)
    return np.asarray(out, dtype=np.float64), np.asarray(source, dtype=np.int64)


def signed_area(polygon: np.ndarray) -> float:
    """Ориентированная площадь контура в пикселях кадра (формула шнурков).

    Args:
        polygon: Контур ``(N, 2)``.

    Returns:
        Площадь со знаком: у контура ``envelope_of`` (лево вниз, низ вправо, …) — отрицательная.
    """
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def outward_normals(polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Внешние нормали, середины и длины звеньев контура с отрицательной площадью.

    Args:
        polygon: Контур ``(N, 2)`` ориентации ``CYCLE``.

    Returns:
        ``(normals, middles, lengths)``: единичные нормали наружу ``(N, 2)``, середины звеньев и
        их длины.
    """
    nxt = np.roll(polygon, -1, axis=0)
    delta = nxt - polygon
    lengths = np.hypot(delta[:, 0], delta[:, 1])
    safe = np.maximum(lengths, 1e-9)
    # Для обхода «лево вниз — низ вправо» наружу смотрит (−dy, dx): у левой стороны (0, 1) → (−1, 0).
    normals = np.column_stack([-delta[:, 1], delta[:, 0]]) / safe[:, None]
    return normals, (polygon + nxt) / 2.0, lengths


def _prepared(envelope: BlockEnvelope, dpi: float) -> tuple[np.ndarray, np.ndarray, bool]:
    """Контур блока, пересобранный с шагом ``DENSIFY_MM`` и приведённый к ориентации ``CYCLE``.

    Args:
        envelope: Огибающая блока (главная, полоса вокруг оси).
        dpi: Разрешение рабочей копии.

    Returns:
        ``(points, source, built)``: вершины, номер исходного звена каждой вершины (в нумерации
        ИСХОДНОГО контура, даже если обход пришлось развернуть) и признак «контур собран обходом
        четырёх кривых» (иначе он растровый, ``_swallow_rows``).
    """
    polygon = np.asarray(envelope.polygon, dtype=np.float64)
    parts = [envelope.left, envelope.bottom, envelope.right[::-1], envelope.top[::-1]]
    stacked = np.vstack(parts)
    built = stacked.shape == polygon.shape and bool(np.allclose(stacked, polygon))
    points, source = densify(polygon, mm_to_px(DENSIFY_MM, dpi))
    if signed_area(points) > 0:
        # Растровый контур cv2.findContours обходит блок в другую сторону — разворачиваем.
        points, source = points[::-1].copy(), source[::-1].copy()
    return points, source, built


def _nearest_curve(middles: np.ndarray, curves: list[np.ndarray]) -> np.ndarray:
    """Для каждой середины звена — номер ближайшей из кривых (по ближайшей вершине кривой).

    Args:
        middles: Середины звеньев ``(N, 2)``.
        curves: Кривые ``(M_k, 2)``.

    Returns:
        Номера кривых ``(N,)``.
    """
    distances = []
    for curve in curves:
        diff = middles[:, None, :] - np.asarray(curve, dtype=np.float64)[None, :, :]
        distances.append(np.hypot(diff[..., 0], diff[..., 1]).min(axis=1))
    return np.argmin(np.column_stack(distances), axis=1)


def _axis_y(points: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Ордината оси в точках ``x`` с продолжением концевых значений за её пределы."""
    return np.interp(x, points[:, 0], points[:, 1])


def _is_full(block: TextBlock, points: np.ndarray) -> bool:
    """Полная ли строка: оба конца её оси не дальше ``TRIM_MM`` от тренда тела блока на её высоте.

    Args:
        block: Текстовый блок (тренд ``core_*`` уже построен огибающей).
        points: Ось строки ``(N, 2)`` слева направо.

    Returns:
        ``True``, если строка покрывает ширину блока.
    """
    y = float(np.median(points[:, 1]))
    left = _core_curve(block.envelope, SideKind.LEFT)
    right = _core_curve(block.envelope, SideKind.RIGHT)
    trim = mm_to_px(TRIM_MM, block.dpi)
    x_left = float(np.interp(y, left[:, 1], left[:, 0]))
    x_right = float(np.interp(y, right[:, 1], right[:, 0]))
    return points[0, 0] <= x_left + trim and points[-1, 0] >= x_right - trim


def _reference_row(block: TextBlock, top: bool) -> tuple[int, np.ndarray] | None:
    """Опора для продления крайней строки: ближайшая ПОЛНАЯ строка среди соседних, иначе самая широкая.

    Соседняя строка сама бывает короткой (конец абзаца над подписью автора, 1968/07 с.93), поэтому
    берётся ближайшая полная среди ``EXTEND_SEARCH_ROWS`` рядов от края. Здесь «полная» проверяема:
    стороны размечаются ПОСЛЕ огибающей, и тренд тела блока известен (у хвоста последней строки в
    ``blocks`` такой возможности нет).

    Args:
        block: Текстовый блок.
        top: Опора для первой строки (ищется вниз), иначе для последней (ищется вверх).

    Returns:
        ``(номер ряда, ось)`` или ``None``, если у соседних рядов нет осей.
    """
    count = len(block.rows)
    order = (
        range(1, min(count, 1 + EXTEND_SEARCH_ROWS))
        if top
        else range(count - 2, max(-1, count - 2 - EXTEND_SEARCH_ROWS), -1)
    )
    axes = [(index, _row_axis(block.rows[index])) for index in order]
    axes = [(index, points) for index, points in axes if points is not None and len(points) >= 2]
    if not axes:
        return None
    full = [item for item in axes if _is_full(block, item[1])]
    if full:
        return full[0]
    return max(axes, key=lambda item: item[1][-1, 0] - item[1][0, 0])


def edge_axis(block: TextBlock, top: bool) -> EdgeAxis | None:
    """Ось крайней строки блока, продлённая до ширины опорной строки по её изгибу.

    Короткая крайняя строка (абзацный отступ первой, конец абзаца или подпись автора в последней)
    не покрывает ширину блока, и лучи из её концов выходят из середины блока (1968/07 с.93:
    подпись «А. ИВАНОВА» x 705..825 при блоке 117..892, луч из её курсивного конца бил в низ блока).
    Продление — как хвост последней строки (``blocks._tail_of``), но в обе стороны и у обеих
    крайних строк: опора (:func:`_reference_row`), сглаженная, сдвигается на межстрочный интервал
    (медиана расхождения на общем участке), на первой ширине символа — плавный переход от
    фактического интервала на конце оси к медианному.

    Args:
        block: Текстовый блок.
        top: Первая строка (иначе последняя).

    Returns:
        Продлённая ось; без продления, если строка полная, опоры нет, опора не шире строки или
        интервал вне ``EXTEND_MIN_PITCHES``…``EXTEND_MAX_PITCHES`` шагов; ``None`` — у ряда нет оси.
    """
    own = _row_axis(block.rows[0] if top else block.rows[-1])
    if own is None:
        return None
    plain = EdgeAxis(own, float(own[0, 0]), float(own[-1, 0]))
    if len(block.rows) < 2 or _is_full(block, own):
        return plain
    found = _reference_row(block, top)
    if found is None:
        return plain
    reference, base = found
    if base[0, 0] >= own[0, 0] and base[-1, 0] <= own[-1, 0]:
        return plain  # опора не шире самой строки — продлевать не до чего
    if len(base) >= 3 and base[-1, 0] - base[0, 0] > 2 * mm_to_px(1.0, block.dpi):
        ref_xs, ref_ys = _smoothed_axis(base, block.dpi)
    else:
        ref_xs, ref_ys = base[:, 0], base[:, 1]
    # Интервал — медиана расхождения на общем участке; вне его np.interp берёт концевое значение опоры.
    gap = float(np.median(own[:, 1] - np.interp(own[:, 0], ref_xs, ref_ys)))
    pitch = max(block.pitch_px, 1e-6)
    sign_ok = gap < 0 if top else gap > 0
    if not sign_ok or not EXTEND_MIN_PITCHES <= abs(gap) / pitch <= EXTEND_MAX_PITCHES:
        return EdgeAxis(own, float(own[0, 0]), float(own[-1, 0]), reference, gap)
    step = mm_to_px(GRID_STEP_MM, block.dpi)
    blend = max(TAIL_BLEND_GLYPHS * block.glyph_size[0], 1e-6)
    parts = []
    if ref_xs[0] < own[0, 0] - step / 2.0:
        # Влево: от начала опоры до начала строки, без самой точки начала строки.
        xs = np.arange(own[0, 0] - step, ref_xs[0], -step)[::-1]
        xs = np.concatenate([[ref_xs[0]], xs]) if xs.size == 0 or xs[0] - ref_xs[0] > 1e-6 else xs
        start_gap = float(own[0, 1] - np.interp(own[0, 0], ref_xs, ref_ys))
        weight = np.clip((own[0, 0] - xs) / blend, 0.0, 1.0)
        parts.append(np.column_stack([xs, np.interp(xs, ref_xs, ref_ys) + (1.0 - weight) * start_gap + weight * gap]))
    parts.append(own)
    if ref_xs[-1] > own[-1, 0] + step / 2.0:
        # Вправо: от конца строки до конца опоры.
        xs = np.arange(own[-1, 0] + step, ref_xs[-1], step)
        xs = np.append(xs, ref_xs[-1]) if xs.size == 0 or ref_xs[-1] - xs[-1] > 1e-6 else xs
        end_gap = float(own[-1, 1] - np.interp(own[-1, 0], ref_xs, ref_ys))
        weight = np.clip((xs - own[-1, 0]) / blend, 0.0, 1.0)
        parts.append(np.column_stack([xs, np.interp(xs, ref_xs, ref_ys) + (1.0 - weight) * end_gap + weight * gap]))
    return EdgeAxis(np.vstack(parts), float(own[0, 0]), float(own[-1, 0]), reference, gap)


def _edge_points(block: TextBlock, top: bool) -> np.ndarray | None:
    """Точки продлённой оси крайней строки или ``None``, если у ряда нет оси."""
    edge = edge_axis(block, top)
    return None if edge is None else edge.points


def _inside_part(axis: np.ndarray, polygon: np.ndarray, at_start: bool) -> np.ndarray:
    """Ось без концевого участка, вышедшего за контур блока.

    Продлённая ось крайней строки (:func:`edge_axis`) идёт до ширины ОПОРНОЙ строки, а контур у
    короткой строки может срезать угол по её концу (1968/07 с.93: низ блока справа идёт по концу
    подписи). Луч из точки вне контура бьёт куда попало, поэтому у этого конца ось укорачивается до
    последней точки внутри контура.

    Args:
        axis: Ось ``(N, 2)`` слева направо.
        polygon: Контур блока.
        at_start: Укорачивать с левого конца (иначе с правого).

    Returns:
        Та же ось или её часть; не короче двух точек.
    """
    hull = polygon.astype(np.float32).reshape(-1, 1, 2)
    inside = np.array([cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0 for x, y in axis])
    if not inside.any():
        return axis
    if at_start:
        first = int(np.argmax(inside))
        return axis[min(first, len(axis) - 2) :]
    last = len(axis) - 1 - int(np.argmax(inside[::-1]))
    return axis[: max(last + 1, 2)]


def sides_construct(block: TextBlock) -> BlockSides:
    """Разметка по построению: звено берёт метку кривой огибающей, из которой собран контур.

    Угловыми считаются звенья верха и низа за концами крайних строк (там кромка — продление
    ``_extend_ends``, а не сама строка) и звенья боковин выше оси первой строки или ниже оси
    последней (там боковина уже поворачивает в крышку). Короткая крайняя строка берётся продлённой
    по опоре (:func:`edge_axis`): иначе весь низ под подписью автора считался бы углом.

    Args:
        block: Текстовый блок.

    Returns:
        Разметка контура.
    """
    envelope = block.envelope
    points, source, built = _prepared(envelope, block.dpi)
    _, middles, _ = outward_normals(points)
    if built:
        # Контур — это left, bottom, right[::-1], top[::-1] подряд: по номеру исходного звена видно,
        # из какой кривой вершина. Звено-стык между кривыми отходит к той, откуда начинается.
        bounds = np.cumsum([len(envelope.left), len(envelope.bottom), len(envelope.right), len(envelope.top)])
        which = np.searchsorted(bounds, source, side="right")
    else:
        which = _nearest_curve(middles, [envelope.left, envelope.bottom, envelope.right, envelope.top])
    labels = tuple(CYCLE[int(item)] for item in which)
    top_axis = _edge_points(block, top=True)
    bottom_axis = _edge_points(block, top=False)
    corner = np.zeros(len(points), dtype=bool)
    for index, (label, middle) in enumerate(zip(labels, middles)):
        if label is SideKind.TOP and top_axis is not None:
            corner[index] = not top_axis[0, 0] <= middle[0] <= top_axis[-1, 0]
        elif label is SideKind.BOTTOM and bottom_axis is not None:
            corner[index] = not bottom_axis[0, 0] <= middle[0] <= bottom_axis[-1, 0]
        elif label.vertical and top_axis is not None and bottom_axis is not None:
            upper = float(_axis_y(top_axis, np.array([middle[0]]))[0])
            lower = float(_axis_y(bottom_axis, np.array([middle[0]]))[0])
            corner[index] = not upper <= middle[1] <= lower
    return BlockSides(SidesMethod.CONSTRUCT, points, labels, corner)


def end_tangent(points: np.ndarray, dpi: float, at_start: bool) -> tuple[np.ndarray, np.ndarray]:
    """Точка конца оси и единичная касательная к ней слева направо, по сглаженной оси.

    Args:
        points: Ось ``(N, 2)`` слева направо.
        dpi: Разрешение рабочей копии.
        at_start: Левый конец (иначе правый).

    Returns:
        ``(point, tangent)``; у оси короче двух узлов — хорда целиком.
    """
    if len(points) >= 3 and points[-1, 0] - points[0, 0] > 2 * mm_to_px(1.0, dpi):
        xs, ys = _smoothed_axis(points, dpi)
    else:
        xs, ys = points[:, 0], points[:, 1]
    reach = mm_to_px(END_TANGENT_MM, dpi)
    near = xs <= xs[0] + reach if at_start else xs >= xs[-1] - reach
    if near.sum() >= 2 and np.ptp(xs[near]) > 0:
        slope = float(np.polyfit(xs[near], ys[near], 1)[0])
    elif xs[-1] - xs[0] > 0:
        slope = float((ys[-1] - ys[0]) / (xs[-1] - xs[0]))
    else:
        slope = 0.0
    tangent = np.array([1.0, slope]) / math.hypot(1.0, slope)
    point = np.array([xs[0], ys[0]]) if at_start else np.array([xs[-1], ys[-1]])
    return point, tangent


def ray_hit(polygon: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> tuple[int, np.ndarray] | None:
    """Первое пересечение луча с контуром.

    Args:
        polygon: Контур ``(N, 2)``.
        origin: Начало луча.
        direction: Направление луча (не обязательно единичное).

    Returns:
        ``(index, point)``: номер ближайшей к пересечению вершины контура и сама точка; ``None``,
        если луч контур не пересёк.
    """
    a = polygon
    b = np.roll(polygon, -1, axis=0)
    edge = b - a
    # origin + lam·d = a + mu·edge  →  lam·d − mu·edge = a − origin, решается по Крамеру.
    det = direction[0] * (-edge[:, 1]) - direction[1] * (-edge[:, 0])
    rhs = a - origin
    ok = np.abs(det) > 1e-12
    lam = np.full(len(a), np.inf)
    mu = np.full(len(a), -1.0)
    lam[ok] = (rhs[ok, 0] * (-edge[ok, 1]) - rhs[ok, 1] * (-edge[ok, 0])) / det[ok]
    mu[ok] = (direction[0] * rhs[ok, 1] - direction[1] * rhs[ok, 0]) / det[ok]
    hit = ok & (lam > 1e-9) & (mu >= 0.0) & (mu <= 1.0)
    if not hit.any():
        return None
    best = int(np.argmin(np.where(hit, lam, np.inf)))
    point = origin + lam[best] * direction
    index = best if mu[best] < 0.5 else (best + 1) % len(polygon)
    return index, point


def _nearest_vertex(polygon: np.ndarray, point: np.ndarray) -> int:
    """Номер вершины контура, ближайшей к точке."""
    return int(np.argmin(np.hypot(*(polygon - point).T)))


def _label_costs(normals: np.ndarray, lengths: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """Цена каждой из четырёх меток для звеньев при данном направлении строк.

    Ожидаемая внешняя нормаль: у левой стороны — против строки, у правой — по строке, у верха —
    нормаль к строке вверх, у низа — вниз. Цена метки — длина звена, умноженная на
    ``(1 − cos)/2`` угла между его нормалью и ожидаемой: 0 при совпадении, длина — при
    противоположной, половина длины — под прямым углом.

    Args:
        normals: Внешние нормали ``(N, 2)``.
        lengths: Длины звеньев ``(N,)``.
        tangent: Направление строк ``(2,)`` или по звену ``(N, 2)``, единичное, слева направо.

    Returns:
        Цены ``(N, 4)`` в порядке ``CYCLE``.
    """
    tangent = np.broadcast_to(tangent, normals.shape)
    up = np.column_stack([tangent[:, 1], -tangent[:, 0]])  # (tx, ty) → (ty, −tx): вверх в кадре
    expected = (-tangent, up * -1.0, tangent, up)  # LEFT, BOTTOM, RIGHT, TOP
    cosines = np.column_stack([np.sum(normals * item, axis=1) for item in expected])
    return lengths[:, None] * (1.0 - cosines) / 2.0


def _split_corner(costs: np.ndarray, first: int, second: int) -> int:
    """Одна точка разреза дуги угла: слева от неё метка ``first``, справа — ``second``.

    Перебор всех разрезов по накопленным суммам, O(n). При равной цене берётся середина лучшего
    диапазона — разрез не жмётся к краю дуги.

    Args:
        costs: Цены меток звеньев дуги ``(n, 4)``.
        first: Номер метки (в ``CYCLE``) до разреза.
        second: Номер метки после.

    Returns:
        Сколько звеньев дуги получает метку ``first`` (от 0 до n).
    """
    n = costs.shape[0]
    before = np.concatenate([[0.0], np.cumsum(costs[:, first])])
    after = np.concatenate([np.cumsum(costs[::-1, second])[::-1], [0.0]])
    total = before + after
    best = np.flatnonzero(np.isclose(total, total.min()))
    return int(best[len(best) // 2]) if n else 0


def sides_rays(block: TextBlock) -> BlockSides:
    """Разметка лучами из концов крайних строк (модель блока из четырёх границ).

    Восемь точек на контуре: из обоих концов первой строки — луч по нормали вверх (граница верха) и
    продление по касательной наружу (граница боковины); из концов последней строки —
    луч вниз и продление наружу. Между соседними точками, лежащими по разные стороны угла, — дуга
    угла; в ней ровно один разрез (:func:`_split_corner`) по цене против ориентации относительно
    строки этого угла. Если точки угла «перехлестнулись», дуга пуста и разрез ставится посередине.

    Args:
        block: Текстовый блок.

    Returns:
        Разметка контура; если у крайних рядов нет осей — разметка по построению.
    """
    points, _, _ = _prepared(block.envelope, block.dpi)
    # Крайние строки — продлённые по опоре (:func:`edge_axis`): у короткой строки лучи иначе выходят
    # из середины блока, а касательная у её загнутого курсивного конца разворачивает луч боковины.
    top_axis = _edge_points(block, top=True)
    bottom_axis = _edge_points(block, top=False)
    if top_axis is None or bottom_axis is None:
        return sides_construct(block)
    normals, _, lengths = outward_normals(points)
    n = len(points)
    anchors, rays, positions, tangents = [], [], {}, {}
    # Для каждого угла: (ось, левый ли конец, направление луча крышки, направление луча боковины).
    for corner, axis, at_start, cap_sign in (
        ("tl", top_axis, True, 1.0),
        ("tr", top_axis, False, 1.0),
        ("bl", bottom_axis, True, -1.0),
        ("br", bottom_axis, False, -1.0),
    ):
        origin, tangent = end_tangent(_inside_part(axis, points, at_start), block.dpi, at_start)
        tangents[corner] = tangent
        up = np.array([tangent[1], -tangent[0]])
        for kind, direction in (("cap", cap_sign * up), ("side", -tangent if at_start else tangent)):
            hit = ray_hit(points, origin, direction)
            if hit is None:
                # Луч мимо контура (конец строки вылез за кромку): берём ближайшую к концу вершину.
                nearest = _nearest_vertex(points, origin)
                hit = (nearest, points[nearest])
            index, point = hit
            positions[(corner, kind)] = index
            anchors.append(point)
            rays.append([origin, point])
    # Порядок точек по обходу (лево вниз, низ вправо, право вверх, верх влево): от боковины у
    # верхнего левого угла — к боковине у нижнего левого, крышке низа слева, … , крышке верха слева.
    order = [
        ("tl", "side"),
        ("bl", "side"),
        ("bl", "cap"),
        ("br", "cap"),
        ("br", "side"),
        ("tr", "side"),
        ("tr", "cap"),
        ("tl", "cap"),
    ]
    start = positions[order[0]]
    rel = np.array([(positions[key] - start) % n for key in order], dtype=np.float64)
    rel = _monotone(rel, n)
    # Стороны между точками и дуги углов: (метка стороны, от, до) и (угол, метка до разреза, после).
    labels = np.empty(n, dtype=np.int64)
    corner = np.zeros(n, dtype=bool)
    spans = [(0, 1, 0), (2, 3, 1), (4, 5, 2), (6, 7, 3)]  # LEFT: боковина tl → боковина bl  # BOTTOM  # RIGHT  # TOP
    for begin, end, label in spans:
        for offset in range(int(rel[begin]), int(rel[end])):
            labels[(start + offset) % n] = label
    corners = [
        (1, 2, 0, 1, "bl"),  # от боковины bl к крышке bl: LEFT → BOTTOM
        (3, 4, 1, 2, "br"),  # BOTTOM → RIGHT
        (5, 6, 2, 3, "tr"),  # RIGHT → TOP
        (7, 8, 3, 0, "tl"),  # TOP → LEFT (через конец обхода)
    ]
    for begin, end, first, second, name in corners:
        lo = int(rel[begin])
        hi = int(rel[end]) if end < len(rel) else n
        idx = np.array([(start + offset) % n for offset in range(lo, hi)], dtype=np.int64)
        if not idx.size:
            continue
        costs = _label_costs(normals[idx], lengths[idx], tangents[name])
        cut = _split_corner(costs, first, second)
        labels[idx[:cut]] = first
        labels[idx[cut:]] = second
        corner[idx] = True
    return BlockSides(
        SidesMethod.RAYS,
        points,
        tuple(CYCLE[int(item)] for item in labels),
        corner,
        anchors=np.asarray(anchors),
        rays=np.asarray(rays),
    )


def _monotone(rel: np.ndarray, n: int) -> np.ndarray:
    """Сделать позиции восьми точек неубывающими по обходу (перехлёст → общая середина).

    Луч, ударивший не в ту сторону контура (короткая крайняя строка, скруглённый угол), ставит свою
    точку раньше предыдущей. Такая пара заменяется их средним — дуга угла между ними пуста, и разрез
    встаёт в эту общую точку. Первая позиция — 0 по построению.

    Args:
        rel: Позиции ``(8,)`` по обходу от первой точки, в вершинах.
        n: Число вершин контура.

    Returns:
        Неубывающие позиции той же длины.
    """
    out = rel.copy()
    for _ in range(len(out)):
        bad = np.flatnonzero(np.diff(out) < 0)
        if not bad.size:
            break
        i = int(bad[0])
        middle = (out[i] + out[i + 1]) / 2.0
        out[i] = out[i + 1] = middle
    out = np.clip(np.round(np.maximum.accumulate(out)), 0, n)
    # Отсчёт идёт от первой точки: она обязана остаться в нуле, иначе звенья до неё без метки.
    out[0] = 0.0
    return out


def _direction_field(block: TextBlock, middles: np.ndarray) -> np.ndarray:
    """Местное направление строк в точках контура: по двум ближайшим по высоте рядам.

    Для каждого ряда берётся сглаженная ось, в абсциссе точки — её ордината и наклон (за концами
    оси — концевые). Направление — средний наклон двух рядов, ближайших по вертикали, с весами
    обратно расстоянию.

    Args:
        block: Текстовый блок.
        middles: Точки ``(N, 2)``.

    Returns:
        Единичные направления ``(N, 2)`` слева направо.
    """
    levels, slopes = [], []
    for row in block.rows:
        points = _row_axis(row, with_tail=True)
        if points is None:
            continue
        if len(points) >= 3 and points[-1, 0] - points[0, 0] > 2 * mm_to_px(1.0, block.dpi):
            xs, ys = _smoothed_axis(points, block.dpi)
        else:
            xs, ys = points[:, 0], points[:, 1]
        grad = np.gradient(ys, xs) if len(xs) >= 2 and np.all(np.diff(xs) > 0) else np.zeros_like(ys)
        levels.append(np.interp(middles[:, 0], xs, ys))
        slopes.append(np.interp(middles[:, 0], xs, grad))
    if not levels:
        return np.tile([1.0, 0.0], (len(middles), 1))
    levels, slopes = np.column_stack(levels), np.column_stack(slopes)
    distance = np.abs(levels - middles[:, 1:2])
    order = np.argsort(distance, axis=1)[:, :2]
    near = np.take_along_axis(distance, order, axis=1)
    weight = 1.0 / np.maximum(near, 1.0)
    slope = np.sum(np.take_along_axis(slopes, order, axis=1) * weight, axis=1) / np.sum(weight, axis=1)
    field = np.column_stack([np.ones_like(slope), slope])
    return field / np.hypot(field[:, 0], field[:, 1])[:, None]


def _cyclic_arcs(costs: np.ndarray) -> np.ndarray:
    """Разбить контур на четыре дуги в порядке ``CYCLE`` с наименьшей суммарной ценой.

    Обход начинается со звена, увереннее всех лежащего на левой стороне (минимум цены LEFT), и
    линейное ДП идёт по состояниям LEFT → BOTTOM → RIGHT → TOP → LEFT (последнее — хвост левой
    стороны, разрезанной началом обхода). Каждая дуга непустая.

    Args:
        costs: Цены меток ``(N, 4)`` в порядке ``CYCLE``.

    Returns:
        Номера меток ``(N,)`` в порядке ``CYCLE``.
    """
    n = costs.shape[0]
    start = int(np.argmin(costs[:, 0] - costs[:, 1:].min(axis=1)))
    rolled = np.roll(costs, -start, axis=0)
    states = 5
    column = [0, 1, 2, 3, 0]  # состояние → метка
    total = np.full((n, states), np.inf)
    back = np.zeros((n, states), dtype=np.int64)
    total[0, 0] = rolled[0, 0]
    for i in range(1, n):
        for state in range(states):
            stay = total[i - 1, state]
            move = total[i - 1, state - 1] if state > 0 else np.inf
            if move < stay:
                total[i, state], back[i, state] = move + rolled[i, column[state]], state - 1
            else:
                total[i, state], back[i, state] = stay + rolled[i, column[state]], state
    # Конец — в TOP (левая сторона целиком в начале) или в хвосте LEFT.
    state = 3 if total[-1, 3] <= total[-1, 4] else 4
    labels = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        labels[i] = column[state]
        state = back[i, state]
    return np.roll(labels, start)


def sides_frame(block: TextBlock) -> BlockSides:
    """Разметка в локальной системе строк: цена меток по нормали звена и разбиение на четыре дуги.

    Угловыми считаются звенья, где итоговая метка не совпала с лучшей для звена самого по себе, и
    звенья, чья нормаль ни к одной из ожидаемых не ближе ``FRAME_AMBIGUOUS_DEG``.

    Args:
        block: Текстовый блок.

    Returns:
        Разметка контура.
    """
    points, _, _ = _prepared(block.envelope, block.dpi)
    normals, middles, lengths = outward_normals(points)
    field = _direction_field(block, middles)
    costs = _label_costs(normals, np.ones_like(lengths), field)  # цена на единицу длины
    labels = _cyclic_arcs(costs * lengths[:, None])
    raw = np.argmin(costs, axis=1)
    # Косинус лучшего совпадения нормали с ожидаемой: (1 − cos)/2 = цена → cos = 1 − 2·цена.
    best_cos = 1.0 - 2.0 * costs.min(axis=1)
    corner = (labels != raw) | (best_cos < math.cos(math.radians(FRAME_AMBIGUOUS_DEG)))
    return BlockSides(SidesMethod.FRAME, points, tuple(CYCLE[int(item)] for item in labels), corner)


SIDES_METHODS = {SidesMethod.CONSTRUCT: sides_construct, SidesMethod.RAYS: sides_rays, SidesMethod.FRAME: sides_frame}


def sides_of(block: TextBlock, method: SidesMethod = DEFAULT_SIDES_METHOD) -> BlockSides:
    """Разметка контура блока выбранным методом.

    Args:
        block: Текстовый блок.
        method: Метод разметки; по умолчанию ``DEFAULT_SIDES_METHOD`` (по построению).

    Returns:
        Разметка контура.
    """
    sides = SIDES_METHODS[method](block)
    return replace(sides, uncertain=_uncertain_ends(sides, block))


def _letter_size(row: Row, block: TextBlock) -> float:
    """Размер буквы ряда (пиксели рабочей копии): больший из медианных размеров глифа ряда, иначе блока."""
    size = max(row.glyph_w, row.glyph_h)
    return size if size > 0 else max(block.glyph_size)


def _uncertain_ends(sides: BlockSides, block: TextBlock) -> np.ndarray:
    """Концы верхней и нижней сторон длиной в букву крайнего ряда — неуверенные звенья.

    У концов верх и низ блока загибаются в угол: кромка там идёт по краске крайних букв и по
    продлению за концом строки, а не по самой строке. Меры наклона и изгиба стороны от этого
    дёргаются, поэтому от каждого конца дуги стороны отмеряется ``UNCERTAIN_GLYPHS`` размеров буквы
    крайнего ряда (первого у верха, последнего у низа) вдоль контура, и эти звенья помечаются.

    Args:
        sides: Разметка контура.
        block: Текстовый блок.

    Returns:
        Флаги ``(N,)`` по звеньям контура.
    """
    _, _, lengths = outward_normals(sides.polygon)
    n = len(lengths)
    out = np.zeros(n, dtype=bool)
    for side, row in ((SideKind.TOP, block.rows[0]), (SideKind.BOTTOM, block.rows[-1])):
        own = np.array([label is side for label in sides.labels])
        if not own.any() or own.all():
            continue
        # Дуга стороны по обходу: начинается сразу после звена чужой стороны.
        start = int(np.flatnonzero(own & ~np.roll(own, 1))[0])
        arc = []
        index = start
        while own[index] and len(arc) < n:
            arc.append(index)
            index = (index + 1) % n
        arc = np.array(arc)
        along = np.cumsum(lengths[arc]) - lengths[arc] / 2.0  # середина звена от начала дуги
        limit = UNCERTAIN_GLYPHS * _letter_size(row, block)
        out[arc[(along < limit) | (along > along[-1] + lengths[arc[-1]] / 2.0 - limit)]] = True
    return out


def _theil_sen(xs: np.ndarray, ys: np.ndarray) -> float:
    """Медианный наклон dy/dx по всем парам точек (устойчив к выбросам)."""
    i, j = np.triu_indices(len(xs), 1)
    dx = xs[j] - xs[i]
    ok = np.abs(dx) > 1e-6
    return float(np.median((ys[j] - ys[i])[ok] / dx[ok])) if ok.any() else 0.0


def side_measures(sides: BlockSides, dpi: float) -> tuple[SideMeasure, ...]:
    """Меры сторон по неугловым звеньям: длина, доля углов, наклон, изгиб.

    Args:
        sides: Разметка контура.
        dpi: Разрешение рабочей копии.

    Returns:
        По мере на каждую сторону в порядке ``CYCLE``.
    """
    _, middles, lengths = outward_normals(sides.polygon)
    labels = np.array([CYCLE.index(item) for item in sides.labels])
    out = []
    for number, side in enumerate(CYCLE):
        own = labels == number
        total = float(lengths[own].sum())
        body = own & ~sides.excluded
        corner_share = float(lengths[own & sides.excluded].sum() / total) if total > 0 else 0.0
        tilt = bend = 0.0
        if body.sum() >= 3:
            tilt, bend = _tilt_bend(middles[body], side.vertical, dpi)
        out.append(SideMeasure(side, px_to_mm(total, dpi), corner_share, tilt, bend))
    return tuple(out)


def _tilt_bend(points: np.ndarray, vertical: bool, dpi: float) -> tuple[float, float]:
    """Наклон и изгиб кривой стороны — общая мера для всей стороны и для дополнительной линии.

    Args:
        points: Точки кривой ``(N, 2)`` в кадре, пиксели рабочей копии; порядок не важен.
        vertical: Вертикальная ли сторона: у неё «абсцисса» — y (x как функция y), у горизонтальной — x.
        dpi: Разрешение рабочей копии.

    Returns:
        ``(tilt_deg, bend_mm)``: наклон Тейла–Сена в градусах (у вертикальной — от вертикали) и размах
        остатка от хорды между крайними точками в мм. Меньше трёх точек — ``(0.0, 0.0)``.
    """
    if len(points) < 3:
        return 0.0, 0.0
    along, across = (points[:, 1], points[:, 0]) if vertical else (points[:, 0], points[:, 1])
    order = np.argsort(along)
    along, across = along[order], across[order]
    sample = slice(None, None, max(1, len(along) // 200))  # Тейл–Сен квадратичен по точкам
    tilt = math.degrees(math.atan(_theil_sen(along[sample], across[sample])))
    bend = 0.0
    if along[-1] - along[0] > 0:
        chord = across[0] + (across[-1] - across[0]) * (along - along[0]) / (along[-1] - along[0])
        bend = px_to_mm(float(np.ptp(across - chord)), dpi)
    return tilt, bend


def label_agreement(first: BlockSides, second: BlockSides) -> float:
    """Доля длины контура, которой два метода дали одну и ту же сторону.

    Args:
        first, second: Разметки одного и того же пересобранного контура.

    Returns:
        Доля от 0 до 1.
    """
    _, _, lengths = outward_normals(first.polygon)
    same = np.array([a is b for a, b in zip(first.labels, second.labels)])
    return float(lengths[same].sum() / max(lengths.sum(), 1e-9))


def _row_end(row: Row, side: SideKind, dpi: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Конец строки у стороны: точка (x края краски, y оси) и касательная к оси там.

    Args:
        row: Ряд блока.
        side: ``LEFT`` — начало строки, ``RIGHT`` — конец.
        dpi: Разрешение рабочей копии.

    Returns:
        ``(point, tangent)`` или ``None``, если у ряда нет осей.
    """
    points = _row_axis(row)
    if points is None:
        return None
    at_start = side is SideKind.LEFT
    _, tangent = end_tangent(points, dpi, at_start)
    x = row.x0 if at_start else row.x1
    return np.array([x, float(_axis_y(points, np.array([x]))[0])]), tangent


def _line_polyline(point: np.ndarray, direction: np.ndarray, curve: np.ndarray) -> float | None:
    """Параметр ``lam`` ближайшего пересечения прямой ``point + lam·direction`` с ломаной.

    Args:
        point: Точка прямой.
        direction: Единичное направление.
        curve: Ломаная ``(M, 2)``.

    Returns:
        ``lam`` с наименьшим модулем или ``None``, если прямая ломаную не пересекает.
    """
    a, b = curve[:-1], curve[1:]
    edge = b - a
    det = direction[0] * (-edge[:, 1]) - direction[1] * (-edge[:, 0])
    rhs = a - point
    ok = np.abs(det) > 1e-12
    if not ok.any():
        return None
    lam = (rhs[ok, 0] * (-edge[ok, 1]) - rhs[ok, 1] * (-edge[ok, 0])) / det[ok]
    mu = (direction[0] * rhs[ok, 1] - direction[1] * rhs[ok, 0]) / det[ok]
    inside = (mu >= 0.0) & (mu <= 1.0)
    if not inside.any():
        return None
    return float(lam[inside][np.argmin(np.abs(lam[inside]))])


def _statuses(resid_mm: np.ndarray) -> list[RowStatus]:
    """Положение концов строк по отклонениям (мм, «внутрь — плюс»)."""
    out = []
    for value in resid_mm:
        if abs(value) <= ALIGN_TOL_MM:
            out.append(RowStatus.ON)
        elif value >= INDENT_MIN_MM:
            out.append(RowStatus.INDENT)
        else:
            out.append(RowStatus.OFF)
    return out


def aligned_runs(statuses: list[RowStatus]) -> np.ndarray:
    """Какие ряды лежат в выровненных сериях.

    Серия — подряд идущие ряды «на кривой». Одиночный ряд между двумя рядами на кривой серию не
    рвёт и входит в неё: отступ (абзацный отступ слева, конец абзаца справа) и ряд «мимо»
    (повреждённая строка, непойманный знак у края). Серия выровнена, если в ней не меньше
    ``MIN_RUN_ROWS`` рядов на кривой и они составляют не меньше ``RUN_MIN_ON_SHARE`` серии.

    Args:
        statuses: Положения концов строк сверху вниз.

    Returns:
        Флаги ``(n,)``.
    """
    n = len(statuses)
    member = np.array([item is RowStatus.ON for item in statuses])
    for i in range(1, n - 1):
        if statuses[i] is not RowStatus.ON and member[i - 1] and statuses[i + 1] is RowStatus.ON:
            member[i] = True
    # Последний ряд блока — конец абзаца: одиночный отступ там тоже не рвёт серию.
    if n >= 2 and statuses[-1] is RowStatus.INDENT and member[-2]:
        member[-1] = True
    out = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if not member[i]:
            i += 1
            continue
        j = i
        while j < n and member[j]:
            j += 1
        on = sum(statuses[k] is RowStatus.ON for k in range(i, j))
        if on >= MIN_RUN_ROWS and on >= RUN_MIN_ON_SHARE * (j - i):
            out[i:j] = True
        i = j
    return out


def _core_curve(envelope: BlockEnvelope, side: SideKind) -> np.ndarray:
    """Тренд тела блока с нужной стороны (``core_*``), иначе внешняя кромка."""
    if side is SideKind.LEFT:
        return envelope.core_left if envelope.core_left is not None else envelope.left
    return envelope.core_right if envelope.core_right is not None else envelope.right


def _resid_trend(block: TextBlock, side: SideKind, points: np.ndarray) -> np.ndarray:
    """Отклонения по x кадра от тренда тела блока, «внутрь — плюс» (пиксели) — как ``alignment.py``."""
    curve = _core_curve(block.envelope, side)
    fitted = np.interp(points[:, 1], curve[:, 1], curve[:, 0])
    return (points[:, 0] - fitted) if side is SideKind.LEFT else (fitted - points[:, 0])


def _resid_tangent(block: TextBlock, side: SideKind, points: np.ndarray, tangents: np.ndarray) -> np.ndarray:
    """Отклонения от тренда тела блока ВДОЛЬ касательной к строке, «внутрь — плюс» (пиксели).

    Прямая через конец строки по её касательной пересекается с кривой тренда. Касательная
    направлена слева направо, поэтому у левой стороны конец внутри блока, когда до кривой надо идти
    назад (``lam < 0``), а у правой — вперёд. Прямая, кривую не встретившая (конец за пределами
    тренда по высоте), меряется по x, как ``trend``.
    """
    curve = _core_curve(block.envelope, side)
    fallback = _resid_trend(block, side, points)
    out = np.empty(len(points))
    for index, (point, tangent) in enumerate(zip(points, tangents)):
        lam = _line_polyline(point, tangent, curve)
        if lam is None:
            out[index] = fallback[index]
        else:
            out[index] = -lam if side is SideKind.LEFT else lam
    return out


def _fit_robust(v: np.ndarray, u: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Устойчивая кривая ``u(v)`` по концам строк: Тейл–Сен для прямой, RANSAC для параболы.

    Как векторы табуляции Tesseract: ищется кривая, на которую ложится больше всего концов, и
    выбросы (отступы, концы абзацев) на неё не влияют. Парабола берётся, только если вписывает хотя
    бы на ``PARABOLA_GAIN_ROWS`` рядов больше прямой.

    Args:
        v: Координата поперёк строк (вниз по блоку).
        u: Координата вдоль строк (концы).
        tol: Допуск «вписался», в тех же единицах.

    Returns:
        ``(coefficients, fitted)``: коэффициенты полинома (``np.polyval``) и значения в ``v``.
    """
    slope = _theil_sen(v, u)
    line = np.array([slope, float(np.median(u - slope * v))])
    best, best_in = line, int((np.abs(u - np.polyval(line, v)) <= tol).sum())
    if len(v) >= 5:
        rng = np.random.default_rng(RANSAC_SEED)
        parabola, parabola_in = None, -1
        for _ in range(RANSAC_TRIALS):
            pick = rng.choice(len(v), 3, replace=False)
            if np.ptp(v[pick]) <= 0:
                continue
            coef = np.polyfit(v[pick], u[pick], 2)
            inliers = np.abs(u - np.polyval(coef, v)) <= tol
            if inliers.sum() > parabola_in and inliers.sum() >= 3:
                parabola, parabola_in = np.polyfit(v[inliers], u[inliers], 2), int(inliers.sum())
        if parabola is not None and parabola_in >= best_in + PARABOLA_GAIN_ROWS:
            best = parabola
    return best, np.polyval(best, v)


def _fit_local(v: np.ndarray, u: np.ndarray, inward: float, start: np.ndarray, tol: float) -> np.ndarray:
    """Местная устойчивая кривая ``u(v)``: у каждого конца — прямая Тейла–Сена по ±``LOCAL_FIT_ROWS`` соседям.

    Отсев односторонний и повторяется ``LOCAL_FIT_ROUNDS`` раз: концы, ушедшие от текущей кривой
    ВНУТРЬ блока дальше допуска (абзацный отступ, серия отступов списка, края рисунка внутри блока),
    в прямые не берутся, а изгиб края бумаги сдвигает концы в обе стороны и остаётся. Соседи —
    ближайшие по ``v`` оставшиеся концы (по порядку, не по расстоянию). Наклон каждой прямой зажат в
    ±``LOCAL_MAX_TURN_DEG`` от наклона общей прямой.

    Args:
        v: Координата поперёк строк.
        u: Координата вдоль строк (концы).
        inward: +1, если внутрь блока — большие ``u`` (левая сторона), −1 — правая.
        start: Общая кривая в ``v`` — с неё начинается отсев.
        tol: Допуск «на кривой».

    Returns:
        Значения кривой в ``v``.
    """
    turn = math.tan(math.radians(LOCAL_MAX_TURN_DEG))
    base_slope = _theil_sen(v, u)
    fitted = start.astype(np.float64)
    for _ in range(LOCAL_FIT_ROUNDS):
        keep = np.nonzero(inward * (u - fitted) <= tol)[0]
        if keep.size < 3:
            return fitted
        order = keep[np.argsort(v[keep])]
        kept_v = v[order]
        new = np.empty(len(v))
        for index in range(len(v)):
            # Место конца среди оставшихся и окно вокруг него.
            rank = int(np.searchsorted(kept_v, v[index]))
            window = order[max(0, rank - LOCAL_FIT_ROWS) : rank + LOCAL_FIT_ROWS + 1]
            if window.size < 3:
                window = order
            slope = float(np.clip(_theil_sen(v[window], u[window]), base_slope - turn, base_slope + turn))
            intercept = float(np.median(u[window] - slope * v[window]))
            new[index] = slope * v[index] + intercept
        fitted = new
    return fitted


def _resid_robust(block: TextBlock, side: SideKind, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Отклонения от устойчивой кривой по самим концам строк в системе координат блока.

    Концы поворачиваются на медианный наклон строк: ``u`` — вдоль строк, ``v`` — поперёк. Кривая
    ``u(v)`` — :func:`_fit_robust` или местная :func:`_fit_local`, если та вписывает больше концов; «внутрь — плюс»: у левой стороны внутрь — большие ``u``.

    Returns:
        ``(resid, curve)``: отклонения (пиксели) и кривая в кадре для оверлея.
    """
    theta = block_frame(list(block.rows))
    cos, sin = math.cos(theta), math.sin(theta)
    centre = points.mean(axis=0)
    shifted = points - centre
    u = shifted[:, 0] * cos + shifted[:, 1] * sin
    v = -shifted[:, 0] * sin + shifted[:, 1] * cos
    tol = mm_to_px(ALIGN_TOL_MM, block.dpi)
    coef, fitted = _fit_robust(v, u, tol)
    grid = np.linspace(v.min(), v.max(), 50)
    gu = np.polyval(coef, grid)
    # Местная кривая — если край меняет наклон по высоте и общая кривая его не описывает.
    if len(v) >= 2 * LOCAL_FIT_ROWS:
        local = _fit_local(v, u, 1.0 if side is SideKind.LEFT else -1.0, fitted, tol)
        if int((np.abs(u - local) <= tol).sum()) >= int((np.abs(u - fitted) <= tol).sum()) + PARABOLA_GAIN_ROWS:
            fitted = local
            order = np.argsort(v)
            gu = np.interp(grid, v[order], local[order])
    resid = (u - fitted) if side is SideKind.LEFT else (fitted - u)
    # Кривая для оверлея: та же u(v) на сетке, повёрнутая обратно в кадр.
    curve = np.column_stack([gu * cos - grid * sin, gu * sin + grid * cos]) + centre
    return resid, curve


def side_alignment(block: TextBlock, side: SideKind, method: AlignMethod) -> SideAlignment | None:
    """Выравнивание по вертикальной стороне: отклонения концов строк, их положение и серии.

    Args:
        block: Текстовый блок.
        side: ``LEFT`` или ``RIGHT``.
        method: Способ меры отклонения.

    Returns:
        Выравнивание стороны; ``None``, если в блоке меньше двух рядов с осями.
    """
    ends = [(index, _row_end(row, side, block.dpi)) for index, row in enumerate(block.rows)]
    ends = [(index, item) for index, item in ends if item is not None]
    if len(ends) < 2:
        return None
    rows = [index for index, _ in ends]
    points = np.array([item[0] for _, item in ends])
    tangents = np.array([item[1] for _, item in ends])
    curve = None
    if method is AlignMethod.TREND:
        resid = _resid_trend(block, side, points)
        curve = _core_curve(block.envelope, side)
    elif method is AlignMethod.TANGENT:
        resid = _resid_tangent(block, side, points, tangents)
        curve = _core_curve(block.envelope, side)
    else:
        resid, curve = _resid_robust(block, side, points)
    resid_mm = np.array([px_to_mm(float(value), block.dpi) for value in resid])
    statuses = _statuses(resid_mm)
    runs = aligned_runs(statuses)
    considered = [item is not RowStatus.INDENT for item in statuses]
    on = [item is RowStatus.ON for item in statuses]
    on_share = float(np.sum(np.array(on) & np.array(considered)) / max(1, sum(considered)))
    return SideAlignment(
        method=method,
        side=side,
        ends=tuple(
            RowEnd(row, point, float(value), status, bool(flag))
            for row, point, value, status, flag in zip(rows, points, resid_mm, statuses, runs)
        ),
        on_share=on_share,
        # Доля выровненных — среди рядов БЕЗ отступов, как у вердикта ``alignment.py``: концы
        # абзацев и подпись автора в знаменателе держали долю ниже порога даже у набора по формату.
        aligned_share=float(np.sum(runs & np.array(considered)) / max(1, sum(considered))),
        curve=curve,
    )


def aligned_segments(sides: BlockSides, alignment: SideAlignment, block: TextBlock) -> np.ndarray:
    """Какие звенья вертикальной стороны лежат против выровненных рядов.

    Звено стороны относится к ряду, ближайшему по высоте в системе координат блока (поперёк строк);
    угловые звенья берут метку ближайшего крайнего ряда.

    Args:
        sides: Разметка контура.
        alignment: Выравнивание этой стороны.
        block: Текстовый блок.

    Returns:
        Флаги ``(N,)`` по звеньям контура: ``True`` — звено стороны против выровненного ряда.
    """
    _, middles, _ = outward_normals(sides.polygon)
    theta = block_frame(list(block.rows))
    across = np.array([-math.sin(theta), math.cos(theta)])
    own = np.array([label is alignment.side for label in sides.labels])
    levels = np.array([float(end.point @ across) for end in alignment.ends])
    flags = np.array([end.aligned for end in alignment.ends])
    out = np.zeros(len(middles), dtype=bool)
    if not levels.size:
        return out
    nearest = np.argmin(np.abs((middles @ across)[:, None] - levels[None, :]), axis=1)
    out[own] = flags[nearest[own]]
    return out


def fill_side_gaps(
    v: np.ndarray, u: np.ndarray, aligned: np.ndarray, step: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Дополнительная линия стороны: невыровненные концы выкинуть, невыровненную середину заменить заплаткой.

    Заплатка — PCHIP (кусочно-кубическая кривая Эрмита) через ВСЕ выровненные точки стороны: касательная
    на стыке с настоящей стороной не ломается, изгиб выгнутой стороны не занижается, как у хорды, и
    кривая не выходит за соседние значения (у сплайна на коротком разрыве бывали бы петли).

    Args:
        v: Координата поперёк строк (вниз по блоку) для точек стороны ``(N,)``.
        u: Координата вдоль строк ``(N,)``.
        aligned: Флаги ``(N,)``: точка стороны лежит против выровненного ряда.
        step: Шаг узлов заплатки по ``v`` (пиксели): длинный разрыв заполняется кривой, а не парой точек.

    Returns:
        ``(v, u, filled)`` по возрастанию ``v`` — точки линии и флаги «заплатка»; ``None``, если
        выровненных точек на разных высотах меньше двух.
    """
    order = np.argsort(v, kind="stable")
    v, u, aligned = v[order], u[order], np.asarray(aligned, dtype=bool)[order]
    green = np.flatnonzero(aligned)
    if green.size < 2:
        return None
    # Опорные точки интерполяции: выровненные, одинаковые высоты слиты медианой (PCHIP требует
    # строго возрастающей абсциссы).
    levels, inverse = np.unique(v[green], return_inverse=True)
    if levels.size < 2:
        return None
    values = np.array([float(np.median(u[green][inverse == index])) for index in range(levels.size)])
    curve = PchipInterpolator(levels, values)
    out_v: list[np.ndarray] = []
    out_u: list[np.ndarray] = []
    out_filled: list[np.ndarray] = []
    # Проход от первой выровненной точки до последней: концы за ними отброшены.
    index, last = int(green[0]), int(green[-1])
    while index <= last:
        if aligned[index]:
            out_v.append(v[index : index + 1])
            out_u.append(u[index : index + 1])
            out_filled.append(np.zeros(1, dtype=bool))
            index += 1
            continue
        # Невыровненная серия в середине: от предыдущей выровненной точки до следующей.
        stop = index
        while not aligned[stop]:
            stop += 1
        v_from, v_to = v[index - 1], v[stop]
        grid = np.unique(np.concatenate([np.arange(v_from + step, v_to, step), v[index:stop]]))
        grid = grid[(grid > v_from) & (grid < v_to)]
        out_v.append(grid)
        out_u.append(curve(grid))
        out_filled.append(np.ones(grid.size, dtype=bool))
        index = stop
    return np.concatenate(out_v), np.concatenate(out_u), np.concatenate(out_filled)


def _aligned_spans(alignment: SideAlignment, theta: float) -> list[tuple[float, float]]:
    """Высоты выровненных серий: от конца первой строки серии до конца последней, поперёк строк.

    Args:
        alignment: Выравнивание стороны; концы строк идут сверху вниз.
        theta: Медианный наклон строк блока (:func:`block_frame`), радианы.

    Returns:
        Пары ``(v_от, v_до)`` в системе блока (``v`` — поперёк строк), по одной на серию.
    """
    across = np.array([-math.sin(theta), math.cos(theta)])
    out = []
    start = None
    for index, end in enumerate(alignment.ends):
        if end.aligned and start is None:
            start = index
        last = index == len(alignment.ends) - 1
        if start is not None and (not end.aligned or last):
            stop = index if end.aligned else index - 1
            out.append((float(alignment.ends[start].point @ across), float(alignment.ends[stop].point @ across)))
            start = None
    return out


def filled_side(
    sides: BlockSides, alignment: SideAlignment, block: TextBlock, unreliable: tuple[tuple[float, float], ...] = ()
) -> FilledSide | None:
    """Дополнительная линия вертикальной стороны и её меры против мер всей стороны.

    Берутся звенья стороны без углов и неуверенных концов (как у :func:`side_measures`). Середины звеньев
    переводятся в систему блока (``u`` вдоль строк, ``v`` поперёк, поворот на медианный наклон строк
    :func:`block_frame`), там строится :func:`fill_side_gaps`, и линия поворачивается обратно в кадр.

    Выровненной считается точка, лежащая по ``v`` между концами первой и последней строки ОДНОЙ
    выровненной серии (:func:`_aligned_spans`), а не точка против ближайшего выровненного ряда, как у
    раскраски :func:`aligned_segments`. Иначе в линию попадала половина перехода от невыровненной
    строки к выровненной: у пятистрочного блока 1970/02 с.90 со ступенькой абзацного отступа наклон
    левой стороны выходил +4.4° вместо −0.15°.

    Args:
        sides: Разметка контура.
        alignment: Выравнивание этой стороны (метод задаёт, какие участки выровнены).
        block: Текстовый блок.
        unreliable: Недостоверные участки этой стороны — отрезки по y кадра (``unreliable_left/right``
            огибающей: ступеньки выноса за колонку, выступы сора ``edge_guard``). Точки стороны в них
            считаются невыровненными, и в середине стороны их заменяет заплатка PCHIP между нормальными
            участками; пусто — как раньше, только по выровненным сериям.

    Returns:
        :class:`FilledSide` или ``None``, если у стороны меньше трёх звеньев или выровненного на ней
        меньше двух точек.
    """
    _, middles, _ = outward_normals(sides.polygon)
    own = np.array([label is alignment.side for label in sides.labels]) & ~sides.excluded
    if own.sum() < 3:
        return None
    points = middles[own]
    theta = block_frame(list(block.rows))
    cos, sin = math.cos(theta), math.sin(theta)
    u = points[:, 0] * cos + points[:, 1] * sin
    v = -points[:, 0] * sin + points[:, 1] * cos
    flags = np.zeros(len(v), dtype=bool)
    for v_from, v_to in _aligned_spans(alignment, theta):
        flags |= (v >= v_from) & (v <= v_to)
    # Недостоверные участки (в y кадра) из выровненного исключаются: там сторону ведёт сор или вынос,
    # а не край текста, и линия идёт заплаткой между нормальными участками.
    for y_from, y_to in unreliable:
        flags &= ~((points[:, 1] >= y_from) & (points[:, 1] <= y_to))
    result = fill_side_gaps(v, u, flags, mm_to_px(DENSIFY_MM, block.dpi))
    if result is None:
        return None
    line_v, line_u, filled = result
    line = np.column_stack([line_u * cos - line_v * sin, line_u * sin + line_v * cos])
    tilt, bend = _tilt_bend(line, True, block.dpi)
    raw_tilt, raw_bend = _tilt_bend(points, True, block.dpi)
    length = px_to_mm(float(line_v[-1] - line_v[0]), block.dpi)
    return FilledSide(alignment.side, alignment.method, line, filled, length, tilt, bend, raw_tilt, raw_bend)


__all__ = [
    "AlignMethod",
    "BlockSides",
    "EdgeAxis",
    "FilledSide",
    "CYCLE",
    "DEFAULT_SIDES_METHOD",
    "RowEnd",
    "RowStatus",
    "SIDES_METHODS",
    "SideAlignment",
    "SideKind",
    "SideMeasure",
    "SidesMethod",
    "aligned_runs",
    "aligned_segments",
    "densify",
    "edge_axis",
    "fill_side_gaps",
    "filled_side",
    "label_agreement",
    "outward_normals",
    "ray_hit",
    "side_alignment",
    "side_measures",
    "sides_construct",
    "sides_frame",
    "sides_of",
    "sides_rays",
    "signed_area",
]
