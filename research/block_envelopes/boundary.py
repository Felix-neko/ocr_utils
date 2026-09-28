"""Граница блока для стенда: переключаемые способы — боевая огибающая, ступенчатое объединение полос строк, выровненные стороны прямой, вогнутая оболочка."""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np
import shapely
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.signal import savgol_filter

from ocr_utils.page_layout import mm_to_px
from ocr_utils.page_layout.text_blocks import blocks as legacy
from ocr_utils.page_layout.text_blocks.blocks import (
    FALLBACK_ASPECT,
    FALLBACK_HEIGHT,
    BlockEnvelope,
    CapKind,
    Row,
    TextBlock,
    envelope_of,
    pitch_of,
)
from research.block_envelopes.capture import BlockInput
from research.block_envelopes.grouping import Group


class Boundary(str, Enum):
    """Как строится граница блока по его рядам."""

    LEGACY = "legacy"  # боевая ``envelope_of`` (полоса вокруг оси, тренд сторон, хвост последней строки)
    BANDS = "bands"  # объединение полос строк по краске, промежутки залиты там, где соседи перекрыты
    FIT = "fit"  # выровненные стороны — по краю выровненных соседей, остальное — ступенями по строкам
    ALPHA = "alpha"  # вогнутая оболочка точек полос строк (как у eynollah / pero)
    FIT_INK = "fit_ink"  # как FIT, но полоса строки — по сглаженному профилю краски, а не ось ± отступ


# Выровненная сторона: столько соседей (и не меньше этой доли) стоят на одной вертикали с таким допуском.
ALIGN_MIN_ROWS = 3
ALIGN_SHARE = 0.5
ALIGN_TOL_MM = 1.2
# Окно соседей, по которым решается, выровнена ли сторона на высоте ряда (рядов в каждую сторону).
ALIGN_WINDOW = 5
# Поле вокруг полосы строки, мм: по x и по y.
BAND_MARGIN_MM = 0.3
# Сглаживание оси строки перед крышками: медиана и Савицкий—Голей на такой длине, мм (зубья на жирном
# заголовке короче, изгиб строки — длиннее).
AXIS_SMOOTH_MM = 6.0
# Окно закрытия промежутков между буквами в профиле краски (полоса по краске), мм.
INK_WINDOW_MM = 3.0
# Наибольший наклон продолжения оси за её концом (добивка строки до стороны), градусы.
EXTEND_MAX_DEG = 3.0
# Упрощение контура, мм.
APPROX_MM = 0.3
# Вогнутая оболочка: доля от длины самого длинного ребра выпуклой оболочки (shapely ``concave_hull``).
ALPHA_RATIO = 0.08


def _glyph(group: list[Row]) -> tuple[float, float]:
    """Средний размер символа группы, как у боевого ``blocks_of``."""
    widths = [row.glyph_w for row in group if row.glyph_w > 0]
    heights = [row.glyph_h for row in group if row.glyph_h > 0]
    if widths:
        return float(np.median(widths)), float(np.median(heights))
    own = float(np.median([row.height for row in group]))
    return own * FALLBACK_ASPECT, own * FALLBACK_HEIGHT


def _axis_points(row: Row) -> np.ndarray:
    """Ось ряда одной ломаной слева направо (все оси ряда вместе)."""
    points = np.vstack([np.asarray(axis.points, dtype=np.float64) for axis in row.axes])
    return points[np.argsort(points[:, 0])]


def _smooth_axis(points: np.ndarray, dpi: float) -> np.ndarray:
    """Ось без зубьев: медиана и сглаживание по окну ``AXIS_SMOOTH_MM`` вдоль x (по сетке в пиксель).

    Args:
        points: Ломаная оси ``(N, 2)``.
        dpi: Разрешение рабочей копии.

    Returns:
        Ломаная ``(M, 2)`` на целой сетке x.
    """
    xs = np.arange(np.floor(points[0, 0]), np.ceil(points[-1, 0]) + 1)
    ys = np.interp(xs, points[:, 0], points[:, 1])
    window = int(mm_to_px(AXIS_SMOOTH_MM, dpi)) | 1
    if xs.size > window:
        ys = median_filter(ys, size=window, mode="nearest")
        ys = savgol_filter(ys, window, 2, mode="interp")
    return np.column_stack([xs, ys])


def _offsets(row: Row) -> tuple[float, float]:
    """Отступы полосы от оси вверх и вниз: постоянные на ряд, квантиль профиля краски (как у боевой полосы).

    Постоянный отступ, а не поточечный профиль, — поэтому у жирного заголовка нет «пилы».
    """
    return legacy._body_offset(row, -1.0), legacy._body_offset(row, +1.0)


def _curve_at(axis: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Ордината сглаженной оси в точках ``xs``; за концами — продолжение прямой по крайней трети оси.

    Наклон продолжения — МНК-прямая по трети оси у этого конца, не круче ``EXTEND_MAX_DEG``: наклон
    по двум крайним точкам ловит крючок конца строки (дефис, выносной элемент), и добивка короткой
    последней строки до стороны уходила вниз клином на 5–6 мм (1966/02 IMG_0098_2R).
    """
    ys = np.interp(xs, axis[:, 0], axis[:, 1])
    limit = np.tan(np.radians(EXTEND_MAX_DEG))
    third = max(2, len(axis) // 3)
    for part, mask, anchor in ((axis[:third], xs < axis[0, 0], axis[0]), (axis[-third:], xs > axis[-1, 0], axis[-1])):
        if not mask.any():
            continue
        slope = 0.0
        if len(part) >= 2 and part[-1, 0] > part[0, 0]:
            slope = float(np.clip(np.polyfit(part[:, 0], part[:, 1], 1)[0], -limit, limit))
        ys[mask] = anchor[1] + slope * (xs[mask] - anchor[0])
    return ys


def _local_side(ys: np.ndarray, xs: np.ndarray, outward: float, dpi: float) -> np.ndarray:
    """Выровненная сторона на высоте каждого ряда — по соседям в окне, а не одной прямой на весь блок.

    Для ряда ``i`` берутся края соседей в окне ``±ALIGN_WINDOW`` рядов (без него самого). Сторона
    выровнена здесь, если не меньше ``ALIGN_SHARE`` соседей (и не меньше ``ALIGN_MIN_ROWS``) стоят в
    пределах ``ALIGN_TOL_MM`` от их медианы по краю; тогда край стороны — наружный из них (края
    выровненных соседей), и изгиб страницы у корешка сторона повторяет. Иначе — ``nan``.

    Args:
        ys: Ординаты рядов сверху вниз.
        xs: Края рядов с этой стороны.
        outward: ``-1`` — левая сторона (наружу — меньше x), ``+1`` — правая.
        dpi: Разрешение рабочей копии.

    Returns:
        Край стороны на высоте каждого ряда или ``nan``, где сторона не выровнена.
    """
    tol = mm_to_px(ALIGN_TOL_MM, dpi)
    out = np.full(len(xs), np.nan)
    for index in range(len(xs)):
        lo, hi = max(0, index - ALIGN_WINDOW), min(len(xs), index + ALIGN_WINDOW + 1)
        neighbours = np.delete(xs[lo:hi], index - lo)
        if len(neighbours) < ALIGN_MIN_ROWS:
            continue
        # Мода края: медиана наружной половины — короткие строки (внутрь) её не тянут.
        order = np.sort(neighbours * outward)[::-1] * outward
        edge = float(np.median(order[: max(ALIGN_MIN_ROWS, len(order) // 2)]))
        aligned = neighbours[np.abs(neighbours - edge) <= tol]
        if len(aligned) >= max(ALIGN_MIN_ROWS, ALIGN_SHARE * len(neighbours)):
            out[index] = float(aligned.min() if outward < 0 else aligned.max())
    return out


def _row_extents(rows: list[Row], dpi: float, fit_sides: bool) -> list[tuple[float, float]]:
    """Края полосы каждого ряда по x: по краске, а у выровненной стороны — до неё.

    Ряд, не дотянувший до выровненной стороны (абзацный отступ, короткая строка конца абзаца),
    добивается до неё. Ряд, вылезший за неё (висячая пунктуация, вынос), сохраняет свой край: в
    границу попадает вся его краска, выступ остаётся местным — только на высоте этого ряда.

    Args:
        rows: Ряды блока сверху вниз.
        dpi: Разрешение рабочей копии.
        fit_sides: Искать ли выровненные стороны (``False`` — все края по краске, чистые ступени).

    Returns:
        ``(левый, правый)`` край для каждого ряда.
    """
    lefts = np.array([row.x0 for row in rows], dtype=np.float64)
    rights = np.array([row.x1 for row in rows], dtype=np.float64)
    if fit_sides and len(rows) > ALIGN_MIN_ROWS:
        ys = np.array([row.y for row in rows], dtype=np.float64)
        left_side = _local_side(ys, lefts, -1.0, dpi)
        right_side = _local_side(ys, rights, +1.0, dpi)
        # Ограничитель вылета: край берётся у соседей, дальше крайнего края рядов блока он не уйдёт.
        lefts = np.where(np.isnan(left_side), lefts, np.minimum(lefts, left_side))
        rights = np.where(np.isnan(right_side), rights, np.maximum(rights, right_side))
    return list(zip(lefts.tolist(), rights.tolist()))


def _band(row: Row, x0: float, x1: float, dpi: float, by_ink: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Верхняя и нижняя кривые полосы ряда между ``x0`` и ``x1`` (с полем ``BAND_MARGIN_MM``).

    По оси (``by_ink=False``): сглаженная ось ± постоянный отступ — полоса не повторяет выносные
    элементы. По краске (``by_ink=True``): профиль краски ряда (верх и низ букв по столбцам), в
    котором промежутки между буквами закрыты скользящим экстремумом на ширину окна
    ``INK_WINDOW_MM``, и сглаженный; там, где профиля нет (за краями краски, добивка до стороны), —
    продолжение по оси с тем же отступом, что у ближайшего края профиля.

    Args:
        row: Ряд.
        x0, x1: Края полосы.
        dpi: Разрешение рабочей копии.
        by_ink: Строить ли по профилю краски.

    Returns:
        Пара кривых ``(верх, низ)`` ``(N, 2)``.
    """
    axis = _smooth_axis(_axis_points(row), dpi)
    xs = np.linspace(x0, x1, max(2, int(x1 - x0) // 2 + 2))
    ys = _curve_at(axis, xs)
    up, down = _offsets(row)
    margin = mm_to_px(BAND_MARGIN_MM, dpi)
    top, bottom = ys - up, ys + down
    if by_ink and row.top_edge is not None and row.bottom_edge is not None and len(row.top_edge) > 3:
        top = _ink_curve(row.top_edge, xs, ys, -1.0, dpi, top)
        bottom = _ink_curve(row.bottom_edge, xs, ys, +1.0, dpi, bottom)
    return np.column_stack([xs, top - margin]), np.column_stack([xs, bottom + margin])


def _ink_curve(
    profile: np.ndarray, xs: np.ndarray, axis_ys: np.ndarray, direction: float, dpi: float, fallback: np.ndarray
) -> np.ndarray:
    """Кривая края краски ряда в точках ``xs``: профиль, закрытый по промежуткам между буквами и сглаженный.

    Args:
        profile: Профиль краски ``(N, 2)`` — верх (``direction=-1``) или низ (``+1``) букв по столбцам.
        xs: Точки, в которых нужна кривая.
        axis_ys: Ордината оси ряда в тех же точках.
        direction: ``-1`` — верх, ``+1`` — низ.
        dpi: Разрешение рабочей копии.
        fallback: Кривая по оси — берётся там, где профиля нет.

    Returns:
        Ординаты кривой в точках ``xs``.
    """
    order = np.argsort(profile[:, 0])
    px, py = profile[order, 0], profile[order, 1]
    grid = np.arange(np.floor(px[0]), np.ceil(px[-1]) + 1)
    # Отступ профиля от оси: верх — вверх (минус), низ — вниз (плюс); столбцы без краски — нет данных.
    offset = direction * (np.interp(grid, px, py) - np.interp(grid, xs, axis_ys))
    window = max(3, int(mm_to_px(INK_WINDOW_MM, dpi)) | 1)
    # Скользящий максимум отступа закрывает промежутки между буквами; медиана и сглаживание — убирают зубья.
    offset = maximum_filter1d(offset, window, mode="nearest")
    offset = median_filter(offset, size=window, mode="nearest")
    if offset.size > window:
        offset = savgol_filter(offset, window, 2, mode="interp")
    offset = np.maximum(offset, 0.0)
    inside = (xs >= grid[0]) & (xs <= grid[-1])
    out = fallback.copy()
    out[inside] = np.interp(xs[inside], grid, np.interp(grid, xs, axis_ys) + direction * offset)
    # Снаружи профиля — ось с отступом ближайшего края профиля (без скачка на стыке).
    out[xs < grid[0]] = np.interp(xs[xs < grid[0]], xs, axis_ys) + direction * offset[0]
    out[xs > grid[-1]] = np.interp(xs[xs > grid[-1]], xs, axis_ys) + direction * offset[-1]
    return out


def stepped_polygon(rows: list[Row], dpi: float, fit_sides: bool, by_ink: bool = False) -> np.ndarray:
    """Контур блока: полосы рядов и заливка промежутков между соседями по их общей ширине.

    Полоса ряда — сглаженная ось ± постоянный отступ, от края до края (:func:`_row_extents`).
    Промежуток между соседними рядами заливается четырёхугольником по их ОБЩЕЙ ширине: где соседи
    не перекрыты, остаётся ступень, а не клин наискосок. Контур — внешний контур растра, упрощённый
    на ``APPROX_MM``.

    Args:
        rows: Ряды блока сверху вниз.
        dpi: Разрешение рабочей копии.
        fit_sides: Добивать ли ряды до выровненных сторон.
        by_ink: Полоса ряда по профилю краски (:func:`_band`), а не ось ± отступ.

    Returns:
        Контур ``(N, 2)`` в пикселях рабочей копии.
    """
    extents = _row_extents(rows, dpi, fit_sides)
    bands = [_band(row, x0, x1, dpi, by_ink) for row, (x0, x1) in zip(rows, extents)]
    shapes = [np.vstack([top, bottom[::-1]]) for top, bottom in bands]
    for index in range(len(rows) - 1):
        (a0, a1), (b0, b1) = extents[index], extents[index + 1]
        lo, hi = max(a0, b0), min(a1, b1)
        if hi <= lo:
            continue
        upper_bottom, lower_top = bands[index][1], bands[index + 1][0]
        xs = np.linspace(lo, hi, max(2, int(hi - lo) // 2 + 2))
        top = np.column_stack([xs, np.interp(xs, upper_bottom[:, 0], upper_bottom[:, 1])])
        bottom = np.column_stack([xs, np.interp(xs, lower_top[:, 0], lower_top[:, 1])])
        # Промежуток заливается, только если нижний ряд и правда ниже (на крутом изгибе полосы перекрыты).
        shapes.append(np.vstack([top, bottom[::-1]]))
    return _outline(shapes, dpi)


def _outline(shapes: list[np.ndarray], dpi: float) -> np.ndarray:
    """Внешний контур объединения многоугольников (через растр), упрощённый на ``APPROX_MM``."""
    points = np.vstack(shapes)
    origin = np.floor(points.min(axis=0)) - 3
    size = tuple((np.ceil(points.max(axis=0) - origin) + 4).astype(int)[::-1])
    mask = np.zeros(size, dtype=np.uint8)
    for shape in shapes:
        cv2.fillPoly(mask, [np.round(shape - origin).astype(np.int32)], 255)
    # Пиксельные щели между полосой и заливкой промежутка закрываются маленьким ядром.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return points
    if len(contours) > 1:
        # Части, не связанные заливкой (соседи не перекрыты по x), соединяются перемычкой по центрам:
        # блок — один контур. Выпуклая оболочка перемычки маленькая, клина она не даёт.
        mask = _bridge(mask, contours)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    simple = cv2.approxPolyDP(contour, mm_to_px(APPROX_MM, dpi), True)
    if simple.shape[0] >= 3:
        contour = simple
    return contour.reshape(-1, 2).astype(np.float64) + origin


def _bridge(mask: np.ndarray, contours) -> np.ndarray:
    """Соединить несвязные части растра перемычками по ближайшим точкам (дерево по порядку сверху вниз)."""
    ordered = sorted(contours, key=lambda item: item[:, 0, 1].min())
    for first, second in zip(ordered[:-1], ordered[1:]):
        a = first.reshape(-1, 2)
        b = second.reshape(-1, 2)
        # Ближайшая пара точек (прореженно: контуры длинные).
        a_s, b_s = a[:: max(1, len(a) // 200)], b[:: max(1, len(b) // 200)]
        distance = ((a_s[:, None, :] - b_s[None, :, :]) ** 2).sum(axis=2)
        i, j = np.unravel_index(np.argmin(distance), distance.shape)
        cv2.line(mask, tuple(int(v) for v in a_s[i]), tuple(int(v) for v in b_s[j]), 255, 5)
    return mask


def alpha_polygon(rows: list[Row], dpi: float) -> np.ndarray:
    """Вогнутая оболочка точек полос рядов (shapely ``concave_hull``), полосы — по краске каждого ряда.

    Args:
        rows: Ряды блока.
        dpi: Разрешение рабочей копии.

    Returns:
        Контур ``(N, 2)``.
    """
    points = []
    for row in rows:
        top, bottom = _band(row, row.x0, row.x1, dpi)
        points.extend([top, bottom])
    cloud = shapely.MultiPoint(np.vstack(points))
    hull = shapely.concave_hull(cloud, ratio=ALPHA_RATIO)
    if hull.geom_type != "Polygon":
        hull = cloud.convex_hull
    if hull.geom_type != "Polygon":
        return np.vstack(points)
    return np.asarray(hull.exterior.coords, dtype=np.float64)[:-1]


def _envelope(polygon: np.ndarray, rows: list[Row]) -> BlockEnvelope:
    """Огибающая для ``TextBlock`` по готовому контуру: кромки — края контура на ординатах рядов."""
    ys = np.array([row.y for row in rows], dtype=np.float64)
    left = np.column_stack([np.full(ys.shape, polygon[:, 0].min()), ys])
    right = np.column_stack([np.full(ys.shape, polygon[:, 0].max()), ys])
    top = polygon[polygon[:, 1] <= np.quantile(polygon[:, 1], 0.1)]
    bottom = polygon[polygon[:, 1] >= np.quantile(polygon[:, 1], 0.9)]
    return BlockEnvelope(left=left, right=right, top=top, bottom=bottom, polygon=polygon, smooth_pitches=0.0)


def block_of(group: Group, kind: Boundary, inp: BlockInput) -> TextBlock:
    """Текстовый блок из группы рядов с границей выбранного способа.

    Args:
        group: Группа рядов (блок).
        kind: Способ границы.
        inp: Вход блоковой стадии (разрешение, краска, параметры боевой огибающей).

    Returns:
        ``TextBlock``; крупная огибающая у всех способов — боевая (для выключки и оверлея не нужна).
    """
    rows = sorted(group.rows, key=lambda row: row.y)
    pitch = pitch_of(rows)
    glyph = _glyph(rows)
    if kind is Boundary.LEGACY:
        rows = legacy._with_tail(rows, pitch, glyph[0], inp.dpi, inp.ink)
        envelope = envelope_of(rows, pitch, inp.dpi, inp.smooth_pitches, glyph, inp.dilate, CapKind.BODY)
    else:
        if kind is Boundary.ALPHA:
            polygon = alpha_polygon(rows, inp.dpi)
        else:
            polygon = stepped_polygon(
                rows, inp.dpi, fit_sides=kind in (Boundary.FIT, Boundary.FIT_INK), by_ink=kind is Boundary.FIT_INK
            )
        envelope = _envelope(polygon, rows)
    return TextBlock(
        column=group.column,
        index=group.index,
        span=group.span,
        rows=tuple(rows),
        pitch_px=pitch,
        dpi=inp.dpi,
        envelope=envelope,
        envelope_coarse=envelope,
    )


__all__ = ["Boundary", "alpha_polygon", "block_of", "stepped_polygon"]
