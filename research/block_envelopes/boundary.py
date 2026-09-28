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
from research.block_envelopes.axes_fix import body_reference
from research.block_envelopes.capture import BlockInput
from ocr_utils.page_layout.text_blocks.smooth_envelope import (  # noqa: F401 — стенд и тесты берут их отсюда
    APPROX_MM,
    BAND_MARGIN_MM,
    SmoothEnvelope,
    _axis_points,
    _axis_span,
    _bridge,
    _curve_at,
    _glyph,
    _guided_at,
    _guides,
    _has_sides,
    _local_side,
    _offsets,
    _outline,
    _row_extents,
    _smooth_axis,
    _within_rows,
    smooth_envelope,
)
from ocr_utils.page_layout.text_blocks.smooth_envelope import _band as _band_axis
from research.block_envelopes.grouping import Group


class Boundary(str, Enum):
    """Как строится граница блока по его рядам."""

    LEGACY = "legacy"  # боевая ``envelope_of`` (полоса вокруг оси, тренд сторон, хвост последней строки)
    BANDS = "bands"  # объединение полос строк по краске, промежутки залиты там, где соседи перекрыты
    FIT = "fit"  # выровненные стороны — по краю выровненных соседей, остальное — ступенями по строкам
    ALPHA = "alpha"  # вогнутая оболочка точек полос строк (как у eynollah / pero)
    FIT_INK = "fit_ink"  # как FIT, но полоса строки — по сглаженному профилю краски, а не ось ± отступ
    SMOOTH = "smooth"  # стороны — гладкие кривые x(y) по краям строк (``smooth_sides``), верх и низ — полосы строк


# Окно закрытия промежутков между буквами в профиле краски (полоса по краске), мм.
INK_WINDOW_MM = 3.0
# Вогнутая оболочка: доля от длины самого длинного ребра выпуклой оболочки (shapely ``concave_hull``).
ALPHA_RATIO = 0.08


def _band(
    row: Row, x0: float, x1: float, dpi: float, by_ink: bool = False, guide: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Полоса ряда: по оси — боевая ``smooth_envelope._band``; по краске (``by_ink``, способ ``FIT_INK``) — профиль краски.

    Args:
        row: Ряд.
        x0, x1: Края полосы.
        dpi: Разрешение рабочей копии.
        by_ink: Строить ли по профилю краски (промежутки между буквами закрыты, :func:`_ink_curve`).
        guide: Ось-проводник за концами своей оси (``smooth_envelope._guided_at``) или ``None``.

    Returns:
        Пара кривых ``(верх, низ)``.
    """
    top, bottom = _band_axis(row, x0, x1, dpi, guide=guide)
    if not by_ink or row.top_edge is None or row.bottom_edge is None or len(row.top_edge) <= 3:
        return top, bottom
    axis = _smooth_axis(_axis_points(row), dpi)
    xs = top[:, 0]
    ys = _guided_at(axis, xs, guide) if guide is not None else _curve_at(axis, xs)
    up, down = _offsets(row)
    margin = mm_to_px(BAND_MARGIN_MM, dpi)
    upper = _ink_curve(row.top_edge, xs, ys, -1.0, dpi, ys - up)
    lower = _ink_curve(row.bottom_edge, xs, ys, +1.0, dpi, ys + down)
    return np.column_stack([xs, upper - margin]), np.column_stack([xs, lower + margin])


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


def stepped_polygon(
    rows: list[Row], dpi: float, fit_sides: bool, by_ink: bool = False, reference_deg: float | None = None
) -> np.ndarray:
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
        reference_deg: Наклон корпуса полосы для добивки строк без длинного соседа (:func:`_guides`).

    Returns:
        Контур ``(N, 2)`` в пикселях рабочей копии.
    """
    extents = _row_extents(rows, dpi, fit_sides)
    guides = _guides(rows, extents, dpi, reference_deg)
    bands = [_band(row, x0, x1, dpi, by_ink, guide) for row, (x0, x1), guide in zip(rows, extents, guides)]
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
    envelope = None
    if kind is Boundary.LEGACY:
        rows = legacy._with_tail(rows, pitch, glyph[0], inp.dpi, inp.ink)
        envelope = envelope_of(rows, pitch, inp.dpi, inp.smooth_pitches, glyph, inp.dilate, CapKind.BODY)
    elif kind is Boundary.SMOOTH and _has_sides(rows):
        envelope = smooth_envelope(rows, inp.dpi, body_reference(inp.axes)[0] if len(inp.axes) >= 3 else None)
        # Предохранитель: гладкий контур не выходит за рамку строк дальше выноса пунктуации. Иначе
        # (центрированный блок без сторон — 1966/02 IMG_0102_2R, «Редколлегия») — прежний FIT.
        if envelope is None or not _within_rows(envelope.polygon, rows, inp.dpi):
            kind = Boundary.FIT
            envelope = None
    if envelope is None and kind is not Boundary.LEGACY:
        if kind is Boundary.ALPHA:
            polygon = alpha_polygon(rows, inp.dpi)
        else:
            # SMOOTH без сторон (короткий или смешанный блок) строится как FIT.
            polygon = stepped_polygon(
                rows,
                inp.dpi,
                fit_sides=kind in (Boundary.FIT, Boundary.FIT_INK, Boundary.SMOOTH),
                by_ink=kind is Boundary.FIT_INK,
                reference_deg=body_reference(inp.axes)[0] if len(inp.axes) >= 3 else None,
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
