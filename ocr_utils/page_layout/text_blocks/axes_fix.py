"""Замена осей-выбросов крупного набора перед сборкой блоков (способ ``BlocksMode.SMOOTH``): ось, наклон которой расходится с корпусом полосы, берётся по центру краски или прямой по глифам с наклоном корпуса."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ocr_utils.page_layout import mm_to_px

# Ось крупного набора: высота строки больше стольких медиан высоты корпуса.
LARGE_HEIGHT_RATIO = 1.3
# Выброс: наклон расходится с медианой наклона корпуса больше стольких градусов...
TILT_DEG = 1.5
# ...при длине оси больше стольких мм (короче — наклон шумит сам по себе).
MIN_LENGTH_MM = 15.0
# Прямая по глифам: наклон зажимается в столько градусов от наклона корпуса.
CLAMP_DEG = 1.0
# Доля крайних по высоте центров глифов, отбрасываемая с каждой стороны (выносные Д, Й, Щ, запятые).
TRIM_SHARE = 0.2
# Высота строки по высоте глифа: у корпуса пака высота ряда ≈ 1.35 медианной высоты глифа
# (ряд 17 px при глифе 12.5 px, 1971/10 IMG_0010_1L).
ROW_PER_GLYPH = 1.35
# Шаг точек новой оси, пиксели рабочей копии.
STEP_PX = 4.0
# Ось, которая ложится на свои глифы, не заменяется, даже если её наклон расходится с корпусом:
# медиана |отклонения середин глифов от оси| меньше этой доли медианной высоты глифа, и ось по
# центру краски идёт с тем же наклоном (не дальше ``AGREE_DEG``). Так остаются заголовок, набранный
# с настоящим наклоном (1966/02 IMG_0077_1L: отклонение 0.01–0.02 высоты, замена уводила ось на
# 9.7 px), и строка корпуса на изгибе бумаги (1971/08 IMG_0067_2R, 1966/04 IMG_0011_2R «снабжения.»:
# прямая с зажатым наклоном резала строку и уходила в межстрочный просвет).
FIT_SHARE = 0.1
AGREE_DEG = 1.0
# Глифы ниже этой доли медианы высот глифов строки (точки, запятые, соринки) в мерах по глифам не
# участвуют.
GLYPH_MIN_SHARE = 0.5


@dataclass(frozen=True)
class AxisFix:
    """Запись протокола: какая ось заменена и чем.

    Attributes:
        cy: Середина оси по высоте, пиксели рабочей копии.
        x0, x1: Концы оси.
        before_deg: Наклон основной оси до замены.
        after_deg: Наклон после замены (``nan`` — ось не тронута).
        source: ``centre`` — прежняя ось по центру краски, ``glyphs`` — прямая по глифам,
            ``none`` — заменить нечем.
        height_before, height_after: Высота строки до и после.
    """

    cy: float
    x0: float
    x1: float
    before_deg: float
    after_deg: float
    source: str
    height_before: float
    height_after: float


def slope_of(points: np.ndarray) -> float:
    """Наклон ломаной в градусах — МНК-прямая по всем точкам (не хорда: крючок конца её не ведёт)."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2 or points[-1, 0] <= points[0, 0]:
        return 0.0
    return float(np.degrees(np.arctan(np.polyfit(points[:, 0], points[:, 1], 1)[0])))


def _letters(axis) -> np.ndarray | None:
    """Боксы букв строки ``(n, 4)`` без точек, запятых и соринок; ``None`` — букв меньше трёх."""
    glyphs = axis.glyphs
    if glyphs is None or len(glyphs) < 3:
        return None
    glyphs = np.asarray(glyphs, dtype=np.float64)
    heights = glyphs[:, 3] - glyphs[:, 1]
    letters = glyphs[heights >= GLYPH_MIN_SHARE * float(np.median(heights))]
    return letters if len(letters) >= 3 else None


def glyph_height(axis) -> float | None:
    """Медианная высота букв строки (px) — кегль строки без раздувания оси изгибом; ``None`` — букв мало."""
    letters = _letters(axis)
    return None if letters is None else float(np.median(letters[:, 3] - letters[:, 1]))


def glyph_fit(axis, points: np.ndarray) -> float | None:
    """Насколько ось ложится на свои буквы: медиана |середина буквы − ось| в долях медианной высоты буквы.

    Args:
        axis: Строка с глифами.
        points: Проверяемая ось ``(n, 2)``.

    Returns:
        Доля (0 — ось точно по серединам букв) или ``None`` — букв меньше трёх.
    """
    letters = _letters(axis)
    if letters is None:
        return None
    points = np.asarray(points, dtype=np.float64)
    cx = (letters[:, 0] + letters[:, 2]) / 2.0
    cy = (letters[:, 1] + letters[:, 3]) / 2.0
    dev = np.abs(cy - np.interp(cx, points[:, 0], points[:, 1]))
    return float(np.median(dev) / max(float(np.median(letters[:, 3] - letters[:, 1])), 1e-6))


def _keeps_own_letters(axis) -> bool:
    """Ложится ли ось на свои буквы и согласна ли по наклону с осью по центру краски (см. ``FIT_SHARE``)."""
    fit = glyph_fit(axis, axis.points)
    if fit is None or fit >= FIT_SHARE:
        return False
    centre = axis.centre_points
    return centre is None or abs(slope_of(centre) - slope_of(axis.points)) < AGREE_DEG


def body_reference(axes: list) -> tuple[float, float]:
    """Наклон и высота корпуса полосы: медианы по осям не выше медианной высоты строки.

    Args:
        axes: Оси строк (``LineAxis``).

    Returns:
        ``(наклон, °; высота, px)``.
    """
    heights = np.array([axis.height for axis in axes], dtype=np.float64)
    median_height = float(np.median(heights))
    body = [axis for axis in axes if axis.height <= median_height]
    return float(np.median([slope_of(axis.points) for axis in body])), median_height


def _glyph_line(axis, reference_deg: float) -> tuple[np.ndarray, float] | None:
    """Прямая по центрам глифов строки с наклоном, зажатым около наклона корпуса, и высота по глифам.

    Args:
        axis: Ось строки с глифами.
        reference_deg: Наклон корпуса полосы.

    Returns:
        ``(точки оси, высота)`` или ``None`` — глифов меньше трёх.
    """
    glyphs = axis.glyphs
    if glyphs is None or len(glyphs) < 3:
        return None
    glyphs = np.asarray(glyphs, dtype=np.float64)
    cx = (glyphs[:, 0] + glyphs[:, 2]) / 2.0
    cy = (glyphs[:, 1] + glyphs[:, 3]) / 2.0
    heights = glyphs[:, 3] - glyphs[:, 1]
    # Отсев выносных: глифы с центром в крайних долях по отклонению от прямой корпуса.
    base = np.tan(np.radians(reference_deg))
    resid = cy - base * cx
    low, high = np.quantile(resid, [TRIM_SHARE, 1.0 - TRIM_SHARE])
    keep = (resid >= low) & (resid <= high)
    if keep.sum() < 3:
        keep = np.ones_like(keep)
    slope = np.degrees(np.arctan(np.polyfit(cx[keep], cy[keep], 1)[0]))
    slope = float(np.clip(slope, reference_deg - CLAMP_DEG, reference_deg + CLAMP_DEG))
    k = np.tan(np.radians(slope))
    intercept = float(np.median(cy[keep] - k * cx[keep]))
    xs = np.arange(axis.x0, axis.x1 + STEP_PX, STEP_PX)
    xs[-1] = min(xs[-1], axis.x1)
    return np.column_stack([xs, intercept + k * xs]), float(np.median(heights[keep]))


def fixed_axes(axes: list, dpi: float) -> tuple[list, list[AxisFix]]:
    """Оси полосы с заменёнными выбросами крупного набора и протокол замен.

    Выброс — ось крупного набора (буквы выше ``LARGE_HEIGHT_RATIO`` букв корпуса) длиннее
    ``MIN_LENGTH_MM``, наклон которой расходится с корпусом больше ``TILT_DEG`` и которая НЕ ложится на
    свои буквы (:func:`glyph_fit`, ``FIT_SHARE``): вторая ось по базовой линии глифов на
    крупном курсиве не отсеивает выносные Д, Й и уходит наискось (1971/08 IMG_0072_1L, «СПЕЦОДЕЖДОЙ»:
    4.9° при 0.13° у корпуса). Замена — прежняя ось по центру краски, если её наклон в пределах
    ``TILT_DEG`` от корпуса, иначе прямая по центрам глифов (:func:`_glyph_line`). Высота строки
    берётся по медиане высот глифов: бокс с выносными раздувает полосу, и строки заголовка налезают.

    Args:
        axes: Оси строк (``LineAxis``).
        dpi: Разрешение рабочей копии.

    Returns:
        ``(оси, протокол)``; оси не выбросов возвращаются теми же объектами.
    """
    if len(axes) < 3:
        return list(axes), []
    reference, body_height = body_reference(axes)
    # Кегль корпуса — по буквам, а не по высоте осей: у строки на изгибе бумаги высота оси раздута
    # (1971/08 IMG_0067_2R: 18.5 при корпусе 14), и строка корпуса считалась крупным набором.
    letter_heights = [height for height in (glyph_height(axis) for axis in axes) if height is not None]
    body_glyph = float(np.median(letter_heights)) if letter_heights else None
    min_length = mm_to_px(MIN_LENGTH_MM, dpi)
    out, log = [], []
    for axis in axes:
        before = slope_of(axis.points)
        own = glyph_height(axis)
        if body_glyph is not None and own is not None:
            large = own > LARGE_HEIGHT_RATIO * body_glyph
        else:
            large = axis.height > LARGE_HEIGHT_RATIO * body_height
        if not large or axis.x1 - axis.x0 < min_length or abs(before - reference) <= TILT_DEG:
            out.append(axis)
            continue
        # Ось, лежащая на своих буквах, — не выброс: наклон у строки настоящий.
        if _keeps_own_letters(axis):
            out.append(axis)
            continue
        centre = axis.centre_points
        if centre is not None and abs(slope_of(centre) - reference) <= TILT_DEG:
            points, height, source = np.asarray(centre, dtype=np.float64), axis.height, "centre"
            glyph = _glyph_line(axis, reference)
            if glyph is not None:
                height = min(height, ROW_PER_GLYPH * glyph[1])
        else:
            glyph = _glyph_line(axis, reference)
            if glyph is None:
                out.append(axis)
                log.append(AxisFix(axis.cy, axis.x0, axis.x1, before, float("nan"), "none", axis.height, axis.height))
                continue
            points, height, source = glyph[0], min(axis.height, ROW_PER_GLYPH * glyph[1]), "glyphs"
        fixed = replace(axis, points=points, height=float(height), body_points=None)
        out.append(fixed)
        log.append(AxisFix(axis.cy, axis.x0, axis.x1, before, slope_of(points), source, axis.height, float(height)))
    return out, log


__all__ = ["AxisFix", "body_reference", "fixed_axes", "glyph_fit", "glyph_height", "slope_of"]
