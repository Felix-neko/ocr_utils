"""Гладкая граница текстового блока (способ ``BlocksMode.SMOOTH``): боковые стороны — одна кривая x(y) по краям строк (:mod:`smooth_sides`), верх и низ — полосы крайних строк; без сторон — ступенчатое объединение полос строк."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from ocr_utils.page_layout import mm_to_px
from ocr_utils.page_layout.text_blocks import blocks as legacy
from ocr_utils.page_layout.text_blocks.blocks import FALLBACK_ASPECT, FALLBACK_HEIGHT, BlockEnvelope, Row, pitch_of
from ocr_utils.page_layout.text_blocks.smooth_sides import PUNCT_MM, EdgeKind, side_curve


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
# Наибольший наклон продолжения оси за её концом (добивка строки до стороны), градусы.
EXTEND_MAX_DEG = 3.0
# Упрощение контура, мм.
APPROX_MM = 0.3


@dataclass(frozen=True)
class SmoothEnvelope(BlockEnvelope):
    """Огибающая со сторонами-кривыми и их недостоверными участками (ступеньки по выносам за колонку).

    ``left``/``right`` — кривые сторон ``(N, 2)`` сверху вниз; ``unreliable_left``/``unreliable_right`` —
    отрезки ``(y0, y1)``, где сторона идёт ступенькой по выносу, а не по колонке.
    """

    unreliable_left: tuple[tuple[float, float], ...] = ()
    unreliable_right: tuple[tuple[float, float], ...] = ()


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


def _axis_span(row: Row) -> tuple[float, float]:
    """Концы осей ряда по x (без виртуального хвоста последней строки)."""
    points = _axis_points(row)
    return float(points[0, 0]), float(points[-1, 0])


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
    # Края ряда — по краске текста, но не уже концов его оси. Маска глифов (``ink_axis.glyph_mask``)
    # не берёт компоненты выше 42 px рабочей копии: буквы мастхеда «СНАБЖЕНИЕ» (52–55 px, 1966/01
    # IMG_0004_2R) в краску текста не попадают, край ряда находился только у «Год издания 1-й», и
    # слово выпадало из границы блока. Ось через эти буквы проходит — по ней край и доводится (так же
    # делает боевая кромка-полоса, ``blocks._row_span``).
    spans = [_axis_span(row) for row in rows]
    lefts = np.array([min(row.x0, span[0]) for row, span in zip(rows, spans)], dtype=np.float64)
    rights = np.array([max(row.x1, span[1]) for row, span in zip(rows, spans)], dtype=np.float64)
    if fit_sides and len(rows) > ALIGN_MIN_ROWS:
        ys = np.array([row.y for row in rows], dtype=np.float64)
        left_side = _local_side(ys, lefts, -1.0, dpi)
        right_side = _local_side(ys, rights, +1.0, dpi)
        # Ограничитель вылета: край берётся у соседей, дальше крайнего края рядов блока он не уйдёт.
        lefts = np.where(np.isnan(left_side), lefts, np.minimum(lefts, left_side))
        rights = np.where(np.isnan(right_side), rights, np.maximum(rights, right_side))
    return list(zip(lefts.tolist(), rights.tolist()))


def _guided_at(axis: np.ndarray, xs: np.ndarray, guide: np.ndarray) -> np.ndarray:
    """Ордината оси в точках ``xs``; за концами своей оси — параллельно оси соседа ``guide``.

    Короткая строка (конец абзаца, 15 мм) несёт свой наклон с ошибкой в градус-два, и добивка до
    выровненной стороны на 100 мм уходила клином на 3 мм к номеру полосы (1968/03 IMG_0149_2R:
    наклон последней строки 1.8° при 0.2° у соседей). Сосед по той же бумаге изогнут так же: за
    концом своей оси строка повторяет его ход со сдвигом, совпадающим на конце.

    Args:
        axis: Сглаженная ось строки ``(N, 2)``.
        xs: Точки, в которых нужна ордината.
        guide: Сглаженная ось соседа ``(M, 2)``.

    Returns:
        Ординаты в точках ``xs``.
    """
    ys = np.interp(xs, axis[:, 0], axis[:, 1])
    for mask, anchor in ((xs < axis[0, 0], axis[0]), (xs > axis[-1, 0], axis[-1])):
        if mask.any():
            shift = anchor[1] - _curve_at(guide, np.array([anchor[0]]))[0]
            ys[mask] = _curve_at(guide, xs[mask]) + shift
    return ys


def _guides(
    rows: list[Row], extents: list[tuple[float, float]], dpi: float, reference_deg: float | None = None
) -> list[np.ndarray | None]:
    """Ось-проводник для каждого ряда, полоса которого шире своей оси: ближайший сосед с длинной осью.

    Args:
        rows: Ряды блока сверху вниз.
        extents: Края полос рядов (:func:`_row_extents`).
        dpi: Разрешение рабочей копии.
        reference_deg: Наклон корпуса полосы; длинного соседа нет — проводник прямая с этим наклоном.
            ``None`` — тогда без проводника (продолжение по своему наклону, :func:`_curve_at`).

    Returns:
        Сглаженная ось соседа, прямая с наклоном корпуса или ``None`` — своя ось и так накрывает полосу.
    """
    axes = [_smooth_axis(_axis_points(row), dpi) for row in rows]
    out: list[np.ndarray | None] = []
    for index, (x0, x1) in enumerate(extents):
        own = axes[index]
        if own[0, 0] <= x0 + 1 and own[-1, 0] >= x1 - 1:
            out.append(None)
            continue
        order = [i for pair in zip(range(index - 1, -1, -1), range(index + 1, len(rows))) for i in pair]
        order += list(range(index - 1 - (len(rows) - 1 - index), -1, -1)) + list(range(2 * index + 1, len(rows)))
        # Проводник — ближайший сосед с ДЛИННОЙ осью: вдвое длиннее своей и не короче половины полосы.
        # Требовать, чтобы он накрыл полосу целиком, нельзя: в шапке таблицы (1976/02 IMG_0094_2R) полоса
        # «До внедре-» добивается влево до 100, а строки ниже кончаются на 698 при краю 706 — проводника
        # не находилось, и продолжение шло по наклону самой шапки, клином вверх на 4 мм.
        own_length = own[-1, 0] - own[0, 0]
        need = max(2.0 * own_length, 0.5 * (x1 - x0))
        guide = next((axes[i] for i in order if axes[i][-1, 0] - axes[i][0, 0] >= need), None)
        if guide is None and reference_deg is not None:
            # Длинного соседа нет (одиночная строка, короткий блок): прямая с наклоном корпуса полосы,
            # а не собственный наклон короткой строки.
            slope = np.tan(np.radians(reference_deg))
            guide = np.array([[x0 - 1.0, 0.0], [x1 + 1.0, slope * (x1 - x0 + 2.0)]])
        out.append(guide)
    return out


def _band(row: Row, x0: float, x1: float, dpi: float, guide: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Верхняя и нижняя кривые полосы ряда между ``x0`` и ``x1`` (с полем ``BAND_MARGIN_MM``).

    Сглаженная ось ± постоянный отступ — полоса не повторяет выносные элементы.

    Args:
        row: Ряд.
        x0, x1: Края полосы.
        dpi: Разрешение рабочей копии.
        guide: Сглаженная ось соседней длинной строки или ``None``: за концами своей оси полоса
            идёт параллельно ей (:func:`_guided_at`), а не по собственному наклону.

    Returns:
        Пара кривых ``(верх, низ)`` ``(N, 2)``.
    """
    axis = _smooth_axis(_axis_points(row), dpi)
    xs = np.linspace(x0, x1, max(2, int(x1 - x0) // 2 + 2))
    ys = _guided_at(axis, xs, guide) if guide is not None else _curve_at(axis, xs)
    up, down = _offsets(row)
    margin = mm_to_px(BAND_MARGIN_MM, dpi)
    top, bottom = ys - up, ys + down
    return np.column_stack([xs, top - margin]), np.column_stack([xs, bottom + margin])


def stepped_polygon(rows: list[Row], dpi: float, fit_sides: bool, reference_deg: float | None = None) -> np.ndarray:
    """Контур блока: полосы рядов и заливка промежутков между соседями по их общей ширине.

    Полоса ряда — сглаженная ось ± постоянный отступ, от края до края (:func:`_row_extents`).
    Промежуток между соседними рядами заливается четырёхугольником по их ОБЩЕЙ ширине: где соседи
    не перекрыты, остаётся ступень, а не клин наискосок. Контур — внешний контур растра, упрощённый
    на ``APPROX_MM``.

    Args:
        rows: Ряды блока сверху вниз.
        dpi: Разрешение рабочей копии.
        fit_sides: Добивать ли ряды до выровненных сторон.
        reference_deg: Наклон корпуса полосы для добивки строк без длинного соседа (:func:`_guides`).

    Returns:
        Контур ``(N, 2)`` в пикселях рабочей копии.
    """
    extents = _row_extents(rows, dpi, fit_sides)
    guides = _guides(rows, extents, dpi, reference_deg)
    bands = [_band(row, x0, x1, dpi, guide) for row, (x0, x1), guide in zip(rows, extents, guides)]
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


# Гладкие стороны строятся только у блока от стольких строк.
SMOOTH_MIN_ROWS = 3
# Хотя бы одна сторона должна держаться на стольких строках и на такой доле строк блока.
SMOOTH_MIN_ALIGNED = 4
SMOOTH_ALIGNED_SHARE = 0.6


def _has_sides(rows: list[Row]) -> bool:
    """Есть ли у блока боковые стороны: не меньше ``SMOOTH_MIN_ROWS`` строк и никакие две не стоят на одной высоте.

    Строки бок о бок (подпись автора рядом со словом заголовка — 1966/01 IMG_0021_2R) — это не колонка, и
    кривая стороны по их краям давала косые перемычки через пустоту.
    """
    if len(rows) < SMOOTH_MIN_ROWS:
        return False
    for upper, lower in zip(rows[:-1], rows[1:]):
        if lower.y - upper.y < 0.5 * min(upper.height, lower.height):
            return False
    return True


def _within_rows(polygon: np.ndarray, rows: list[Row], dpi: float) -> bool:
    """Лежит ли контур в рамке строк блока (края не уже концов осей) с полем ``PUNCT_MM``."""
    spans = [_axis_span(row) for row in rows]
    x0 = min(min(row.x0, span[0]) for row, span in zip(rows, spans))
    x1 = max(max(row.x1, span[1]) for row, span in zip(rows, spans))
    y0 = min(row.y - row.height for row in rows)
    y1 = max(row.y + row.height for row in rows)
    pad = mm_to_px(PUNCT_MM, dpi)
    return bool(
        polygon[:, 0].min() >= x0 - pad
        and polygon[:, 0].max() <= x1 + pad
        and polygon[:, 1].min() >= y0 - pad
        and polygon[:, 1].max() <= y1 + pad
    )


def smooth_envelope(rows: list[Row], dpi: float, reference_deg: float | None) -> SmoothEnvelope | None:
    """Огибающая с гладкими сторонами: левая и правая — :func:`smooth_sides.side_curve`, верх и низ — полосы крайних строк.

    Края строк — не уже концов их осей (:func:`_axis_span`). Контур обходится как у боевой огибающей:
    левая сторона сверху вниз, низ слева направо, правая снизу вверх, верх справа налево; сторона
    обрезается по верху и низу там, где с ними встречается.

    Args:
        rows: Ряды блока сверху вниз.
        dpi: Разрешение рабочей копии.
        reference_deg: Наклон корпуса полосы (для продолжения осей крайних строк, :func:`_guides`).

    Returns:
        :class:`SmoothEnvelope`; ``None`` — ни одна сторона не выровнена.
    """
    pitch = pitch_of(rows) if len(rows) > 1 else 1.6 * rows[0].height
    spans = [_axis_span(row) for row in rows]
    ys = np.array([row.y for row in rows], dtype=np.float64)
    lefts = np.array([min(row.x0, span[0]) for row, span in zip(rows, spans)], dtype=np.float64)
    rights = np.array([max(row.x1, span[1]) for row, span in zip(rows, spans)], dtype=np.float64)
    margin = mm_to_px(BAND_MARGIN_MM, dpi)
    # Зона строки по высоте для сторон — полоса строки (ось ± отступы) с полем.
    offsets = [_offsets(row) for row in rows]
    tops = np.array([row.y - up - margin for row, (up, _) in zip(rows, offsets)])
    bottoms = np.array([row.y + down + margin for row, (_, down) in zip(rows, offsets)])
    right = side_curve(ys, rights + margin, tops, bottoms, pitch, dpi)
    left = side_curve(ys, -(lefts - margin), tops, bottoms, pitch, dpi)
    # Ни одна сторона не выровнена (подпись автора по центру, заголовок) — гладким сторонам не за что
    # держаться: блок строится ступенчатым объединением полос строк.
    # Сторона считается выровненной, если на ней не меньше ``SMOOTH_MIN_ALIGNED`` строк и не меньше
    # ``SMOOTH_ALIGNED_SHARE`` блока: у центрированной подписи из пяти строк три края случайно ложатся
    # на прямую, и сторона шла клином через пустоту (1974/12 IMG_0128_2R).
    aligned = {EdgeKind.SPLINE, EdgeKind.BUMP}
    need = max(SMOOTH_MIN_ALIGNED, SMOOTH_ALIGNED_SHARE * len(rows))
    if all(sum(kind in aligned for kind in side.kinds) < need for side in (left, right)):
        return None
    grid, x_left, x_right = right.ys, -left.us, right.us
    # Верх и низ — полосы первой и последней строки между сторонами на их высоте.
    extents = [(float(np.interp(row.y, grid, x_left)), float(np.interp(row.y, grid, x_right))) for row in rows]
    guides = _guides(rows, extents, dpi, reference_deg)
    top, _ = _band(rows[0], *extents[0], dpi, guide=guides[0])
    _, bottom = _band(rows[-1], *extents[-1], dpi, guide=guides[-1])
    # Стороны обрезаются по крышкам: от ординаты крышки на своём конце до ординаты другой крышки.
    left_mask = (grid > top[0, 1]) & (grid < bottom[0, 1])
    right_mask = (grid > top[-1, 1]) & (grid < bottom[-1, 1])
    left_side = np.vstack([top[:1], np.column_stack([x_left[left_mask], grid[left_mask]]), bottom[:1]])
    right_side = np.vstack([top[-1:], np.column_stack([x_right[right_mask], grid[right_mask]]), bottom[-1:]])
    polygon = np.vstack([left_side, bottom[1:-1], right_side[::-1], top[-2:0:-1]])
    simple = cv2.approxPolyDP(polygon.astype(np.float32).reshape(-1, 1, 2), mm_to_px(APPROX_MM, dpi), True)
    if simple.shape[0] >= 3:
        polygon = simple.reshape(-1, 2).astype(np.float64)
    # Тренд сторон для мер выключки (``alignment``): сплайн по краям строк без поля и сдвига наружу, в
    # пределах строк; у невыровненной стороны — сама сторона.
    rows_span = (grid >= ys[0]) & (grid <= ys[-1])
    core_left = np.column_stack([-left.core[rows_span] + margin, grid[rows_span]]) if left.core is not None else None
    core_right = np.column_stack([right.core[rows_span] - margin, grid[rows_span]]) if right.core is not None else None
    return SmoothEnvelope(
        core_left=core_left if core_left is not None and len(core_left) >= 2 else None,
        core_right=core_right if core_right is not None and len(core_right) >= 2 else None,
        left=left_side,
        right=right_side,
        top=top,
        bottom=bottom,
        polygon=polygon,
        smooth_pitches=0.0,
        unreliable_left=left.unreliable,
        unreliable_right=right.unreliable,
    )


def _stepped_envelope(rows: list[Row], dpi: float, reference_deg: float | None) -> BlockEnvelope:
    """Огибающая блока без гладких сторон: контур — ступенчатое объединение полос строк (:func:`stepped_polygon`).

    Кромки для мер сторон (``sides.py``, выключка) — края полос строк сверху вниз (левая и правая), верх
    и низ — полосы первой и последней строки.

    Args:
        rows: Ряды блока сверху вниз.
        dpi: Разрешение рабочей копии.
        reference_deg: Наклон корпуса полосы.

    Returns:
        :class:`BlockEnvelope`.
    """
    polygon = stepped_polygon(rows, dpi, fit_sides=True, reference_deg=reference_deg)
    extents = _row_extents(rows, dpi, fit_sides=True)
    guides = _guides(rows, extents, dpi, reference_deg)
    top, _ = _band(rows[0], *extents[0], dpi, guide=guides[0])
    _, bottom = _band(rows[-1], *extents[-1], dpi, guide=guides[-1])
    ys = np.array([row.y for row in rows], dtype=np.float64)
    left = np.column_stack([[x0 for x0, _ in extents], ys])
    right = np.column_stack([[x1 for _, x1 in extents], ys])
    return BlockEnvelope(left=left, right=right, top=top, bottom=bottom, polygon=polygon, smooth_pitches=0.0)


def envelope_smooth(rows: list[Row], dpi: float, reference_deg: float | None) -> BlockEnvelope:
    """Граница блока способа ``SMOOTH``: гладкие стороны, где они есть, иначе ступенчатое объединение полос строк.

    Гладкие стороны строятся у блока от ``SMOOTH_MIN_ROWS`` строк без строк бок о бок, если хотя бы
    одна сторона выровнена (:func:`smooth_envelope`) и контур не выходит за рамку строк дальше
    выноса пунктуации (:func:`_within_rows`). Иначе — :func:`_stepped_envelope`.

    Args:
        rows: Ряды блока сверху вниз.
        dpi: Разрешение рабочей копии.
        reference_deg: Наклон корпуса полосы (:func:`axes_fix.body_reference`) или ``None``.

    Returns:
        :class:`SmoothEnvelope` или :class:`BlockEnvelope`.
    """
    if _has_sides(rows):
        envelope = smooth_envelope(rows, dpi, reference_deg)
        if envelope is not None and _within_rows(envelope.polygon, rows, dpi):
            return envelope
    return _stepped_envelope(rows, dpi, reference_deg)


__all__ = ["SmoothEnvelope", "envelope_smooth", "smooth_envelope", "stepped_polygon"]
