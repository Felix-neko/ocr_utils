"""Текстовый блок = колонка: ряды строк, края рядов по краске и гладкая огибающая блока.

Огибающая строится не по крайним точкам краски (тогда её тянут абзацные отступы и висячие
строки), а КВАНТИЛЬНЫМ ТРЕНДОМ: в скользящем окне шириной в несколько межстрочных интервалов
берётся квантиль краёв рядов (слева нижний, справа верхний) и сглаживается. Получается плавная
кривая по телу блока — то, что нельзя описать прямой на изогнутой бумаге.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from ocr_utils.curved_layout import RENDER_DPI, WORK_DPI
from ocr_utils.curved_layout.columns import bounds_at
from ocr_utils.curved_layout.lines import LineAxis, with_column
from ocr_utils.page_layout import mm_to_px, px_to_mm

# Край ряда ищется в пределах колонки с этим припуском: выносные элементы и кавычки вылезают за
# границу колонки, найденную по профилю.
COLUMN_PAD_MM = 5.0
# Столбец считается краем, если в полосе ряда в нём не меньше стольких пикселей краски и краска
# есть ещё не меньше чем в EDGE_CONFIRM_MIN из следующих EDGE_CONFIRM_COLS столбцов (пыль не край).
EDGE_MIN_INK_PX = 2
EDGE_CONFIRM_COLS = 3
EDGE_CONFIRM_MIN = 2
# Полоса ряда расширяется на эту долю высоты: край буквы с выносным элементом иначе теряется.
ROW_PAD_HEIGHTS = 0.15
# Сегменты с центрами ближе этой доли высоты — один ряд.
ROW_TOL_HEIGHTS = 0.5
# Окно огибающей по умолчанию — в межстрочных интервалах (решение пользователя: 2–3, настраивается).
SMOOTH_PITCHES = 2.5
# Во сколько раз крупнее окно у «крупной» огибающей, с которой сравнивается основная.
COARSE_FACTOR = 3.0
# Квантили краёв в окне: слева берём нижний, справа верхний — отступ абзаца и короткий конец
# строки остаются за кривой и не тянут её внутрь блока.
EDGE_QUANTILE = 0.25
# Шаг сетки огибающей по y.
GRID_STEP_MM = 1.0
# Блок короче стольких рядов не строится. Один ряд — это тоже блок (заголовок в одну строку,
# колонтитул, подпись автора): его огибающая идёт по оси строки с полем в полвысоты.
MIN_BLOCK_ROWS = 1
# Насколько край ряда может выйти за границу колонки (мм): дальше — чужая колонка.
OVERHANG_MM = 1.5
# Доля длины строки, которая должна лежать в колонке, чтобы строка считалась её строкой.
INSIDE_SHARE = 0.9
# Заход края ряда внутрь блока дальше этого (мм) — абзацный отступ или короткий конец строки:
# в тренд кромки такой ряд не идёт (но в мерах выключки считается отдельно).
TRIM_MM = 2.5
# Рядов в окне локальной прямой не меньше этого: по двум-трём кромку не провести.
MIN_TREND_ROWS = 5
# Вертикальный разрыв больше стольких межстрочных интервалов делит колонку на два блока.
BLOCK_GAP_PITCHES = 3.0
# Медианные высоты по HEIGHT_WINDOW рядам с обеих сторон границы различаются во столько раз —
# другой кегль, другой блок (заголовок над корпусом, подпись автора под ним).
HEIGHT_BREAK_RATIO = 1.45
HEIGHT_WINDOW = 3
# Сплошная черта делит блок, если она перекрывает колонку не меньше чем на эту долю ширины:
# разделитель сноски в журнале — короткая черта примерно в четверть колонки.
RULE_OVERLAP_SHARE = 0.15
# Обрывок короче MIN_BLOCK_ROWS приклеивается к соседней группе, если разрыв не больше этого.
MERGE_GAP_PITCHES = 6.0
# Куски колонки из соседних зон считаются одной колонкой при таком перекрытии по ширине и
# совпадении левого края (мм).
MERGE_OVERLAP = 0.6
MERGE_EDGE_MM = 3.0
MERGE_WIDTH_RATIO = 1.5
# Блок шире типичного во столько раз и короче стольких рядов — колонтитул, а не блок.
WIDE_BLOCK_RATIO = 1.6
WIDE_BLOCK_ROWS = 5
# Столько раз повторяется «тренд → отбор рядов тела → тренд».
TREND_PASSES = 2
# Насколько кромке позволено выйти за диапазон краёв рядов (мм): дальше — экстраполяция.
TREND_CLIP_MM = 3.0


@dataclass(frozen=True)
class Row:
    """Ряд блока: одна строка (или слитые сегменты одной y-полосы) с краями по краске."""

    y: float  # ордината середины ряда, пиксели рабочей копии
    height: float
    x0: float  # край краски слева (None-строки сюда не попадают)
    x1: float
    axes: tuple[LineAxis, ...]

    @property
    def axis(self) -> LineAxis:
        """Самая длинная ось ряда — она представляет ряд в верхней и нижней кромках блока."""
        return max(self.axes, key=lambda item: item.x1 - item.x0)


@dataclass(frozen=True)
class BlockEnvelope:
    """Огибающая блока: четыре кривые и замкнутый контур по ним (пиксели рабочей копии)."""

    left: np.ndarray
    right: np.ndarray
    top: np.ndarray
    bottom: np.ndarray
    polygon: np.ndarray
    smooth_pitches: float


@dataclass(frozen=True)
class TextBlock:
    """Текстовый блок: ряды колонки, шаг строк и две огибающие разного масштаба сглаживания.

    Блок — колонка целиком (решение пользователя), но колонка, разорванная по вертикали большим
    пробелом (заголовок между двумя заметками — 1975/05 с.97), делится на блоки по этому пробелу:
    иначе огибающая тянется через пустоту в двадцать строк.
    """

    column: int
    index: int  # номер блока внутри колонки, сверху вниз
    span: tuple[int, int]  # границы колонки по x
    rows: tuple[Row, ...]
    pitch_px: float
    dpi: float
    envelope: BlockEnvelope
    envelope_coarse: BlockEnvelope

    @property
    def pitch_mm(self) -> float:
        return px_to_mm(self.pitch_px, self.dpi)

    @property
    def lines(self) -> int:
        return len(self.rows)


def ink_edge(ink: np.ndarray, x_lo: int, x_hi: int, y0: int, y1: int, side: str) -> float | None:
    """Край краски в полосе ``[y0, y1)`` между столбцами ``x_lo`` и ``x_hi`` (пиксели ``ink``).

    Слева — первый столбец с краской, подтверждённой соседями справа, справа — наоборот.
    Тот же приём, что у кромок блока в детекторе геометрии (v16): одиночная точка (пыль,
    пробившаяся засечка) краем не становится.
    """
    height, width = ink.shape
    x_lo, x_hi, y0, y1 = max(0, x_lo), min(width, x_hi), max(0, y0), min(height, y1)
    if x_hi - x_lo < EDGE_CONFIRM_COLS + 1 or y1 <= y0:
        return None
    counts = (ink[y0:y1, x_lo:x_hi] > 0).sum(axis=0)
    filled = counts >= EDGE_MIN_INK_PX
    order = range(len(filled)) if side == "left" else range(len(filled) - 1, -1, -1)
    for i in order:
        if not filled[i]:
            continue
        window = (
            filled[i + 1 : i + 1 + EDGE_CONFIRM_COLS] if side == "left" else filled[max(0, i - EDGE_CONFIRM_COLS) : i]
        )
        if window.sum() >= EDGE_CONFIRM_MIN:
            return float(x_lo + i)
    return None


def rows_of(
    axes: list[LineAxis],
    ink: np.ndarray,
    span: tuple[int, int],
    dpi: float,
    gutters: list | None = None,
    width: int | None = None,
    pad: float | None = None,
) -> list[Row]:
    """Ряды колонки: оси с близкими ординатами сливаются, края берутся по краске рендера.

    Args:
        axes: Оси строк колонки.
        ink: Краска рендера ``RENDER_DPI`` (ненулевое — краска).
        span: Границы колонки ``(x0, x1)`` в пикселях рабочей копии (границы зоны).
        dpi: Разрешение рабочей копии.
        gutters: Межколонники-ломаные: окно поиска края берётся по ним НА ВЫСОТЕ РЯДА, чтобы на
            трапеции не заехать в соседнюю колонку (при межколоннике в 4 мм припуск в 5 мм уводил
            правый край ряда на букву соседней колонки — 1975/05 с.97).
        width: Ширина рабочей копии (нужна вместе с ``gutters``).
        pad: Припуск к границам колонки; по умолчанию ``COLUMN_PAD_MM``.

    Returns:
        Ряды сверху вниз; ряды, у которых край не нашёлся, выбрасываются.
    """
    k = RENDER_DPI / dpi
    pad = mm_to_px(COLUMN_PAD_MM, dpi) if pad is None else pad
    groups: list[list[LineAxis]] = []
    for axis in sorted(axes, key=lambda item: item.cy):
        if groups and abs(axis.cy - np.median([item.cy for item in groups[-1]])) <= ROW_TOL_HEIGHTS * axis.height:
            groups[-1].append(axis)
        else:
            groups.append([axis])
    rows: list[Row] = []
    for group in groups:
        height = float(np.median([item.height for item in group]))
        y = float(np.median([item.cy for item in group]))
        row_x0 = float(min(item.x0 for item in group))
        row_x1 = float(max(item.x1 for item in group))
        margin = ROW_PAD_HEIGHTS * height
        y0 = int((y - height / 2.0 - margin) * k)
        y1 = int((y + height / 2.0 + margin) * k) + 1
        # Окно поиска края — границы колонки на высоте ряда плюс припуск, но не дальше середины
        # межколонника: иначе крайняя буква соседней колонки становится краем этого ряда.
        if gutters is not None and width is not None:
            left_bound, right_bound = bounds_at(gutters, row_x0, row_x1, y, width)
        else:
            left_bound, right_bound = float(span[0]), float(span[1])
        x_lo = max(0.0, left_bound - pad)
        x_hi = min(float(ink.shape[1] / k), right_bound + pad)
        if gutters is not None:
            for gutter in gutters:
                if not gutter.alive_at(y):
                    continue
                gx0, gx1 = gutter.x0_at(y), gutter.x1_at(y)
                if gx1 <= left_bound + 1:
                    x_lo = max(x_lo, (gx0 + gx1) / 2.0)
                elif gx0 >= right_bound - 1:
                    x_hi = min(x_hi, (gx0 + gx1) / 2.0)
        left = ink_edge(ink, int(x_lo * k), int(x_hi * k), y0, y1, "left")
        right = ink_edge(ink, int(x_lo * k), int(x_hi * k), y0, y1, "right")
        if left is None or right is None:
            continue
        left, right = left / k, right / k
        # Край, вылезший за колонку дальше OVERHANG_MM, — не край этого ряда: так выглядит
        # строка, сшитая через межколонник (1973/07 с.88), и колонтитул во всю ширину.
        overhang = mm_to_px(OVERHANG_MM, dpi)
        if left < left_bound - overhang or right > right_bound + overhang:
            continue
        rows.append(Row(y=y, height=height, x0=left, x1=right, axes=tuple(group)))
    return rows


def row_gap(previous: Row, row: Row) -> float:
    """Расстояние между соседними рядами ВДОЛЬ НАКЛОНА: по ординатам осей в общей точке по x.

    Разница ординат центров на наклонных строках врёт: у двух рядов одной колонки центры могут
    разойтись на высоту строки просто из-за наклона, а чужой ряд (номер полосы, уехавший под
    блок) — наоборот, оказаться рядом. В общей точке по x такого не происходит.
    """
    x_lo = max(previous.axis.x0, row.axis.x0)
    x_hi = min(previous.axis.x1, row.axis.x1)
    if x_hi <= x_lo:
        return row.y - previous.y
    x = (x_lo + x_hi) / 2.0
    return row.axis.y_at(x) - previous.axis.y_at(x)


def _height_break(rows: list[Row], index: int) -> bool:
    """Устойчивая смена кегля на границе после ряда ``index``.

    Сравниваются МЕДИАНЫ высот по нескольким рядам с каждой стороны, а не сами соседи: высота
    отдельного ряда корпуса скачет на выносных элементах и прописных (13 против 17 px), и по
    соседям колонка дробилась на куски. Заголовок же выше корпуса устойчиво.
    """
    # На коротких группах кегль не меряют: у крупного набора высота «строки» скачет (слипшиеся
    # строки заголовка дают 65 px против 44), и блок заголовка дробился на куски по одной строке
    # (1973/06 с.65). Там границу и так ставят разрыв и линейка.
    if len(rows) < 2 * HEIGHT_WINDOW:
        return False
    before = [row.height for row in rows[max(0, index - HEIGHT_WINDOW + 1) : index + 1]]
    after = [row.height for row in rows[index + 1 : index + 1 + HEIGHT_WINDOW]]
    if not before or not after:
        return False
    lo, hi = sorted((float(np.median(before)), float(np.median(after))))
    return hi > HEIGHT_BREAK_RATIO * max(lo, 1e-6)


def _rule_between(previous: Row, row: Row, rules: list, span: tuple[int, int]) -> bool:
    """Проходит ли между рядами сплошная черта, перекрывающая колонку.

    Разделитель сноски — именно такая черта: текст под ней относится к сноске, а не к блоку
    (1973/06 с.65, правая колонка).
    """
    width = max(1, span[1] - span[0])
    for rule in rules:
        if not (previous.y < rule.cy < row.y):
            continue
        overlap = min(rule.x1, span[1]) - max(rule.x0, span[0])
        if overlap >= RULE_OVERLAP_SHARE * width:
            return True
    return False


def split_blocks(
    rows: list[Row], pitch: float, rules: list | None = None, span: tuple[int, int] | None = None
) -> list[list[Row]]:
    """Разделить ряды колонки на блоки по вертикальным разрывам больше ``BLOCK_GAP_PITCHES`` шагов.

    Обрывок из одной-двух строк (пропала строка при сегментации — жирный или разрядка) к блоку
    не относится сам по себе: он приклеивается к соседней группе, если разрыв до неё меньше
    ``MERGE_GAP_PITCHES`` шагов. Настоящий разрыв между заметками (1975/05 с.97 — полтора десятка
    шагов) так не склеится.
    """
    if len(rows) < 2:
        return [rows]
    groups: list[list[Row]] = [[rows[0]]]
    limit = BLOCK_GAP_PITCHES * pitch
    rules = rules or []
    for index, (previous, row) in enumerate(zip(rows[:-1], rows[1:])):
        divided = (
            row_gap(previous, row) > limit
            or _height_break(rows, index)
            or (span is not None and _rule_between(previous, row, rules, span))
        )
        if divided:
            groups.append([row])
        else:
            groups[-1].append(row)
    merged: list[list[Row]] = []
    for group in groups:
        gap = row_gap(merged[-1][-1], group[0]) if merged else None
        small = len(group) < MIN_BLOCK_ROWS or (merged and len(merged[-1]) < MIN_BLOCK_ROWS)
        joinable = (
            merged
            and small
            and gap is not None
            and gap <= MERGE_GAP_PITCHES * pitch
            and not _height_break([*merged[-1], *group], len(merged[-1]) - 1)
            and not (span is not None and _rule_between(merged[-1][-1], group[0], rules, span))
        )
        if joinable:
            merged[-1].extend(group)
        else:
            merged.append(group)
    return merged


def pitch_of(rows: list[Row]) -> float:
    """Межстрочный шаг блока: медиана расстояний между серединами соседних рядов (пиксели)."""
    if len(rows) < 2:
        return float(rows[0].height * 1.6) if rows else 0.0
    gaps = np.array([row_gap(previous, row) for previous, row in zip(rows[:-1], rows[1:])])
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return float(rows[0].height * 1.6)
    # Между абзацами и вокруг заголовка шаг больше: берём медиану, а не среднее.
    return float(np.median(gaps))


def _quantile_trend(ys: np.ndarray, xs: np.ndarray, grid: np.ndarray, window: float, quantile: float) -> np.ndarray:
    """Грубый квантильный тренд ``x(y)``: в окне ``window`` вокруг узла сетки — квантиль ``xs``.

    Нужен только как первое приближение, по которому отбираются ряды тела блока: сам по себе
    квантиль в скользящем окне прыгает там, где в окно попало два-три ряда.
    """
    out = np.empty(grid.shape, dtype=np.float64)
    half = window / 2.0
    for i, node in enumerate(grid):
        own = xs[np.abs(ys - node) <= half]
        if own.size == 0:
            # Окно пустое (разрыв между абзацами шире окна) — берём ближайшую точку.
            own = xs[np.argsort(np.abs(ys - node))[:1]]
        out[i] = float(np.quantile(own, quantile))
    return out


def _local_line(ys: np.ndarray, xs: np.ndarray, grid: np.ndarray, window: float) -> np.ndarray:
    """Локальная взвешенная прямая (LOWESS первой степени) ``x(y)`` в узлах сетки.

    Прямая в окне, а не среднее: на наклонном блоке среднее сдвигает кромку внутрь на полокна.
    Веса — трикубические по расстоянию до узла, поэтому кривая не дёргается, когда ряд входит в
    окно и выходит из него. Если в окно попало меньше ``MIN_TREND_ROWS`` рядов, окно
    расширяется до ближайших ``MIN_TREND_ROWS``: по двум-трём рядам кромку не провести, и
    именно на этом ломалась кромка у коротких концов абзацев (1973/07 с.77).
    """
    out = np.empty(grid.shape, dtype=np.float64)
    half = max(window / 2.0, 1e-6)
    if ys.size == 0:
        return out
    for i, node in enumerate(grid):
        distance = np.abs(ys - node)
        reach = half
        if (distance <= reach).sum() < MIN_TREND_ROWS:
            nearest = np.sort(distance)[: min(MIN_TREND_ROWS, ys.size)]
            reach = max(reach, float(nearest[-1]))
        own = distance <= reach
        if own.sum() == 0:
            out[i] = float(xs[np.argmin(distance)])
            continue
        weights = (1.0 - (distance[own] / reach) ** 3) ** 3
        if own.sum() >= 2 and weights.sum() > 0:
            slope, intercept = np.polyfit(ys[own], xs[own], 1, w=np.sqrt(weights))
            out[i] = slope * node + intercept
        else:
            out[i] = float(np.average(xs[own], weights=weights))
    return out


def _smooth(values: np.ndarray, window_nodes: int) -> np.ndarray:
    """Сглаживание кромки по сетке: Савицкий–Голей второй степени, окно в узлах."""
    window = max(3, int(window_nodes) | 1)
    if values.size <= window:
        return values
    return savgol_filter(values, window_length=window, polyorder=2, mode="nearest")


def _single_row_envelope(row: Row, smooth_pitches: float) -> BlockEnvelope:
    """Огибающая блока из одного ряда: по его оси, с полем в половину высоты.

    Тренд по одному ряду не построить, поэтому кромки — вертикальные отрезки по краям краски, а
    верх и низ — сама ось, поднятая и опущенная на полвысоты.
    """
    axis = row.axis
    half = row.height / 2.0
    top = _cap(row, row.x0, row.x1, -1.0)
    bottom = _cap(row, row.x0, row.x1, 1.0)
    left = np.array([[row.x0, axis.y_at(row.x0) - half], [row.x0, axis.y_at(row.x0) + half]])
    right = np.array([[row.x1, axis.y_at(row.x1) - half], [row.x1, axis.y_at(row.x1) + half]])
    polygon = np.vstack([left, bottom, right[::-1], top[::-1]])
    return BlockEnvelope(
        left=left, right=right, top=top, bottom=bottom, polygon=polygon, smooth_pitches=float(smooth_pitches)
    )


def envelope_of(rows: list[Row], pitch: float, dpi: float, smooth_pitches: float) -> BlockEnvelope:
    """Гладкая огибающая блока с окном ``smooth_pitches`` межстрочных интервалов.

    Левая и правая кромки — квантильный тренд краёв рядов по сетке шага ``GRID_STEP_MM``;
    верх и низ — оси крайних рядов, поднятые (опущенные) на половину высоты и продлённые до
    кромок. Контур собирается обходом: левая сверху вниз, низ слева направо, правая снизу
    вверх, верх справа налево.
    """
    if len(rows) == 1:
        return _single_row_envelope(rows[0], smooth_pitches)
    ys = np.array([row.y for row in rows], dtype=np.float64)
    xs_left = np.array([row.x0 for row in rows], dtype=np.float64)
    xs_right = np.array([row.x1 for row in rows], dtype=np.float64)
    step = mm_to_px(GRID_STEP_MM, dpi)
    top_row, bottom_row = rows[0], rows[-1]
    y_top = top_row.y - top_row.height / 2.0
    y_bottom = bottom_row.y + bottom_row.height / 2.0
    grid = np.arange(y_top, y_bottom + step, step)
    window = max(smooth_pitches * pitch, 2 * step)
    left = _side_trend(ys, xs_left, grid, window, dpi, inward=+1.0)
    right = _side_trend(ys, xs_right, grid, window, dpi, inward=-1.0)
    nodes = max(3, int(round(window / step)))
    left, right = _smooth(left, nodes), _smooth(right, nodes)
    left_curve = np.column_stack([left, grid])
    right_curve = np.column_stack([right, grid])
    top_curve = _cap(top_row, left[0], right[0], -1.0)
    bottom_curve = _cap(bottom_row, left[-1], right[-1], 1.0)
    polygon = np.vstack([left_curve, bottom_curve, right_curve[::-1], top_curve[::-1]])
    return BlockEnvelope(
        left=left_curve,
        right=right_curve,
        top=top_curve,
        bottom=bottom_curve,
        polygon=polygon,
        smooth_pitches=float(smooth_pitches),
    )


def _side_trend(
    ys: np.ndarray, xs: np.ndarray, grid: np.ndarray, window: float, dpi: float, inward: float
) -> np.ndarray:
    """Кромка блока: грубый квантиль → отбор рядов тела → локальная прямая по ним.

    Args:
        ys, xs: Ординаты рядов и их края с этой стороны.
        grid: Узлы сетки по y, на которых считается кромка.
        window: Окно сглаживания в пикселях.
        dpi: Разрешение рабочей копии (для перевода допуска из мм).
        inward: Куда от кромки смотрит внутренность блока: ``+1`` для левой стороны, ``-1`` для правой.

    Returns:
        Значения кромки в узлах сетки.
    """
    quantile = EDGE_QUANTILE if inward > 0 else 1.0 - EDGE_QUANTILE
    trend = _quantile_trend(ys, xs, grid, max(window * 2.0, window), quantile)
    trim = mm_to_px(TRIM_MM, dpi)
    keep = np.ones(ys.shape, dtype=bool)
    # Два прохода: по грубому тренду отбрасываются ряды, ушедшие внутрь блока (абзацный отступ,
    # короткий конец абзаца), затем кромка пересчитывается по оставшимся и отбор повторяется.
    for _ in range(TREND_PASSES):
        fitted = np.interp(ys, grid, trend)
        inside = inward * (xs - fitted) > trim
        keep = ~inside
        if keep.sum() < MIN_TREND_ROWS:
            keep = np.ones(ys.shape, dtype=bool)
        trend = _local_line(ys[keep], xs[keep], grid, window)
    # Кромка не должна уходить за пределы самих краёв рядов: на краях сетки локальная прямая
    # экстраполирует, и у коротких блоков огибающая улетала на десятки миллиметров.
    margin = mm_to_px(TREND_CLIP_MM, dpi)
    return np.clip(trend, xs[keep].min() - margin, xs[keep].max() + margin)


def _cap(row: Row, x_left: float, x_right: float, direction: float) -> np.ndarray:
    """Верхняя (``direction=-1``) или нижняя (``+1``) кромка блока по оси крайнего ряда."""
    axis = row.axis
    offset = direction * row.height / 2.0
    xs = np.linspace(x_left, x_right, num=max(2, int(abs(x_right - x_left) / 4) + 2))
    ys = np.array([axis.y_at(x) + offset for x in xs], dtype=np.float64)
    return np.column_stack([xs, ys])


def blocks_of(
    axes: list[LineAxis],
    zones: list,
    gutters: list,
    width: int,
    ink: np.ndarray,
    rules: list | None = None,
    dpi: float = WORK_DPI,
    smooth_pitches: float = SMOOTH_PITCHES,
    coarse_factor: float = COARSE_FACTOR,
) -> list[TextBlock]:
    """Блоки страницы: колонки зон вёрстки, склеенные по вертикали и разделённые разрывами.

    Зона вёрстки кончается там, где появляется или пропадает межколонник (врезка сверху справа —
    1967/10 с.63), но колонка основного текста при этом продолжается: соседние по вертикали
    куски одной колонки склеиваются обратно, если их границы совпадают (перекрытие не меньше
    ``MERGE_OVERLAP``) и разрыв между ними меньше ``BLOCK_GAP_PITCHES`` шагов строк.

    Args:
        axes: Оси всех строк страницы.
        zones: Зоны вёрстки (``columns.Zone``).
        gutters: Локальные межколонники-ломаные (``columns.Gutter``).
        width: Ширина рабочей копии.
        ink: Краска текста рендера ``RENDER_DPI``.
        rules: Сплошные черты страницы (``segment.Rule``): делят блок (сноска под линейкой).
        dpi: Разрешение рабочей копии.
        smooth_pitches: Окно основной огибающей в межстрочных интервалах.
        coarse_factor: Во сколько раз шире окно крупной огибающей.

    Returns:
        Блоки, слева направо и сверху вниз.
    """
    pad = mm_to_px(COLUMN_PAD_MM, dpi)
    pieces: list[tuple[tuple[int, int], list[Row]]] = []
    placed: set[tuple[int, int, int]] = set()
    for zone in zones:
        own_zone = [axis for axis in axes if zone.y0 <= axis.cy < zone.y1]
        for column, span in enumerate(zone.columns):
            own = [
                with_column(axis, column) for axis in own_zone if not axis.cross and _inside(axis, span, gutters, width)
            ]
            if not own:
                continue
            placed.update(_axis_key(axis) for axis in own)
            rows = rows_of(own, ink, span, dpi, gutters=gutters, width=width, pad=pad)
            if rows:
                pieces.append((span, rows))
    # Строки, которым колонки не нашлось (набраны через межколонник, шире любой колонки —
    # колонтитул, заголовок во всю ширину), собираются в блоки ПО ВСЕЙ СТРАНИЦЕ, а не по зонам:
    # заголовок и его подзаголовок часто попадают в соседние зоны и иначе разъезжаются.
    rest = [axis for axis in axes if _axis_key(axis) not in placed]
    if rest:
        rows = rows_of(rest, ink, (0, width), dpi, pad=pad)
        if rows:
            pieces.append(((0, width), rows))
    merged = _merge_pieces(pieces, dpi)
    widths = [span[1] - span[0] for span, _ in merged]
    typical = float(np.median(widths)) if widths else 0.0
    blocks: list[TextBlock] = []
    for column, (span, rows) in enumerate(merged):
        if len(rows) < MIN_BLOCK_ROWS:
            continue
        # Обрывок во всю ширину страницы из пары строк — колонтитул или строки, слипшиеся над
        # межколонником: у него ширина заметно больше типичной, а строк мало.
        if typical and span[1] - span[0] > WIDE_BLOCK_RATIO * typical and len(rows) < WIDE_BLOCK_ROWS:
            continue
        for number, group in enumerate(split_blocks(rows, pitch_of(rows), rules, span)):
            if len(group) < MIN_BLOCK_ROWS:
                continue
            pitch = pitch_of(group)
            blocks.append(
                TextBlock(
                    column=column,
                    index=number,
                    span=span,
                    rows=tuple(group),
                    pitch_px=pitch,
                    dpi=float(dpi),
                    envelope=envelope_of(group, pitch, dpi, smooth_pitches),
                    envelope_coarse=envelope_of(group, pitch, dpi, smooth_pitches * coarse_factor),
                )
            )
    return blocks


def _merge_pieces(
    pieces: list[tuple[tuple[int, int], list[Row]]], dpi: float
) -> list[tuple[tuple[int, int], list[Row]]]:
    """Склеить куски одной колонки из соседних зон: тот же левый край и небольшой разрыв по высоте."""
    out: list[tuple[tuple[int, int], list[Row]]] = []
    for span, rows in sorted(pieces, key=lambda item: (item[0][0], item[1][0].y)):
        joined = False
        for index, (other_span, other_rows) in enumerate(out):
            overlap = min(span[1], other_span[1]) - max(span[0], other_span[0])
            # Одна колонка: левые края совпали (по ним равняется набор) и куски заметно
            # перекрываются по ширине. У верхнего куска колонки правый край бывает шире — там,
            # где сбоку ещё нет врезки (1967/10 с.63), и требовать совпадения обоих краёв нельзя.
            same_left = abs(span[0] - other_span[0]) <= mm_to_px(MERGE_EDGE_MM, dpi)
            widths = (span[1] - span[0], other_span[1] - other_span[0])
            # Ширины не должны различаться больше чем в MERGE_WIDTH_RATIO раз: иначе к колонке
            # приклеивается колонтитул во всю ширину страницы (1973/07 с.77 — кромка блока
            # уходила через всю страницу к номеру полосы).
            same_width = max(widths) <= MERGE_WIDTH_RATIO * min(widths)
            same = same_left and same_width and overlap >= MERGE_OVERLAP * min(widths)
            pitch = pitch_of(other_rows)
            gap = row_gap(other_rows[-1], rows[0])
            if same and 0 < gap <= BLOCK_GAP_PITCHES * pitch:
                out[index] = ((min(span[0], other_span[0]), max(span[1], other_span[1])), other_rows + rows)
                joined = True
                break
        if not joined:
            out.append((span, list(rows)))
    return sorted(out, key=lambda item: (item[0][0], item[1][0].y))


def _fits_any(axis: LineAxis, zones: list, gutters: list, width: int) -> bool:
    """Есть ли зона и колонка в ней, в которую ось помещается целиком."""
    for zone in zones:
        if not (zone.y0 <= axis.cy < zone.y1):
            continue
        if any(_inside(axis, span, gutters, width) for span in zone.columns):
            return True
    return False


def _axis_key(axis: LineAxis) -> tuple[int, int, int]:
    """Ключ оси для сравнения между копиями: округлённые середина и концы."""
    return (round(axis.cy), round(axis.x0), round(axis.x1))


def _inside(axis: LineAxis, span: tuple[int, int], gutters: list, width: int) -> bool:
    """Ось лежит в колонке зоны — по границам колонки НА ВЫСОТЕ ЭТОЙ ОСИ.

    Границы берутся у межколонников-ломаных (:func:`columns.bounds_at`), а не у прямоугольника
    зоны: на трапеции колонка уезжает вбок, и прямая граница либо режет её, либо цепляет чужое
    (номер полосы, уехавший под блок).
    """
    left, right = bounds_at(gutters, axis.x0, axis.x1, axis.cy, width)
    # Колонка зоны и локальные границы должны быть той же колонкой: сравниваем по перекрытию.
    if min(right, span[1]) - max(left, span[0]) < 0.5 * (span[1] - span[0]):
        return False
    # Строка принадлежит колонке, если лежит в ней ОСНОВНОЙ своей частью: требовать «целиком
    # внутри» нельзя — у последней строки абзаца выносной элемент вылезает за кромку, и строка
    # уходила из колонки в блок-остаток (1970/02 с.90, «справочниках»).
    pad = 0.05 * (right - left)
    inside = min(axis.x1, right + pad) - max(axis.x0, left - pad)
    return inside >= INSIDE_SHARE * max(1.0, axis.x1 - axis.x0)


__all__ = [
    "row_gap",
    "BlockEnvelope",
    "Row",
    "TextBlock",
    "blocks_of",
    "envelope_of",
    "ink_edge",
    "pitch_of",
    "rows_of",
]
