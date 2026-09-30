"""Блоки текста и их кромки по краске рядов «было | стало»: перекос и рваность.

Ядро (``edges.py``) подбирало строки кромки в A и B независимо и сопоставляло кромки по
перекрытию — на перекошенном блоке (1969/12 с.34: левый край ушёл на 5 мм при горизонтальных
строках) оно видело 0.2–1.2 мм. Первая версия стенда строила кромку по боксам строк и их парам
(``lines.match_lines``), и её портили разрезанные строки: у выключенного текста строка часто
распадается на два сегмента по широкому пробелу, а одиночная первая буква «в»/«и» отваливается от
строки — бокс начинается со второго слова; обрезки вставали в последовательность как отдельные
«строки», окно огибающей видело вместо соседей обрезки, и строка с отступом попадала в кромку
(1966/03 с.13, 1967/11 с.75 — кружки в отступах и внутри блока).

Здесь кромка меряется ПО КРАСКЕ: сегменты одной колонки с близкими центрами по y сливаются в
РЯД (полоса ``[y0, y1]``), и край ряда — первый (последний) столбец рендера 300 dpi в пределах
колонки, где в полосе ряда есть краска, с проверкой, что краска есть и в соседних столбцах
(пыль не край). В A полоса ряда переносится полем смещений и край меряется там же — партнёр из
``match_lines`` не нужен. Огибающая — «липкая лента» пользователя в одномерном виде:
последовательность краёв рядов сверху вниз сглаживается морфологическим открытием (закрытием)
окном ``ENVELOPE_LINES`` рядов — абзацный отступ и короткая последняя строка абзаца заливаются,
вырез под иллюстрацию в 3+ ряда остаётся ямой. Блок выключенный, если в B на кромке не меньше
``MIN_EDGE_LINES`` рядов и p80 их остатка от прямой не больше ``MAX_ROUGH_B_MM``.

Перекос кромки — её наклон от вертикали МИНУС наклон строк блока (шир, не поворот): доворот
всей страницы, при котором кромка и строки поворачиваются вместе, порчей не считается.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import grey_closing, grey_opening

from ocr_utils.geometry_regression import mm_to_px, px_to_mm
from ocr_utils.geometry_regression.edges import BLOCK_GAP_HEIGHTS, _theil_sen
from ocr_utils.geometry_regression.field import Field
from ocr_utils.geometry_regression.regions import TextLine, column_spans
from ocr_utils.geometry_regression.render import RENDER_DPI

Box = tuple[int, int, int, int]

# Рядов на кромке меньше — блок не мерится.
MIN_EDGE_LINES = 8
# Окно огибающей (рядов): открытие/закрытие заливает ямы короче окна − 1.
ENVELOPE_LINES = 3
# Ряд «на кромке», если его край не дальше этой доли высоты строки от огибающей.
EDGE_TOL_HEIGHTS = 0.5
# Кромка в B рваная сильнее — блок не выключенный (оглавление, стихи, список), не мерится.
MAX_ROUGH_B_MM = 1.5
# Перцентиль остатка для рваности: один выпавший ряд (переносный дефис) её не задирает.
ROUGH_PERCENTILE = 80.0
# Полуширина рамки кромки на оверлее.
EDGE_BOX_MM = 1.7
# Сегменты с центрами ближе этой доли высоты — один ряд.
ROW_TOL_HEIGHTS = 0.5
# Окно поиска края: от границы колонки или от края сегментов ряда (что дальше наружу) с припуском
# (мм): у трапеции 1975/05 с.61 левый край колонки уходит за межколонник на 8 мм, а бокс ряда без
# первой буквы начинается на 4 мм правее видимого края.
COLUMN_PAD_MM = 5.0
# Край ряда: столбец с не меньше стольких пикселей краски в полосе ряда (300 dpi) …
EDGE_MIN_INK_PX = 2
# … и с краской в следующих (внутрь строки) стольких столбцах из этих — иначе пылинка.
EDGE_CONFIRM_COLS = 3
EDGE_CONFIRM_MIN = 2
# Полоса ряда в A — та же высота с припуском в долях высоты (поле ошибается на доли миллиметра).
ROW_PAD_HEIGHTS = 0.15


@dataclass(frozen=True)
class Row:
    """Ряд текста колонки: слитые сегменты одной y-полосы (пиксели рабочей копии)."""

    x0: int
    y0: int
    x1: int
    y1: int
    height: float
    slope_deg: float  # медианный наклон сегментов ряда
    column: int

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass(frozen=True)
class BlockEdge:
    """Одна кромка блока по тем же рядам в обеих версиях.

    Наклоны — в градусах от вертикали (положительный — низ кромки правее верха), шир —
    наклон кромки плюс наклон строк блока (у повёрнутого блока они гасят друг друга).
    """

    side: str  # "left" | "right"
    lines: int
    length_mm: float
    tilt_b_deg: float
    tilt_a_deg: float
    shear_b_deg: float
    shear_a_deg: float
    rough_b_mm: float
    rough_a_mm: float
    jitter_b_mm: float
    jitter_a_mm: float
    points_b: np.ndarray  # N × 2, (x, y) краёв рядов в B, пиксели рабочей копии
    points_a: np.ndarray

    @property
    def shear_move_mm(self) -> float:
        """На сколько мм ушёл конец кромки ОТНОСИТЕЛЬНО СТРОК: |sin(шир A) − sin(шир B)| × длина."""
        sa, sb = np.sin(np.radians(self.shear_a_deg)), np.sin(np.radians(self.shear_b_deg))
        return float(self.length_mm * abs(sa - sb))

    @property
    def shear_delta_mm(self) -> float:
        """Порча: блок перестал быть тем прямоугольником, каким был, в мм ухода конца кромки.

        Мерится ИЗМЕНЕНИЕ перекоса, а не прирост его модуля: блок, который в B клонился на 0.7°
        в одну сторону, а в A клонится на 0.7° в другую, деформирован на 1.4°, хотя модуль
        перекоса тот же — разность модулей давала на таких страницах 0.05 мм и теряла порчу,
        которую видно глазом (1968/11 с.94, 1966/01 с.11, 1971/07 с.58, 1970/02 с.70).
        Результат ограничен остаточным перекосом A: если в B блок был перекошен сильно, а в A
        перекошен слабее (пусть и в другую сторону), порча — только то, что осталось видно в A.

        Кромка, ставшая прямее В ТУ ЖЕ СТОРОНУ (FineReader убрал часть перекоса), — не порча,
        а выигрыш: см. :attr:`shear_gain_mm`.
        """
        sa, sb = np.sin(np.radians(self.shear_a_deg)), np.sin(np.radians(self.shear_b_deg))
        if abs(sa) < abs(sb) and sa * sb >= 0:
            return 0.0
        return float(self.length_mm * min(abs(sa - sb), abs(sa)))

    @property
    def shear_gain_mm(self) -> float:
        """Выигрыш: насколько перекос кромки относительно строк уменьшился по модулю (мм)."""
        sa, sb = np.sin(np.radians(self.shear_a_deg)), np.sin(np.radians(self.shear_b_deg))
        return float(max(0.0, self.length_mm * (abs(sb) - abs(sa))))

    @property
    def rough_delta_mm(self) -> float:
        return self.rough_a_mm - self.rough_b_mm


def text_rows(lines: list[TextLine]) -> list[Row]:
    """Сегменты строк → ряды: в одной колонке сегменты с центрами ближе ``ROW_TOL_HEIGHTS`` высот сливаются.

    Строки через межколонник (column −1) не участвуют. Ряд получает объединённую полосу по y,
    крайние x сегментов и медианный наклон.
    """
    rows: list[Row] = []
    for column in sorted({line.column for line in lines if line.column >= 0}):
        own = sorted((line for line in lines if line.column == column), key=lambda line: line.cy)
        groups: list[list[TextLine]] = []
        for line in own:
            if groups and abs(line.cy - np.median([g.cy for g in groups[-1]])) <= ROW_TOL_HEIGHTS * line.height:
                groups[-1].append(line)
            else:
                groups.append([line])
        for group in groups:
            rows.append(
                Row(
                    min(g.x0 for g in group),
                    min(g.y0 for g in group),
                    max(g.x1 for g in group),
                    max(g.y1 for g in group),
                    float(np.median([g.height for g in group])),
                    float(np.median([g.slope_deg for g in group])),
                    column,
                )
            )
    return sorted(rows, key=lambda row: (row.column, row.cy))


def text_blocks(rows: list[Row]) -> list[list[Row]]:
    """Ряды одной колонки → блоки по разрыву больше ``BLOCK_GAP_HEIGHTS`` высот."""
    blocks: list[list[Row]] = []
    for column in sorted({row.column for row in rows}):
        current: list[Row] = []
        for row in (r for r in rows if r.column == column):
            if current and row.y0 - current[-1].y1 > BLOCK_GAP_HEIGHTS * row.height:
                blocks.append(current)
                current = []
            current.append(row)
        if current:
            blocks.append(current)
    return [block for block in blocks if len(block) >= MIN_EDGE_LINES]


def _ink_edge(ink: np.ndarray, x_lo: int, x_hi: int, y0: int, y1: int, side: str) -> float | None:
    """Край краски в полосе ``[y0, y1)`` между столбцами ``x_lo`` и ``x_hi`` (пиксели ``ink``).

    Для левой кромки — первый столбец слева с краской, подтверждённой соседями справа; для
    правой — последний столбец с подтверждением слева. ``None`` — краски в полосе нет.
    """
    h, w = ink.shape
    x_lo, x_hi, y0, y1 = max(0, x_lo), min(w, x_hi), max(0, y0), min(h, y1)
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


def _row_edge_b(ink_b: np.ndarray, row: Row, column: tuple[int, int], side: str, dpi: float) -> float | None:
    """Край ряда в B (пиксели рабочей копии) по краске рендера ``ink_b``."""
    k = RENDER_DPI / dpi
    pad = mm_to_px(COLUMN_PAD_MM, dpi)
    x_lo, x_hi = int((min(column[0], row.x0) - pad) * k), int((max(column[1], row.x1) + pad) * k)
    x = _ink_edge(ink_b, x_lo, x_hi, int(row.y0 * k), int(row.y1 * k) + 1, side)
    return None if x is None else x / k


def _row_edge_a(
    ink_a: np.ndarray, row: Row, column: tuple[int, int], side: str, field: Field | None, dpi: float
) -> tuple[float, float] | None:
    """Край и центр ряда в A: полоса ряда переносится полем, край ищется в той же колонке."""
    k = RENDER_DPI / dpi
    pad = mm_to_px(COLUMN_PAD_MM, dpi)
    c0, c1 = min(column[0], row.x0), max(column[1], row.x1)
    corners = np.array([[c0, row.y0], [c1, row.y1], [c0, row.y1], [c1, row.y0]], float)
    moved = field.transform(corners) if field is not None else corners
    margin = ROW_PAD_HEIGHTS * row.height
    y0, y1 = moved[:, 1].min() - margin, moved[:, 1].max() + margin
    x_lo, x_hi = moved[:, 0].min() - pad, moved[:, 0].max() + pad
    x = _ink_edge(ink_a, int(x_lo * k), int(x_hi * k), int(y0 * k), int(y1 * k) + 1, side)
    if x is None:
        return None
    return x / k, float((y0 + y1) / 2.0)


def _fit(xs: np.ndarray, ys: np.ndarray) -> tuple[float, np.ndarray]:
    """Прямая x = s·y + c по Тейлу–Сену: наклон dx/dy и остатки."""
    slope = _theil_sen(xs, ys)
    intercept = float(np.median(xs - slope * ys))
    return slope, xs - (slope * ys + intercept)


def block_edge(
    block: list[Row],
    column: tuple[int, int],
    side: str,
    ink_b: np.ndarray,
    ink_a: np.ndarray,
    field: Field | None,
    dpi: float,
    slopes_a: dict[int, float] | None = None,
) -> BlockEdge | None:
    """Кромка блока по краске рядов, доходящих до огибающей в B, и тех же рядов в A.

    Args:
        block: Ряды блока в B (одна колонка, сверху вниз).
        column: Границы колонки ``(x0, x1)`` в пикселях рабочей копии.
        side: ``left`` или ``right``.
        ink_b, ink_a: Краска рендеров ``RENDER_DPI`` обеих версий (``255 − серый``).
        field: Поле смещений B → A (пиксели ``dpi``) или ``None``.
        dpi: Разрешение рабочей копии, в которой заданы ряды и колонка.
        slopes_a: Наклон строк в A по рядам (``id(row) → градусы``); без него наклон строк
            A берётся равным наклону B плюс поворот поля.

    Returns:
        :class:`BlockEdge` или ``None``, если рядов на кромке мало или кромка в B рваная.
    """
    height = float(np.median([row.height for row in block]))
    edges_b = [_row_edge_b(ink_b, row, column, side, dpi) for row in block]
    found = [i for i, x in enumerate(edges_b) if x is not None]
    if len(found) < MIN_EDGE_LINES:
        return None
    xs = np.array([edges_b[i] for i in found], dtype=np.float64)
    envelope = grey_opening(xs, size=ENVELOPE_LINES) if side == "left" else grey_closing(xs, size=ENVELOPE_LINES)
    on_edge = np.abs(xs - envelope) <= EDGE_TOL_HEIGHTS * height
    chosen: list[tuple[Row, float, tuple[float, float]]] = []
    for i, ok in zip(found, on_edge):
        if not ok:
            continue
        edge_a = _row_edge_a(ink_a, block[i], column, side, field, dpi)
        if edge_a is not None:
            chosen.append((block[i], edges_b[i], edge_a))
    if len(chosen) < MIN_EDGE_LINES:
        return None
    xb = np.array([x for _, x, _ in chosen])
    yb = np.array([row.cy for row, _, _ in chosen])
    xa = np.array([a[0] for _, _, a in chosen])
    ya = np.array([a[1] for _, _, a in chosen])
    slope_b, resid_b = _fit(xb, yb)
    slope_a, resid_a = _fit(xa, ya)
    rough_b = px_to_mm(float(np.percentile(np.abs(resid_b), ROUGH_PERCENTILE)), dpi)
    if rough_b > MAX_ROUGH_B_MM:
        return None
    rough_a = px_to_mm(float(np.percentile(np.abs(resid_a), ROUGH_PERCENTILE)), dpi)
    # Шир: наклон кромки от вертикали (dx/dy) плюс медианный наклон строк (dy/dx) — у чистого
    # поворота они противоположны по знаку и сумма ≈ 0.
    line_tilt_b = float(np.median([row.slope_deg for row, _, _ in chosen]))
    rot = field.rot_deg if field is not None else 0.0
    line_tilt_a = (
        float(np.median([slopes_a.get(id(row), row.slope_deg + rot) for row, _, _ in chosen]))
        if slopes_a
        else line_tilt_b + rot
    )
    tilt_b = float(np.degrees(np.arctan(slope_b)))
    tilt_a = float(np.degrees(np.arctan(slope_a)))
    return BlockEdge(
        side,
        len(chosen),
        px_to_mm(float(yb.max() - yb.min()), dpi),
        tilt_b,
        tilt_a,
        tilt_b + line_tilt_b,
        tilt_a + line_tilt_a,
        rough_b,
        rough_a,
        px_to_mm(float(np.median(np.abs(np.diff(xb)))), dpi),
        px_to_mm(float(np.median(np.abs(np.diff(xa)))), dpi),
        np.column_stack([xb, yb]),
        np.column_stack([xa, ya]),
    )


def _blank_boxes(ink: np.ndarray, boxes: list[Box], k: float) -> np.ndarray:
    """Копия краски с погашенными рамками (пиксели ``dpi`` × ``k``): линейка у поля — не край текста."""
    if not boxes:
        return ink
    out = ink.copy()
    h, w = out.shape
    for x0, y0, x1, y1 in boxes:
        out[max(0, int(y0 * k)) : min(h, int(y1 * k) + 1), max(0, int(x0 * k)) : min(w, int(x1 * k) + 1)] = 0
    return out


def block_edges(
    lines_b: list[TextLine],
    separators: list[tuple[int, int]],
    width: int,
    gray300_b: np.ndarray,
    gray300_a: np.ndarray,
    field: Field | None,
    dpi: float,
    slopes_a: dict[int, float] | None = None,
    exclude: list[Box] | None = None,
) -> list[BlockEdge]:
    """Все кромки выключенных блоков страницы по краске рядов.

    Args:
        lines_b: Строки B (сегменты ``text_lines``) вне рисунков, таблиц и растра.
        separators: Межколонники B в пикселях ``dpi`` (из ``text_lines``).
        width: Ширина рабочей копии B.
        gray300_b, gray300_a: Серые рендеры ``RENDER_DPI``.
        field: Поле смещений B → A.
        dpi: Разрешение рабочей копии.
        slopes_a: Наклон строк A по рядам (см. :func:`block_edge`).
        exclude: Рамки рисунков, таблиц и растра на B (пиксели ``dpi``): их краска в край не идёт —
            вертикальная линейка у поля (1975/05 с.61) иначе становится «краем» первых рядов.
    """
    k = RENDER_DPI / dpi
    exclude = list(exclude or [])
    ink_b = _blank_boxes(255 - gray300_b, exclude, k)
    exclude_a = exclude
    if field is not None and exclude:
        exclude_a = []
        for x0, y0, x1, y1 in exclude:
            corners = field.transform(np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], float))
            exclude_a.append(
                (int(corners[:, 0].min()), int(corners[:, 1].min()), int(corners[:, 0].max()), int(corners[:, 1].max()))
            )
    ink_a = _blank_boxes(255 - gray300_a, exclude_a, k)
    columns = column_spans(separators, width, dpi)
    edges: list[BlockEdge] = []
    for block in text_blocks(text_rows(lines_b)):
        column = columns[block[0].column] if block[0].column < len(columns) else (0, width)
        for side in ("left", "right"):
            edge = block_edge(block, column, side, ink_b, ink_a, field, dpi, slopes_a)
            if edge is not None:
                edges.append(edge)
    return edges


def _edge_box(points: np.ndarray, half: int) -> Box:
    x = float(np.median(points[:, 0]))
    return (int(x - half), int(points[:, 1].min()), int(x + half), int(points[:, 1].max()))


def _polyline(points: np.ndarray) -> list[list[float]]:
    return [[*points[i], *points[i + 1]] for i in range(len(points) - 1)]


def edge_metrics(edges: list[BlockEdge], dpi: float) -> tuple[dict[str, float], dict]:
    """Худшие разности по кромкам: перекос (мм), рваность (мм), дрожание; выигрыш по перекосу.

    Returns:
        Метрики ``edges_matched``, ``edge_shear_delta_mm``, ``edge_shear_gain_mm``, ``edge_shear_move_mm``,
        ``edge_rough_delta_mm``, ``edge_jitter_delta_mm``, ``edge_shear_max_a_deg`` и виновники
        (рамка кромки и ряд точек краёв — на оверлее кружки и прямая по ним).
    """
    metrics = {
        "edges_matched": float(len(edges)),
        "edge_shear_delta_mm": 0.0,
        "edge_shear_gain_mm": 0.0,
        "edge_shear_move_mm": 0.0,
        "edge_rough_delta_mm": 0.0,
        "edge_jitter_delta_mm": 0.0,
        "edge_shear_max_a_deg": 0.0,
    }
    culprits: dict = {}
    if not edges:
        return metrics, culprits
    half = mm_to_px(EDGE_BOX_MM, dpi)
    shears = [edge.shear_delta_mm for edge in edges]
    worst = int(np.argmax(shears))
    metrics["edge_shear_delta_mm"] = float(shears[worst])
    metrics["edge_shear_gain_mm"] = float(max(edge.shear_gain_mm for edge in edges))
    metrics["edge_shear_move_mm"] = float(max(edge.shear_move_mm for edge in edges))
    metrics["edge_shear_max_a_deg"] = float(max(abs(edge.shear_a_deg) for edge in edges))
    roughs = [edge.rough_delta_mm for edge in edges]
    rough_worst = int(np.argmax(roughs))
    metrics["edge_rough_delta_mm"] = float(roughs[rough_worst])
    metrics["edge_jitter_delta_mm"] = float(max(edge.jitter_a_mm - edge.jitter_b_mm for edge in edges))
    for name, index in (("edge_shear_delta_mm", worst), ("edge_rough_delta_mm", rough_worst)):
        edge = edges[index]
        culprits[name] = {
            "b": _edge_box(edge.points_b, half),
            "a": _edge_box(edge.points_a, half),
            "segments_b": _polyline(edge.points_b),
            "segments_a": _polyline(edge.points_a),
        }
    return metrics, culprits


__all__ = ["Row", "BlockEdge", "text_rows", "text_blocks", "block_edge", "block_edges", "edge_metrics"]
