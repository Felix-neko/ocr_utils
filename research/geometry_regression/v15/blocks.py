"""Блоки текста и их кромки по одним и тем же строкам «было | стало»: перекос и рваность.

Ядро (``edges.py``) подбирало строки кромки в A и B независимо и сопоставляло кромки по
перекрытию — на перекошенном блоке (1969/12 с.34: левый край ушёл на 5 мм при горизонтальных
строках) оно видело 0.2–1.2 мм. Здесь кромка строится по ПАРАМ строк (``lines.match_lines``):
в B выбираются строки, доходящие до огибающей блока, и те же строки берутся в A. Огибающая —
«липкая лента» пользователя в одномерном виде: последовательность начал (концов) строк сверху
вниз сглаживается морфологическим открытием (закрытием) окном ``ENVELOPE_LINES`` строк —
абзацный отступ и короткая последняя строка абзаца (1–2 строки) заливаются, вырез под
иллюстрацию в 3+ строки остаётся ямой (там кромки нет). Блок считается выключенным, если в B
на кромке не меньше ``MIN_EDGE_LINES`` строк и p80 их остатка от прямой не больше
``MAX_ROUGH_B_MM`` (строки на кромке отбираются с допуском полвысоты, ~1.5 мм, и p80 их остатка у
выключенного блока 1.0–1.4 мм; у оглавления 1967/07 с.97 — 5.3 мм);
оглавления, стихи и списки так отсекаются.

Перекос кромки — её наклон от вертикали МИНУС наклон строк блока (шир, не поворот): доворот
всей страницы, при котором кромка и строки поворачиваются вместе, порчей не считается.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import grey_closing, grey_opening

from ocr_utils.geometry_regression import mm_to_px, px_to_mm
from ocr_utils.geometry_regression.edges import BLOCK_GAP_HEIGHTS, _theil_sen
from ocr_utils.geometry_regression.regions import TextLine

Box = tuple[int, int, int, int]

# Строк на кромке меньше — блок не мерится.
MIN_EDGE_LINES = 8
# Окно огибающей (строк): открытие/закрытие заливает ямы короче окна − 1.
ENVELOPE_LINES = 3
# Строка «на кромке», если её край не дальше этой доли высоты строки от огибающей.
EDGE_TOL_HEIGHTS = 0.5
# Кромка в B рваная сильнее — блок не выключенный (оглавление, стихи, список), не мерится.
MAX_ROUGH_B_MM = 1.5
# Перцентиль остатка для рваности: одна выпавшая строка (переносный дефис) её не задирает.
ROUGH_PERCENTILE = 80.0
# Полуширина рамки кромки на оверлее.
EDGE_BOX_MM = 1.7


@dataclass(frozen=True)
class BlockEdge:
    """Одна кромка блока по тем же строкам в обеих версиях.

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
    points_b: np.ndarray  # N × 2, (x, y) краёв строк в B
    points_a: np.ndarray

    @property
    def shear_delta_mm(self) -> float:
        """Уход конца кромки от вертикали в мм: стало − было (относительно строк)."""
        return self.length_mm * (abs(np.sin(np.radians(self.shear_a_deg))) - abs(np.sin(np.radians(self.shear_b_deg))))

    @property
    def rough_delta_mm(self) -> float:
        return self.rough_a_mm - self.rough_b_mm


def text_blocks(lines: list[TextLine]) -> list[list[TextLine]]:
    """Строки одной колонки → блоки по разрыву больше ``BLOCK_GAP_HEIGHTS`` высот; строки через межколонник (column −1) не участвуют."""
    blocks: list[list[TextLine]] = []
    for column in sorted({line.column for line in lines if line.column >= 0}):
        own = sorted((line for line in lines if line.column == column), key=lambda line: line.cy)
        current: list[TextLine] = []
        for line in own:
            if current and line.y0 - current[-1].y1 > BLOCK_GAP_HEIGHTS * line.height:
                blocks.append(current)
                current = []
            current.append(line)
        if current:
            blocks.append(current)
    return [block for block in blocks if len(block) >= MIN_EDGE_LINES]


def _fit(xs: np.ndarray, ys: np.ndarray) -> tuple[float, np.ndarray]:
    """Прямая x = s·y + c по Тейлу–Сену: наклон dx/dy и остатки."""
    slope = _theil_sen(xs, ys)
    intercept = float(np.median(xs - slope * ys))
    return slope, xs - (slope * ys + intercept)


def block_edge(block: list[TextLine], partner: dict[int, TextLine], side: str, dpi: float) -> BlockEdge | None:
    """Кромка блока по строкам, доходящим до огибающей в B, и их парам в A.

    Args:
        block: Строки блока в B (одна колонка, сверху вниз).
        partner: ``id(строка B) → строка A`` из ``lines.match_lines``.
        side: ``left`` (начала строк) или ``right`` (концы).
        dpi: Разрешение копии, в которой заданы строки.

    Returns:
        :class:`BlockEdge` или ``None``, если строк на кромке мало или кромка в B рваная.
    """
    height = float(np.median([line.height for line in block]))
    xs = np.array([line.x0 if side == "left" else line.x1 for line in block], dtype=np.float64)
    envelope = grey_opening(xs, size=ENVELOPE_LINES) if side == "left" else grey_closing(xs, size=ENVELOPE_LINES)
    on_edge = np.abs(xs - envelope) <= EDGE_TOL_HEIGHTS * height
    chosen = [line for line, ok in zip(block, on_edge) if ok and id(line) in partner]
    if len(chosen) < MIN_EDGE_LINES:
        return None
    xb = np.array([line.x0 if side == "left" else line.x1 for line in chosen], dtype=np.float64)
    yb = np.array([line.cy for line in chosen], dtype=np.float64)
    xa = np.array([partner[id(line)].x0 if side == "left" else partner[id(line)].x1 for line in chosen])
    ya = np.array([partner[id(line)].cy for line in chosen], dtype=np.float64)
    slope_b, resid_b = _fit(xb, yb)
    slope_a, resid_a = _fit(xa.astype(np.float64), ya)
    rough_b = px_to_mm(float(np.percentile(np.abs(resid_b), ROUGH_PERCENTILE)), dpi)
    if rough_b > MAX_ROUGH_B_MM:
        return None
    rough_a = px_to_mm(float(np.percentile(np.abs(resid_a), ROUGH_PERCENTILE)), dpi)
    # Шир: наклон кромки от вертикали (dx/dy) плюс медианный наклон строк (dy/dx) — у чистого
    # поворота они противоположны по знаку и сумма ≈ 0.
    line_tilt_b = float(np.median([line.slope_deg for line in chosen]))
    line_tilt_a = float(np.median([partner[id(line)].slope_deg for line in chosen]))
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


def block_edges(lines_b: list[TextLine], pairs: list[tuple[TextLine, TextLine]], dpi: float) -> list[BlockEdge]:
    """Все кромки выключенных блоков страницы по парам строк."""
    partner = {id(b): a for b, a in pairs}
    edges: list[BlockEdge] = []
    for block in text_blocks(lines_b):
        for side in ("left", "right"):
            edge = block_edge(block, partner, side, dpi)
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
        Метрики ``edges_matched``, ``edge_shear_delta_mm``, ``edge_shear_gain_mm``,
        ``edge_rough_delta_mm``, ``edge_jitter_delta_mm``, ``edge_shear_max_a_deg`` и виновники
        (рамка кромки и ломаная краёв строк в A — на оверлей).
    """
    metrics = {
        "edges_matched": float(len(edges)),
        "edge_shear_delta_mm": 0.0,
        "edge_shear_gain_mm": 0.0,
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
    metrics["edge_shear_gain_mm"] = float(max(0.0, -min(shears)))
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


__all__ = ["BlockEdge", "text_blocks", "block_edge", "block_edges", "edge_metrics"]
