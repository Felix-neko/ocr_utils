"""Кромки колонок: наклон левого и правого края выключенного текста.

Абзацные отступы и висячие строки шумят сильнее наклона (это уже выяснено в
``curved_lines``: край колонки как признак кривизны отвергнут). Здесь задача проще —
сравнить один и тот же край на двух версиях страницы, и шум одинаков в обеих. Край берётся
только по строкам, которые до него ДОХОДЯТ (начало в пределах полувысоты строки от общего
левого края колонки), наклон — медиана попарных наклонов (Тейл–Сен), чтобы одна ошибочная
строка не тянула прямую.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocr_utils.geometry_regression import WORK_DPI, mm_to_px, px_to_mm
from ocr_utils.geometry_regression.regions import TextLine

# Строк у кромки меньше — наклон не считается.
MIN_EDGE_LINES = 8
# Колонка режется на блоки по вертикальному промежутку больше стольких высот строки: наклон
# верхнего блока (1967/02 с.57) тонет в кромке всей полосы, если мерить её целиком.
BLOCK_GAP_HEIGHTS = 3.0
# Сопоставление кромок B и A: та же сторона и перекрытие по вертикали не меньше этой доли
# длинной кромки. По номеру колонки нельзя: сегментация колонок в A и B расходится, и кромке
# всей колонки в B доставалась кромка её нижней трети в A (1967/04 с.9).
MATCH_OVERLAP = 0.7
# Допуск «строка доходит до кромки» в долях высоты строки.
EDGE_TOL_HEIGHTS = 0.5
# Колонка — промежуток между межколонниками; уже 20 мм не рассматривается.
MIN_COLUMN_MM = 20.0
# Полуширина рамки кромки на оверлее.
EDGE_BOX_MM = 1.7


@dataclass(frozen=True)
class Edge:
    side: str  # "left" | "right"
    column: int
    tilt_deg: float  # наклон кромки от вертикали, со знаком
    lines: int
    box: tuple[int, int, int, int]


def _theil_sen(xs: np.ndarray, ys: np.ndarray) -> float:
    """Медианный наклон dx/dy по всем парам точек."""
    n = len(xs)
    i, j = np.triu_indices(n, 1)
    dy = ys[j] - ys[i]
    ok = np.abs(dy) > 1e-6
    return float(np.median((xs[j] - xs[i])[ok] / dy[ok])) if ok.any() else 0.0


def column_edges(
    lines: list[TextLine], separators: list[tuple[int, int]], width: int, dpi: float = WORK_DPI
) -> list[Edge]:
    """Кромки всех колонок страницы."""
    min_column = mm_to_px(MIN_COLUMN_MM, dpi)
    half = mm_to_px(EDGE_BOX_MM, dpi)
    bounds = [0] + [x for pair in separators for x in pair] + [width]
    columns = [(bounds[i], bounds[i + 1]) for i in range(0, len(bounds), 2) if bounds[i + 1] - bounds[i] >= min_column]
    edges: list[Edge] = []
    for index, (x0, x1) in enumerate(columns):
        column = sorted((line for line in lines if x0 <= line.cx < x1), key=lambda line: line.cy)
        for own in _blocks(column):
            edges.extend(_block_edges(own, index, half))
    return edges


def _blocks(column: list[TextLine]) -> list[list[TextLine]]:
    """Колонка → блоки строк, разделённые промежутком больше ``BLOCK_GAP_HEIGHTS`` высот."""
    blocks: list[list[TextLine]] = []
    for line in column:
        if blocks and line.y0 - blocks[-1][-1].y1 <= BLOCK_GAP_HEIGHTS * line.height:
            blocks[-1].append(line)
        else:
            blocks.append([line])
    return [block for block in blocks if len(block) >= MIN_EDGE_LINES]


def _block_edges(own: list[TextLine], index: int, half: int) -> list[Edge]:
    """Левая и правая кромки одного блока строк по строкам, доходящим до края."""
    edges: list[Edge] = []
    height = float(np.median([line.height for line in own]))
    starts = np.array([line.x0 for line in own], dtype=np.float64)
    ends = np.array([line.x1 for line in own], dtype=np.float64)
    ys = np.array([line.cy for line in own], dtype=np.float64)
    for side, values, limit in (
        ("left", starts, np.percentile(starts, 15) + EDGE_TOL_HEIGHTS * height),
        ("right", ends, np.percentile(ends, 85) - EDGE_TOL_HEIGHTS * height),
    ):
        keep = values <= limit if side == "left" else values >= limit
        if keep.sum() < MIN_EDGE_LINES:
            continue
        slope = _theil_sen(values[keep], ys[keep])
        x_edge = float(np.median(values[keep]))
        box = (int(x_edge - half), int(ys[keep].min()), int(x_edge + half), int(ys[keep].max()))
        edges.append(Edge(side, index, float(np.degrees(np.arctan(slope))), int(keep.sum()), box))
    return edges


def _overlap(b: Edge, a: Edge) -> float:
    top, bottom = max(b.box[1], a.box[1]), min(b.box[3], a.box[3])
    longest = max(b.box[3] - b.box[1], a.box[3] - a.box[1], 1)
    return max(0, bottom - top) / longest


def edge_metrics(before: list[Edge], after: list[Edge], dpi: float = WORK_DPI) -> tuple[dict[str, float], dict]:
    """Худшая разность ухода кромки от вертикали (мм, длина × sin наклона) по сопоставленным кромкам."""
    pairs: list[tuple[Edge, Edge]] = []
    taken: set[int] = set()
    for edge in before:
        candidates = [
            (j, _overlap(edge, other)) for j, other in enumerate(after) if other.side == edge.side and j not in taken
        ]
        candidates = [(j, o) for j, o in candidates if o >= MATCH_OVERLAP]
        if candidates:
            j = max(candidates, key=lambda c: c[1])[0]
            taken.add(j)
            pairs.append((edge, after[j]))
    metrics = {"edges_matched": float(len(pairs))}
    culprits: dict = {}
    if not pairs:
        metrics["edge_dev_max_delta_mm"] = 0.0
        metrics["edge_gain_mm"] = 0.0
        return metrics, culprits
    deltas = [
        px_to_mm(b.box[3] - b.box[1], dpi) * (np.sin(np.radians(abs(a.tilt_deg))) - np.sin(np.radians(abs(b.tilt_deg))))
        for b, a in pairs
    ]
    worst = int(np.argmax(deltas))
    metrics["edge_dev_max_delta_mm"] = float(deltas[worst])
    metrics["edge_gain_mm"] = float(max(0.0, -min(deltas)))
    metrics["edge_tilt_max_b"] = float(max(abs(b.tilt_deg) for b, _ in pairs))
    metrics["edge_tilt_max_a"] = float(max(abs(a.tilt_deg) for _, a in pairs))
    culprits["edge_dev_max_delta_mm"] = {"b": pairs[worst][0].box, "a": pairs[worst][1].box}
    return metrics, culprits
