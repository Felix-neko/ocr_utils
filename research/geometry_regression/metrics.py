"""Все измерения одной пары страниц в один плоский словарь.

Порядок: поле смещений → области на B → линейки, строки, кромки на обеих версиях →
сопоставление через поле → разности. Сырьё (тайлы поля, рамки-виновники) возвращается
отдельно: оно идёт в JSON-кэш и на оверлей, но не в CSV.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from research.geometry_regression import WORK_DPI
from research.geometry_regression.edges import column_edges, edge_metrics
from research.geometry_regression.field import estimate_field, field_metrics
from research.geometry_regression.lines import line_metrics, match_lines
from research.geometry_regression.regions import lineart_boxes, text_boxes, text_lines
from research.geometry_regression.render import RENDER_DPI, to_work
from research.geometry_regression.strokes import find_strokes, match_strokes, stroke_metrics


@dataclass(frozen=True)
class Params:
    """Размеры, привязанные к бумаге. Умолчания — для пака-1 (журнальная полоса ~170×260 мм)."""

    dpi: float = WORK_DPI
    # Штрих короче не считается прямым штрихом: дробная черта формулы ≈ 13-25 мм, линейки
    # таблиц от 30 мм, а у текста длинных прямых кромок нет.
    stroke_min_mm: float = 8.0
    # Строка короче в попарные метрики не идёт: наклон короткой строки шумит.
    line_min_mm: float = 25.0


@dataclass
class PageMeasure:
    metrics: dict[str, float] = field(default_factory=dict)
    culprits: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


def _boxes(items) -> list[list[int]]:
    return [[int(v) for v in box] for box in items]


def _scale_culprits(culprits: dict, k: float) -> dict:
    """Рамки-виновники из другого разрешения — в пиксели рабочей копии."""
    return {
        name: {side: tuple(int(round(v * k)) for v in box) for side, box in pair.items()}
        for name, pair in culprits.items()
    }


def measure_pair(gray300_b: np.ndarray, gray300_a: np.ndarray, params: Params = Params()) -> PageMeasure:
    """Метрики пары «без коррекции (B) | с коррекцией (A)» по рендерам 300 dpi."""
    dpi = params.dpi
    b = to_work(gray300_b)
    a = to_work(gray300_a)
    out = PageMeasure()

    warp = estimate_field(b, a, dpi)
    lineart = lineart_boxes(b, dpi)
    lines_b, separators_b = text_lines(gray300_b, dpi)
    lines_a, separators_a = text_lines(gray300_a, dpi)
    out.metrics.update(field_metrics(warp, lineart, text_boxes(lines_b)))
    out.metrics["lineart_boxes"] = float(len(lineart))

    strokes_b = find_strokes(gray300_b, params.stroke_min_mm, RENDER_DPI)
    strokes_a = find_strokes(gray300_a, params.stroke_min_mm, RENDER_DPI)
    pairs = match_strokes(strokes_b, strokes_a, warp, RENDER_DPI, dpi)
    metrics, culprits = stroke_metrics(strokes_b, strokes_a, pairs, warp.rot_deg if warp else 0.0, RENDER_DPI)
    out.metrics.update(metrics)
    out.culprits.update(_scale_culprits(culprits, dpi / RENDER_DPI))

    metrics, culprits = line_metrics(
        lines_b, lines_a, match_lines(lines_b, lines_a, warp, dpi), params.line_min_mm, dpi
    )
    out.metrics.update(metrics)
    out.culprits.update(culprits)

    edges_b = column_edges(lines_b, separators_b, b.shape[1], dpi)
    edges_a = column_edges(lines_a, separators_a, a.shape[1], dpi)
    metrics, culprits = edge_metrics(edges_b, edges_a, dpi)
    out.metrics.update(metrics)
    out.culprits.update(culprits)

    out.raw = {
        "size_b": [int(b.shape[1]), int(b.shape[0])],
        "size_a": [int(a.shape[1]), int(a.shape[0])],
        "lineart": _boxes(lineart),
        "field": (
            None
            if warp is None
            else {
                "tiles": np.round(warp.tiles, 2).tolist(),
                "weight": np.round(warp.weight, 3).tolist(),
                "affine": np.round(warp.affine, 6).tolist(),
            }
        ),
    }
    return out
