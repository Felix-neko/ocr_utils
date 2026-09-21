"""Все измерения одной пары страниц в один плоский словарь.

Порядок: поле смещений → области на B → линейки, строки, кромки на обеих версиях →
сопоставление через поле → разности. Сырьё (тайлы поля, рамки-виновники) возвращается
отдельно: оно идёт в JSON-кэш и на оверлей, но не в CSV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ocr_utils.geometry_regression import WORK_DPI
from ocr_utils.geometry_regression.bend import bend_metrics
from ocr_utils.geometry_regression.edges import column_edges, edge_metrics
from ocr_utils.geometry_regression.field import estimate_field, field_metrics
from ocr_utils.geometry_regression.lines import line_metrics, match_lines
from ocr_utils.geometry_regression.regions import lineart_boxes, text_boxes, text_lines
from ocr_utils.geometry_regression.render import RENDER_DPI, to_work
from ocr_utils.geometry_regression.stretch import glyph_line_metrics
from ocr_utils.geometry_regression.strokes import LOGO_TOP_FRAC, find_strokes, match_strokes, stroke_metrics


@dataclass(frozen=True)
class Params:
    """Размеры, привязанные к бумаге. Умолчания — для пака-1 (журнальная полоса ~170×260 мм)."""

    dpi: float = WORK_DPI
    # Корень кэша surya (``page_layout``): рамки таблиц и line art берутся разбором page_layout
    # по варианту ``fr_nogeo``; ``None`` — по одним пикселям (как в v12). В воркерах кэш только
    # читается, промах при ``layout_on_miss="fail"`` — ошибка страницы: кэш набивает
    # ``page_layout prefill-surya`` заранее.
    layout_cache_dir: "Path | None" = None
    layout_on_miss: str = "fail"
    # Штрих короче не считается прямым штрихом: дробные черты формул от 5 мм (1966/05 с.70),
    # линейки таблиц от 30 мм, а у текста прямых кромок такой длины почти нет.
    stroke_min_mm: float = 4.0
    # Строка короче в попарные метрики не идёт: наклон короткой строки шумит.
    line_min_mm: float = 25.0


@dataclass
class PageMeasure:
    metrics: dict[str, float] = field(default_factory=dict)
    culprits: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


def _transform_boxes(boxes, warp) -> list:
    """Рамки из кадра B в кадр A через аффинную часть поля (без поля — как есть)."""
    if warp is None or not boxes:
        return list(boxes)
    out = []
    for x0, y0, x1, y1 in boxes:
        corners = warp.transform(np.array([[x0, y0], [x1, y1]], dtype=np.float64))
        out.append(
            (
                int(corners[:, 0].min()),
                int(corners[:, 1].min()),
                int(corners[:, 0].max()) + 1,
                int(corners[:, 1].max()) + 1,
            )
        )
    return out


def _inside_any(line, boxes) -> bool:
    """Центр строки лежит в одной из рамок line art."""
    return any(x0 <= line.cx < x1 and y0 <= line.cy < y1 for x0, y0, x1, y1 in boxes)


def _boxes(items) -> list[list[int]]:
    return [[int(v) for v in box] for box in items]


def _scale_culprits(culprits: dict, k: float) -> dict:
    """Рамки-виновники (и отрезки групп) из другого разрешения — в пиксели рабочей копии."""

    def scale(value):
        if value and isinstance(value[0], (list, tuple)):
            return [[int(round(v * k)) for v in segment] for segment in value]
        return tuple(int(round(v * k)) for v in value)

    return {name: {side: scale(value) for side, value in pair.items()} for name, pair in culprits.items()}


def measure_pair(
    gray300_b: np.ndarray,
    gray300_a: np.ndarray,
    params: Params = Params(),
    lineart: list | None = None,
    tables: list | None = None,
) -> PageMeasure:
    """Метрики пары «без коррекции (B) | с коррекцией (A)» по рендерам 300 dpi.

    Args:
        gray300_b: Рендер страницы без коррекции (B), 300 dpi.
        gray300_a: Рендер страницы с коррекцией (A), 300 dpi.
        params: Размеры, привязанные к бумаге.
        lineart: Рамки line art (рисунков) на B в пикселях ``params.dpi`` (от ``page_layout``,
            см. :func:`regions.layout_regions`); ``None`` — найти по одним пикселям (запасной ход v12).
        tables: Рамки таблиц на B: внутри них меряются штрихи и изгиб линеек (как в рисунках), но
            поле смещений и строки текста считаются как у текста — содержимое таблицы текст.

    Returns:
        :class:`PageMeasure`: метрики «стало − было», виновники, сырые ряды.
    """
    dpi = params.dpi
    b = to_work(gray300_b)
    a = to_work(gray300_a)
    out = PageMeasure()

    lineart = lineart_boxes(b, dpi) if lineart is None else [tuple(int(v) for v in box) for box in lineart]
    tables = [tuple(int(v) for v in box) for box in tables or []]
    ruled = lineart + tables  # где штрихам разрешена толщина рисунка и где линейки нельзя гнуть
    warp = estimate_field(b, a, dpi, lineart)
    lines_b, separators_b = text_lines(gray300_b, dpi)
    lines_a, separators_a = text_lines(gray300_a, dpi)
    # Надписи внутри рисунков (полки шкафа на 1966/01 с.78) — не строки текста: их «выравнивание»
    # засчитывалось как выигрыш, а на деле это порча чертежа, которую ловят штрихи.
    lines_b = [line for line in lines_b if not _inside_any(line, lineart)]
    # Логотип рубрики (рисунок с центром в шапке) в долю тайлов без пары не идёт: FineReader
    # двигает его отдельно от текста (1973/07 с.67 — логотип на месте, текст ниже на 3 мм),
    # аффинный прогноз мимо, все его тайлы «без пары» — а сам логотип цел.
    top = LOGO_TOP_FRAC * b.shape[0]
    lineart_body = [box for box in lineart if (box[1] + box[3]) / 2.0 >= top]
    out.metrics.update(field_metrics(warp, lineart_body, text_boxes(lines_b)))
    out.metrics["lineart_boxes"] = float(len(lineart))
    out.metrics["table_boxes"] = float(len(tables))

    # Рамки рисунков и таблиц даны на копии B; для A они переносятся аффинной частью поля.
    ruled_a = _transform_boxes(ruled, warp)
    strokes_b = find_strokes(gray300_b, params.stroke_min_mm, RENDER_DPI, ruled, dpi)
    strokes_a = find_strokes(gray300_a, params.stroke_min_mm, RENDER_DPI, ruled_a, dpi, drop_lone=False)
    pairs = match_strokes(strokes_b, strokes_a, warp, RENDER_DPI, dpi)
    metrics, culprits = stroke_metrics(strokes_b, strokes_a, pairs, warp.rot_deg if warp else 0.0, RENDER_DPI)
    out.metrics.update(metrics)
    out.culprits.update(_scale_culprits(culprits, dpi / RENDER_DPI))
    # Изгиб длинных линий — по краске вдоль штриха B и той же линии в A, без пар LSD.
    metrics, culprits = bend_metrics(gray300_b, gray300_a, strokes_b, warp, RENDER_DPI, dpi)
    out.metrics.update(metrics)
    out.culprits.update(_scale_culprits(culprits, dpi / RENDER_DPI))

    # Пары строк подтверждаются по глифам, и только подтверждённые идут в сводки и выигрыш.
    line_pairs = match_lines(lines_b, lines_a, warp, dpi)
    metrics, culprits, verified = glyph_line_metrics(gray300_b, gray300_a, line_pairs, warp, dpi, params.line_min_mm)
    out.metrics.update(metrics)
    out.culprits.update(culprits)
    out.metrics.update(line_metrics(lines_b, lines_a, verified, dpi))

    edges_b = column_edges(lines_b, separators_b, b.shape[1], dpi)
    edges_a = column_edges(lines_a, separators_a, a.shape[1], dpi)
    metrics, culprits = edge_metrics(edges_b, edges_a, dpi)
    out.metrics.update(metrics)
    out.culprits.update(culprits)

    out.raw = {
        "size_b": [int(b.shape[1]), int(b.shape[0])],
        "size_a": [int(a.shape[1]), int(a.shape[0])],
        "lineart": _boxes(lineart),
        "tables": _boxes(tables),
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
