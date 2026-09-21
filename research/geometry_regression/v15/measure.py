"""Все измерения пары страниц v15 в один плоский словарь — из кирпичей ядра с новыми метриками.

Порядок тот же, что в ``metrics.measure_pair`` ядра: поле смещений → области на B → штрихи,
строки, кромки на обеих версиях → сопоставление через поле → разности. Отличия — в
докстринге пакета ``research.geometry_regression.v15``.
"""

from __future__ import annotations

import numpy as np

from ocr_utils.geometry_regression.field import estimate_field, field_metrics
from ocr_utils.geometry_regression.lines import line_metrics, match_lines
from ocr_utils.geometry_regression.metrics import (
    PageMeasure,
    Params,
    _boxes,
    _inside_any,
    _mid_inside,
    _scale_culprits,
    _transform_boxes,
)
from ocr_utils.geometry_regression.regions import TextLine, lineart_boxes, text_boxes, text_lines
from ocr_utils.geometry_regression.render import RENDER_DPI, to_work
from ocr_utils.geometry_regression.strokes import LOGO_TOP_FRAC, find_strokes, match_strokes, stroke_metrics
from research.geometry_regression.v15 import ENGINE_VERSION
from research.geometry_regression.v15.blocks import block_edges, edge_metrics
from research.geometry_regression.v15.field_shear import shear_metrics
from research.geometry_regression.v15.lines import LineTilt, glyph_line_metrics, tilt_summary
from research.geometry_regression.v15.raster import raster_edge_metrics
from research.geometry_regression.v15.ridge import (
    StrokeTilt,
    bend_metrics,
    lsd_fallback,
    ridge_stroke_metrics,
    trace_strokes,
)

# Доворот страницы подтверждён, если |поворот поля| не меньше этого (градусы), элементы
# (строки или штрихи) довернулись на тот же угол (разность медиан наклонов B → A совпадает с
# поворотом в допуске) и стали ближе к оси: медиана |наклона| упала не меньше чем на
# DESKEW_MIN_GAIN_DEG. Тогда элемент, ушедший от оси ровно на доворот (боковая линейка на
# 1975/06 с.39 при выпрямленной схеме), — не порча, а выигрыш — сам доворот.
DESKEW_MIN_ROT_DEG = 0.5
DESKEW_MIN_GAIN_DEG = 0.3
DESKEW_TOL_DEG = 0.35
# Элементов для оценки доворота меньше — не подтверждается.
DESKEW_MIN_ITEMS = 5
# Группы параллельных внутри рисунков — из штрихов не короче стольких мм (полки шкафа 1966/01
# с.78 — от 56 мм; рёбра столбиков и легенда диаграммы 1975/09 с.10 — 13–17 мм; формулы вне
# рисунков — от 8 мм, как в ядре).
PARALLEL_LINEART_MIN_MM = 20.0


def _wrap(angle: float) -> float:
    return (angle + 90.0) % 180.0 - 90.0


def _deskew(lines: list[LineTilt], strokes: list[StrokeTilt], rot_deg: float) -> tuple[float, float, str]:
    """Выигрыш по довороту страницы и подтверждённый доворот.

    Доворот подтверждён, когда элементы страницы (строки или штрихи, не меньше
    ``DESKEW_MIN_ITEMS``) довернулись на угол поля И хоть одно семейство стало прямее: строки
    (медиана |наклона|) либо горизонтали или вертикали (средний по длине |наклон|) — на
    ``DESKEW_MIN_GAIN_DEG``. На 1975/06 с.39 горизонтали схемы выпрямились с 1.96° до 0.19°, а
    вертикали ушли на тот же угол — это доворот, не порча.

    Args:
        lines: Наклоны пар строк по проекции (корпус и строки внутри таблиц).
        strokes: Направления штрихов B и A.
        rot_deg: Поворот аффинной части поля B → A.

    Returns:
        ``(page_deskew_gain_deg, rot_adjust_deg, источник)``: выигрыш в градусах (0, если
        не подтверждён), поправка к наклонам A (равна ``rot_deg`` или 0) и чем подтверждено.
    """
    if abs(rot_deg) < DESKEW_MIN_ROT_DEG:
        return 0.0, 0.0, ""
    axis = lambda a: _wrap(a) if abs(_wrap(a)) <= 45.0 else _wrap(a - 90.0)
    turned = []
    if len(lines) >= DESKEW_MIN_ITEMS:
        turned.append(("lines", float(np.median([t.tilt_a - t.tilt_b for t in lines]))))
    if len(strokes) >= DESKEW_MIN_ITEMS:
        turned.append(("strokes", float(np.median([axis(t.angle_a) - axis(t.angle_b) for t in strokes]))))
    sources = [name for name, turn in turned if abs(turn - rot_deg) <= DESKEW_TOL_DEG]
    if not sources:
        return 0.0, 0.0, ""
    gains = []
    if len(lines) >= 3:
        gains.append(float(np.median([abs(t.tilt_b) for t in lines]) - np.median([abs(t.tilt_a) for t in lines])))
    for horizontal in (True, False):
        family = (
            [t for t in strokes if (abs(axis(t.angle_b)) <= 45.0) == horizontal and abs(_wrap(t.angle_b)) <= 45.0]
            if horizontal
            else [t for t in strokes if abs(_wrap(t.angle_b)) > 45.0]
        )
        if len(family) >= 3:
            weights = np.array([t.stroke.length for t in family])
            before = np.array([abs(axis(t.angle_b)) for t in family])
            after = np.array([abs(axis(t.angle_a)) for t in family])
            gains.append(float((weights * (before - after)).sum() / weights.sum()))
    if gains and max(gains) >= DESKEW_MIN_GAIN_DEG:
        return abs(rot_deg), rot_deg, sources[0]
    return 0.0, 0.0, ""


def _merge_max(target: dict[str, float], source: dict[str, float], names: tuple[str, ...]) -> None:
    for name in names:
        target[name] = max(float(target.get(name, 0.0) or 0.0), float(source.get(name, 0.0) or 0.0))


def measure_pair(
    gray300_b: np.ndarray,
    gray300_a: np.ndarray,
    params: Params = Params(),
    lineart: list | None = None,
    tables: list | None = None,
    raster: list | None = None,
) -> PageMeasure:
    """Метрики пары «без коррекции (B) | с коррекцией (A)» v15 по рендерам 300 dpi.

    Args:
        gray300_b, gray300_a: Рендеры обеих версий, 300 dpi.
        params: Размеры, привязанные к бумаге (те же, что у ядра).
        lineart, tables, raster: Рамки рисунков, таблиц и растра на B в пикселях ``params.dpi``
            (см. ``regions.layout_regions``); ``lineart=None`` — найти по одним пикселям.

    Returns:
        :class:`PageMeasure`: метрики «стало − было», виновники (пиксели ``params.dpi``), сырьё.
    """
    dpi = params.dpi
    b, a = to_work(gray300_b), to_work(gray300_a)
    out = PageMeasure()
    lineart = lineart_boxes(b, dpi) if lineart is None else [tuple(int(v) for v in box) for box in lineart]
    tables = [tuple(int(v) for v in box) for box in tables or []]
    raster = [tuple(int(v) for v in box) for box in raster or []]
    warp = estimate_field(b, a, dpi, lineart + raster)
    rot_deg = warp.rot_deg if warp is not None else 0.0

    lines_b, separators_b = text_lines(gray300_b, dpi)
    lines_a, _ = text_lines(gray300_a, dpi)
    # Корпус — строки вне рисунков, таблиц и растра: и порча, и выигрыш. Строки внутри таблиц и
    # рисунков — только выигрыш (их «наклон» и «волна» — не порча, а вот выпрямление — довод за
    # коррекцию: 1973/08 с.18, 1975/08 с.39). Строки в растре не строки.
    body_b = [line for line in lines_b if not _inside_any(line, lineart + tables + raster)]
    inner_b = [line for line in lines_b if _inside_any(line, lineart + tables) and not _inside_any(line, raster)]
    top = LOGO_TOP_FRAC * b.shape[0]
    lineart_body = [box for box in lineart if (box[1] + box[3]) / 2.0 >= top]
    out.metrics.update(field_metrics(warp, lineart_body, text_boxes(body_b), raster))
    out.metrics["lineart_boxes"] = float(len(lineart))
    out.metrics["table_boxes"] = float(len(tables))
    out.metrics["raster_boxes"] = float(len(raster))

    # Штрихи: LSD в обеих версиях (как в ядре) — для счётчиков, потерь, параллельности и запасного
    # хода наклона; сам наклон в A — по краске вдоль штриха B.
    lineart_a = _transform_boxes(lineart, warp)
    raster_a = _transform_boxes(raster, warp)
    strokes_b = find_strokes(gray300_b, params.stroke_min_mm, RENDER_DPI, lineart, dpi)
    strokes_a = find_strokes(gray300_a, params.stroke_min_mm, RENDER_DPI, lineart_a, dpi, drop_lone=False)
    k_render = RENDER_DPI / dpi
    strokes_b = [s for s in strokes_b if not _mid_inside(s, raster, k_render)]
    strokes_a = [s for s in strokes_a if not _mid_inside(s, raster_a, k_render)]
    pairs = match_strokes(strokes_b, strokes_a, warp, RENDER_DPI, dpi)
    metrics, culprits = stroke_metrics(strokes_b, strokes_a, pairs, rot_deg, RENDER_DPI)
    for name in list(metrics):
        if name.startswith(("hstroke", "vstroke", "parallel")):
            metrics.pop(name)
            culprits.pop(name, None)
    out.metrics.update(metrics)
    out.culprits.update(_scale_culprits(culprits, dpi / RENDER_DPI))
    # Параллельность — только околоосевые штрихи (полки шкафа под −5° на 1966/01 с.78 — в допуске):
    # штриховка диаграмм под 45° (1975/09 с.6, с.10; 1975/06 с.69) — не линейки, разброс её
    # углов ничего не значит. Внутри рисунков — да: там параллельность и была придумана.
    # Внутри рисунков — только штрихи от PARALLEL_LINEART_MIN_MM: рёбра столбиков диаграммы
    # по 5–8 мм (1975/09 с.10) и обломки в чертеже (1974/10 с.30) дают разброс углов из ничего.
    min_lineart = PARALLEL_LINEART_MIN_MM * RENDER_DPI / 25.4
    axis_pairs = [
        (sb, sa) for sb, sa in pairs if sb.axis_tilt is not None and (not sb.in_lineart or sb.length >= min_lineart)
    ]
    metrics, culprits = stroke_metrics(strokes_b, strokes_a, axis_pairs, rot_deg, RENDER_DPI)
    out.metrics.update({name: value for name, value in metrics.items() if name.startswith("parallel")})
    out.culprits.update(
        _scale_culprits({n: c for n, c in culprits.items() if n.startswith("parallel")}, dpi / RENDER_DPI)
    )

    # Строки корпуса по глифам: волна, растяжение, наклоны по проекции (сводятся после деcкью).
    body_pairs = match_lines(body_b, lines_a, warp, dpi)
    metrics, culprits, verified, tilts = glyph_line_metrics(
        gray300_b, gray300_a, body_pairs, body_b, warp, dpi, params.line_min_mm
    )
    out.metrics.update(metrics)
    out.culprits.update(culprits)
    out.metrics.update(line_metrics(body_b, lines_a, verified, dpi))
    # Строки внутри таблиц и рисунков — только выигрыш, максимумом с корпусом.
    inner_pairs = match_lines(inner_b, lines_a, warp, dpi)
    _, _, verified_in, tilts_in = glyph_line_metrics(
        gray300_b, gray300_a, inner_pairs, inner_b, warp, dpi, params.line_min_mm
    )
    inner_stats = line_metrics(inner_b, lines_a, verified_in, dpi)
    out.metrics["table_lines_verified"] = float(len(verified_in))
    out.metrics["table_sag_gain_mm"] = inner_stats["text_sag_gain_mm"]
    out.metrics["table_spread_gain_deg"] = inner_stats["text_spread_gain_deg"]
    _merge_max(out.metrics, inner_stats, ("text_sag_gain_mm", "text_spread_gain_deg"))

    # Направления штрихов — по краске (следы), запасной ход — пары LSD.
    tilts_strokes = lsd_fallback(trace_strokes(gray300_b, gray300_a, strokes_b, warp, RENDER_DPI, dpi), pairs)
    # Доворот страницы, подтверждённый строками или штрихами: выигрыш и поправка к наклонам A.
    deskew_gain, rot_adjust, source = _deskew(tilts + tilts_in, tilts_strokes, rot_deg)
    out.metrics["page_deskew_gain_deg"] = deskew_gain
    out.metrics["deskew_rot_adjust_deg"] = rot_adjust
    out.metrics["deskew_by_strokes"] = float(source == "strokes")
    metrics, culprits = tilt_summary(tilts, rot_adjust)
    out.metrics.update(metrics)
    out.culprits.update(culprits)
    metrics_in, _ = tilt_summary(tilts_in, rot_adjust)
    _merge_max(out.metrics, metrics_in, ("line_tilt_gain_mm", "line_tilt_gain_deg"))
    # Наклон штрихов к оси и изгиб (след с ограничением шага).
    metrics, culprits = ridge_stroke_metrics(tilts_strokes, RENDER_DPI, rot_adjust)
    out.metrics.update(metrics)
    out.culprits.update(_scale_culprits(culprits, dpi / RENDER_DPI))
    metrics, culprits = bend_metrics(gray300_b, gray300_a, strokes_b, warp, RENDER_DPI, dpi)
    out.metrics.update(metrics)
    out.culprits.update(_scale_culprits(culprits, dpi / RENDER_DPI))

    # Кромки фотографий — по каждому снимку в рамке, наклон по снимку целиком.
    metrics, culprits = raster_edge_metrics(b, a, raster, warp, dpi)
    out.metrics.update(metrics)
    out.culprits.update(culprits)

    # Перекос: сдвиг по полю на тайлах текста и кромки выключенных блоков по тем же строкам.
    metrics, culprits = shear_metrics(warp, text_boxes(body_b), lineart + tables + raster, dpi)
    out.metrics.update(metrics)
    out.culprits.update(culprits)
    metrics, culprits = edge_metrics(block_edges(body_b, body_pairs, dpi), dpi)
    out.metrics.update(metrics)
    out.culprits.update(culprits)

    out.raw = {
        "engine": ENGINE_VERSION,
        "size_b": [int(b.shape[1]), int(b.shape[0])],
        "size_a": [int(a.shape[1]), int(a.shape[0])],
        "lineart": _boxes(lineart),
        "tables": _boxes(tables),
        "raster": _boxes(raster),
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


__all__ = ["measure_pair"]
