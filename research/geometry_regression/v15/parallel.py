"""Параллельность групп штрихов «было | стало» — как в ядре, но с порогами по месту штриха.

Группы отрезков B по углу вокруг самого длинного (``GROUP_TOL_DEG``), разброс углов группы до и
после: законная правка (поворот, сдвиг) разброс не меняет, местная — увеличивает (полки шкафа
1966/01 с.78, лучи номограммы 1974/10 с.30). Отличия от ``strokes.stroke_metrics`` ядра:

* только околоосевые штрихи (``axis_tilt`` есть): штриховка диаграмм под 45° (1975/09 с.6, с.10;
  1975/06 с.69) давала разброс из ничего;
* внутри рисунков — штрихи от ``LINEART_MIN_MM`` и группы от ``LINEART_MIN_GROUP`` членов: три
  ребра легенды диаграммы по 13–17 мм (1975/09 с.10) — не группа, пять лучей номограммы — группа;
  вне рисунков — штрихи от ``TEXT_MIN_MM`` (в ядре 8 мм).
"""

from __future__ import annotations

import numpy as np

from ocr_utils.geometry_regression.strokes import GROUP_MIN, GROUP_TOL_DEG, Stroke, _wrap

LINEART_MIN_MM = 12.0
LINEART_MIN_GROUP = 4
# Вне рисунков — штрихи от стольких мм: стороны квадратиков легенды диаграммы по 12–13 мм
# (1975/09 с.10, один квадратик нарисован под 3°) вне рамки line art собирались в группу.
TEXT_MIN_MM = 15.0


def parallel_metrics(pairs: list[tuple[Stroke, Stroke]], dpi: float) -> tuple[dict[str, float], dict]:
    """Худший по группам прирост разброса углов и её отрезки для оверлея (пиксели ``dpi``)."""
    metrics = {"parallel_groups": 0.0, "parallel_spread_delta_max": 0.0}
    culprits: dict = {}
    mm = dpi / 25.4
    eligible = [
        (b, a)
        for b, a in pairs
        if b.axis_tilt is not None and b.length >= (LINEART_MIN_MM if b.in_lineart else TEXT_MIN_MM) * mm
    ]
    if not eligible:
        return metrics, culprits
    ang_b = np.array([b.angle_deg for b, _ in eligible])
    ang_a = np.array([a.angle_deg for _, a in eligible])
    order = np.argsort([-b.length for b, _ in eligible])
    used = np.zeros(len(eligible), dtype=bool)
    best = None
    for k in order:
        if used[k]:
            continue
        group = np.nonzero((np.abs(_wrap(ang_b - ang_b[k])) < GROUP_TOL_DEG) & ~used)[0]
        need = LINEART_MIN_GROUP if eligible[k][0].in_lineart else GROUP_MIN
        if len(group) < need:
            continue
        used[group] = True
        delta = float(np.ptp(_wrap(ang_a[group] - ang_a[k])) - np.ptp(_wrap(ang_b[group] - ang_b[k])))
        metrics["parallel_groups"] += 1.0
        if best is None or delta > best[0]:
            best = (delta, group)
    if best is not None:
        delta, group = best
        metrics["parallel_spread_delta_max"] = delta
        segs_b = [[eligible[i][0].x0, eligible[i][0].y0, eligible[i][0].x1, eligible[i][0].y1] for i in group]
        segs_a = [[eligible[i][1].x0, eligible[i][1].y0, eligible[i][1].x1, eligible[i][1].y1] for i in group]
        box = lambda segs: (
            int(min(min(s[0], s[2]) for s in segs)),
            int(min(min(s[1], s[3]) for s in segs)),
            int(max(max(s[0], s[2]) for s in segs)) + 1,
            int(max(max(s[1], s[3]) for s in segs)) + 1,
        )
        culprits["parallel_spread_delta_max"] = {
            "b": box(segs_b),
            "a": box(segs_a),
            "segments_b": segs_b,
            "segments_a": segs_a,
        }
    return metrics, culprits


__all__ = ["parallel_metrics"]
