"""Оверлей пары «без коррекции | с коррекцией» v17–v18: строки и выровненные стороны с их мерами, в шапке — вердикт, правило и все метрики с порогами."""

from __future__ import annotations

from pathlib import Path

import cv2
import fitz
import numpy as np

from ocr_utils.geometry_regression.render import render_gray, to_work
from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, header_strip, legend_strip
from ocr_utils.geometry_regression.quality.scoring import Assessment, Group, Thresholds17

# Палитра (BGR, навык draw-overlay): порча — красный «отвергнутого», выигрыш — зелёный «принятого»,
# выровненная сторона — синий дополнительной линии стороны, объекты B — приглушённая заливка.
COLOUR_WORSE = (0, 0, 220)
COLOUR_BETTER = (0, 150, 0)
COLOUR_SIDE = (220, 90, 20)
COLOUR_OBJECT = (200, 140, 60)
COLOUR_TEXT = (20, 20, 20)
SIDE_ALPHA = 0.55
OBJECT_ALPHA = 0.18
# Строка рисуется, если её порча или выигрыш (мм с весом) не меньше этой доли порога качества строк.
LINE_SHOW_FRAC = 0.3


def _label(canvas: np.ndarray, text: str, x: float, y: float, colour: tuple[int, int, int]) -> None:
    """Короткая числовая подпись у сущности (ASCII — шрифт OpenCV), на белой подложке."""
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    x, y = int(max(0, min(canvas.shape[1] - w - 2, x))), int(max(h + 2, y))
    cv2.rectangle(canvas, (x - 1, y - h - 2), (x + w + 1, y + 2), (255, 255, 255), -1)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)


def _dashed_rect(canvas: np.ndarray, box, colour: tuple[int, int, int], dash: int = 6) -> None:
    """Пунктирная рамка (у OpenCV пунктира нет): отрезки по ``dash`` px через такой же промежуток."""
    x0, y0, x1, y1 = (int(v) for v in box)
    for a, b in (((x0, y0), (x1, y0)), ((x0, y1), (x1, y1)), ((x0, y0), (x0, y1)), ((x1, y0), (x1, y1))):
        length = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        for start in range(0, length, 2 * dash):
            stop = min(length, start + dash)
            p0 = (a[0] + (b[0] - a[0]) * start // max(length, 1), a[1] + (b[1] - a[1]) * start // max(length, 1))
            p1 = (a[0] + (b[0] - a[0]) * stop // max(length, 1), a[1] + (b[1] - a[1]) * stop // max(length, 1))
            cv2.line(canvas, p0, p1, colour, 1, cv2.LINE_AA)


def _draw_side(page: np.ndarray, payload: dict, which: str, line_threshold: float) -> np.ndarray:
    """Разметка одной стороны пары (``which`` — ``b`` или ``a``) на её рабочей копии."""
    canvas = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    if which == "b":
        layer = canvas.copy()
        for x0, y0, x1, y1 in payload["raw"]["objects_b"]:
            cv2.rectangle(layer, (int(x0), int(y0)), (int(x1), int(y1)), COLOUR_OBJECT, -1)
        cv2.addWeighted(layer, OBJECT_ALPHA, canvas, 1 - OBJECT_ALPHA, 0, canvas)
    # Выровненные стороны пар — толсто и полупрозрачно, подпись: размах этой стороны и вес.
    layer = canvas.copy()
    for edge in payload["edges"]:
        points = np.asarray(edge[f"points_{which}"], dtype=np.int32)
        if len(points) >= 2:
            cv2.polylines(layer, [points], False, COLOUR_SIDE, 3, cv2.LINE_AA)
    cv2.addWeighted(layer, SIDE_ALPHA, canvas, 1 - SIDE_ALPHA, 0, canvas)
    for edge in payload["edges"]:
        points = np.asarray(edge[f"points_{which}"])
        if len(points) < 2:
            continue
        delta = edge["w"] * (edge["e_a"] - edge["e_b"])
        colour = COLOUR_WORSE if delta > 0.3 else COLOUR_BETTER if delta < -0.3 else COLOUR_SIDE
        x, y = points[len(points) // 2]
        _label(canvas, f"E{edge[f'e_{which}']:.2f} w{edge['w']:.2f}", x + 4, y, colour)
    # Строки с заметной порчей (красный) или выигрышем (зелёный): рамка общего участка и размах Q.
    for line in payload["lines"]:
        delta = line["w"] * (line["q_a"] - line["q_b"])
        if abs(delta) < LINE_SHOW_FRAC * line_threshold:
            continue
        colour = COLOUR_WORSE if delta > 0 else COLOUR_BETTER
        x0, y0, x1, y1 = line[which]
        # Строка без пары в разборе A (ось A перенесена полем) — пунктиром.
        if line.get("tr"):
            _dashed_rect(canvas, (x0, y0 - 2, x1, y1 + 2), colour)
        else:
            cv2.rectangle(canvas, (int(x0), int(y0) - 2), (int(x1), int(y1) + 2), colour, 1)
        _label(canvas, f"Q{line[f'q_{which}']:.2f}", x1 + 3, y1, colour)
    # Виновники метрик — толстой рамкой.
    for name, where in payload.get("culprits", {}).items():
        if which in where:
            x0, y0, x1, y1 = where[which]
            colour = COLOUR_BETTER if name.endswith("gain_mm") else COLOUR_WORSE
            cv2.rectangle(canvas, (int(x0) - 3, int(y0) - 3), (int(x1) + 3, int(y1) + 3), colour, 2)
    return canvas


def header_lines(payload: dict, assessment: Assessment, thresholds: Thresholds17) -> list[str]:
    """Шапка: страница, вердикт и правило, группы и сумма, выигрыш, таблица метрик порчи с порогами."""
    lines = [
        f"{payload['pdf']} с.{payload['page']}   вердикт {payload.get('version', 'v17')}: {assessment.verdict.value.upper()}   правило: "
        f"{assessment.rule.value}{' — ' + assessment.culprit if assessment.culprit else ''}",
        "группы: "
        + ", ".join(f"{g.value} {assessment.groups[g]:.2f}" for g in Group)
        + f"   Σ = {assessment.total:.2f} (bad при Σ ≥ S = {thresholds.total:g} и Σ ≥ {thresholds.ratio:g} × выигрыш = {thresholds.ratio * assessment.gain:.2f})",
        f"порча (жёсткая) {assessment.damage:.2f}, не-текста {assessment.damage_other:.2f}, с мягкими {assessment.worst:.2f}; "
        f"выигрыш {assessment.gain:.2f}, для порчи не-текста {assessment.gain_other:.2f}"
        f" ({assessment.gain_reason or '—'}; min_gain {thresholds.min_gain:g}, порча ≥ {thresholds.ratio:g} × выигрыш"
        f" = {thresholds.ratio * assessment.gain:.2f} → bad)",
        "метрика: значение | порог | жёсткий | score | группа",
    ]
    if payload.get("previous"):
        lines.insert(1, f"прежний прогон: {payload['previous']}")
    shown = sorted(assessment.scores.values(), key=lambda s: -s.score)
    for item in shown:
        if item.value == 0.0 and item.spec.group not in (Group.HORIZONTALS, Group.VERTICALS):
            continue
        marks = [item.spec.group.value]
        if item.spec.unforgivable:
            marks.append("непрощаемая")
        if item.soft:
            marks.append("мягкая")
        if not item.applicable:
            marks.append("не применима")
        flag = "  ≥1" if item.score >= 1 else ""
        lines.append(
            f"  {item.name}: {item.value:.3f} | {item.spec.threshold:g} | {item.spec.hard:g} | {item.score:.2f}{flag} | "
            + ", ".join(marks)
        )
    lines.append("выигрыш: значение | порог | score")
    for name, threshold in thresholds.gains.items():
        value = float(payload["metrics"].get(name, 0.0) or 0.0)
        lines.append(f"  {name}: {value:.3f} | {threshold:g} | {assessment.gains.get(name, 0.0):.2f}")
    metrics = payload["metrics"]
    if metrics.get("lineart_shape_raw_mm"):
        lines.append(
            f"форма line art (v18): групповая {metrics.get('lineart_shape_mm', 0.0):.2f} мм (сырая "
            f"{metrics.get('lineart_shape_raw_mm', 0.0):.2f}), перекос+пропорции {metrics.get('lineart_nonsim_mm', 0.0):.2f}, "
            f"изгиб+трапеция {metrics.get('lineart_curve_mm', 0.0):.2f}, перекос осей {metrics.get('lineart_shape_shear_deg', 0.0):.2f}°, "
            f"поворот частей {metrics.get('lineart_part_turn_deg', 0.0):.2f}°; прямота черт A−B "
            f"{metrics.get('lineart_straight_delta_mm', 0.0):+.2f} мм ({int(metrics.get('lineart_straight_lines', 0))} черт), "
            f"участков {int(metrics.get('lineart_patches', 0))}"
        )
    if metrics.get("line_transferred"):
        lines.append(
            f"строк без пары в A, перенесённых полем: {int(metrics['line_transferred'])}, худшая порча среди них "
            f"{metrics.get('line_quality_transferred_mm', 0.0):.2f} мм"
        )
    lines.append(
        f"пар строк {int(metrics.get('line_pairs', 0))}, сторон {int(metrics.get('edge_pairs', 0))}; "
        f"край по сырой стороне {metrics.get('edge_quality_raw_mm', 0.0):.2f} мм; вердикт v16 {metrics.get('v16_verdict', '—')}"
    )
    return lines


def draw_pair(
    payload: dict, assessment: Assessment, thresholds: Thresholds17, geo_pdf: Path, nogeo_pdf: Path
) -> np.ndarray:
    """Склейка «без коррекции | с коррекцией» с шапкой и легендой в полях.

    Args:
        payload: Кэш страницы (:func:`measure.measure_page`).
        assessment: Вердикт v17.
        thresholds: Пороги (в шапку).
        geo_pdf: PDF с коррекцией.
        nogeo_pdf: PDF без коррекции.

    Returns:
        Картинка BGR.
    """
    index = payload["page"] - 1
    with fitz.open(str(nogeo_pdf)) as nogeo, fitz.open(str(geo_pdf)) as geo:
        page_b, page_a = to_work(render_gray(nogeo, index)), to_work(render_gray(geo, index))
    threshold = thresholds.specs["line_quality_mm"].threshold
    tiles = [_draw_side(page_b, payload, "b", threshold), _draw_side(page_a, payload, "a", threshold)]
    titled = [
        np.vstack([header_strip([title], tile.shape[1]), tile])
        for tile, title in zip(tiles, ["без коррекции (B)", "с коррекцией FineReader (A)"])
    ]
    # Страницы B и A бывают разного размера (FineReader меняет поля), и шапки над ними переносятся по-разному:
    # высота выравнивается белым полем снизу уже после шапок.
    height = max(t.shape[0] for t in titled)
    titled = [np.vstack([t, np.full((height - t.shape[0], t.shape[1], 3), 255, np.uint8)]) for t in titled]
    canvas = np.hstack([titled[0], np.full((height, 12, 3), 255, np.uint8), titled[1]])
    width = canvas.shape[1]
    legend = [
        LegendEntry("строка: порча (рамка общего участка, Q — размах оси, мм)", COLOUR_WORSE),
        LegendEntry("строка: выигрыш", COLOUR_BETTER),
        LegendEntry("пунктир — строки без пары в разборе A: ось A перенесена полем", COLOUR_WORSE),
        LegendEntry(
            "выровненная сторона блока с заплатками (E — размах, мм; w — вес по рядам)", COLOUR_SIDE, SIDE_ALPHA
        ),
        LegendEntry(
            "таблица, рисунок, формула на B (строки там — только выигрыш)", COLOUR_OBJECT, OBJECT_ALPHA, SampleStyle.BOX
        ),
        LegendEntry("виновник метрики — толстая рамка", COLOUR_WORSE),
    ]
    return np.vstack([header_strip(header_lines(payload, assessment, thresholds), width), canvas,
                      legend_strip(legend, width, "обозначения")])  # fmt: skip


__all__ = ["draw_pair", "header_lines"]
