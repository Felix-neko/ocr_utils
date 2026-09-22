"""Порча line art коррекцией: угол между осями, поворот отдельных линий, изгиб, потерянные линии.

Разбор 2026-09-22 (1968/12 с.74 шкаф, 1973/03 с.59 стеллаж, 1974/12 с.73 график): снимая трапецию,
FineReader правит рисунок сильнее соседнего текста — ось X графика повернулась на 1.1° при неподвижной
оси Y, стеллаж стал параллелограммом, перспективное ребро шкафа 81 мм под 4.3° довёрнуто к горизонтали
на 3°, полка 50 мм получила сагитту 2.2 мм. Решение пользователя: поворот рисунка целиком (как деcкью)
прощается, изменение углов между линиями, изгиб прямых и поворот отдельных линий — нет.

Меряется по самим линиям (LSD-штрихи B внутри рамки рисунка, направление в A — по следу краски вдоль
перенесённого полем штриха, с широким окном; запасной ход — двойник LSD в A с широкими допусками):

* ``lineart_axis_delta_deg`` — изменение угла между семействами: медиана поворота горизонталей
  (|угол| ≤ ``H_MAX_DEG``) минус медиана поворота вертикалей (≥ ``V_MIN_DEG``), по модулю; у чистого
  поворота рисунка обе медианы равны и разность нулевая;
* ``lineart_rot_max_deg`` — поворот отдельной линии сверх поворота рисунка (медианы по всем его линиям);
* ``lineart_spread_delta_deg`` — рост разброса углов внутри семейства (линии от ``ROT_MIN_MM``, не
  меньше трёх), A − B: параллельные в B разошлись в A; выравнивание перекошенной сканом блок-схемы
  (1972/11 с.27) разброс уменьшает и порчей не считается;
* ``lineart_bend_mm`` — рост сагитты линии от ``BEND_MIN_MM`` (короче, чем у линеек страницы: полки,
  оси графиков);
* ``lineart_lost_mm`` — длина самой длинной линии, найденной в B и не найденной в A ни следом, ни
  LSD (контекст: линия сдвинута дальше окна поиска или порвана).

Второй источник тех же величин — тайлы поля внутри рамки (когда их с парой не меньше ``MIN_TILES``):
локальный аффин → полярное разложение (DIC): угол между образами осей минус 90° и неконформность
(λ₁ − λ₂)/(λ₁ + λ₂), за вычетом тех же величин по странице. У половины страниц с line art тайлов в
рамке нет (рисунок меньше тайла, периодика), поэтому штрихи — основной источник.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import polar

from ocr_utils.geometry_regression import mm_to_px, px_to_mm
from ocr_utils.geometry_regression.field import Field, robust_affine
from ocr_utils.geometry_regression.render import RENDER_DPI
from ocr_utils.geometry_regression.strokes import Stroke, _wrap
from research.geometry_regression.v15.ridge import Trace, trace_line

Box = tuple[int, int, int, int]

# Штрихи внутри рамки от стольких мм идут в семейства и в поворот; в изгиб — от BEND_MIN_MM.
MIN_STROKE_MM = 15.0
ROT_MIN_MM = 20.0
BEND_MIN_MM = 25.0
# Припуск рамки рисунка (мм): линии рамки лежат на её границе.
BOX_PAD_MM = 2.0
# Семейства: горизонтали и вертикали по углу в B; каждое — суммарно не короче FAMILY_MIN_MM.
H_MAX_DEG = 30.0
V_MIN_DEG = 60.0
FAMILY_MIN_MM = 30.0
# След краски: шаг проб и окно поиска (перспективное ребро уходит от прогноза поля на миллиметры).
TILT_STEP_MM = 1.0
TILT_WINDOW_MM = 3.0
TILT_MIN_SAMPLES = 6
BEND_STEP_MM = 2.0
BEND_WINDOW_MM = 3.0
BEND_MIN_SAMPLES = 6
# Запасной двойник LSD в A: перпендикуляр до прямой B′, угол, длина.
MATCH_PERP_MM = 5.0
MATCH_ANGLE_DEG = 8.0
MATCH_LENGTH_FRAC = 0.4
# Разброс углов считается по семейству не меньше чем из стольких линий от SPREAD_MIN_MM: у коротких
# рёбер блок-схемы (20–25 мм, 1972/11 с.27) угол по следу гуляет на ±0.5°.
SPREAD_MIN_LINES = 3
SPREAD_MIN_MM = 25.0
# Потерянной считается линия, чей след в B покрыт не хуже этого.
LOST_MIN_COVERAGE = 0.9
# Тайлов с парой внутри рамки для локального аффина.
MIN_TILES = 6


@dataclass(frozen=True)
class LineArtStroke:
    """Линия рисунка: штрих B, направления в обеих версиях и след в A (если найден)."""

    stroke: Stroke
    angle_b: float
    angle_a: float | None  # None — не найдена в A
    source: str  # "ridge" | "lsd" | "lost"
    segment_a: list[float] | None


def _inside(stroke: Stroke, box: Box, pad: float) -> bool:
    mx, my = stroke.mid
    x0, y0, x1, y1 = box
    return x0 - pad <= mx < x1 + pad and y0 - pad <= my < y1 + pad


def _segment(trace: Trace) -> list[float]:
    pts = trace.points
    return [float(pts[0, 0]), float(pts[0, 1]), float(pts[-1, 0]), float(pts[-1, 1])]


def _lsd_twin(stroke: Stroke, after: list[Stroke], field: Field | None, k: float, dpi: float) -> Stroke | None:
    """Двойник штриха B среди штрихов A с широкими допусками (середина переносится полем)."""
    if not after:
        return None
    mid = np.array(stroke.mid, dtype=np.float64)
    if field is not None:
        mid = field.transform(mid[None] * k)[0] / k
    theta = np.radians(stroke.angle_deg)
    normal = np.array([-np.sin(theta), np.cos(theta)])
    along = np.array([np.cos(theta), np.sin(theta)])
    perp_tol = mm_to_px(MATCH_PERP_MM, dpi)
    best, best_perp = None, perp_tol
    for other in after:
        if abs(float(_wrap(other.angle_deg - stroke.angle_deg))) > MATCH_ANGLE_DEG:
            continue
        if abs(other.length - stroke.length) > MATCH_LENGTH_FRAC * stroke.length:
            continue
        delta = np.array(other.mid) - mid
        if abs(float(delta @ along)) > 0.5 * stroke.length:
            continue
        perp = abs(float(delta @ normal))
        if perp < best_perp:
            best, best_perp = other, perp
    return best


def lineart_strokes(
    gray_b: np.ndarray,
    gray_a: np.ndarray,
    strokes_b: list[Stroke],
    strokes_a: list[Stroke],
    box: Box,
    field: Field | None,
    dpi: float,
    field_dpi: float,
) -> list[LineArtStroke]:
    """Линии рисунка внутри рамки ``box`` (пиксели ``field_dpi``) с направлениями в B и A.

    Args:
        gray_b, gray_a: Серые рендеры в ``dpi``.
        strokes_b, strokes_a: LSD-штрихи обеих версий в пикселях ``dpi``.
        box: Рамка рисунка в пикселях ``field_dpi``.
        field: Поле смещений B → A в пикселях ``field_dpi``.
        dpi: Разрешение рендеров.
        field_dpi: Разрешение поля и рамки.
    """
    k = field_dpi / dpi
    pad = mm_to_px(BOX_PAD_MM, dpi)
    box_px = tuple(v / k for v in box)
    out: list[LineArtStroke] = []
    for stroke in strokes_b:
        if stroke.length < mm_to_px(MIN_STROKE_MM, dpi) or not _inside(stroke, box_px, pad):
            continue
        args = (stroke.x0, stroke.y0, stroke.x1, stroke.y1, dpi)
        trace_b = trace_line(gray_b, *args, None, k, TILT_STEP_MM, TILT_WINDOW_MM, TILT_MIN_SAMPLES)
        angle_b = trace_b.angle_deg if trace_b is not None else stroke.angle_deg
        # Без поля (страница без текста) линия в A ищется на месте линии B: окно ±TILT_WINDOW_MM.
        trace_a = trace_line(gray_a, *args, field, k, TILT_STEP_MM, TILT_WINDOW_MM, TILT_MIN_SAMPLES)
        if trace_a is not None:
            out.append(LineArtStroke(stroke, angle_b, trace_a.angle_deg, "ridge", _segment(trace_a)))
            continue
        twin = _lsd_twin(stroke, strokes_a, field, k, dpi)
        if twin is not None:
            out.append(LineArtStroke(stroke, angle_b, twin.angle_deg, "lsd", [twin.x0, twin.y0, twin.x1, twin.y1]))
            continue
        lost = trace_b is not None and trace_b.coverage >= LOST_MIN_COVERAGE
        out.append(LineArtStroke(stroke, angle_b, None, "lost" if lost else "none", None))
    return out


def _tile_distortion(field: Field | None, box: Box) -> tuple[float, float, int]:
    """По тайлам поля внутри рамки: угол между образами осей − 90° и неконформность сверх страницы.

    Returns:
        ``(axis_delta_deg, anisotropy, tiles)``; нули, если тайлов с парой меньше ``MIN_TILES``.
    """
    if field is None or len(field.tiles) == 0:
        return 0.0, 0.0, 0
    good = field.weight > 0
    pts = field.tiles[:, :2]
    x0, y0, x1, y1 = box
    inside = good & (pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)
    if inside.sum() < MIN_TILES:
        return 0.0, 0.0, int(inside.sum())
    local, _, _ = robust_affine(pts[inside], pts[inside] + field.tiles[inside, 2:4])
    axis_local, aniso_local = _decompose(local[:, :2])
    axis_page, aniso_page = _decompose(field.affine[:, :2])
    return abs(axis_local - axis_page), max(0.0, aniso_local - aniso_page), int(inside.sum())


def _decompose(matrix: np.ndarray) -> tuple[float, float]:
    """Линейная часть аффина → (угол между образами осей − 90°, неконформность (λ₁ − λ₂)/(λ₁ + λ₂))."""
    ex, ey = matrix @ np.array([1.0, 0.0]), matrix @ np.array([0.0, 1.0])
    cos = float(ex @ ey / max(np.linalg.norm(ex) * np.linalg.norm(ey), 1e-9))
    axis = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) - 90.0)
    _, stretch = polar(matrix)
    lam = np.linalg.eigvalsh(stretch)
    aniso = float((lam.max() - lam.min()) / max(lam.max() + lam.min(), 1e-9))
    return axis, aniso


def _family_turn(items: list[LineArtStroke], rot_deg: float) -> tuple[float, float]:
    """Медианный поворот линий сверх поворота страницы (градусы) и их суммарная длина (px)."""
    turns = [float(_wrap(s.angle_a - s.angle_b)) - rot_deg for s in items if s.angle_a is not None]
    if not turns:
        return 0.0, 0.0
    return float(np.median(turns)), float(sum(s.stroke.length for s in items if s.angle_a is not None))


def lineart_metrics(
    gray_b: np.ndarray,
    gray_a: np.ndarray,
    strokes_b: list[Stroke],
    strokes_a: list[Stroke],
    drawings: list[Box],
    field: Field | None,
    rot_deg: float,
    dpi: float,
    field_dpi: float,
) -> tuple[dict[str, float], dict]:
    """Порча рисунков страницы: худшие значения по рамкам ``drawings`` и виновники (пиксели ``dpi``).

    Args:
        gray_b, gray_a: Серые рендеры в ``dpi``.
        strokes_b, strokes_a: LSD-штрихи обеих версий (пиксели ``dpi``).
        drawings: Рамки line art на B в пикселях ``field_dpi``.
        field: Поле смещений B → A.
        rot_deg: Поворот аффинной части поля (вычитается из поворотов линий).
        dpi: Разрешение рендеров.
        field_dpi: Разрешение рамок и поля.

    Returns:
        Метрики ``lineart_axis_delta_deg``, ``lineart_rot_max_deg``, ``lineart_spread_delta_deg``, ``lineart_bend_mm``,
        ``lineart_lost_mm``, ``lineart_tile_axis_delta_deg``, ``lineart_tile_aniso``,
        ``lineart_strokes`` и виновники с отрезками для оверлея.
    """
    metrics = {
        "lineart_axis_delta_deg": 0.0,
        "lineart_rot_max_deg": 0.0,
        "lineart_spread_delta_deg": 0.0,
        "lineart_bend_mm": 0.0,
        "lineart_lost_mm": 0.0,
        "lineart_tile_axis_delta_deg": 0.0,
        "lineart_tile_aniso": 0.0,
        "lineart_strokes": 0.0,
    }
    culprits: dict = {}
    k = field_dpi / dpi
    for box in drawings:
        items = lineart_strokes(gray_b, gray_a, strokes_b, strokes_a, box, field, dpi, field_dpi)
        metrics["lineart_strokes"] += float(len(items))
        box_px = tuple(int(v / k) for v in box)
        found = [s for s in items if s.angle_a is not None]
        # Угол между семействами: горизонтали против вертикалей, каждое — от FAMILY_MIN_MM.
        horizontals = [s for s in found if abs(s.angle_b) <= H_MAX_DEG]
        verticals = [s for s in found if abs(s.angle_b) >= V_MIN_DEG]
        turn_h, len_h = _family_turn(horizontals, rot_deg)
        turn_v, len_v = _family_turn(verticals, rot_deg)
        family_min = mm_to_px(FAMILY_MIN_MM, dpi)
        if len_h >= family_min and len_v >= family_min:
            axis = abs(turn_h - turn_v)
            if axis > metrics["lineart_axis_delta_deg"]:
                metrics["lineart_axis_delta_deg"] = axis
                shown = horizontals + verticals
                culprits["lineart_axis_delta_deg"] = _culprit(box_px, shown)
        # Поворот отдельной линии сверх поворота рисунка (медиана по всем его линиям).
        turns = {id(s): float(_wrap(s.angle_a - s.angle_b)) - rot_deg for s in found}
        if turns:
            own_rot = float(np.median(list(turns.values())))
            long = [s for s in found if s.stroke.length >= mm_to_px(ROT_MIN_MM, dpi)]
            for s in long:
                extra = abs(turns[id(s)] - own_rot)
                if extra > metrics["lineart_rot_max_deg"]:
                    metrics["lineart_rot_max_deg"] = extra
                    culprits["lineart_rot_max_deg"] = _culprit(box_px, [s])
            # Разброс углов внутри семейства: параллельные в B разошлись в A (трапеция стеллажа
            # 1973/03 с.59). Именно A − B, а не разброс поворотов: перекошенную сканом блок-схему
            # (1972/11 с.27, рёбра от −2° до +2°) FineReader выровнял к осям — разброс упал, это не порча.
            for family in (horizontals, verticals):
                members = [s for s in family if s.stroke.length >= mm_to_px(SPREAD_MIN_MM, dpi)]
                if len(members) < SPREAD_MIN_LINES:
                    continue
                ref = members[0].angle_b
                spread_b = float(np.ptp([_wrap(s.angle_b - ref) for s in members]))
                spread_a = float(np.ptp([_wrap(s.angle_a - ref) for s in members]))
                if spread_a - spread_b > metrics["lineart_spread_delta_deg"]:
                    metrics["lineart_spread_delta_deg"] = spread_a - spread_b
                    culprits["lineart_spread_delta_deg"] = _culprit(box_px, members)
        # Потерянные линии — контекст.
        for s in items:
            if s.source == "lost" and s.stroke.length >= mm_to_px(ROT_MIN_MM, dpi):
                metrics["lineart_lost_mm"] = max(metrics["lineart_lost_mm"], px_to_mm(s.stroke.length, dpi))
        # Изгиб — по следу с широким окном, линии от BEND_MIN_MM.
        for s in items:
            if s.stroke.length < mm_to_px(BEND_MIN_MM, dpi):
                continue
            st = s.stroke
            args = (st.x0, st.y0, st.x1, st.y1, dpi)
            trace_b = trace_line(gray_b, *args, None, k, BEND_STEP_MM, BEND_WINDOW_MM, BEND_MIN_SAMPLES)
            trace_a = trace_line(gray_a, *args, field, k, BEND_STEP_MM, BEND_WINDOW_MM, BEND_MIN_SAMPLES)
            if trace_b is None or trace_a is None:
                continue
            delta = trace_a.sag_mm - trace_b.sag_mm
            if delta > metrics["lineart_bend_mm"]:
                metrics["lineart_bend_mm"] = delta
                pts = trace_a.points
                culprits["lineart_bend_mm"] = {
                    "b": st.box,
                    "a": (
                        int(pts[:, 0].min()),
                        int(pts[:, 1].min()),
                        int(pts[:, 0].max()) + 1,
                        int(pts[:, 1].max()) + 1,
                    ),
                    "segments_b": [[st.x0, st.y0, st.x1, st.y1]],
                    "segments_a": [[*pts[i], *pts[i + 1]] for i in range(len(pts) - 1)],
                }
        # Тайлы поля внутри рамки — второй источник угла между осями.
        tile_axis, tile_aniso, _ = _tile_distortion(field, box)
        metrics["lineart_tile_axis_delta_deg"] = max(metrics["lineart_tile_axis_delta_deg"], tile_axis)
        metrics["lineart_tile_aniso"] = max(metrics["lineart_tile_aniso"], tile_aniso)
    return metrics, culprits


def _culprit(box_px: Box, shown: list[LineArtStroke]) -> dict:
    """Рамка рисунка в обеих версиях и отрезки линий: в B — штрихи, в A — их следы или двойники."""
    segs_b = [[s.stroke.x0, s.stroke.y0, s.stroke.x1, s.stroke.y1] for s in shown]
    segs_a = [s.segment_a for s in shown if s.segment_a is not None]
    box_a = box_px
    if segs_a:
        xs = [v for seg in segs_a for v in (seg[0], seg[2])]
        ys = [v for seg in segs_a for v in (seg[1], seg[3])]
        box_a = (int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1)
    return {"b": box_px, "a": box_a, "segments_b": segs_b, "segments_a": segs_a}


__all__ = ["LineArtStroke", "lineart_strokes", "lineart_metrics"]
