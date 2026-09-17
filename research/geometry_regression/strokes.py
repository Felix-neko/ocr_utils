"""Длинные прямые штрихи (LSD): наклон к оси, поворот и параллельность попарно «было | стало».

Отрезки ищет LSD (``cv2.createLineSegmentDetector``) на рендере 300 dpi — он находит
штрих любой ориентации (наклонную дробную черту 1967/01 с.85, полки шкафа под −5° на
1966/01 с.78) и не требует морфологии, которая рвёт наклонённую линейку. Отрезки короче
``min_len_mm`` отбрасываются: у текста длинных прямых кромок нет.

Три вещи, которые FineReader портит и которые видны только на штрихах:

* **Наклон к оси.** Линейка таблицы или рамка, бывшая вертикальной, наклонилась (с.38,
  с.95): у околоосевых отрезков сравнивается |угол от оси|, взвешенно по длине.
* **Параллельность.** Полки шкафа в перспективе были параллельны, а после «распрямления»
  разошлись (с.78); рамки блок-схемы — то же (с.80). Отрезки B группируются по углу, и у
  группы сравнивается разброс углов до и после: законная правка (поворот, сдвиг) разброс
  не меняет, местная — увеличивает.
* **Поворот отрезка сверх общего.** ``|Δугла − поворот страницы|`` — сколько FineReader
  довернул именно этот штрих; для контекста, флага не ставит.

Сопоставление B → A: середина отрезка B переносится полем смещений, ищется отрезок A
близкого угла и длины, чья середина лежит у прямой B′.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from research.geometry_regression import px_to_mm
from research.geometry_regression.field import Field

# LSD на слегка размытом бинарном рендере: без размытия ступеньки бинаризации дробят отрезки.
BLUR_SIGMA_PX = 2.0
# Склейка коллинеарных кусков: наклонная черта на бинарном рендере — лестница, LSD режет её
# на 2-3 отрезка (1967/01 с.85). Кандидаты в склейку — от этой длины (мм); склеиваются при
# близких углах, малом перпендикулярном зазоре и разрыве вдоль не больше ``MERGE_GAP_MM``.
MERGE_MIN_MM = 4.0
MERGE_ANGLE_DEG = 4.0
MERGE_PERP_MM = 0.25
MERGE_GAP_MM = 1.3
# Околоосевой отрезок — в пределах этого угла от горизонтали или вертикали.
AXIS_TOL_DEG = 10.0
# Сопоставление: допуск перпендикулярного расстояния до прямой B′ (мм), по углу и по длине.
MATCH_PERP_MM = 1.3
MATCH_ANGLE_DEG = 8.0
MATCH_LENGTH_FRAC = 0.4
# Группа параллельных: отрезки B в пределах этого угла от самого длинного; меньше стольких — не группа.
GROUP_TOL_DEG = 1.0
GROUP_MIN = 3


@dataclass(frozen=True)
class Stroke:
    x0: float
    y0: float
    x1: float
    y1: float
    length: float  # px рендера
    angle_deg: float  # в (−90, 90]; 0 — горизонталь, ±90 — вертикаль

    @property
    def mid(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0

    @property
    def axis_tilt(self) -> float | None:
        """Отклонение от ближайшей оси, если отрезок околоосевой; иначе None."""
        a = abs(self.angle_deg)
        if a <= AXIS_TOL_DEG:
            return a
        if a >= 90.0 - AXIS_TOL_DEG:
            return 90.0 - a
        return None

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (
            int(min(self.x0, self.x1)),
            int(min(self.y0, self.y1)),
            int(max(self.x0, self.x1)) + 1,
            int(max(self.y0, self.y1)) + 1,
        )


def _wrap(angle: np.ndarray | float):
    """Разность направлений в (−90, 90]."""
    return (np.asarray(angle) + 90.0) % 180.0 - 90.0


def find_strokes(gray: np.ndarray, min_len_mm: float, dpi: float) -> list[Stroke]:
    """Отрезки LSD длиной от ``min_len_mm`` на рендере ``gray`` с разрешением ``dpi``."""
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    blurred = cv2.GaussianBlur(gray, (0, 0), BLUR_SIGMA_PX)
    found = detector.detect(blurred)[0]
    if found is None:
        return []
    segments = found.reshape(-1, 4).astype(np.float64)
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    segments = _merge_collinear(segments[lengths >= MERGE_MIN_MM * dpi / 25.4], dpi)
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    angles = _wrap(np.degrees(np.arctan2(segments[:, 3] - segments[:, 1], segments[:, 2] - segments[:, 0])))
    min_len = min_len_mm * dpi / 25.4
    return [Stroke(*segments[i], float(lengths[i]), float(angles[i])) for i in np.nonzero(lengths >= min_len)[0]]


def _merge_collinear(segments: np.ndarray, dpi: float) -> np.ndarray:
    """Склейка коллинеарных отрезков с малым разрывом; обход от длинных к коротким.

    Погнутая черта (1967/01 с.85: −6.9° слева, −1.6° справа) НЕ склеивается — углы
    кусков расходятся сильнее допуска, и это правильно: она и есть порча.
    """
    if len(segments) < 2:
        return segments
    perp_tol, gap_tol = MERGE_PERP_MM * dpi / 25.4, MERGE_GAP_MM * dpi / 25.4
    segs = sorted((row.copy() for row in segments), key=lambda r: -np.hypot(r[2] - r[0], r[3] - r[1]))
    i = 0
    while i < len(segs):
        a = segs[i]
        j = i + 1
        while j < len(segs):
            b = segs[j]
            da, db = a[2:] - a[:2], b[2:] - b[:2]
            la = np.hypot(*da)
            ua = da / la
            na = np.array([-ua[1], ua[0]])
            angle = abs(_wrap(np.degrees(np.arctan2(db[1], db[0]) - np.arctan2(da[1], da[0]))))
            perp = max(abs((b[:2] - a[:2]) @ na), abs((b[2:] - a[:2]) @ na))
            if angle < MERGE_ANGLE_DEG and perp <= perp_tol:
                t = np.array([0.0, la, (b[:2] - a[:2]) @ ua, (b[2:] - a[:2]) @ ua])
                gap = max(t[2:].min() - la, -t[2:].max())
                if gap <= gap_tol:
                    a = np.r_[a[:2] + ua * t.min(), a[:2] + ua * t.max()]
                    segs[i] = a
                    del segs[j]
                    continue
            j += 1
        i += 1
    return np.array(segs).reshape(-1, 4)


def match_strokes(
    before: list[Stroke], after: list[Stroke], field: Field | None, dpi: float, field_dpi: float
) -> list[tuple[Stroke, Stroke]]:
    """Пары «отрезок B — тот же отрезок в A». Поле смещений задано в пикселях ``field_dpi``."""
    if not before or not after:
        return []
    k = field_dpi / dpi
    mids_b = np.array([s.mid for s in before])
    moved = field.transform(mids_b * k) / k if field is not None else mids_b
    mids_a = np.array([s.mid for s in after])
    len_a = np.array([s.length for s in after])
    ang_a = np.array([s.angle_deg for s in after])
    perp_tol = MATCH_PERP_MM * dpi / 25.4
    pairs: list[tuple[Stroke, Stroke]] = []
    taken: set[int] = set()
    for stroke, mid in zip(before, moved):
        theta = np.radians(stroke.angle_deg)
        normal = np.array([-np.sin(theta), np.cos(theta)])
        along = np.array([np.cos(theta), np.sin(theta)])
        delta = mids_a - mid
        ok = (
            (np.abs(_wrap(ang_a - stroke.angle_deg)) < MATCH_ANGLE_DEG)
            & (np.abs(len_a - stroke.length) < MATCH_LENGTH_FRAC * stroke.length)
            & (np.abs(delta @ along) < 0.5 * stroke.length)
            & (np.abs(delta @ normal) <= perp_tol)
        )
        candidates = [j for j in np.nonzero(ok)[0] if j not in taken]
        if not candidates:
            continue
        j = min(candidates, key=lambda j: abs(float(delta[j] @ normal)))
        taken.add(j)
        pairs.append((stroke, after[j]))
    return pairs


def _weighted_tilt(items: list[tuple[float, float]]) -> float:
    """Средний |наклон к оси|, взвешенный по длине; ``items`` — (наклон, длина)."""
    if not items:
        return 0.0
    tilt = np.array([t for t, _ in items])
    weight = np.array([w for _, w in items])
    return float((tilt * weight).sum() / weight.sum())


def stroke_metrics(
    before: list[Stroke], after: list[Stroke], pairs: list[tuple[Stroke, Stroke]], rot_deg: float, dpi: float
) -> tuple[dict[str, float], dict]:
    """Метрики по парам отрезков и рамки-виновники для оверлея (в пикселях ``dpi``)."""
    metrics: dict[str, float] = {
        "strokes_b": float(len(before)),
        "strokes_a": float(len(after)),
        "strokes_matched": float(len(pairs)),
        "strokes_lost_frac": 0.0 if not before else 1.0 - len(pairs) / len(before),
    }
    culprits: dict = {}

    # Самый длинный штрих B, которому в A не нашлось прямого двойника: он погнулся или порвался.
    matched_b = {id(b) for b, _ in pairs}
    lost = [b for b in before if id(b) not in matched_b]
    if lost:
        worst = max(lost, key=lambda s: s.length)
        metrics["stroke_lost_max_mm"] = px_to_mm(worst.length, dpi)
        culprits["stroke_lost_max_mm"] = {"b": worst.box, "a": worst.box}
    else:
        metrics["stroke_lost_max_mm"] = 0.0

    # Наклон к оси. Что видно глазу — не угол, а на сколько миллиметров конец штриха ушёл
    # от оси: короткий штрих под 2° (с.2 в 1967/02, 9 мм → 0.3 мм) незаметен, линейка
    # таблицы в 50 мм под 1.2° (с.38 в 1967/01, 1 мм) бросается в глаза. Поэтому максимум
    # берётся по отклонению ``длина × sin(наклон)``, а средний наклон по длине — контекст.
    for orient, low, high in (("h", 0.0, AXIS_TOL_DEG), ("v", 90.0 - AXIS_TOL_DEG, 90.0)):
        own = [(b, a) for b, a in pairs if low <= abs(b.angle_deg) <= high and b.axis_tilt is not None]
        key = f"{orient}stroke"
        metrics[f"{key}_tilt_wmean_b"] = _weighted_tilt([(b.axis_tilt, b.length) for b, _ in own])
        metrics[f"{key}_tilt_wmean_a"] = _weighted_tilt([(a.axis_tilt or 0.0, a.length) for _, a in own])
        metrics[f"{key}_tilt_wmean_delta"] = metrics[f"{key}_tilt_wmean_a"] - metrics[f"{key}_tilt_wmean_b"]
        if own:
            deltas = [
                px_to_mm(b.length, dpi) * (np.sin(np.radians(a.axis_tilt or 0.0)) - np.sin(np.radians(b.axis_tilt)))
                for b, a in own
            ]
            worst = int(np.argmax(deltas))
            metrics[f"{key}_dev_max_delta_mm"] = float(deltas[worst])
            culprits[f"{key}_dev_max_delta_mm"] = {"b": own[worst][0].box, "a": own[worst][1].box}
        else:
            metrics[f"{key}_dev_max_delta_mm"] = 0.0

    # Поворот сверх общего.
    if pairs:
        extra = np.abs(_wrap(np.array([a.angle_deg - b.angle_deg for b, a in pairs]) - rot_deg))
        metrics["stroke_rot_p90"] = float(np.percentile(extra, 90))
        metrics["stroke_rot_max"] = float(extra.max())
    else:
        metrics["stroke_rot_p90"] = metrics["stroke_rot_max"] = 0.0

    # Параллельность: группы отрезков B по углу вокруг самого длинного, разброс до и после.
    metrics["parallel_groups"] = 0.0
    metrics["parallel_spread_delta_max"] = 0.0
    if pairs:
        ang_b = np.array([b.angle_deg for b, _ in pairs])
        ang_a = np.array([a.angle_deg for _, a in pairs])
        order = np.argsort([-b.length for b, _ in pairs])
        used = np.zeros(len(pairs), dtype=bool)
        best = None
        for k in order:
            if used[k]:
                continue
            group = np.nonzero((np.abs(_wrap(ang_b - ang_b[k])) < GROUP_TOL_DEG) & ~used)[0]
            if len(group) < GROUP_MIN:
                continue
            used[group] = True
            spread_b = float(np.ptp(_wrap(ang_b[group] - ang_b[k])))
            spread_a = float(np.ptp(_wrap(ang_a[group] - ang_a[k])))
            metrics["parallel_groups"] += 1.0
            delta = spread_a - spread_b
            if best is None or delta > best[0]:
                best = (delta, group)
        if best is not None:
            delta, group = best
            metrics["parallel_spread_delta_max"] = float(delta)
            xs = [v for i in group for v in (pairs[i][1].x0, pairs[i][1].x1)]
            ys = [v for i in group for v in (pairs[i][1].y0, pairs[i][1].y1)]
            xb = [v for i in group for v in (pairs[i][0].x0, pairs[i][0].x1)]
            yb = [v for i in group for v in (pairs[i][0].y0, pairs[i][0].y1)]
            culprits["parallel_spread_delta_max"] = {
                "b": (int(min(xb)), int(min(yb)), int(max(xb)) + 1, int(max(yb)) + 1),
                "a": (int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1),
            }
    return metrics, culprits
