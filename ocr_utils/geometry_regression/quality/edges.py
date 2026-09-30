"""Качество краёв текстовых блоков: размах выровненной стороны по горизонтали на линии с заплатками поверх аномалий, порча и выигрыш FineReader по парам сторон B ↔ A.

КАКИЕ СТОРОНЫ. Только те, по которым текст блока выровнен (``BlockRecord.aligned_left/right`` —
выключка ``text_blocks.alignment``): у рваного края нет линии, которую FineReader мог бы испортить.

КАКАЯ ЛИНИЯ. Дополнительная линия стороны (``sides.filled_side``, в сайдкаре ``SideRecord.points``):
невыровненные концы отброшены, невыровненная середина (отступ, короткая строка) и недостоверные участки
(ступенька выноса, выступ сора ``edge_guard``) заменены гладкой интерполяцией PCHIP между нормальными
участками. Для сравнения считается и сырая сторона огибающей на той же высоте (``edge_quality_raw_mm``).

МЕРА. Размах линии по x (мм): наклон стороны длиной L под углом θ даёт L·tg θ, изгиб — свою сагитту;
как и у строк, одна мера на наклон и кривизну. Сравнение — на общем для B и A отрезке по высоте.

ВЕС. Большой блок весит больше заголовка в 2–3 строки: ``w = min(1, ряды / EDGE_FULL_ROWS)``. Сама мера в мм
и так растёт с длиной стороны; вес гасит короткие стороны, где размах в пару пикселей — уже градус.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocr_utils.geometry_regression.field import Field
from ocr_utils.page_layout import px_to_mm
from ocr_utils.page_layout.text_blocks.store import PageGeometry, PointKind, SideCode
from ocr_utils.geometry_regression.quality.lines import spread_mm

# Рядов в блоке для полного веса стороны.
EDGE_FULL_ROWS = 8
# Блок меньше — сторона не меряется (одна-две строки — не край колонки).
EDGE_MIN_ROWS = 2
# Общий отрезок сторон короче — пара не меряется, мм.
EDGE_MIN_MM = 10.0
# Точек линии меньше — сторона не берётся.
MIN_POINTS = 4
# Пара B ↔ A: перекрытие по высоте от короткой и допуск по x, мм.
MATCH_OVERLAP = 0.6
MATCH_DX_MM = 3.0


@dataclass(frozen=True)
class EdgePair:
    """Пара выровненных сторон B ↔ A с мерами на общем отрезке.

    Attributes:
        block_b: Блок в B.
        block_a: Блок в A.
        side: Сторона.
        rows: Рядов в блоке B.
        e_b: Размах линии B, мм.
        e_a: Размах линии A, мм.
        e_raw_b: Размах сырой стороны огибающей B на том же отрезке, мм.
        e_raw_a: То же у A.
        patched_b: Доля точек линии B на заплатках над аномалиями.
        length_mm: Длина общего отрезка, мм.
        points_b: Линия B на общем отрезке (для оверлея).
        points_a: Линия A на общем отрезке.
    """

    block_b: int
    block_a: int
    side: SideCode
    rows: int
    e_b: float
    e_a: float
    e_raw_b: float
    e_raw_a: float
    patched_b: float
    length_mm: float
    points_b: np.ndarray
    points_a: np.ndarray

    @property
    def weight(self) -> float:
        """Вес стороны по числу рядов блока."""
        return min(1.0, self.rows / EDGE_FULL_ROWS)

    @property
    def delta(self) -> float:
        """Порча стороны со знаком, мм с весом: плюс — в A край стал косее или кривее."""
        return self.weight * (self.e_a - self.e_b)

    @property
    def delta_raw(self) -> float:
        """То же по сырой стороне огибающей (без заплаток) — для сравнения шумности."""
        return self.weight * (self.e_raw_a - self.e_raw_b)


def _sides(geometry: PageGeometry) -> list[tuple[int, SideCode, np.ndarray, np.ndarray, np.ndarray, int]]:
    """Выровненные стороны разбора: ``(блок, сторона, линия, виды точек, сырая сторона, ряды)``."""
    out = []
    for block in geometry.blocks:
        if block.rows < EDGE_MIN_ROWS:
            continue
        for code, aligned in ((SideCode.LEFT, block.aligned_left), (SideCode.RIGHT, block.aligned_right)):
            side = block.sides.get(code)
            if not aligned or side is None or len(side.points) < MIN_POINTS:
                continue
            order = np.argsort(side.points[:, 1], kind="stable")
            out.append((block.index, code, side.points[order], side.kinds[order], side.raw, block.rows))
    return out


def _between(points: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Точки с y в отрезке ``[lo, hi]``."""
    return points[(points[:, 1] >= lo) & (points[:, 1] <= hi)]


def match_edges(b: PageGeometry, a: PageGeometry, field: Field | None) -> list[EdgePair]:
    """Пары выровненных сторон B ↔ A с мерами на общем отрезке по высоте.

    Args:
        b: Разбор без коррекции.
        a: Разбор с коррекцией.
        field: Поле смещений B → A; ``None`` — без переноса.

    Returns:
        Пары; каждая сторона A — не больше чем в одной паре (ближайшая по x).
    """
    dpi = b.dpi
    sides_a = _sides(a)
    candidates = []
    for block_b, code, line_b, kinds_b, raw_b, rows in _sides(b):
        mapped = field.transform(line_b) if field is not None else line_b
        lo_b, hi_b = float(mapped[:, 1].min()), float(mapped[:, 1].max())
        best = None
        for index, (block_a, code_a, line_a, _, _, _) in enumerate(sides_a):
            if code_a is not code:
                continue
            lo, hi = max(lo_b, line_a[0, 1]), min(hi_b, line_a[-1, 1])
            shorter = min(hi_b - lo_b, line_a[-1, 1] - line_a[0, 1])
            if shorter <= 0 or hi - lo < MATCH_OVERLAP * shorter:
                continue
            ys = np.linspace(lo, hi, 12)
            order = np.argsort(mapped[:, 1], kind="stable")
            dx = np.median(
                np.abs(np.interp(ys, mapped[order, 1], mapped[order, 0]) - np.interp(ys, line_a[:, 1], line_a[:, 0]))
            )
            if px_to_mm(float(dx), dpi) > MATCH_DX_MM:
                continue
            if best is None or dx < best[0]:
                best = (float(dx), index, lo, hi)
        if best is not None:
            candidates.append((best, block_b, code, line_b, kinds_b, raw_b, rows, mapped))
    used: set[int] = set()
    out = []
    for (dx, index, lo, hi), block_b, code, line_b, kinds_b, raw_b, rows, mapped in sorted(
        candidates, key=lambda c: c[0][0]
    ):
        if index in used:
            continue
        block_a, _, line_a, _, raw_a, _ = sides_a[index]
        inside = (mapped[:, 1] >= lo) & (mapped[:, 1] <= hi)
        common_b, common_a = line_b[inside], _between(line_a, lo, hi)
        if len(common_b) < MIN_POINTS or len(common_a) < MIN_POINTS or px_to_mm(hi - lo, dpi) < EDGE_MIN_MM:
            continue
        used.add(index)
        # Сырая сторона — на той же высоте в своём кадре: у B — отрезок по y самой линии B.
        raw_common_b = _between(raw_b, float(common_b[:, 1].min()), float(common_b[:, 1].max()))
        raw_common_a = _between(raw_a, lo, hi)
        out.append(
            EdgePair(
                block_b=block_b,
                block_a=block_a,
                side=code,
                rows=rows,
                e_b=spread_mm(common_b, dpi, column=0),
                e_a=spread_mm(common_a, dpi, column=0),
                e_raw_b=spread_mm(raw_common_b, dpi, column=0),
                e_raw_a=spread_mm(raw_common_a, dpi, column=0),
                patched_b=float(np.mean(kinds_b[inside] == PointKind.ANOMALY)) if inside.any() else 0.0,
                length_mm=px_to_mm(hi - lo, dpi),
                points_b=common_b,
                points_a=common_a,
            )
        )
    return out


def _box(points: np.ndarray) -> list[float]:
    return [float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())]


def edge_metrics(pairs: list[EdgePair]) -> tuple[dict[str, float], dict[str, dict]]:
    """Метрики страницы по парам сторон.

    * ``edge_quality_mm`` — худшая порча выровненной стороны: ``max w·(E_A − E_B)``;
    * ``edge_quality_gain_mm`` — лучший выигрыш стороны ``max w·(E_B − E_A)``;
    * ``edge_quality_raw_mm`` — та же порча по сырой стороне огибающей (без заплаток), для сравнения;
    * счётчики.

    Args:
        pairs: Пары сторон (:func:`match_edges`).

    Returns:
        ``(метрики, виновники)``.
    """
    metrics = {
        "edge_pairs": float(len(pairs)),
        "edge_quality_mm": 0.0,
        "edge_quality_gain_mm": 0.0,
        "edge_quality_raw_mm": 0.0,
    }
    culprits: dict[str, dict] = {}
    if not pairs:
        return metrics, culprits
    worst = max(pairs, key=lambda p: p.delta)
    if worst.delta > 0:
        metrics["edge_quality_mm"] = worst.delta
        culprits["edge_quality_mm"] = {"b": _box(worst.points_b), "a": _box(worst.points_a)}
    best = min(pairs, key=lambda p: p.delta)
    if best.delta < 0:
        metrics["edge_quality_gain_mm"] = -best.delta
        culprits["edge_quality_gain_mm"] = {"b": _box(best.points_b), "a": _box(best.points_a)}
    metrics["edge_quality_raw_mm"] = max(0.0, max(p.delta_raw for p in pairs))
    return metrics, culprits


__all__ = ["EdgePair", "edge_metrics", "match_edges"]
