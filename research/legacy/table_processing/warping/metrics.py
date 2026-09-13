"""Меры геометрии таблицы: кривизна линеек и регулярность решётки.

ВСЕ МЕРЫ БЕЗЭТАЛОННЫЕ. Эталона выпрямленной таблицы у нас нет и быть не может (в
литературе для этого держат специально снятые пары), поэтому меряется не «похоже на
правильное», а «насколько само себе противоречит»: прямая линейка обязана быть прямой,
параллельные — параллельными, а пересечения линеек обязаны складываться в прямоугольную
решётку. Всё это проверяется по самой картинке.

ЕДИНИЦЫ — МИЛЛИМЕТРЫ, а не пиксели. Одна и та же таблица меряется на скане 600 dpi и на
рендере страницы PDF 150 dpi, и сравнивать их в пикселях нельзя.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research.legacy.table_processing.detection.ruling import find_lines
from research.legacy.table_processing.warping.trace import Polyline, crossings, trace

MM_PER_INCH = 25.4


@dataclass(frozen=True)
class Geometry:
    """Геометрия одной таблицы. Пустые поля означают «мерить было не по чему»."""

    rules_h: int
    rules_v: int
    sagitta_max_mm: float
    sagitta_p90_mm: float
    angle_median_deg: float
    angle_spread_deg: float
    lattice_rms_mm: float
    crossings: int

    def as_row(self) -> dict[str, float | int]:
        return {
            "rules_h": self.rules_h,
            "rules_v": self.rules_v,
            "sagitta_max_mm": round(self.sagitta_max_mm, 3),
            "sagitta_p90_mm": round(self.sagitta_p90_mm, 3),
            "angle_median_deg": round(self.angle_median_deg, 3),
            "angle_spread_deg": round(self.angle_spread_deg, 3),
            "lattice_rms_mm": round(self.lattice_rms_mm, 3),
            "crossings": self.crossings,
        }

    @property
    def measurable(self) -> bool:
        return self.rules_h + self.rules_v >= 2


HEADER = (
    "rules_h",
    "rules_v",
    "sagitta_max_mm",
    "sagitta_p90_mm",
    "angle_median_deg",
    "angle_spread_deg",
    "lattice_rms_mm",
    "crossings",
)

EMPTY = Geometry(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)


def lattice_rms(horizontal: list[Polyline], vertical: list[Polyline]) -> tuple[float, int]:
    """Насколько пересечения линеек отклоняются от прямоугольной решётки, в пикселях.

    Идеальная решётка — та, где у всех пересечений одной колонки один и тот же x, а у всех
    пересечений одной строки один и тот же y. Отклонение от неё и есть мера: перекос её не
    портит (он одинаков для всех), а кривизна и трапеция портят.
    """
    points = crossings(horizontal, vertical)
    if len(points) < 4:
        return 0.0, len(points)
    xs: dict[int, list[float]] = {}
    ys: dict[int, list[float]] = {}
    for h_index, v_index, x, y in points:
        xs.setdefault(v_index, []).append(x)
        ys.setdefault(h_index, []).append(y)
    residuals: list[float] = []
    for values in xs.values():
        if len(values) > 1:
            centre = float(np.mean(values))
            residuals.extend(abs(value - centre) for value in values)
    for values in ys.values():
        if len(values) > 1:
            centre = float(np.mean(values))
            residuals.extend(abs(value - centre) for value in values)
    if not residuals:
        return 0.0, len(points)
    return float(np.sqrt(np.mean(np.square(residuals)))), len(points)


def measure(gray: np.ndarray, dpi: int) -> Geometry:
    """Геометрия таблицы по её вырезке."""
    lines = find_lines(gray, dpi)
    horizontal, vertical = trace(lines, dpi)
    if not horizontal and not vertical:
        return EMPTY

    scale = MM_PER_INCH / dpi
    sagittas = np.array([line.sagitta for line in horizontal + vertical], dtype=float)
    angles = np.array([line.angle_deg for line in horizontal], dtype=float)
    if angles.size == 0:
        angles = np.array([line.angle_deg for line in vertical], dtype=float)
    rms, count = lattice_rms(horizontal, vertical)
    return Geometry(
        rules_h=len(horizontal),
        rules_v=len(vertical),
        sagitta_max_mm=float(sagittas.max()) * scale if sagittas.size else 0.0,
        sagitta_p90_mm=float(np.percentile(sagittas, 90)) * scale if sagittas.size else 0.0,
        angle_median_deg=float(np.median(angles)) if angles.size else 0.0,
        angle_spread_deg=float(np.percentile(angles, 90) - np.percentile(angles, 10)) if angles.size > 2 else 0.0,
        lattice_rms_mm=rms * scale,
        crossings=count,
    )
