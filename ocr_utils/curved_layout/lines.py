"""Осевая кривая строки: ломаная через все символы строки примерно по её центру.

Ось строится не подгонкой параболы (как в ``curved_lines.fitting``), а пересборкой самой
центр-линии: точки краски усредняются по окну ``AXIS_STEP_MM``, затем сглаживаются медианой и
Савицким–Голеем окном в долях высоты строки. Парабола на кривой бумаге размазывает прогиб,
сидящий в одной четверти строки (это же было причиной отказа от неё в ``dewarp.textline``), а
ось по краске повторяет форму строки как есть.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from ocr_utils.curved_layout import WORK_DPI
from ocr_utils.curved_layout.engines.base import EngineLine
from ocr_utils.page_layout import mm_to_px, px_to_mm
from ocr_utils.curved_layout.leaders import inside_spans
from ocr_utils.scan_markup.curved_lines.fitting import fit_line, smooth_median

# Шаг, с которым ось пересобирается: 1 мм — мельче кегля, но крупнее формы буквы.
AXIS_STEP_MM = 1.0
# Окно сглаживания оси в долях высоты строки (умолчание; настраивается снаружи). 0.6 высоты
# гасит прыжки центра масс на выносных элементах и знаках препинания, но оставляет форму строки.
SMOOTH_HEIGHTS = 0.6
# Короче этого строка осью не описывается. 4 мм — столько занимают три буквы корпуса, а строка
# от трёх букв уже строка (решение пользователя): при 8 мм без оси оставались концевые строки
# сносок («ч. II, с. 421» — 1973/08 с.85) и числовые графы таблиц.
MIN_LENGTH_MM = 4.0
# Точек оси меньше — строку не берём (нужны хотя бы три отсчёта для сглаживания).
MIN_POINTS = 4


@dataclass(frozen=True)
class LineAxis:
    """Ось строки: ломаная ``(N, 2)`` в пикселях рабочей копии плюс сводки её формы.

    ``points`` идут слева направо от первого символа до последнего; ``height`` — высота строки
    в тех же пикселях; ``column`` — номер колонки; ``cross`` — строка на самом деле кусок
    широкой строки (заголовка), разрезанной межколонником.
    """

    points: np.ndarray
    height: float
    column: int
    dpi: float
    cross: bool  # кусок строки, разрезанной межколонником (заголовок во всю ширину)
    sagitta_mm: float  # прогиб относительно прямой по концам, со знаком (вниз — плюс)
    slope_deg: float  # наклон прямой по концам оси
    bend_mm: float  # размах остатка оси от прямой: «насколько строка непрямая»
    resid_parabola_mm: float  # размах остатка оси от ПАРАБОЛЫ: где параболы не хватает
    # Отрезки по x, занятые точками и запятыми: там ось провисает к базовой линии, и в меры
    # формы строки эти участки не входят (см. ``_shape_stats``).
    mark_spans: tuple[tuple[float, float], ...] = ()

    @property
    def x0(self) -> float:
        return float(self.points[0, 0])

    @property
    def x1(self) -> float:
        return float(self.points[-1, 0])

    @property
    def cy(self) -> float:
        return float(np.median(self.points[:, 1]))

    @property
    def length_mm(self) -> float:
        return px_to_mm(self.x1 - self.x0, self.dpi)

    def y_at(self, x: float) -> float:
        """Ордината оси в точке ``x`` (линейная интерполяция, за концами — концевое значение)."""
        return float(np.interp(x, self.points[:, 0], self.points[:, 1]))


def resample(points: np.ndarray, step_px: float) -> np.ndarray:
    """Пересборка ломаной с постоянным шагом по x: в каждом окне — медиана ординат.

    Args:
        points: Исходные точки ``(N, 2)``, отсортированные по x.
        step_px: Шаг сетки по x в пикселях.

    Returns:
        Ломаная ``(M, 2)`` с шагом ``step_px``; окна без точек пропускаются.
    """
    xs, ys = points[:, 0], points[:, 1]
    if xs.size == 0:
        return points
    grid = np.arange(xs.min(), xs.max() + step_px, step_px)
    if xs.size < grid.size:
        # Редкая ломаная (базовая линия нейросетевого движка — три-четыре точки на строку):
        # сетка заполняется интерполяцией, иначе после пересборки точек меньше минимума.
        return np.column_stack([grid, np.interp(grid, xs, ys)])
    index = np.clip(np.searchsorted(grid, xs, side="right") - 1, 0, len(grid) - 1)
    out_x: list[float] = []
    out_y: list[float] = []
    for cell in np.unique(index):
        own = ys[index == cell]
        out_x.append(float(grid[cell] + step_px / 2.0))
        out_y.append(float(np.median(own)))
    return np.column_stack([out_x, out_y])


def smooth_axis(points: np.ndarray, window_px: float) -> np.ndarray:
    """Сглаживание оси: медиана окном (выбросы) и Савицкий–Голей (гладкость), окно в пикселях."""
    if points.shape[0] < MIN_POINTS:
        return points
    step = float(np.median(np.diff(points[:, 0]))) if points.shape[0] > 1 else 1.0
    window = max(3, int(round(window_px / max(step, 1e-6))) | 1)
    ys = smooth_median(points[:, 1], window)
    if ys.size > window:
        # Полином второй степени в окне: сохраняет дугу и не срезает перегибы, в отличие от среднего.
        ys = savgol_filter(ys, window_length=window, polyorder=2, mode="nearest")
    return np.column_stack([points[:, 0], ys])


def _shape_stats(
    points: np.ndarray, dpi: float, mark_spans: tuple[tuple[float, float], ...] = ()
) -> tuple[float, float, float, float]:
    """Сводки формы оси: сагитта, наклон, размах остатка от прямой и от параболы (мм, градусы).

    Столбцы точек и запятых из счёта исключаются: эти знаки стоят на БАЗОВОЙ линии, ось там
    провисает на полвысоты строчной, и без отсева строка выглядит круче и кривее, чем она есть
    («Энгельс Ф.» в конце сноски 1973/08 с.85 задирал размах остатка вдвое). Если после отсева
    точек осталось меньше четырёх, меры считаются по всем — лучше огрублённая мера, чем никакой.
    """
    if mark_spans:
        keep = ~inside_spans(points[:, 0], list(mark_spans))
        if int(keep.sum()) >= MIN_POINTS:
            points = points[keep]
    xs, ys = points[:, 0], points[:, 1]
    length = float(xs[-1] - xs[0])
    if length <= 0:
        return 0.0, 0.0, 0.0, 0.0
    slope = (ys[-1] - ys[0]) / length
    chord = ys[0] + slope * (xs - xs[0])
    resid = ys - chord
    middle = float(np.interp((xs[0] + xs[-1]) / 2.0, xs, resid))
    bend = float(resid.max() - resid.min())
    fit = fit_line(xs, ys)
    resid_parabola = 0.0
    if fit is not None:
        # Остаток от параболы: подгоняем её заново по тем же точкам и берём размах.
        coeffs = np.polyfit(xs - xs.mean(), ys, 2)
        resid_quad = ys - np.polyval(coeffs, xs - xs.mean())
        resid_parabola = float(resid_quad.max() - resid_quad.min())
    return (
        px_to_mm(middle, dpi),
        float(np.degrees(np.arctan(slope))),
        px_to_mm(bend, dpi),
        px_to_mm(resid_parabola, dpi),
    )


def axis_of(line: EngineLine, dpi: float = WORK_DPI, smooth_heights: float = SMOOTH_HEIGHTS) -> LineAxis | None:
    """Ось одной строки из результата поставщика.

    Args:
        line: Строка от движка: центр-линия или базовая линия плюс высота.
        dpi: Разрешение рабочей копии, в котором заданы точки.
        smooth_heights: Окно сглаживания в долях высоты строки.

    Returns:
        :class:`LineAxis` или ``None``, если строка коротка или точек мало. Базовая линия
        поднимается на половину высоты: ось должна идти по центру символов, а не под ними.
    """
    points = np.asarray(line.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2:
        return None
    points = points[np.argsort(points[:, 0])]
    if px_to_mm(points[-1, 0] - points[0, 0], dpi) < MIN_LENGTH_MM:
        return None
    if line.baseline:
        points = points.copy()
        points[:, 1] -= line.height / 2.0
    points = resample(points, mm_to_px(AXIS_STEP_MM, dpi))
    if points.shape[0] < MIN_POINTS:
        return None
    points = smooth_axis(points, max(smooth_heights * line.height, mm_to_px(AXIS_STEP_MM, dpi) * 3))
    sagitta, slope, bend, resid = _shape_stats(points, dpi, tuple(line.mark_spans))
    return LineAxis(
        points=points,
        height=float(line.height),
        column=0,
        dpi=float(dpi),
        cross=False,
        sagitta_mm=sagitta,
        slope_deg=slope,
        bend_mm=bend,
        resid_parabola_mm=resid,
        mark_spans=tuple(line.mark_spans),
    )


def axes_of(lines: list[EngineLine], dpi: float = WORK_DPI, smooth_heights: float = SMOOTH_HEIGHTS) -> list[LineAxis]:
    """Оси всех строк страницы, сверху вниз; строки без оси выбрасываются."""
    axes = [axis for axis in (axis_of(line, dpi, smooth_heights) for line in lines) if axis is not None]
    return sorted(axes, key=lambda axis: axis.cy)


def with_column(axis: LineAxis, column: int, cross: bool | None = None) -> LineAxis:
    """Та же ось с проставленным номером колонки и, если задан, признаком разрезанной строки."""
    return LineAxis(
        points=axis.points,
        height=axis.height,
        column=column,
        dpi=axis.dpi,
        cross=axis.cross if cross is None else cross,
        sagitta_mm=axis.sagitta_mm,
        slope_deg=axis.slope_deg,
        bend_mm=axis.bend_mm,
        resid_parabola_mm=axis.resid_parabola_mm,
        mark_spans=axis.mark_spans,
    )


__all__ = ["LineAxis", "axes_of", "axis_of", "resample", "smooth_axis", "with_column"]
