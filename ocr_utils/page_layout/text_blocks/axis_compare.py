"""Стенд сравнения двух осей строки: прежней (центр масс краски) и второй (по базовой линии, :mod:`baseline_axis`).

На каждую полосу — меры обеих осей в CSV и картинки «было — стало» на ОДНОМ холсте: прежняя ось
полупрозрачным оранжевым, вторая — полупрозрачным зелёным, низы глифов точками; вырезки с
увеличением там, где оси расходятся сильнее всего. Шапка и легенда — в полях картинки.

Меры (все в миллиметрах бумаги или градусах):

* ``mid_*`` — отклонение середины строчной без выносных от оси (медиана, p90): мера, в которой
  класс соседних букв не участвует; ``tall_*`` — то же рядом с прописными и цифрами, где прежняя ось
  прыгает;
* ``wobble`` — медиана по строкам СКО второй разности оси на шаге 1 мм;
* ``nb_p90``, ``nb_end_p90`` — p90 разности наклонов соседних строк в общей точке x, по всей длине и
  на крайних 15 % (изгиб у края полосы);
* ``crossings`` — скрещивания осей (:func:`metrics.crossings_of`).
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout import mm_to_px, px_to_mm
from ocr_utils.page_layout.overlay_frame import LegendEntry, framed
from ocr_utils.page_layout.text_blocks import RENDER_DPI
from ocr_utils.page_layout.text_blocks.baseline_axis import line_x_height
from ocr_utils.page_layout.text_blocks.lines import LineAxis
from ocr_utils.page_layout.text_blocks.metrics import crossings_of
from ocr_utils.page_layout.text_blocks.page import PageAnalysis

# Цвета (BGR) и прозрачность осей на оверлее: прежняя — оранжевая, вторая — зелёная (как ось строки
# в палитре проекта), низы глифов — фиолетовые.
COLOUR_CENTRE = (0, 140, 255)
COLOUR_BODY = (40, 170, 40)
COLOUR_BOTTOM = (200, 60, 200)
AXIS_ALPHA = 0.6
# Ширина страницы на общем оверлее, увеличение вырезок и их размер (пиксели рабочей копии).
PAGE_WIDTH = 1400
CROP_ZOOM = 3
CROP_HALF_WIDTH = 170
CROP_HALF_HEIGHT = 70
CROPS = 4
# Строчная без выносных: высота в пределах этой доли от высоты строчной строки; «высокий» глиф
# (прописная, цифра) — выше стольких строчных, но не мостик.
X_CLASS_TOLERANCE = 0.2
TALL_FROM_XH = 1.3
TALL_TO_XH = 2.3
# Строчная «рядом с высоким глифом» — не дальше стольких строчных по x.
NEAR_TALL_XH = 2.5
# Шаг, на котором считается дрожание оси, мм.
WOBBLE_STEP_MM = 1.0
# Соседняя строка: ниже не больше чем на столько высот строки и перекрыта по x на такую долю.
NEIGHBOUR_HEIGHTS = 2.2
NEIGHBOUR_OVERLAP = 0.5
EDGE_SHARE = 0.15


@dataclass(frozen=True)
class AxisMeasures:
    """Меры одной оси (прежней или второй) по всей полосе; ``nan`` — посчитать не из чего."""

    mid_med: float
    mid_p90: float
    tall_med: float
    tall_p90: float
    wobble: float
    nb_p90: float
    nb_end_p90: float
    crossings: int


def _curve(axis: LineAxis, which: str) -> np.ndarray | None:
    """Точки оси: ``centre`` — прежняя (``centre_points`` или ``points``), ``body`` — вторая."""
    if which == "body":
        return axis.body_points
    return axis.centre_points if axis.centre_points is not None else axis.points


def _percentile(values: list[float], q: float) -> float:
    """Перцентиль списка; ``nan``, если он пуст."""
    return float(np.percentile(values, q)) if values else float("nan")


def measures_of(analysis: PageAnalysis, which: str) -> AxisMeasures:
    """Меры одной из осей по всем строкам полосы.

    Args:
        analysis: Разбор полосы (у осей должны быть глифы и вторая ось).
        which: ``centre`` или ``body``.

    Returns:
        :class:`AxisMeasures`.
    """
    dpi = analysis.dpi
    mids: list[float] = []
    near_tall: list[float] = []
    wobble: list[float] = []
    curves: list[np.ndarray | None] = []
    for axis in analysis.axes:
        points = _curve(axis, which)
        glyphs = axis.glyphs
        if points is None or glyphs is None or glyphs.shape[0] < 3 or points.shape[0] < 2:
            curves.append(None)
            continue
        curves.append(points)
        x_h = line_x_height(glyphs)
        heights = glyphs[:, 3] - glyphs[:, 1]
        centres = 0.5 * (glyphs[:, 0] + glyphs[:, 2])
        inside = (centres >= points[0, 0]) & (centres <= points[-1, 0])
        x_class = (np.abs(heights - x_h) <= X_CLASS_TOLERANCE * x_h) & inside
        tall = (heights > TALL_FROM_XH * x_h) & (heights < TALL_TO_XH * x_h)
        deviation = np.abs(0.5 * (glyphs[:, 1] + glyphs[:, 3]) - np.interp(centres, points[:, 0], points[:, 1]))
        mids.extend(px_to_mm(deviation[x_class], dpi))
        if tall.any():
            close = np.abs(centres[:, None] - centres[tall][None, :]).min(axis=1) <= NEAR_TALL_XH * x_h
            near_tall.extend(px_to_mm(deviation[x_class & close], dpi))
        grid = np.arange(points[0, 0], points[-1, 0], mm_to_px(WOBBLE_STEP_MM, dpi))
        if grid.size >= 5:
            wobble.append(float(np.std(np.diff(px_to_mm(np.interp(grid, points[:, 0], points[:, 1]), dpi), 2))))
    slopes_all, slopes_end = _neighbour_slopes(analysis.axes, curves)
    crossing = crossings_of([curve for curve in curves if curve is not None])
    return AxisMeasures(
        mid_med=_percentile(mids, 50),
        mid_p90=_percentile(mids, 90),
        tall_med=_percentile(near_tall, 50),
        tall_p90=_percentile(near_tall, 90),
        wobble=float(np.median(wobble)) if wobble else float("nan"),
        nb_p90=_percentile(slopes_all, 90),
        nb_end_p90=_percentile(slopes_end, 90),
        crossings=int(crossing),
    )


def _neighbour_slopes(axes: tuple[LineAxis, ...], curves: list[np.ndarray | None]) -> tuple[list[float], list[float]]:
    """Разности наклонов строки и ближайшей строки ниже (градусы): по всей общей длине и на краях."""
    everywhere: list[float] = []
    edges: list[float] = []
    order = sorted(range(len(axes)), key=lambda index: axes[index].cy)
    for position, index in enumerate(order):
        own = curves[index]
        if own is None:
            continue
        for other_index in order[position + 1 :]:
            other = curves[other_index]
            if other is None:
                continue
            if axes[other_index].cy - axes[index].cy > NEIGHBOUR_HEIGHTS * axes[index].height:
                break
            low, high = max(own[0, 0], other[0, 0]), min(own[-1, 0], other[-1, 0])
            if high - low < NEIGHBOUR_OVERLAP * min(own[-1, 0] - own[0, 0], other[-1, 0] - other[0, 0]):
                continue
            grid = np.linspace(low, high, 60)
            first = np.degrees(np.arctan(np.gradient(np.interp(grid, own[:, 0], own[:, 1]), grid)))
            second = np.degrees(np.arctan(np.gradient(np.interp(grid, other[:, 0], other[:, 1]), grid)))
            difference = np.abs(first - second)
            everywhere.extend(difference)
            edge = (grid - low < EDGE_SHARE * (high - low)) | (high - grid < EDGE_SHARE * (high - low))
            edges.extend(difference[edge])
            break
    return everywhere, edges


def _legend() -> list[LegendEntry]:
    """Легенда оверлея сравнения осей."""
    return [
        LegendEntry("прежняя ось: центр масс краски", COLOUR_CENTRE, AXIS_ALPHA),
        LegendEntry("вторая ось: базовая линия + ½ строчной, соседи", COLOUR_BODY, AXIS_ALPHA),
        LegendEntry("низ глифа", COLOUR_BOTTOM),
    ]


def _draw(canvas: np.ndarray, analysis: PageAnalysis, scale: float, x0: float, y0: float, dots: bool) -> np.ndarray:
    """Обе оси (полупрозрачно) и, если нужно, низы глифов на холсте; координаты — ``(x − x0)·scale``."""
    layer = canvas.copy()
    thickness = max(2, int(round(scale)))
    for axis in analysis.axes:
        for which, colour in (("centre", COLOUR_CENTRE), ("body", COLOUR_BODY)):
            points = _curve(axis, which)
            if points is None:
                continue
            curve = np.column_stack([(points[:, 0] - x0) * scale, (points[:, 1] - y0) * scale]).round().astype(np.int32)
            cv2.polylines(layer, [curve], False, colour, thickness, cv2.LINE_AA)
    out = cv2.addWeighted(layer, AXIS_ALPHA, canvas, 1.0 - AXIS_ALPHA, 0)
    if dots:
        for axis in analysis.axes:
            if axis.glyphs is None:
                continue
            for gx0, _, gx1, gy1 in axis.glyphs:
                centre = (int(((gx0 + gx1) / 2 - x0) * scale), int((gy1 - y0) * scale))
                cv2.circle(out, centre, 2, COLOUR_BOTTOM, -1)
    return out


def _worst_places(analysis: PageAnalysis, count: int) -> list[tuple[float, float]]:
    """Точки, где оси расходятся сильнее всего (по одной на строку, самые сильные строки)."""
    places = []
    for axis in analysis.axes:
        body = axis.body_points
        centre = _curve(axis, "centre")
        if body is None or centre is None:
            continue
        difference = np.abs(np.interp(body[:, 0], centre[:, 0], centre[:, 1]) - body[:, 1])
        index = int(np.argmax(difference))
        places.append((float(difference[index]), float(body[index, 0]), float(body[index, 1])))
    chosen: list[tuple[float, float]] = []
    for _, x, y in sorted(places, reverse=True):
        # Вырезки не накладываются: следующая — не ближе полувырезки к уже выбранным.
        if all(abs(x - cx) > CROP_HALF_WIDTH or abs(y - cy) > CROP_HALF_HEIGHT for cx, cy in chosen):
            chosen.append((x, y))
        if len(chosen) == count:
            break
    return chosen


def write_pictures(analysis: PageAnalysis, gray300: np.ndarray, out_dir: Path) -> list[Path]:
    """Оверлей полосы и вырезки «было — стало» в ``out_dir``.

    Args:
        analysis: Разбор полосы.
        gray300: Серый рендер полосы ``RENDER_DPI``.
        out_dir: Папка выхода.

    Returns:
        Пути записанных картинок.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    k = RENDER_DPI / analysis.dpi
    quality = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    written = []
    scale = PAGE_WIDTH / analysis.width
    page = cv2.resize(gray300, (PAGE_WIDTH, int(round(analysis.height * scale))), interpolation=cv2.INTER_AREA)
    canvas = _draw(cv2.cvtColor(page, cv2.COLOR_GRAY2BGR), analysis, scale, 0.0, 0.0, dots=False)
    path = out_dir / f"{analysis.name}_полоса.jpg"
    cv2.imwrite(str(path), framed(canvas, [f"{analysis.name}: оси строк, вся полоса"], _legend()), quality)
    written.append(path)
    for number, (x, y) in enumerate(_worst_places(analysis, CROPS), 1):
        x0 = max(0.0, x - CROP_HALF_WIDTH)
        y0 = max(0.0, y - CROP_HALF_HEIGHT)
        x1 = min(float(analysis.width), x + CROP_HALF_WIDTH)
        y1 = min(float(analysis.height), y + CROP_HALF_HEIGHT)
        piece = gray300[int(y0 * k) : int(y1 * k), int(x0 * k) : int(x1 * k)]
        piece = cv2.resize(piece, None, fx=CROP_ZOOM / k, fy=CROP_ZOOM / k, interpolation=cv2.INTER_AREA)
        canvas = _draw(cv2.cvtColor(piece, cv2.COLOR_GRAY2BGR), analysis, CROP_ZOOM, x0, y0, dots=True)
        header = [f"{analysis.name}: оси строк, вырезка {number} ({x0:.0f}, {y0:.0f}) ×{CROP_ZOOM}"]
        path = out_dir / f"{analysis.name}_вырезка_{number}.jpg"
        cv2.imwrite(str(path), framed(canvas, header, _legend()), quality)
        written.append(path)
    return written


def write_csv(rows: list[tuple[str, AxisMeasures, AxisMeasures]], path: Path) -> None:
    """Меры обеих осей по полосам в CSV: ``полоса``, затем ``centre_*`` и ``body_*``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(AxisMeasures.__dataclass_fields__)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["полоса"] + [f"centre_{name}" for name in names] + [f"body_{name}" for name in names])
        for key, centre, body in rows:
            writer.writerow([key] + [asdict(centre)[name] for name in names] + [asdict(body)[name] for name in names])


def summary(rows: list[tuple[str, AxisMeasures, AxisMeasures]]) -> str:
    """Таблица-сводка: медианы мер по полосам для обеих осей (markdown)."""
    names = list(AxisMeasures.__dataclass_fields__)
    lines = ["| ось | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for label, index in (("прежняя", 1), ("вторая", 2)):
        values = np.array([[float(asdict(row[index])[name]) for name in names] for row in rows], dtype=float)
        medians = np.nanmedian(values, axis=0)
        lines.append(f"| {label} | " + " | ".join(f"{value:.3f}" for value in medians) + " |")
    return "\n".join(lines)


def compare_page(key: str, sharpened_dir: Path, out_dir: Path) -> tuple[str, AxisMeasures, AxisMeasures]:
    """Разбор одной полосы пака, её картинки «было — стало» и меры обеих осей (воркер пула).

    Args:
        key: Ключ полосы «год/выпуск/полоса».
        sharpened_dir: Корень заострённых копий пака.
        out_dir: Папка выхода (картинки — в ``out_dir/картинки``).

    Returns:
        ``(ключ, меры прежней оси, меры второй оси)``.
    """
    from ocr_utils.page_layout.image import PageImage, Variant
    from ocr_utils.page_layout.pack_analysis.stages import DEFAULT_DPI
    from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
    from ocr_utils.page_layout.text_blocks.page import analyse_gray

    image = PageImage.from_file(sharpened_dir / f"{key}.jpg", Variant.SHARPENED, key, default_dpi=DEFAULT_DPI)
    gray300 = image.gray_at(RENDER_DPI)
    analysis = analyse_gray(gray300, InkEngine(), name=key.replace("/", "_"))
    write_pictures(analysis, gray300, out_dir / "картинки")
    return key, measures_of(analysis, "centre"), measures_of(analysis, "body")


__all__ = ["AxisMeasures", "compare_page", "measures_of", "summary", "write_csv", "write_pictures"]
