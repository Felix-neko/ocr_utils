"""Линейка как ломаная: для каждого x — свой y.

ЗАЧЕМ. Детектор таблиц знает про линейку только габаритный прямоугольник и общий наклон.
Для геометрии этого мало: прямоугольник одинаков и у прямой линейки, и у дуги той же
длины. Чтобы отличить перекос от кривизны — а это разные болезни и разное лечение, — надо
знать, где линейка проходит В КАЖДОЙ точке.

КАК. Маска линеек уже построена морфологией (``detection.ruling.find_lines``), и в ней нет
ничего, кроме линеек. Значит, прослеживать нечего: достаточно взять габарит отрезка и в
каждом столбце посчитать центр тяжести краски. Это векторно, устойчиво к разрывам (столбец
без краски просто выпадает) и не требует ни порогов, ни начального приближения.

ПРЕДЕЛ, КОТОРЫЙ НАДО ЗНАТЬ. Прослеживать можно только то, что нашёл детектор, а он ищет
линейку морфологическим открытием: 8 мм краски подряд В ОДНОЙ СТРОКЕ. Значит, линейка видна,
пока её уход по вертикали внутри этого окна не превышает её собственной толщины (около
0.35 мм). Замер на синтетике: дуга амплитудой 8 px при ширине 520 px уже рвётся на куски, и
прослеженная сагитта не растёт, а падает. На паке до этого далеко — там сагитта 0.2-1.0 мм
на ширине 130 мм, то есть уход внутри окна в двадцать раз меньше предела, — но таблицу,
изогнутую сильнее, детектор не найдёт вовсе, и чинить будет нечего.

ЗАЧЕМ ЦЕНТР ТЯЖЕСТИ, А НЕ СЕРЕДИНА ГАБАРИТА. Линейка на скане размыта, и её край дрожит на
пиксель-другой; центр тяжести усредняет это дрожание, а середина габарита следует за ним.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocr_utils.scan_markup.curved_lines.fitting import fit_line

from research.legacy.table_processing.detection.ruling import Lines, Segment, mm_to_px

# Короче этого линейку не прослеживаем: 10 мм. На более коротком отрезке кривизну не
# отличить от шума разметки края.
MIN_TRACE_MM = 10.0

# Меньше этой доли точек с краской — линейка рваная, мерить по ней нечего.
MIN_COVERAGE = 0.6


@dataclass(frozen=True)
class Polyline:
    """Прослеженная линейка в координатах поданного изображения.

    ``along`` — координата вдоль линейки (x у горизонтальной, y у вертикальной),
    ``across`` — поперёк. Хранится именно так, чтобы обе меры считались одной формулой.
    """

    along: np.ndarray
    across: np.ndarray
    horizontal: bool
    thickness: float

    @property
    def length(self) -> float:
        return float(self.along[-1] - self.along[0]) if self.along.size else 0.0

    @property
    def points(self) -> np.ndarray:
        """Точки ломаной в порядке (x, y) — для контрольных точек выпрямителя."""
        if self.horizontal:
            return np.stack([self.along, self.across], axis=1)
        return np.stack([self.across, self.along], axis=1)

    def straight(self) -> np.ndarray:
        """Прямая, подогнанная по методу наименьших квадратов, в тех же точках."""
        if self.along.size < 2:
            return self.across.copy()
        slope, intercept = np.polyfit(self.along, self.across, 1)
        return slope * self.along + intercept

    @property
    def sagitta(self) -> float:
        """Наибольшее отклонение от прямой, в пикселях. Это и есть «кривизна» на глаз."""
        if self.along.size < 3:
            return 0.0
        return float(np.abs(self.across - self.straight()).max())

    @property
    def angle_deg(self) -> float:
        """Наклон в градусах ПО ЧАСОВОЙ — валюта та же, что у ``scan_markup.rotation``."""
        if self.along.size < 2:
            return 0.0
        slope = float(np.polyfit(self.along, self.across, 1)[0])
        angle = float(np.degrees(np.arctan(slope)))
        # У вертикали положительный наклон «вправо вниз» означает поворот против часовой.
        return angle if self.horizontal else -angle

    def fit(self):
        """Подгонка прямой и параболы функцией из ``curved_lines.fitting``.

        Своей арифметики здесь нет намеренно: ``fit_line`` уже отлажен на кривых строках
        пака и отдаёт сразу наклон, кривизну, сагитту и остатки обеих подгонок.
        """
        return fit_line(self.along.astype(float), self.across.astype(float))


def trace_segment(mask: np.ndarray, segment: Segment) -> "Polyline | None":
    """Ломаная одного отрезка: центр тяжести краски в каждом столбце (или строке)."""
    box = segment.box
    window = mask[box.y0 : box.y1 + 1, box.x0 : box.x1 + 1]
    if window.size == 0:
        return None
    ink = window > 0

    if segment.horizontal:
        # Идём по столбцам: для каждого x — центр тяжести краски по y.
        rows = np.arange(box.y0, box.y0 + window.shape[0], dtype=float)
        count = ink.sum(axis=0).astype(float)
        weighted = (ink * rows[:, None]).sum(axis=0)
        start_coordinate = box.x0
    else:
        # Идём по строкам: для каждого y — центр тяжести краски по x.
        columns = np.arange(box.x0, box.x0 + window.shape[1], dtype=float)
        count = ink.sum(axis=1).astype(float)
        weighted = (ink * columns[None, :]).sum(axis=1)
        start_coordinate = box.y0

    valid = count > 0
    if int(valid.sum()) < 2 or float(valid.mean()) < MIN_COVERAGE:
        return None
    along = (np.arange(count.size) + start_coordinate)[valid].astype(float)
    across = (weighted[valid] / count[valid]).astype(float)
    return Polyline(along, across, segment.horizontal, float(count[valid].mean()))


def trace(lines: Lines, dpi: int, min_length_mm: float = MIN_TRACE_MM) -> tuple[list[Polyline], list[Polyline]]:
    """Все линейки как ломаные: горизонтальные и вертикальные."""
    minimal = mm_to_px(min_length_mm, dpi)
    horizontal: list[Polyline] = []
    vertical: list[Polyline] = []
    for segment in lines.horizontal:
        if segment.length < minimal:
            continue
        line = trace_segment(lines.horizontal_mask, segment)
        if line is not None:
            horizontal.append(line)
    for segment in lines.vertical:
        if segment.length < minimal:
            continue
        line = trace_segment(lines.vertical_mask, segment)
        if line is not None:
            vertical.append(line)
    return horizontal, vertical


def crossings(horizontal: list[Polyline], vertical: list[Polyline]) -> list[tuple[int, int, float, float]]:
    """Пересечения ломаных: ``(индекс горизонтали, индекс вертикали, x, y)``.

    Пересечение ищется как точка, где вертикаль пересекает горизонталь по интерполяции обеих
    ломаных: у горизонтали известно y(x), у вертикали x(y), и решается их пересечение
    простым поиском ближайшего согласования. Точность до пикселя здесь и не нужна — эти
    точки идут опорами выпрямителя, а он всё равно сглаживает.
    """
    found: list[tuple[int, int, float, float]] = []
    for h_index, row in enumerate(horizontal):
        if row.along.size < 2:
            continue
        for v_index, column in enumerate(vertical):
            if column.along.size < 2:
                continue
            x_low, x_high = row.along[0], row.along[-1]
            y_low, y_high = column.along[0], column.along[-1]
            x_guess = float(np.median(column.across))
            if not (x_low <= x_guess <= x_high):
                continue
            y_at_x = float(np.interp(x_guess, row.along, row.across))
            if not (y_low <= y_at_x <= y_high):
                continue
            # Одно уточнение: взять x вертикали на найденной высоте.
            x_refined = float(np.interp(y_at_x, column.along, column.across))
            y_refined = float(np.interp(x_refined, row.along, row.across))
            found.append((h_index, v_index, x_refined, y_refined))
    return found
