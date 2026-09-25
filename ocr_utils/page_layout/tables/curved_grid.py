"""Сетка ячеек по изогнутым линейкам: ячейки — многоугольники из кусков кривых, без выпрямления картинки.

Сетка ``grid.grid_from_lines`` — решётка по осям: разделитель строки — одна координата y. На наклонной
или изогнутой таблице (1968/03 с.4: линейки от −3,4° слева до −0,3° справа) одна линейка даёт
несколько разделителей, и ячейки режутся вдоль наклона. Здесь разделитель — КРИВАЯ (трасса линейки,
``traces.RuleTrace``), а ячейка — область между двумя соседними горизонтальными и двумя соседними
вертикальными кривыми. Логика та же, что у ``grid.py``, и константы оттуда же:

1. РАЗДЕЛИТЕЛИ. Трассы одной оси, идущие на одной высоте (медиана разности кривых на общем участке
   не больше ``SEPARATOR_TOL_MM``), — один разделитель. Две линейки на расстоянии до
   ``DOUBLE_RULE_MM`` друг от друга и ПАРАЛЛЕЛЬНЫЕ (разброс разности p95 − p5 не больше
   ``DOUBLE_STABLE_MM``) — двойная линейка, одна граница; она же отбивает шапку. Условие
   параллельности нужно волнистой линейке: её куски на разной высоте прежняя сетка принимала за
   двойную (1968/05 с.70, таблица 5).
2. ПРОДОЛЖЕНИЕ НА ВСЮ ТАБЛИЦУ. Неполная линейка (под частью граф) задаёт границу строки на всю
   ширину: за своими концами она идёт «параллельно» соседним длинным линейкам — между ними на той же
   относительной высоте, что у своего конца.
3. ВНЕШНИЕ ГРАНИЦЫ. Если внешней линейки нет (крайний разделитель дальше ``OUTER_COLUMN_MM`` от края
   вырезки), граница — копия крайней линейки, сдвинутая к краю вырезки; крайняя полоса без текста
   отбрасывается.
4. УГЛЫ — пересечения кривых ``y = fh(x)`` и ``x = fv(y)`` простой итерацией: при наклонах в
   несколько градусов отображение сжимающее, хватает пары шагов.
5. СТОРОНА ЯЧЕЙКИ ЕСТЬ, если вдоль куска кривой между углами краска линейки этой оси стоит не
   меньше чем на половине отсчётов (``SEPARATOR_FILL``). Нет стороны — ячейки объединяются;
   сквозные границы строк (``GLOBAL_ROW_SHARE``) не объединяются никогда.
6. МНОГОУГОЛЬНИКИ. Ячейка — обход кусков четырёх кривых между углами; внутренность — те же кривые,
   сдвинутые внутрь на измеренный край штриха и зазор ``RULE_CLEARANCE_MM``.

Всё считается на вырезке таблицы в рабочем разрешении (300 dpi, ``grid.WORK_DPI``) и переносится в
пиксели страницы ``mapped(scale, dx, dy)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

import cv2
import numpy as np

from ocr_utils.page_layout.geometry import Box, Cell, Grid
from ocr_utils.page_layout.tables.grid import (
    DOUBLE_RULE_MM,
    GLOBAL_ROW_MIN_COLUMNS,
    GLOBAL_ROW_SHARE,
    MIN_CELL_MM,
    NO_RULE_PAD_MM,
    OUTER_COLUMN_MM,
    RULE_CLEARANCE_MM,
    SEPARATOR_FILL,
    SEPARATOR_TOL_MM,
    has_text,
)
from ocr_utils.page_layout.tables.rules import CROSS_TOL_MM, components
from ocr_utils.page_layout.tables.ruling import MIN_RULE_MM, binarize, mm_to_px
from ocr_utils.page_layout.tables.traces import Axis, Frame, RuleTrace, RuleTraces, trace_rules, with_masks

# Самая короткая линейка, которую сетка берёт вообще: 4 мм (вертикали подшапки в строках высотой 5 мм).
SHORT_RULE_MM = 4.0

# ОПОРА (``table_rules``). Линейки таблицы — одна связная сеть. Опорные — не короче этой доли охвата
# линеек своей оси (края таблицы, сквозные линейки, в том числе у таблиц без внешней рамки); к ним по
# цепочке касаний добавляются остальные: линейка от 8 мм — если хоть где-то касается принятой
# перпендикулярной, короткая — если оба её конца вплотную к принятым или она встык продолжает
# принятую. «Обоими концами» для всех оказалось слишком строго: линейка под «Годы» 1967/10 с.33 одним
# концом не доходит до вертикали 6 мм. Касание не с сетью, а с любой трассой — слишком мягко: рамка
# штампа опиралась сама на себя, штрихи крупных букв заголовка — друг на друга (обложки 1971/06,
# 1970/11). Без правила разделителем становилась и черта дроби «2,9/3,5» (1973/08 с.62): сетка
# продолжала её на всю таблицу, и вместо 17 × 4 получалось 17 × 26.
SUPPORT_SPAN = 0.5

# Короткая линейка (от SHORT_RULE_MM до MIN_RULE_MM) — «мостик» — обязана стоять обоими концами
# вплотную к перпендикулярным: не дальше 0,5 мм плюс толщина. Черта дроби в узкой строке боковой
# таблицы 1973/08 с.62 длиной 6 мм не доходит до линеек строки на 13–26 px (1–2 мм) — при допуске
# 1,5 мм она проходила бы мостиком.
BRIDGE_TOL_MM = 0.5

# Нелинейка толще этого множителя от медианной толщины опорных линеек — штрих жирной буквы (10–13 px
# против 3–4 px у линеек) или рамка штампа (5 px против 3 px, 1971/06 обложка: штамп касается линейки).
GLYPH_THICKNESS_FACTOR = 1.5

# Двойная линейка — две ПАРАЛЛЕЛЬНЫЕ: разброс расстояния между ними (p95 − p5) не больше 0,4 мм.
# Куски одной волнистой линейки на разной высоте этому не удовлетворяют.
DOUBLE_STABLE_MM = 0.4

# Общий участок двух трасс, по которому их сравнивают: не меньше 2 мм.
MIN_OVERLAP_MM = 2.0

# Линейка, накрывающая эту долю таблицы, — опорная: по ней продолжаются неполные.
ANCHOR_SPAN = 0.8

# Шаг обхода кривых при построении многоугольника ячейки.
POLYGON_STEP_MM = 1.0

# Окно поиска краски линейки поперёк кривой при проверке стороны: полтолщины + 1 px + 0,3 мм.
PRESENCE_SLACK_MM = 0.3

# У концов куска стороны пересечение с перпендикулярной линейкой не считается: отступ не меньше 0,5 мм.
CORNER_TRIM_MM = 0.5

# Итерации поиска угла.
CORNER_ITERATIONS = 12


class Side(StrEnum):
    """Сторона ячейки."""

    TOP = "сверху"
    RIGHT = "справа"
    BOTTOM = "снизу"
    LEFT = "слева"


@dataclass(frozen=True)
class CurvedEdge:
    """Сторона ячейки на кривой: есть ли линейка, какая доля стороны в краске, толщина и края штриха."""

    present: bool
    fill: float
    thickness: float  # медианная толщина штриха на этой стороне, пиксели
    before: float  # на сколько штрих заходит в сторону МЕНЬШИХ координат поперёк (вверх / влево), p95
    after: float  # на сколько заходит в сторону больших (вниз / вправо), p95


NO_EDGE = CurvedEdge(False, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True, eq=False)
class Boundary:
    """Граница строк (у оси HORIZONTAL) или колонок (VERTICAL): кривая на всю таблицу.

    ``values[i]`` — поперечная координата границы в точке ``i`` вдоль (рабочие пиксели, от 0 до
    размера вырезки). У двойной линейки две кривые: ``before`` (верхняя / левая) и ``after``.
    """

    axis: Axis
    before: np.ndarray
    after: np.ndarray
    traces: tuple[RuleTrace, ...]  # трассы, из которых собрана граница (пусто у виртуальной)
    own_start: float  # где вдоль лежат её собственные линейки
    own_end: float
    is_double: bool = False
    is_virtual: bool = False

    @property
    def middle(self) -> float:
        """Медиана кривой на собственном участке — по ней границы упорядочиваются."""
        start, end = int(max(0, self.own_start)), int(min(len(self.before) - 1, self.own_end))
        return float(np.median((self.before[start : end + 1] + self.after[start : end + 1]) / 2.0))


@dataclass(frozen=True, eq=False)
class CellRegion:
    """Область ячейки на картинке: вырезка по рамке, маска криволинейного многоугольника, рамка вырезки."""

    image: np.ndarray  # вырезка; пиксели вне многоугольника залиты бумагой
    mask: np.ndarray  # bool той же формы: True — внутри многоугольника ячейки
    box: Box  # где вырезка стоит на исходной картинке


@dataclass(frozen=True, eq=False)
class CurvedCell:
    """Ячейка сетки по кривым: многоугольники ячейки и её внутренности, объединение, стороны."""

    row: int
    col: int
    row_span: int
    col_span: int
    is_header: bool
    polygon: np.ndarray  # (N, 2) float: обход x, y по кривым через углы, по часовой начиная с левого верхнего
    inner_polygon: np.ndarray  # то же без линеек: кривые сдвинуты внутрь на край штриха и зазор
    sides: dict[Side, tuple[CurvedEdge, ...]]  # куски каждой стороны (у объединённой ячейки их несколько)
    irregular: bool = False  # группа клеток не прямоугольная (объединение «уголком»); рамка — по охвату

    @property
    def key(self) -> tuple[int, int]:
        return (self.row, self.col)

    @property
    def box(self) -> Box:
        """Целая рамка, накрывающая многоугольник ячейки."""
        return _bounding(self.polygon)

    @property
    def inner_box(self) -> Box:
        """Целая рамка, накрывающая внутренность."""
        return _bounding(self.inner_polygon)

    def covers(self, row: int, col: int) -> bool:
        return self.row <= row < self.row + self.row_span and self.col <= col < self.col + self.col_span

    @property
    def region_area(self) -> float:
        """Площадь области ячейки без линеек (внутреннего многоугольника), квадратные пиксели."""
        x, y = self.inner_polygon[:, 0], self.inner_polygon[:, 1]
        return abs(0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def mask(self, shape: tuple[int, int], inner: bool = False) -> np.ndarray:
        """Маска ячейки (или её внутренности) размером ``shape``: uint8 0/255.

        Args:
            shape: ``(высота, ширина)`` картинки, в пикселях которой многоугольник.
            inner: Взять внутренность вместо всей ячейки.

        Returns:
            Маска.
        """
        canvas = np.zeros(shape, np.uint8)
        polygon = self.inner_polygon if inner else self.polygon
        cv2.fillPoly(canvas, [np.round(polygon).astype(np.int32)], 255)
        return canvas

    def extract(self, image: np.ndarray, inner: bool = True, fill: "int | None" = None) -> "CellRegion":
        """Область ячейки на картинке: вырезка по рамке многоугольника, всё вне многоугольника — цветом бумаги.

        Это и есть «область ячейки» по кривым границам: не рамка и не параллелограмм, а ровно то, что
        лежит между четырьмя кривыми (по умолчанию — внутренность без линеек, под распознавание).

        Args:
            image: Картинка, в пикселях которой ячейка (серая ``(H, W)`` или цветная ``(H, W, 3)``).
            inner: Взять внутренность без линеек (``inner_polygon``) вместо всей ячейки.
            fill: Чем залить пиксели вне многоугольника; ``None`` — бумагой (90-й перцентиль яркости вырезки).

        Returns:
            Вырезка, маска многоугольника в ней и рамка вырезки на картинке.
        """
        height, width = image.shape[:2]
        polygon = self.inner_polygon if inner else self.polygon
        box = _bounding(polygon).clipped(width, height)
        crop = image[box.slice].copy()
        local = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(local, [np.round(polygon - np.array([box.x0, box.y0])).astype(np.int32)], 255)
        inside = local > 0
        paper = fill if fill is not None else (np.percentile(crop, 90, axis=(0, 1)) if crop.size else 255)
        crop[~inside] = paper
        return CellRegion(crop, inside, box)

    def mapped(self, scale: float, dx: float, dy: float) -> "CurvedCell":
        """Ячейка в пикселях другой картинки: ``x · scale + dx``, ``y · scale + dy``."""
        shift = np.array([dx, dy])
        return replace(self, polygon=self.polygon * scale + shift, inner_polygon=self.inner_polygon * scale + shift)

    def to_json(self) -> dict:
        """Ячейка в JSON: номер, объединение, шапка, многоугольники (пиксели, 0,1 px) и стороны."""
        return {
            "row": self.row,
            "col": self.col,
            "row_span": self.row_span,
            "col_span": self.col_span,
            "is_header": self.is_header,
            "irregular": self.irregular,
            "box": list(self.box.as_tuple()),
            "polygon": np.round(self.polygon, 1).tolist(),
            "inner_polygon": np.round(self.inner_polygon, 1).tolist(),
            "sides": {side.value: [edge.present for edge in edges] for side, edges in self.sides.items()},
        }


@dataclass(frozen=True, eq=False)
class CurvedGrid:
    """Сетка таблицы по кривым: границы строк и колонок, ячейки-многоугольники, шапка.

    Границы хранятся в рабочих пикселях вырезки, ячейки — уже в пикселях картинки ``frame``.
    """

    rows: tuple[Boundary, ...]  # границы строк сверху вниз (горизонтальные кривые)
    cols: tuple[Boundary, ...]  # границы колонок слева направо
    cells: tuple[CurvedCell, ...]
    header_rows: int
    dpi: int
    frame: Frame = field(default_factory=Frame)
    source: str = "curved_grid"

    @property
    def n_rows(self) -> int:
        return max(0, len(self.rows) - 1)

    @property
    def n_cols(self) -> int:
        return max(0, len(self.cols) - 1)

    def cell_at(self, row: int, col: int) -> "CurvedCell | None":
        for cell in self.cells:
            if cell.covers(row, col):
                return cell
        return None

    def header_cells(self) -> list[CurvedCell]:
        return [cell for cell in self.cells if cell.is_header]

    def boundary_points(self, boundary: Boundary, step_px: float = 4.0) -> np.ndarray:
        """Точки ``(x, y)`` кривой границы в пикселях картинки ``frame`` (середина двойной линейки).

        Args:
            boundary: Граница из ``rows`` или ``cols``.
            step_px: Шаг по рабочим пикселям.

        Returns:
            Массив ``(N, 2)``.
        """
        along = np.arange(0, len(boundary.before), step_px)
        across = np.interp(along, np.arange(len(boundary.before)), (boundary.before + boundary.after) / 2.0)
        points = np.column_stack([along, across] if boundary.axis == Axis.HORIZONTAL else [across, along])
        return points * self.frame.scale + np.array([self.frame.dx, self.frame.dy])

    def mapped(self, scale: float, dx: float, dy: float) -> "CurvedGrid":
        """Сетка в пикселях другой картинки: ячейки переносятся, границы остаются рабочими с новым ``frame``."""
        return replace(
            self, cells=tuple(cell.mapped(scale, dx, dy) for cell in self.cells), frame=self.frame.then(scale, dx, dy)
        )

    def as_axis_grid(self) -> Grid:
        """Та же сетка в типах ``geometry.Grid``: ячейки — рамки многоугольников (для потребителей старой сетки).

        Returns:
            Сетка с ``xs``/``ys`` по медианам границ и ячейками-рамками в пикселях картинки ``frame``.
        """
        scale, dx, dy = self.frame.scale, self.frame.dx, self.frame.dy
        xs = [int(round(boundary.middle * scale + dx)) for boundary in self.cols]
        ys = [int(round(boundary.middle * scale + dy)) for boundary in self.rows]
        cells = [Cell(c.row, c.col, c.box, c.row_span, c.col_span, c.is_header, c.inner_box) for c in self.cells]
        return Grid(xs=xs, ys=ys, cells=cells, header_rows=self.header_rows, source=self.source)

    def to_json(self) -> dict:
        """Сетка в JSON: размеры, шапка, ячейки и кривые границ (пиксели картинки ``frame``)."""
        return {
            "source": self.source,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "header_rows": self.header_rows,
            "rows": [np.round(self.boundary_points(b), 1).tolist() for b in self.rows],
            "cols": [np.round(self.boundary_points(b), 1).tolist() for b in self.cols],
            "cells": [cell.to_json() for cell in self.cells],
        }


def _bounding(polygon: np.ndarray) -> Box:
    """Целая рамка, накрывающая многоугольник."""
    x0, y0 = np.floor(polygon.min(axis=0)).astype(int)
    x1, y1 = np.ceil(polygon.max(axis=0)).astype(int) + 1
    return Box(int(x0), int(y0), int(x1), int(y1))


# --- трассы → границы ----------------------------------------------------------------


def _near(point: np.ndarray, others: list[RuleTrace], tolerance: float) -> np.ndarray:
    """Для точек ``(N, 2)``: лежит ли каждая на какой-нибудь из трасс ``others`` с допуском ``tolerance``."""
    near = np.zeros(len(point), bool)
    for other in others:
        position, across = (point[:, 0], point[:, 1]) if other.horizontal else (point[:, 1], point[:, 0])
        near |= np.abs(other.across_at(position, extend_px=tolerance) - across) <= tolerance
    return near


def _touching(trace: RuleTrace, accepted_same: list[RuleTrace], accepted_other: list[RuleTrace], dpi: int) -> bool:
    """Опирается ли линейка на уже принятые линейки таблицы (правило опоры, см. ``SUPPORT_SPAN``).

    Линейка от ``MIN_RULE_MM`` — если хоть где-то касается принятой перпендикулярной (допуск
    ``CROSS_TOL_MM`` + толщина); короткая — если ОБА её конца стоят вплотную к принятым
    перпендикулярным (``BRIDGE_TOL_MM`` + толщина) или она встык продолжает принятую своей оси.
    """
    if trace.end - trace.start >= mm_to_px(MIN_RULE_MM, dpi):
        points = trace.point_at(trace.along_grid(mm_to_px(1.0, dpi)))
        return bool(_near(points, accepted_other, mm_to_px(CROSS_TOL_MM, dpi) + trace.thickness_px).any())
    ends = trace.point_at(np.array([trace.start_out, trace.end_out]))
    if _near(ends, accepted_other, mm_to_px(BRIDGE_TOL_MM, dpi) + trace.thickness_px).all():
        return True
    # Черта «|» строки «Итого» 1968/03 с.4 продолжает вертикаль над ней со сдвигом 1,4 мм.
    return any(_same_line(trace, other, dpi) for other in accepted_same)


def table_rules(
    horizontal: list[RuleTrace], vertical: list[RuleTrace], dpi: int
) -> tuple[list[RuleTrace], list[RuleTrace]]:
    """Линейки таблицы среди трасс вырезки: длинные и всё, что с ними связано касаниями.

    ЗАЧЕМ. Трассы находят всё линейное: штрихи крупных букв заголовка, рамку штампа, черты дробей,
    тире. Настоящие линейки таблицы — одна связная сеть. Опорные — линейки не короче
    ``SUPPORT_SPAN`` охвата своей оси (края и сквозные линейки); к ним по цепочке касаний добавляются
    остальные (``_touching``). Рамка штампа (1971/06, обложка) опирается сама на себя, но с сетью не
    связана; буквы «ИЮНЬ» и цифры номера висят свободно. Сверх того нелинейка толще
    ``GLYPH_THICKNESS_FACTOR`` медианной толщины опорных — штрих жирной буквы (10–13 px против
    3–4 px у линеек пака), даже если он касается сети.

    Args:
        horizontal: Горизонтальные трассы.
        vertical: Вертикальные трассы.
        dpi: Разрешение.

    Returns:
        ``(горизонтальные, вертикальные)`` линейки таблицы.
    """
    minimal = mm_to_px(MIN_RULE_MM, dpi)
    anchors: list[list[RuleTrace]] = []
    for traces in (horizontal, vertical):
        long_enough = [t for t in traces if t.end - t.start >= minimal]
        if not long_enough:
            anchors.append([])
            continue
        extent = max(t.end for t in long_enough) - min(t.start for t in long_enough)
        anchors.append([t for t in traces if t.end - t.start >= SUPPORT_SPAN * extent])
    reference = [t.thickness_px for t in anchors[0] + anchors[1]]
    limit = GLYPH_THICKNESS_FACTOR * float(np.median(reference)) if reference else np.inf
    accepted = [list(anchors[0]), list(anchors[1])]
    pending = [
        [t for t in traces if not any(t is a for a in anchors[index]) and t.thickness_px <= limit]
        for index, traces in enumerate((horizontal, vertical))
    ]
    # Наращивать сеть, пока к ней что-то добавляется: линейка подшапки опирается на вертикаль,
    # которая сама опирается на край таблицы.
    grown = True
    while grown:
        grown = False
        for index in (0, 1):
            other = 1 - index
            still: list[RuleTrace] = []
            for trace in pending[index]:
                if _touching(trace, accepted[index], accepted[other], dpi):
                    accepted[index].append(trace)
                    grown = True
                else:
                    still.append(trace)
            pending[index] = still
    return accepted[0], accepted[1]


def _overlap(first: RuleTrace, second: RuleTrace) -> np.ndarray:
    """Общий участок двух трасс одной оси вдоль, шаг 1 px (может быть пустым)."""
    start, end = max(first.start_out, second.start_out), min(first.end_out, second.end_out)
    return np.arange(np.ceil(start), np.floor(end) + 1) if end > start else np.zeros(0)


def _same_line(first: RuleTrace, second: RuleTrace, dpi: int) -> bool:
    """Одна ли это граница: трассы идут на одной высоте (или одна продолжает другую встык)."""
    tolerance = mm_to_px(SEPARATOR_TOL_MM, dpi)
    common = _overlap(first, second)
    if common.size >= mm_to_px(MIN_OVERLAP_MM, dpi):
        return abs(float(np.median(first.across_at(common) - second.across_at(common)))) <= tolerance
    # Без общего участка — встык: продолжения концов сходятся ближе ширины самой узкой клетки
    # (``MIN_CELL_MM``, 2 мм). Шире допуска разделителя, потому что кусок линейки в соседней строке
    # бывает набран со сдвигом: черта «|» в строке «Итого» 1968/03 с.4 стоит на 1,4 мм правее
    # вертикали над ней, и при допуске 1 мм между ними появлялась графа шириной 1,4 мм.
    left, right = (first, second) if first.start_out <= second.start_out else (second, first)
    gap = right.start_out - left.end_out
    if gap > mm_to_px(MIN_RULE_MM, dpi):
        return False
    joint = (left.end_out + right.start_out) / 2.0
    reach = abs(gap) + 1.0
    a, b = left.across_at(joint, reach)[0], right.across_at(joint, reach)[0]
    return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) < mm_to_px(MIN_CELL_MM, dpi))


def _is_double(first: RuleTrace, second: RuleTrace, dpi: int) -> bool:
    """Двойная ли линейка: близко, но дальше допуска разделителя, на общем участке не меньше половины короткой, параллельно."""
    common = _overlap(first, second)
    shorter = min(first.end_out - first.start_out, second.end_out - second.start_out)
    if common.size < max(mm_to_px(MIN_OVERLAP_MM, dpi), 0.5 * shorter):
        return False
    distance = np.abs(first.across_at(common) - second.across_at(common))
    middle = float(np.median(distance))
    stable = float(np.percentile(distance, 95) - np.percentile(distance, 5))
    return mm_to_px(SEPARATOR_TOL_MM, dpi) < middle <= mm_to_px(DOUBLE_RULE_MM, dpi) and stable <= mm_to_px(
        DOUBLE_STABLE_MM, dpi
    )


def _group(traces: list[RuleTrace], dpi: int) -> list[list[RuleTrace]]:
    """Трассы одной оси → группы «одна граница»."""
    edges = [
        (i, j) for i in range(len(traces)) for j in range(i + 1, len(traces)) if _same_line(traces[i], traces[j], dpi)
    ]
    return [[traces[index] for index in group] for group in components(len(traces), edges)]


def _own_curve(group: list[RuleTrace], size: int) -> tuple[np.ndarray, np.ndarray]:
    """Кривая группы на собственном участке: среднее трасс (с весом по длине) там, где они есть; NaN вне их.

    Вес по длине — чтобы короткая черта у длинной линейки (пометка, обрывок) не сдвигала её кривую.

    Args:
        group: Трассы одной границы.
        size: Длина вырезки вдоль.

    Returns:
        ``(значения длины size с NaN, маска «определено»)``.
    """
    along = np.arange(size, dtype=float)
    stack = np.vstack([trace.across_at(along) for trace in group])
    defined = np.isfinite(stack).any(axis=0)
    values = np.full(size, np.nan)
    lengths = np.array([max(trace.end - trace.start, 1.0) for trace in group])[:, None]
    present = np.isfinite(stack[:, defined])
    weighted = np.where(present, stack[:, defined], 0.0) * lengths
    values[defined] = weighted.sum(axis=0) / (present * lengths).sum(axis=0)
    # Разрывы между трассами одной границы (встык) — линейно.
    if defined.any() and not defined.all():
        inside = np.arange(np.flatnonzero(defined)[0], np.flatnonzero(defined)[-1] + 1)
        values[inside] = np.interp(inside, along[defined], values[defined])
        defined = np.zeros(size, bool)
        defined[inside] = True
    return values, defined


@dataclass(frozen=True)
class _Partial:
    """Кривая границы на собственном участке, до продолжения на всю таблицу."""

    group: tuple[RuleTrace, ...]
    values: np.ndarray  # NaN вне собственного участка
    defined: np.ndarray

    @property
    def start(self) -> int:
        return int(np.flatnonzero(self.defined)[0])

    @property
    def end(self) -> int:
        return int(np.flatnonzero(self.defined)[-1])

    @property
    def middle(self) -> float:
        return float(np.median(self.values[self.defined]))


def _tangent_extension(partial: _Partial, size: int) -> np.ndarray:
    """Продолжение за концами по касательной на концах (наклон по последним 2 мм)."""
    values = partial.values.copy()
    start, end = partial.start, partial.end
    span = max(2, min(24, (end - start) // 4))
    left = (values[start + span] - values[start]) / span
    right = (values[end] - values[end - span]) / span
    before = np.arange(0, start)
    after = np.arange(end + 1, size)
    values[before] = values[start] + left * (before - start)
    values[after] = values[end] + right * (after - end)
    return values


def _extend(partial: _Partial, anchors: list[np.ndarray], size: int) -> np.ndarray:
    """Кривая границы на всю таблицу: за своими концами — «параллельно» соседним опорным кривым.

    Между опорными выше и ниже — на той же относительной высоте, что у своего конца; если опорная
    только с одной стороны — на том же расстоянии от неё; если опорных нет — по касательной.

    Args:
        partial: Кривая на собственном участке.
        anchors: Опорные кривые той же оси на всю таблицу.
        size: Длина вырезки вдоль.

    Returns:
        Кривая длины ``size``.
    """
    values = partial.values.copy()
    for end, outside in ((partial.start, np.arange(0, partial.start)), (partial.end, np.arange(partial.end + 1, size))):
        if outside.size == 0:
            continue
        own = values[end]
        above = [a for a in anchors if a[end] < own - 0.5]
        below = [a for a in anchors if a[end] > own + 0.5]
        upper = max(above, key=lambda a: a[end]) if above else None
        lower = min(below, key=lambda a: a[end]) if below else None
        if upper is not None and lower is not None:
            share = (own - upper[end]) / max(lower[end] - upper[end], 1e-6)
            values[outside] = upper[outside] + share * (lower[outside] - upper[outside])
        elif upper is not None or lower is not None:
            anchor = upper if upper is not None else lower
            values[outside] = anchor[outside] + (own - anchor[end])
        else:
            values[outside] = _tangent_extension(partial, size)[outside]
    return values


def _partial_curves(groups: list[list[RuleTrace]], size: int) -> tuple[list[_Partial], list[np.ndarray]]:
    """Группы трасс → кривые на собственных участках и их продолжения на всю таблицу.

    Args:
        groups: Трассы, сгруппированные по границам.
        size: Длина вырезки вдоль.

    Returns:
        ``(кривые на собственных участках, кривые на всю таблицу)`` в одном порядке.
    """
    partials = []
    for group in groups:
        values, defined = _own_curve(group, size)
        if defined.any():
            partials.append(_Partial(tuple(group), values, defined))
    if not partials:
        return [], []
    # Опорные — накрывающие ANCHOR_SPAN охвата всех линеек оси; их продолжение — по касательной.
    extent_start = min(p.start for p in partials)
    extent_end = max(p.end for p in partials)
    span = max(1, extent_end - extent_start)
    anchor_parts = [p for p in partials if (p.end - p.start) >= ANCHOR_SPAN * span]
    anchors = [_tangent_extension(p, size) for p in anchor_parts]
    full = [
        _tangent_extension(p, size) if any(p is a for a in anchor_parts) else _extend(p, anchors, size)
        for p in partials
    ]
    return partials, full


def _merge_close(partials: list[_Partial], full: list[np.ndarray], dpi: int) -> list[list[RuleTrace]]:
    """Слить границы, чьи кривые на всю таблицу идут ближе ``MIN_CELL_MM``, кроме двойных линеек.

    Клетка уже 2 мм не бывает (``grid.MIN_CELL_MM``): две такие границы — куски одной линейки,
    прерванной строками или графами (1973/08: 17 × 26 вместо 17 × 4), или случайная черта у линейки —
    карандашная пометка «XX 15» у рамки обложки 1971/06 давала графу шириной 15 px. Двойная линейка
    (``_is_double``) — две границы и остаётся двумя.

    Args:
        partials: Кривые на собственных участках.
        full: Их продолжения на всю таблицу.
        dpi: Разрешение.

    Returns:
        Группы трасс после слияния.
    """
    limit = mm_to_px(MIN_CELL_MM, dpi)
    overlap_limit = mm_to_px(MIN_OVERLAP_MM, dpi)
    links: list[tuple[int, int]] = []
    for i in range(len(partials)):
        for j in range(i + 1, len(partials)):
            if int((partials[i].defined & partials[j].defined).sum()) >= overlap_limit and any(
                _is_double(a, b, dpi) for a in partials[i].group for b in partials[j].group
            ):
                continue
            gaps = np.concatenate(
                [
                    np.abs(partials[i].values[partials[i].defined] - full[j][partials[i].defined]),
                    np.abs(partials[j].values[partials[j].defined] - full[i][partials[j].defined]),
                ]
            )
            if float(np.median(gaps)) < limit:
                links.append((i, j))
    return [[trace for index in group for trace in partials[index].group] for group in components(len(partials), links)]


def _boundaries(traces: list[RuleTrace], axis: Axis, size: int, dpi: int) -> list[Boundary]:
    """Трассы одной оси → границы на всю таблицу, упорядоченные по положению; двойные слиты в одну.

    Args:
        traces: Трассы одной оси (рабочие пиксели).
        axis: Их ось.
        size: Длина вырезки вдоль этой оси.
        dpi: Разрешение.

    Returns:
        Границы по возрастанию поперечной координаты.
    """
    if not traces:
        return []
    groups = _group(traces, dpi)
    partials, full = _partial_curves(groups, size)
    # Куски одной линейки без общего участка и далеко друг от друга вдоль (колонка, прерванная
    # строками) — одна граница, если их кривые на всю таблицу идут ближе самой узкой клетки.
    merged = _merge_close(partials, full, dpi)
    if len(merged) < len(partials):
        partials, full = _partial_curves(merged, size)
    if not partials:
        return []
    order = sorted(range(len(partials)), key=lambda index: partials[index].middle)

    # Двойные: соседние по порядку пары, прошедшие проверку параллельности.
    boundaries: list[Boundary] = []
    index = 0
    while index < len(order):
        current = order[index]
        if index + 1 < len(order):
            following = order[index + 1]
            pairs = [(a, b) for a in partials[current].group for b in partials[following].group]
            if any(_is_double(a, b, dpi) for a, b in pairs):
                boundaries.append(
                    Boundary(
                        axis,
                        full[current],
                        full[following],
                        partials[current].group + partials[following].group,
                        float(min(partials[current].start, partials[following].start)),
                        float(max(partials[current].end, partials[following].end)),
                        is_double=True,
                    )
                )
                index += 2
                continue
        partial = partials[current]
        boundaries.append(
            Boundary(axis, full[current], full[current], partial.group, float(partial.start), float(partial.end))
        )
        index += 1
    return boundaries


def _virtual(boundary: Boundary, edge: float, before: bool) -> Boundary:
    """Внешняя граница без линейки: копия крайней кривой, сдвинутая к краю вырезки ``edge``.

    Args:
        boundary: Крайняя настоящая граница.
        edge: Координата края вырезки поперёк (0 или размер − 1).
        before: Сдвиг к меньшим координатам (верх / лево).

    Returns:
        Виртуальная граница.
    """
    curve = boundary.before if before else boundary.after
    shift = edge - (curve.min() if before else curve.max())
    values = curve + shift
    return Boundary(boundary.axis, values, values, (), boundary.own_start, boundary.own_end, is_virtual=True)


def _with_outer(boundaries: list[Boundary], size_across: int, dpi: int) -> list[Boundary]:
    """Добавить внешние границы там, где крайняя линейка дальше ``OUTER_COLUMN_MM`` от края вырезки."""
    if not boundaries:
        return boundaries
    minimal = mm_to_px(OUTER_COLUMN_MM, dpi)
    result = list(boundaries)
    if result[0].before.min() > minimal:
        result.insert(0, _virtual(result[0], 0.0, True))
    if (size_across - 1) - result[-1].after.max() > minimal:
        result.append(_virtual(result[-1], float(size_across - 1), False))
    return result


# --- углы, стороны --------------------------------------------------------------------


def _corner(horizontal: np.ndarray, vertical: np.ndarray) -> tuple[float, float]:
    """Пересечение кривых ``y = horizontal[x]`` и ``x = vertical[y]`` простой итерацией.

    Args:
        horizontal: Кривая горизонтальной границы, индекс — x.
        vertical: Кривая вертикальной границы, индекс — y.

    Returns:
        Точка ``(x, y)``.
    """
    xs = np.arange(len(horizontal), dtype=float)
    ys = np.arange(len(vertical), dtype=float)
    x = float(np.median(vertical))
    y = float(np.interp(x, xs, horizontal))
    for _ in range(CORNER_ITERATIONS):
        new_x = float(np.interp(y, ys, vertical))
        new_y = float(np.interp(new_x, xs, horizontal))
        converged = abs(new_x - x) < 0.01 and abs(new_y - y) < 0.01
        x, y = new_x, new_y
        if converged:
            break
    return x, y


def _edge(
    mask: np.ndarray, curve: np.ndarray, start: float, end: float, horizontal: bool, dpi: int, trim: float
) -> CurvedEdge:
    """Есть ли линейка на куске кривой ``[start, end]`` вдоль и какие у неё толщина и края штриха.

    Args:
        mask: Маска пикселей линеек этой оси (uint8), размером с вырезку.
        curve: Кривая границы (значения поперёк по индексу вдоль).
        start: Начало куска вдоль (угол ячейки).
        end: Конец куска вдоль.
        horizontal: Ось границы.
        dpi: Разрешение.
        trim: Сколько отрезать у концов куска — там пересечение с перпендикулярной линейкой.

    Returns:
        Сторона ячейки.
    """
    lo, hi = int(np.ceil(start + trim)), int(np.floor(end - trim))
    if hi < lo:
        lo, hi = int(np.ceil(start)), int(np.floor(end))
    if hi < lo:
        return NO_EDGE
    # Маска индексируется [поперёк, вдоль]: у горизонтали это [y, x], у вертикали — транспонированная.
    grid = mask if horizontal else mask.T
    window = mm_to_px(PRESENCE_SLACK_MM, dpi) + 3
    hits: list[tuple[float, float, int]] = []
    samples = 0
    for position in range(max(lo, 0), min(hi, grid.shape[1] - 1, len(curve) - 1) + 1):
        centre = curve[position]
        if not np.isfinite(centre):
            continue
        samples += 1
        low = int(max(0, np.floor(centre - window)))
        high = int(min(grid.shape[0] - 1, np.ceil(centre + window)))
        ink = np.flatnonzero(grid[low : high + 1, position])
        if ink.size:
            # Края штриха относительно кривой: сколько он заходит к меньшим и к большим координатам.
            hits.append((centre - (low + ink[0]), (low + ink[-1]) - centre, ink.size))
    if samples == 0:
        return NO_EDGE
    fill = len(hits) / samples
    if not hits:
        return CurvedEdge(False, 0.0, 0.0, 0.0, 0.0)
    before = float(np.percentile([h[0] for h in hits], 95))
    after = float(np.percentile([h[1] for h in hits], 95))
    thickness = float(np.median([h[2] for h in hits]))
    return CurvedEdge(fill >= SEPARATOR_FILL, fill, thickness, max(0.0, before), max(0.0, after))


# --- многоугольники ------------------------------------------------------------------


def _piece(curve: np.ndarray, start: float, end: float, step: float, horizontal: bool) -> np.ndarray:
    """Точки кривой от ``start`` до ``end`` вдоль (в любом направлении), концы включены."""
    count = max(2, int(np.ceil(abs(end - start) / step)) + 1)
    along = np.linspace(start, end, count)
    across = np.interp(along, np.arange(len(curve), dtype=float), curve)
    return np.column_stack([along, across] if horizontal else [across, along])


def _polygon(top: np.ndarray, right: np.ndarray, bottom: np.ndarray, left: np.ndarray, step: float) -> np.ndarray:
    """Обход ячейки по четырём кривым через их пересечения: верх → право → низ → лево."""
    tl, tr = _corner(top, left), _corner(top, right)
    br, bl = _corner(bottom, right), _corner(bottom, left)
    parts = [
        _piece(top, tl[0], tr[0], step, True),
        _piece(right, tr[1], br[1], step, False)[1:],
        _piece(bottom, br[0], bl[0], step, True)[1:],
        _piece(left, bl[1], tl[1], step, False)[1:-1],
    ]
    return np.vstack(parts)


def _valid(polygon: np.ndarray) -> bool:
    """Многоугольник не вывернут: площадь положительна (обход по часовой в координатах с y вниз)."""
    x, y = polygon[:, 0], polygon[:, 1]
    area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area > 1.0


def _shrunk(polygon: np.ndarray, pad: int) -> np.ndarray:
    """Запасная внутренность: рамка многоугольника, ужатая на ``pad`` (не больше трети стороны)."""
    box = _bounding(polygon)
    horizontal = min(pad, max(0, (box.width - 1) // 3))
    vertical = min(pad, max(0, (box.height - 1) // 3))
    x0, y0, x1, y1 = box.x0 + horizontal, box.y0 + vertical, box.x1 - 1 - horizontal, box.y1 - 1 - vertical
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


# --- сборка -------------------------------------------------------------------------


@dataclass(frozen=True)
class _Lattice:
    """Решётка границ со сторонами клеток: ``h_edges[граница строки][колонка]``, ``v_edges[граница колонки][строка]``."""

    rows: list[Boundary]
    cols: list[Boundary]
    h_edges: list[list[CurvedEdge]]
    v_edges: list[list[CurvedEdge]]
    corners: np.ndarray  # (строк границ, колонок границ, 2): точка x, y


def _lattice(rows: list[Boundary], cols: list[Boundary], traces: RuleTraces, dpi: int) -> _Lattice:
    """Углы решётки и стороны всех клеток."""
    corners = np.zeros((len(rows), len(cols), 2))
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            corners[i, j] = _corner((row.before + row.after) / 2.0, (col.before + col.after) / 2.0)
    trim_min = mm_to_px(CORNER_TRIM_MM, dpi)
    h_edges: list[list[CurvedEdge]] = []
    for i, row in enumerate(rows):
        pieces: list[CurvedEdge] = []
        for j in range(len(cols) - 1):
            if row.is_virtual:
                pieces.append(NO_EDGE)
                continue
            trim = max(trim_min, max(_thickness(cols[j]), _thickness(cols[j + 1])) / 2.0 + 1)
            start, end = corners[i, j, 0], corners[i, j + 1, 0]
            candidates = [_edge(traces.horizontal_mask, curve, start, end, True, dpi, trim) for curve in _curves(row)]
            pieces.append(max(candidates, key=lambda edge: edge.fill))
        h_edges.append(pieces)
    v_edges: list[list[CurvedEdge]] = []
    for j, col in enumerate(cols):
        pieces = []
        for i in range(len(rows) - 1):
            if col.is_virtual:
                pieces.append(NO_EDGE)
                continue
            trim = max(trim_min, max(_thickness(rows[i]), _thickness(rows[i + 1])) / 2.0 + 1)
            start, end = corners[i, j, 1], corners[i + 1, j, 1]
            candidates = [_edge(traces.vertical_mask, curve, start, end, False, dpi, trim) for curve in _curves(col)]
            pieces.append(max(candidates, key=lambda edge: edge.fill))
        v_edges.append(pieces)
    return _Lattice(rows, cols, h_edges, v_edges, corners)


def _curves(boundary: Boundary) -> list[np.ndarray]:
    return [boundary.before] if not boundary.is_double else [boundary.before, boundary.after]


def _thickness(boundary: Boundary) -> float:
    return max((trace.thickness_px for trace in boundary.traces), default=0.0)


def _band_has_text(ink: np.ndarray, first: Boundary, second: Boundary, dpi: int) -> bool:
    """Есть ли текст в полосе между двумя соседними границами одной оси (по маске полосы)."""
    size = len(first.before)
    along = np.arange(size, dtype=float)
    upper, lower = first.after, second.before
    if first.axis == Axis.HORIZONTAL:
        outline = np.vstack([np.column_stack([along, upper]), np.column_stack([along[::-1], lower[::-1]])])
    else:
        outline = np.vstack([np.column_stack([upper, along]), np.column_stack([lower[::-1], along[::-1]])])
    band = np.zeros(ink.shape, np.uint8)
    cv2.fillPoly(band, [np.round(outline).astype(np.int32)], 255)
    width = float(np.median(lower - upper))
    return width >= mm_to_px(MIN_CELL_MM, dpi) and has_text(cv2.bitwise_and(ink, band), dpi)


def _trim_virtual(boundaries: list[Boundary], ink: np.ndarray, dpi: int) -> list[Boundary]:
    """Убрать виртуальные внешние границы, если полоса между ними и таблицей пуста."""
    result = list(boundaries)
    if len(result) > 2 and result[0].is_virtual and not _band_has_text(ink, result[0], result[1], dpi):
        result = result[1:]
    if len(result) > 2 and result[-1].is_virtual and not _band_has_text(ink, result[-2], result[-1], dpi):
        result = result[:-1]
    return result


def _trim_ruleless_rows(lattice: _Lattice) -> tuple[list[Boundary], bool]:
    """Выбросить крайние строки без единой внутренней вертикальной линейки (как ``grid._trim_ruleless_rows``).

    Returns:
        ``(границы строк, изменилось ли что-то)``.
    """
    rows = lattice.rows
    count = len(rows) - 1
    inner_cols = range(1, len(lattice.cols) - 1)
    ruled = [any(lattice.v_edges[j][row].present for j in inner_cols) for row in range(count)]
    first, last = 0, count - 1
    while first < last and not ruled[first]:
        first += 1
    while last > first and not ruled[last]:
        last -= 1
    trimmed = rows[first : last + 2]
    return trimmed, len(trimmed) != len(rows)


def curved_grid_from_traces(traces: RuleTraces, shape: tuple[int, int], dpi: int, ink: np.ndarray) -> CurvedGrid:
    """Сетка ячеек по трассам линеек вырезки таблицы.

    Args:
        traces: Линейки таблицы (уже отобранные ``table_rules``, с масками ``traces.with_masks``;
            рабочие пиксели).
        shape: ``(высота, ширина)`` вырезки.
        dpi: Её разрешение.
        ink: Краска текста без линеек (uint8 0/255) — для отсечения пустых крайних полос.

    Returns:
        Сетка по кривым в пикселях вырезки; пустая, если строк или колонок не набралось.
    """
    height, width = shape
    rows = _with_outer(_boundaries(traces.horizontal, Axis.HORIZONTAL, width, dpi), height, dpi)
    cols = _with_outer(_boundaries(traces.vertical, Axis.VERTICAL, height, dpi), width, dpi)
    rows = _trim_virtual(rows, ink, dpi)
    cols = _trim_virtual(cols, ink, dpi)
    if len(rows) < 2 or len(cols) < 2:
        return CurvedGrid(tuple(rows), tuple(cols), (), 0, dpi)
    lattice = _lattice(rows, cols, traces, dpi)
    rows, changed = _trim_ruleless_rows(lattice)
    if changed:
        if len(rows) < 2:
            return CurvedGrid(tuple(rows), tuple(cols), (), 0, dpi)
        lattice = _lattice(rows, cols, traces, dpi)
    return _assemble(lattice, dpi)


def _assemble(lattice: _Lattice, dpi: int) -> CurvedGrid:
    """Объединения по отсутствующим сторонам, шапка, многоугольники ячеек."""
    rows, cols = lattice.rows, lattice.cols
    n_rows, n_cols = len(rows) - 1, len(cols) - 1
    # Сквозные границы строк: линейка под достаточной долей колонок — через неё не объединяем.
    through = {
        i
        for i in range(1, n_rows)
        if sum(edge.present for edge in lattice.h_edges[i]) >= max(GLOBAL_ROW_MIN_COLUMNS, GLOBAL_ROW_SHARE * n_cols)
    }
    links: list[tuple[int, int]] = []
    for r in range(n_rows):
        for c in range(n_cols):
            if c + 1 < n_cols and not lattice.v_edges[c + 1][r].present:
                links.append((r * n_cols + c, r * n_cols + c + 1))
            if r + 1 < n_rows and r + 1 not in through and not lattice.h_edges[r + 1][c].present:
                links.append((r * n_cols + c, (r + 1) * n_cols + c))
    groups = components(n_rows * n_cols, links)

    header_rows = next((i for i, row in enumerate(rows) if row.is_double and i > 0), 0)
    if header_rows == 0:
        header_rows = 1 if n_rows > 1 else 0
    step = mm_to_px(POLYGON_STEP_MM, dpi)
    clearance = mm_to_px(RULE_CLEARANCE_MM, dpi)
    pad = mm_to_px(NO_RULE_PAD_MM, dpi)
    cells = [
        _cell(sorted(divmod(index, n_cols) for index in group), lattice, header_rows, step, clearance, pad)
        for group in groups
    ]
    cells.sort(key=lambda cell: (cell.row, cell.col))
    return CurvedGrid(tuple(rows), tuple(cols), tuple(cells), header_rows, dpi)


def _offset(edges: list[CurvedEdge], inward_after: bool, clearance: int, pad: int) -> float:
    """На сколько сдвинуть сторону внутрь ячейки: самый строгий кусок (край штриха + зазор, без линейки — pad)."""
    values = [(edge.after if inward_after else edge.before) + clearance if edge.present else pad for edge in edges]
    return float(max(values)) if values else float(pad)


def _cell(
    members: list[tuple[int, int]], lattice: _Lattice, header_rows: int, step: int, clearance: int, pad: int
) -> CurvedCell:
    """Ячейка из клеток решётки: многоугольник по кривым, внутренность по краям штрихов."""
    top = min(r for r, _ in members)
    bottom = max(r for r, _ in members)
    left = min(c for _, c in members)
    right = max(c for _, c in members)
    irregular = len(members) != (bottom - top + 1) * (right - left + 1)
    upper, lower = lattice.rows[top], lattice.rows[bottom + 1]
    first, last = lattice.cols[left], lattice.cols[right + 1]
    # У двойной линейки ячейка ниже берёт нижнюю кривую, выше — верхнюю.
    top_curve, bottom_curve = upper.after, lower.before
    left_curve, right_curve = first.after, last.before
    polygon = _polygon(top_curve, right_curve, bottom_curve, left_curve, step)

    sides = {
        Side.TOP: tuple(lattice.h_edges[top][c] for c in range(left, right + 1)),
        Side.BOTTOM: tuple(lattice.h_edges[bottom + 1][c] for c in range(left, right + 1)),
        Side.LEFT: tuple(lattice.v_edges[left][r] for r in range(top, bottom + 1)),
        Side.RIGHT: tuple(lattice.v_edges[right + 1][r] for r in range(top, bottom + 1)),
    }
    inner = _polygon(
        top_curve + _offset(list(sides[Side.TOP]), True, clearance, pad),
        right_curve - _offset(list(sides[Side.RIGHT]), False, clearance, pad),
        bottom_curve - _offset(list(sides[Side.BOTTOM]), False, clearance, pad),
        left_curve + _offset(list(sides[Side.LEFT]), True, clearance, pad),
        step,
    )
    if not _valid(inner) or not _valid(polygon):
        inner = _shrunk(polygon, pad)
    return CurvedCell(
        top, left, bottom - top + 1, right - left + 1, top < header_rows, polygon, inner, sides, irregular
    )


def text_ink(gray: np.ndarray, traces: RuleTraces) -> np.ndarray:
    """Краска текста без линеек: бинаризация минус маска трасс, раздутая на пару пикселей."""
    fat = cv2.dilate(traces.mask, np.ones((3, 3), np.uint8))
    return cv2.bitwise_and(binarize(gray), cv2.bitwise_not(fat))


def curved_table(gray: np.ndarray, dpi: int) -> tuple[RuleTraces, CurvedGrid]:
    """Линейки таблицы кривыми и сетка по ним для вырезки таблицы.

    Args:
        gray: Серая вырезка таблицы с полем в несколько миллиметров (обычно 300 dpi).
        dpi: Её разрешение.

    Returns:
        ``(линейки таблицы, сетка)`` в пикселях вырезки. В линейки входят только отобранные
        ``table_rules`` — штрихи букв, штампы и черты дробей отброшены; маски — краска по этим трассам.
    """
    found = trace_rules(gray, dpi, SHORT_RULE_MM)
    horizontal, vertical = table_rules(found.horizontal, found.vertical, dpi)
    traces = with_masks(horizontal, vertical, binarize(gray), dpi)
    return traces, curved_grid_from_traces(traces, gray.shape[:2], dpi, text_ink(gray, traces))


__all__ = [
    "table_rules",
    "CellRegion",
    "Boundary",
    "CurvedCell",
    "CurvedEdge",
    "CurvedGrid",
    "Side",
    "curved_grid_from_traces",
    "curved_table",
    "text_ink",
]
