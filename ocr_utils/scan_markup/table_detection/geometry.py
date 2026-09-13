"""Общие типы: рамка, таблица, ячейка, сетка. Плюс арифметика, которую иначе перепишут трижды.

СИСТЕМА КООРДИНАТ. Всё в пикселях того изображения, которое подавали алгоритму, и поле
``origin`` у :class:`TableBox` говорит, какого именно: ``"page"`` — полоса целиком,
``"crop"`` — вырезанная таблица. Пересчёт между ними — единственное место, где ошибка
не видна глазами до самого рендера, поэтому он собран в :func:`shift` и :func:`scale`.

СТРОКИ И КОЛОНКИ считаются от нуля сверху и слева. Объединённая ячейка записывается ОДИН
раз — в своей левой верхней клетке, с ``row_span``/``col_span`` больше единицы; остальные
клетки, которые она накрывает, в сетке не появляются вовсе. Иначе рендер стёр бы одну и ту
же шапку столько раз, сколько колонок она накрывает.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Box:
    """Прямоугольник в пикселях; ``x1``/``y1`` — за последним пикселем, как срез питона."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError(f"рамка вывернута: {self}")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def slice(self) -> tuple[slice, slice]:
        """Готовый срез для numpy: ``gray[box.slice]``."""
        return slice(self.y0, self.y1), slice(self.x0, self.x1)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)

    def padded(self, pad: int) -> "Box":
        return Box(self.x0 - pad, self.y0 - pad, self.x1 + pad, self.y1 + pad)

    def clipped(self, width: int, height: int) -> "Box":
        return Box(
            max(0, min(self.x0, width)),
            max(0, min(self.y0, height)),
            max(0, min(self.x1, width)),
            max(0, min(self.y1, height)),
        )

    def shifted(self, dx: int, dy: int) -> "Box":
        return Box(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)

    def scaled(self, factor: float) -> "Box":
        return Box(round(self.x0 * factor), round(self.y0 * factor), round(self.x1 * factor), round(self.y1 * factor))


def union(boxes: Sequence[Box]) -> Box | None:
    """Объемлющая рамка; None для пустой последовательности."""
    if not boxes:
        return None
    return Box(min(b.x0 for b in boxes), min(b.y0 for b in boxes), max(b.x1 for b in boxes), max(b.y1 for b in boxes))


def intersection(first: Box, second: Box) -> Box | None:
    x0, y0 = max(first.x0, second.x0), max(first.y0, second.y0)
    x1, y1 = min(first.x1, second.x1), min(first.y1, second.y1)
    return Box(x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def iou(first: Box, second: Box) -> float:
    """Пересечение к объединению — мера совпадения рамок в сравнении детекторов."""
    common = intersection(first, second)
    if common is None:
        return 0.0
    denominator = first.area + second.area - common.area
    return common.area / denominator if denominator > 0 else 0.0


KIND_TABLE = "таблица"
KIND_DIAGRAM = "схема"
KIND_DRAWING = "рисунок"
KINDS = (KIND_TABLE, KIND_DIAGRAM, KIND_DRAWING)


@dataclass(frozen=True)
class TableBox:
    """Найденная таблица.

    ``skew_deg`` — угол линеек: положительный означает, что таблицу надо повернуть ПО
    ЧАСОВОЙ на столько градусов, чтобы линейки стали горизонтальными. Валюта та же, что
    у ``ocr_utils.scan_markup.rotation``, и по той же причине: второе соглашение по
    соседству рано или поздно повернуло бы в другую сторону.

    ДВЕ РАМКИ, И ЭТО НЕ ИЗБЫТОЧНОСТЬ. ``box`` — то, по чему режут: она подвинута до чистого
    просвета, чтобы вырезка не разрубила буквы пополам. ``rule_box`` — огибающая самих
    линеек, и по ней считается ``detection.verify``. Разводить их приходится потому, что
    признаки проверки считаются по вырезке РОВНО по рамке и меряют расстояние до её края
    (``INNER_MARGIN_MM``): сдвинь границу на два миллиметра — и внешняя линейка таблицы
    станет «внутренней вертикалью», а все пороги, калиброванные на 61 находке, поедут.
    ``None`` означает «рамку не двигали», то есть ``rule_box`` совпадает с ``box``.
    """

    box: Box
    score: float = 1.0
    source: str = ""
    origin: str = "page"
    skew_deg: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    rule_box: "Box | None" = None
    # Вид объекта: «таблица», «схема» (блок-схема: коробки с текстом и связи между ними) или
    # «рисунок» (график, чертёж, line art). Появился в четвёртой версии, потому что схему и
    # таблицу лечат по-разному: у таблицы графу можно расширить под горизонтальный текст, у
    # схемы блок обязан остаться на месте. По умолчанию «таблица», чтобы старые вызовы, которые
    # про виды не знают, не изменили поведения.
    kind: str = KIND_TABLE

    @property
    def rules(self) -> Box:
        """Рамка, по которой считают признаки проверки."""
        return self.rule_box if self.rule_box is not None else self.box

    @property
    def is_table(self) -> bool:
        return self.kind == KIND_TABLE


@dataclass(frozen=True)
class Edge:
    """Линейка на границе одной полосы: где она проходит и какая она толщины.

    ЗАЧЕМ НЕ ПРОСТО ЧИСЛО. Разделитель колонки — это не одна прямая на всю таблицу.
    Линейка шапки и линейка тела печатались и сканировались по-разному, и замер по 80
    вырезкам пака даёт разброс центров внутри одной колонки: медиана 5 px при 300 dpi,
    p90 7, максимум 20 — то есть до ширины цифры. Поэтому у каждой полосы своя линейка.

    ``present=False`` значит «линейки здесь нет вовсе»: объединённая ячейка или внешняя
    граница без рамки. Тогда ``position`` — это глобальный разделитель, взятый как есть,
    а ``thickness`` равна нулю, и внутренность ячейки с этой стороны отступает на глазок.
    """

    position: int
    thickness: int = 0
    present: bool = False
    # Края краски линейки в этой полосе. Именно по ним, а не по половине толщины, режется
    # внутренность ячейки: линейка бывает наклонной (в паке до 1.7°), и на высоте строки
    # в 300 px её штрих уходит вбок на несколько пикселей — половина толщины этого не знает.
    low: int = 0
    high: int = 0

    def inner_offset(self) -> int:
        """На сколько отступить от ``position`` внутрь ячейки, чтобы не задеть линейку."""
        return (self.thickness + 1) // 2 + 1 if self.present else 0

    def inner_after(self) -> int:
        """Первая координата ЗА линейкой, если идти вправо или вниз."""
        return self.high + 1 if self.present else self.position

    def inner_before(self) -> int:
        """Последняя координата ПЕРЕД линейкой, если идти влево или вверх."""
        return self.low - 1 if self.present else self.position


@dataclass(frozen=True)
class Cell:
    """Клетка сетки. Объединённая записана один раз, в левой верхней клетке.

    ``box`` — геометрия ячейки: рамка по центрам линеек. ``inner`` — её внутренность без
    линеек, посчитанная по измеренной толщине каждой из четырёх; если она не задана
    (синтетика, разметка), внутренность берут отступом на глазок — см. ``ruling_grid.interior``.
    """

    row: int
    col: int
    box: Box
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False
    inner: "Box | None" = None

    @property
    def key(self) -> tuple[int, int]:
        return (self.row, self.col)

    def covers(self, row: int, col: int) -> bool:
        return self.row <= row < self.row + self.row_span and self.col <= col < self.col + self.col_span


@dataclass
class Grid:
    """Сетка таблицы: разделители и ячейки.

    ``xs``/``ys`` — координаты разделителей (на единицу больше, чем колонок и строк).
    ``header_rows`` — сколько верхних строк относится к шапке; определяется двойной линейкой,
    которой в этих журналах отделена шапка, и именно в шапке живёт боковой текст.
    """

    xs: list[int]
    ys: list[int]
    cells: list[Cell]
    header_rows: int = 0
    double_rule_ys: list[int] = field(default_factory=list)
    source: str = ""
    # Где можно разрезать картинку по вертикали, НЕ рассекая линейку: левый край её штриха.
    # Резать по ``xs`` (центру линейки) нельзя — половина штриха остаётся слева от разреза,
    # половина справа, и после вставки линейка двоится. Пусто — значит резать по ``xs``.
    column_cuts: list[int] = field(default_factory=list)

    @property
    def cuts(self) -> list[int]:
        return self.column_cuts or list(self.xs)

    @property
    def n_cols(self) -> int:
        return max(0, len(self.xs) - 1)

    @property
    def n_rows(self) -> int:
        return max(0, len(self.ys) - 1)

    def shifted(self, dx: int, dy: int) -> "Grid":
        """Сетка, сдвинутая вместе с картинкой.

        Нужна ровно затем, чтобы обрезка картинки и сетка не разъезжались. Один раз они
        уже разъехались: обрезанная по таблице картинка рисовалась с координатами
        необрезанной вырезки, и все рамки оверлея уехали на 15-24 px.
        """
        return Grid(
            xs=[x + dx for x in self.xs],
            ys=[y + dy for y in self.ys],
            cells=[
                Cell(
                    row=cell.row,
                    col=cell.col,
                    box=cell.box.shifted(dx, dy),
                    row_span=cell.row_span,
                    col_span=cell.col_span,
                    is_header=cell.is_header,
                    inner=cell.inner.shifted(dx, dy) if cell.inner else None,
                )
                for cell in self.cells
            ],
            header_rows=self.header_rows,
            double_rule_ys=[y + dy for y in self.double_rule_ys],
            source=self.source,
            column_cuts=[x + dx for x in self.column_cuts],
        )

    def cell_at(self, row: int, col: int) -> Cell | None:
        for cell in self.cells:
            if cell.covers(row, col):
                return cell
        return None

    def header_cells(self) -> list[Cell]:
        return [cell for cell in self.cells if cell.is_header]

    def to_json(self) -> dict:
        return {
            "xs": list(self.xs),
            "ys": list(self.ys),
            "header_rows": self.header_rows,
            "double_rule_ys": list(self.double_rule_ys),
            "source": self.source,
            "cells": [
                {
                    "row": c.row,
                    "col": c.col,
                    "box": list(c.box.as_tuple()),
                    "row_span": c.row_span,
                    "col_span": c.col_span,
                    "is_header": c.is_header,
                    "inner": list(c.inner.as_tuple()) if c.inner else None,
                }
                for c in self.cells
            ],
        }

    @staticmethod
    def from_json(payload: dict) -> "Grid":
        cells = [
            Cell(
                row=int(item["row"]),
                col=int(item["col"]),
                box=Box(*(int(v) for v in item["box"])),
                row_span=int(item.get("row_span", 1)),
                col_span=int(item.get("col_span", 1)),
                is_header=bool(item.get("is_header", False)),
                inner=Box(*(int(v) for v in item["inner"])) if item.get("inner") else None,
            )
            for item in payload.get("cells", [])
        ]
        return Grid(
            xs=[int(v) for v in payload.get("xs", [])],
            ys=[int(v) for v in payload.get("ys", [])],
            cells=cells,
            header_rows=int(payload.get("header_rows", 0)),
            double_rule_ys=[int(v) for v in payload.get("double_rule_ys", [])],
            source=str(payload.get("source", "")),
        )


def match_boxes(predicted: Sequence[Box], truth: Sequence[Box], threshold: float = 0.5) -> tuple[int, list[float]]:
    """Жадное сопоставление рамок: сколько эталонных нашлось и с каким IoU.

    Жадно, а не венгерским алгоритмом, намеренно: таблиц на полосе единицы, а жадный ответ
    читается глазами и не требует объяснений в отчёте.
    """
    used: set[int] = set()
    scores: list[float] = []
    for reference in truth:
        best_index, best_score = -1, 0.0
        for index, candidate in enumerate(predicted):
            if index in used:
                continue
            score = iou(candidate, reference)
            if score > best_score:
                best_index, best_score = index, score
        if best_index >= 0 and best_score >= threshold:
            used.add(best_index)
        scores.append(best_score)
    return len(used), scores


def boxes_from_rows(rows: Iterable[dict], prefix: str = "") -> list[Box]:
    """Рамки из строк CSV с колонками ``x0,y0,x1,y1`` (с необязательной приставкой)."""
    return [Box(*(int(float(row[f"{prefix}{name}"])) for name in ("x0", "y0", "x1", "y1"))) for row in rows]
