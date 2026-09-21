"""Проверка находки детектора: таблица это или что-то другое.

ЗАЧЕМ. Детектор ищет скопления линеек, а линейки бывают не только у таблиц. Разведка по
шести выпускам 1966 года: из 27 находок на страницах, где в DOCX таблицы нет, 15 оказались
настоящими таблицами, а 12 — нет, и все двенадцать распались на четыре ясных вида:

* РАМКА ОБЪЯВЛЕНИЯ, некролога, книжной обложки (8 из 12) — прямоугольник вокруг сплошного
  набора. Внутренних разделителей нет ни одного, строки идут во всю ширину рамки;
* ЧЕРТЁЖ, генплан, блок-схема (3) — короткие линии, решётки не образуют;
* ГРАФИК с координатной сеткой (1) — решётка есть, но в клетках нет букв.

Отсюда пять признаков ниже. Ни один не работает в одиночку, и это не осторожность, а
требование вёрстки: у таблицы может не быть внешней рамки, а может не быть части
внутренних линеек. «Внутренних разделителей нет» — не приговор: такую находку спасает
колонность текста.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.page_layout.tables.ruling import Lines, binarize, mm_to_px

# Линейка считается ВНУТРЕННЕЙ, если её середина отстоит от края находки не меньше чем на
# столько: 4 мм. Ближе — это сама рамка, и у рамки объявления других линеек нет.
INNER_MARGIN_MM = 4.0

# Просвет между графами: столбец без единого пикселя краски глифов шириной от 1.5 мм.
# Уже — это межсловный пробел, который на сумме по всем строкам всё равно заполнится.
GUTTER_MIN_MM = 1.5

# Зазор смыкания строки при измерении её длины: 2 мм. Больше — и строка перепрыгивает
# просвет между графами, меньше — и рвётся на слова.
LINE_GAP_MM = 2.0

# Окно вокруг предсказанного пересечения, в котором ищется краска: 1 мм.
CROSS_TOL_MM = 1.0


@dataclass(frozen=True)
class Features:
    """Признаки одной находки. Все безразмерные или в штуках."""

    inner_vertical: int
    inner_horizontal: int
    gutters: int
    gutter_share: float
    line_span: float
    lattice: float
    glyph_share: float
    glyph_of_ink: float
    edge_fill: float
    cells: int
    filled_cells: float

    def as_row(self) -> dict[str, float | int]:
        return {
            "inner_v": self.inner_vertical,
            "inner_h": self.inner_horizontal,
            "gutters": self.gutters,
            "gutter_share": round(self.gutter_share, 3),
            "line_span": round(self.line_span, 3),
            "lattice": round(self.lattice, 3),
            "glyph_share": round(self.glyph_share, 4),
            "glyph_of_ink": round(self.glyph_of_ink, 3),
            "edge_fill": round(self.edge_fill, 3),
            "cells": self.cells,
            "filled_cells": round(self.filled_cells, 3),
        }


HEADER = (
    "inner_v",
    "inner_h",
    "gutters",
    "gutter_share",
    "line_span",
    "lattice",
    "glyph_share",
    "glyph_of_ink",
    "edge_fill",
    "cells",
    "filled_cells",
)

# Ячейка считается заполненной, если краски глифов в ней не меньше этой доли площади.
# 0.5% при кегле петита — это примерно два знака.
CELL_INK_SHARE = 0.005

# --- Правило ---------------------------------------------------------------
# Все пороги стоят в ПУСТЫХ промежутках замера по 60 находкам, размеченным глазами
# (37 таблиц, 23 не таблицы; 27 из разведки по 1966 году и 33 из аудита по 396 полосам
# всех одиннадцати лет пака).
#
# ВНУТРЕННИЕ ВЕРТИКАЛИ. У всех 15 рамок объявлений, некрологов и книжных обложек их ровно
# ноль: рамка и есть рамка, прямоугольник вокруг сплошного набора.
MIN_INNER_VERTICAL = 1

# ОДНА ВНУТРЕННЯЯ ВЕРТИКАЛЬ — вырожденный случай, и решётка тут ничего не значит. Так
# выглядят колонтитул с содержанием («редколлегия» плюс перечень статей) и обложка журнала.
# Их отличают две вещи: у колонтитула ячеек одна-две (у таблиц с одной вертикалью — 4-6),
# у обложки краска наполовину из штрихов заголовка (0.42-0.44 против 0.52-0.92 у таблиц).
SINGLE_COLUMN_MIN_CELLS = 4
SINGLE_COLUMN_GLYPH_OF_INK = 0.50

# ДОЛЯ ГЛИФОВ В КРАСКЕ. В таблице краска — это буквы, в чертеже и блок-схеме — штрихи.
# Порогов два, потому что признак работает в паре с решёткой: при правдоподобной решётке
# хватает 0.41 (у графика работ с крестиками ровно 0.42, у ближайшего чертежа 0.40), при
# неполной нужно 0.50.
MIN_GLYPH_OF_INK = 0.41
CLEAR_GLYPH_OF_INK = 0.50

# ЗАПОЛНЕННОСТЬ РЁБЕР. 0.70 — решётка заведомо есть, и тогда текст не нужен вовсе: так
# проходит схема-матрица с залитыми квадратами (доля глифов 0.13). 0.45 — решётка
# правдоподобна: у настоящих таблиц с неполной решёткой 0.48-0.49, у блок-схем 0.33-0.41.
STRONG_EDGE_FILL = 0.70
PLAUSIBLE_EDGE_FILL = 0.45

# МНОГО КОЛОНОК. Бланк с пустыми графами даёт неполную решётку (0.39), и от блок-схемы его
# отличает число внутренних вертикалей: у бланка их 14 и 26, у блок-схемы 3.
MANY_COLUMNS = 8


def is_table(found: Features) -> tuple[bool, str]:
    """Таблица ли это. Второе значение — причина отказа словами, для отчёта.

    Правило проверено на всех 60 размеченных находках: ошибок нет.

    ЧЕГО ОНО НЕ УМЕЕТ. Оно не отличает таблицу от бланка и не пытается: бланк с пустыми
    графами (лимитная карточка, «Отметки ОТК») — такая же таблица, и FineReader ломает её
    так же. А вот технический чертёж с выносными размерными линиями оно ловит только пока
    в нём мало текста; чертёж, подписанный густо, пройдёт.
    """
    if found.inner_vertical < MIN_INNER_VERTICAL:
        return False, "нет внутренних вертикальных линеек — это рамка вокруг текста"
    if found.inner_vertical == 1:
        if found.cells < SINGLE_COLUMN_MIN_CELLS:
            return False, f"одна вертикаль и всего {found.cells} ячейки — это колонтитул, а не таблица"
        if found.glyph_of_ink < SINGLE_COLUMN_GLYPH_OF_INK:
            return False, f"одна вертикаль, и краска на {1 - found.glyph_of_ink:.0%} состоит из штрихов"
        return True, ""
    if found.edge_fill >= STRONG_EDGE_FILL:
        return True, ""
    plausible = found.edge_fill >= PLAUSIBLE_EDGE_FILL
    if found.glyph_of_ink >= CLEAR_GLYPH_OF_INK and (plausible or found.inner_vertical >= MANY_COLUMNS):
        return True, ""
    if found.glyph_of_ink >= MIN_GLYPH_OF_INK and plausible:
        return True, ""
    return False, (
        f"не таблица: рёбер на месте {found.edge_fill:.0%}, "
        f"букв в краске {found.glyph_of_ink:.0%}, колонок {found.inner_vertical}"
    )


# --- Правило третьей версии --------------------------------------------------

# ПОЛ ПО БУКВАМ И ПУСТЫМ ЯЧЕЙКАМ. Сверка по паку вскрыла то, чего не было видно на выборке:
# восемь полос, размеченных человеком как «график, а не таблица», правило выше ПРИНИМАЕТ —
# через ярлык ``STRONG_EDGE_FILL``. У координатной сетки графика рёбра действительно все на
# месте, поэтому ``edge_fill`` её пропускает, а букв в ней нет вовсе: ``glyph_of_ink`` от
# 0.012 до 0.203.
#
# Отличает их ``filled_cells`` — доля ячеек, в которых есть краска глифов. У графиков она
# 0.066-0.360, у чертежей 0.222-0.308, у настоящих пустых бланков 0.547 и 0.725, а у всех
# 879 принятых находок пака медиана 1.000.
#
# Порог по буквам стоит на 0.12, а не выше, из-за одного случая: 1966/03 с.33 — таблица-матрица
# сроков, у которой правая половина занята чёрными плашками, и букв в краске 0.131. При 0.15
# правило теряло её, а она размечена человеком как таблица. При 0.12 уходят семь графиков из
# восьми, оба line art и все три чертежа, и ни одной размеченной таблицы.
FLOOR_GLYPH_OF_INK = 0.12
FLOOR_FILLED_CELLS = 0.40

# ПУСТОЙ БЛАНК. Разреженная матрица (1969/06 с.67) и пустой бланк (1972/06 с.73) — настоящие
# таблицы, но текста в них так мало, что ``glyph_of_ink`` проваливается ниже ``MIN_GLYPH_OF_INK``.
# Спасает их размер решётки: сорок ячеек при полной решётке и хотя бы наполовину заполненных
# ячейках — это разлинованная форма, а не рисунок. Порог 40, а не 20: при 20 в находки
# добавляются ещё две неразмеченные полосы, а обе цели забираются и так.
BLANK_MIN_CELLS = 40
BLANK_MIN_LATTICE = 0.9
BLANK_MIN_FILLED = 0.5


# БАННЕР РУБРИКИ. Послабление по размеру («длинная сторона ≥ 40 мм, короткая ≥ 10») открыло
# дорогу новому виду шума, которого на выборке не было: заголовок рубрики в рамке —
# «ПРОБЛЕМЫ, СУЖДЕНИЯ, ПОИСК», «ПИСЬМА ЧИТАТЕЛЕЙ». Замер на 29 новых срабатываниях по паку:
# у баннера внутренних ГОРИЗОНТАЛЬНЫХ линеек ноль, ячеек 2-3, высота 15-23 мм. То есть это
# одна строка из двух-трёх клеток, а таблица — это всегда хотя бы две строки.
#
# Проверено на 21 находке восемнадцати полос, где таблица есть и FineReader её нашёл: у всех
# внутренних горизонталей хотя бы одна, под нож не попадает ни одна.
BANNER_MAX_CELLS = 3


def is_table_v3(found: Features) -> tuple[bool, str]:
    """Правило третьей версии: то же, что :func:`is_table`, плюс пол и ветка для бланка.

    Замер на разметке: 0 ошибок на 61 находке ``labels/detector.csv`` — столько же, сколько
    у второй версии. По паку: минус 12 находок (семь графиков, два line art, три чертежа),
    плюс две (разреженная матрица и пустой бланк), минус заголовки рубрик в рамке.

    ЧЕГО ОНО ПО-ПРЕЖНЕМУ НЕ УМЕЕТ. 1967/03 с.9 остаётся отвергнутой: ей не хватает 0.006 до
    ``PLAUSIBLE_EDGE_FILL`` и одной вертикали до ``MANY_COLUMNS``. Двигать порог на 0.006 под
    один случай — это подгонка, а не калибровка, и она сломается на первой же новой полосе.
    """
    if found.inner_horizontal == 0 and found.cells <= BANNER_MAX_CELLS:
        return False, (
            f"внутренних горизонталей нет, ячеек {found.cells} — это одна строка, "
            "заголовок рубрики в рамке, а не таблица"
        )
    if found.glyph_of_ink < FLOOR_GLYPH_OF_INK and found.filled_cells < FLOOR_FILLED_CELLS:
        return False, (
            f"букв в краске {found.glyph_of_ink:.0%}, заполнено ячеек {found.filled_cells:.0%} — "
            "это сетка графика или чертёж"
        )
    if (
        found.cells >= BLANK_MIN_CELLS
        and found.lattice >= BLANK_MIN_LATTICE
        and found.edge_fill >= PLAUSIBLE_EDGE_FILL
        and found.filled_cells >= BLANK_MIN_FILLED
        and found.inner_vertical >= MIN_INNER_VERTICAL
    ):
        return True, ""
    return is_table(found)


def _inner_counts(lines: Lines, shape: tuple[int, int], dpi: int) -> tuple[int, int]:
    """Сколько линеек проходит внутри находки, а не по её краю."""
    height, width = shape
    margin = mm_to_px(INNER_MARGIN_MM, dpi)
    vertical = sum(1 for s in lines.vertical if margin <= (s.box.x0 + s.box.x1) // 2 <= width - margin)
    horizontal = sum(1 for s in lines.horizontal if margin <= (s.box.y0 + s.box.y1) // 2 <= height - margin)
    return vertical, horizontal


def _gutters(mask: np.ndarray, dpi: int) -> tuple[int, float]:
    """Число сквозных просветов между графами и их суммарная доля ширины.

    Просвет ищется по СУММЕ краски за все строки: у сплошного набора межсловные пробелы
    разных строк не совпадают по x и сумму не обнуляют, а просвет между графами обнуляет.
    """
    if mask.size == 0:
        return 0, 0.0
    margin = mm_to_px(INNER_MARGIN_MM, dpi)
    profile = (mask > 0).sum(axis=0)
    inner = profile[margin : max(margin + 1, profile.size - margin)]
    if inner.size == 0:
        return 0, 0.0
    minimal = mm_to_px(GUTTER_MIN_MM, dpi)
    count = 0
    total = 0
    run = 0
    for value in inner:
        if value == 0:
            run += 1
            continue
        if run >= minimal:
            count += 1
            total += run
        run = 0
    if run >= minimal:
        count += 1
        total += run
    return count, total / inner.size


def _line_span(mask: np.ndarray, dpi: int) -> float:
    """Медианная длина строки в долях ширины находки.

    У сплошного набора строка идёт почти во всю ширину рамки, у таблицы обрывается на
    границе графы.
    """
    if mask.size == 0:
        return 0.0
    gap = mm_to_px(LINE_GAP_MM, dpi)
    smeared = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, gap), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(smeared, 8)
    if count <= 1:
        return 0.0
    widths = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    keep = heights >= mm_to_px(0.8, dpi)
    if not keep.any():
        return 0.0
    return float(np.median(widths[keep])) / mask.shape[1]


def _lattice(lines: Lines, dpi: int) -> float:
    """Доля предсказанных пересечений линеек, где краска действительно есть."""
    if not lines.horizontal or not lines.vertical:
        return 0.0
    tolerance = mm_to_px(CROSS_TOL_MM, dpi)
    mask = lines.mask
    height, width = mask.shape[:2]
    hits = total = 0
    for horizontal in lines.horizontal:
        y = (horizontal.box.y0 + horizontal.box.y1) // 2
        for vertical in lines.vertical:
            x = (vertical.box.x0 + vertical.box.x1) // 2
            # Пересечение предсказывается только там, где отрезки действительно
            # перекрываются по своей длинной оси.
            if not (horizontal.box.x0 - tolerance <= x <= horizontal.box.x1 + tolerance):
                continue
            if not (vertical.box.y0 - tolerance <= y <= vertical.box.y1 + tolerance):
                continue
            total += 1
            window = mask[
                max(0, y - tolerance) : min(height, y + tolerance + 1),
                max(0, x - tolerance) : min(width, x + tolerance + 1),
            ]
            hits += int(window.any())
    return hits / total if total else 0.0


def _filled_cells(mask: np.ndarray, lines: Lines, dpi: int) -> tuple[int, float]:
    """Сколько в находке ячеек и какая доля из них с текстом.

    Главный признак против координатной сетки и чертежа: у таблицы клетки заполнены
    текстом, у сетки графика они пустые, у чертежа заполнены как попало.
    """
    from ocr_utils.page_layout.tables.grid import grid_from_lines

    grid = grid_from_lines(lines, mask.shape[:2], dpi)
    if not grid.cells:
        return 0, 0.0
    filled = 0
    for cell in grid.cells:
        box = cell.inner or cell.box
        patch = mask[box.clipped(mask.shape[1], mask.shape[0]).slice]
        if patch.size and np.count_nonzero(patch) >= CELL_INK_SHARE * patch.size:
            filled += 1
    return len(grid.cells), filled / len(grid.cells)


def _edge_fill(lines: Lines, shape: tuple[int, int], dpi: int) -> float:
    """Доля ВНУТРЕННИХ рёбер сетки, под которыми действительно есть линейка."""
    from ocr_utils.page_layout.tables.grid import _edge_tables, separators

    xs, ys, _ = separators(lines, shape, dpi)
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    column_edges, row_edges = _edge_tables(lines, xs, ys, dpi)
    present = [column_edges[row][column].present for row in range(len(ys) - 1) for column in range(1, len(xs) - 1)]
    present += [row_edges[column][row].present for column in range(len(xs) - 1) for row in range(1, len(ys) - 1)]
    return sum(present) / len(present) if present else 0.0


def features(gray: np.ndarray, lines: Lines, dpi: int) -> Features:
    """Все признаки одной находки. ``gray`` — вырезка находки, ``lines`` — её линейки."""
    mask = glyph_mask(gray, dpi)
    ink = int(np.count_nonzero(binarize(gray)))
    vertical, horizontal = _inner_counts(lines, gray.shape[:2], dpi)
    gutters, gutter_share = _gutters(mask, dpi)
    cells, filled = _filled_cells(mask, lines, dpi)
    return Features(
        inner_vertical=vertical,
        inner_horizontal=horizontal,
        gutters=gutters,
        gutter_share=gutter_share,
        line_span=_line_span(mask, dpi),
        lattice=_lattice(lines, dpi),
        glyph_share=float(np.count_nonzero(mask)) / max(1, mask.size),
        glyph_of_ink=float(np.count_nonzero(mask)) / max(1, ink),
        edge_fill=_edge_fill(lines, gray.shape[:2], dpi),
        cells=cells,
        filled_cells=filled,
    )


# --- Маска глифов ---------------------------------------------------------------------

# Размер глифа в миллиметрах: от 0.5 мм (точка над «й», запятая) до 5 мм (заглавная
# заголовка). Замер по ячейкам таблиц пака: медианная высота компоненты 14-21 px при
# 300 dpi, то есть 1.2-1.8 мм.
GLYPH_MIN_MM = 0.5
GLYPH_MAX_MM = 5.0

# Компонента длиннее трёх максимальных высот глифа — обрывок линейки, а не буква.
GLYPH_MAX_LONG_FACTOR = 3.0

# Заполнение габаритного прямоугольника краской: буква заполняет его хотя бы на 10%.
GLYPH_MIN_FILL = 0.10

# Зазор смыкания RLSA: 0.7 мм — межбуквенный и межсловный пробел петита. При 300 dpi это
# 8 px, на которых и сделан замер выше.
RLSA_GAP_MM = 0.7

# Что считается «строкой» после смыкания: вытянуто хотя бы втрое и длиной от 3.4 мм.
LINE_ASPECT = 3.0
LINE_MIN_LENGTH_MM = 3.4

# Меньше этой доли краски в «строках» — мерить нечего.
MIN_LINE_INK_FRAC = 0.004

# Ниже этого перевеса ось считается неразличимой.
AXIS_MARGIN_THR = 0.25

# Меньше этого числа строк — асимметрия считается по шуму, знак не определён.
MIN_LINES_FOR_SIGN = 2


def glyph_mask(gray: np.ndarray, dpi: int) -> np.ndarray:
    """Маска компонент размера глифа: краска без линеек и без пыли."""
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    if count <= 1:
        return np.zeros_like(ink)
    minimum = mm_to_px(GLYPH_MIN_MM, dpi)
    maximum = mm_to_px(GLYPH_MAX_MM, dpi)
    width = stats[:, cv2.CC_STAT_WIDTH]
    height = stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]
    long_side = np.maximum(width, height)
    keep = (
        (np.minimum(width, height) >= 2)
        & (long_side >= minimum)
        & (long_side <= maximum * GLYPH_MAX_LONG_FACTOR)
        & (np.minimum(width, height) <= maximum)
        & (area >= GLYPH_MIN_FILL * width * height)
    )
    keep[0] = False
    return np.where(keep[labels], np.uint8(255), np.uint8(0))
