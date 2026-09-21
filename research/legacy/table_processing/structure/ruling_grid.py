"""Сетка ячеек по линейкам: разделители, объединения, шапка.

ЛИНЕЙКИ УЖЕ НАЙДЕНЫ детектором таблиц — здесь они превращаются в сетку.

1. РАЗДЕЛИТЕЛИ. Отрезки одного направления, стоящие на одной координате, — это один
   разделитель. Внешние границы таблицы добавляются всегда: у этих таблиц внешней рамки
   слева и справа часто нет вовсе, а колонка у края есть.

2. ЛИНЕЙКА У КАЖДОЙ ПОЛОСЫ СВОЯ. Разделитель — не одна прямая на всю таблицу: линейка
   шапки и линейка тела печатались и сканировались по-разному. Замер по 80 вырезкам пака:
   разброс центров отрезков внутри ОДНОЙ колонки — медиана 5 px при 300 dpi, p90 7, p99 16,
   максимум 20, то есть до ширины цифры; угол линейки — медиана 0.19°, p90 0.61°, до 1.69°.
   Поэтому каждая клетка получает свои четыре :class:`Edge` — положение и толщину линейки
   ИМЕННО В ЭТОЙ полосе, — и рамка ячейки собирается из них, а не из общего разделителя.

3. ДВОЙНАЯ ЛИНЕЙКА. Шапку в этих журналах отбивают двумя близкими линейками. Две линейки
   ближе 1.5 мм — это одна граница, и она же говорит, где кончается шапка. Знать это важно:
   боковой текст живёт в шапке.

4. ОБЪЕДИНЁННЫЕ ЯЧЕЙКИ. Разделителя между двумя соседними клетками может не быть — тогда
   это одна ячейка на две клетки. Проверяется не наличие отрезка вообще, а доля стороны
   клетки, вдоль которой есть краска разделителя: заголовок «Хвойные породы» над двумя
   графами именно так и выглядит — вертикальная линейка есть ниже него, но не на его высоте.

5. ВНУТРЕННОСТЬ ЯЧЕЙКИ (``Cell.inner``) отступает от каждой линейки на её ИЗМЕРЕННУЮ
   половину толщины, а не на общее число. Толщина линейки в этих вырезках гуляет от 2 до
   9 px при 300 dpi, и единый отступ 0.6 мм либо оставлял обрывок линейки в ячейке (его
   потом дважды подчищали ниже по конвейеру), либо съедал выносные элементы.

ЖИВОЙ КОД ПЕРЕЕХАЛ в ``ocr_utils.page_layout.tables.grid``: здесь остался реэкспорт, чтобы стенд исследований
(сравнение версий, добыча, отчёты) мерил тот же детектор, что стоит в конвейере.
Здесь остались извлечение ячеек под распознавание (``extract``, ``strip_rules``, ``cell_image``).
"""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.tables.grid import (  # noqa: F401 — реэкспорт для стенда
    WORK_DPI,
    SEPARATOR_TOL_MM,
    DOUBLE_RULE_MM,
    SEPARATOR_FILL,
    GLOBAL_ROW_SHARE,
    GLOBAL_ROW_MIN_COLUMNS,
    MIN_CELL_MM,
    OUTER_COLUMN_MM,
    NO_RULE_PAD_MM,
    RULE_EDGE_SHARE,
    RULE_CLEARANCE_MM,
    BAND_TEXT_COMPONENTS,
    BAND_GLYPH_MM,
    RULE_MARGIN_PX,
    _cluster,
    has_text,
    _trim_empty_bands,
    separators,
    edge_at,
    _edge_tables,
    _trim_ruleless_rows,
    _weighted,
    grid_from_lines,
    _column_cuts,
    _build_cell,
    _fallback_inner,
    _header_rows,
    text_ink,
)
from research.legacy.table_processing.detection.ruling import Lines, binarize, find_lines, mm_to_px
from research.legacy.table_processing.geometry import Box, Cell, Grid
from research.legacy.table_processing.structure.base import CellExtractor


def extract(gray: np.ndarray, dpi: int = WORK_DPI) -> Grid:
    """Сетка вырезанной таблицы."""
    lines = find_lines(gray, dpi)
    return grid_from_lines(lines, gray.shape[:2], dpi, text_ink(gray, lines))


def strip_rules(gray: np.ndarray, dpi: int, level: "int | None" = None) -> np.ndarray:
    """Копия вырезки, из которой убраны обрывки линеек.

    ЗАЧЕМ, ЕСЛИ ВНУТРЕННОСТЬ ЯЧЕЙКИ УЖЕ ОТСТУПАЕТ ОТ ЛИНЕЙКИ на её измеренную толщину.
    Затем, что отступ спасает от прямой линейки, а не от косой: у линейки под углом 1.7°
    (такие в паке есть) край заходит в ячейку на несколько пикселей. Дальше он портит всё
    сразу — в счёте формы букв это компонента с диким отношением сторон, а tesseract читает
    его как «|» или «—» и вставляет в текст.

    Убираются именно ДЛИННЫЕ И ТОНКИЕ штрихи — те же, что ищет детектор таблиц, но порог
    длины здесь свой: внутри ячейки шириной 8 мм обрывок разделителя короче общего порога.
    Берётся 40% меньшей стороны вырезки, но не меньше 2 мм. Буква под это описание не
    подходит: непрерывного штриха в 40% ширины ячейки у неё не бывает.
    """
    side = min(gray.shape[:2])
    lines = find_lines(gray, dpi, max(mm_to_px(2.0, dpi), int(side * 0.4)))
    mask = cv2.bitwise_or(lines.mask, _border_stubs(gray, dpi))
    if not mask.any():
        return gray
    cleaned = gray.copy()
    paper = int(np.percentile(gray, 90)) if level is None else level
    cleaned[mask > 0] = paper
    return cleaned


# Обрубок разделителя у края вырезки: тонкий (не толще 1 мм), вытянутый (втрое) и не короче
# 1.2 мм. По длине его не отличить от вертикального штриха буквы «П», поэтому решает
# КАСАНИЕ КРАЯ: текст внутри ячейки отбит от разделителя отступом и края не касается,
# а обрубок линейки касается по построению.
STUB_MAX_THICKNESS_MM = 1.0
STUB_MIN_LENGTH_MM = 1.2
STUB_ASPECT = 3.0


def _border_stubs(gray: np.ndarray, dpi: int) -> np.ndarray:
    """Маска обрубков линеек, приросших к краю вырезки."""
    height, width = gray.shape[:2]
    binary = binarize(gray)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    stub = np.zeros_like(binary)
    if count <= 1:
        return stub
    thickness_px = mm_to_px(STUB_MAX_THICKNESS_MM, dpi)
    length_px = mm_to_px(STUB_MIN_LENGTH_MM, dpi)
    for index in range(1, count):
        left, top, box_width, box_height, _ = stats[index]
        touches = left == 0 or top == 0 or left + box_width >= width or top + box_height >= height
        if not touches:
            continue
        thin, long_side = min(box_width, box_height), max(box_width, box_height)
        if thin <= thickness_px and long_side >= length_px and long_side >= STUB_ASPECT * max(thin, 1):
            stub[labels == index] = 255
    return stub


def cell_image(table: np.ndarray, cell: Cell, dpi: int, pad_mm: float = NO_RULE_PAD_MM) -> np.ndarray:
    """Внутренность ячейки, готовая и для детекторов, и для распознавания."""
    box = interior(cell, dpi, pad_mm).clipped(table.shape[1], table.shape[0])
    if box.width <= 0 or box.height <= 0:
        return np.full((1, 1), 255, np.uint8)
    return strip_rules(table[box.slice], dpi)


def interior(cell: Cell, dpi: int, pad_mm: float = NO_RULE_PAD_MM) -> Box:
    """Внутренность ячейки без линеек.

    Если сетка посчитала её по измеренным линейкам (``Cell.inner``), берётся она. Иначе —
    отступ на глазок: так приходят ячейки из синтетики и из ручной разметки, где линеек
    никто не мерил.
    """
    if cell.inner is not None:
        return cell.inner
    return _fallback_inner(cell.box, mm_to_px(pad_mm, dpi))


ALGORITHM = CellExtractor(
    name="ruling_grid",
    summary="классика: разделители по линейкам, объединения по их отсутствию (CPU)",
    stage="cpu",
    run=extract,
)
