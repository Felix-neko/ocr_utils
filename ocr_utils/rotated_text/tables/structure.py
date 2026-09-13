"""Сетка ячеек вырезанной таблицы и всё, что о ней можно измерить без распознавания.

Сетка строится тем же кодом, что проверяет находки детектора таблиц
(``scan_markup.table_detection.grid``): линейки → разделители → объединения по отсутствующим
линейкам → шапка по двойной линейке. Здесь только обёртка: рабочая копия в 300 dpi,
внутренность ячейки без обрубков линеек и пересчёт сетки в другое разрешение.

ПОЧЕМУ 300 DPI. Порог сетки калиброван на нём; при 150 петит шапки даёт 7 px, и буква не
отличима от точки; при 600 морфология стоит вчетверо дороже и не даёт ничего нового
(``table_detection/README``). Рендер потом идёт в исходном разрешении, и сетка просто
умножается на два — ошибка округления в пиксель против отступа от линейки в 0.3 мм (7 px
при 600 dpi) не страшна.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import cv2
import numpy as np

from ocr_utils.scan_markup.rotation import rotate_cw
from ocr_utils.scan_markup.table_detection.geometry import Box, Cell, Grid
from ocr_utils.scan_markup.table_detection.grid import NO_RULE_PAD_MM, WORK_DPI, grid_from_lines, text_ink
from ocr_utils.scan_markup.table_detection.ruling import Lines, binarize, find_lines, mm_to_px
from ocr_utils.scan_markup.table_detection.verify import glyph_mask

# Обрубок линейки у края внутренности: тонкий (до 1 мм), вытянутый (втрое) и от 1.2 мм
# длиной. От штриха буквы «П» его отличает КАСАНИЕ КРАЯ: текст отбит от линейки отступом,
# а обрубок линейки касается края по построению.
STUB_MAX_THICKNESS_MM = 1.0
STUB_MIN_LENGTH_MM = 1.2
STUB_ASPECT = 3.0

# Кегль по высоте глифа: DejaVu Sans Condensed имеет высоту заглавной ≈0.73 em и строчной
# ≈0.55 em; смешанная медиана компонент ≈0.62 em, откуда множитель 1.6.
FONT_FROM_GLYPH = 1.6

# Компонента мельче 0.15 мм² (21 px² при 300 dpi) — пыль или точка над «й», в медиану
# высоты она не идёт.
MIN_GLYPH_MM2 = 0.15


@dataclass
class TableStructure:
    """Сетка и уровни бумаги и краски по всей таблице."""

    grid: Grid
    lines: Lines
    paper: int
    ink: int


def work_copy(gray: np.ndarray, dpi: int, work_dpi: int = WORK_DPI) -> np.ndarray:
    """Копия в рабочем разрешении."""
    if dpi == work_dpi:
        return gray
    factor = work_dpi / dpi
    interpolation = cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
    return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=interpolation)


def analyse_structure(work: np.ndarray, dpi: int = WORK_DPI) -> TableStructure:
    lines = find_lines(work, dpi)
    grid = grid_from_lines(lines, work.shape[:2], dpi, text_ink(work, lines))
    return TableStructure(grid=grid, lines=lines, paper=paper_level(work), ink=ink_level(work))


def paper_level(gray: np.ndarray) -> int:
    return int(np.percentile(gray, 90)) if gray.size else 255


def ink_level(gray: np.ndarray) -> int:
    """Краска всей таблицы, а не ячейки: у закрашенной ячейки квантиль уплыл бы вверх."""
    return int(np.percentile(gray, 5)) if gray.size else 0


def interior_box(cell: Cell, dpi: int) -> Box:
    """Внутренность ячейки без линеек: измеренная сеткой, иначе отступ на глазок."""
    if cell.inner is not None:
        return cell.inner
    pad = mm_to_px(NO_RULE_PAD_MM, dpi)
    return Box(
        cell.box.x0 + pad,
        cell.box.y0 + pad,
        max(cell.box.x0 + pad, cell.box.x1 - pad),
        max(cell.box.y0 + pad, cell.box.y1 - pad),
    )


def cell_interior(gray: np.ndarray, cell: Cell, dpi: int) -> np.ndarray:
    """Вырезка внутренности ячейки, очищенная от обрубков линеек. Пустая ячейка — 1×1 бумага."""
    box = interior_box(cell, dpi).clipped(gray.shape[1], gray.shape[0])
    if box.width <= 0 or box.height <= 0:
        return np.full((1, 1), 255, np.uint8)
    return strip_stubs(gray[box.slice], dpi)


def strip_stubs(gray: np.ndarray, dpi: int) -> np.ndarray:
    """Убрать длинные тонкие штрихи, приросшие к краю вырезки: наклонная линейка заходит в
    ячейку, и tesseract читает её как «|» или «—»."""
    height, width = gray.shape[:2]
    binary = binarize(gray)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return gray
    thickness_px = mm_to_px(STUB_MAX_THICKNESS_MM, dpi)
    length_px = mm_to_px(STUB_MIN_LENGTH_MM, dpi)
    left, top = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    box_w, box_h = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    touches = (left == 0) | (top == 0) | (left + box_w >= width) | (top + box_h >= height)
    thin, long_side = np.minimum(box_w, box_h), np.maximum(box_w, box_h)
    stub = (
        touches & (thin <= thickness_px) & (long_side >= length_px) & (long_side >= STUB_ASPECT * np.maximum(thin, 1))
    )
    stub[0] = False
    if not stub.any():
        return gray
    cleaned = gray.copy()
    cleaned[stub[labels]] = paper_level(gray)
    return cleaned


def glyph_height(gray: np.ndarray, dpi: int) -> float:
    """Медианная высота глифа в ВЫПРЯМЛЕННОЙ вырезке; 0 — мерить не по чему."""
    mask = glyph_mask(gray, dpi)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return 0.0
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    widths = stats[1:, cv2.CC_STAT_WIDTH].astype(float)
    keep = (widths * heights) >= MIN_GLYPH_MM2 * (dpi / 25.4) ** 2
    return float(np.median(heights[keep])) if keep.any() else 0.0


def native_font_px(gray: np.ndarray, cells: Iterable[tuple[Cell, int]], dpi: int) -> float:
    """Собственный кегль таблицы в пикселях: медиана высот глифов × 1.6 по ячейкам с
    известным поворотом (каждая сперва выпрямляется своим углом)."""
    heights: list[float] = []
    for cell, rotate in cells:
        value = glyph_height(rotate_cw(cell_interior(gray, cell, dpi), rotate), dpi)
        if value > 0:
            heights.append(value)
    return float(np.median(heights)) * FONT_FROM_GLYPH if heights else 0.0


# Склейка объединённых ячеек, которые сетка разрезала. Линейка на границе считается
# существующей при той же доле покрытия, что у сетки (``grid.SEPARATOR_FILL``). Текст
# «пересекает» границу в двух случаях: буква лежит поперёк неё, заходя не меньше чем на
# 0.3 мм с каждой стороны (меньше — выносной элемент или засечка), либо две буквы стоят по
# разные стороны границы вплотную — просвет между ними не больше межбуквенного (0.7 мм,
# тот же ``RLSA_GAP_MM``) — и выровнены по одной строке (перекрытие по протяжению границы
# не меньше 0.6 большей из них). Второй случай нужен потому, что граница попадает в
# просвет между буквами так же часто, как на букву.
BOUNDARY_FILL = 0.5
BOUNDARY_TOL_MM = 1.0
CROSS_DEPTH_MM = 0.3
CROSS_GAP_MM = 0.7
CROSS_ALIGN = 0.6


def split_overlapping(grid: Grid) -> Grid:
    """Разобрать на клетки ячейки сетки, накрывающие чужие клетки.

    Сетка объединяет клетки по отсутствию линейки и строит ячейку по габариту группы;
    Г-образная группа даёт ячейку, накрывающую клетки, которые ей не принадлежат
    (1966/06 IMG_0137: «стоимость» над двумя графами и «сумма отклонений» на две строки
    рядом — одна ячейка 2×3 поверх «по прейскуранту» и «фактическая»). Такая ячейка
    разбирается на свои клетки, а склейка по тексту (``merge_split_cells``) собирает из
    них настоящие объединения."""
    owners: dict[tuple[int, int], list[int]] = {}
    for index, cell in enumerate(grid.cells):
        for r in range(cell.row, cell.row + cell.row_span):
            for c in range(cell.col, cell.col + cell.col_span):
                owners.setdefault((r, c), []).append(index)
    overlapping = {i for members in owners.values() if len(members) > 1 for i in members}
    if not overlapping:
        return grid
    # Из перекрывшихся клетка достаётся той ячейке, у которой она единственная, иначе —
    # меньшей; у остальных перекрывшихся ячеек все клетки становятся отдельными.
    result: list[Cell] = []
    for index, cell in enumerate(grid.cells):
        if index not in overlapping:
            result.append(cell)
            continue
        for r in range(cell.row, cell.row + cell.row_span):
            for c in range(cell.col, cell.col + cell.col_span):
                claimants = owners[(r, c)]
                if min(claimants, key=lambda i: grid.cells[i].row_span * grid.cells[i].col_span) != index:
                    continue
                if c + 1 >= len(grid.xs) or r + 1 >= len(grid.ys):
                    continue
                result.append(
                    Cell(
                        row=r,
                        col=c,
                        box=Box(grid.xs[c], grid.ys[r], grid.xs[c + 1], grid.ys[r + 1]),
                        is_header=cell.is_header,
                    )
                )
    result.sort(key=lambda c: (c.row, c.col))
    return Grid(
        xs=list(grid.xs),
        ys=list(grid.ys),
        cells=result,
        header_rows=grid.header_rows,
        double_rule_ys=list(grid.double_rule_ys),
        source=grid.source,
        column_cuts=list(grid.column_cuts),
    )


def merge_split_cells(grid: Grid, lines: Lines, work: np.ndarray, dpi: int) -> Grid:
    """Склеить соседние ячейки, между которыми нет линейки, а текст границу пересекает.

    ЗАЧЕМ. Сетка объединяет клетки по отсутствию линейки, но через СКВОЗНУЮ строку
    (линейка под большинством колонок) объединять отказывается — иначе шапка склеивалась
    бы с телом там, где линейка под одной колонкой не пропечаталась. Цена правила: боковая
    надпись, тянущаяся через несколько строк без линеек поперёк (приписка в крайней графе
    1969/08, шапка глубже соседних), режется на куски по строкам и читается как «мос»,
    «олеб», «про» — а набор потом рубит слова и буквы.

    ПРИЗНАК ОБЪЕДИНЕНИЯ — тот же, что у Camelot и у split-and-merge распознавателей
    структуры таблиц: границы между клетками нет, если на ней нет линейки И через неё
    проходит текст. Второе условие — решающее: строки прямого текста границ не пересекают
    никогда, поэтому шапка с телом не склеится, даже если линейки между ними нет; буква же,
    физически лежащая поперёк границы, — прямое свидетельство, что границы там нет.
    Компоненты берутся по краске БЕЗ линеек: вертикальная линейка пересекает каждую
    горизонтальную границу, и по сырой краске она сошла бы за букву.

    Склеиваются только пары с одинаковым протяжением вдоль общей границы, и только группы,
    сложившиеся в прямоугольник: Г-образное объединение в таблице невозможно.
    """
    if not grid.cells:
        return grid
    grid = split_overlapping(grid)
    tol = mm_to_px(BOUNDARY_TOL_MM, dpi)
    depth = (mm_to_px(CROSS_DEPTH_MM, dpi), mm_to_px(CROSS_GAP_MM, dpi))
    boxes = _glyph_boxes(work, lines, dpi)

    index = {cell.key: i for i, cell in enumerate(grid.cells)}
    by_key = {cell.key: cell for cell in grid.cells}
    parent = list(range(len(grid.cells)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for cell in grid.cells:
        below = by_key.get((cell.row + cell.row_span, cell.col))
        if below is not None and below.col_span == cell.col_span:
            if _boundary_open(lines.horizontal_mask, boxes, cell.box.y1, cell.box.x0, cell.box.x1, True, tol, depth):
                union(index[cell.key], index[below.key])
        right = by_key.get((cell.row, cell.col + cell.col_span))
        if right is not None and right.row_span == cell.row_span:
            if _boundary_open(lines.vertical_mask, boxes, cell.box.x1, cell.box.y0, cell.box.y1, False, tol, depth):
                union(index[cell.key], index[right.key])

    groups: dict[int, list[Cell]] = {}
    for i, cell in enumerate(grid.cells):
        groups.setdefault(find(i), []).append(cell)

    merged: list[Cell] = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(members[0])
            continue
        rows = range(min(c.row for c in members), max(c.row + c.row_span for c in members))
        cols = range(min(c.col for c in members), max(c.col + c.col_span for c in members))
        covered = sum(c.row_span * c.col_span for c in members)
        if covered != len(rows) * len(cols):
            merged.extend(members)  # не прямоугольник — оставляем как было
            continue
        inners = [c.inner for c in members if c.inner is not None]
        merged.append(
            Cell(
                row=rows.start,
                col=cols.start,
                box=Box(
                    min(c.box.x0 for c in members),
                    min(c.box.y0 for c in members),
                    max(c.box.x1 for c in members),
                    max(c.box.y1 for c in members),
                ),
                row_span=len(rows),
                col_span=len(cols),
                is_header=any(c.is_header for c in members),
                inner=(
                    Box(
                        min(b.x0 for b in inners),
                        min(b.y0 for b in inners),
                        max(b.x1 for b in inners),
                        max(b.y1 for b in inners),
                    )
                    if len(inners) == len(members)
                    else None
                ),
            )
        )
    merged.sort(key=lambda c: (c.row, c.col))
    return Grid(
        xs=list(grid.xs),
        ys=list(grid.ys),
        cells=merged,
        header_rows=grid.header_rows,
        double_rule_ys=list(grid.double_rule_ys),
        source=grid.source,
        column_cuts=list(grid.column_cuts),
    )


def _glyph_boxes(work: np.ndarray, lines: Lines, dpi: int) -> np.ndarray:
    """Габариты компонент размера буквы по краске без линеек: ``[x0, y0, x1, y1]`` построчно."""
    from ocr_utils.scan_markup.table_detection.grid import text_ink
    from ocr_utils.scan_markup.table_detection.verify import GLYPH_MAX_MM, GLYPH_MIN_MM

    ink = text_ink(work, lines)
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    if count <= 1:
        return np.zeros((0, 4), int)
    minimum, maximum = mm_to_px(GLYPH_MIN_MM, dpi), mm_to_px(GLYPH_MAX_MM, dpi)
    w, h = stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]
    keep = (np.maximum(w, h) >= minimum) & (np.maximum(w, h) <= maximum) & (np.minimum(w, h) >= 2)
    x0, y0 = stats[1:, cv2.CC_STAT_LEFT][keep], stats[1:, cv2.CC_STAT_TOP][keep]
    return np.stack([x0, y0, x0 + w[keep], y0 + h[keep]], axis=1)


def _boundary_open(
    mask: np.ndarray,
    boxes: np.ndarray,
    position: int,
    start: int,
    end: int,
    horizontal: bool,
    tol: int,
    depth: tuple[int, int],
) -> bool:
    """Нет ли линейки на границе и пересекает ли её текст. ``position`` — координата
    границы поперёк, ``[start, end)`` — её протяжение вдоль, ``depth`` — (глубина захода
    буквы за границу, допустимый просвет между буквами по разные стороны)."""
    a, b = start + tol, end - tol
    if b <= a:
        return False
    lo, hi = max(0, position - tol), position + tol + 1
    band = mask[lo:hi, a:b] if horizontal else mask[a:b, lo:hi]
    if band.size and band.any(axis=0 if horizontal else 1).mean() >= BOUNDARY_FILL:
        return False
    if boxes.size == 0:
        return False
    if horizontal:
        along0, along1, across0, across1 = boxes[:, 0], boxes[:, 2], boxes[:, 1], boxes[:, 3]
    else:
        along0, along1, across0, across1 = boxes[:, 1], boxes[:, 3], boxes[:, 0], boxes[:, 2]
    inside = (along1 > a) & (along0 < b)
    cross, gap = depth
    if bool(((across0 <= position - cross) & (across1 >= position + cross) & inside).any()):
        return True
    # Две буквы вплотную по разные стороны границы, на одной строке.
    before = inside & (across1 <= position + cross) & (across1 >= position - gap)
    after = inside & (across0 >= position - cross) & (across0 <= position + gap)
    if not before.any() or not after.any():
        return False
    b0, b1, b_end = along0[before], along1[before], across1[before]
    a0, a1, a_start = along0[after], along1[after], across0[after]
    close = (a_start[None, :] - b_end[:, None]) <= gap
    overlap = np.minimum(b1[:, None], a1[None, :]) - np.maximum(b0[:, None], a0[None, :])
    widest = np.maximum((b1 - b0)[:, None], (a1 - a0)[None, :])
    return bool((close & (overlap >= CROSS_ALIGN * widest)).any())


def scale_grid(grid: Grid, factor: float) -> Grid:
    """Сетка в другом разрешении. Единица — исходные пиксели, округление к ближайшему."""
    if factor == 1:
        return grid

    def scaled(value: int) -> int:
        return int(round(value * factor))

    cells = [
        replace(cell, box=cell.box.scaled(factor), inner=cell.inner.scaled(factor) if cell.inner is not None else None)
        for cell in grid.cells
    ]
    return Grid(
        xs=[scaled(v) for v in grid.xs],
        ys=[scaled(v) for v in grid.ys],
        cells=cells,
        header_rows=grid.header_rows,
        double_rule_ys=[scaled(v) for v in grid.double_rule_ys],
        source=grid.source,
        column_cuts=[scaled(v) for v in grid.column_cuts],
    )
