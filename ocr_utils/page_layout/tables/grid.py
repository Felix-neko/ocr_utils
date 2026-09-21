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
"""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.tables.ruling import Lines, binarize, find_lines, mm_to_px
from ocr_utils.page_layout.geometry import Box, Cell, Edge, Grid

# Рабочее разрешение разбора вырезанной таблицы. Выше, чем у поиска таблиц на полосе
# (150 dpi): там нужна рамка с точностью до миллиметра, здесь — граница ячейки, по которой
# будут резать текст, и промах в миллиметр съел бы выносные элементы.
WORK_DPI = 300

# Два отрезка считаются одним разделителем, если их координаты ближе этого: 1.0 мм.
SEPARATOR_TOL_MM = 1.0

# Две горизонтальные линейки ближе этого — двойная линейка, то есть одна граница: 1.5 мм.
DOUBLE_RULE_MM = 1.5

# Разделитель считается существующим, если краска покрывает такую долю стороны клетки.
# 0.5, а не 0.9: линейка рвётся там, где её пересекает цифра, и в вырезке с полями края
# сторон часто пустые.
SEPARATOR_FILL = 0.5

# Доля колонок, при которой горизонтальная линейка считается СКВОЗНОЙ границей строк: через
# такую границу ячейки не объединяются, даже если под конкретной колонкой линейки нет.
#
# Зачем. У таблицы 1972/06 «Форма лимитной карточки» под шапкой идёт строка с номерами граф,
# и линейка над ней есть под девятью колонками из пятнадцати (0.60), а под остальными шестью
# её нет — там шапка нарисована глубже. Без этого правила шапка этих шести колонок
# объединялась с номером графы, и замена бокового текста стирала номер вместе с ним.
# У настоящих объединений по вертикали доля ниже: у таблицы 1966/01 «Поставки, тыс. шт.»
# линейка под подзаголовком идёт под двумя колонками из четырёх (0.50), и объединять через
# неё МОЖНО — иначе боковая шапка справа разрежется пополам. Отсюда и порог, и второе
# условие: сквозной считается линейка не меньше чем под тремя колонками.
GLOBAL_ROW_SHARE = 0.6
GLOBAL_ROW_MIN_COLUMNS = 3

# Клетка тоньше этого не бывает: 2 мм. Всё, что уже, — расщеплённая пополам линейка.
MIN_CELL_MM = 2.0

# На сколько край вырезки должен отстоять от крайней линейки, чтобы там признали колонку
# без линейки: 5 мм. Меньше — это поля вырезки (их добавляет ``mining.export``, 3 мм), а
# не графа. Без отдельного порога синтетическая таблица с полями в 30 px получала лишнюю
# пустую колонку с каждой стороны.
OUTER_COLUMN_MM = 5.0

# Отступ внутрь ячейки там, где линейки нет вовсе и мерить нечего: 0.6 мм.
NO_RULE_PAD_MM = 0.6

# Доля строк полосы, в которых должна стоять краска, чтобы колонка считалась частью штриха
# линейки, а не случайной буквой, заехавшей в окно поиска.
RULE_EDGE_SHARE = 0.2

# Зазор ЗА краем штриха линейки, на который дополнительно отступает внутренность ячейки.
# Ставится по замеру на 37 размеченных ячейках: срез ровно по краю краски (зазор 0) даёт
# tesseract CER 0.058 против 0.036 у прежнего отступа в 0.6 мм от центра линейки — у самой
# линейки остаётся серая кайма, и распознаватель теряет на ней строку или дописывает «_|».
RULE_CLEARANCE_MM = 0.3

# Крайняя полоса выбрасывается, если в ней нет ТЕКСТА. Так уходят полосы-призраки: между
# верхом вырезки и верхней линейкой таблицы часто стоит линейка над заголовком страницы
# (1966/06, таблица 7), и из неё получалась пустая строка сетки.
#
# Наличие текста проверяется по компонентам размера буквы, а не по доле краски. Доля не
# годится: после вычитания линеек от них остаётся кайма в сотню пикселей, и пустая полоса
# набирает столько же, сколько полоса с текстом (замерено на синтетике: 0.0070 против
# 0.0072). Компонент же размера буквы в кайме не бывает.
BAND_TEXT_COMPONENTS = 3
BAND_GLYPH_MM = 0.8
RULE_MARGIN_PX = 3


def _cluster(values: list[int], tolerance: int) -> list[int]:
    """Близкие координаты — в одну; представитель группы — медиана.

    Медиана, а не среднее: группа собирается цепочкой (каждое значение сравнивается с
    предыдущим), и при трёх отрезках подряд с шагом чуть меньше допуска среднее уезжает от
    настоящей линейки, а медиана остаётся на ней.
    """
    if not values:
        return []
    ordered = sorted(values)
    groups: list[list[int]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [int(round(float(np.median(group)))) for group in groups]


def has_text(band: np.ndarray, dpi: int) -> bool:
    """Есть ли в куске краски хоть сколько-то компонент размера буквы."""
    if band.size == 0:
        return False
    count, _, stats, _ = cv2.connectedComponentsWithStats(band, 8)
    if count <= 1:
        return False
    width = stats[1:, cv2.CC_STAT_WIDTH]
    height = stats[1:, cv2.CC_STAT_HEIGHT]
    minimal = mm_to_px(BAND_GLYPH_MM, dpi)
    glyphs = (np.minimum(width, height) >= 2) & (np.maximum(width, height) >= minimal)
    return int(glyphs.sum()) >= BAND_TEXT_COMPONENTS


def _trim_empty_bands(values: list[int], ink: np.ndarray, vertical: bool, dpi: int) -> list[int]:
    """Выбросить крайние полосы без текста — линейку над заголовком и обрезной край."""
    minimal = mm_to_px(MIN_CELL_MM, dpi)

    def keep(start: int, end: int) -> bool:
        band = ink[:, start:end] if vertical else ink[start:end, :]
        return end - start >= minimal and has_text(band, dpi)

    while len(values) > 2 and not keep(values[0], values[1]):
        values = values[1:]
    while len(values) > 2 and not keep(values[-2], values[-1]):
        values = values[:-1]
    return values


def separators(
    lines: Lines, shape: tuple[int, int], dpi: int, ink: "np.ndarray | None" = None
) -> tuple[list[int], list[int], list[int]]:
    """Координаты разделителей: вертикальные, горизонтальные и двойные горизонтальные."""
    height, width = shape
    tolerance = mm_to_px(SEPARATOR_TOL_MM, dpi)
    minimal = mm_to_px(OUTER_COLUMN_MM, dpi)

    xs = _cluster([(s.box.x0 + s.box.x1) // 2 for s in lines.vertical], tolerance)
    ys = _cluster([(s.box.y0 + s.box.y1) // 2 for s in lines.horizontal], tolerance)

    # Внешние границы: у этих таблиц внешней рамки слева и справа часто нет.
    if not xs or xs[0] > minimal:
        xs.insert(0, 0)
    if not xs or width - xs[-1] > minimal:
        xs.append(width - 1)
    if not ys or ys[0] > minimal:
        ys.insert(0, 0)
    if not ys or height - ys[-1] > minimal:
        ys.append(height - 1)

    double_gap = mm_to_px(DOUBLE_RULE_MM, dpi)
    doubles = [ys[index] for index in range(len(ys) - 1) if ys[index + 1] - ys[index] <= double_gap]
    # Двойная линейка — одна граница: вторую из пары выбрасываем, но помним, где она была.
    collapsed: list[int] = []
    skip_next = False
    for index, value in enumerate(ys):
        if skip_next:
            skip_next = False
            continue
        if index + 1 < len(ys) and ys[index + 1] - value <= double_gap:
            collapsed.append((value + ys[index + 1]) // 2)
            skip_next = True
        else:
            collapsed.append(value)

    if ink is not None:
        xs = _trim_empty_bands(xs, ink, True, dpi)
        collapsed = _trim_empty_bands(collapsed, ink, False, dpi)
    return xs, collapsed, doubles


def edge_at(mask: np.ndarray, position: int, start: int, end: int, vertical: bool, tolerance: int) -> Edge:
    """Линейка в окне ``position ± tolerance``, но только внутри полосы ``[start, end)``.

    Положение уточняется взвешенным по краске центроидом, толщина считается как площадь,
    делённая на длину полосы — та же мера, что у детектора таблиц, и по той же причине:
    габарит наклонной линейки шире её штриха.
    """
    if end <= start:
        return Edge(position)
    low = max(0, position - tolerance)
    high = position + tolerance + 1
    strip = mask[start:end, low:high] if vertical else mask[low:high, start:end]
    if strip.size == 0:
        return Edge(position)
    ink = strip > 0
    along = ink.any(axis=1) if vertical else ink.any(axis=0)
    if along.size == 0 or float(np.count_nonzero(along)) / along.size < SEPARATOR_FILL:
        return Edge(position)
    profile = ink.sum(axis=0 if vertical else 1).astype(float)
    total = float(profile.sum())
    if total <= 0.0:
        return Edge(position)
    centre = low + float((np.arange(profile.size) * profile).sum() / total)
    thickness = int(round(total / max(1, end - start)))
    # Края штриха: колонки, где краска стоит хотя бы в пятой части строк полосы. Порог
    # нужен, чтобы буква, случайно заехавшая в окно поиска, не растянула края линейки:
    # она задевает несколько строк из сотен, линейка — почти все.
    solid = np.nonzero(profile >= RULE_EDGE_SHARE * (end - start))[0]
    if solid.size == 0:
        solid = np.nonzero(profile)[0]
    return Edge(int(round(centre)), thickness, True, low + int(solid[0]), low + int(solid[-1]))


def _edge_tables(lines: Lines, xs: list[int], ys: list[int], dpi: int) -> tuple[list[list[Edge]], list[list[Edge]]]:
    """``column_edges[строка][граница колонки]`` и ``row_edges[колонка][граница строки]``."""
    tolerance = max(2, mm_to_px(SEPARATOR_TOL_MM, dpi))
    column_edges = [
        [edge_at(lines.vertical_mask, xs[column], ys[row], ys[row + 1], True, tolerance) for column in range(len(xs))]
        for row in range(len(ys) - 1)
    ]
    row_edges = [
        [
            edge_at(lines.horizontal_mask, ys[row], xs[column], xs[column + 1], False, tolerance)
            for row in range(len(ys))
        ]
        for column in range(len(xs) - 1)
    ]
    return column_edges, row_edges


def _trim_ruleless_rows(ys: list[int], column_edges: list[list[Edge]]) -> list[int]:
    """Выбросить крайние строки, внутри которых нет ни одной вертикальной линейки.

    ЗАЧЕМ. В вырезку попадает то, что стоит вплотную к таблице: заголовок страницы сверху и
    сноска снизу («1 Расстояние от поставщика до потребителя — 2150 км»). Наружная рамка
    таблицы часто тянется и через них, поэтому они становятся строками сетки — а разделителей
    колонок в них нет, и объединение по отсутствующему разделителю склеивает такую строку с
    телом таблицы. На 1966/05, таблица 7 из-за этого всё тело сливалось в одну ячейку.

    Строка данных без единого разделителя колонок — не строка данных. Тексту внутри таблицы,
    занимающему всю ширину (заголовок в рамке), ячейка тоже не нужна: боковых надписей в нём
    не бывает.
    """

    def ruled(row: int) -> bool:
        return any(edge.present for edge in column_edges[row][1:-1])

    first, last = 0, len(column_edges) - 1
    while first < last and not ruled(first):
        first += 1
    while last > first and not ruled(last):
        last -= 1
    return ys[first : last + 2]


def _weighted(positions: list[int], weights: list[int]) -> int:
    total = sum(weights)
    if total <= 0:
        return int(round(float(np.mean(positions)))) if positions else 0
    return int(round(sum(p * w for p, w in zip(positions, weights)) / total))


def grid_from_lines(lines: Lines, shape: tuple[int, int], dpi: int, ink: "np.ndarray | None" = None) -> Grid:
    """Сетка ячеек: разделители, объединения по отсутствующим разделителям, шапка."""
    xs, ys, doubles = separators(lines, shape, dpi, ink)
    columns, rows = len(xs) - 1, len(ys) - 1
    if columns < 1 or rows < 1:
        return Grid(xs=xs, ys=ys, cells=[], source="ruling_grid")

    column_edges, row_edges = _edge_tables(lines, xs, ys, dpi)
    ys = _trim_ruleless_rows(ys, column_edges)
    if len(ys) - 1 != rows:
        rows = len(ys) - 1
        if rows < 1:
            return Grid(xs=xs, ys=ys, cells=[], source="ruling_grid")
        column_edges, row_edges = _edge_tables(lines, xs, ys, dpi)

    parent = list(range(rows * columns))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def merge(first: int, second: int) -> None:
        a, b = find(first), find(second)
        if a != b:
            parent[max(a, b)] = min(a, b)

    # Сперва — какие горизонтальные границы сквозные: через них объединять нельзя вовсе.
    through = {
        boundary
        for boundary in range(1, rows)
        if sum(1 for column in range(columns) if row_edges[column][boundary].present)
        >= max(GLOBAL_ROW_MIN_COLUMNS, GLOBAL_ROW_SHARE * columns)
    }

    for row in range(rows):
        for column in range(columns):
            if column + 1 < columns and not column_edges[row][column + 1].present:
                merge(row * columns + column, row * columns + column + 1)
            if row + 1 < rows and row + 1 not in through and not row_edges[column][row + 1].present:
                merge(row * columns + column, (row + 1) * columns + column)

    groups: dict[int, list[tuple[int, int]]] = {}
    for row in range(rows):
        for column in range(columns):
            groups.setdefault(find(row * columns + column), []).append((row, column))

    header_rows = _header_rows(ys, doubles)
    pad = mm_to_px(NO_RULE_PAD_MM, dpi)
    clearance = mm_to_px(RULE_CLEARANCE_MM, dpi)
    cells = [
        _build_cell(members, xs, ys, column_edges, row_edges, header_rows, pad, clearance)
        for members in groups.values()
    ]
    cells.sort(key=lambda cell: (cell.row, cell.col))
    return Grid(
        xs=xs,
        ys=ys,
        cells=cells,
        header_rows=header_rows,
        double_rule_ys=doubles,
        source="ruling_grid",
        column_cuts=_column_cuts(xs, column_edges),
    )


def _column_cuts(xs: list[int], column_edges: list[list[Edge]]) -> list[int]:
    """Левый край штриха каждой вертикальной линейки — там и только там можно резать.

    Берётся минимум по всем полосам: разрез идёт через всю высоту кадра, и он не должен
    задеть линейку ни в одной строке.
    """
    cuts: list[int] = []
    for column in range(len(xs)):
        edges = [row[column] for row in column_edges if row[column].present]
        cuts.append(min(edge.low for edge in edges) if edges else xs[column])
    return cuts


def _build_cell(
    members: list[tuple[int, int]],
    xs: list[int],
    ys: list[int],
    column_edges: list[list[Edge]],
    row_edges: list[list[Edge]],
    header_rows: int,
    pad: int,
    clearance: int,
) -> Cell:
    """Ячейка из клеток решётки: рамка по своим линейкам, внутренность по их толщине."""
    top = min(row for row, _ in members)
    bottom = max(row for row, _ in members)
    left = min(column for _, column in members)
    right = max(column for _, column in members)

    own_rows = range(top, bottom + 1)
    own_columns = range(left, right + 1)
    row_weights = [ys[row + 1] - ys[row] for row in own_rows]
    column_weights = [xs[column + 1] - xs[column] for column in own_columns]

    left_edges = [column_edges[row][left] for row in own_rows]
    right_edges = [column_edges[row][right + 1] for row in own_rows]
    top_edges = [row_edges[column][top] for column in own_columns]
    bottom_edges = [row_edges[column][bottom + 1] for column in own_columns]

    box = Box(
        _weighted([edge.position for edge in left_edges], row_weights),
        _weighted([edge.position for edge in top_edges], column_weights),
        _weighted([edge.position for edge in right_edges], row_weights),
        _weighted([edge.position for edge in bottom_edges], column_weights),
    )

    # Внутренность — по САМОЙ строгой полосе: рамка обязана не задеть линейку ни в одной
    # своей строке, иначе обрывок останется в вырезке ячейки. Числа считаются ДО сборки
    # рамки: у графы шириной в пару миллиметров отступы с двух сторон встречаются, и
    # ``Box`` на вывернутых координатах падает.
    left_inner = max(edge.inner_after() + (clearance if edge.present else pad) for edge in left_edges)
    right_inner = min(edge.inner_before() - (clearance if edge.present else pad) for edge in right_edges)
    top_inner = max(edge.inner_after() + (clearance if edge.present else pad) for edge in top_edges)
    bottom_inner = min(edge.inner_before() - (clearance if edge.present else pad) for edge in bottom_edges)
    if right_inner > left_inner and bottom_inner > top_inner:
        inner = Box(left_inner, top_inner, right_inner, bottom_inner)
    else:
        inner = _fallback_inner(box, pad)

    return Cell(
        row=top,
        col=left,
        box=box,
        row_span=bottom - top + 1,
        col_span=right - left + 1,
        is_header=top < header_rows,
        inner=inner,
    )


def _fallback_inner(box: Box, pad: int) -> Box:
    """Внутренность на глазок: отступ, ужатый до трети стороны у совсем узкой ячейки."""
    horizontal = min(pad, max(0, (box.width - 1) // 3))
    vertical = min(pad, max(0, (box.height - 1) // 3))
    return Box(box.x0 + horizontal, box.y0 + vertical, box.x1 - horizontal, box.y1 - vertical)


def _header_rows(ys: list[int], doubles: list[int]) -> int:
    """Сколько строк относится к шапке: всё, что выше двойной линейки.

    Если двойной линейки нет, шапкой считается первая строка: в этих журналах шапка есть
    почти всегда, а ошибка в одну строку тут не страшна — она только сужает поиск бокового
    текста, а ищут его всё равно по всем ячейкам.
    """
    if not doubles:
        return 1 if len(ys) > 2 else 0
    boundary = min(doubles)
    for index, value in enumerate(ys):
        if value >= boundary:
            return index
    return 1


def text_ink(gray: np.ndarray, lines: Lines) -> np.ndarray:
    """Краска без линеек: бинаризация минус маска линеек, раздутая на пару пикселей."""
    fat = cv2.dilate(lines.mask, np.ones((RULE_MARGIN_PX, RULE_MARGIN_PX), np.uint8))
    return cv2.bitwise_and(binarize(gray), cv2.bitwise_not(fat))
