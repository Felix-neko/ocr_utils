"""Группировка рядов в блоки для стенда: куски колонок как в боевом ``blocks_of`` и переключаемые способы делить их на блоки (прежний, доработанный, граф)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from ocr_utils.page_layout import mm_to_px
from ocr_utils.page_layout.text_blocks import blocks as legacy
from ocr_utils.page_layout.text_blocks.blocks import (
    BLOCK_GAP_PITCHES,
    COLUMN_PAD_MM,
    GLUED_ROWS_SHARE,
    LARGE_TYPE_RATIO,
    MIN_BLOCK_ROWS,
    SAME_STYLE_DISTANCE,
    SOFT_GAP_PITCHES,
    WIDE_BLOCK_RATIO,
    WIDE_BLOCK_ROWS,
    Row,
    pitch_of,
    row_gap,
    rows_of,
    split_blocks,
    style_distance,
)
from ocr_utils.page_layout.text_blocks.columns import inside_gutter
from ocr_utils.page_layout.text_blocks.lines import with_column
from research.block_envelopes.capture import BlockInput


class Grouping(str, Enum):
    """Как ряды куска колонки делятся на блоки."""

    LEGACY = "legacy"  # боевой ход: потоки + ``split_blocks`` (разрыв, кегль, жирность, перекрытие, черты)
    REACH = "reach"  # боевой ход + склейка ложных разрезов и разрез по «дотягиванию» строк
    GRAPH = "graph"  # с нуля: ряды — вершины, рёбра между соседями в столбик, компоненты — блоки


class SplitReason(str, Enum):
    """Правило боевого ``split_blocks``, разрезавшее блок на границе двух соседних рядов."""

    GAP = "gap"  # разрыв больше BLOCK_GAP_PITCHES шагов
    LARGE_GAP = "large_gap"  # крупный набор, разрыв больше предела в кеглях
    SOFT = "soft"  # разрыв больше SOFT_GAP_PITCHES и слабая смена набора
    WEAK_OVERLAP = "weak_overlap"  # ряды расходятся по горизонтали
    RULE = "rule"  # черта между рядами
    BARRIER = "barrier"  # линейка-барьер между рядами
    STYLE = "style"  # смена набора по окнам (``_style_break``)
    STREAM = "stream"  # ряды в разных потоках (бок о бок другого набора)


@dataclass(frozen=True)
class Piece:
    """Кусок колонки до деления на блоки: границы колонки по x и ряды сверху вниз."""

    column: int
    span: tuple[int, int]
    rows: list[Row]


@dataclass(frozen=True)
class Group:
    """Готовая группа рядов — будущий блок: номер колонки, номер внутри неё, границы колонки, ряды."""

    column: int
    index: int
    span: tuple[int, int]
    rows: list[Row]


@dataclass(frozen=True)
class PageContext:
    """Общие меры страницы, нужные делению кусков: медианная высота ряда (кегль корпуса)."""

    body_height: float


def pieces_of(inp: BlockInput) -> tuple[list[Piece], PageContext]:
    """Куски колонок страницы — ровно как в боевом ``blocks_of`` до деления на блоки.

    Повторяет ход ``blocks_of``: оси зон раскладываются по колонкам, в каждой собираются ряды,
    бесхозные оси — в кусок во всю ширину; затем возврат продолжения вводки, склейка кусков одной
    колонки из соседних зон, развод «проливов», выброс широких обрывков из слипшихся строк.

    Args:
        inp: Вход блоковой стадии полосы.

    Returns:
        Пара ``(куски, контекст страницы)``; куски в том же порядке, что колонки у ``blocks_of``.
    """
    dpi, width = inp.dpi, inp.width
    pad = mm_to_px(COLUMN_PAD_MM, dpi)
    raw: list[tuple[tuple[int, int], list[Row]]] = []
    placed: set[tuple[int, int, int]] = set()
    # Оси каждой зоны раскладываются по её колонкам, в колонке собираются ряды.
    for zone in inp.zones:
        own_zone = [axis for axis in inp.axes if zone.y0 <= axis.cy < zone.y1]
        homes = legacy._column_homes(
            [axis for axis in own_zone if not axis.cross], zone, inp.gutters, width, inp.leaders
        )
        for column, span in enumerate(zone.columns):
            own = [with_column(axis, column) for axis in own_zone if homes.get(legacy._axis_key(axis)) == column]
            if not own:
                continue
            rows = rows_of(
                own,
                inp.ink,
                span,
                dpi,
                gutters=inp.gutters,
                width=width,
                pad=pad,
                leaders=inp.leaders,
                barriers=inp.barriers,
            )
            placed.update(legacy._axis_key(axis) for row in rows for axis in row.axes)
            if rows:
                raw.append((span, rows))
    # Оси без колонки — в один кусок во всю ширину страницы.
    rest = [axis for axis in inp.axes if legacy._axis_key(axis) not in placed]
    if rest:
        rows = rows_of(
            rest,
            inp.ink,
            (0, width),
            dpi,
            gutters=inp.gutters,
            width=width,
            pad=pad,
            leaders=inp.leaders,
            fallback=True,
            barriers=inp.barriers,
        )
        if rows:
            raw.append(((0, width), rows))
    raw = legacy._absorb_continuation(raw, width, dpi, inp.barriers)
    merged = legacy._merge_pieces(raw, dpi)
    widths = [span[1] - span[0] for span, _ in merged]
    typical = float(np.median(widths)) if widths else 0.0
    all_heights = [row.height for _, rows in merged for row in rows]
    body_height = float(np.median(all_heights)) if all_heights else 0.0
    merged, spilled = legacy._spilled_rows(merged, dpi)
    if spilled:
        full = next((index for index, (span, _) in enumerate(merged) if span == (0, width)), None)
        if full is None:
            merged.append(((0, width), spilled))
        else:
            span, rows = merged[full]
            merged[full] = (span, sorted([*rows, *spilled], key=lambda row: row.y))
    pieces: list[Piece] = []
    for column, (span, rows) in enumerate(merged):
        if len(rows) < MIN_BLOCK_ROWS:
            continue
        # Широкий обрывок из слипшихся строк соседних колонок — выбрасывается, как в боевом коде.
        wide = typical and span[1] - span[0] > WIDE_BLOCK_RATIO * typical and len(rows) < WIDE_BLOCK_ROWS
        if wide and np.mean([len(row.axes) > 1 for row in rows]) >= GLUED_ROWS_SHARE:
            continue
        pieces.append(Piece(column, span, list(rows)))
    return pieces, PageContext(body_height=body_height)


def legacy_groups(piece: Piece, inp: BlockInput, context: PageContext) -> list[list[Row]]:
    """Деление куска боевым ходом: потоки бок о бок, затем ``split_blocks`` в каждом.

    Args:
        piece: Кусок колонки.
        inp: Вход блоковой стадии (черты, барьеры, разрешение).
        context: Меры страницы.

    Returns:
        Группы рядов сверху вниз.
    """
    rows = piece.rows
    column_pitch = pitch_of(rows)
    streams = legacy._streams(rows, column_pitch, inp.dpi)
    groups = [
        group
        for stream in streams
        for group in split_blocks(stream, column_pitch, inp.rules, piece.span, context.body_height, inp.barriers)
    ]
    groups.sort(key=lambda group: group[0].y)
    return groups


def split_reasons(
    previous: Row, row: Row, rows: list[Row], index: int, pitch: float, piece: Piece, inp: BlockInput, context
) -> list[SplitReason]:
    """Какие правила боевого ``split_blocks`` режут границу между ``previous`` и ``row``.

    Повторяет условия ``split_blocks`` по отдельности, чтобы в протоколе было видно, какое из них
    дало ложный разрез. Смена набора по окнам (``STYLE``) проверяется на всей колонке, а не внутри
    уже разрезанной группы, как в боевом коде, — это приближение, достаточное для протокола.

    Args:
        previous: Верхний ряд.
        row: Нижний ряд.
        rows: Все ряды колонки (для окон смены набора и опорной длины).
        index: Номер ``previous`` в ``rows``.
        pitch: Шаг колонки.
        piece: Кусок колонки (границы — для черт).
        inp: Вход блоковой стадии.
        context: Меры страницы.

    Returns:
        Список сработавших правил; пусто — боевой код здесь не резал бы по этим правилам.
    """
    reasons: list[SplitReason] = []
    gap = row_gap(previous, row)
    reference = max(item.x1 - item.x0 for item in rows)
    smaller = min(previous.height, row.height)
    large_type = context.body_height > 0 and smaller > LARGE_TYPE_RATIO * context.body_height
    one_headline = large_type and style_distance(previous, row) < SAME_STYLE_DISTANCE
    if gap > BLOCK_GAP_PITCHES * pitch and not one_headline:
        reasons.append(SplitReason.GAP)
    if large_type and gap > legacy._large_gap_limit(previous, row):
        reasons.append(SplitReason.LARGE_GAP)
    if gap > SOFT_GAP_PITCHES * pitch and legacy._soft_style_break(previous, row):
        reasons.append(SplitReason.SOFT)
    if legacy._weak_overlap(previous, row, reference):
        reasons.append(SplitReason.WEAK_OVERLAP)
    if legacy._rule_between(previous, row, inp.rules, piece.span):
        reasons.append(SplitReason.RULE)
    if legacy._barrier_between_rows(previous, row, inp.barriers, piece.span):
        reasons.append(SplitReason.BARRIER)
    if legacy._style_break(rows, index):
        reasons.append(SplitReason.STYLE)
    return reasons


# --- Доработка боевого хода (REACH) и граф рядов (GRAPH) ---------------------------------------------

# Штрих текста, а не разделитель: линейка целиком в полосе строки — от стольких высот над осью (верх
# крупных букв) до стольких под ней (подчёркивание).
INSIDE_HEIGHTS = 0.6
UNDERLINE_HEIGHTS = 0.75
# «Дотягивание»: ряд ищет перекрытого по x соседа выше не дальше стольких высот строки.
REACH_HEIGHTS = 2.5
# Склейка ложного разреза: разрыв между блоками не больше стольких шагов — обычный межстрочный шаг с
# запасом; подпись автора под письмом отбита на 1.7 шага (1966/02 IMG_0099_2R) и не склеивается...
REJOIN_GAP_PITCHES = 1.35
# ...края совпадают (левый, правый или центр) с таким допуском...
REJOIN_EDGE_MM = 3.0
# ...блоки перекрыты по x на такую долю более узкого...
REJOIN_OVERLAP = 0.6
# ...и набор на стыке один (балл :func:`style_distance` меньше).
REJOIN_STYLE = SAME_STYLE_DISTANCE
# Куски одной строки (заголовок, разорванный при сборке рядов): перекрыты по высоте на такую долю
# меньшей высоты и разнесены по x не дальше стольких высот строки.
SAME_LINE_OVERLAP = 0.5
SAME_LINE_GAP_HEIGHTS = 1.5
# Граф: ребро вниз — к ближайшему ряду, перекрытому по x на эту долю более короткого...
GRAPH_OVERLAP = 0.35
# ...не дальше стольких шагов (корпус) или высот ряда (крупный набор)...
GRAPH_GAP_PITCHES = 2.0
GRAPH_GAP_HEIGHTS = 2.4
# ...и одного набора; при зазоре больше SOFT_GAP_PITCHES шага набор должен совпасть строже.
GRAPH_STYLE = 0.75
GRAPH_STYLE_STRICT = 0.35


def underline_free(barriers, axes: list) -> object:
    """Линейки-барьеры без штрихов текста: подчёркиваний и черт самих букв крупного набора.

    Линейка — не разделитель, если ВСЕ её точки лежат в полосе какой-нибудь строки: от
    ``INSIDE_HEIGHTS`` высоты над осью до ``UNDERLINE_HEIGHTS`` высоты под ней. Так выглядят:

    * подчёркивание — черта под словом «организаций.» (1966/01 IMG_0036_2R, x 56–186, на 8 px ниже
      оси) резала абзац пополам;
    * перекладины и ножки букв заголовка — верх букв «МЕЛКООПТОВОЙ ТОРГОВЛИ» (1968/04 IMG_0043_1L) и
      ножка «В» в «В ДЕЛО» (1971/10 IMG_0010_1L) отрывали вторую строку заголовка от первой.

    Разделитель (сноски, заметки) стоит в промежутке между строками, вне их полос.

    Args:
        barriers: Линейки-барьеры (``barriers.BarrierLines``) или ``None``.
        axes: Оси строк страницы.

    Returns:
        Новый набор линеек (или тот же объект, если убирать нечего).
    """
    if barriers is None or barriers.empty:
        return barriers
    kept = [line for line in barriers.lines if not any(_inside_band(line, axis) for axis in axes)]
    if len(kept) == len(barriers.lines):
        return barriers
    return type(barriers).of(kept)


def _inside_band(line: np.ndarray, axis) -> bool:
    """Лежит ли ломаная линейки целиком в полосе строки (по x — в пределах оси с допуском в полвысоты)."""
    pad = 0.5 * axis.height
    xs, ys = line[:, 0], line[:, 1]
    if xs.min() < axis.x0 - pad or xs.max() > axis.x1 + pad:
        return False
    dy = ys - np.interp(xs, axis.points[:, 0], axis.points[:, 1])
    return bool((dy >= -INSIDE_HEIGHTS * axis.height).all() and (dy <= UNDERLINE_HEIGHTS * axis.height).all())


def merge_same_line(rows: list[Row], dpi: float, barriers=None, gutters: list | None = None) -> list[Row]:
    """Слить куски одной строки в один ряд: перекрыты по высоте, близко по x, одного набора.

    Крупный заголовок при сборке рядов рвётся на куски по буквам (1966/03 IMG_0120_2R, 1966/02
    IMG_0099_2R — «ПОТРЕБЛЕНИЯ» двумя рядами на одной высоте), и каждый кусок уходил в свой блок.

    Куски через линейку-барьер или межколонник не сливаются: «СОВЕТА МИНИСТРОВ» в левой графе
    обложки и строка оглавления справа от вертикальной линейки стоят на одной высоте в двух десятках
    пикселей друг от друга (1972/08 IMG_0054_2R).

    Args:
        rows: Ряды (любой порядок).
        dpi: Разрешение рабочей копии.
        barriers: Линейки-барьеры (``barriers.BarrierLines``) или ``None``.
        gutters: Локальные межколонники (``columns.Gutter``) или ``None``.

    Returns:
        Ряды сверху вниз; слитые куски — одним рядом с общими краями и осями.
    """
    out = sorted(rows, key=lambda row: row.x0)
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                overlap = min(a.y + a.height / 2, b.y + b.height / 2) - max(a.y - a.height / 2, b.y - b.height / 2)
                if overlap < SAME_LINE_OVERLAP * min(a.height, b.height):
                    continue
                gap = max(a.x0, b.x0) - min(a.x1, b.x1)
                if gap > SAME_LINE_GAP_HEIGHTS * max(a.height, b.height):
                    continue
                if style_distance(a, b) >= SAME_STYLE_DISTANCE:
                    continue
                if _divided(a, b, barriers, gutters):
                    continue
                out[i] = _joined(a, b)
                del out[j]
                changed = True
                break
            if changed:
                break
    return sorted(out, key=lambda row: row.y)


def _divided(a: Row, b: Row, barriers, gutters: list | None) -> bool:
    """Разделяет ли два куска на одной высоте линейка-барьер или межколонник (по отрезку между их краями)."""
    left, right = (a, b) if a.x0 <= b.x0 else (b, a)
    x0, x1 = left.x1, right.x0
    y = (a.y + b.y) / 2.0
    if barriers is not None and not barriers.empty and barriers.crosses((x0 - 1.0, y), (x1 + 1.0, y)):
        return True
    if gutters and x1 > x0:
        middle = (x0 + x1) / 2.0
        if inside_gutter(gutters, middle - 1.0, middle + 1.0, y):
            return True
    return False


def _joined(a: Row, b: Row) -> Row:
    """Один ряд из двух кусков строки: края — крайние, оси и профили — вместе, высота — большая."""
    edges = {}
    for name in ("top_edge", "bottom_edge", "body"):
        parts = [item for item in (getattr(a, name), getattr(b, name)) if item is not None and len(item)]
        edges[name] = np.vstack(parts)[np.argsort(np.vstack(parts)[:, 0])] if parts else None
    return replace(
        a,
        y=(a.y * (a.x1 - a.x0) + b.y * (b.x1 - b.x0)) / max(1.0, (a.x1 - a.x0) + (b.x1 - b.x0)),
        height=max(a.height, b.height),
        x0=min(a.x0, b.x0),
        x1=max(a.x1, b.x1),
        axes=tuple(sorted([*a.axes, *b.axes], key=lambda axis: axis.x0)),
        tail=None,
        cut=None,
        **edges,
    )


def reach_components(rows: list[Row]) -> list[list[Row]]:
    """Разбить группу на части, связанные «дотягиванием»: ряд связан с ближайшим выше, с которым перекрыт по x.

    Правило пользователя: слияние по горизонтали допустимо, только если строки одной части хоть в
    одной строке дотягиваются до другой. Хвост абзаца «ленности.» и «* * *» правее него (1966/02
    IMG_0073_1L) по x не перекрываются — и в один блок не идут.

    Args:
        rows: Ряды группы сверху вниз.

    Returns:
        Части сверху вниз (одна — если группа связна).
    """
    if len(rows) < 2:
        return [rows]
    parent = list(range(len(rows)))
    for index, row in enumerate(rows):
        for above in range(index - 1, -1, -1):
            other = rows[above]
            # Дотягиваться можно только до соседей по вертикали: полная строка через одну выше —
            # не сосед «* * *» под хвостом абзаца.
            if row.y - other.y > REACH_HEIGHTS * max(row.height, other.height):
                break
            if min(other.x1, row.x1) - max(other.x0, row.x0) > 0:
                parent[legacy._root(parent, index)] = legacy._root(parent, above)
                break
    parts: dict[int, list[Row]] = {}
    for index, row in enumerate(rows):
        parts.setdefault(legacy._root(parent, index), []).append(row)
    return sorted(parts.values(), key=lambda part: part[0].y)


def _extent(rows: list[Row]) -> tuple[float, float]:
    """Крайние края рядов по x."""
    return min(row.x0 for row in rows), max(row.x1 for row in rows)


def _rejoinable(upper: list[Row], lower: list[Row], inp: BlockInput, barriers, body_height: float) -> bool:
    """Можно ли склеить два блока в один: один под другим вплотную, края сходятся, набор один, черты между ними нет.

    Args:
        upper: Ряды верхнего блока.
        lower: Ряды нижнего блока.
        inp: Вход блоковой стадии (черты, разрешение).
        barriers: Линейки-барьеры (уже без подчёркиваний).
        body_height: Медианная высота ряда страницы (кегль корпуса): крупный набор меряется в своих высотах.

    Returns:
        ``True`` — это один блок, разрезанный ложно.
    """
    last, first = upper[-1], lower[0]
    pitch = max(pitch_of(upper) if len(upper) > 1 else 0.0, pitch_of(lower) if len(lower) > 1 else 0.0)
    pitch = pitch or 1.6 * max(last.height, first.height)
    gap = row_gap(last, first)
    # Крупный набор (заголовок) меряется в своих высотах: у него разрежённый интервал.
    large = body_height > 0 and min(last.height, first.height) > LARGE_TYPE_RATIO * body_height
    limit = GAP_HEIGHTS_REJOIN * min(last.height, first.height) if large else REJOIN_GAP_PITCHES * pitch
    if not 0 < gap <= limit:
        return False
    if style_distance(last, first) >= REJOIN_STYLE:
        return False
    (ux0, ux1), (lx0, lx1) = _extent(upper), _extent(lower)
    overlap = min(ux1, lx1) - max(ux0, lx0)
    if overlap < REJOIN_OVERLAP * min(ux1 - ux0, lx1 - lx0):
        return False
    tol = mm_to_px(REJOIN_EDGE_MM, inp.dpi)
    same_left = abs(ux0 - lx0) <= tol
    same_right = abs(ux1 - lx1) <= tol
    same_centre = abs((ux0 + ux1) - (lx0 + lx1)) / 2.0 <= tol
    # Конец абзаца (короткий последний ряд верхнего) и абзацный отступ (первый ряд нижнего) сдвигают
    # только одну сторону; строки заголовка по центру — центр.
    if not (same_left or same_right or same_centre):
        return False
    span = (int(min(ux0, lx0)), int(max(ux1, lx1)))
    if legacy._rule_between(last, first, inp.rules, span):
        return False
    if legacy._barrier_between_rows(last, first, barriers, span):
        return False
    return True


# Предел разрыва для склейки крупного набора — в высотах строки (как ``GAP_HEIGHTS`` боевого кода).
GAP_HEIGHTS_REJOIN = 2.4


def rejoin(groups: list[list[Row]], inp: BlockInput, barriers, body_height: float) -> list[list[Row]]:
    """Склеивать соседние по вертикали группы страницы, пока находятся ложные разрезы (:func:`_rejoinable`).

    Работает поверх кусков разных колонок и зон: ложный разрез 1966/02 IMG_0098_2R — это восемь
    строк правой колонки, ушедших в кусок во всю ширину.

    Args:
        groups: Группы рядов сверху вниз каждая.
        inp: Вход блоковой стадии.
        barriers: Линейки-барьеры.
        body_height: Медианная высота ряда страницы.

    Returns:
        Группы после склеек.
    """
    out = [list(group) for group in groups if group]
    changed = True
    while changed:
        changed = False
        for i, upper in enumerate(out):
            # Ближайшая снизу группа, перекрытая по x, — кандидат на склейку.
            below = [
                (lower[0].y - upper[-1].y, j)
                for j, lower in enumerate(out)
                if j != i
                and lower[0].y > upper[-1].y
                and min(_extent(upper)[1], _extent(lower)[1]) > max(_extent(upper)[0], _extent(lower)[0])
            ]
            if not below:
                continue
            _, j = min(below)
            if _rejoinable(upper, out[j], inp, barriers, body_height):
                out[i] = upper + out[j]
                del out[j]
                changed = True
                break
    return out


def reach_groups(inp: BlockInput) -> list[Group]:
    """Доработанный боевой ход: без подчёркиваний-барьеров, куски строк слиты, части без «дотягивания» врозь, ложные разрезы склеены.

    Args:
        inp: Вход блоковой стадии.

    Returns:
        Группы-блоки.
    """
    barriers = underline_free(inp.barriers, inp.axes)
    local = replace_input(inp, barriers)
    pieces, context = pieces_of(local)
    groups: list[tuple[Piece, list[Row]]] = []
    for piece in pieces:
        for group in legacy_groups(piece, local, context):
            for part in reach_components(merge_same_line(group, inp.dpi, barriers, inp.gutters)):
                groups.append((piece, part))
    joined = rejoin([rows for _, rows in groups], local, barriers, context.body_height)
    return _numbered(joined, inp)


def replace_input(inp: BlockInput, barriers) -> BlockInput:
    """Тот же вход с другими линейками-барьерами (без копирования больших массивов)."""
    return BlockInput(**{**inp.__dict__, "barriers": barriers})


def _numbered(groups: list[list[Row]], inp: BlockInput) -> list[Group]:
    """Группы по порядку чтения: слева направо по колонкам (левому краю), сверху вниз."""
    ordered = sorted(groups, key=lambda rows: (rows[0].y, _extent(rows)[0]))
    return [
        Group(column=0, index=index, span=(int(_extent(rows)[0]), int(_extent(rows)[1])), rows=rows)
        for index, rows in enumerate(ordered)
    ]


def graph_groups(inp: BlockInput) -> list[Group]:
    """Группировка с нуля: ряды всей страницы — вершины, рёбра между соседями в столбик, компоненты — блоки.

    1. Ряды — как у боевого хода (колонки зон, запасной кусок), куски одной строки слиты.
    2. Ребро вниз — к БЛИЖАЙШЕМУ ряду ниже, перекрытому по x (``GRAPH_OVERLAP`` более короткого), если
       зазор в пределах (``GRAPH_GAP_PITCHES`` шага корпуса или ``GRAPH_GAP_HEIGHTS`` высот крупного
       набора), набор близок (``GRAPH_STYLE``, при большом зазоре — ``GRAPH_STYLE_STRICT``), между ними
       нет черты и барьера. Если под рядом стоят два ряда бок о бок через межколонник (ряд — общий
       заголовок над колонками), ребра нет: иначе колонки срослись бы через него.
    3. Компоненты связности — блоки; внутри компоненты — боевое деление по смене набора
       (``_split_by_style``: подпись автора жирнее корпуса).

    Горизонтальных рёбер нет: соседние колонки не сливаются никогда.

    Args:
        inp: Вход блоковой стадии.

    Returns:
        Группы-блоки.
    """
    barriers = underline_free(inp.barriers, inp.axes)
    local = replace_input(inp, barriers)
    pieces, context = pieces_of(local)
    rows = merge_same_line([row for piece in pieces for row in piece.rows], inp.dpi, barriers, inp.gutters)
    # Шаг корпуса страницы — медиана шагов кусков.
    pitches = [pitch_of(piece.rows) for piece in pieces if len(piece.rows) > 2]
    body_pitch = float(np.median(pitches)) if pitches else 1.6 * context.body_height
    parent = list(range(len(rows)))
    for index, row in enumerate(rows):
        candidates = []
        for other_index in range(index + 1, len(rows)):
            other = rows[other_index]
            if other.y - other.height / 2 <= row.y:
                continue  # не ниже
            overlap = min(row.x1, other.x1) - max(row.x0, other.x0)
            if overlap < GRAPH_OVERLAP * min(row.x1 - row.x0, other.x1 - other.x0):
                continue
            candidates.append((row_gap(row, other), other_index))
        if not candidates:
            continue
        gap, nearest = min(candidates)
        other = rows[nearest]
        # Бок о бок под рядом: ещё один кандидат почти на той же высоте, но через пустоту по x.
        twins = [
            item
            for _, item in candidates
            if item != nearest
            and abs(rows[item].y - other.y) < 0.5 * max(rows[item].height, other.height)
            and max(rows[item].x0, other.x0) - min(rows[item].x1, other.x1) > 0
        ]
        if twins:
            continue
        large = context.body_height > 0 and min(row.height, other.height) > LARGE_TYPE_RATIO * context.body_height
        limit = GRAPH_GAP_HEIGHTS * min(row.height, other.height) if large else GRAPH_GAP_PITCHES * body_pitch
        if gap > limit:
            continue
        distance = style_distance(row, other)
        if distance >= GRAPH_STYLE or (gap > SOFT_GAP_PITCHES * body_pitch and distance >= GRAPH_STYLE_STRICT):
            continue
        span = (int(min(row.x0, other.x0)), int(max(row.x1, other.x1)))
        if legacy._rule_between(row, other, inp.rules, span) or legacy._barrier_between_rows(
            row, other, barriers, span
        ):
            continue
        parent[legacy._root(parent, nearest)] = legacy._root(parent, index)
    components: dict[int, list[Row]] = {}
    for index, row in enumerate(rows):
        components.setdefault(legacy._root(parent, index), []).append(row)
    groups = []
    for component in components.values():
        component.sort(key=lambda row: row.y)
        groups.extend(legacy._split_by_style(component))
    return _numbered(groups, inp)


def legacy_all(inp: BlockInput) -> list[Group]:
    """Боевой ход целиком: группы всех кусков страницы, пронумерованные как у ``blocks_of``."""
    pieces, context = pieces_of(inp)
    out = []
    for piece in pieces:
        for index, group in enumerate(legacy_groups(piece, inp, context)):
            if len(group) >= MIN_BLOCK_ROWS:
                out.append(Group(piece.column, index, piece.span, group))
    return out


def groups_of(kind: Grouping, inp: BlockInput) -> list[Group]:
    """Группы-блоки полосы выбранным способом.

    Args:
        kind: Способ группировки.
        inp: Вход блоковой стадии.

    Returns:
        Группы.
    """
    if kind is Grouping.LEGACY:
        return legacy_all(inp)
    if kind is Grouping.REACH:
        return reach_groups(inp)
    return graph_groups(inp)


__all__ = ["Group", "Grouping", "PageContext", "Piece", "SplitReason", "legacy_groups", "pieces_of", "split_reasons"]
