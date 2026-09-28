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
from ocr_utils.page_layout.text_blocks.regroup import (  # noqa: F401 — стенд и тесты берут их отсюда
    SAME_LINE_OVERLAP,
    _divided,
    _extent,
    _joined,
    merge_overlapping_pieces,
    merge_same_line,
    reach_components,
    rejoin,
    resolve_shared_rows,
    underline_free,
)
from research.block_envelopes.axes_fix import fixed_input
from research.block_envelopes.capture import BlockInput


class Grouping(str, Enum):
    """Как ряды куска колонки делятся на блоки."""

    LEGACY = "legacy"  # боевой ход: потоки + ``split_blocks`` (разрыв, кегль, жирность, перекрытие, черты)
    REACH = "reach"  # боевой ход + склейка ложных разрезов и разрез по «дотягиванию» строк
    GRAPH = "graph"  # с нуля: ряды — вершины, рёбра между соседями в столбик, компоненты — блоки
    REACH_ROWS = "reach_rows"  # REACH + куски одной строки слиты ДО деления + страховка от общих рядов
    REACH_AXES = "reach_axes"  # REACH_ROWS + оси-выбросы крупного набора заменены (``axes_fix``)


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

# Граф: ребро вниз — к ближайшему ряду, перекрытому по x на эту долю более короткого...
GRAPH_OVERLAP = 0.35
# ...не дальше стольких шагов (корпус) или высот ряда (крупный набор)...
GRAPH_GAP_PITCHES = 2.0
GRAPH_GAP_HEIGHTS = 2.4
# ...и одного набора; при зазоре больше SOFT_GAP_PITCHES шага набор должен совпасть строже.
GRAPH_STYLE = 0.75
GRAPH_STYLE_STRICT = 0.35


def reach_groups(inp: BlockInput, rows_fix: bool = False) -> list[Group]:
    """Доработанный боевой ход: без подчёркиваний-барьеров, куски строк слиты, части без «дотягивания» врозь, ложные разрезы склеены.

    Args:
        inp: Вход блоковой стадии.
        rows_fix: Сливать ли налезающие куски одной строки ДО деления куска колонки на блоки
            (:func:`merge_overlapping_pieces`) и разводить ли после группировки ряды одной высоты,
            попавшие в разные блоки (:func:`resolve_shared_rows`).

    Returns:
        Группы-блоки.
    """
    barriers = underline_free(inp.barriers, inp.axes)
    local = replace_input(inp, barriers)
    pieces, context = pieces_of(local)
    if rows_fix:
        pieces = [
            Piece(piece.column, piece.span, merge_overlapping_pieces(piece.rows, barriers, inp.gutters))
            for piece in pieces
        ]
    groups: list[tuple[Piece, list[Row]]] = []
    for piece in pieces:
        for group in legacy_groups(piece, local, context):
            for part in reach_components(merge_same_line(group, inp.dpi, barriers, inp.gutters)):
                groups.append((piece, part))
    joined = rejoin([rows for _, rows in groups], local.rules, local.dpi, barriers, context.body_height)
    if rows_fix:
        joined = resolve_shared_rows(joined)
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
    if kind is Grouping.REACH_ROWS:
        return reach_groups(inp, rows_fix=True)
    if kind is Grouping.REACH_AXES:
        # Оси-выбросы заменяются на входе: и ряды, и полосы строятся уже по исправленным осям.
        return reach_groups(fixed_input(inp)[0], rows_fix=True)
    return graph_groups(inp)


__all__ = ["Group", "Grouping", "PageContext", "Piece", "SplitReason", "legacy_groups", "pieces_of", "split_reasons"]
