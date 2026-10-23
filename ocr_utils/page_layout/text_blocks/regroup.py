"""Доработки группировки строк в блоки (способ ``BlocksMode.SMOOTH``): штрихи текста — не барьеры, куски одной строки — один ряд, слияние только при «дотягивании» строк, склейка ложных разрезов, развод общих рядов; сборка блоков с гладкой границей."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ocr_utils.page_layout import mm_to_px
from ocr_utils.page_layout.text_blocks import WORK_DPI
from ocr_utils.page_layout.text_blocks import blocks as legacy
from ocr_utils.page_layout.text_blocks.axes_fix import body_reference
from ocr_utils.page_layout.text_blocks.blocks import (
    COARSE_FACTOR,
    DILATE_GLYPHS,
    LARGE_TYPE_RATIO,
    MIN_BLOCK_ROWS,
    ROW_GLYPH_GAP_MM,
    ROW_GLYPH_RATIO,
    ROW_GLYPH_STRONG_GAP_MM,
    ROW_GLYPH_STRONG_RATIO,
    SAME_BASELINE_MAX_GAP_KEGL,
    SAME_STYLE_DISTANCE,
    SMOOTH_PITCHES,
    Row,
    TextBlock,
    column_pieces,
    envelope_of,
    pitch_of,
    row_gap,
    split_blocks,
    style_distance,
)
from ocr_utils.page_layout.text_blocks.columns import inside_gutter
from ocr_utils.page_layout.text_blocks.smooth_envelope import _glyph, envelope_smooth
from ocr_utils.page_layout.text_blocks.typeset import axes_glyphs, axes_kegl, barrier_between, same_baseline

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
# Разный набор кусков на одной высоте (:func:`_different_type`) решается по кеглю, только если у обоих кусков не
# меньше стольких глифов; основной кегль — до стольких мм (выше — заголовок, защита просвета по меньшему кеглю).
TYPE_MIN_GLYPHS = 5
# Широкий пробел строки по формату: куски на одной высоте сливаются и при просвете до стольких высот строки, если
# над и под просветом в пределах стольких высот (через одну короткую строку — конец записи) идут строки через
# весь просвет (:func:`_bridged`).
SAME_LINE_WIDE_GAP_HEIGHTS = 4.0
BRIDGE_ROWS_HEIGHTS = 3.5
BODY_KEGL_MAX_MM = 2.5


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
                limit = SAME_LINE_GAP_HEIGHTS * max(a.height, b.height)
                bridged = _bridged(a, b, out)
                if gap > limit and not (gap <= SAME_LINE_WIDE_GAP_HEIGHTS * max(a.height, b.height) and bridged):
                    continue
                if style_distance(a, b) >= SAME_STYLE_DISTANCE:
                    continue
                # Просвет, перекрытый строками сверху и снизу, — пробел внутри строки: ни межколонник (столбец широких
                # пробелов библиографии по формату принимался за межколонник — ХиП 1991/01 IMG_1016), ни разный кегль
                # (жирная фамилия автора записи в начале строки — 01/IMG_1030) её не делят; линейка-барьер — делит.
                if _divided(a, b, barriers, None if bridged else gutters, check_type=not bridged):
                    continue
                out[i] = _joined(a, b)
                del out[j]
                changed = True
                break
            if changed:
                break
    return sorted(out, key=lambda row: row.y)


def _bridged(a: Row, b: Row, rows: list[Row]) -> bool:
    """Перекрыт ли просвет между кусками строки соседними строками сверху и снизу — просвет не канал колонок.

    Строка по формату с широкими пробелами (библиография: «формы, методы, перспективы.— Л.: Лениздат, 1990.— 144 с.»,
    ХиП 1991/01 IMG_1016 — просвет 6.7 мм, три высоты строки) рвётся на куски шире ``SAME_LINE_GAP_HEIGHTS``. Если
    над просветом и под ним (не дальше ``BRIDGE_ROWS_HEIGHTS`` высот) идут строки через весь просвет, это не
    межколонник и не графа таблицы, а пробел внутри строки.

    Args:
        a, b: Куски на одной высоте.
        rows: Все ряды.

    Returns:
        ``True`` — просвет перекрыт строками и сверху, и снизу.
    """
    # Куски разной высоты строки — не одна строка с широкими пробелами: номер выпуска «4-й» (ось 23 px) рядом с
    # «СНАБЖЕНИЕ» шапки журнала (ось 60 px) считался перекрытым строками «Год издания» и «ОРГАН…» и сливался с ним
    # в ряд без сравнения набора (1969/06 IMG_0105_2R).
    heights = sorted((max(axis.height for axis in a.axes), max(axis.height for axis in b.axes)))
    if heights[1] >= ROW_GLYPH_STRONG_RATIO * heights[0]:
        return False
    left, right = (a, b) if a.x0 <= b.x0 else (b, a)
    gx0, gx1 = left.x1, right.x0
    y, reach = (a.y + b.y) / 2.0, BRIDGE_ROWS_HEIGHTS * max(a.height, b.height)
    above = below = False
    for row in rows:
        if row is a or row is b or row.x0 > gx0 or row.x1 < gx1:
            continue
        if y - reach <= row.y < y - 0.5 * row.height:
            above = True
        elif y + 0.5 * row.height < row.y <= y + reach:
            below = True
    return above and below


def _divided(a: Row, b: Row, barriers, gutters: list | None, check_type: bool = True) -> bool:
    """Разделяет ли два куска на одной высоте линейка-барьер, межколонник или разный кегль через пустоту.

    Разный кегль (:func:`_different_type`) — подпись автора рядом с заголовком: ряды налезают по краске, и
    ``merge_overlapping_pieces`` сливал их без сравнения набора (1966/01 IMG_0046_2R, 1971/02 IMG_0071_2R).
    Линейка проверяется и по глифам нормальной высоты (:func:`typeset.barrier_between`): краска линейки
    оглавления входит в строку, ось кончается на самой линейке, край ряда по краске уходит за неё в соседнюю
    графу (1973/01 IMG_0004_2R: оси «1973» до 358, край по краске 392, линейка 354, оглавление с 376).

    Args:
        a, b: Ряды.
        barriers: Линейки-барьеры (``barriers.BarrierLines``) или ``None``.
        gutters: Межколонники или ``None``.
        check_type: Сравнивать набор кусков (:func:`_different_type`).

    Returns:
        ``True`` — ряды не сливать.
    """
    left, right = (a, b) if a.x0 <= b.x0 else (b, a)
    x0, x1 = left.x1, right.x0
    y = (a.y + b.y) / 2.0
    if barriers is not None and not barriers.empty and barriers.crosses((x0 - 1.0, y), (x1 + 1.0, y)):
        return True
    if gutters and x1 > x0:
        middle = (x0 + x1) / 2.0
        if inside_gutter(gutters, middle - 1.0, middle + 1.0, y):
            return True
    if barrier_between(left.axes, right.axes, y, y, barriers):
        return True
    return check_type and _different_type(left, right)


def _different_type(left: Row, right: Row) -> bool:
    """Разный ли набор у двух рядов бок о бок: кегль по боксам глифов, просвет — между ОСЯМИ рядов.

    Пороги — как у ``blocks._split_by_glyph``: просвет шире ``ROW_GLYPH_GAP_MM`` при разнице кегля от
    ``ROW_GLYPH_RATIO`` или шире ``ROW_GLYPH_STRONG_GAP_MM`` при разнице от ``ROW_GLYPH_STRONG_RATIO``; защита
    «одна базовая линия» — до ``SAME_BASELINE_MAX_GAP_KEGL`` меньшего кегля. Кегль — по боксам глифов
    (``typeset.axes_kegl``): у шапки журнала буквы выше 7 мм, и мера ряда ``glyph_h`` там молчит. Просвет —
    по осям: края по краске у подписи рядом с заголовком налезают.

    Args:
        left: Левый ряд.
        right: Правый ряд.

    Returns:
        ``True`` — ряды разного набора.
    """
    gap = min(axis.x0 for axis in right.axes) - max(axis.x1 for axis in left.axes)
    left_glyphs, right_glyphs = axes_glyphs(left.axes), axes_glyphs(right.axes)
    # Кегль по горстке глифов (тире, точка, обломок слова) не мера набора: обломок «—» в 3 глифа с кеглем 0.68 мм
    # отделялся от своей строки библиографии (ХиП 1991/01 IMG_1016), строка оставалась тремя кусками, и сторона блока
    # по ним не строилась.
    if min(_glyph_count(left_glyphs), _glyph_count(right_glyphs)) < TYPE_MIN_GLYPHS:
        # Высота осей сравнивается и у горстки глифов: номер выпуска «4-й» (3 глифа, ось 23 px) рядом с «СНАБЖЕНИЕ»
        # шапки журнала (ось 60 px) иначе сливался с ним в ряд (1969/06 IMG_0105_2R). У обломка строки библиографии
        # высота оси — высота своей строки, разницы нет.
        heights = sorted((max(axis.height for axis in left.axes), max(axis.height for axis in right.axes)))
        return heights[1] >= ROW_GLYPH_STRONG_RATIO * heights[0] and gap > mm_to_px(ROW_GLYPH_STRONG_GAP_MM, WORK_DPI)
    low, high = sorted((axes_kegl(left.axes), axes_kegl(right.axes)))
    if low <= 0:
        return False
    # Защита «одна базовая линия»: в наборе основного кегля просвет меряется кеглем большего куска. У кусков
    # одной строки кегль меньшего может быть занижен обломками глифов (бледное начало абзаца после выравнивания
    # освещения: глифы 0.2–1.0 мм при кегле строки 1.9 мм, ХиП 1991/01 IMG_0937), и обычный пробел между словами
    # (2 мм) превышал защиту по меньшему кеглю. Подпись рядом с заголовком (кегль заголовка крупнее основного) —
    # по-прежнему по меньшему кеглю.
    guard = high if high <= mm_to_px(BODY_KEGL_MAX_MM, WORK_DPI) else low
    if gap <= SAME_BASELINE_MAX_GAP_KEGL * guard and same_baseline(left_glyphs, right_glyphs, low):
        return False
    if high >= ROW_GLYPH_RATIO * low and gap > mm_to_px(ROW_GLYPH_GAP_MM, WORK_DPI):
        return True
    return high >= ROW_GLYPH_STRONG_RATIO * low and gap > mm_to_px(ROW_GLYPH_STRONG_GAP_MM, WORK_DPI)


def _glyph_count(glyphs) -> int:
    """Число глифов группы (``None`` — ноль)."""
    return 0 if glyphs is None else len(glyphs)


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
            # Половинки ОДНОЙ строки (текст графы и число за отточием — 1970/06 IMG_0119_2R, «53,2 тыс.»)
            # связаны всегда: бок о бок другого набора боевой ход уже развёл по потокам, а внутри
            # группы соседи на одной высоте — одна строка.
            same_line = abs(row.y - other.y) < SAME_LINE_OVERLAP * min(row.height, other.height)
            if same_line or min(other.x1, row.x1) - max(other.x0, row.x0) > 0:
                parent[legacy._root(parent, index)] = legacy._root(parent, above)
                break
    parts: dict[int, list[Row]] = {}
    for index, row in enumerate(rows):
        parts.setdefault(legacy._root(parent, index), []).append(row)
    return sorted(parts.values(), key=lambda part: part[0].y)


def _extent(rows: list[Row]) -> tuple[float, float]:
    """Крайние края рядов по x."""
    return min(row.x0 for row in rows), max(row.x1 for row in rows)


def _rejoinable(upper: list[Row], lower: list[Row], rules: list, dpi: float, barriers, body_height: float) -> bool:
    """Можно ли склеить два блока в один: один под другим вплотную, края сходятся, набор один, черты между ними нет.

    Args:
        upper: Ряды верхнего блока.
        lower: Ряды нижнего блока.
        rules: Сплошные черты страницы.
        dpi: Разрешение рабочей копии.
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
    tol = mm_to_px(REJOIN_EDGE_MM, dpi)
    same_left = abs(ux0 - lx0) <= tol
    same_right = abs(ux1 - lx1) <= tol
    same_centre = abs((ux0 + ux1) - (lx0 + lx1)) / 2.0 <= tol
    # Конец абзаца (короткий последний ряд верхнего) и абзацный отступ (первый ряд нижнего) сдвигают
    # только одну сторону; строки заголовка по центру — центр.
    if not (same_left or same_right or same_centre):
        return False
    span = (int(min(ux0, lx0)), int(max(ux1, lx1)))
    if legacy._rule_between(last, first, rules, span):
        return False
    if legacy._barrier_between_rows(last, first, barriers, span):
        return False
    return True


# Предел разрыва для склейки крупного набора — в высотах строки (как ``GAP_HEIGHTS`` боевого кода).
GAP_HEIGHTS_REJOIN = 2.4


def rejoin(groups: list[list[Row]], rules: list, dpi: float, barriers, body_height: float) -> list[list[Row]]:
    """Склеивать соседние по вертикали группы страницы, пока находятся ложные разрезы (:func:`_rejoinable`).

    Работает поверх кусков разных колонок и зон: ложный разрез 1966/02 IMG_0098_2R — это восемь
    строк правой колонки, ушедших в кусок во всю ширину.

    Args:
        groups: Группы рядов сверху вниз каждая.
        rules: Сплошные черты страницы.
        dpi: Разрешение рабочей копии.
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
            if _rejoinable(upper, out[j], rules, dpi, barriers, body_height):
                out[i] = upper + out[j]
                del out[j]
                changed = True
                break
    return out


# Куски одной строки: середины ближе этой доли меньшей высоты...
PIECE_SAME_Y = 0.5
# ...и промежуток по x меньше стольких высот строки (отрицательный — перекрытие). При промежутке
# больше этого разница набора запрещает слияние: это законные ряды бок о бок (подпись и заголовок).
PIECE_GAP_HEIGHTS = 1.0


def merge_overlapping_pieces(rows: list[Row], barriers=None, gutters: list | None = None) -> list[Row]:
    """Слить куски одной строки до деления на блоки — независимо от набора, если они налезают или почти касаются.

    1972/03 IMG_0108_2R: строка «Первый этап — механизация … экономической» разобрана двумя рядами на
    одной высоте (x 64–558 и 414–826 — они НАЛЕЗАЮТ друг на друга). Начало набрано жирным, балл набора
    велик, и потоки с делением по набору развели куски в разные блоки: один закрыл верхний блок,
    другой открыл нижний, и блоки налезли друг на друга. Два ряда, налезающие по x, бок о бок не
    стоят — это одна строка.

    Args:
        rows: Ряды куска колонки.
        barriers: Линейки-барьеры: через них куски не сливаются.
        gutters: Межколонники: через них куски не сливаются.

    Returns:
        Ряды сверху вниз.
    """
    out = sorted(rows, key=lambda row: row.x0)
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                if abs(a.y - b.y) >= PIECE_SAME_Y * min(a.height, b.height):
                    continue
                gap = max(a.x0, b.x0) - min(a.x1, b.x1)
                if gap > PIECE_GAP_HEIGHTS * max(a.height, b.height):
                    continue
                # Линейка-барьер или межколонник между кусками запрещают слияние и у налезающих кусков: края
                # ряда по краске выходят за ось, и куски строки через барьер тоже «налезают».
                if _divided(a, b, barriers, gutters):
                    continue
                out[i] = _joined(a, b)
                del out[j]
                changed = True
                break
            if changed:
                break
    return sorted(out, key=lambda row: row.y)


def resolve_shared_rows(groups: list[list[Row]]) -> list[list[Row]]:
    """Развести ряды одной высоты, налезающие по x и попавшие в разные блоки: ряд уходит к блоку более длинного куска.

    Страховка на случай, когда куски одной строки всё же разошлись (разный кусок колонки, склейка).
    Из каждой такой пары рядов переносится более короткий; при равной длине — ряд из блока с большим
    номером (строгий порядок: иначе куски равной длины перебрасывались туда-обратно без конца —
    1969/08 IMG_0068_2R). Перенос за один проход: решения принимаются по исходной раскладке, поэтому
    цикл невозможен. Пустые блоки выбрасываются.

    Args:
        groups: Группы рядов.

    Returns:
        Группы после переноса.
    """
    home = {id(row): index for index, group in enumerate(groups) for row in group}
    rows = [row for group in groups for row in group]
    target = dict(home)
    for a in rows:
        for b in rows:
            if a is b or home[id(a)] == home[id(b)]:
                continue
            if abs(a.y - b.y) >= PIECE_SAME_Y * min(a.height, b.height):
                continue
            if min(a.x1, b.x1) - max(a.x0, b.x0) <= 0:
                continue
            # a уходит к b, если a строго «младше»: короче, а при равной длине — из блока с большим номером.
            if (a.x1 - a.x0, -home[id(a)]) < (b.x1 - b.x0, -home[id(b)]):
                target[id(a)] = home[id(b)]
    out: list[list[Row]] = [[] for _ in groups]
    for row in rows:
        out[target[id(row)]].append(row)
    return [sorted(group, key=lambda row: row.y) for group in out if group]


def column_groups(
    rows: list[Row], span: tuple[int, int], rules: list, dpi: float, body_height: float, barriers
) -> list[list[Row]]:
    """Деление куска колонки прежним ходом: потоки бок о бок, затем ``split_blocks`` в каждом.

    Args:
        rows: Ряды куска сверху вниз.
        span: Границы колонки.
        rules: Сплошные черты страницы.
        dpi: Разрешение рабочей копии.
        body_height: Медианная высота ряда страницы.
        barriers: Линейки-барьеры.

    Returns:
        Группы рядов сверху вниз.
    """
    column_pitch = pitch_of(rows)
    streams = legacy._streams(rows, column_pitch, dpi)
    groups = [
        group for stream in streams for group in split_blocks(stream, column_pitch, rules, span, body_height, barriers)
    ]
    groups.sort(key=lambda group: group[0].y)
    return groups


def smooth_groups(
    axes: list,
    zones: list,
    gutters: list,
    width: int,
    ink: np.ndarray,
    rules: list | None,
    dpi: float,
    leaders: list | None,
    barriers,
) -> list[tuple[tuple[int, int], list[Row]]]:
    """Группы рядов страницы способом ``SMOOTH``.

    1. Линейки, целиком лежащие в полосе строки (подчёркивания, штрихи букв крупного набора), — не
       барьеры (:func:`underline_free`).
    2. Куски колонок — как у прежнего хода (``blocks.column_pieces``); куски одной строки, налезающие
       по x или почти касающиеся, сливаются в ряд ДО деления (:func:`merge_overlapping_pieces`).
    3. Деление куска — прежним ходом (:func:`column_groups`); в каждой группе — куски одной строки
       одного набора в один ряд (:func:`merge_same_line`), части без «дотягивания» строк — врозь
       (:func:`reach_components`).
    4. Склейка ложных разрезов через колонки и зоны (:func:`rejoin`), развод рядов одной высоты, попавших
       в разные блоки (:func:`resolve_shared_rows`).

    Args:
        axes: Оси строк (уже с заменёнными осями-выбросами, :func:`axes_fix.fixed_axes`).
        zones, gutters, width, ink, rules, dpi, leaders, barriers: Как у ``blocks.blocks_of``.

    Returns:
        Группы ``(границы по x, ряды сверху вниз)`` в порядке чтения.
    """
    rules = rules or []
    barriers = underline_free(barriers, axes)
    pieces, body_height = column_pieces(axes, zones, gutters, width, ink, dpi, leaders, barriers)
    groups: list[list[Row]] = []
    for _, span, rows in pieces:
        rows = merge_overlapping_pieces(rows, barriers, gutters)
        for group in column_groups(rows, span, rules, dpi, body_height, barriers):
            groups.extend(reach_components(merge_same_line(group, dpi, barriers, gutters)))
    groups = resolve_shared_rows(rejoin(groups, rules, dpi, barriers, body_height))
    # Куски одной строки, попавшие в одну группу только после склейки ложных разрезов (до неё они были в разных
    # группах и сливаться не могли): строка библиографии с широкими пробелами шла в блок двумя рядами на одной
    # высоте, и у блока не строились стороны (ХиП 1991/01 IMG_1016).
    groups = [merge_same_line(rows, dpi, barriers, gutters) for rows in groups]
    ordered = sorted(groups, key=lambda rows: (rows[0].y, _extent(rows)[0]))
    return [((int(_extent(rows)[0]), int(_extent(rows)[1])), rows) for rows in ordered]


def smooth_blocks(
    axes: list,
    zones: list,
    gutters: list,
    width: int,
    ink: np.ndarray,
    rules: list | None = None,
    dpi: float = WORK_DPI,
    smooth_pitches: float = SMOOTH_PITCHES,
    coarse_factor: float = COARSE_FACTOR,
    dilate: float = DILATE_GLYPHS,
    leaders: list | None = None,
    barriers=None,
) -> list[TextBlock]:
    """Блоки страницы способом ``SMOOTH``: группы :func:`smooth_groups`, граница — :func:`smooth_envelope.envelope_smooth`.

    Крупная огибающая (``envelope_coarse``, для оверлея и отчёта) — прежняя ``envelope_of`` с окном в
    ``coarse_factor`` раз шире; справочная кромка по краске не считается (``envelope_ink = None``).

    Args:
        axes, zones, gutters, width, ink, rules, dpi, smooth_pitches, coarse_factor, dilate, leaders,
        barriers: Как у ``blocks.blocks_of``.

    Returns:
        Блоки в порядке чтения; ``column`` — номер блока, ``index`` — 0 (колонки у групп способа нет:
        склейка идёт через колонки и зоны).
    """
    reference = body_reference(axes)[0] if len(axes) >= 3 else None
    blocks: list[TextBlock] = []
    for number, (span, rows) in enumerate(
        smooth_groups(axes, zones, gutters, width, ink, rules, dpi, leaders, barriers)
    ):
        if len(rows) < MIN_BLOCK_ROWS:
            continue
        rows = sorted(rows, key=lambda row: row.y)
        pitch = pitch_of(rows)
        glyph = _glyph(rows)
        envelope = envelope_smooth(rows, dpi, reference)
        envelope = replace(
            envelope,
            polygon_dilated=legacy._dilated(envelope.polygon, glyph, dilate, dpi),
            dilate_px=(dilate * glyph[0], dilate * glyph[1]),
        )
        blocks.append(
            TextBlock(
                column=number,
                index=0,
                span=span,
                rows=tuple(rows),
                pitch_px=pitch,
                dpi=float(dpi),
                envelope=envelope,
                envelope_coarse=envelope_of(rows, pitch, dpi, smooth_pitches * coarse_factor),
                envelope_ink=None,
            )
        )
    return blocks


__all__ = [
    "column_groups",
    "merge_overlapping_pieces",
    "merge_same_line",
    "reach_components",
    "rejoin",
    "resolve_shared_rows",
    "smooth_blocks",
    "smooth_groups",
    "underline_free",
]
