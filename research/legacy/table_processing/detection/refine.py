"""Доводка рамки находки третьей версии: обрезка по линейкам и прилипание к просвету.

ЖИВОЙ КОД ПЕРЕЕХАЛ в ``ocr_utils.page_layout.tables.refine``: здесь остался реэкспорт, чтобы стенд исследований
(сравнение версий, добыча, отчёты) мерил тот же детектор, что стоит в конвейере.
Здесь остались ``trim_ruleless`` и ``snap_edges`` третьей версии; ``drop_border_rules``,
``fit_rows`` и ``push_edges`` берутся из конвейера.

ТРИ РАЗНЫЕ ЗАДАЧИ, и они решаются в разных местах конвейера.

1. ``drop_border_rules`` работает ДО кластеризации. Тень корешка и обрез страницы дают
   длинную тонкую вертикаль у самого края скана; при склейке зазором 6 мм она смыкается с
   подчёркиваниями и заголовками абзацев выше таблицы, и рамка растягивается на пол-полосы.
   На 1968/01 с.31 такая вертикаль длиной 119 мм в 8 мм от края превращала таблицу 129x49 мм
   в находку 135x119 мм.

2. ``trim_ruleless`` работает ПОСЛЕ проверки. Он срезает сверху и снизу полосы, под которыми
   нет ни одной вертикали, — заголовок над таблицей и сноску под ней, через которые прошла
   внешняя рамка. Та же мысль, что в ``structure.ruling_grid._trim_ruleless_rows``, но там
   она срабатывает поздно, на вырезке, когда чужой текст уже внутри.

3. ``snap_edges`` работает ПОСЛЕДНИМ и меняет только ту рамку, по которой режут. Замер по
   800 сторонам принятых находок: 709 сторон уже проходят по чистому месту, 78 надо двинуть
   наружу, 12 внутрь, и медианный сдвиг — 0.2 мм. Работа тут точечная, поэтому и правило
   осторожное: сторона трогается, только если под ней действительно стоит краска букв.

ПОЧЕМУ ПРИЛИПАНИЕ НЕ МЕНЯЕТ ПРИЗНАКИ ПРОВЕРКИ. ``detection.verify`` считает всё по вырезке
ровно по рамке, и ``INNER_MARGIN_MM = 4`` отсчитывается от её края: сдвиг границы на два
миллиметра превратил бы внешнюю линейку таблицы во «внутреннюю вертикаль» и сдвинул бы
``inner_v`` и ``edge_fill``, на которых стоят все пороги. Поэтому у находки две рамки:
``rule_box`` — огибающая линеек, по ней считается проверка, и ``box`` — то, что режут.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocr_utils.page_layout.tables.refine import (  # noqa: F401 — реэкспорт для стенда
    BORDER_MM,
    BORDER_LONG_SHARE,
    CROSS_TOL_MM,
    crosses,
    drop_border_rules,
    _centre_inside,
    _rule_bounds,
    MIN_VERTICALS_FOR_ROWS,
    fit_rows,
    PUSH_LIMIT_MM,
    PUSH_PAD_MM,
    OUTSIDE_SHARE,
    PUSH_PASSES,
    SLIGHT_MM,
    LINE_INSIDE_SHARE,
    OWN_LIMIT_MM,
    Pushed,
    _with_side,
    _settle,
    _side_position,
    _line_inside_share,
    push_edges,
)
from research.legacy.table_processing.detection.ruling import Lines, mm_to_px
from research.legacy.table_processing.geometry import Box


# --- 2. Полосы без опоры на линейки -----------------------------------------

# Меньше двух вертикалей — судить о теле таблицы нельзя. Замер на 200 хороших находках:
# единственная крупная потеря (1975/10 с.4, рамка 221 мм, срезалось 105 мм) была именно там,
# где вертикаль в рамке одна.
MIN_VERTICALS_TO_TRIM = 2

# Горизонталь остаётся в рамке, если её центр не дальше этого от области, занятой вертикалями:
# верхняя и нижняя линейки таблицы часто выходят чуть за концы вертикалей.
TRIM_SLACK_MM = 3.0

# Полоса между двумя сквозными горизонталями — СТРОКА ТАБЛИЦЫ, даже если вертикалей в ней
# не нашлось. Это не поблажка, а исправление конкретной поломки: у шапки таблицы
# «Наименование грузов | 1971 г. | 1972 г.» (1973/01 с.22) разделители граф идут только на
# высоту самой шапки — 5.8 мм, то есть короче порога поиска линеек (8 мм). Обрезка видела
# полосу без вертикалей и срезала шапку целиком, а вместе с ней и заголовки колонок.
#
# Выше этого полоса перестаёт быть строкой: 20 мм — это уже врезка или отдельный абзац.
# Заголовок НАД таблицей под правило не подпадает: над ним сквозной горизонтали нет.
TRIM_ROW_MM = 20.0

# Сквозная горизонталь — та, что перекрывает не меньше этой доли ширины рамки.
TRIM_LONG_SPAN = 0.8


def trim_ruleless(box: Box, lines: Lines, dpi: int) -> Box:
    """Сузить рамку до полос, у которых есть опора на вертикальные линейки.

    Сверху и снизу к таблице часто прилипает заголовок или сноска: внешняя рамка тянется и
    через них, и они становятся строками сетки. Признак чужого прост — под ним нет ни одной
    вертикали графы. Исключение — строка, зажатая между двумя сквозными горизонталями:
    см. ``TRIM_ROW_MM``.
    """
    inside_v = [segment for segment in lines.vertical if _centre_inside(segment.box, box)]
    if len(inside_v) < MIN_VERTICALS_TO_TRIM:
        return box
    slack = mm_to_px(TRIM_SLACK_MM, dpi)
    top = min(segment.box.y0 for segment in inside_v)
    bottom = max(segment.box.y1 for segment in inside_v)

    long_ys = sorted(
        (segment.box.y0 + segment.box.y1) // 2
        for segment in lines.horizontal
        if _centre_inside(segment.box, box) and segment.length >= TRIM_LONG_SPAN * box.width
    )
    top = _extend_to_row(top, long_ys, dpi, upwards=True)
    bottom = _extend_to_row(bottom, long_ys, dpi, upwards=False)

    inside_h = [
        segment
        for segment in lines.horizontal
        if _centre_inside(segment.box, box) and top - slack <= (segment.box.y0 + segment.box.y1) // 2 <= bottom + slack
    ]
    kept = inside_v + inside_h
    return Box(
        min(segment.box.x0 for segment in kept),
        min(segment.box.y0 for segment in kept),
        max(segment.box.x1 for segment in kept),
        max(segment.box.y1 for segment in kept),
    )


def _extend_to_row(edge: int, long_ys: list[int], dpi: int, upwards: bool) -> int:
    """Дотянуть край до соседней сквозной горизонтали, если между ними помещается строка.

    Шаг повторяется: у таблицы бывает многоуровневая шапка, и над первой линейкой стоит ещё
    одна. Каждый шаг не длиннее ``TRIM_ROW_MM``, поэтому уехать на весь лист нельзя.
    """
    limit = mm_to_px(TRIM_ROW_MM, dpi)
    slack = mm_to_px(TRIM_SLACK_MM, dpi)
    current = edge
    while True:
        if upwards:
            candidates = [y for y in long_ys if y < current - slack and current - y <= limit]
            nearest = max(candidates, default=None)
        else:
            candidates = [y for y in long_ys if y > current + slack and y - current <= limit]
            nearest = min(candidates, default=None)
        if nearest is None:
            return current
        current = nearest


# --- 3. Прилипание границы к чистому просвету --------------------------------

# Ниже этой доли краски под границей сторона считается чистой — тот же порог, что в
# ``detection.quality``, и намеренно тот же: мера и лечение должны говорить об одном.
CLEAN_SHARE = 0.01

# Дальше этого границу не двигают ни наружу, ни внутрь. Замер: 90% сторон укладываются
# в 1.2 мм, максимум 6.3 мм. Хвост за 3.5 мм — это уже не «буква торчит за линейку», а кусок
# соседнего блока, и его должно добирать доращивание, а не прилипание.
SNAP_LIMIT_MM = 3.5


@dataclass(frozen=True)
class Snapped:
    box: Box
    moved_mm: dict[str, float]
    failed: int  # сторон, где чистого места не нашлось ни внутри, ни снаружи


def snap_edges(box: Box, text_ink: np.ndarray, lines: Lines, dpi: int) -> Snapped:
    """Отодвинуть границы рамки так, чтобы ни одна из них не шла по буквам.

    Наружу или внутрь — куда ближе чистый просвет; при равенстве наружу, потому что торчащая
    за линейку буква чаще принадлежит таблице, чем соседнему абзацу.

    Внутрь есть жёсткий предел: НИКОГДА не заходить за крайнюю линейку рамки. Между границей
    и крайней линейкой — единственное место, где может лежать прихваченный чужой текст; всё,
    что за линейкой, принадлежит таблице, и резать его нельзя ни при каком пороге.
    """
    height, width = text_ink.shape[:2]
    box = box.clipped(width, height)
    limit = mm_to_px(SNAP_LIMIT_MM, dpi)
    scale = 25.4 / dpi
    ink = text_ink > 0

    inner = _rule_bounds(lines, box)
    columns = ink[:, box.x0 : box.x1].mean(axis=1) if box.width > 0 else np.zeros(height)
    rows = ink[box.y0 : box.y1, :].mean(axis=0) if box.height > 0 else np.zeros(width)

    sides = {
        "сверху": (columns, box.y0, -1, inner[1]),
        "снизу": (columns, box.y1 - 1, +1, inner[3]),
        "слева": (rows, box.x0, -1, inner[0]),
        "справа": (rows, box.x1 - 1, +1, inner[2]),
    }
    shifts: dict[str, int] = {}
    moved: dict[str, float] = {}
    failed = 0
    for side, (profile, position, outward, rule_limit) in sides.items():
        shifts[side] = 0
        moved[side] = 0.0
        if not 0 <= position < profile.size or profile[position] <= CLEAN_SHARE:
            continue
        # Внутрь двигаться можно только до крайней линейки — дальше уже тело таблицы.
        inward_room = abs(rule_limit - position)
        out = _first_clean(profile, position, outward, limit)
        into = _first_clean(profile, position, -outward, min(limit, inward_room))
        if out is None and into is None:
            failed += 1
            continue
        if into is None or (out is not None and out <= into):
            shifts[side] = outward * out
        else:
            shifts[side] = -outward * into
        moved[side] = shifts[side] * outward * scale

    # Сначала числа, потом ``Box``: конструктор бросает на вывернутой рамке, а у узкой находки
    # два встречных сдвига по 3.5 мм её вывернуть могут. Проверять надо ДО, а не после.
    left, top = box.x0 + shifts["слева"], box.y0 + shifts["сверху"]
    right, bottom = box.x1 + shifts["справа"], box.y1 + shifts["снизу"]
    if right <= left or bottom <= top:
        return Snapped(box, {side: 0.0 for side in sides}, failed)
    return Snapped(Box(left, top, right, bottom).clipped(width, height), moved, failed)


def _first_clean(profile: np.ndarray, start: int, step: int, limit: int) -> "int | None":
    for distance in range(1, int(limit) + 1):
        index = start + step * distance
        if not 0 <= index < profile.size:
            return None
        if profile[index] <= CLEAN_SHARE:
            return distance
    return None
