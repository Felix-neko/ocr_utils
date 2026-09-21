"""Линейки четвёртой версии: цепочки фрагментов и связные ядра вместо склейки дилатацией.

ДВЕ БОЛЕЗНИ ТРЕТЬЕЙ ВЕРСИИ, которые лечатся здесь.

1. Линейка ищется морфологическим открытием ядром 8 мм: краска обязана тянуться по ОДНОЙ
   строке пикселей восемь миллиметров подряд. На изогнутой полосе (1975/07 с.42) загнутый
   конец линейки уходит вниз быстрее, чем на свою толщину, и открытие его отрезает — а с ним
   и последнюю графу таблицы. Здесь ядро втрое короче (3 мм), и фрагменты СШИВАЮТСЯ в
   цепочку по соосности: тот же приём, которым сшивают штрихи в ``curved_lines``. Так линейка
   идёт вслед за изгибом, как и просил человек — «следить за продолжениями рёбер, даже если
   они не отвесные».

2. Скопление собиралось дилатацией на 6 мм: всё, что стоит ближе шести миллиметров, — одна
   таблица. Поэтому к таблице липли подчёркивание из соседней колонки (1966/05 с.29: 32 мм в
   4 мм от угла), линии бланка в другой колонке (1974/06 с.97), линейка колонтитула (есть на
   трети полос пака). Здесь скопление — СВЯЗНОЕ ЯДРО: линейки, которые пересекаются или
   продолжают друг друга. Одиночная линейка ядром не является и ни к кому не клеится.

КЛЯКСЫ. Ядро недосвета у корешка бинаризуется в тонкую тёмную полосу и проходит все пороги
линейки (1976/12 с.30: толщина 3.4 px при линейках 2.0–2.4). Толщина её не отделяет: замер по
290 полосам показал, что 279 из 4301 линеек настоящих таблиц тоньше 1.5 px, а кляксы бывают и
тонкими. Отделяет СОСЕДСТВО: у линейки по обе стороны чистая бумага, у ядра кляксы — серая
муть, которая местами тоже проходит порог. Доля не-линеечной краски в полосах шириной 1 мм по
обе стороны: у 99 % настоящих линеек не выше 0.38, у клякс 0.86–0.99.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.page_layout.tables.refine import crosses
from ocr_utils.page_layout.tables.ruling import (
    MAX_RULE_THICKNESS_MM,
    MIN_RULE_MM,
    Lines,
    Segment,
    _angle_of,
    _axis_mask,
    binarize,
    mm_to_px,
)
from ocr_utils.page_layout.geometry import Box

# Ядро поиска фрагмента: 3 мм. Короче — и тире, перекладины «Т» и «Г» становятся фрагментами
# сотнями; длиннее — и загнутый конец линейки снова теряется. Одиночный фрагмент в линейку не
# превращается: цепочка обязана набрать ``MIN_RULE_MM``.
FRAGMENT_MM = 3.0

# Два фрагмента одной оси сшиваются, если зазор вдоль оси не больше этого: 8 мм. Столько же
# сшивало закрытие в ``ruling._axis_mask`` (ядро равно порогу длины линейки), и меньший зазор
# терял паритет: при 2 мм 625 линеек третьей версии из 7600 оставались без пары — линейка,
# продавленная или пунктирная, рвётся и на 3-6 мм. Сшить через такой зазор два ЧУЖИХ штриха
# мешают соосность и согласный наклон, которых у закрытия не было вовсе.
CHAIN_GAP_MM = 8.0

# Через зазор шире этого сшиваются только фрагменты не короче ``CHAIN_LONG_MM``: стебель буквы
# («ф», «р», «у» — 3 мм) в 6 мм над таблицей стоит ровно на оси вертикали графы и через
# широкий зазор пришивался к ней, поднимая рамку на строку абзаца (1966/03 с.81).
CHAIN_SHORT_GAP_MM = 2.0
CHAIN_LONG_MM = 5.0

# И если на стыке они расходятся поперёк не больше этого (с учётом наклона первого): 0.35 мм,
# два пикселя при 150 dpi. Порванная сканом линейка продолжается на той же строке, изогнутая
# уходит между соседними фрагментами меньше чем на пиксель. Допуск шире (0.7 мм) сшивал
# нижнюю линейку таблицы с подчёркиванием в соседней колонке, стоящим на 0.9 мм ниже
# (1966/05 с.29): наклон линейки «предсказывал» как раз такой уход.
CHAIN_OFFSET_MM = 0.35

# И если их наклоны согласны в этих пределах. Линейки пака лежат в 1.7°, изгиб добавляет доли.
CHAIN_ANGLE_DEG = 2.0

# Ширина полос по обе стороны линейки, в которых меряется соседство: 1 мм.
ISOLATION_BAND_MM = 1.0

# Выше этой доли краски в соседних полосах (по МЕНЬШЕЙ из сторон) отрезок — не линейка, а ядро
# кляксы. Замер по большей стороне: 99 % настоящих линеек не выше 0.38, кляксы 0.86–0.99; по
# меньшей стороне настоящие линейки ещё ниже. Порог в пустом промежутке.
MAX_ISOLATION = 0.55

# Допуск на пересечение линеек в ядре — тот же, что в ``refine.drop_border_rules``.
CROSS_TOL_MM = 1.5

# Две стопки линеек (шапка, отбитая от тела просветом) — одно ядро, если перекрываются по
# длинной оси на эту долю и стоят не дальше ``STACK_GAP_MM``. Это замена дилатации 6 мм ровно
# для того случая, где она была права. Но две ЦЕЛЫЕ таблицы в 5 мм друг от друга (1975/05
# с.96) сливать нельзя, поэтому стопка сливается, только если одна из частей неполна (меньше
# ``COMPLETE_LONG_RULES`` сквозных горизонталей: одна строка шапки, бланк без тела) или графы
# обеих частей стоят на одних x (``ALIGNED_SHARE``). Между частями могут лежать сирые
# горизонтали бланка («Периодичность завоза ___»): они служат мостиком, если лежат внутри
# общей ширины стопки, — в соседней колонке линии бланка мостиком не считаются (1974/06 с.97).
STACK_OVERLAP = 0.8
STACK_GAP_MM = 6.0
# Ближе этого две стопки сливаются без оговорок: две РАЗНЫЕ таблицы вплотную не печатают.
# Шапку бланка без вертикалей в 5 мм над телом (1972/10 с.36, «Форма классификатора
# соответствия») это правило НЕ берёт: её линейки ни с чем не пересекаются, а тянуть сирые
# горизонтали на 5 мм — значит тянуть и линейку колонтитула над таблицей у верха полосы.
STACK_TOUCH_MM = 2.5
COMPLETE_LONG_RULES = 3
LONG_RULE_SPAN = 0.8
ALIGNED_SHARE = 0.5
ALIGNED_TOL_MM = 1.0


@dataclass(frozen=True)
class Fragment:
    """Кусок линейки: габарит, ось, наклон, площадь краски."""

    box: Box
    horizontal: bool
    angle_deg: float
    area: int

    @property
    def length(self) -> int:
        return self.box.width if self.horizontal else self.box.height

    @property
    def along(self) -> tuple[int, int]:
        return (self.box.x0, self.box.x1) if self.horizontal else (self.box.y0, self.box.y1)

    @property
    def across(self) -> float:
        return (self.box.y0 + self.box.y1) / 2 if self.horizontal else (self.box.x0 + self.box.x1) / 2


def fragments(binary: np.ndarray, dpi: int, min_mm: float = FRAGMENT_MM) -> tuple[list[Fragment], list[Fragment]]:
    """Фрагменты линеек обеих осей: открытие коротким ядром, компоненты, порог толщины."""
    length = mm_to_px(min_mm, dpi)
    thickness = mm_to_px(MAX_RULE_THICKNESS_MM, dpi)
    result: list[list[Fragment]] = []
    for horizontal in (True, False):
        mask = _axis_mask(binary, length, horizontal)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        found: list[Fragment] = []
        for index in range(1, count):
            left, top, width, height, area = stats[index]
            extent = width if horizontal else height
            if area / max(extent, 1) > thickness:
                continue
            box = Box(int(left), int(top), int(left + width), int(top + height))
            found.append(Fragment(box, horizontal, _angle_of(labels, index, horizontal, box), int(area)))
        result.append(found)
    return result[0], result[1]


def _predicted_across(first: Fragment, at_along: float) -> float:
    """Где прошла бы поперечная координата первого фрагмента, продолжи его до ``at_along``."""
    centre_along = sum(first.along) / 2
    slope = np.tan(np.radians(first.angle_deg if first.horizontal else -first.angle_deg))
    return first.across + slope * (at_along - centre_along)


def continues(first: Fragment, second: Fragment, dpi: int, gap_mm: float = CHAIN_GAP_MM) -> bool:
    """Продолжает ли второй фрагмент первый: зазор мал, стык соосен, наклон согласен."""
    if first.horizontal is not second.horizontal:
        return False
    a0, a1 = first.along
    b0, b1 = second.along
    gap = max(b0 - a1, a0 - b1)
    if gap > mm_to_px(gap_mm, dpi):
        return False
    if gap > mm_to_px(CHAIN_SHORT_GAP_MM, dpi) and min(first.length, second.length) < mm_to_px(CHAIN_LONG_MM, dpi):
        return False
    if abs(first.angle_deg - second.angle_deg) > CHAIN_ANGLE_DEG:
        return False
    # Стык — ближний к первому конец второго. Поперёк сравнивается предсказание по наклону
    # первого с фактическим положением второго на том же месте.
    joint = b0 if b0 >= a1 else b1
    predicted = _predicted_across(first, joint)
    actual = _predicted_across(second, joint)
    return abs(predicted - actual) <= mm_to_px(CHAIN_OFFSET_MM, dpi)


def _components(count: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Компоненты связности простого графа на ``count`` вершинах."""
    parent = list(range(count))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for first, second in edges:
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            parent[root_a] = root_b
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def _chain_edges(items: list[Fragment], dpi: int, gap_mm: float) -> list[tuple[int, int]]:
    """Пары фрагментов, которые продолжают друг друга. Сравниваются только соседи по оси."""
    order = sorted(range(len(items)), key=lambda index: items[index].along[0])
    reach = mm_to_px(gap_mm, dpi)
    edges: list[tuple[int, int]] = []
    for position, index in enumerate(order):
        first = items[index]
        for other in order[position + 1 :]:
            second = items[other]
            if second.along[0] - first.along[1] > reach:
                break
            if continues(first, second, dpi, gap_mm) or continues(second, first, dpi, gap_mm):
                edges.append((index, other))
    return edges


def _merge(items: list[Fragment], labels: np.ndarray | None = None) -> Segment:
    """Одна линейка из цепочки фрагментов: объединённый габарит, наклон по всем точкам."""
    box = Box(
        min(f.box.x0 for f in items),
        min(f.box.y0 for f in items),
        max(f.box.x1 for f in items),
        max(f.box.y1 for f in items),
    )
    horizontal = items[0].horizontal
    # Наклон цепочки — регрессия по центрам фрагментов, взвешенная их длиной: у изогнутой
    # линейки это средний наклон, а он и нужен для продолжения за конец.
    if len(items) >= 2:
        along = np.array([sum(f.along) / 2 for f in items], dtype=float)
        across = np.array([f.across for f in items], dtype=float)
        weights = np.array([max(1, f.length) for f in items], dtype=float)
        if along.max() - along.min() >= 4:
            slope = np.polyfit(along, across, 1, w=np.sqrt(weights))[0]
            angle = float(np.degrees(np.arctan(slope)))
            angle = angle if horizontal else -angle
        else:
            angle = float(np.average([f.angle_deg for f in items], weights=weights))
    else:
        angle = items[0].angle_deg
    return Segment(box, horizontal, angle)


def chain(items: list[Fragment], dpi: int, min_mm: float = MIN_RULE_MM, gap_mm: float = CHAIN_GAP_MM) -> list[Segment]:
    """Сшить фрагменты в линейки и оставить те, что набрали ``min_mm``."""
    if not items:
        return []
    groups = _components(len(items), _chain_edges(items, dpi, gap_mm))
    minimal = mm_to_px(min_mm, dpi)
    rules: list[Segment] = []
    for group in groups:
        merged = _merge([items[index] for index in group])
        if merged.length >= minimal:
            rules.append(merged)
    return rules


def isolation(ink_without_rules: np.ndarray, segment: Segment, dpi: int) -> float:
    """Доля краски в полосах 1 мм по обе стороны отрезка: у линейки ноль, у ядра кляксы — много.

    Краска подаётся уже БЕЗ линеек (``text_ink``), иначе двойная линейка шапки затемняла бы
    соседку и обе уходили бы в кляксы.
    """
    box = segment.box
    band = mm_to_px(ISOLATION_BAND_MM, dpi)
    height, width = ink_without_rules.shape[:2]
    if segment.horizontal:
        above = ink_without_rules[max(0, box.y0 - band - 1) : max(0, box.y0 - 1), box.x0 : box.x1]
        below = ink_without_rules[box.y1 + 1 : min(height, box.y1 + 1 + band), box.x0 : box.x1]
    else:
        above = ink_without_rules[box.y0 : box.y1, max(0, box.x0 - band - 1) : max(0, box.x0 - 1)]
        below = ink_without_rules[box.y0 : box.y1, box.x1 + 1 : min(width, box.x1 + 1 + band)]
    shares = [float((side > 0).mean()) for side in (above, below) if side.size]
    # МЕНЬШАЯ из двух сторон, не большая: подчёркивание под словом и линейка, к которой вплотную
    # прижата строка таблицы, темны с ОДНОЙ стороны, а ядро кляксы — с обеих.
    return min(shares) if shares else 0.0


def find_rules(
    gray: np.ndarray,
    dpi: int,
    binary: "np.ndarray | None" = None,
    found_fragments: "tuple[list[Fragment], list[Fragment]] | None" = None,
) -> Lines:
    """Линейки полосы: фрагменты → цепочки → фильтр клякс. Маски — по самим фрагментам.

    ``found_fragments`` — уже посчитанные фрагменты, если вызывающий считает их и для себя
    (детектор использует их ещё и при доращивании).
    """
    binary = binarize(gray) if binary is None else binary
    horizontal_fragments, vertical_fragments = found_fragments or fragments(binary, dpi)
    horizontal = chain(horizontal_fragments, dpi)
    vertical = chain(vertical_fragments, dpi)

    # Маски линеек — только те фрагменты, что вошли в линейку: по ним считаются краска без
    # линеек и трассы, а обрывки тире в них не нужны.
    height, width = binary.shape[:2]
    h_mask = np.zeros((height, width), np.uint8)
    v_mask = np.zeros((height, width), np.uint8)
    fragment_h = _axis_mask(binary, mm_to_px(FRAGMENT_MM, dpi), True)
    fragment_v = _axis_mask(binary, mm_to_px(FRAGMENT_MM, dpi), False)
    for segment in horizontal:
        h_mask[segment.box.slice] = fragment_h[segment.box.slice]
    for segment in vertical:
        v_mask[segment.box.slice] = fragment_v[segment.box.slice]

    rules = Lines(horizontal, vertical, h_mask, v_mask)
    fat = cv2.dilate(rules.mask, np.ones((3, 3), np.uint8))
    ink = cv2.bitwise_and(binary, cv2.bitwise_not(fat))
    kept_h = [s for s in horizontal if isolation(ink, s, dpi) <= MAX_ISOLATION]
    kept_v = [s for s in vertical if isolation(ink, s, dpi) <= MAX_ISOLATION]
    if len(kept_h) != len(horizontal) or len(kept_v) != len(vertical):
        h_mask = np.zeros((height, width), np.uint8)
        v_mask = np.zeros((height, width), np.uint8)
        for segment in kept_h:
            h_mask[segment.box.slice] = fragment_h[segment.box.slice]
        for segment in kept_v:
            v_mask[segment.box.slice] = fragment_v[segment.box.slice]
    return Lines(kept_h, kept_v, h_mask, v_mask)


# --- Связные ядра --------------------------------------------------------------


def _as_fragment(segment: Segment) -> Fragment:
    return Fragment(segment.box, segment.horizontal, segment.angle_deg, 0)


def _long_rules(group: list[Segment], box: Box) -> int:
    return sum(1 for s in group if s.horizontal and s.length >= LONG_RULE_SPAN * max(1, box.width))


def _aligned(first: list[Segment], second: list[Segment], dpi: int) -> bool:
    """Стоят ли вертикали двух ядер на одних x: шапка и тело одной таблицы делят графы."""
    xs_a = [(s.box.x0 + s.box.x1) / 2 for s in first if not s.horizontal]
    xs_b = [(s.box.x0 + s.box.x1) / 2 for s in second if not s.horizontal]
    if not xs_a or not xs_b:
        return False
    small, big = (xs_a, xs_b) if len(xs_a) <= len(xs_b) else (xs_b, xs_a)
    tolerance = mm_to_px(ALIGNED_TOL_MM, dpi)
    hits = sum(1 for x in small if any(abs(x - y) <= tolerance for y in big))
    return hits / len(small) >= ALIGNED_SHARE


def _bridged(first: Box, second: Box, orphans: list[Segment], dpi: int) -> bool:
    """Соединяют ли сирые горизонтали две рамки стопкой: цепочка шагов не длиннее зазора."""
    gap = mm_to_px(STACK_GAP_MM, dpi)
    top, bottom = (first, second) if first.y1 <= second.y0 else (second, first)
    if top.y1 > bottom.y0:
        return False
    left, right = max(top.x0, bottom.x0), min(top.x1, bottom.x1)
    if right - left < STACK_OVERLAP * min(top.width, bottom.width):
        return False
    lo, hi = min(top.x0, bottom.x0), max(top.x1, bottom.x1)
    slack = mm_to_px(2.0, dpi)
    steps = sorted(
        (s.box.y0 + s.box.y1) // 2
        for s in orphans
        if s.horizontal
        and top.y1 <= (s.box.y0 + s.box.y1) // 2 <= bottom.y0
        and s.box.x0 >= lo - slack
        and s.box.x1 <= hi + slack
    )
    current = top.y1
    for y in steps + [bottom.y0]:
        if y - current > gap:
            return False
        current = y
    return True


def _stacked(first: Box, second: Box, dpi: int, gap_mm: float = STACK_GAP_MM) -> bool:
    """Стоят ли две рамки стопкой или рядом: перекрытие по одной оси, малый зазор по другой."""
    gap = mm_to_px(gap_mm, dpi)
    overlap_x = min(first.x1, second.x1) - max(first.x0, second.x0)
    overlap_y = min(first.y1, second.y1) - max(first.y0, second.y0)
    if overlap_x >= STACK_OVERLAP * min(first.width, second.width) and -gap <= overlap_y <= 0:
        return True
    if overlap_y >= STACK_OVERLAP * min(first.height, second.height) and -gap <= overlap_x <= 0:
        return True
    return False


def _overlapping(first: Box, second: Box) -> bool:
    return min(first.x1, second.x1) > max(first.x0, second.x0) and min(first.y1, second.y1) > max(first.y0, second.y0)


def _should_merge(
    group_a: list[Segment], box_a: Box, group_b: list[Segment], box_b: Box, orphans: list[Segment], dpi: int
) -> bool:
    if _overlapping(box_a, box_b) or _stacked(box_a, box_b, dpi, STACK_TOUCH_MM):
        return True
    if not (_stacked(box_a, box_b, dpi) or _bridged(box_a, box_b, orphans, dpi)):
        return False
    incomplete = _long_rules(group_a, box_a) < COMPLETE_LONG_RULES or _long_rules(group_b, box_b) < COMPLETE_LONG_RULES
    return incomplete or _aligned(group_a, group_b, dpi)


def cores(lines: Lines, dpi: int) -> list[list[Segment]]:
    """Связные ядра линеек: группы, в которых линейки пересекаются или продолжают друг друга.

    Одиночная линейка ядром не считается — колонтитул, подчёркивание и дробная черта здесь и
    отпадают. Ядра, стоящие стопкой (шапка, отбитая от тела просветом), сливаются.
    """
    segments = list(lines.horizontal) + list(lines.vertical)
    if not segments:
        return []
    tolerance = mm_to_px(CROSS_TOL_MM, dpi)
    edges: list[tuple[int, int]] = []
    n_h = len(lines.horizontal)
    for h_index, horizontal in enumerate(lines.horizontal):
        for v_offset, vertical in enumerate(lines.vertical):
            if crosses(vertical, horizontal, tolerance):
                edges.append((h_index, n_h + v_offset))
    for axis_items, base in ((lines.horizontal, 0), (lines.vertical, n_h)):
        as_fragments = [_as_fragment(s) for s in axis_items]
        edges += [(base + a, base + b) for a, b in _chain_edges(as_fragments, dpi, CHAIN_GAP_MM)]
    groups = [g for g in _components(len(segments), edges) if len(g) >= 2]
    in_core = {i for g in groups for i in g}
    orphans = [segments[i] for i in range(len(segments)) if i not in in_core]

    # Слияние стопок — по рамкам ядер, до сходимости.
    boxes = [_extent([segments[i] for i in g]) for g in groups]
    merged = True
    while merged and len(groups) > 1:
        merged = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if _should_merge(
                    [segments[i] for i in groups[a]], boxes[a], [segments[i] for i in groups[b]], boxes[b], orphans, dpi
                ):
                    groups[a] = groups[a] + groups[b]
                    boxes[a] = _extent([segments[i] for i in groups[a]])
                    del groups[b]
                    del boxes[b]
                    merged = True
                    break
            if merged:
                break

    taken = {i for g in groups for i in g}
    # Поглощение: линейка не из ядра, целиком лежащая внутри рамки ядра, добавляется к нему.
    slack = mm_to_px(1.0, dpi)
    for index, segment in enumerate(segments):
        if index in taken:
            continue
        for g_index, box in enumerate(boxes):
            if _inside(segment.box, box, slack):
                groups[g_index].append(index)
                taken.add(index)
                break
    return [[segments[i] for i in g] for g in groups]


def _extent(segments: list[Segment]) -> Box:
    return Box(
        min(s.box.x0 for s in segments),
        min(s.box.y0 for s in segments),
        max(s.box.x1 for s in segments),
        max(s.box.y1 for s in segments),
    )


def _inside(rule: Box, box: Box, slack: int) -> bool:
    return (
        rule.x0 >= box.x0 - slack
        and rule.y0 >= box.y0 - slack
        and rule.x1 <= box.x1 + slack
        and rule.y1 <= box.y1 + slack
    )


__all__ = ["Fragment", "chain", "continues", "cores", "find_rules", "fragments", "isolation"]
