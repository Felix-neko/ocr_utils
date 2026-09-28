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
from ocr_utils.page_layout.geometry import Box, LooseRule

# Ядро поиска фрагмента: 3 мм. Короче — и тире, перекладины «Т» и «Г» становятся фрагментами
# сотнями; длиннее — и загнутый конец линейки снова теряется. Одиночный фрагмент в линейку не
# превращается: цепочка обязана набрать ``MIN_RULE_MM``.
FRAGMENT_MM = 3.0

# Разрывы до этой длины между найденными пробегами зашиваются ещё при поиске кусков, без проверок; всё,
# что длиннее, сшивает цепочка (``continues``) — с проверкой соосности и правилом коротких кусков. Раньше
# зашивалось ядром самого куска (3 мм), и два ствола букв соседних строк («р» над «р», разрыв 2,7 мм)
# склеивались в «линейку» мимо этого правила (1966/03 IMG_0143_1L). Замер по паку-1 (2026-09-28, 914
# таблиц): при 2 мм сетка ячеек не изменилась у 913; линеек-сирот исчезло 770 вертикалей и 259
# горизонталей, в просмотренной выборке — стволы и буквы заголовков.
FRAGMENT_CLOSE_MM = 2.0

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

# ОТКРЫТАЯ ТАБЛИЦА: шапка в линейках, а в теле только вертикали граф, и те разорваны
# подзаголовками разделов («а) продукция машиностроения…») — нижней линейки нет, горизонталей
# в теле нет, куски вертикалей ни с чем не пересекаются и в ядро не попадают. Затравкой
# становилась одна шапка и отклонялась как «заголовок в рамке» (1970/12 с.87, 1975/02 с.75).
# Кусок тела продолжает графы ядра, если начинается не дальше этого зазора ниже него: замер
# разрывов на двух полосах — 6,3–11,2 мм (строка подзаголовка с отбивками при шаге строк 3,8 мм),
# порог с запасом на подзаголовок в две строки не берём.
OPEN_BODY_GAP_MM = 13.0
# Шаг тела принимается, только если продолжаются хотя бы столько граф ядра одновременно:
# одна вертикаль под шапкой — это и межколонная линейка вёрстки, её тянуть нельзя.
OPEN_BODY_MIN_COLUMNS = 2
# Куски одного шага тела перекрываются по y хотя бы на эту долю более короткого.
OPEN_BODY_ROW_OVERLAP = 0.5
# Графы считаются только внутренние: вертикаль ближе этого к боку рамки — боковина, а не графа.
OPEN_BODY_SIDE_MM = 4.0
# Верхняя линейка раздела ищется в этой полосе от его верха: есть — это отдельная таблица.
OPEN_BODY_TOP_RULE_MM = 2.0

# МОСТ: две полные таблицы одна под другой связывает в одно ядро одна-единственная длинная
# линейка — правая вертикаль, идущая вдоль обеих (1969/02 IMG_0074_2R: таблица 2 и таблица ниже
# через вертикаль x ≈ 815 px на 150 dpi). Правило стопок (``_should_merge``: две полные таблицы с
# несовпадающими графами не сливать) до них не доходило — связность решала раньше. Теперь ядро
# режется по такой линейке, если без неё куски не слились бы и по правилу стопок.
# Кандидаты в мосты — самые длинные линейки ядра по каждой оси (мост всегда длиннее кусков).
BRIDGE_CANDIDATES_PER_AXIS = 4
# Ядра крупнее этого не режутся: разбор идёт по кандидатам за квадрат от числа линеек.
BRIDGE_MAX_RULES = 150


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


@dataclass(frozen=True)
class FragmentLayer:
    """Фрагменты одной оси вместе с картой меток компонент, из которой они вырезаны.

    Карта нужна трассировке изогнутых линеек (``traces``): чтобы взять ИМЕННО пиксели линейки, а не
    всё, что попало в её габарит, — у наклонной линейки габарит высотой в сантиметр и тянет чужую
    краску. ``label_ids[i]`` — номер компоненты в ``labels`` у фрагмента ``fragments[i]``.
    """

    fragments: list[Fragment]
    labels: np.ndarray
    label_ids: list[int]


def fragment_layers(binary: np.ndarray, dpi: int, min_mm: float = FRAGMENT_MM) -> tuple[FragmentLayer, FragmentLayer]:
    """Фрагменты линеек обеих осей с картами меток: открытие коротким ядром, компоненты, порог толщины.

    Args:
        binary: Краска белым на чёрном (``ruling.binarize``).
        dpi: Разрешение картинки — для перевода миллиметров в пиксели.
        min_mm: Длина ядра открытия, то есть минимальная длина фрагмента.

    Returns:
        Слои горизонтальных и вертикальных фрагментов (в этом порядке).
    """
    length = mm_to_px(min_mm, dpi)
    thickness = mm_to_px(MAX_RULE_THICKNESS_MM, dpi)
    result: list[FragmentLayer] = []
    for horizontal in (True, False):
        mask = _axis_mask(binary, length, horizontal, mm_to_px(FRAGMENT_CLOSE_MM, dpi))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        found: list[Fragment] = []
        label_ids: list[int] = []
        for index in range(1, count):
            left, top, width, height, area = stats[index]
            extent = width if horizontal else height
            # Толщина как площадь на длину: толще миллиметра — не линейка, а пятно или буква.
            if area / max(extent, 1) > thickness:
                continue
            box = Box(int(left), int(top), int(left + width), int(top + height))
            found.append(Fragment(box, horizontal, _angle_of(labels, index, horizontal, box), int(area)))
            label_ids.append(index)
        result.append(FragmentLayer(found, labels, label_ids))
    return result[0], result[1]


def fragments(binary: np.ndarray, dpi: int, min_mm: float = FRAGMENT_MM) -> tuple[list[Fragment], list[Fragment]]:
    """Фрагменты линеек обеих осей: открытие коротким ядром, компоненты, порог толщины."""
    horizontal, vertical = fragment_layers(binary, dpi, min_mm)
    return horizontal.fragments, vertical.fragments


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


def components(count: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Компоненты связности простого графа на ``count`` вершинах (союз-поиск).

    Args:
        count: Число вершин, они нумеруются ``0 … count-1``.
        edges: Рёбра графа парами номеров вершин.

    Returns:
        Списки номеров вершин по компонентам; одиночная вершина — компонента из одной вершины.
    """
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


# Прежнее имя: его импортирует замороженный стенд research/legacy/table_processing.
_components = components


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


@dataclass(frozen=True)
class ChainedRule:
    """Линейка-цепочка вместе с номерами фрагментов, из которых она сшита (индексы во входном списке)."""

    segment: Segment
    members: tuple[int, ...]


def chain_groups(
    items: list[Fragment], dpi: int, min_mm: float = MIN_RULE_MM, gap_mm: float = CHAIN_GAP_MM
) -> list[ChainedRule]:
    """Сшить фрагменты в линейки, сохранив состав каждой; оставить те, что набрали ``min_mm``.

    Состав нужен трассировке (``traces``): изгиб линейки живёт в её фрагментах, а ``Segment`` хранит
    только габарит и средний наклон.

    Args:
        items: Фрагменты одной оси.
        dpi: Разрешение картинки.
        min_mm: Минимальная длина линейки-цепочки.
        gap_mm: Наибольший зазор между сшиваемыми фрагментами вдоль оси.

    Returns:
        Линейки в порядке компонент графа сшивки; у каждой — отрезок и номера её фрагментов.
    """
    if not items:
        return []
    groups = components(len(items), _chain_edges(items, dpi, gap_mm))
    minimal = mm_to_px(min_mm, dpi)
    rules: list[ChainedRule] = []
    for group in groups:
        merged = _merge([items[index] for index in group])
        if merged.length >= minimal:
            rules.append(ChainedRule(merged, tuple(group)))
    return rules


def chain(items: list[Fragment], dpi: int, min_mm: float = MIN_RULE_MM, gap_mm: float = CHAIN_GAP_MM) -> list[Segment]:
    """Сшить фрагменты в линейки и оставить те, что набрали ``min_mm``."""
    return [rule.segment for rule in chain_groups(items, dpi, min_mm, gap_mm)]


# Линейка-сирота (не вошедшая ни в таблицу, ни в схему) должна стоять на ЧИСТОЙ бумаге с обеих сторон.
# Ложная «линейка» — это низы, верхи или перекладины букв жирного заголовка, сшитые через узкие
# промежутки между буквами, или стволы букв соседних строк («р» над «р»): с одной стороны вплотную
# стоят сами буквы. Мера — доля краски в полосе ``LOOSE_SIDE_BAND_MM`` по БОЛЬШЕЙ из сторон, вдоль
# оси ломаной. Замер на паке-1 (2026-09-28, случайные выборки по 80 горизонталей и вертикалей, размечены
# глазами; ``ai_slop/rule_side_probe.py``): у ложных горизонталей 0.27–0.82, у отбивок 0.00–0.02, у дробных
# черт и подчёркиваний до 0.38; у ложных вертикалей 0.19–0.56, у межколонных линеек с текстом вплотную
# до 0.21. Длинные линейки (колонтитул с текстом над ним, пунктир, орнамент дают до 0.41) не трогаются.
LOOSE_SIDE_BAND_MM = 1.0
LOOSE_MAX_SIDE_INK_HORIZONTAL = 0.4
LOOSE_MAX_SIDE_INK_VERTICAL = 0.25
LOOSE_KEEP_MM = 60.0


def side_ink(binary: np.ndarray, rule: LooseRule, dpi: int) -> tuple[float, float]:
    """Доля краски в полосах ``LOOSE_SIDE_BAND_MM`` по обе стороны линейки, вдоль оси её ломаной.

    Args:
        binary: Краска белым на чёрном (``ruling.binarize``) в пикселях линейки.
        rule: Линейка (ось ломаной и толщина).
        dpi: Разрешение картинки.

    Returns:
        ``(сверху, снизу)`` для горизонтали, ``(слева, справа)`` для вертикали; 0, если полоса вне кадра.
    """
    points = np.asarray(rule.points, dtype=np.float64)
    if not rule.horizontal:
        # Вертикаль сводится к горизонтали транспонированием: вдоль оси — y, поперёк — x.
        binary, points = binary.T, points[:, ::-1]
    points = points[np.argsort(points[:, 0])]
    height, width = binary.shape[:2]
    xs = np.arange(int(points[0, 0]), int(points[-1, 0]) + 1)
    xs = xs[(xs >= 0) & (xs < width)]
    if xs.size == 0:
        return 0.0, 0.0
    ys = np.interp(xs, points[:, 0], points[:, 1])
    band = mm_to_px(LOOSE_SIDE_BAND_MM, dpi)
    # Полоса начинается сразу за штрихом линейки: полтолщины и пиксель запаса на размытие края.
    gap = rule.thickness_px / 2.0 + 1.0
    first, second = [], []
    for x, y in zip(xs, ys):
        first.append(binary[max(0, int(round(y - gap - band))) : max(0, int(round(y - gap))), x] > 0)
        second.append(binary[min(height, int(round(y + gap))) : min(height, int(round(y + gap + band))), x] > 0)
    shares = []
    for side in (np.concatenate(first), np.concatenate(second)):
        shares.append(float(side.mean()) if side.size else 0.0)
    return shares[0], shares[1]


def loose_rule_is_clean(binary: np.ndarray, rule: LooseRule, dpi: int) -> bool:
    """Линейка-сирота стоит на чистой бумаге (или длинная): годится в барьеры, а не буквы строки.

    Args:
        binary: Краска белым на чёрном в пикселях линейки.
        rule: Линейка-сирота.
        dpi: Разрешение картинки.

    Returns:
        ``True`` — линейку оставить; ``False`` — это буквы заголовка или стволы соседних строк.
    """
    points = np.asarray(rule.points, dtype=np.float64)
    along = points[:, 0] if rule.horizontal else points[:, 1]
    if float(along.max() - along.min()) >= mm_to_px(LOOSE_KEEP_MM, dpi):
        return True
    limit = LOOSE_MAX_SIDE_INK_HORIZONTAL if rule.horizontal else LOOSE_MAX_SIDE_INK_VERTICAL
    return max(side_ink(binary, rule, dpi)) <= limit


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
    close = mm_to_px(FRAGMENT_CLOSE_MM, dpi)
    fragment_h = _axis_mask(binary, mm_to_px(FRAGMENT_MM, dpi), True, close)
    fragment_v = _axis_mask(binary, mm_to_px(FRAGMENT_MM, dpi), False, close)
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


def _centre_x(segment: Segment) -> float:
    """Центр отрезка по x, px."""
    return (segment.box.x0 + segment.box.x1) / 2


def _body_step(column_xs: list[float], box: Box, pool: list[int], segments: list[Segment], dpi: int) -> list[int]:
    """Одна «строка» тела открытой таблицы под рамкой ``box``: куски вертикалей на графах ядра.

    Args:
        column_xs: Центры вертикалей ядра по x — графы, которые продолжаются вниз.
        box: Текущая рамка ядра; кусок ищется ниже её низа.
        pool: Индексы вертикалей, которые ещё можно взять (не из ядер с горизонталями).
        segments: Все линейки полосы; индексы ``pool`` — в этот список.
        dpi: Разрешение кадра.

    Returns:
        Индексы кусков этого шага; пустой список — тело кончилось.
    """
    gap = mm_to_px(OPEN_BODY_GAP_MM, dpi)
    tolerance = mm_to_px(ALIGNED_TOL_MM, dpi)
    # Кандидаты: начало не выше низа рамки (с допуском) и не дальше зазора, x — на графе ядра.
    near = [
        index
        for index in pool
        if box.y1 - tolerance <= segments[index].box.y0 <= box.y1 + gap
        and any(abs(_centre_x(segments[index]) - x) <= tolerance for x in column_xs)
    ]
    if not near:
        return []
    # Строка тела — куски, перекрывающиеся по y с самым верхним из кандидатов.
    anchor = min(near, key=lambda index: segments[index].box.y0)
    top, bottom = segments[anchor].box.y0, segments[anchor].box.y1
    step = []
    for index in near:
        piece = segments[index].box
        overlap = min(bottom, piece.y1) - max(top, piece.y0)
        if overlap >= OPEN_BODY_ROW_OVERLAP * min(bottom - top, piece.y1 - piece.y0):
            step.append(index)
    # Сколько разных граф ядра продолжено: меньше порога — это не тело таблицы.
    matched = {
        min(range(len(column_xs)), key=lambda k: abs(column_xs[k] - _centre_x(segments[index]))) for index in step
    }
    return step if len(matched) >= OPEN_BODY_MIN_COLUMNS else []


def _continues_columns(column_xs: list[float], box: Box, other: list[int], segments: list[Segment], dpi: int) -> bool:
    """Продолжает ли ядро ``other`` графы ядра над ним: начинается в пределах зазора открытого тела,
    и наверху у него вертикали хотя бы на ``OPEN_BODY_MIN_COLUMNS`` графах ядра.

    Args:
        column_xs: Центры вертикалей верхнего ядра.
        box: Рамка верхнего ядра.
        other: Индексы линеек нижнего ядра.
        segments: Все линейки полосы.
        dpi: Разрешение кадра.

    Returns:
        ``True`` — это следующий раздел той же открытой таблицы.
    """
    gap = mm_to_px(OPEN_BODY_GAP_MM, dpi)
    tolerance = mm_to_px(ALIGNED_TOL_MM, dpi)
    below = _extent([segments[i] for i in other])
    if not (box.y1 - tolerance <= below.y0 <= box.y1 + gap):
        return False
    overlap = min(box.x1, below.x1) - max(box.x0, below.x0)
    if overlap < STACK_OVERLAP * min(box.width, below.width):
        return False
    # Раздел тела начинается голыми вертикалями; своя верхняя линейка во всю ширину — это уже
    # отдельная таблица со своей шапкой. Без этого условия склеивались «Таблица 1» и «Таблица 2»
    # одна под другой (1966/06 с.52, 1967/05 с.69, 1972/07 с.73, 1969/12 с.53 — 8 полос пака).
    top_band = mm_to_px(OPEN_BODY_TOP_RULE_MM, dpi)
    if any(
        segments[i].horizontal
        and (segments[i].box.y0 + segments[i].box.y1) // 2 <= below.y0 + top_band
        and segments[i].length >= LONG_RULE_SPAN * max(1, below.width)
        for i in other
    ):
        return False
    # Внутренние вертикали, начинающиеся у верха нижнего ядра (с допуском на зазор), на графах
    # верхнего. Совпасть должна не просто пара граф, а не меньше ``ALIGNED_SHARE`` граф меньшей
    # из частей — как у шапки и тела одной таблицы в правиле стопок (``_aligned``); две разные
    # таблицы одной ширины совпадают разве что боковинами, а они сюда не входят.
    side = mm_to_px(OPEN_BODY_SIDE_MM, dpi)
    lower_xs = sorted(
        {
            _centre_x(segments[i])
            for i in other
            if not segments[i].horizontal
            and segments[i].box.y0 <= below.y0 + gap
            and below.x0 + side <= _centre_x(segments[i]) <= below.x1 - side
        }
    )
    if not lower_xs:
        return False
    matched = {k for x_low in lower_xs for k, x in enumerate(column_xs) if abs(x_low - x) <= tolerance}
    return len(matched) >= max(OPEN_BODY_MIN_COLUMNS, ALIGNED_SHARE * min(len(column_xs), len(lower_xs)))


def _inner_columns(members: list[int], box: Box, segments: list[Segment], dpi: int) -> list[float]:
    """Центры ВНУТРЕННИХ вертикалей ядра — графы без боковин рамки.

    Боковины есть у любой обрамлённой таблицы, и по ним две разные таблицы одной ширины
    «совпадали бы графами»; поэтому вертикали ближе ``OPEN_BODY_SIDE_MM`` к бокам не считаются.

    Args:
        members: Индексы линеек ядра.
        box: Рамка ядра.
        segments: Все линейки полосы.
        dpi: Разрешение кадра.

    Returns:
        Отсортированные центры внутренних вертикалей, px.
    """
    side = mm_to_px(OPEN_BODY_SIDE_MM, dpi)
    return sorted(
        {
            _centre_x(segments[i])
            for i in members
            if not segments[i].horizontal and box.x0 + side <= _centre_x(segments[i]) <= box.x1 - side
        }
    )


def extend_open_bodies(groups: list[list[int]], segments: list[Segment], dpi: int) -> list[list[int]]:
    """Дотянуть ядра-шапки вниз по кускам вертикалей, продолжающим их графы (открытая таблица).

    Args:
        groups: Ядра — списки индексов в ``segments``.
        segments: Все линейки полосы (сначала горизонтали, потом вертикали, как в :func:`cores`).
        dpi: Разрешение кадра.

    Returns:
        Новые ядра: ядра с горизонталями, дотянутые вниз; ядра из одних вертикалей, целиком
        ушедшие в тело чужой таблицы, удалены.
    """
    with_rules = [g for g in groups if any(segments[i].horizontal for i in g)]
    # Вертикали, которые можно брать в тело: не из ядер с горизонталями.
    owned = {i for g in with_rules for i in g}
    pool = [i for i, s in enumerate(segments) if not s.horizontal and i not in owned]
    taken: set[int] = set()
    extended: list[list[int]] = []
    # Сверху вниз: верхнее ядро забирает нижние разделы своей таблицы раньше, чем они сами
    # начнут расти.
    order = sorted(range(len(with_rules)), key=lambda k: _extent([segments[i] for i in with_rules[k]]).y0)
    absorbed: set[int] = set()
    for k in order:
        if k in absorbed:
            continue
        members = list(with_rules[k])
        box = _extent([segments[i] for i in members])
        column_xs = _inner_columns(members, box, segments, dpi)
        if len(column_xs) >= OPEN_BODY_MIN_COLUMNS:
            while True:
                # Шаг — сирые куски вертикалей на графах ядра…
                step = _body_step(column_xs, box, [i for i in pool if i not in taken], segments, dpi)
                if step:
                    taken.update(step)
                    members += step
                    box = _extent([segments[i] for i in members])
                    continue
                # …или следующий раздел той же таблицы, ставший своим ядром (у него есть горизонталь:
                # двойная линейка над «Итого»).
                follower = next(
                    (
                        j
                        for j in order
                        if j != k
                        and j not in absorbed
                        and _continues_columns(column_xs, box, with_rules[j], segments, dpi)
                    ),
                    None,
                )
                if follower is None:
                    break
                absorbed.add(follower)
                members += with_rules[follower]
                box = _extent([segments[i] for i in members])
        extended.append(members)
    # Ядра из одних вертикалей: остаток без ушедших в тело кусков, если в нём ещё ≥ 2 линеек.
    for group in groups:
        if any(segments[i].horizontal for i in group):
            continue
        rest = [i for i in group if i not in taken]
        if len(rest) >= 2:
            extended.append(rest)
    return extended


def _local_components(members: list[int], segments: list[Segment], dpi: int) -> list[list[int]]:
    """Связные компоненты набора линеек по тем же рёбрам, что у :func:`cores`.

    Args:
        members: Индексы линеек в ``segments``.
        segments: Все линейки полосы.
        dpi: Разрешение кадра.

    Returns:
        Компоненты — списки индексов в ``segments``.
    """
    tolerance = mm_to_px(CROSS_TOL_MM, dpi)
    edges: list[tuple[int, int]] = []
    horizontal = [k for k, i in enumerate(members) if segments[i].horizontal]
    vertical = [k for k, i in enumerate(members) if not segments[i].horizontal]
    # Пересечения горизонталей с вертикалями.
    for a in horizontal:
        for b in vertical:
            if crosses(segments[members[b]], segments[members[a]], tolerance):
                edges.append((a, b))
    # Продолжения вдоль оси (порванная линейка).
    for axis in (horizontal, vertical):
        as_fragments = [_as_fragment(segments[members[k]]) for k in axis]
        edges += [(axis[a], axis[b]) for a, b in _chain_edges(as_fragments, dpi, CHAIN_GAP_MM)]
    return [[members[k] for k in comp] for comp in components(len(members), edges)]


def _stacked_apart(first: Box, second: Box) -> bool:
    """Стоят ли две рамки одна над другой без перекрытия по y и с общей шириной (``STACK_OVERLAP``)."""
    overlap_x = min(first.x1, second.x1) - max(first.x0, second.x0)
    apart_y = first.y1 <= second.y0 or second.y1 <= first.y0
    return apart_y and overlap_x >= STACK_OVERLAP * min(first.width, second.width)


def _clipped_to(bridge: Segment, box: Box) -> "Segment | None":
    """Кусок моста в пределах рамки части по его оси; ``None`` — мост её не касается."""
    b = bridge.box
    if bridge.horizontal:
        x0, x1 = max(b.x0, box.x0), min(b.x1, box.x1)
        return Segment(Box(x0, b.y0, x1, b.y1), True, bridge.angle_deg) if x1 > x0 else None
    y0, y1 = max(b.y0, box.y0), min(b.y1, box.y1)
    return Segment(Box(b.x0, y0, b.x1, y1), False, bridge.angle_deg) if y1 > y0 else None


def split_bridged(group: list[int], segments: list[Segment], dpi: int) -> list[list[int]]:
    """Разрезать ядро по линейке-мосту, если без неё куски — отдельные таблицы по правилу стопок.

    Каждой части достаётся копия моста, обрезанная по её рамке: у обеих таблиц остаётся своя
    боковая линейка. Копии дописываются в конец ``segments`` (список расширяется на месте —
    это список самой :func:`cores`, наружу он не отдаётся).

    Args:
        group: Индексы линеек ядра в ``segments``.
        segments: Все линейки полосы; сюда же дописываются обрезки моста.
        dpi: Разрешение кадра.

    Returns:
        Ядра после разрезов (одно — если резать нечего).
    """
    if len(group) > BRIDGE_MAX_RULES:
        return [group]
    # Кандидаты в мосты: самые длинные линейки каждой оси.
    candidates: list[int] = []
    for horizontal in (True, False):
        axis = sorted((i for i in group if segments[i].horizontal == horizontal), key=lambda i: -segments[i].length)
        candidates += axis[:BRIDGE_CANDIDATES_PER_AXIS]
    for bridge in candidates:
        rest = [i for i in group if i != bridge]
        parts = [p for p in _local_components(rest, segments, dpi) if len(p) >= 2]
        if len(parts) < 2:
            continue
        boxes = [_extent([segments[i] for i in p]) for p in parts]
        # Режем только две и больше ПОЛНЫХ таблиц одна над другой. Без этого условия мостом
        # становилась линия-стрелка блок-схемы: схема 1970/11 с.82 распадалась на коробки, и ни
        # одна из них уже не была схемой — находка пропадала целиком.
        complete = all(
            _long_rules([segments[i] for i in p], box) >= COMPLETE_LONG_RULES for p, box in zip(parts, boxes)
        )
        stacked = all(_stacked_apart(boxes[a], boxes[b]) for a in range(len(parts)) for b in range(a + 1, len(parts)))
        if not (complete and stacked):
            continue
        # Если хоть одна пара частей слилась бы и без моста — это одна таблица, мост не мост.
        glued = any(
            _should_merge([segments[i] for i in parts[a]], boxes[a], [segments[i] for i in parts[b]], boxes[b], [], dpi)
            for a in range(len(parts))
            for b in range(a + 1, len(parts))
        )
        if glued:
            continue
        result: list[list[int]] = []
        for part, box in zip(parts, boxes):
            piece = _clipped_to(segments[bridge], box)
            if piece is not None:
                segments.append(piece)
                part = part + [len(segments) - 1]
            result += split_bridged(part, segments, dpi)
        return result
    return [group]


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
    groups = [g for g in components(len(segments), edges) if len(g) >= 2]
    # Две полные таблицы, связанные одной длинной линейкой, — два ядра (см. ``split_bridged``).
    groups = [part for g in groups for part in split_bridged(g, segments, dpi)]
    in_core = {i for g in groups for i in g}
    orphans = [segments[i] for i in range(len(segments)) if i not in in_core]

    # Слияние стопок — по рамкам ядер, до сходимости.
    groups = _merge_stacks(groups, segments, orphans, dpi)
    boxes = [_extent([segments[i] for i in g]) for g in groups]

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
    # Открытые таблицы: шапка забирает куски вертикалей тела, продолжающие её графы.
    groups = extend_open_bodies(groups, segments, dpi)
    # Ещё одно слияние стопок: открытое тело, дотянутое вниз, встаёт вплотную к своей итоговой
    # строке под двойной линейкой («Итого в среднем…», 1973/08 с.20) — у той свои вертикали и своё ядро.
    groups = _merge_stacks(groups, segments, orphans, dpi)
    return [[segments[i] for i in g] for g in groups]


def _merge_stacks(
    groups: list[list[int]], segments: list[Segment], orphans: list[Segment], dpi: int
) -> list[list[int]]:
    """Слить ядра, стоящие стопкой, по правилу :func:`_should_merge`, до сходимости.

    Args:
        groups: Ядра — списки индексов в ``segments``.
        segments: Все линейки полосы.
        orphans: Линейки вне ядер — мостики между частями бланка.
        dpi: Разрешение кадра.

    Returns:
        Новый список ядер; вход не меняется.
    """
    groups = [list(g) for g in groups]
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
    return groups


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
