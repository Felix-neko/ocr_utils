"""Сцепка кусков строк по ЗОНАМ ПОИСКА: каждый кусок ведёт свою ось и смотрит вдоль неё.

Прежний ход растил цепочку жадно: от последнего куска брался ближайший справа, у которого
ордината не расходится с предсказанием по последним звеньям. На искажённой бумаге сгустки
соседних строк чередуются по x, каждое звено законно (излом 1–2°), а ошибка копится — за пять
звеньев ось уходит на соседнюю строку, и поправить это порогом нельзя: порог локальный, а беда
накопительная.

Здесь копить нечего. У куска есть СВОЯ ось по буквам внутри него; от её концов выпускаются
«колбаски» — зоны поиска; соединяются только те куски, чьи зоны встретились, и притом встретились
ПО ХОДУ обеих осей (угол подхода в допуске). Каждое соединение проверяется от исходного куска, а
не от конца растущей цепочки, поэтому ошибка не наследуется.

Ход в два круга. Первый: длинные куски (у них своя ось надёжна) плюс первичные зоны коротких —
короткий кусок принимает соединение без проверки угла, потому что своего направления у «и», «в»
или точки нет. Второй: у коротких появляются вторичные зоны влево и вправо по наклону соседних
длинных кусков, и короткие сцепляются между собой — так собирается строка, набранная вразрядку,
и ряд точек отточия.

Все пороги — в долях высоты строчной буквы (``x_h``) и межстрочного шага, поэтому крупный набор
работает теми же числами.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ocr_utils.curved_layout import WORK_DPI
from ocr_utils.curved_layout.capsules import Capsule, contact_of
from ocr_utils.curved_layout.pieces import (
    AXIS_MIN_LETTERS,
    Piece,
    axis_residual,
    chord_slope,
    merged,
    page_x_height,
    pieces_of,
    pitch_of,
)

# Длина зоны длинного куска в высотах строчной буквы. 2.5 икса ≈ 25 px на корпусе пака-1: больше
# межсловного пробела (медиана зазора до соседа 12 px, p75 — 22 px), но заметно меньше прежнего
# разрыва сцепки (2.5 высоты БОКСА = 33–45 px). Замер по десяти трудным полосам: при 2.5 сближений
# 5, при 3.0 — 6, при 3.5 — 6; половинок соответственно 26, 22, 20. Выбрано 2.5 по решению
# пользователя «лучше разорвать, чем склеить»: разорванную строку соберёт в один ряд блока
# ``blocks.rows_of``, а перескок исправить нечем.
LONG_REACH_XH = 2.5
# Зона длинного куска — КОНУС: у самого куска радиус 0.6 икса (6 px), у дальнего конца 1.1 икса
# (11 px). У куска направление известно точно, а дальше оценка наклона шумит, и колбаска
# постоянной толщины проходила мимо соседа: замер на 1973/08 с.85 — 23 пары соседних слов одной
# строки не срослись именно так. Соседнюю строку конус всё равно не достаёт: на дальнем конце
# 11 + 6 = 17 px против шага 22 px.
LONG_RADIUS_XH = 0.6
LONG_RADIUS_END_XH = 1.1
# Но не шире этой доли межстрочного шага: раскрытие конуса задано в высотах БУКВЫ, а опасность
# дотянуться до соседней строки задаётся ШАГОМ. На плотном наборе (1975/05 с.97: шаг 18.5 px при
# тех же буквах) конус с зоной соседа давали 17 px против шага 18.5 — и ось ныряла в соседнюю
# строку. С этим потолком дальний конец плюс зона соседа не превышают двух третей шага.
CONE_MAX_PITCH_SHARE = 0.35
# Первичная зона короткого куска: вверх шире, чем вниз. Точка и запятая сидят на базовой линии,
# и симметричная зона провисала бы к соседней строке.
SHORT_UP_XH = 0.75
SHORT_DOWN_XH = 0.25
# Вторичные зоны коротких кусков (второй круг): длина в межстрочных шагах и радиус в иксах.
# 0.8 шага ≈ 18 px — ровно шаг точек отточия и заметно меньше зазора между графами таблицы.
SECOND_REACH_PITCHES = 0.8
SECOND_RADIUS_XH = 0.5
# Допуск на угол подхода: под каким углом зона видит точку встречи.
ANGLE_LIMIT_DEG = 20.0
# Рычаг угла: вершина отодвигается назад по оси на столько высот строчной. При коротком зазоре
# угол от самого конца куска бессмыслен — см. ``capsules.angle_at``.
ANGLE_LEVER_XH = 1.5
# Кругов слияния: за круг сливаются ВСЕ взаимно ближайшие пары, то есть длина цепочки удваивается.
MAX_ROUNDS = 10
# Второй круг не делается, если длинных кусков меньше: на странице-схеме подписи разбросаны, и
# сцеплять их вторичными зонами бессмысленно (тот же порог, что был у поля хода строк).
MIN_LONG_PIECES = 12
# Столько точек подряд — уже отточие: такая цепочка не приращивает справа ничего, кроме отточия.
LEADER_RUN_MIN = 4
# Насколько наклон соединения может отличаться от МЕСТНОГО наклона строк (медиана по соседним
# длинным кускам). Абсолютный предел тут не годится: в изогнутом углу полосы соседние слова
# одной строки законно стоят с перепадом в 10° (замер на 1973/08 с.85: 16 верных пар отбито
# порогом в 6°), а в середине полосы 6° — уже перескок. Предел нужен против ДИАГОНАЛЬНЫХ слияний
# через три строки: такой кусок ложится на прямую отлично, и предохранитель по остатку оси его
# не ловит (1967/10 с.63 geo: кусок 108 px длиной падал на 44 px — 22°).
MERGE_SLOPE_TOLERANCE_DEG = 8.0
# Предохранитель слияния: остаток якорей от оси после слияния. Перескок на соседнюю строку даёт
# ступеньку в полшага (11 px на корпусе пака-1), и остаток подскакивает сразу.
AXIS_MAX_RESID_XH = 0.35
# Сколько ближайших длинных кусков опрашивается ради наклона короткого и сколько их нужно.
SLOPE_NEIGHBOURS = 7
SLOPE_MIN_NEIGHBOURS = 3
# Наклон строки на скане не круче этого: медиана по соседям зажимается.
SLOPE_LIMIT_DEG = 5.0


@dataclass(frozen=True)
class Zone:
    """Зона поиска куска: капсула, чей это кусок, с какой стороны и проверять ли угол подхода."""

    piece: int
    side: int  # −1 зона смотрит влево, +1 вправо, 0 — первичная зона короткого куска
    capsule: Capsule
    checked: bool  # False — угол подхода не проверяется (у куска нет своего направления)


@dataclass(frozen=True)
class Link:
    """Вероятное соединение двух кусков: кто слева, кто справа и как далеко точка встречи."""

    left: int
    right: int
    reach_left: float  # как далеко точка встречи от начала зоны ЛЕВОГО куска
    reach_right: float
    gap: float


def long_zones(piece: Piece, index: int, pitch: float = 0.0) -> list[Zone]:
    """Зоны длинного куска: от концов его оси по касательной наружу.

    Args:
        piece: Кусок.
        index: Его номер в списке кусков.
        pitch: Межстрочный шаг страницы; им ограничивается раскрытие конуса, чтобы зона не
            дотягивалась до соседней строки. Ноль — ограничения нет.
    """
    reach = LONG_REACH_XH * piece.x_h
    radius = LONG_RADIUS_XH * piece.x_h
    radius_end = LONG_RADIUS_END_XH * piece.x_h
    if pitch > 0:
        radius_end = min(radius_end, CONE_MAX_PITCH_SHARE * pitch)
    radius_end = max(radius_end, radius)
    out: list[Zone] = []
    for at_start, side in ((True, -1), (False, 1)):
        x, y, dx, dy = piece.tangent(at_start)
        out.append(
            Zone(
                piece=index,
                side=side,
                capsule=Capsule(ax=x, ay=y, bx=x + dx * reach, by=y + dy * reach, radius=radius, radius_end=radius_end),
                checked=True,
            )
        )
    return out


def primary_zone(piece: Piece, index: int) -> Zone:
    """Первичная зона короткого куска: его собственный охват, раздутый вверх сильнее, чем вниз.

    Угол подхода у такой зоны не проверяется: у куска из одной-двух букв своего направления нет,
    и требовать от него «смотреть навстречу» нечего.
    """
    up = SHORT_UP_XH * piece.x_h
    down = SHORT_DOWN_XH * piece.x_h
    # Ось берётся по якорям куска (у низкой метки якорь уже поднят к уровню строки), а не по
    # середине бокса: иначе зона точки снова уедет вниз.
    y = piece.cy
    shift = (up - down) / 2.0
    return Zone(
        piece=index,
        side=0,
        capsule=Capsule(ax=piece.x0, ay=y - shift, bx=piece.x1, by=y - shift, radius=(up + down) / 2.0),
        checked=False,
    )


def secondary_zones(piece: Piece, index: int, pitch: float, slope: float) -> list[Zone]:
    """Вторичные зоны короткого куска: влево и вправо по наклону соседних длинных кусков."""
    reach = SECOND_REACH_PITCHES * pitch
    radius = SECOND_RADIUS_XH * piece.x_h
    radius_end = max(radius, min(LONG_RADIUS_END_XH * piece.x_h, CONE_MAX_PITCH_SHARE * pitch))
    length = float(np.hypot(1.0, slope))
    dx, dy = 1.0 / length, slope / length
    y = piece.cy
    return [
        Zone(
            piece=index,
            side=-1,
            capsule=Capsule(
                ax=piece.x0, ay=y, bx=piece.x0 - dx * reach, by=y - dy * reach, radius=radius, radius_end=radius_end
            ),
            checked=True,
        ),
        Zone(
            piece=index,
            side=1,
            capsule=Capsule(
                ax=piece.x1, ay=y, bx=piece.x1 + dx * reach, by=y + dy * reach, radius=radius, radius_end=radius_end
            ),
            checked=True,
        ),
    ]


def neighbour_slopes(pieces: list[Piece]) -> np.ndarray:
    """Наклон строки в месте каждого куска: медиана по ближайшим ДЛИННЫМ кускам.

    Заменяет прежнее поле хода строк там, где оно только и было нужно, — коротким кускам, у
    которых своего направления нет. Медиана зажимается ``SLOPE_LIMIT_DEG``: строка на скане не
    наклонена круче.

    Args:
        pieces: Куски страницы.

    Returns:
        Тангенс наклона для каждого куска; ноль, если длинных соседей не набралось.
    """
    out = np.zeros(len(pieces), dtype=np.float64)
    long_index = [index for index, piece in enumerate(pieces) if piece.long and piece.letters >= AXIS_MIN_LETTERS]
    if len(long_index) < SLOPE_MIN_NEIGHBOURS:
        return out
    centres = np.array([[(pieces[i].x0 + pieces[i].x1) / 2.0, pieces[i].cy] for i in long_index])
    # Наклон берётся по ХОРДЕ куска: конец шумит, а хорда устойчива.
    slopes = np.array([chord_slope(pieces[i]) for i in long_index])
    tree = cKDTree(centres)
    limit = float(np.tan(np.radians(SLOPE_LIMIT_DEG)))
    for index, piece in enumerate(pieces):
        point = np.array([(piece.x0 + piece.x1) / 2.0, piece.cy])
        count = min(SLOPE_NEIGHBOURS, len(long_index))
        _, own = tree.query(point, k=count)
        own = np.atleast_1d(own)
        own = own[own < len(long_index)]
        if own.size < SLOPE_MIN_NEIGHBOURS:
            continue
        out[index] = float(np.clip(np.median(slopes[own]), -limit, limit))
    return out


def zones_of(pieces: list[Piece], pitch: float, secondary: bool, slopes: np.ndarray | None = None) -> list[Zone]:
    """Зоны поиска всех кусков: длинным — по концам оси, коротким — первичная (и вторичные)."""
    slopes = neighbour_slopes(pieces) if slopes is None else slopes
    out: list[Zone] = []
    for index, piece in enumerate(pieces):
        if piece.long and piece.letters >= AXIS_MIN_LETTERS:
            out.extend(long_zones(piece, index, pitch))
            continue
        out.append(primary_zone(piece, index))
        # Вторичные зоны нужны и куску из ОДНОЙ буквы: строка вразрядку («М а р к с» в сноске
        # 1973/08 с.85) вся состоит из таких, и без них она не собирается вовсе. А вот НИЗКИМ
        # МЕТКАМ (точка, запятая) вторичные зоны не даются: ряд точек отточия сцеплялся бы сам с
        # собой и тянул бы за собой соседние графы — замер: сближений 7 → 17, блоков 122 → 143,
        # покрытие краски 99.1 → 98.7 %. В строку отточия входят своим ходом
        # (``segment.extend_with_leaders``).
        if secondary and not bool(piece.marks.all()):
            out.extend(secondary_zones(piece, index, pitch, float(slopes[index])))
    return out


def _pairs(zones: list[Zone]) -> set[tuple[int, int]]:
    """Пары зон, которые вообще могут соприкоснуться: предварительный отбор по ``cKDTree``."""
    if len(zones) < 2:
        return set()
    centres = np.array([[(z.capsule.ax + z.capsule.bx) / 2.0, (z.capsule.ay + z.capsule.by) / 2.0] for z in zones])
    reaches = np.array([z.capsule.length / 2.0 + z.capsule.radius for z in zones])
    tree = cKDTree(centres)
    return tree.query_pairs(r=float(2.0 * reaches.max()))


def links_of(
    zones: list[Zone],
    pieces: list[Piece],
    scale,
    separators: list[tuple[int, int, int, int]],
    rules: list | None,
    ink300: np.ndarray,
    k: float,
    slopes: np.ndarray | None = None,
) -> list[Link]:
    """Вероятные соединения: зоны встретились, углы подхода в допуске, запрета между кусками нет.

    Args:
        zones: Зоны поиска всех кусков.
        pieces: Сами куски.
        scale: Размеры масштаба набора (нужно отношение высот соседних кусков).
        separators: Межколонники и вертикальные линейки — через них строка не собирается.
        rules: Сплошные горизонтальные черты (ребро ячейки таблицы).
        ink300: Краска рендера (нужна проверке межколонника).
        k: Во сколько раз рендер крупнее рабочей копии.
        slopes: Местный наклон строк в месте каждого куска (медиана по соседям). С ним сверяется
            наклон соединения; ``None`` — сверка с горизонталью.

    Returns:
        Соединения без дубликатов: для каждой пары кусков — самое близкое.
    """
    from ocr_utils.curved_layout.segment import _crosses

    best: dict[tuple[int, int], Link] = {}
    for first, second in _pairs(zones):
        one, other = zones[first], zones[second]
        if one.piece == other.piece:
            continue
        lever_one = ANGLE_LEVER_XH * pieces[one.piece].x_h
        lever_other = ANGLE_LEVER_XH * pieces[other.piece].x_h
        contact = contact_of(one.capsule, other.capsule, lever_one, lever_other)
        if contact is None:
            continue
        if one.checked and contact.angle_first > ANGLE_LIMIT_DEG:
            continue
        if other.checked and contact.angle_second > ANGLE_LIMIT_DEG:
            continue
        left_index, right_index = one.piece, other.piece
        left_zone, right_zone = one, other
        if pieces[left_index].x0 > pieces[right_index].x0:
            left_index, right_index = right_index, left_index
            left_zone, right_zone = other, one
        # Зона, смотрящая не в ту сторону, соединения не даёт: правый кусок ловится ПРАВОЙ зоной.
        if left_zone.side == -1 or right_zone.side == 1:
            continue
        left_piece, right_piece = pieces[left_index], pieces[right_index]
        local = 0.0 if slopes is None else float((slopes[left_index] + slopes[right_index]) / 2.0)
        if not _allowed(left_piece, right_piece, scale, separators, rules, ink300, k, _crosses, local):
            continue
        reach_left = contact.reach_first if left_zone is one else contact.reach_second
        reach_right = contact.reach_second if left_zone is one else contact.reach_first
        key = (left_index, right_index)
        link = Link(left=left_index, right=right_index, reach_left=reach_left, reach_right=reach_right, gap=contact.gap)
        if key not in best or link.reach_left < best[key].reach_left:
            best[key] = link
    return list(best.values())


def _allowed(left: Piece, right: Piece, scale, separators, rules, ink300, k, crosses, local_slope: float = 0.0) -> bool:
    """Можно ли вообще соединять эти два куска: наклон, запреты, кегль, отточия.

    Наклон соединения сверяется с МЕСТНЫМ наклоном строк ``local_slope``, а не с горизонталью:
    в изогнутом углу полосы соседние слова одной строки законно стоят с перепадом в десять
    градусов, а в середине полосы столько же — уже перескок.
    """
    heights = sorted((max(left.x_h, 1e-6), max(right.x_h, 1e-6)))
    if heights[1] > scale.link_height_ratio * heights[0]:
        return False
    # Куски одной строки не стоят друг над другом: наклон соединения не должен уходить от
    # местного наклона строк дальше допуска.
    run = (right.x0 + right.x1) / 2.0 - (left.x0 + left.x1) / 2.0
    rise = right.cy - left.cy
    expected = local_slope * run
    if abs(rise - expected) > np.tan(np.radians(MERGE_SLOPE_TOLERANCE_DEG)) * max(abs(run), 1e-6):
        return False
    cy = (left.cy + right.cy) / 2.0
    height = max(left.height, right.height)
    if crosses(separators, left.x1, right.x0, cy, height, ink300, k):
        return False
    if rules and _rule_between(rules, left.x1, right.x0, left.cy, right.cy):
        return False
    # Цепочка, в которой уже есть отточие, справа приращивает только другое отточие: иначе текст
    # левой графы таблицы сошьётся с числом правой через ряд точек (1971/10 с.93).
    if left.leader_dots >= LEADER_RUN_MIN and right.leader_dots == 0:
        return False
    return True


def _rule_between(rules: list, left: float, right: float, y_left: float, y_right: float) -> bool:
    """Лежит ли между кусками сплошная черта — ребро ячейки таблицы."""
    low, high = sorted((y_left, y_right))
    x0, x1 = sorted((left, right))
    return any(low <= rule.cy <= high and rule.x0 <= x1 and rule.x1 >= x0 for rule in rules)


def accept(links: list[Link], count: int) -> list[Link]:
    """Взаимно ближайшие пары: с каждой стороны куска не больше одного соединения.

    Со своей стороны кусок выбирает то соединение, чья точка встречи ближе к началу его зоны.
    Принимается пара, только если выбор ВЗАИМЕН: иначе получается несогласованная сцепка, где A
    тянется к B, а B — к C.

    Args:
        links: Вероятные соединения.
        count: Сколько всего кусков.

    Returns:
        Принятые соединения.
    """
    best_right: dict[int, Link] = {}
    best_left: dict[int, Link] = {}
    for link in links:
        current = best_right.get(link.left)
        if current is None or link.reach_left < current.reach_left:
            best_right[link.left] = link
        current = best_left.get(link.right)
        if current is None or link.reach_right < current.reach_right:
            best_left[link.right] = link
    out: list[Link] = []
    for index in range(count):
        link = best_right.get(index)
        if link is not None and best_left.get(link.right) is link:
            out.append(link)
    return out


def _guard(left: Piece, right: Piece, pitch: float) -> bool:
    """Держит ли слияние проверку по собственной оси: остаток якорей от неё.

    Это абсолютный тормоз вместо прежнего поля хода строк. Если слияние свело в один кусок две
    строки, ось по всем якорям перестаёт их описывать: остаток подскакивает с единиц пикселей до
    полушага.

    Прогиб оси от хорды в проверку НЕ входит: у честно изогнутой строки он большой по
    определению, ради таких строк весь подпакет и сделан. Замер на 1973/08 с.85: верное слияние
    двадцати восьми букв дало остаток 0.73 px при пороге 3.5 и прогиб 12.6 px при пороге 11.1 —
    прогиб отбивал правильное слияние, остаток его пропускал.
    """
    candidate = merged(left, right)
    if candidate.letters < 3:
        return True
    return axis_residual(candidate) <= AXIS_MAX_RESID_XH * max(candidate.x_h, 1e-6)


def _round(
    pieces: list[Piece],
    pitch: float,
    secondary: bool,
    scale,
    separators: list[tuple[int, int, int, int]],
    rules: list | None,
    ink300: np.ndarray,
    k: float,
) -> tuple[list[Piece], int]:
    """Один круг: построить зоны, отобрать взаимно ближайшие пары, слить их."""
    slopes = neighbour_slopes(pieces)
    zones = zones_of(pieces, pitch, secondary, slopes)
    links = accept(links_of(zones, pieces, scale, separators, rules, ink300, k, slopes), len(pieces))
    if not links:
        return pieces, 0
    taken = [False] * len(pieces)
    out: list[Piece] = []
    joined = 0
    for link in links:
        if taken[link.left] or taken[link.right]:
            continue
        if not _guard(pieces[link.left], pieces[link.right], pitch):
            continue
        taken[link.left] = taken[link.right] = True
        out.append(merged(pieces[link.left], pieces[link.right]))
        joined += 1
    out.extend(piece for index, piece in enumerate(pieces) if not taken[index])
    out.sort(key=lambda piece: piece.x0)
    return out, joined


def link_by_zones(
    stats: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    separators: list[tuple[int, int, int, int]],
    scale,
    dpi: float,
    ink300: np.ndarray,
    k: float,
    leaders: list | None = None,
    rules: list | None = None,
) -> list[list[int]]:
    """Сцепить куски строк по зонам поиска; отдаёт наборы индексов сгустков, как прежняя сцепка.

    Args:
        stats: Сгустки после смыкания RLSA.
        labels: Карта компонент после смыкания.
        mask: Маска глифов до смыкания (по ней куски разбираются на буквы).
        separators: Межколонники и вертикальные линейки.
        scale: Размеры масштаба набора.
        dpi: Разрешение рабочей копии.
        ink300: Краска рендера.
        k: Во сколько раз рендер крупнее рабочей копии.
        leaders: Отточия страницы.
        rules: Сплошные горизонтальные черты.

    Returns:
        Наборы индексов сгустков — по одному на строку.
    """
    from ocr_utils.curved_layout.segment import candidates_of

    candidates = candidates_of(stats, scale, dpi)
    pieces = pieces_of(stats, labels, mask, candidates, dpi, leaders)
    if not pieces:
        return []
    pitch = pitch_of(pieces, dpi)
    long_count = sum(1 for piece in pieces if piece.long)
    # Первый круг: длинные куски и первичные зоны коротких. Второй: у коротких появляются
    # вторичные зоны, и они сцепляются между собой — так собирается строка вразрядку и ряд точек.
    for secondary in (False, True):
        if secondary and long_count < MIN_LONG_PIECES:
            break
        for _ in range(MAX_ROUNDS):
            pieces, joined = _round(pieces, pitch, secondary, scale, separators, rules, ink300, k)
            if not joined:
                break
    return [list(piece.blobs) for piece in pieces]


__all__ = ["Link", "Zone", "accept", "link_by_zones", "links_of", "long_zones", "primary_zone", "zones_of"]
