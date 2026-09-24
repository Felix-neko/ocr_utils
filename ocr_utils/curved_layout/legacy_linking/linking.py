"""Прежняя сцепка кусков строк: жадная цепочка с ограничителем по полю хода строк.

Цепочка растёт слева направо: от последнего куска берётся ближайший справа, у которого ордината
не расходится с ПРЕДСКАЗАНИЕМ по последним звеньям. На сильно искажённой бумаге сгустки соседних
строк чередуются по x, каждое звено законно (излом 1–2°), а ошибка копится — за пять звеньев ось
уходит на соседнюю строку. Поле хода строк (:mod:`.baselines`) как ограничитель помогает лишь
отчасти: оно правит общий ход полосы, а местный изгиб в углу не отслеживает.

Ход оставлен рабочим: ``--linking greedy``. Замена — :mod:`ocr_utils.curved_layout.zones`.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from ocr_utils.curved_layout.legacy_linking.baselines import LineField
from ocr_utils.curved_layout.segment import Rule, Scale, X_OVERLAP_SHARE, _crosses, candidates_of


# Второй проход: насколько выпрямленная ордината куска может уйти от ординаты цепочки — в долях
# межстрочного шага и в долях полувысоты самого куска (заглавная буква и слово с выносными
# элементами законно смещены от центра строки).
FIELD_GUARD_PITCHES = 0.35
FIELD_GUARD_SPAN = 0.5
# Сшивание обрывков: допуск по выпрямленной ординате между цепочками одной строки.
FIELD_MERGE_PITCHES = 0.3
# По скольким первым кускам берётся уровень цепочки.
FIELD_LEVEL_HEAD = 3
# Доля более короткой цепочки, на которую двум цепочкам одной строки позволено перекрываться.
MERGE_OVERLAP_SHARE = 0.5
# Наклон сцепки: допуск по вертикали растёт с расстоянием между кусками, но не быстрее этого
# угла (строки на скане наклонены на 2–3°, изгиб добавляет локально ещё пару градусов).
MAX_LINK_SLOPE_DEG = 4.0
SLOPE_BASE_PX = 3.0
MIN_SLOPE_SPAN_PX = 60.0
# Ход цепочки предсказывается прямой по её последним звеньям: сколько звеньев нужно, чтобы
# прямую строить, и по скольким последним она строится.
LINK_FIT_MIN = 3
LINK_FIT_SPAN = 6


def _predicted_cy(stats: np.ndarray, chain: list[int], x: float, fallback: float) -> float:
    """Где цепочка ждёт следующий кусок по вертикали: прямая по её последним звеньям.

    Args:
        stats: Статистика компонент (``connectedComponentsWithStats``).
        chain: Индексы уже собранных кусков строки.
        x: Абсцисса центра кандидата.
        fallback: Что вернуть, если звеньев мало (ордината последнего куска).

    Returns:
        Ожидаемая ордината центра строки в точке ``x``.
    """
    if len(chain) < LINK_FIT_MIN:
        return fallback
    own = stats[chain[-LINK_FIT_SPAN:]]
    xs = own[:, cv2.CC_STAT_LEFT] + own[:, cv2.CC_STAT_WIDTH] / 2.0
    ys = own[:, cv2.CC_STAT_TOP] + own[:, cv2.CC_STAT_HEIGHT] / 2.0
    if float(xs.max() - xs.min()) < 1.0:
        return fallback
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope * x + intercept)


def link_spans(
    stats: np.ndarray,
    separators: list[tuple[int, int, int, int]],
    scale: Scale,
    dpi: float,
    ink300: np.ndarray,
    k: float,
    field: LineField | None = None,
) -> list[list[int]]:
    """Сцепить куски одной строки в цепочки (индексы сгустков) по размерам ``scale``.

    Повторяет ``line_fit.link_spans``, но пороги приходят из масштаба: готовый отбрасывает всё
    выше 45 px, а строка наклонного логотипа (буквы 22 px, разбег по высоте 30 px) выше — и
    заголовок терял ось целиком (1973/06 с.65).

    Args:
        stats: Сгустки страницы от ``connectedComponentsWithStats``.
        separators: Живые межколонники (через них строка не сцепляется).
        scale: Размеры масштаба набора (корпус или крупный).
        dpi: Разрешение рабочей копии.
        ink300: Краска копии ``RENDER_DPI`` (нужна проверке межколонника).
        k: Во сколько раз рендер крупнее рабочей копии.
        field: Поле хода строк второго прохода. Если задано, кусок не берётся в цепочку, когда
            его ВЫПРЯМЛЕННАЯ ордината уходит от ординаты цепочки дальше ``FIELD_GUARD_PITCHES``
            межстрочного шага: локальные пороги по наклону и излому перескок на соседнюю строку
            не ловят (изломы внутри цепочек — медиана 1.6°, максимум 10°, то есть неотличимы от
            нормальных), а выпрямленная ордината даёт абсолютный отсчёт «это уже другая строка».

    Returns:
        Цепочки индексов сгустков, куски внутри упорядочены слева направо.
    """
    candidates = candidates_of(stats, scale, dpi)
    # Выпрямленные ординаты центров всех сгустков считаются один раз на страницу.
    straight = None
    if field is not None and field.pitch > 0:
        boxes = stats[:, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]].astype(np.float64)
        straight = field.straighten(boxes[:, 0] + boxes[:, 2] / 2.0, boxes[:, 1] + boxes[:, 3] / 2.0)
    guard = FIELD_GUARD_PITCHES * field.pitch if field is not None else 0.0
    used: set[int] = set()
    spans: list[list[int]] = []
    for head in candidates:
        if head in used:
            continue
        chain = [head]
        used.add(head)
        while True:
            x, y, w, h = (int(stats[chain[-1], k]) for k in (0, 1, 2, 3))
            cy, right = y + h / 2.0, x + w
            best = None
            for other in candidates:
                if other in used:
                    continue
                ox, oy, ow, oh = (int(stats[other, k]) for k in (0, 1, 2, 3))
                # Кандидату позволено начинаться левее конца цепочки на долю своей ширины: у
                # набора вразрядку и у засечек куски перекрываются по x, и строгое «строго правее»
                # рвало строку сноски «М а р к с К. и Энгельс Ф.» (1973/08 с.85).
                if ox < right - max(2.0, X_OVERLAP_SHARE * ow) or ox - right > scale.link_gap_heights * max(h, oh):
                    continue
                # Сверяемся не с последним куском, а с ПРЕДСКАЗАНИЕМ по ходу самой цепочки: иначе
                # цепочка «дрейфует» — каждый следующий кусок ниже предыдущего на треть высоты, и
                # через четыре куска строка уезжает на соседнюю (1970/02 с.90, низ левой колонки:
                # оси шли с наклоном 3–7° при норме ±1°).
                # Наклон ЦЕПОЧКИ ограничен: допуск «0.35 высоты» на каждом шаге складывается, и
                # через несколько кусков ось уезжает на соседнюю строку (1973/08 с.85, низ левой
                # колонки: наклоны −10.7°…+8.8° при норме ±2°). Считается от начала цепочки,
                # поэтому короткие куски (предлог, обрывок слова) сцепляться не мешают.
                head_box = stats[chain[0]]
                run = (ox + ow / 2.0) - (int(head_box[0]) + int(head_box[2]) / 2.0)
                if run >= MIN_SLOPE_SPAN_PX:
                    rise = oy + oh / 2.0 - (int(head_box[1]) + int(head_box[3]) / 2.0)
                    # Наклон цепочки ограничен абсолютным порогом: строки на скане наклонены на
                    # 2–3°, изгиб добавляет локально ещё пару градусов. Сверка с наклоном СОСЕДНИХ
                    # строк осталась нереализованной — параметр ``slopes`` никогда не передавался.
                    limit = math.tan(math.radians(MAX_LINK_SLOPE_DEG)) * run + SLOPE_BASE_PX
                    if abs(rise) > limit:
                        continue
                predicted = _predicted_cy(stats, chain, ox + ow / 2.0, cy)
                if abs(oy + oh / 2.0 - predicted) > scale.link_dy_heights * max(h, oh):
                    continue
                if straight is not None:
                    # Уровень берётся по НАЧАЛУ цепочки, а не по всем её кускам: медиана по всей
                    # цепочке ползёт вместе с ней, и сборка, добавляя каждый раз кусок чуть ниже
                    # медианы, законно уходит на соседнюю строку за десяток звеньев (1973/08 с.85,
                    # низ левой колонки: «еще К. Маркс, когда говорил, что „…по»). Выпрямленная
                    # ордината вдоль настоящей строки постоянна, поэтому начало — верный отсчёт.
                    level = float(np.median(straight[chain[:FIELD_LEVEL_HEAD]]))
                    if abs(straight[other] - level) > guard + oh / 2.0 * FIELD_GUARD_SPAN:
                        continue
                if max(h, oh) > scale.link_height_ratio * min(h, oh) or _crosses(
                    separators, right, ox, cy, max(h, oh), ink300, k
                ):
                    continue
                if best is None or ox < int(stats[best, cv2.CC_STAT_LEFT]):
                    best = other
            if best is None:
                break
            chain.append(best)
            used.add(best)
        spans.append(chain)
    return spans


def _all_leader_dots(boxes: np.ndarray, span: list[int], leaders: list | None) -> bool:
    """Состоит ли цепочка целиком из точек отточий (координаты — пиксели рабочей копии)."""
    if not leaders:
        return False
    centres = np.array(
        [(float(boxes[blob, 0] + boxes[blob, 2] / 2.0), float(boxes[blob, 1] + boxes[blob, 3] / 2.0)) for blob in span]
    )
    for x, y in centres:
        if not any(leader.covers(x) and abs(leader.y - y) <= max(leader.thickness, 2.0) for leader in leaders):
            return False
    return True


def _crosses_rule(rules: list["Rule"] | None, left: float, right: float, y_left: float, y_right: float) -> bool:
    """Лежит ли между ординатами двух кусков сплошная черта, накрывающая разрыв между ними по x."""
    if not rules:
        return False
    low, high = sorted((y_left, y_right))
    x0, x1 = sorted((left, right))
    return any(low <= rule.cy <= high and rule.x0 <= x1 and rule.x1 >= x0 for rule in rules)


def merge_by_field(
    spans: list[list[int]],
    stats: np.ndarray,
    field: LineField,
    separators: list[tuple[int, int, int, int]],
    scale: Scale,
    ink300: np.ndarray,
    k: float,
    leaders: list | None = None,
    rules: list["Rule"] | None = None,
) -> list[list[int]]:
    """Сшить цепочки, лежащие на одном уровне поля: обрывки одной строки — снова одна строка.

    Ограничитель поля (``link_spans(field=...)``) не даёт цепочке уйти на соседнюю строку, но
    кусок, который жадный ход успел забрать в чужую цепочку, своей строке уже не достаётся: он
    начинает собственную цепочку, и строка выходит разорванной пополам. Этот проход сшивает
    такие половинки обратно — по тому же признаку, по которому ограничитель их разделил:
    ВЫПРЯМЛЕННАЯ ордината. Сшиваются только соседи слева направо, не разделённые живым
    межколонником и отстоящие не дальше обычного разрыва строки.

    Args:
        spans: Цепочки после сцепки с ограничителем.
        stats: Сгустки страницы.
        field: Поле хода строк.
        separators: Живые межколонники.
        scale: Размеры масштаба набора.
        ink300: Краска копии ``RENDER_DPI``.
        k: Во сколько раз рендер крупнее рабочей копии.
        leaders: Отточия страницы. Их точки сшивать нельзя: каждая точка — отдельная цепочка из
            одного куска, и сшивание набирало строку оглавления из двух десятков таких цепочек
            (1971/10 с.93: блоков 9 → 23, покрытие краски 99 → 86 %). В строку отточия и так
            входят отдельным ходом (:func:`extend_with_leaders`), который не трогает геометрию
            ряда.
        rules: Сплошные горизонтальные черты. Через ребро ячейки сшивать нельзя: в ячейках
            встречается повёрнутый текст, и «тот же уровень» по разные стороны ребра —
            это разные ячейки, а не одна строка.

    Returns:
        Цепочки после сшивания, куски внутри упорядочены слева направо.
    """
    if not spans or field.pitch <= 0:
        return spans
    boxes = stats[:, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]].astype(np.float64)
    straight = field.straighten(boxes[:, 0] + boxes[:, 2] / 2.0, boxes[:, 1] + boxes[:, 3] / 2.0)
    # Паспорт цепочки: уровень, края по x, ордината и высота на краях.
    order = sorted(range(len(spans)), key=lambda index: float(boxes[spans[index], 0].min()))
    levels = [float(np.median(straight[span])) for span in spans]
    dotted = [_all_leader_dots(boxes, span, leaders) for span in spans]
    merged: list[list[int]] = []
    taken = [False] * len(spans)
    for position, index in enumerate(order):
        if taken[index]:
            continue
        chain = list(spans[index])
        taken[index] = True
        if dotted[index]:
            merged.append(chain)
            continue
        # Присоединяем следующие правее цепочки, пока они лежат на том же уровне.
        for other in order[position + 1 :]:
            if taken[other] or dotted[other]:
                continue
            if abs(levels[other] - float(np.median(straight[chain]))) > FIELD_MERGE_PITCHES * field.pitch:
                continue
            left = float(boxes[spans[other], 0].min())
            right = float((boxes[chain, 0] + boxes[chain, 2]).max())
            height = max(float(boxes[chain, 3].max()), float(boxes[spans[other], 3].max()))
            if left - right > scale.link_gap_heights * height:
                continue
            # Цепочкам одной строки разрешено ПЕРЕКРЫВАТЬСЯ по x: на сильно искажённой бумаге
            # сгустки соседних строк чередуются, и строка собирается двумя цепочками, идущими по
            # одному и тому же отрезку (1973/11 с.79: ряды x111..342 и x290..476 при шаге 22 px).
            # Именно такие половинки и рвут блок — правило «слабое перекрытие по x» режет колонку
            # между ними. Перекрытие ограничено долей от более короткой цепочки: иначе сшиваются
            # соседние по вертикали строки одной ширины.
            other_right = float((boxes[spans[other], 0] + boxes[spans[other], 2]).max())
            other_left = float(boxes[spans[other], 0].min())
            chain_left = float(boxes[chain, 0].min())
            overlap = min(right, other_right) - max(chain_left, other_left)
            shorter = min(right - chain_left, other_right - other_left)
            if overlap > MERGE_OVERLAP_SHARE * max(shorter, 1.0):
                continue
            cy = float(np.median(boxes[chain, 1] + boxes[chain, 3] / 2.0))
            if _crosses(separators, right, left, cy, height, ink300, k):
                continue
            other_cy = float(np.median(boxes[spans[other], 1] + boxes[spans[other], 3] / 2.0))
            if _crosses_rule(rules, right, left, cy, other_cy):
                continue
            chain.extend(spans[other])
            taken[other] = True
        merged.append(sorted(chain, key=lambda blob: float(boxes[blob, 0])))
    return merged


__all__ = ["link_spans", "merge_by_field"]
