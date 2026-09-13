"""Доводка рамки находки: выбросить чужое, поставить рамку по рёбрам, не разрезать буквы.

ТРИ РАЗНЫЕ ЗАДАЧИ, и они решаются в разных местах конвейера.

1. ``drop_border_rules`` работает ДО кластеризации. Тень корешка и обрез страницы дают
   длинную тонкую вертикаль у самого края скана; при склейке зазором 6 мм она смыкается с
   подчёркиваниями и заголовками абзацев выше таблицы, и рамка растягивается на пол-полосы.
   На 1968/01 с.31 такая вертикаль длиной 119 мм в 8 мм от края превращала таблицу 129x49 мм
   в находку 135x119 мм.

2. ``fit_rows`` работает ПОСЛЕ роста и только зажимает верх и низ рамки по крайним линейкам
   внутри неё. Обрезка третьей версии (``trim_ruleless``, осталась в
   ``research/legacy/table_processing``) пересобирала рамку из линеек всей полосы и молча
   откатывала доращённое; здесь рамка — поданная.

3. ``push_edges`` работает ПОСЛЕДНИМ и меняет только ту рамку, по которой режут: ни одна
   сторона не должна пересекать компоненту краски. Прилипание третьей версии
   (``snap_edges``) считало сторону чистой при доле краски ниже процента — это три
   срезанные первые буквы на высокой таблице (1973/12 с.86), — и не могло двигаться дальше
   3.5 мм.

ПОЧЕМУ ДОВОДКА НЕ МЕНЯЕТ ПРИЗНАКИ ПРОВЕРКИ. ``detection.verify`` считает всё по вырезке
ровно по рамке, и ``INNER_MARGIN_MM = 4`` отсчитывается от её края: сдвиг границы на два
миллиметра превратил бы внешнюю линейку таблицы во «внутреннюю вертикаль» и сдвинул бы
``inner_v`` и ``edge_fill``, на которых стоят все пороги. Поэтому у находки две рамки:
``rule_box`` — огибающая линеек, по ней считается проверка, и ``box`` — то, что режут.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocr_utils.scan_markup.table_detection.quality import straddling
from ocr_utils.scan_markup.table_detection.ruling import Lines, Segment, mm_to_px
from ocr_utils.scan_markup.table_detection.geometry import Box

# --- 1. Паразитная линейка у края скана -------------------------------------

# Ближе этого к краю кадра — уже поле, а не полоса набора. Замер: на 1968/01 с.31 паразитная
# вертикаль стоит в 8.1 мм от края, поэтому 5 мм мало.
BORDER_MM = 10.0

# Короче этой доли стороны кадра тень корешка не бывает: она идёт через всю полосу.
BORDER_LONG_SHARE = 0.4

# Допуск на пересечение линеек. 1.5 мм — это разброс центров линеек внутри одной колонки
# (см. geometry.Edge). С допуском в три пикселя проверка врала: у 17 длинных вертикалей из 154
# внутри НАСТОЯЩИХ таблиц пересечений не находилось, потому что горизонталь стоит ровно на краю.
CROSS_TOL_MM = 1.5


def crosses(vertical: Segment, horizontal: Segment, tolerance_px: int) -> bool:
    """Пересекаются ли вертикаль и горизонталь — по габаритам, с допуском."""
    centre_x = (vertical.box.x0 + vertical.box.x1) // 2
    centre_y = (horizontal.box.y0 + horizontal.box.y1) // 2
    return (
        horizontal.box.x0 - tolerance_px <= centre_x <= horizontal.box.x1 + tolerance_px
        and vertical.box.y0 - tolerance_px <= centre_y <= vertical.box.y1 + tolerance_px
    )


def drop_border_rules(lines: Lines, shape: tuple[int, int], dpi: int) -> Lines:
    """Убрать линейки, которые на деле — край скана, а не разметка таблицы.

    Три условия сразу, и третье главное: у настоящей внешней вертикали таблицы ВСЕГДА есть
    пересечение с её горизонталями, у тени корешка — нет. Замер на 250 принятых находках
    (1963 вертикали): под все три условия попадает ровно одна вертикаль настоящей таблицы,
    а по паку такие линейки встречаются примерно на одной полосе из ста тридцати.
    """
    height, width = shape
    border = mm_to_px(BORDER_MM, dpi)
    tolerance = mm_to_px(CROSS_TOL_MM, dpi)

    def parasitic(segment: Segment, along: int, distance: int) -> bool:
        if distance > border:
            return False
        if segment.length < BORDER_LONG_SHARE * along:
            return False
        others = lines.horizontal if not segment.horizontal else lines.vertical
        if segment.horizontal:
            return not any(crosses(other, segment, tolerance) for other in others)
        return not any(crosses(segment, other, tolerance) for other in others)

    kept_vertical = [
        segment
        for segment in lines.vertical
        if not parasitic(segment, height, min(segment.box.x0, width - segment.box.x1))
    ]
    kept_horizontal = [
        segment
        for segment in lines.horizontal
        if not parasitic(segment, width, min(segment.box.y0, height - segment.box.y1))
    ]
    return Lines(kept_horizontal, kept_vertical, lines.horizontal_mask, lines.vertical_mask)


# --- 2. Полосы без опоры на линейки -----------------------------------------


def _centre_inside(rule: Box, box: Box) -> bool:
    return box.x0 <= (rule.x0 + rule.x1) // 2 <= box.x1 and box.y0 <= (rule.y0 + rule.y1) // 2 <= box.y1


def _rule_bounds(lines: Lines, box: Box) -> tuple[int, int, int, int]:
    """Крайние линейки внутри рамки: левее/выше них внутрь двигаться нельзя."""
    inside = [
        segment.box for segment in list(lines.horizontal) + list(lines.vertical) if _centre_inside(segment.box, box)
    ]
    if not inside:
        return box.x0, box.y0, box.x1 - 1, box.y1 - 1
    return (
        min(rule.x0 for rule in inside),
        min(rule.y0 for rule in inside),
        max(rule.x1 for rule in inside) - 1,
        max(rule.y1 for rule in inside) - 1,
    )


# --- 2. Рамка строго по линейкам ----------------------------------------------------------

# Меньше двух вертикалей — о крайних линейках рамки судить не о чем, рамка остаётся как есть.
MIN_VERTICALS_FOR_ROWS = 2


def fit_rows(box: Box, lines: Lines, ink: np.ndarray, dpi: int) -> Box:
    """Верх и низ рамки — по крайним линейкам внутри неё. Ни строки сверху, ни строки снизу.

    Раньше отсюда добирались строки без вертикалей — «Итого» под последней линейкой — по тесту
    «текст не задевает просветы граф». Замер по 209 полосам: из 48 таких добираний 41 было
    заголовком «Таблица N», подписью или номером страницы в одной графе, и даже с оговоркой
    «текст хотя бы в двух графах» последняя строка абзаца вместе с подписью на той же высоте
    проходила тест (1968/06 с.38), а верхушка строки набора, срезанная краем окна, — тоже
    (1966/02 с.16). Человек решил: строку «Итого» без рёбер лучше оставить снаружи, чем
    захватывать чужой текст. Границу таблицы он понимает как минимальную область по рёбрам, и
    расширять её вправе только ``push_edges`` — ради не разрезанных букв.

    ``ink`` и ``dpi`` не используются и оставлены в сигнатуре ради вызывающих.
    """
    del ink, dpi
    inside_v = [s for s in lines.vertical if _centre_inside(s.box, box)]
    if len(inside_v) < MIN_VERTICALS_FOR_ROWS:
        return box
    inside_h = [s for s in lines.horizontal if _centre_inside(s.box, box)]
    # Линейка с центром внутри рамки может торчать наружу (чужая вертикаль, прошедшая мимо
    # ядра); без зажима её конец тянул низ рамки на абзац под таблицей (1966/02 с.16).
    rule_top = max(box.y0, min([s.box.y0 for s in inside_v] + [s.box.y0 for s in inside_h]))
    rule_bottom = min(box.y1, max([s.box.y1 for s in inside_v] + [s.box.y1 for s in inside_h]))
    if rule_bottom <= rule_top:
        return box
    return Box(box.x0, rule_top, box.x1, rule_bottom)


# --- 3. Граница не пересекает компоненты краски -----------------------------------------

# Дальше этого границу наружу не двигают: 12 мм. Компонента крупнее — клякса или рисунок,
# и её не обойти; сторона остаётся на месте и помечается ``push_failed``.
PUSH_LIMIT_MM = 12.0

# Зазор между краем компоненты и границей после сдвига.
PUSH_PAD_MM = 0.5

# Компонента, лежащая снаружи больше чем на эту долю, выталкивается наружу (граница уходит
# внутрь), если внутри есть место до крайней линейки. Иначе граница уходит наружу.
OUTSIDE_SHARE = 0.6

# Сколько раз повторять: сдвиг открывает новые компоненты под новой границей.
PUSH_PASSES = 6

# Чужая компонента, прижатая к линейке снаружи (подпись под таблицей, строка абзаца над ней),
# отрезается по линейке — кроме случая, когда граница задевает её лишь краем и уйти наружу
# можно на эти доли миллиметра: полбуквы резать нельзя, а тянуть рамку на целую строку ради
# чужого текста — тоже.
SLIGHT_MM = 1.5

# Компонента, торчащая за линейку, — СВОЯ, если её строка лежит внутри рамки: полоса высотой
# в компоненту (для левой и правой сторон; шириной — для верха и низа), взятая по всей длине
# стороны, несёт краску внутри рамки не меньше чем на эту долю. Первая буква «М» в
# «Министерство Грузинской ССР» открытой таблицы (1971/03 с.38) торчит за левую линейку на
# 73 % своей ширины, а вся её строка — внутри; подпись под таблицей и последняя строка абзаца
# над ней лежат снаружи целиком. За своей компонентой граница уходит наружу на её ширину, но не
# дальше ``OWN_LIMIT_MM``: это размер слова, не строки.
LINE_INSIDE_SHARE = 0.7
OWN_LIMIT_MM = 6.0


@dataclass(frozen=True)
class Pushed:
    box: Box
    moved_mm: dict[str, float]
    failed: int


def _with_side(box: Box, side: str, position: int, shape: tuple[int, int]) -> "Box | None":
    """Рамка с одной стороной, переставленной в ``position``; ``None``, если рамка вывернулась."""
    height, width = shape
    x0, y0, x1, y1 = box.as_tuple()
    if side == "сверху":
        y0 = max(0, position)
    elif side == "снизу":
        y1 = min(height, position)
    elif side == "слева":
        x0 = max(0, position)
    else:
        x1 = min(width, position)
    return Box(x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _settle(
    box: Box, side: str, components: np.ndarray, outward: bool, bound: int, pad: int, shape: tuple[int, int]
) -> "int | None":
    """Положение стороны, при котором она не пересекает ни одной компоненты.

    Идёт шагами: за край всех пересечённых компонент, проверка заново, и так до чистого места
    или до ``bound``. Один шаг не годится: сдвиг внутрь на верх чужой строки ставит границу
    ровно на буквы своей строки, и без повторной проверки граница колебалась между двумя
    положениями, ни разу не став чистой (1973/12 с.85).
    """
    current = box
    for _ in range(PUSH_PASSES):
        hits = straddling(components, current, side)
        if hits.shape[0] == 0:
            return _side_position(current, side)
        x0, y0, x1, y1 = hits.T
        if side == "сверху":
            position = int(y0.min()) - pad if outward else int(y1.max()) + 1
            legal = position >= bound if outward else position <= bound
        elif side == "снизу":
            position = int(y1.max()) + pad if outward else int(y0.min())
            legal = position <= bound if outward else position >= bound
        elif side == "слева":
            position = int(x0.min()) - pad if outward else int(x1.max()) + 1
            legal = position >= bound if outward else position <= bound
        else:
            position = int(x1.max()) + pad if outward else int(x0.min())
            legal = position <= bound if outward else position >= bound
        if not legal:
            return None
        current = _with_side(current, side, position, shape)
        if current is None:
            return None
    return None


def _side_position(box: Box, side: str) -> int:
    return {"сверху": box.y0, "снизу": box.y1, "слева": box.x0, "справа": box.x1}[side]


def _line_inside_share(ink: np.ndarray, box: Box, side: str, hits: np.ndarray) -> float:
    """Какая доля краски строки пересечённых компонент лежит внутри рамки."""
    height, width = ink.shape[:2]
    x0, y0, x1, y1 = hits.T
    if side in ("слева", "справа"):
        lo, hi = max(0, int(y0.min())), min(height, int(y1.max()))
        band = ink[lo:hi, :] > 0
        inside = band[:, box.x0 : box.x1]
    else:
        lo, hi = max(0, int(x0.min())), min(width, int(x1.max()))
        band = ink[:, lo:hi] > 0
        inside = band[box.y0 : box.y1, :]
    total = int(band.sum())
    return float(inside.sum()) / total if total else 0.0


def push_edges(
    box: Box, components: np.ndarray, lines: Lines, dpi: int, shape: tuple[int, int], ink: "np.ndarray | None" = None
) -> Pushed:
    """Сдвинуть границы так, чтобы ни одна из них не пересекала компоненту краски.

    Отличие от ``snap_edges`` третьей версии — единица счёта. Там граница считалась чистой, если
    под ней меньше процента краски, и искался просвет не дальше 3.5 мм; здесь граница обязана не
    пересекать НИ ОДНОЙ компоненты, а двигаться она может на 12 мм: столько занимают крупные
    буквы заголовка на линейке рамки (1970/05 с.3) и «Продолжение» над таблицей.

    Компонента, лежащая по большей части снаружи, — чужой текст: граница уходит внутрь, но
    только до крайней линейки (``_rule_bounds``): за линейкой — тело таблицы, его не режут.
    Если внутрь нельзя, граница уходит наружу; если и наружу дальше потолка — сторона остаётся
    и считается в ``failed``.
    """
    height, width = shape
    box = box.clipped(width, height)
    limit = mm_to_px(PUSH_LIMIT_MM, dpi)
    pad = mm_to_px(PUSH_PAD_MM, dpi)
    scale = 25.4 / dpi
    inner = _rule_bounds(lines, box)
    start = box
    failed: set[str] = set()
    sides = ("сверху", "снизу", "слева", "справа")
    outward_bound = {
        "сверху": start.y0 - limit,
        "снизу": start.y1 + limit,
        "слева": start.x0 - limit,
        "справа": start.x1 + limit,
    }
    inward_bound = {"сверху": inner[1], "снизу": inner[3], "слева": inner[0], "справа": inner[2]}

    for _ in range(2):  # сдвиг одной стороны меняет отрезок соседней — второй круг подчищает
        for side in sides:
            if side in failed:
                continue
            hits = straddling(components, box, side)
            if hits.shape[0] == 0:
                continue
            x0, y0, x1, y1 = hits.T
            if side == "сверху":
                outside = (box.y0 - y0) / np.maximum(1, y1 - y0)
            elif side == "снизу":
                outside = (y1 - box.y1) / np.maximum(1, y1 - y0)
            elif side == "слева":
                outside = (box.x0 - x0) / np.maximum(1, x1 - x0)
            else:
                outside = (x1 - box.x1) / np.maximum(1, x1 - x0)
            prefer_inward = bool(np.all(outside > OUTSIDE_SHARE))
            inward = _settle(box, side, components, False, inward_bound[side], pad, shape)
            outward = _settle(box, side, components, True, outward_bound[side], pad, shape)
            if prefer_inward and inward is None:
                # Внутрь до чистого места нельзя (упёрлись в крайнюю линейку), а компоненты
                # лежат по большей части снаружи. Своя строка (её краска внутри рамки) —
                # идём наружу за компонентой, но не дальше размера слова. Чужая: наружу только
                # на доли миллиметра (граница задела край буквы), иначе граница ставится
                # вплотную к линейке — чужое отрезается по линейке, а не тянет рамку на
                # подпись, номер страницы или строку абзаца.
                slight = mm_to_px(SLIGHT_MM, dpi)
                own_limit = mm_to_px(OWN_LIMIT_MM, dpi)
                rule_position = inward_bound[side]
                own = ink is not None and _line_inside_share(ink, box, side, hits) >= LINE_INSIDE_SHARE
                allowed = own_limit if own else slight
                if outward is not None and abs(outward - rule_position) <= allowed:
                    inward = None
                else:
                    inward = rule_position
            if prefer_inward and inward is not None:
                position = inward
            elif outward is not None:
                position = outward
            elif inward is not None:
                position = inward
            else:
                failed.add(side)
                continue
            moved = _with_side(box, side, position, shape)
            if moved is None:
                failed.add(side)
                continue
            box = moved

    moved_mm = {
        "сверху": (start.y0 - box.y0) * scale,
        "снизу": (box.y1 - start.y1) * scale,
        "слева": (start.x0 - box.x0) * scale,
        "справа": (box.x1 - start.x1) * scale,
    }
    return Pushed(box, moved_mm, len(failed))
