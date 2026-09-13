"""Детектор таблиц, третья версия: те же линейки, но рамка доводится до ума.

ЧТО ИЗМЕНИЛОСЬ ПРОТИВ ВТОРОЙ ВЕРСИИ. Вторая версия отвечала на вопрос «таблица ли это» и
отвечала хорошо: на 396 просмотренных полосах ложных срабатываний не осталось. Дальше человек
просмотрел 105 оверлеев принятых находок и разложил их по диагнозам, и оказалось, что почти
все оставшиеся дефекты — не про распознавание таблицы, а про РАМКУ: она режет буквы, хватает
соседний абзац, берёт половину таблицы или кусок блок-схемы. Третья версия — про рамку.

ПЯТЬ ПРАВОК, каждая замерена до того, как была написана:

1. Сквозной считается линейка ЛЮБОЙ оси. У шести полос из восемнадцати, где таблица есть, а
   детектор её не видел, сквозная горизонталь одна, зато сквозных вертикалей от двух до
   десяти: это бланки, размеченные вертикалями граф, без внешней рамки сверху и снизу.
2. К правилу «обе стороны не меньше 25 мм» ДОБАВЛЕНО «длинная не меньше 40, короткая не
   меньше 10». Ещё восемь полос из тех восемнадцати — широкие короткие шапки бланков:
   126x25, 130x16, 91x21 мм. Заменить одно правило другим нельзя: на синтетике это теряло
   небольшую квадратную таблицу 39x29 мм, которую вторая версия находила.
3. Линейка, оказавшаяся тенью корешка, выбрасывается до кластеризации (``refine``).
4. Рамка доращивается по связным рёбрам, пока рядом есть их продолжения (``grow`` ниже).
5. Границы прилипают к чистому просвету, чтобы вырезка не разрубала буквы (``refine``).

ЦЕНА ПОСЛАБЛЕНИЙ ЗАМЕРЕНА, а не оценена на глаз: на 900 случайных полосах пака, где сейчас
нет ни находок, ни таблиц в выгрузке DOCX, послабления дают находку на трёх полосах, и
проверка принимает одну. В пересчёте на пак это около двенадцати новых принятых полос из
одиннадцати тысяч.

ПОРЯДОК ВАЖЕН. Проверка стоит ДО доращивания — растить можно только то, что уже признано
таблицей, иначе рамка вокруг объявления дорастёт до половины полосы. И проверка же стоит
после доращивания: если выросшая рамка перестала быть таблицей, рост откатывается. Поэтому
самый рискованный шаг в худшем случае просто ничего не делает.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from research.legacy.table_processing.detection import quality, refine
from research.legacy.table_processing.detection.base import TableDetector
from research.legacy.table_processing.detection.ruling import (
    WORK_DPI,
    ClusterPolicy,
    Lines,
    Segment,
    binarize,
    cluster_tables,
    find_lines,
    mm_to_px,
)
from research.legacy.table_processing.geometry import Box, TableBox, intersection

logger = logging.getLogger(__name__)

# Политика кластеризации третьей версии — правки 1 и 2 из шапки модуля.
POLICY_V3 = ClusterPolicy(
    # Старое правило «обе стороны не меньше 25 мм» остаётся: без него теряются небольшие
    # квадратные таблицы. Новая пара «длинная 40, короткая 10» добавляет к нему широкие
    # короткие шапки бланков — 126x25, 130x16, 91x21 мм.
    min_sides_mm=((25.0, 25.0), (40.0, 10.0)),
    long_rules_any_axis=True,
)

# --- Доращивание -------------------------------------------------------------

# Ореол, в котором ищутся продолжения рёбер. 20 мм: замер на семи полосах, где таблица взята
# не целиком, показывает, что недостающие линейки лежат в 3-17 мм от рамки.
HALO_MM = 20.0

# Зазор склейки при доращивании — вдвое больше обычного. Замер на десяти полосах, где взят
# лишь кусок блок-схемы: расстояния до оставшихся фрагментов лежат в 6-15 мм, то есть обычные
# 6 мм их не достают, а 12 достают.
GROW_GAP_MM = 12.0

# Слабый порог длины линейки внутри ореола. У недобранных таблиц нижняя линейка бывает
# продавлена до 4-6 мм, и общий порог 8 мм её просто не видит. Слабый порог применяется
# ТОЛЬКО вокруг уже подтверждённой таблицы: там короткий штрих почти наверняка её ребро,
# а не тире в тексте.
GROW_MIN_RULE_MM = 4.0

# Общий потолок роста стороны на все проходы. Замер по классу «кусок блок-схемы»: при 25 мм
# доля чужого текста в рамке 0.024, при 40 мм — 0.010, при 60 мм — та же 0.010, и ни один
# другой класс от этого не портится. Берём 40: дальше выигрыша нет, а риск растёт.
GROW_CAP_MM = 40.0

# Сколько проходов доращивания. Три хватает: замер показывает, что после второго прирост
# падает ниже миллиметра.
GROW_PASSES = 3

# Насколько кандидат может отклоняться от предсказанного продолжения ребра.
COLINEAR_MM = 2.0

# И насколько может отличаться его наклон. Линейки пака лежат в пределах 1.7°, поэтому
# согласный наклон — сильный признак того, что это то же ребро, а не чужой штрих.
COLINEAR_ANGLE_DEG = 1.0

# Насколько может вырасти доля чужого текста в рамке, чтобы рост всё же приняли. 0.02 — это
# пара слов на всю рамку; целая строка набора даёт заметно больше. Замер на 200 хороших
# находках: у 90% находок доля чужого текста не превышает 0.001, так что запас щедрый.
MAX_FOREIGN_GAIN = 0.02


def _extent(segments: list[Segment], width: int, height: int) -> Box:
    return Box(
        min(s.box.x0 for s in segments),
        min(s.box.y0 for s in segments),
        max(s.box.x1 for s in segments),
        max(s.box.y1 for s in segments),
    ).clipped(width, height)


def is_continuation(candidate: Segment, owned: list[Segment], dpi: int) -> bool:
    """Похож ли кандидат на продолжение одного из рёбер уже найденной таблицы.

    Соосность меряется по ПОПЕРЕЧНОЙ координате: у продолжения горизонтали та же высота, у
    продолжения вертикали та же абсцисса. Плюс согласный наклон — именно он и позволяет
    ловить продолжения на искривлённой полосе, где ребро перестало быть отвесным: там центр
    поперёк уплывает, но угол остаётся тем же.
    """
    tolerance = mm_to_px(COLINEAR_MM, dpi)
    for rule in owned:
        if rule.horizontal is not candidate.horizontal:
            continue
        if candidate.horizontal:
            gap = abs(_centre(candidate.box)[1] - _centre(rule.box)[1])
        else:
            gap = abs(_centre(candidate.box)[0] - _centre(rule.box)[0])
        if gap > tolerance:
            continue
        if abs(candidate.angle_deg - rule.angle_deg) <= COLINEAR_ANGLE_DEG:
            return True
    return False


def _centre(box: Box) -> tuple[int, int]:
    return (box.x0 + box.x1) // 2, (box.y0 + box.y1) // 2


def _candidates(gray: np.ndarray, lines: Lines, box: Box, dpi: int) -> list[Segment]:
    """Отрезки в ореоле вокруг рамки, которые в неё не вошли.

    Два поставщика. Первый — линейки самой полосы, найденные общим порогом: ими живут
    блок-схемы, у которых рядом стоят целые фрагменты. Второй — повторный поиск в ореоле со
    слабым порогом длины: им живут недобранные таблицы, у которых нижнее ребро продавлено
    короче общего порога. Кандидаты второго поставщика обязаны пройти ``is_continuation``,
    первого — нет: целый фрагмент блок-схемы продолжением ребра не является по определению.
    """
    height, width = gray.shape[:2]
    halo = mm_to_px(HALO_MM, dpi)
    grown = Box(box.x0 - halo, box.y0 - halo, box.x1 + halo, box.y1 + halo).clipped(width, height)

    found: list[Segment] = []
    for segment in list(lines.horizontal) + list(lines.vertical):
        if _inside(segment.box, box):
            continue
        if _overlaps(segment.box, grown):
            found.append(segment)

    owned = [s for s in list(lines.horizontal) + list(lines.vertical) if _inside(s.box, box)]
    if owned and grown.width > 8 and grown.height > 8:
        weak = find_lines(gray[grown.slice], dpi, min_length_px=mm_to_px(GROW_MIN_RULE_MM, dpi))
        for segment in list(weak.horizontal) + list(weak.vertical):
            shifted = Segment(segment.box.shifted(grown.x0, grown.y0), segment.horizontal, segment.angle_deg)
            if _inside(shifted.box, box) or not is_continuation(shifted, owned, dpi):
                continue
            found.append(shifted)
    return found


def _inside(rule: Box, box: Box) -> bool:
    return rule.x0 >= box.x0 and rule.y0 >= box.y0 and rule.x1 <= box.x1 and rule.y1 <= box.y1


def _overlaps(first: Box, second: Box) -> bool:
    return first.x0 < second.x1 and second.x0 < first.x1 and first.y0 < second.y1 and second.y0 < first.y1


def grow(gray: np.ndarray, lines: Lines, box: Box, dpi: int, blocked: "list[Box] | None" = None) -> Box:
    """Дорастить рамку по связным рёбрам, пока рядом есть их продолжения.

    Механика — та же склейка, что в кластеризации, но зазор вдвое больше и применяется только
    в ореоле вокруг уже подтверждённой таблицы.

    ``blocked`` — рамки ДРУГИХ подтверждённых таблиц этой же полосы. Через них рост не идёт, и
    это единственная надёжная преграда, которую удалось найти. Разделитель «не расти через
    полосу сплошного текста» проверялся замером и не годится: у зазора до продолжения СВОЕЙ
    таблицы максимальная доля строки в зазоре 0.53 по медиане, у зазора до СОСЕДНЕЙ таблицы —
    0.40, распределения перекрываются полностью. А вот «там уже стоит другая таблица» —
    признак безошибочный: на 1972/07 с.75 без него две шапки по 29 мм срастались в шесть
    перекрывающихся рамок высотой до 153 мм.

    Потолок роста ``GROW_CAP_MM`` — общий на все проходы, а не на каждый: иначе за три прохода
    сторона уезжала на 75 мм, и потолок переставал что-либо ограничивать.
    """
    height, width = gray.shape[:2]
    gap = mm_to_px(GROW_GAP_MM, dpi)
    cap = mm_to_px(GROW_CAP_MM, dpi)
    start = box.clipped(width, height)
    current = start
    barriers = [item for item in (blocked or []) if item != box]

    for _ in range(GROW_PASSES):
        extra = [
            segment
            for segment in _candidates(gray, lines, current, dpi)
            if not any(_overlaps(segment.box, barrier) for barrier in barriers)
        ]
        if not extra:
            break
        canvas = np.zeros((height, width), np.uint8)
        cv2.rectangle(canvas, (current.x0, current.y0), (current.x1 - 1, current.y1 - 1), 255, -1)
        for segment in extra:
            rule = segment.box
            cv2.rectangle(canvas, (rule.x0, rule.y0), (max(rule.x0, rule.x1 - 1), max(rule.y0, rule.y1 - 1)), 255, -1)
        glued = cv2.dilate(canvas, np.ones((gap, gap), np.uint8))
        count, labels, _, _ = cv2.connectedComponentsWithStats(glued, 8)
        if count <= 1:
            break
        label = int(labels[_centre(current)[1], _centre(current)[0]])
        if label == 0:
            break
        joined = [current] + [s.box for s in extra if labels[_centre(s.box)[1], _centre(s.box)[0]] == label]
        grown_box = Box(
            min(item.x0 for item in joined),
            min(item.y0 for item in joined),
            max(item.x1 for item in joined),
            max(item.y1 for item in joined),
        ).clipped(width, height)
        capped = Box(
            max(grown_box.x0, start.x0 - cap),
            max(grown_box.y0, start.y0 - cap),
            min(grown_box.x1, start.x1 + cap),
            min(grown_box.y1, start.y1 + cap),
        )
        capped = _stop_before(capped, current, barriers)
        if capped.area <= current.area:
            break
        gained = capped.area - current.area
        current = capped
        if gained < mm_to_px(1.0, dpi) * max(current.width, current.height):
            break
    return current


def _stop_before(grown: Box, current: Box, barriers: list[Box]) -> Box:
    """Обрезать рост так, чтобы рамка не наехала на чужую подтверждённую таблицу."""
    result = grown
    for barrier in barriers:
        if not _overlaps(result, barrier):
            continue
        if barrier.y0 >= current.y1:  # чужая таблица ниже
            result = Box(result.x0, result.y0, result.x1, min(result.y1, barrier.y0))
        elif barrier.y1 <= current.y0:  # выше
            result = Box(result.x0, max(result.y0, barrier.y1), result.x1, result.y1)
        elif barrier.x0 >= current.x1:  # правее
            result = Box(result.x0, result.y0, min(result.x1, barrier.x0), result.y1)
        elif barrier.x1 <= current.x0:  # левее
            result = Box(max(result.x0, barrier.x1), result.y0, result.x1, result.y1)
        else:  # рамки пересекались ещё до роста — не наш случай, рост отменяем
            return current
        if result.width <= 0 or result.height <= 0:
            return current
    return result


# --- Конвейер ----------------------------------------------------------------


def detect(gray: np.ndarray, dpi: int = WORK_DPI, verify_findings: bool = True) -> list[TableBox]:
    """Таблицы третьей версии. Координаты — в пикселях поданного изображения.

    Импорт ``verify`` отложен внутрь функции: он тянет разбор сетки, который импортирует
    модуль линеек.
    """
    from research.legacy.table_processing.detection import verify as verification
    from research.legacy.table_processing.structure.ruling_grid import text_ink

    height, width = gray.shape[:2]
    raw = find_lines(gray, dpi)
    lines = refine.drop_border_rules(raw, (height, width), dpi)
    found = cluster_tables(lines, dpi, binarize(gray), POLICY_V3)
    if not found:
        return []
    if not verify_findings:
        return found

    def signs_of(box: Box):
        crop = gray[box.slice]
        return verification.features(crop, find_lines(crop, dpi), dpi)

    # Первый проход: кто вообще таблица. Растить можно только подтверждённое, иначе рамка
    # вокруг объявления дорастёт до половины полосы.
    accepted: list[tuple[TableBox, Box, object]] = []
    for table in found:
        rule_box = table.box.clipped(width, height)
        if rule_box.width < 8 or rule_box.height < 8:
            continue
        signs = signs_of(rule_box)
        good, reason = verification.is_table_v3(signs)
        if good:
            accepted.append((table, rule_box, signs))
        else:
            logger.debug("находка %s отклонена: %s", rule_box.as_tuple(), reason)
    if not accepted:
        return []

    # Второй проход: доращивание. Соседи известны только теперь — поэтому проходов два.
    neighbours = [box for _, box, _ in accepted]
    ink = text_ink(gray, lines)
    kept: list[TableBox] = []
    for table, rule_box, signs in accepted:
        grown = grow(gray, lines, rule_box, dpi, neighbours)
        if grown != rule_box:
            grown_signs = signs_of(grown)
            # Рост принимается, если выросшая рамка ЛИБО осталась таблицей, ЛИБО не набрала
            # чужого текста. Второе условие нужно ради блок-схем: целая схема таблицей не
            # является, проверка её отвергает, и по одной только проверке рост всегда
            # откатывался бы — а половина схемы вместо целой схемы это как раз то, что
            # человек назвал плохим. Мерой «чужого» служит краска в полосах без вертикалей:
            # у схемы её нет ни до, ни после роста, у абзаца рядом — сразу появляется.
            before = quality.foreign_text(ink, lines, rule_box, dpi)
            after = quality.foreign_text(ink, lines, grown, dpi)
            if verification.is_table_v3(grown_signs)[0] or after <= before + MAX_FOREIGN_GAIN:
                rule_box, signs = grown, grown_signs
            else:
                logger.debug(
                    "рост %s → %s откачен: чужого %.3f → %.3f", rule_box.as_tuple(), grown.as_tuple(), before, after
                )

        rule_box = refine.trim_ruleless(rule_box, lines, dpi).clipped(width, height)
        snapped = refine.snap_edges(rule_box, ink, lines, dpi)

        metrics = {**table.metrics, **{key: float(value) for key, value in signs.as_row().items()}}
        metrics["verified"] = 1.0
        metrics["snap_failed"] = float(snapped.failed)
        for side, value in snapped.moved_mm.items():
            metrics[f"snap_{side}"] = round(value, 2)
        kept.append(
            TableBox(
                box=snapped.box,
                score=table.score,
                source="ruling_v3",
                skew_deg=table.skew_deg,
                metrics=metrics,
                rule_box=rule_box,
            )
        )
    return _deduplicate(kept)


def _deduplicate(tables: list[TableBox]) -> list[TableBox]:
    """Убрать находки, съеденные соседкой после доращивания.

    Слияние в кластеризации происходит ДО роста и о нём ничего не знает. Если две рамки
    выросли навстречу и одна почти целиком легла внутрь другой, человек видит одну таблицу,
    обведённую дважды, и это надо убрать.
    """
    ordered = sorted(tables, key=lambda item: -item.box.area)
    kept: list[TableBox] = []
    for table in ordered:
        swallowed = False
        for bigger in kept:
            overlap = intersection(bigger.box, table.box)
            if overlap is not None and overlap.area >= 0.8 * table.box.area:
                swallowed = True
                break
        if not swallowed:
            kept.append(table)
    return sorted(kept, key=lambda item: item.box.y0)


ALGORITHM = TableDetector(
    name="ruling_v3",
    summary="линейки плюс доводка рамки: доращивание по рёбрам и прилипание к просвету (CPU)",
    stage="cpu",
    run=detect,
)
