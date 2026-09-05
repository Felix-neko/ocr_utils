"""Группировка связных областей маски в «операции закраса».

ЗАЧЕМ. Сеть заливает дыру по её окрестности, поэтому закрашивать надо тесным ROI
вокруг области (см. :mod:`ocr_utils.inpainting.roi`). Но «область» и «связная
область» — не одно и то же. Рукописную надпись разметчик нередко обводит по букве,
и одна метка распадается на шесть-семь связных кусков. Закрасить их по одному
значит шесть раз показать сети окрестность одной буквы, где соседние буквы той же
надписи остались на месте, — сеть честно продолжит их внутрь дыры, и вместо чистой
бумаги получится рукописная каша.

ПРАВИЛО. Каждая связная область раздувается на ``dilate_frac`` СВОЕЙ ширины и
высоты в каждую сторону; те, у кого раздутые рамки пересеклись, идут в закрас одной
операцией — одной маской с несколькими связными областями. Правило транзитивно:
если A задел B, а B задел C, все трое в одной группе, даже когда A и C далеко друг
от друга. Отсюда система непересекающихся множеств, а не простая попарная склейка.

Доля от собственного размера, а не фиксированные пиксели: буква и печать во всю
ширину полосы требуют разного зазора, и единая константа была бы либо велика для
одной, либо мала для другой. При ``dilate_frac = 1/3`` две области склеиваются,
если зазор между ними меньше ``w1/3 + w2/3``.

ПЕРЕСЕКАЮТСЯ РАМКИ, А НЕ ПИКСЕЛИ. Прямоугольники точны, дёшевы и проверяются
тестом на бумаге; областей на полосе единицы (замер по паку-1: 363 группы
«полоса + вид разметки» односвязны, 37 состоят из двух областей, максимум — семь),
так что сильно ошибиться нечем. Лишняя склейка стоит лишь большего ROI, то есть
как раз того, ради чего группировка и делается. Если длинный диагональный росчерк
начнёт затягивать соседнюю печать, попиксельный вариант строится поверх
``finger_removal.asymmetric_dilation.dilate_zones_elliptical``: раздуть каждую
область эллипсом в 1/3 её размера и взять связность результата.

Раздутая рамка НЕ обрезается кадром: обрезка могла бы только ослабить склейку у
края, а сам ROI всё равно обрезается позже, в :func:`roi.roi_bounds`.
"""

import cv2
import numpy as np


# Доля собственной ширины/высоты области, на которую она раздувается В КАЖДУЮ
# СТОРОНУ перед проверкой пересечения. У квадрата 300×300 рамка растёт до 500×500.
DEFAULT_GROUP_DILATE_FRAC = 1.0 / 3.0

# Связные области мельче этого (пикс.) в закрас не идут вовсе: это одиночные
# пиксели от растеризации краёв маски, а не размеченный объект.
MIN_ZONE_AREA = 20


def expand_box(
    box: "tuple[int, int, int, int]", dilate_frac: float, min_dilate_px: int = 0
) -> "tuple[float, float, float, float]":
    """Рамка, раздутая на ``dilate_frac`` своей ширины и высоты в каждую сторону.

    ``min_dilate_px`` — нижняя граница припуска в пикселях: у совсем мелкой области
    доля от собственного размера вырождается в ноль, и она не склеилась бы с
    соседкой, даже лежащей вплотную.

    Возвращает ``(x1, y1, x2, y2)`` во float и БЕЗ обрезки по кадру (зачем — см.
    докстринг модуля).
    """
    x1, y1, x2, y2 = box
    dx = max((x2 - x1) * dilate_frac, float(min_dilate_px))
    dy = max((y2 - y1) * dilate_frac, float(min_dilate_px))
    return x1 - dx, y1 - dy, x2 + dx, y2 + dy


def boxes_overlap(a: "tuple[float, float, float, float]", b: "tuple[float, float, float, float]") -> bool:
    """Пересекаются ли два прямоугольника ``(x1, y1, x2, y2)``.

    Касание краями пересечением НЕ считается: рамки полуинтервальные, у соседних
    без зазора ``a.x2 == b.x1``, и строгое сравнение оставляет их разными группами.
    """
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def union_indices(count: int, pairs: "list[tuple[int, int]]") -> "list[list[int]]":
    """Индексы ``0..count-1``, разбитые на классы отношением «связаны парой».

    Система непересекающихся множеств со сжатием путей. Классы возвращаются в
    порядке НАИМЕНЬШЕГО индекса в каждом, а внутри класса индексы упорядочены, —
    чтобы результат не зависел от порядка пар и прогон был воспроизводим.
    """
    parent = list(range(count))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    groups: "dict[int, list[int]]" = {}
    for i in range(count):
        groups.setdefault(find(i), []).append(i)
    return [sorted(members) for _, members in sorted(groups.items(), key=lambda kv: min(kv[1]))]


def group_boxes(
    boxes: "list[tuple[int, int, int, int]]", dilate_frac: float = DEFAULT_GROUP_DILATE_FRAC, min_dilate_px: int = 0
) -> "list[list[int]]":
    """Индексы рамок, разбитые на группы по пересечению раздутых версий.

    Чистая геометрия: ни картинок, ни масок, поэтому правило проверяется тестом
    напрямую. ``dilate_frac = 0`` отключает склейку — каждая рамка сама по себе
    (это же значение служит контрольным вариантом при сравнении).
    """
    expanded = [expand_box(b, dilate_frac, min_dilate_px) for b in boxes]
    pairs = [
        (i, j)
        for i in range(len(expanded))
        for j in range(i + 1, len(expanded))
        if boxes_overlap(expanded[i], expanded[j])
    ]
    return union_indices(len(boxes), pairs)


def component_boxes(
    mask: np.ndarray, min_area: int = MIN_ZONE_AREA
) -> "tuple[np.ndarray, list[tuple[int, int, int, int]], list[int]]":
    """Связные области маски: карта меток, их рамки и сами метки.

    Возвращает ``(labels, boxes, label_ids)``, где ``boxes[k]`` — рамка
    ``(x1, y1, x2, y2)`` области с меткой ``label_ids[k]``. Области мельче
    ``min_area`` пикселей отбрасываются.
    """
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    boxes: "list[tuple[int, int, int, int]]" = []
    label_ids: "list[int]" = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        boxes.append((int(x), int(y), int(x + w), int(y + h)))
        label_ids.append(label)
    return labels, boxes, label_ids


def group_masks(
    mask: np.ndarray,
    dilate_frac: float = DEFAULT_GROUP_DILATE_FRAC,
    min_dilate_px: int = 0,
    min_area: int = MIN_ZONE_AREA,
) -> "list[np.ndarray]":
    """Связные области маски, склеенные в маски-«операции» (uint8 0/255).

    Внутри одной группы областей может быть несколько, и они НЕ связаны между собой
    — именно это и нужно: обведённая по буквам рукописная надпись подаётся
    закрасчику одной операцией, а не буква за буквой.

    Группы возвращаются в порядке первой (левой верхней) области и вместе
    ПОКРЫВАЮТ исходную маску без пересечений — за вычетом мелочи, отсеянной
    ``min_area``. Пустая маска → пустой список.

    Маски групп собираются одним проходом по карте меток через таблицу
    «метка → номер группы»: отдельное ``labels == label`` на каждую область стоило
    бы прохода по кадру в 21 Мп на каждую из них.
    """
    labels, boxes, label_ids = component_boxes(mask, min_area)
    if not boxes:
        return []

    groups = group_boxes(boxes, dilate_frac, min_dilate_px)

    # 0 — «ни в какой группе»: и фон, и отсеянная по площади мелочь.
    lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    for number, members in enumerate(groups, start=1):
        for k in members:
            lut[label_ids[k]] = number
    grouped = lut[labels]
    return [(grouped == number).astype(np.uint8) * 255 for number in range(1, len(groups) + 1)]
