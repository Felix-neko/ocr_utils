"""Тесты группировки связных областей в «операции закраса» (``inpainting.grouping``)."""

import numpy as np
import pytest

from ocr_utils.inpainting.grouping import (
    DEFAULT_GROUP_DILATE_FRAC,
    MIN_ZONE_AREA,
    boxes_overlap,
    expand_box,
    group_boxes,
    group_masks,
    union_indices,
)


def square(x: int, y: int, side: int = 30) -> "tuple[int, int, int, int]":
    """Квадратная рамка со стороной ``side`` в точке (x, y)."""
    return (x, y, x + side, y + side)


def paint(shape, boxes) -> np.ndarray:
    """Маска uint8 0/255 с залитыми прямоугольниками."""
    mask = np.zeros(shape, np.uint8)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = 255
    return mask


# ----------------------------------------------------------------------
# Раздувание рамки и пересечение
# ----------------------------------------------------------------------


def test_expand_box_scales_with_own_size():
    # Квадрат 300x300 при 1/3 растёт на 100 в каждую сторону.
    assert expand_box((0, 0, 300, 300), 1 / 3) == (-100.0, -100.0, 400.0, 400.0)
    # Прямоугольник — по своей стороне отдельно.
    assert expand_box((0, 0, 300, 30), 1 / 3) == (-100.0, -10.0, 400.0, 40.0)


def test_expand_box_min_dilate_px_saves_tiny_zones():
    # У области 3x3 доля от собственного размера — это один пиксель; нижняя граница
    # припуска нужна, чтобы такая мелочь всё же склеилась с соседкой рядом.
    assert expand_box((0, 0, 3, 3), 1 / 3, min_dilate_px=10) == (-10.0, -10.0, 13.0, 13.0)


def test_boxes_touching_edges_do_not_overlap():
    # Рамки полуинтервальные: у смежных без зазора a.x2 == b.x1, и это не пересечение.
    assert not boxes_overlap((0, 0, 10, 10), (10, 0, 20, 10))
    assert boxes_overlap((0, 0, 10, 10), (9, 0, 20, 10))


# ----------------------------------------------------------------------
# Разбиение на группы
# ----------------------------------------------------------------------


def test_far_boxes_stay_separate():
    # Зазор 200 при стороне 30 — намного больше, чем 30/3 + 30/3.
    assert group_boxes([square(0, 0), square(230, 0)]) == [[0], [1]]


def test_near_boxes_merge():
    # Зазор 15 меньше, чем 30/3 + 30/3 = 20.
    assert group_boxes([square(0, 0), square(45, 0)]) == [[0, 1]]


@pytest.mark.parametrize("gap,expected", [(19, [[0, 1]]), (21, [[0], [1]])])
def test_merge_threshold_is_sum_of_thirds(gap, expected):
    """Обещание правила: склейка ровно при зазоре меньше ``w1/3 + w2/3``.

    Порог закрепляется тестом, потому что на него опирается выбор ``dilate_frac``
    в вызывающем коде: сдвинь его молча — и группировка на паке изменится вся.
    """
    assert group_boxes([square(0, 0), square(30 + gap, 0)]) == expected


def test_grouping_is_transitive():
    # A и C далеко друг от друга, но обе задевают B — значит все трое вместе.
    boxes = [square(0, 0), square(45, 0), square(90, 0)]
    assert group_boxes(boxes) == [[0, 1, 2]]
    # Без середины крайние остаются порознь.
    assert group_boxes([boxes[0], boxes[2]]) == [[0], [1]]


def test_zero_frac_disables_merging():
    # Контрольный вариант при сравнении: каждая область сама по себе.
    assert group_boxes([square(0, 0), square(31, 0)], dilate_frac=0.0) == [[0], [1]]


def test_union_indices_order_is_stable():
    # Классы — в порядке наименьшего индекса, внутри класса индексы упорядочены,
    # и порядок пар на результат не влияет.
    assert union_indices(4, [(2, 3), (0, 1)]) == [[0, 1], [2, 3]]
    assert union_indices(4, [(3, 2), (1, 0)]) == [[0, 1], [2, 3]]


# ----------------------------------------------------------------------
# Маски групп
# ----------------------------------------------------------------------


def test_group_masks_empty():
    assert group_masks(np.zeros((50, 50), np.uint8)) == []


def test_group_masks_partition_the_source():
    """Группы вместе покрывают исходную маску и попарно не пересекаются."""
    boxes = [square(10, 10), square(55, 10), square(300, 200)]
    mask = paint((400, 400), boxes)

    groups = group_masks(mask)
    assert len(groups) == 2  # две близкие склеились, дальняя отдельно

    union = np.zeros_like(mask, bool)
    for g in groups:
        assert not (union & (g > 0)).any(), "группы пересекаются"
        union |= g > 0
    assert np.array_equal(union, mask > 0)


def test_group_masks_keeps_unconnected_components_together():
    # Главный случай: надпись, обведённая по буквам, идёт одной операцией.
    letters = [square(x, 0, side=20) for x in (0, 25, 50, 75, 100)]
    groups = group_masks(paint((60, 200), letters))
    assert len(groups) == 1
    # Внутри группы области по-прежнему НЕ связаны — их пять.
    import cv2

    count, _ = cv2.connectedComponents((groups[0] > 0).astype(np.uint8), connectivity=8)
    assert count - 1 == 5


def test_group_masks_drops_specks():
    """Мелочь мельче ``MIN_ZONE_AREA`` не образует своей группы и не тянет соседей."""
    mask = paint((200, 200), [square(10, 10)])
    mask[150, 150] = 255  # одиночный пиксель — растеризационный мусор
    groups = group_masks(mask, min_area=MIN_ZONE_AREA)
    assert len(groups) == 1
    assert groups[0][150, 150] == 0


def test_default_frac_is_one_third():
    assert DEFAULT_GROUP_DILATE_FRAC == pytest.approx(1 / 3)
