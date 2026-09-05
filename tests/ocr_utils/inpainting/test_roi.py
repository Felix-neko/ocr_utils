"""Тесты геометрии ROI и вклейки (``inpainting.roi``).

Модуль переехал из ``finger_removal`` и обзавёлся вторым потребителем (закрас
разметки из CVAT), поэтому его обещания пора закрепить: раньше они держались
только на докстрингах.
"""

import numpy as np

from ocr_utils.inpainting.roi import blend_roi, mask_bbox, mask_components, roi_bounds, roi_bounds_list


def paint(shape, boxes) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = 255
    return mask


def test_mask_bbox_is_half_open():
    mask = paint((100, 100), [(10, 20, 30, 40)])
    assert mask_bbox(mask) == (10, 20, 30, 40)


def test_mask_bbox_of_empty_mask():
    assert mask_bbox(np.zeros((10, 10), np.uint8)) is None


def test_roi_bounds_none_for_empty_mask():
    assert roi_bounds(np.zeros((10, 10), np.uint8), 5, 1.5, (10, 10)) is None


def test_roi_bounds_padding_then_scale():
    # (10..30) + padding 5 → (5..35), сторона 30, центр 20; scale 2 → (-10..50),
    # и левый край обрезается кадром до 0.
    assert roi_bounds(paint((200, 200), [(10, 10, 30, 30)]), 5, 2.0, (200, 200)) == (0, 0, 50, 50)


def test_roi_bounds_scale_one_is_just_padding():
    assert roi_bounds(paint((200, 200), [(50, 50, 90, 90)]), 10, 1.0, (200, 200)) == (40, 40, 100, 100)


def test_mask_components_splits_by_connectivity():
    mask = paint((100, 200), [(10, 10, 30, 30), (100, 10, 130, 30)])
    comps = mask_components(mask)
    assert len(comps) == 2
    assert all(set(np.unique(c)) <= {0, 255} for c in comps)


def test_roi_bounds_list_one_per_component():
    mask = paint((100, 200), [(10, 10, 30, 30), (100, 10, 130, 30)])
    assert len(roi_bounds_list(mask, padding=2, roi_scale=1.0)) == 2


def test_blend_roi_never_spills_outside_the_mask():
    """Обещание, на котором стоит весь закрас: вне маски результат бит-в-бит исходный.

    Заливка приходит от сети через ресайз ROI туда-обратно, и подмешать её в
    неиспорченные пиксели вокруг зоны нельзя даже узкой каймой.
    """
    rng = np.random.default_rng(1)
    orig = rng.integers(0, 255, size=(80, 80, 3), dtype=np.uint8)
    filled = np.full_like(orig, 9)
    mask = paint((80, 80), [(20, 20, 60, 60)])

    out = blend_roi(orig, filled, mask, feather=9)

    outside = mask == 0
    assert np.array_equal(out[outside], orig[outside])


def test_blend_roi_feather_zero_replaces_exactly_the_mask():
    orig = np.full((40, 40, 3), 100, np.uint8)
    filled = np.full_like(orig, 200)
    mask = paint((40, 40), [(10, 10, 30, 30)])

    out = blend_roi(orig, filled, mask, feather=0)

    assert (out[10:30, 10:30] == 200).all()
    assert (out[mask == 0] == 100).all()


def test_blend_roi_feather_ramps_inward():
    """Растушёвка идёт ВНУТРЬ: на самой границе маски — ещё исходник, в глубине — заливка."""
    orig = np.full((100, 100, 3), 0, np.uint8)
    filled = np.full_like(orig, 255)
    mask = paint((100, 100), [(20, 20, 80, 80)])

    out = blend_roi(orig, filled, mask, feather=10)

    assert out[20, 50, 0] < 60  # первый ряд внутри маски — почти исходник
    assert out[50, 50, 0] == 255  # центр — целиком заливка
