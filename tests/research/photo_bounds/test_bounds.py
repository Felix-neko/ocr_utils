"""Тесты стенда «границы фото» на синтетике: меры рамок (IoU, промах сторон, доля светлого растра у стороны) и область фото многоугольником с выборками сторон; сами границы — тесты ``tests/ocr_utils/page_layout/raster/test_photo_edges.py``."""

from __future__ import annotations

import numpy as np

from research.photo_bounds.bounds import Side, find_photos
from research.photo_bounds.measure import band_cover, iou, side_deltas_mm
from research.photo_bounds.supports import PhotoRegion, photo_samples, region_of

from tests.ocr_utils.page_layout.raster.test_photo_edges import DPI, mm, photo_page


def test_measures():
    """IoU и промахи сторон со знаком; доля светлого растра у стороны."""
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0
    d = side_deltas_mm((0, 0, 100, 100), (10, 10, 90, 90), 1.0)
    assert d[Side.TOP] == 10 and d[Side.BOTTOM] == 10 and d[Side.LEFT] == 10 and d[Side.RIGHT] == 10
    light = np.zeros((200, 200), bool)
    light[50:150, 50:150] = True
    assert band_cover(light, (0, 0), (50, 50, 150, 150), Side.TOP, True, DPI) == 1.0
    assert band_cover(light, (0, 0), (50, 50, 150, 150), Side.TOP, False, DPI) == 0.0


def test_photo_samples_and_region():
    """Выборки сторон: верх и низ — в ``dy`` (2 группы), боковые — в ``dx``; у прямого снимка смещения ~0.
    Область фото многоугольником: точка подписи под снимком — вне области, внутри снимка — в ней."""
    img, true = photo_page()
    photo = find_photos(img, [(true[0] - mm(2), true[1] - mm(2), true[2] + mm(2), true[3] + mm(9))], DPI)[0]
    dx, dy, count = photo_samples(photo, DPI)
    assert count == 2
    assert len(np.unique(dy.group)) == 2 and len(np.unique(dx.group)) == 2
    assert np.abs(dy.value).max() < mm(0.15) and np.abs(dx.value).max() < mm(0.15)
    region = region_of(photo)
    assert isinstance(region, PhotoRegion)
    inside = np.array([[(true[0] + true[2]) / 2, (true[1] + true[3]) / 2]])
    caption = np.array([[mm(50), true[3] + mm(2.4)]])
    assert region.contains(inside).all()
    assert not region.contains(caption, mm(1.0)).any()
