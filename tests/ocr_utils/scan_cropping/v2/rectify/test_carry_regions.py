"""Тесты переноса областей второго преобразования за содержимым (``rectify.text_run.carry_regions``)."""

from __future__ import annotations

import numpy as np
import pytest

from ocr_utils.scan_cropping.v2.rectify.text_field import PageRegion, RegionKindR
from ocr_utils.scan_cropping.v2.rectify.text_run import carried_points, carry_regions

DPI = 300.0


def _field(dx, dy, shape=(400, 500)) -> tuple[np.ndarray, np.ndarray]:
    """Поле ``(D_x, D_y)`` холста: числа — постоянное смещение, функции ``f(xx, yy)`` — по координатам."""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    fx = dx(xx, yy) if callable(dx) else np.full(shape, dx, np.float32)
    fy = dy(xx, yy) if callable(dy) else np.full(shape, dy, np.float32)
    return fx.astype(np.float32), fy.astype(np.float32)


def test_constant_field_moves_box_against_displacement():
    """Холст берёт пиксель P из P + D: при D = (5, −3) содержимое и рамка едут на (−5, +3)."""
    region = PageRegion(RegionKindR.LINE_ART, (100.0, 100.0, 200.0, 180.0))
    ((moved,),) = carry_regions([(region,)], _field(5.0, -3.0), DPI)
    assert moved.box == pytest.approx((95.0, 103.0, 195.0, 183.0), abs=0.05)


def test_stretch_widens_box():
    """Растяжение по x (D_x = −0.1·x: точка Q стоит в Q / 0.9) расширяет рамку, а не только сдвигает."""
    region = PageRegion(RegionKindR.RASTER, (90.0, 50.0, 180.0, 120.0))
    ((moved,),) = carry_regions([(region,)], _field(lambda xx, yy: -0.1 * xx, 0.0), DPI)
    x0, y0, x1, y1 = moved.box
    assert x0 == pytest.approx(100.0, abs=0.5) and x1 == pytest.approx(200.0, abs=0.5)
    assert (y0, y1) == pytest.approx((50.0, 120.0), abs=0.05)


def test_polygon_carried_and_box_is_its_extent():
    """Многоугольник фото переносится по вершинам, рамка — его охват; найденное фото остаётся прежним объектом."""
    polygon = np.array([[100.0, 100.0], [200.0, 105.0], [195.0, 200.0], [102.0, 195.0]])
    marker = object()
    region = PageRegion(RegionKindR.RASTER, (100.0, 100.0, 200.0, 200.0), polygon, marker)
    ((moved,),) = carry_regions([(region,)], _field(-4.0, 2.0), DPI)
    assert moved.polygon == pytest.approx(polygon + np.array([4.0, -2.0]), abs=0.05)
    assert moved.box == pytest.approx((104.0, 98.0, 204.0, 198.0), abs=0.05)
    assert moved.photo is marker


def test_zero_field_returns_same_regions():
    """Без смещения области не меняются (тот же объект)."""
    regions = [(PageRegion(RegionKindR.TABLE, (1.0, 2.0, 3.0, 4.0)),), ()]
    assert carry_regions(regions, _field(0.0, 0.0), DPI) is regions


def test_carried_points_inverts_smooth_field():
    """Обращение гладкого поля: P + D(P) = Q с точностью до сотых пикселя."""
    total = _field(lambda xx, yy: 3.0 * np.sin(xx / 60.0), lambda xx, yy: 2.0 * np.cos(yy / 50.0))
    targets = np.array([[150.0, 120.0], [300.0, 250.0], [420.0, 80.0]])
    found = carried_points(targets, total)
    back = found + np.stack([3.0 * np.sin(found[:, 0] / 60.0), 2.0 * np.cos(found[:, 1] / 50.0)], axis=1)
    assert back == pytest.approx(targets, abs=0.05)
