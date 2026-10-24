"""Тесты фото во втором преобразовании на синтетике: область фото многоугольником — отсев выборок и преграда разбора по охвату, стороны фото — опоры поля (``photo_side_samples``), ограничение веса сторон фото у текста (``limit_photo_weights``) и его действие на поле."""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.raster.photo_edges import find_photos
from ocr_utils.scan_cropping.v2.rectify.coons import Layout, PageRect
from ocr_utils.scan_cropping.v2.rectify.text_field import (
    PHOTO_NEAR_MM,
    PHOTO_NEAR_WEIGHT,
    PageRegion,
    RegionKindR,
    SampleKind,
    Samples,
    fit_field,
    limit_photo_weights,
    no_text_zones,
    outside,
    photo_side_samples,
    region_barriers,
)
from tests.ocr_utils.page_layout.raster.test_photo_edges import DPI, mm, photo_page

W, H = 1600, 1200


def skewed_polygon() -> np.ndarray:
    """Скошенный четырёхугольник фото: верх идёт вниз на 40 px слева направо."""
    return np.array([[200.0, 200.0], [800.0, 240.0], [800.0, 640.0], [200.0, 600.0]])


def test_polygon_region_contains_and_outside():
    """Многоугольник фото: точка подписи под скошенным низом — вне области (опора), хотя в охвате рамки; внутри — нет."""
    poly = skewed_polygon()
    box = (poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max())
    region = PageRegion(RegionKindR.RASTER, box, poly)
    rect = PageRegion(RegionKindR.RASTER, box)
    caption = np.array([[250.0, 620.0]])  # под низом у левого края: в охвате (y ≤ 640), но ниже стороны (y 603)
    inside = np.array([[500.0, 400.0]])
    assert rect.contains(caption).all() and not region.contains(caption, 5.0).any()
    assert region.contains(inside).all()
    zones = no_text_zones((region,))
    assert outside(caption, zones, 5.0).all() and not outside(inside, zones, 5.0).any()
    # Преграда разбора — охват многоугольника (рамки в px рабочей копии 150 dpi).
    barriers = region_barriers((region,), np.zeros(2), DPI, (W, H))
    assert barriers == ((100, 100, 400, 320),)


def test_photo_side_samples():
    """Стороны снимка — опоры: верх и низ — ``LINE`` в ``dy`` (две группы), боковые — ``SIDE`` в ``dx``."""
    img, true = photo_page()
    photo = find_photos(img, [(true[0] - mm(2), true[1] - mm(2), true[2] + mm(2), true[3] + mm(9))], DPI)[0]
    dx, dy, count = photo_side_samples(photo, DPI)
    assert count == 2
    assert set(dy.kind) == {SampleKind.LINE} and set(dx.kind) == {SampleKind.SIDE}
    assert len(np.unique(dy.group)) == 2 and len(np.unique(dx.group)) == 2
    # Сторона ровная: смещения к цели меньше 0.15 мм; выборки — между углами, не дальше 1 мм к углу.
    assert np.abs(dy.value).max() < mm(0.15) and np.abs(dx.value).max() < mm(0.15)
    assert dy.at[:, 0].min() >= true[0] + mm(0.9) - 2


def test_limit_photo_weights_ramp():
    """Вес выборки фото: у текста — ``PHOTO_NEAR_WEIGHT``, дальше двух порогов — полный, между — линейно."""
    near = mm(10.0)
    photo = Samples(np.array([[100.0, 0.0], [100.0 + 1.5 * near, 0.0], [100.0 + 3 * near, 0.0]]), np.zeros(3),
                    np.ones(3), np.zeros((3, 2)), np.full(3, SampleKind.SIDE), np.zeros(3, int))  # fmt: skip
    text = Samples(np.array([[100.0, 5.0]]), np.zeros(1), np.ones(1), np.zeros((1, 2)), np.full(1, SampleKind.SIDE),
                   np.zeros(1, int))  # fmt: skip
    w = limit_photo_weights(photo, text, DPI, 10.0, PHOTO_NEAR_WEIGHT).weight
    assert w[0] == PHOTO_NEAR_WEIGHT and w[2] == 1.0
    assert PHOTO_NEAR_WEIGHT < w[1] < 1.0
    # Без опор текста и при весе 1 — без изменений.
    assert np.array_equal(limit_photo_weights(photo, Samples.empty(), DPI).weight, photo.weight)
    assert np.array_equal(limit_photo_weights(photo, text, DPI, 10.0, 1.0).weight, photo.weight)


def test_limited_photo_side_yields_to_text():
    """Сторона фото (наклон 1 мм на высоте) в 2 мм от отбивки колонки (ровной): с ограничением веса поле на отбивке
    почти не трогается, без ограничения — гнёт её к стороне фото."""
    lay = Layout(W, H, (PageRect(40, 40, W / 2, H - 40), PageRect(W / 2, 40, W - 40, H - 40)), W / 2)
    ys = np.arange(150.0, 1050.0, mm(3.0))
    n = len(ys)
    x_text, x_photo = 400.0, 400.0 + mm(2.0)
    text = Samples(np.stack([np.full(n, x_text), ys], 1), np.zeros(n), np.ones(n), np.stack([np.full(n, x_text), ys], 1),
                   np.full(n, SampleKind.SIDE), np.zeros(n, int))  # fmt: skip
    tilt = (ys - ys.mean()) / np.ptp(ys) * mm(1.0)
    photo = Samples(np.stack([np.full(n, x_photo), ys], 1), tilt, np.ones(n), np.stack([x_photo + tilt, ys], 1),
                    np.full(n, SampleKind.SIDE), np.ones(n, int))  # fmt: skip
    spread = {}
    for name, weight in (("full", 1.0), ("limited", PHOTO_NEAR_WEIGHT)):
        h = text + limit_photo_weights(photo, text, DPI, PHOTO_NEAR_MM, weight)
        dx, _ = fit_field(h, Samples.empty(), lay, DPI, 4.0, 10.0).dense(W, H)
        spread[name] = float(np.ptp(dx[ys.astype(int), int(x_text)]))
    # Вес 0.5 у текста: изгиб отбивки от стороны фото — заметно меньше, чем при полном весе.
    assert spread["limited"] < 0.8 * spread["full"]


def test_partial_photo_gives_no_supports():
    """Фото без четырёхугольника (одна чёткая сторона) — ни одной выборки в опоры, хотя сторона чёткая."""
    from dataclasses import replace

    img, true = photo_page()
    photo = find_photos(img, [(true[0] - mm(2), true[1] - mm(2), true[2] + mm(2), true[3] + mm(9))], DPI)[0]
    partial = replace(photo, corners=None, sharp_sides=(False, False, True, False), quad_reason="не все стороны чёткие")
    dx, dy, count = photo_side_samples(partial, DPI)
    assert len(dx) == 0 and len(dy) == 0 and count == 0
