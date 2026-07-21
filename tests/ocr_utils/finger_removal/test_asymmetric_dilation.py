"""Тесты асимметричной дилатации зон пальца (``FingerZoneDilation``)."""

import cv2
import numpy as np
import pytest

from ocr_utils.finger_removal.asymmetric_dilation import (
    DEFAULT_MAX_ASYMMETRIC_DILATION_RATIO,
    FingerZoneDilation,
    dilate_finger_zones,
)

# Кадр 1000x1000 — квадратный, чтобы нормировки по X и Y были симметричны
H = W = 1000
# Зоны пальца (x_min, y_min, x_max, y_max), прилегающие к разным сторонам.
# По «длинной» оси зона вытянута — как реальный палец вдоль края.
BBOX_LEFT = (0, 350, 60, 650)
BBOX_RIGHT = (940, 350, 1000, 650)
BBOX_TOP = (350, 0, 650, 60)
BBOX_BOTTOM = (350, 940, 650, 1000)
BBOX_CORNER_TL = (0, 0, 60, 60)
BBOX_CORNER_BR = (940, 940, 1000, 1000)


@pytest.fixture
def helper():
    return FingerZoneDilation((H, W))


# ----------------------------------------------------------------------
# Определение ближайшей стороны
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "bbox,expected", [(BBOX_LEFT, "left"), (BBOX_RIGHT, "right"), (BBOX_TOP, "top"), (BBOX_BOTTOM, "bottom")]
)
def test_nearest_side(helper, bbox, expected):
    assert helper.nearest_side(bbox) == expected


def test_nearest_side_corner_is_one_of_adjacent(helper):
    """В углу ближайшей может быть любая из двух прилегающих сторон."""
    assert helper.nearest_side(BBOX_CORNER_TL) in ("left", "top")


# ----------------------------------------------------------------------
# Коэффициенты дилатации
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bbox", [BBOX_LEFT, BBOX_RIGHT])
def test_side_finger_grows_along_y(helper, bbox):
    """Боковой палец: x_ratio → 1, y_ratio → 1 + MAX."""
    x_ratio, y_ratio = helper.ratios(bbox)
    assert x_ratio == pytest.approx(1.0, abs=0.2)
    assert y_ratio == pytest.approx(1.0 + DEFAULT_MAX_ASYMMETRIC_DILATION_RATIO, abs=0.2)
    assert y_ratio > x_ratio


@pytest.mark.parametrize("bbox", [BBOX_TOP, BBOX_BOTTOM])
def test_top_bottom_finger_grows_along_x(helper, bbox):
    """Верхний/нижний палец: x_ratio → 1 + MAX, y_ratio → 1."""
    x_ratio, y_ratio = helper.ratios(bbox)
    assert x_ratio == pytest.approx(1.0 + DEFAULT_MAX_ASYMMETRIC_DILATION_RATIO, abs=0.2)
    assert y_ratio == pytest.approx(1.0, abs=0.2)
    assert x_ratio > y_ratio


@pytest.mark.parametrize("bbox", [BBOX_CORNER_TL, BBOX_CORNER_BR])
def test_corner_finger_grows_evenly(helper, bbox):
    """Угловой палец: оба коэффициента → 1 + MAX/2 и равны между собой."""
    x_ratio, y_ratio = helper.ratios(bbox)
    expected = 1.0 + DEFAULT_MAX_ASYMMETRIC_DILATION_RATIO / 2.0
    assert x_ratio == pytest.approx(expected, abs=0.2)
    assert y_ratio == pytest.approx(expected, abs=0.2)
    assert x_ratio == pytest.approx(y_ratio, abs=1e-6)


def test_weights_sum_to_one(helper):
    """Веса перекрёстные и нормированы: w_lr + w_tb == 1 для любой зоны."""
    for bbox in (BBOX_LEFT, BBOX_TOP, BBOX_CORNER_TL, (400, 400, 600, 600)):
        w_lr, w_tb = helper.side_weights(bbox)
        assert w_lr + w_tb == pytest.approx(1.0)


def test_ratios_within_bounds(helper):
    """Коэффициенты всегда в [1, 1 + MAX] — дилатация не может уменьшать зону."""
    for bbox in (BBOX_LEFT, BBOX_RIGHT, BBOX_TOP, BBOX_BOTTOM, BBOX_CORNER_TL, (400, 400, 600, 600)):
        for r in helper.ratios(bbox):
            assert 1.0 <= r <= 1.0 + DEFAULT_MAX_ASYMMETRIC_DILATION_RATIO + 1e-9


def test_max_ratio_zero_is_symmetric():
    """max_ratio=0 (из CLI --max-asymmetric-dilation-ratio=0) — прежняя круговая дилатация."""
    zero = FingerZoneDilation((H, W), max_ratio=0.0)
    for bbox in (BBOX_LEFT, BBOX_TOP, BBOX_CORNER_TL):
        assert zero.ratios(bbox) == pytest.approx((1.0, 1.0))


@pytest.mark.parametrize("ratio", [1.0, 2.0, 4.0])
def test_larger_max_ratio_grows_more(ratio):
    """Чем больше max_ratio, тем сильнее перекос: y_ratio бокового пальца ≈ 1 + ratio."""
    helper = FingerZoneDilation((H, W), max_ratio=ratio)
    x_ratio, y_ratio = helper.ratios(BBOX_LEFT)
    assert y_ratio == pytest.approx(1.0 + ratio, abs=0.1 * ratio + 0.05)
    assert 1.0 <= x_ratio <= 1.0 + ratio


def test_max_ratio_propagates_to_dilation():
    """max_ratio доходит до реальной дилатации: больший ratio даёт больший прирост по Y."""
    mask = _mask_with(BBOX_LEFT)
    small = dilate_finger_zones(mask, dilate_px=20, max_ratio=0.0)
    large = dilate_finger_zones(mask, dilate_px=20, max_ratio=4.0)
    ys_s = np.where(small > 0)[0]
    ys_l = np.where(large > 0)[0]
    assert (ys_l.max() - ys_l.min()) > (ys_s.max() - ys_s.min())


def test_disabled_gives_symmetric_ratios():
    """Флаг enabled=False возвращает прежнее круговое поведение (1, 1)."""
    off = FingerZoneDilation((H, W), enabled=False)
    for bbox in (BBOX_LEFT, BBOX_TOP, BBOX_CORNER_TL):
        assert off.ratios(bbox) == (1.0, 1.0)


def test_kernel_radii_scale_with_base(helper):
    """Радиусы ядра пропорциональны базовому dilate_px и не меньше 1."""
    kx1, ky1 = helper.kernel_radii(BBOX_LEFT, 10)
    kx2, ky2 = helper.kernel_radii(BBOX_LEFT, 20)
    assert kx2 == pytest.approx(2 * kx1, abs=1)
    assert ky2 == pytest.approx(2 * ky1, abs=1)
    assert min(helper.kernel_radii(BBOX_LEFT, 0)) >= 1


# ----------------------------------------------------------------------
# Применение к маске
# ----------------------------------------------------------------------


def _mask_with(bbox):
    m = np.zeros((H, W), dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    m[y0:y1, x0:x1] = 255
    return m


def test_dilate_side_finger_extends_more_vertically():
    """Боковая зона после дилатации прирастает по высоте сильнее, чем по ширине."""
    mask = _mask_with(BBOX_LEFT)
    out = dilate_finger_zones(mask, dilate_px=20)
    ys, xs = np.where(out > 0)
    grew_h = (ys.max() - ys.min()) - (BBOX_LEFT[3] - BBOX_LEFT[1])
    grew_w = (xs.max() - xs.min()) - (BBOX_LEFT[2] - BBOX_LEFT[0])
    assert grew_h > grew_w


def test_dilate_top_finger_extends_more_horizontally():
    """Верхняя зона после дилатации прирастает по ширине сильнее, чем по высоте."""
    mask = _mask_with(BBOX_TOP)
    out = dilate_finger_zones(mask, dilate_px=20)
    ys, xs = np.where(out > 0)
    grew_h = (ys.max() - ys.min()) - (BBOX_TOP[3] - BBOX_TOP[1])
    grew_w = (xs.max() - xs.min()) - (BBOX_TOP[2] - BBOX_TOP[0])
    assert grew_w > grew_h


def test_dilate_never_shrinks():
    """Дилатация только расширяет: исходная маска целиком внутри результата."""
    mask = _mask_with(BBOX_LEFT)
    out = dilate_finger_zones(mask, dilate_px=15)
    assert np.all(out[mask > 0] > 0)
    assert int(np.count_nonzero(out)) > int(np.count_nonzero(mask))


def test_two_zones_dilated_independently():
    """Два пальца с РАЗНЫХ сторон получают разный перекос дилатации."""
    mask = cv2.bitwise_or(_mask_with(BBOX_LEFT), _mask_with(BBOX_TOP))
    out = dilate_finger_zones(mask, dilate_px=20)
    # левая зона: смотрим её половину кадра — прирост по Y больше
    left_half = out[:, : W // 2].copy()
    left_half[: BBOX_TOP[3] + 100, BBOX_TOP[0] :] = 0  # убираем влияние верхней зоны
    ys, xs = np.where(left_half > 0)
    assert (ys.max() - ys.min()) > (xs.max() - xs.min())


def test_empty_mask_returns_empty():
    """Пустая маска остаётся пустой (и не падает)."""
    empty = np.zeros((H, W), dtype=np.uint8)
    out = dilate_finger_zones(empty, dilate_px=20)
    assert int(np.count_nonzero(out)) == 0


def test_zero_dilate_px_is_noop():
    """dilate_px=0 не меняет маску."""
    mask = _mask_with(BBOX_LEFT)
    out = dilate_finger_zones(mask, dilate_px=0)
    assert np.array_equal(out, mask)
