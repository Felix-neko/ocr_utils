"""Тесты меры перспективного масштаба страницы по её высоте (``rectify/page_scale.py``): равная высота — без изменений, перспектива — точная ширина ближней к камере страницы и равномерные буквы по плоскости, впадина у корешка и поворот книги в кадре не в счёт, наклонённая страница и пределы."""

import numpy as np
import pytest

from ocr_utils.scan_cropping.v2.page_frames import PageFrame
from ocr_utils.scan_cropping.v2.page_pose import PagePose
from ocr_utils.scan_cropping.v2.rectify.coons import page_patch, plan_layout
from ocr_utils.scan_cropping.v2.rectify.page_scale import ScaleParams, ScaleSkip, page_scale

PX_PER_MM = 10.0
TANGENT_PX = 20.0
# Главная точка снимка синтетики — на линии корешка посередине высоты (камера над корешком).
CENTRE = np.array([0.0, 0.0])


def identity(points: np.ndarray) -> np.ndarray:
    """Перевод координат без сдвига."""
    return np.asarray(points, np.float64)


def perspective_right_page(depth_ratio: float, dip_px: float = 0.0) -> PageFrame:
    """Правая страница 110×190 мм (корешок на x = 0), снятая так, что внешний край в ``depth_ratio`` раз ближе к камере.

    Точка плоскости ``X ∈ [0, W]`` на расстоянии ``z(X) = 1 − (1 − 1/depth_ratio)·X/W`` (у корешка 1): на снимке
    ``x = X / z``, высота страницы ``H / z``. ``dip_px`` — ложная «впадина» маски у корешка (верх ниже на столько px в
    первых 10 мм).
    """
    w, h = 110.0 * PX_PER_MM, 190.0 * PX_PER_MM
    xs = np.linspace(0.0, w, 300)
    z = 1.0 - (1.0 - 1.0 / depth_ratio) * xs / w
    img_x = xs / z
    top_y, bottom_y = -h / 2 / z, h / 2 / z
    top_y = top_y + dip_px * np.clip(1.0 - img_x / (10.0 * PX_PER_MM), 0.0, 1.0)
    top = np.c_[img_x, top_y][::-1]  # от внешнего угла к корешку
    bottom = np.c_[img_x, bottom_y][::-1]
    far = np.array([top[0], bottom[0]])
    spine = np.array([top[-1], bottom[-1]])
    return PageFrame(top, bottom, far, spine, np.zeros((0, 2)))


def test_flat_page_keeps_width():
    frame = perspective_right_page(1.0)
    patch = page_patch(frame, False, identity, TANGENT_PX)
    sc = page_scale(patch, 190.0 * PX_PER_MM, False, PagePose.FLAT, CENTRE, PX_PER_MM, ScaleParams())
    assert sc.skip is ScaleSkip.NONE and sc.ratio == pytest.approx(1.0, abs=1e-3)


def test_closer_outer_edge_narrows_page_and_evens_letters():
    # Внешний край на 8 % ближе: по дуге страница шире 110 мм, по масштабу высоты — 110 мм.
    frame = perspective_right_page(1.08)
    patch = page_patch(frame, False, identity, TANGENT_PX)
    sc = page_scale(patch, 190.0 * PX_PER_MM, False, PagePose.FLAT, CENTRE, PX_PER_MM, ScaleParams())
    assert sc.arc_width > 113.0 * PX_PER_MM
    assert sc.target_width == pytest.approx(110.0 * PX_PER_MM, rel=0.002)
    # Параметр по плоскости: равные доли ширины → равные мм страницы (точки X через 11 мм).
    us, ts = sc.u_warp
    u = np.linspace(0.0, 1.0, 11)
    x_img = patch.top.at(np.interp(u, us, ts))[:, 0]
    w = 110.0 * PX_PER_MM
    # Обратный перевод снимка в плоскость: X = x·z, z = 1 − k·X/W ⇒ X = x / (1 + k·x/W).
    k = 1.0 - 1.0 / 1.08
    plane = x_img / (1.0 + k * x_img / w)
    np.testing.assert_allclose(np.sort(plane), np.linspace(0.0, w, 11), atol=0.002 * w)


def test_spine_dip_ignored():
    # Ложная впадина маски у корешка (верх ниже на 40 px в первых 10 мм) мера не замечает: зона у корешка не в счёт.
    clean = page_patch(perspective_right_page(1.08), False, identity, TANGENT_PX)
    dipped = page_patch(perspective_right_page(1.08, dip_px=40.0), False, identity, TANGENT_PX)
    a = page_scale(clean, 1900.0, False, PagePose.FLAT, CENTRE, PX_PER_MM, ScaleParams())
    b = page_scale(dipped, 1900.0, False, PagePose.FLAT, CENTRE, PX_PER_MM, ScaleParams())
    assert b.target_width == pytest.approx(a.target_width, rel=0.002)


def test_tilted_page_and_limits():
    patch = page_patch(perspective_right_page(1.3), False, identity, TANGENT_PX)
    tilted = page_scale(patch, 1900.0, False, PagePose.RAISED, CENTRE, PX_PER_MM, ScaleParams())
    assert tilted.skip is ScaleSkip.TILTED and tilted.target_width == tilted.arc_width and tilted.u_warp is None
    # Внешний край на 30 % ближе: сужение больше 10 % упирается в предел.
    sc = page_scale(patch, 1900.0, False, PagePose.FLAT, CENTRE, PX_PER_MM, ScaleParams(max_shrink=0.10))
    assert sc.clamped and sc.ratio == pytest.approx(0.90)


def test_layout_uses_scaled_width():
    patch = page_patch(perspective_right_page(1.08), False, identity, TANGENT_PX)
    sc = page_scale(patch, 1900.0, False, PagePose.FLAT, CENTRE, PX_PER_MM, ScaleParams())
    layout = plan_layout((patch,), (2500, 1400), widths=(sc.target_width,))
    assert layout.rects[0].width == pytest.approx(sc.target_width)


def test_rotated_book_same_scale():
    # Книга повёрнута в кадре на 6° вокруг главной точки: мера та же (высота — по парам точек, координата — по хорде).
    frame = perspective_right_page(1.08)
    a = np.radians(6.0)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    turned = PageFrame(*(np.asarray(getattr(frame, n)) @ rot.T for n in ("top", "bottom", "far", "spine")),
                       np.zeros((0, 2)))  # fmt: skip
    base = page_scale(page_patch(frame, False, identity, TANGENT_PX), 1900.0, False, PagePose.FLAT, CENTRE, PX_PER_MM,
                      ScaleParams())  # fmt: skip
    rotated = page_scale(page_patch(turned, False, identity, TANGENT_PX), 1900.0, False, PagePose.FLAT, CENTRE,
                         PX_PER_MM, ScaleParams())  # fmt: skip
    assert rotated.target_width == pytest.approx(base.target_width, rel=1e-3)


def test_warp_table_strictly_increasing_with_hooked_midline():
    # Средняя линия у корешка загибается назад на 3 px: таблица параметра строго возрастает, продолжение за край конечно.
    frame = perspective_right_page(1.05)
    top, bottom = frame.top.copy(), frame.bottom.copy()
    top[-6:, 0] += np.linspace(0.0, 3.0, 6)
    bottom[-6:, 0] += np.linspace(0.0, 3.0, 6)
    hooked = PageFrame(top, bottom, frame.far, np.array([top[-1], bottom[-1]]), np.zeros((0, 2)))
    patch = page_patch(hooked, False, identity, TANGENT_PX)
    sc = page_scale(patch, 1900.0, False, PagePose.FLAT, CENTRE, PX_PER_MM, ScaleParams())
    us, _ = sc.u_warp
    assert np.all(np.diff(us) > 0)
    from dataclasses import replace

    warped = replace(patch, u_warp=sc.u_warp)
    assert np.all(np.isfinite(warped.arc_of(np.array([-0.2, 0.5, 1.2]))))
