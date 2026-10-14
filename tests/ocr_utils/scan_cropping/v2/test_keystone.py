"""Тесты коррекции трапеций перед surya на синтетике: одна и две страницы, вертикальность и длины сторон, обратный перенос."""

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.keystone import plan_warp, tilt, unwarp_polygons, vertical_sides, warp_polygons


def moved(quad: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Вершины трапеции под гомографией."""
    return cv2.perspectiveTransform(quad.reshape(-1, 1, 2).astype(np.float32), h).reshape(-1, 2)


def length(side: np.ndarray) -> float:
    """Длина стороны ``(верх, низ)``."""
    return float(np.linalg.norm(side[1] - side[0]))


def rotated(quad: np.ndarray, deg: float) -> np.ndarray:
    """Трапеция, повёрнутая на ``deg`` вокруг своего центра."""
    a = np.radians(deg)
    c = quad.mean(axis=0)
    r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (quad - c) @ r.T + c


def test_single_page():
    """Одна страница: короткая сторона той же длины, длинная стала такой же, обе вертикальны."""
    quad = rotated(np.array([[100, 100], [500, 150], [500, 450], [100, 500]], float), 5)
    warp = plan_warp([quad])
    assert warp is not None and warp.spine_x is None
    left0, right0 = vertical_sides(quad)
    left, right = vertical_sides(moved(quad, warp.hs[0]))
    short = min(length(left0), length(right0))
    assert abs(length(left) - short) < 1 and abs(length(right) - short) < 1
    assert abs(np.degrees(tilt(left))) < 0.5 and abs(np.degrees(tilt(right))) < 0.5


def test_two_pages():
    """Две страницы: ближние стороны сохраняют длину и вертикальны, дальние — вертикальны и той же длины."""
    lq = rotated(np.array([[100, 140], [500, 100], [500, 520], [100, 480]], float), 3)
    rq = rotated(np.array([[510, 100], [900, 150], [900, 470], [510, 520]], float), 3)
    warp = plan_warp([lq, rq])
    assert warp is not None and warp.spine_x is not None
    for k, (q, near_right) in enumerate(((lq, True), (rq, False))):
        l0, r0 = vertical_sides(q)
        left, right = vertical_sides(moved(q, warp.hs[k]))
        near0, near, far = (r0, right, left) if near_right else (l0, left, right)
        assert abs(length(near) - length(near0)) < 1
        assert abs(length(far) - length(near0)) < 1
        assert abs(np.degrees(tilt(near))) < 0.5 and abs(np.degrees(tilt(far))) < 0.5


def test_roundtrip_and_tilted_blocks():
    """Полигоны туда-обратно совпадают; осевая рамка на холсте возвращается наклонным четырёхугольником."""
    quad = rotated(np.array([[100, 100], [500, 150], [500, 450], [100, 500]], float), 5)
    warp = plan_warp([quad])
    block = np.array([[200, 250], [400, 250], [400, 320], [200, 320]], np.float32)
    back = unwarp_polygons(warp_polygons([block], warp), warp)[0]
    assert np.abs(back - block).max() < 1
    box = warp_polygons([quad.astype(np.float32)], warp)[0]
    x0, y0 = box.min(axis=0) + 20
    x1, y1 = box.max(axis=0) - 20
    rect = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    tilted = unwarp_polygons([rect], warp)[0]
    assert abs(tilted[0, 1] - tilted[1, 1]) > 5  # верхняя сторона уже не горизонтальна
    inside = [
        cv2.pointPolygonTest(quad.astype(np.float32).reshape(-1, 1, 2), tuple(map(float, p)), False) for p in tilted
    ]
    assert min(inside) >= 0


def test_no_pages():
    """Без трапеций выпрямления нет."""
    assert plan_warp([]) is None
