"""Выпрямление криволинейной рамки страницы в прямоугольник: гомография по углам (перспектива), затем Coons-патч по четырём кривым в выпрямленных координатах (изгиб), ``cv2.remap`` итога v2."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.page_boundary import PageBoundary


@dataclass(frozen=True)
class PageWarp:
    """Отображение прямоугольника страницы в картинку-источник.

    Поля:
        homography: Гомография «источник → выпрямленные координаты» по четырём углам рамки (3×3).
        top: Верх рамки в выпрямленных координатах (N, 2).
        bottom: Низ.
        left: Левая сторона.
        right: Правая сторона.
        size: Размер прямоугольника ``(W, H)``, px источника.
    """

    homography: np.ndarray
    top: np.ndarray
    bottom: np.ndarray
    left: np.ndarray
    right: np.ndarray
    size: tuple[int, int]


def apply_h(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Точки (…, 2) через гомографию ``h``."""
    shape = pts.shape
    flat = pts.reshape(-1, 2).astype(np.float64)
    hom = np.hstack([flat, np.ones((len(flat), 1))]) @ h.T
    return (hom[:, :2] / hom[:, 2:3]).reshape(shape)


def arc_resample(curve: np.ndarray, n: int) -> np.ndarray:
    """Ломаная, перевыбранная в ``n`` точек равномерно по длине дуги (концы сохраняются).

    Args:
        curve: Ломаная (M, 2).
        n: Число точек.

    Returns:
        (n, 2).
    """
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    target = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(target, s, curve[:, 0]), np.interp(target, s, curve[:, 1])], axis=1)


def arc_length(curve: np.ndarray) -> float:
    """Длина ломаной."""
    return float(np.linalg.norm(np.diff(curve, axis=0), axis=1).sum())


def plan_page_warp(boundary: PageBoundary, scale: float, offset: tuple[float, float]) -> PageWarp:
    """Отображение для рамки страницы в координатах источника.

    Рамка переводится в координаты источника (``x * scale - offset``); гомография ставит её углы в углы прямоугольника
    со сторонами — средними длинами хорд противоположных сторон; кривые рамки переносятся туда же. Размер итога —
    средние длины дуг противоположных сторон в выпрямленных координатах.

    Args:
        boundary: Рамка (px маски).
        scale: Множитель px маски → px кадра.
        offset: Сдвиг начала источника в кадре ``(x0, y0)`` (рамка итога v2).

    Returns:
        План выпрямления.
    """
    off = np.asarray(offset, np.float64)
    top, bottom = boundary.top * scale - off, boundary.bottom * scale - off
    left, right = boundary.left * scale - off, boundary.right * scale - off
    tl, tr, bl, br = top[0], top[-1], bottom[0], bottom[-1]
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    src = np.float32([tl, tr, br, bl])
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    hom = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
    r_top, r_bottom, r_left, r_right = (apply_h(hom, c) for c in (top, bottom, left, right))
    width = round((arc_length(r_top) + arc_length(r_bottom)) / 2)
    height = round((arc_length(r_left) + arc_length(r_right)) / 2)
    return PageWarp(hom, r_top, r_bottom, r_left, r_right, (width, height))


def coons_grid(warp: PageWarp) -> np.ndarray:
    """Coons-патч: для каждого пикселя прямоугольника (H, W) — точка в выпрямленных координатах.

    ``S(u, v) = (1−v)·T(u) + v·B(u) + (1−u)·L(v) + u·R(v) − билинейная по углам``; стороны перевыбраны по длине дуги,
    так что вдоль каждой стороны шаг равномерный.

    Args:
        warp: План выпрямления.

    Returns:
        Массив (H, W, 2) float32.
    """
    width, height = warp.size
    top = arc_resample(warp.top, width)
    bottom = arc_resample(warp.bottom, width)
    left = arc_resample(warp.left, height)
    right = arc_resample(warp.right, height)
    u = np.linspace(0.0, 1.0, width)[None, :, None]
    v = np.linspace(0.0, 1.0, height)[:, None, None]
    p00, p10, p01, p11 = top[0], top[-1], bottom[0], bottom[-1]
    grid = (1 - v) * top[None] + v * bottom[None] + (1 - u) * left[:, None] + u * right[:, None]
    grid -= (1 - u) * (1 - v) * p00 + u * (1 - v) * p10 + (1 - u) * v * p01 + u * v * p11
    return grid.astype(np.float32)


def warp_page(src: np.ndarray, warp: PageWarp) -> np.ndarray:
    """Страница-прямоугольник из источника: Coons-патч → обратная гомография → ``cv2.remap`` (кубическая).

    Args:
        src: Источник (итоговый TIFF v2).
        warp: План выпрямления.

    Returns:
        Картинка страницы размером ``warp.size``.
    """
    grid = coons_grid(warp)
    inv = np.linalg.inv(warp.homography)
    pts = apply_h(inv, grid).astype(np.float32)
    return cv2.remap(src, pts[..., 0], pts[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
