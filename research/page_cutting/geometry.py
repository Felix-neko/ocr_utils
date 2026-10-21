"""Стороны страницы по её трапеции: какая сторона верх, низ, у корешка и дальняя; точки контура маски по сторонам с отклонением от прямой стороны."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class Side(Enum):
    """Сторона страницы."""

    TOP = "top"
    BOTTOM = "bottom"
    # Сторона, обращённая к соседней странице разворота (у корешка).
    SPINE = "spine"
    # Сторона у рамки кадра (наружный обрез листа).
    FAR = "far"


@dataclass(frozen=True)
class SideProfile:
    """Точки контура маски у одной стороны трапеции.

    Поля:
        side: Какая это сторона.
        a: Начало стороны (px), для верха и низа — левый конец, для боковых — верхний.
        b: Конец стороны (px).
        t: Положение точки вдоль стороны, доля от 0 (у ``a``) до 1 (у ``b``).
        d: Отклонение точки от прямой стороны, px; плюс — внутрь страницы, минус — наружу.
        clean: Точка годится для подгонки (не на рамке кадра и не в зоне пальца).
    """

    side: Side
    a: np.ndarray
    b: np.ndarray
    t: np.ndarray
    d: np.ndarray
    clean: np.ndarray


def quad_sides(quad: np.ndarray, spine_right: bool | None) -> dict[Side, tuple[np.ndarray, np.ndarray]]:
    """Стороны трапеции по месту: верх, низ, у корешка, дальняя.

    Вершины берутся по осям: две верхние — по наименьшему Y, левая из них — по X; так же низ. Порядок вершин
    в ``pages.json`` для этого не нужен.

    Args:
        quad: Вершины трапеции (4, 2), px.
        spine_right: Корешок справа от страницы (левая страница разворота) — True, слева — False; одиночная страница —
            None, тогда «у корешка» считается правая сторона, а «дальней» — левая.

    Returns:
        Словарь сторона → (начало, конец): верх и низ слева направо, боковые сверху вниз.
    """
    pts = np.asarray(quad, np.float64)
    order = np.argsort(pts[:, 1])
    top = pts[order[:2]][np.argsort(pts[order[:2], 0])]
    bottom = pts[order[2:]][np.argsort(pts[order[2:], 0])]
    left = (top[0], bottom[0])
    right = (top[1], bottom[1])
    spine, far = (right, left) if spine_right in (True, None) else (left, right)
    return {Side.TOP: (top[0], top[1]), Side.BOTTOM: (bottom[0], bottom[1]), Side.SPINE: spine, Side.FAR: far}


def mask_contour(mask: np.ndarray) -> np.ndarray:
    """Точки наружного контура самой крупной компоненты маски, (N, 2) float, px.

    Args:
        mask: Булева маска.

    Returns:
        Точки контура (пустой массив, если маска пуста).
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2))
    return max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)


def inward_normal(a: np.ndarray, b: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Единичная нормаль к отрезку ``ab``, смотрящая в сторону ``centre`` (внутрь страницы)."""
    d = b - a
    n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-12)
    return n if (centre - a) @ n >= 0 else -n


def side_profiles(
    mask: np.ndarray, quad: np.ndarray, spine_right: bool | None, exclude: np.ndarray | None = None, border_px: int = 2
) -> dict[Side, SideProfile]:
    """Раздача точек контура маски по сторонам трапеции и отклонение каждой точки от своей стороны.

    Точка контура уходит к ближайшей стороне (по расстоянию до отрезка). Точки у рамки кадра (там маска обрезана
    кадром, а не листом) и в зоне ``exclude`` (палец) помечаются как негодные.

    Args:
        mask: Маска страницы.
        quad: Трапеция страницы (4, 2), в тех же px, что маска.
        spine_right: Где корешок (см. ``quad_sides``).
        exclude: Зона, точки в которой негодны (палец), или None.
        border_px: Ширина полосы у рамки кадра, где точки негодны, px.

    Returns:
        Профили четырёх сторон.
    """
    pts = mask_contour(mask)
    sides = quad_sides(quad, spine_right)
    centre = np.asarray(quad, np.float64).mean(axis=0)
    h, w = mask.shape
    # Расстояние точки до каждого отрезка-стороны; точка достаётся ближайшей стороне.
    dists = []
    for a, b in sides.values():
        ab = b - a
        t = np.clip(((pts - a) @ ab) / (ab @ ab + 1e-12), 0.0, 1.0)
        dists.append(np.linalg.norm(pts - (a + t[:, None] * ab), axis=1))
    owner = np.argmin(np.stack(dists), axis=0) if len(pts) else np.zeros(0, int)
    on_border = (pts[:, 0] < border_px) | (pts[:, 1] < border_px)
    on_border |= (pts[:, 0] >= w - 1 - border_px) | (pts[:, 1] >= h - 1 - border_px)
    if exclude is not None and len(pts):
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
        on_border |= exclude[yi, xi]
    out = {}
    for i, (side, (a, b)) in enumerate(sides.items()):
        sel = owner == i
        p = pts[sel]
        ab = b - a
        t = ((p - a) @ ab) / (ab @ ab + 1e-12)
        d = (p - a) @ inward_normal(a, b, centre)
        # Вдоль стороны — по возрастанию t, чтобы профиль можно было сглаживать окном.
        order = np.argsort(t)
        out[side] = SideProfile(side, a, b, t[order], d[order], ~on_border[sel][order])
    return out


def smooth_profile(t: np.ndarray, d: np.ndarray, length_px: float, window_px: float) -> np.ndarray:
    """Медианное сглаживание профиля стороны окном вдоль стороны.

    Args:
        t: Положение точек вдоль стороны, доля (по возрастанию).
        d: Отклонения точек, px.
        length_px: Длина стороны, px (переводит окно в доли).
        window_px: Ширина окна вдоль стороны, px.

    Returns:
        Сглаженные отклонения (по точке на точку).
    """
    if not len(t):
        return d
    half = window_px / 2 / max(length_px, 1.0)
    lo = np.searchsorted(t, t - half)
    hi = np.searchsorted(t, t + half, side="right")
    return np.array([np.median(d[i:j]) for i, j in zip(lo, hi)])
