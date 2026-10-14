"""Коррекция трапеций кадра перед surya layout: гомографии страниц по их трапециям (ближние стороны на месте, дальние — той же длины и вертикальны), выпрямленный холст и обратный перенос блоков."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Поле холста вокруг выпрямленных страниц, доля размера их общей рамки: края книги и выступы блоков за трапецию.
CANVAS_MARGIN = 0.08
# Преобразование считается вырожденным, если площадь страницы меняется сильнее этого (в обе стороны).
MAX_AREA_RATIO = 2.0


@dataclass(frozen=True)
class KeystoneWarp:
    """Выпрямление кадра по трапециям страниц.

    Поля:
        hs: Гомографии страниц слева направо (координаты кадра → холст).
        size: Размер холста ``(W, H)``.
        spine_x: Абсцисса стыка страниц на холсте (две страницы) или None (одна).
    """

    hs: list[np.ndarray]
    size: tuple[int, int]
    spine_x: float | None


def vertical_sides(quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Левая и правая «вертикальные» стороны трапеции, каждая — ``(верх, низ)``.

    Args:
        quad: Вершины (4, 2).

    Returns:
        ``(левая (2, 2), правая (2, 2))``, точки сверху вниз.
    """
    order = np.argsort(quad[:, 0])
    left, right = quad[order[:2]], quad[order[2:]]
    return left[np.argsort(left[:, 1])], right[np.argsort(right[:, 1])]


def tilt(side: np.ndarray) -> float:
    """Наклон стороны от вертикали, рад (плюс — низ правее верха).

    Args:
        side: ``(верх, низ)``.

    Returns:
        Угол.
    """
    d = side[1] - side[0]
    return float(np.arctan2(d[0], d[1]))


def rotate(points: np.ndarray, angle: float, centre: np.ndarray) -> np.ndarray:
    """Поворот точек вокруг ``centre`` так, что сторона с наклоном ``angle`` становится вертикальной.

    Args:
        points: Точки (N, 2).
        angle: Наклон стороны, рад (``tilt``).
        centre: Центр поворота.

    Returns:
        Точки (N, 2).
    """
    c, s = np.cos(angle), np.sin(angle)
    p = points - centre
    return np.c_[p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c] + centre


def upright(side: np.ndarray, length: float, angle: float, centre: np.ndarray) -> np.ndarray:
    """Сторона после поворота, поставленная вертикально через свою середину с длиной ``length``.

    Args:
        side: ``(верх, низ)``.
        length: Длина, px.
        angle: Угол поворота кадра (``rotate``).
        centre: Центр поворота.

    Returns:
        ``(верх, низ)``.
    """
    mid = rotate(side.mean(axis=0, keepdims=True), angle, centre)[0]
    half = np.array([0.0, length / 2])
    return np.array([mid - half, mid + half])


def page_target(near: np.ndarray, far: np.ndarray, angle: float, centre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Исходный и целевой четырёхугольники страницы: ближняя сторона только поворачивается, дальняя — вертикальна и
    той же длины, что ближняя.

    Args:
        near: Ближняя (сохраняемая) сторона ``(верх, низ)``.
        far: Дальняя сторона ``(верх, низ)``.
        angle: Угол поворота кадра.
        centre: Центр поворота.

    Returns:
        ``(исходные 4 точки, целевые 4 точки)`` в порядке: ближняя верх, ближняя низ, дальняя низ, дальняя верх.
    """
    length = float(np.linalg.norm(near[1] - near[0]))
    dst_near = rotate(near, angle, centre)
    dst_far = upright(far, length, angle, centre)
    src = np.array([near[0], near[1], far[1], far[0]], np.float32)
    dst = np.array([dst_near[0], dst_near[1], dst_far[1], dst_far[0]], np.float32)
    return src, dst


def area(quad: np.ndarray) -> float:
    """Площадь четырёхугольника (формула Гаусса)."""
    x, y = quad[:, 0], quad[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def plan_warp(quads: list[np.ndarray]) -> KeystoneWarp | None:
    """Гомографии выпрямления по трапециям страниц (координаты кадра).

    Одна страница: короткая вертикальная сторона остаётся, кадр поворачивается так, чтобы она стала вертикальной,
    длинная сторона — вертикальна через свою середину и той же длины, что короткая. Две страницы: ближние к стыку
    стороны остаются, поворот — на средний наклон ближних сторон, дальние стороны — вертикальны и той же длины, что
    ближняя сторона своей страницы.

    Args:
        quads: Трапеции страниц слева направо (4, 2), px кадра.

    Returns:
        Выпрямление или None (страниц нет / больше двух / преобразование вырождено).
    """
    if not quads or len(quads) > 2:
        return None
    sides = [vertical_sides(q) for q in quads]
    centre = np.concatenate(quads).mean(axis=0)
    pairs = []
    if len(quads) == 1:
        left, right = sides[0]
        ln, rn = np.linalg.norm(left[1] - left[0]), np.linalg.norm(right[1] - right[0])
        near, far = (left, right) if ln <= rn else (right, left)
        pairs.append((near, far))
        angle = tilt(near)
    else:
        (l_left, l_right), (r_left, r_right) = sides
        pairs = [(l_right, l_left), (r_left, r_right)]
        angle = (tilt(l_right) + tilt(r_left)) / 2
    targets = [page_target(near, far, angle, centre) for near, far in pairs]
    for src, dst in targets:
        ratio = area(dst) / max(area(src), 1.0)
        if not cv2.isContourConvex(dst.reshape(-1, 1, 2)) or not 1 / MAX_AREA_RATIO <= ratio <= MAX_AREA_RATIO:
            return None
    # Холст — общая рамка целевых страниц с полем.
    allpts = np.concatenate([dst for _, dst in targets])
    lo, hi = allpts.min(axis=0), allpts.max(axis=0)
    margin = CANVAS_MARGIN * (hi - lo)
    lo, hi = lo - margin, hi + margin
    shift = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]], [0, 0, 1]], np.float64)
    hs = [shift @ cv2.getPerspectiveTransform(src, dst) for src, dst in targets]
    size = (int(np.ceil(hi[0] - lo[0])), int(np.ceil(hi[1] - lo[1])))
    spine_x = None
    if len(quads) == 2:
        spine_x = float((targets[0][1][:2, 0].mean() + targets[1][1][:2, 0].mean()) / 2 - lo[0])
    return KeystoneWarp(hs, size, spine_x)


def warp_for_layout(bgr: np.ndarray, warp: KeystoneWarp) -> np.ndarray:
    """Выпрямленный холст: каждая страница своей гомографией, при двух страницах — склейка по стыку.

    Args:
        bgr: Кадр.
        warp: Выпрямление.

    Returns:
        Холст BGR размера ``warp.size``.
    """
    first = cv2.warpPerspective(bgr, warp.hs[0], warp.size, flags=cv2.INTER_AREA, borderValue=(0, 0, 0))
    if warp.spine_x is None:
        return first
    second = cv2.warpPerspective(bgr, warp.hs[1], warp.size, flags=cv2.INTER_AREA, borderValue=(0, 0, 0))
    split = int(round(warp.spine_x))
    first[:, split:] = second[:, split:]
    return first


def page_of(point: np.ndarray, warp: KeystoneWarp) -> int:
    """Номер страницы точки холста (по стыку)."""
    if warp.spine_x is None:
        return 0
    return 0 if point[0] < warp.spine_x else 1


def warp_polygons(polys: list[np.ndarray], warp: KeystoneWarp) -> list[np.ndarray]:
    """Полигоны кадра → холст (каждая точка гомографией страницы, к которой относится центр полигона на холсте).

    Нужна для проверок: страница выбирается по центру после переноса первой гомографией.

    Args:
        polys: Полигоны (N, 2), px кадра.
        warp: Выпрямление.

    Returns:
        Полигоны на холсте.
    """
    out = []
    for poly in polys:
        pts = np.asarray(poly, np.float32).reshape(-1, 1, 2)
        moved = cv2.perspectiveTransform(pts, warp.hs[0]).reshape(-1, 2)
        k = page_of(moved.mean(axis=0), warp)
        out.append(cv2.perspectiveTransform(pts, warp.hs[k]).reshape(-1, 2))
    return out


def unwarp_polygons(polys: list[np.ndarray], warp: KeystoneWarp) -> list[np.ndarray]:
    """Блоки surya с холста → кадр: обратная гомография страницы, к которой относится центр блока.

    Осевая рамка блока превращается в наклонный четырёхугольник по перспективе страницы.

    Args:
        polys: Полигоны блоков на холсте (N, 2).
        warp: Выпрямление.

    Returns:
        Полигоны в px кадра (float32).
    """
    inverse = [np.linalg.inv(h) for h in warp.hs]
    out = []
    for poly in polys:
        pts = np.asarray(poly, np.float32).reshape(-1, 1, 2)
        k = page_of(pts.reshape(-1, 2).mean(axis=0), warp)
        out.append(cv2.perspectiveTransform(pts, inverse[k]).reshape(-1, 2).astype(np.float32))
    return out
