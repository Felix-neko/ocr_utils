"""Охватывающая трапеция маски листа: стороны Хубером по контуру (без рамки кадра), сдвиг наружу до квантиля, углы — пересечения."""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

# Доля точек стороны, которая должна оказаться внутри четырёхугольника: 0.995 — заусенцы рваного
# края, кусок соседней полосы в один-два пикселя и выступ пальца не раздувают сторону.
SIDE_QUANTILE = 0.995
# Допуск RANSAC для прямой стороны, px рабочего масштаба.
RANSAC_TOL = 3.0
# Полоса у прямой стороны, внутри которой точки контура считаются краем листа (заусенцы, надрывы),
# px рабочего масштаба; дальше — выступы, сторону не двигают.
SIDE_BAND = 25.0


class SideFit(Enum):
    """Подгонка стороны четырёхугольника: RANSAC-хорда со сдвигом по полосе у края или Хубер по всем точкам стороны."""

    RANSAC_BAND = "ransac_band"
    HUBER_ALL = "huber_all"


# Боевой способ: Хубер по всем точкам — край листа у корешка выгнут, хорда RANSAC срезает горб (1970 1_1570).
DEFAULT_SIDE_FIT = SideFit.HUBER_ALL


def _initial_quad(contour: np.ndarray) -> np.ndarray:
    """Начальный четырёхугольник — повёрнутый прямоугольник наименьшей площади (устойчив к выступам пальцев).

    Args:
        contour: Контур (N, 1, 2) int.

    Returns:
        Вершины (4, 2) float.
    """
    return cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float64)


def _ransac_line(pts: np.ndarray, tol: float, trials: int = 300, seed: int = 0) -> np.ndarray:
    """Прямая по точкам методом RANSAC (наибольшее число точек ближе ``tol``) с уточнением по МНК на опорных.

    Args:
        pts: Точки (N, 2).
        tol: Допуск опорной точки, px.
        trials: Число случайных пар.
        seed: Зерно генератора.

    Returns:
        Прямая (a, b, c) с единичной нормалью.
    """
    rng = np.random.default_rng(seed)
    best, best_count = None, -1
    for _ in range(trials):
        i, j = rng.choice(len(pts), 2, replace=False)
        if np.allclose(pts[i], pts[j]):
            continue
        line = _line_through(pts[i], pts[j] - pts[i])
        count = int((np.abs(pts @ line[:2] + line[2]) < tol).sum())
        if count > best_count:
            best, best_count = line, count
    inliers = pts[np.abs(pts @ best[:2] + best[2]) < tol]
    vx, vy, x0, y0 = cv2.fitLine(inliers.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return _line_through(np.array([x0, y0]), np.array([vx, vy]))


def _frame_edge_line(point: np.ndarray, h: int, w: int) -> np.ndarray:
    """Ближайшая к точке рамка кадра как прямая ``(a, b, c)``.

    Args:
        point: Точка (середина стороны).
        h: Высота кадра.
        w: Ширина кадра.

    Returns:
        Прямая рамки.
    """
    x, y = point
    edges = [
        (x, np.array([1.0, 0.0, 0.0])),
        (w - 1 - x, np.array([1.0, 0.0, -(w - 1.0)])),
        (y, np.array([0.0, 1.0, 0.0])),
        (h - 1 - y, np.array([0.0, 1.0, -(h - 1.0)])),
    ]
    return min(edges, key=lambda e: e[0])[1]


def _line_through(p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Прямая ``a·x + b·y + c = 0`` с единичной нормалью через точку ``p`` с направлением ``d``.

    Args:
        p: Точка (2,).
        d: Направление (2,).

    Returns:
        Коэффициенты (a, b, c).
    """
    d = d / (np.linalg.norm(d) + 1e-12)
    n = np.array([-d[1], d[0]])
    return np.array([n[0], n[1], -float(n @ p)])


def _intersect(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    """Точка пересечения двух прямых ``(a, b, c)``.

    Args:
        l1: Первая прямая.
        l2: Вторая прямая.

    Returns:
        Точка (2,) (почти параллельные дают далёкую точку — отсекается проверкой выпуклости у вызывающего).
    """
    p = np.cross(l1, l2)
    return p[:2] / (p[2] if abs(p[2]) > 1e-9 else 1e-9)


def fit_enclosing_quad(
    mask: np.ndarray, quantile: float = SIDE_QUANTILE, side_fit: "SideFit" = None
) -> np.ndarray | None:
    """Охватывающий четырёхугольник маски листа с перспективой (трапеция, не прямоугольник).

    Каждая сторона — прямая RANSAC по точкам контура, ближайшим к этой стороне начального прямоугольника;
    затем прямая сдвигается наружу параллельно себе, пока внутри не окажется доля ``quantile`` её точек
    из полосы :data:`SIDE_BAND` у прямой. Углы — пересечения соседних сторон. Выемки (палец, надрыв) лежат внутри и
    сторону не тянут; заусенцы наружу отсекает квантиль.

    Args:
        mask: Булева маска листа на рабочем масштабе.
        quantile: Доля точек стороны, которая должна быть внутри.
        side_fit: Способ подгонки стороны (:class:`SideFit`); None — :data:`DEFAULT_SIDE_FIT`.

    Returns:
        Вершины (4, 2) float в пикселях маски по часовой стрелке, либо None (пустая маска, вырожденный ответ).
    """
    side_fit = DEFAULT_SIDE_FIT if side_fit is None else side_fit
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    quad0 = _initial_quad(contour)
    # Порядок вершин по часовой стрелке вокруг центра — стороны идут подряд.
    centre = quad0.mean(axis=0)
    order = np.argsort(np.arctan2(quad0[:, 1] - centre[1], quad0[:, 0] - centre[0]))
    quad0 = quad0[order]
    pts = contour[:, 0, :].astype(np.float64)
    # Каждой точке контура — ближайшая сторона начального четырёхугольника.
    dists = np.stack([_segment_distance(pts, quad0[i], quad0[(i + 1) % 4]) for i in range(4)], axis=1)
    owner = np.argmin(dists, axis=1)
    h, w = mask.shape
    # Точки на рамке кадра — не край листа, а обрез кадра: в подгонку сторон не идут.
    on_border = (pts[:, 0] <= 1) | (pts[:, 1] <= 1) | (pts[:, 0] >= w - 2) | (pts[:, 1] >= h - 2)
    lines = []
    for i in range(4):
        side = pts[(owner == i) & ~on_border]
        a, b = quad0[i], quad0[(i + 1) % 4]
        if len(side) < 20:
            # Сторона целиком на обрезе кадра: берётся ближайшая рамка кадра.
            lines.append(_frame_edge_line((a + b) / 2, h, w))
            continue
        if side_fit is SideFit.RANSAC_BAND:
            line = _ransac_line(side, RANSAC_TOL)
        else:
            vx, vy, x0, y0 = cv2.fitLine(side.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            line = _line_through(np.array([x0, y0]), np.array([vx, vy]))
        # Нормаль — наружу (от центра).
        if line[:2] @ centre + line[2] > 0:
            line = -line
        signed = side @ line[:2] + line[2]
        if side_fit is SideFit.RANSAC_BAND:
            # Сдвиг только по точкам у самой стороны: выступы дальше полосы сторону не тянут.
            signed = signed[np.abs(signed) < SIDE_BAND]
        if len(signed):
            line[2] -= max(float(np.quantile(signed, quantile)), 0.0)
        lines.append(line)
    quad = np.array([_intersect(lines[i - 1], lines[i]) for i in range(4)])
    if not np.all(np.isfinite(quad)) or np.abs(quad).max() > 4 * max(h, w):
        return None
    if not cv2.isContourConvex(quad.astype(np.float32).reshape(-1, 1, 2)):
        return None
    return quad


def _segment_distance(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Расстояния от точек до отрезка ``ab``.

    Args:
        pts: Точки (N, 2).
        a: Начало отрезка.
        b: Конец отрезка.

    Returns:
        Расстояния (N,).
    """
    ab = b - a
    t = np.clip(((pts - a) @ ab) / (ab @ ab + 1e-12), 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)


def quad_mask(quad: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Растровая маска четырёхугольника.

    Args:
        quad: Вершины (4, 2).
        shape: (H, W) маски.

    Returns:
        Булева маска.
    """
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(quad).astype(np.int32)], 1)
    return m.astype(bool)


def keystone_deg(quad: np.ndarray) -> float:
    """Наибольший угол между противоположными сторонами четырёхугольника (мера трапеции), градусы.

    Args:
        quad: Вершины (4, 2) по кругу.

    Returns:
        0 для параллелограмма; растёт с перспективой.
    """
    sides = [quad[(i + 1) % 4] - quad[i] for i in range(4)]
    out = 0.0
    for i in (0, 1):
        a, b = sides[i], -sides[i + 2]
        cos = abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        out = max(out, float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
    return out
