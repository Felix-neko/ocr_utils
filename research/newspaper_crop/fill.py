"""Выход под ScanTailor: прямоугольник копирования из четырёхугольника зоны — «по месту» или с выпрямлением перспективы, не-бумага дорисована."""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.fill import FILL_SCALE, TELEA_RADIUS, fill_non_paper  # noqa: F401 — ядро, ре-экспорт



class CopyMode(Enum):
    """Как получить прямоугольник: габарит четырёхугольника без преобразования или гомография в прямоугольник."""

    IN_PLACE = "in_place"
    WARP = "warp"




def quad_homography(quad: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Гомография четырёхугольника в прямоугольник по длинам его сторон (без смены масштаба).

    Args:
        quad: Вершины (4, 2) по часовой стрелке в пикселях кадра.

    Returns:
        (матрица 3×3, ширина, высота) выпрямленного прямоугольника.
    """
    # Порядок вершин: левая верхняя — с наименьшей суммой x + y, дальше по часовой стрелке.
    start = int(np.argmin(quad.sum(axis=1)))
    q = np.roll(quad, -start, axis=0).astype(np.float32)
    width = int(round(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))))
    height = int(round(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    return cv2.getPerspectiveTransform(q, dst), width, height


def copy_rectangle(
    bgr: np.ndarray, paper: np.ndarray, quad: np.ndarray, mode: CopyMode, keep: np.ndarray | None = None
) -> np.ndarray:
    """Прямоугольник копирования: вырезка по габариту (или выпрямление) четырёхугольника с дорисованной не-бумагой.

    Args:
        bgr: Кадр BGR (масштаб любой).
        paper: Маска листа того же размера, что ``bgr``.
        quad: Вершины четырёхугольника (4, 2) в пикселях ``bgr`` по часовой стрелке.
        mode: Способ (:class:`CopyMode`).
        keep: Защитный зазор у своей печати вне листа (того же размера) или None: закрашивается ровным цветом
            бумаги (:func:`research.newspaper_crop.text_guard.paint_margin`) и служит Telea известной границей —
            заливка не тянет краску букв.

    Returns:
        Картинка BGR прямоугольника.
    """
    from research.newspaper_crop.text_guard import paint_margin

    h, w = paper.shape
    keep = np.zeros_like(paper) if keep is None else keep
    if mode is CopyMode.WARP:
        m, width, height = quad_homography(quad)
        img = cv2.warpPerspective(bgr, m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        pm = cv2.warpPerspective(paper.astype(np.uint8), m, (width, height), flags=cv2.INTER_NEAREST) > 0
        km = cv2.warpPerspective(keep.astype(np.uint8), m, (width, height), flags=cv2.INTER_NEAREST) > 0
        return fill_non_paper(paint_margin(img, km, pm), pm | km)
    x0, y0 = np.floor(quad.min(axis=0)).astype(int)
    x1, y1 = np.ceil(quad.max(axis=0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    img, pm, km = bgr[y0:y1, x0:x1], paper[y0:y1, x0:x1], keep[y0:y1, x0:x1]
    return fill_non_paper(paint_margin(img, km, pm), pm | km)
