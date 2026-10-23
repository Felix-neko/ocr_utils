"""Лист отсмотра выпрямления: «было» — итог v2 с подгоночными областями страниц и сеткой изолиний u, v; «стало» — выпрямленный кадр с прямоугольниками и той же сеткой (там она прямая)."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, framed
from ocr_utils.scan_cropping.v2.overlay import COLOUR_FRAME
from research.page_rectify.coons import Layout, Patch, PageRect

# Высота каждой половины листа, px: страница целиком видна, строки различимы.
SHEET_HEIGHT = 1400
# Зазор между «было» и «стало», px листа.
GAP = 24
# Шаг сетки изолиний — доля стороны прямоугольника страницы.
GRID_STEP = 0.1
# Точек на изолинию (кривая на «было»).
GRID_SAMPLES = 80
# Сетка внутри области — вспомогательный тон (``stages.COLOUR_HINT``), продолжение за область — приглушённый.
COLOUR_GRID = (200, 140, 60)
COLOUR_GRID_OUT = (150, 170, 120)
GRID_ALPHA = 0.6
# Толщина рамки страницы на листе, px.
FRAME_PX = 2


def grid_segments(
    patch: Patch, rect: PageRect, x_range: tuple[float, float], height: int
) -> list[tuple[bool, np.ndarray, np.ndarray]]:
    """Изолинии ``u = const`` и ``v = const`` с шагом :data:`GRID_STEP` на половине холста страницы.

    Каждая изолиния режется на куски внутри области (``u, v ∈ [0, 1]``) и за ней.

    Args:
        patch: Область страницы.
        rect: Её прямоугольник на холсте.
        x_range: Столбцы холста этой страницы ``(от, до)``.
        height: Высота холста.

    Returns:
        Куски ``(внутри, точки на холсте (N, 2), точки на итоге v2 (N, 2))``.
    """
    u_lo, u_hi = (x_range[0] - rect.x0) / rect.width, (x_range[1] - rect.x0) / rect.width
    v_lo, v_hi = -rect.y0 / rect.height, (height - rect.y0) / rect.height
    out = []
    # Изолинии по каждой оси: постоянная доля и пределы второй доли.
    for along_u, (lo, hi), (o_lo, o_hi) in ((True, (u_lo, u_hi), (v_lo, v_hi)), (False, (v_lo, v_hi), (u_lo, u_hi))):
        for k in range(int(np.ceil(lo / GRID_STEP)), int(np.floor(hi / GRID_STEP)) + 1):
            fixed = k * GRID_STEP
            fixed_in = -1e-9 <= fixed <= 1 + 1e-9
            # Куски второй доли: до области, в ней, после.
            for a, b in ((o_lo, min(0.0, o_hi)), (max(0.0, o_lo), min(1.0, o_hi)), (max(1.0, o_lo), o_hi)):
                if b - a <= 1e-6:
                    continue
                t = np.linspace(a, b, GRID_SAMPLES)
                u, v = (np.full_like(t, fixed), t) if along_u else (t, np.full_like(t, fixed))
                canvas = np.stack([rect.x0 + u * rect.width, rect.y0 + v * rect.height], axis=1)
                out.append((fixed_in and 0 <= a and b <= 1, canvas, patch.points(u, v)))
    return out


def draw_grid(image: np.ndarray, segments: list[np.ndarray], colour: tuple[int, int, int], scale: float) -> None:
    """Полупрозрачные ломаные на картинке листа (на месте).

    Args:
        image: Картинка листа BGR.
        segments: Ломаные (N, 2) в px полного размера.
        colour: Цвет BGR.
        scale: Масштаб листа к полному размеру.
    """
    layer = image.copy()
    pts = [np.round(s * scale).astype(np.int32) for s in segments]
    cv2.polylines(layer, pts, False, colour, 1, cv2.LINE_AA)
    cv2.addWeighted(layer, GRID_ALPHA, image, 1 - GRID_ALPHA, 0, image)


def patch_outline(patch: Patch) -> np.ndarray:
    """Контур области: верх, правая сторона, низ обратно, левая обратно (N, 2)."""
    return np.vstack([patch.top.points, patch.right.points, patch.bottom.points[::-1], patch.left.points[::-1]])


def rect_outline(rect: PageRect) -> np.ndarray:
    """Контур прямоугольника (4, 2)."""
    return np.array([[rect.x0, rect.y0], [rect.x1, rect.y0], [rect.x1, rect.y1], [rect.x0, rect.y1]])


def draw_sheet(
    before: np.ndarray, after: np.ndarray, patches: tuple[Patch, Patch], layout: Layout, header: list[str]
) -> np.ndarray:
    """Лист «было | стало».

    Args:
        before: Итог v2 (BGR).
        after: Выпрямленный кадр (BGR).
        patches: Области левой и правой страницы (px итога v2).
        layout: Расклад выпрямленного кадра.
        header: Строки шапки.

    Returns:
        Лист BGR со шапкой и легендой в полях.
    """
    f_before = SHEET_HEIGHT / before.shape[0]
    f_after = SHEET_HEIGHT / after.shape[0]
    left_img = cv2.resize(before, None, fx=f_before, fy=f_before, interpolation=cv2.INTER_AREA)
    right_img = cv2.resize(after, None, fx=f_after, fy=f_after, interpolation=cv2.INTER_AREA)
    split = int(np.ceil(layout.spine_x))
    ranges = ((0.0, float(split)), (float(split), float(layout.width)))
    pieces = [s for p, r, xr in zip(patches, layout.rects, ranges) for s in grid_segments(p, r, xr, layout.height)]
    for inside, colour in ((False, COLOUR_GRID_OUT), (True, COLOUR_GRID)):
        draw_grid(left_img, [src for i, _, src in pieces if i == inside], colour, f_before)
        draw_grid(right_img, [cnv for i, cnv, _ in pieces if i == inside], colour, f_after)
    for patch, rect in zip(patches, layout.rects):
        cv2.polylines(left_img, [np.round(patch_outline(patch) * f_before).astype(np.int32)], True, COLOUR_FRAME,
                      FRAME_PX, cv2.LINE_AA)  # fmt: skip
        cv2.polylines(right_img, [np.round(rect_outline(rect) * f_after).astype(np.int32)], True, COLOUR_FRAME,
                      FRAME_PX, cv2.LINE_AA)  # fmt: skip
    gap = np.full((SHEET_HEIGHT, GAP, 3), 255, np.uint8)
    canvas = np.hstack([left_img, gap, right_img])
    legend = [
        LegendEntry("подгоночная область страницы (было) → прямоугольник (стало)", COLOUR_FRAME),
        LegendEntry(f"сетка u, v внутри области, шаг {GRID_STEP:.0%}", COLOUR_GRID, alpha=GRID_ALPHA),
        LegendEntry("продолжение сетки за область", COLOUR_GRID_OUT, alpha=GRID_ALPHA),
    ]
    return framed(canvas, header, legend)
