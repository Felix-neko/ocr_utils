"""Холст surya layout по поверхности Кунса подгоночных областей страниц (первое преобразование выпрямления) и перенос блоков с холста на кадр сгущёнными многоугольниками."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np

from ocr_utils.page_layout.image import SURYA_MAX_SIDE
from ocr_utils.scan_cropping.v2.page_frames import PageFrame
from ocr_utils.scan_cropping.v2.rectify.coons import (
    TANGENT_MM,
    Layout,
    Patch,
    coons_at,
    page_patch,
    patch_outline,
    plan_layout,
    rect_fractions,
    rectify_maps_scaled,
    scaled_size,
)
from ocr_utils.scan_cropping.v2.rectify.geometry import SpreadGeometry


class LayoutWarp(StrEnum):
    """Чем выпрямляется кадр перед surya layout."""

    # Кадр как есть.
    NONE = "none"
    # Гомографии страниц по трапециям (``keystone``).
    KEYSTONE = "keystone"
    # Поверхность Кунса по подгоночным областям страниц; не построилась — откат на трапеции.
    COONS = "coons"


class FramesReason(StrEnum):
    """Почему холст Кунса не построен (откат на трапеции); пусто — построен."""

    OK = ""
    # Подгоночных областей нет или их больше двух: бумага не найдена, страниц с трапецией нет (значение — прежнее, когда
    # одна страница тоже откатывалась на трапецию).
    NOT_TWO_PAGES = "not_two_pages"
    # Подгонка областей (``spread_fit``) упала.
    FIT_FAILED = "fit_failed"
    # Проверка поверхности не прошла: складка, площадь, самопересечение контура (:func:`check_patches`).
    DEGENERATE = "degenerate"
    # Файла областей в кэше нет (шаг страниц считался без ``--layout-warp coons``).
    MISSING = "missing"


# Поле холста вокруг прямоугольников страниц, мм бумаги (≈ 8 % поля keystone: 25 мм ≈ 345 px при 13.8 px/мм). Поле
# нужно фильтру мусорных блоков surya (книга по Оцу на чёрном коврике) и тексту, вылезшему за область.
CANVAS_MARGIN_MM = 25.0
# Шаг сгущения вершин блока перед переносом на кадр, мм бумаги (≈ 28 px при 13.8 px/мм).
DENSIFY_MM = 2.0
# Допуск упрощения перенесённого многоугольника (``cv2.approxPolyDP``), мм бумаги (≈ 1.4 px при 13.8 px/мм).
SIMPLIFY_MM = 0.1
# Допустимое отношение площади прямоугольника страницы к площади её области (в обе стороны).
MAX_AREA_RATIO = 1.5
# Шаг сетки проверки якобиана поверхности, мм бумаги (≈ 138 px при 13.8 px/мм).
JACOBIAN_STEP_MM = 10.0
# Точек контура области в проверке на самопересечение (контур перевыбирается по длине дуги).
OUTLINE_SAMPLES = 400
# Обратный перенос кадр → холст: узлов грубой сетки начального приближения по длинной стороне холста, итераций Ньютона,
# шаг конечных разностей якобиана (px холста).
INVERSE_GRID_NODES = 160
NEWTON_ITERATIONS = 20
NEWTON_STEP_PX = 0.5
# Сколько точек обратного переноса за раз сравнивать с узлами сетки (память на матрицу расстояний).
INVERSE_CHUNK = 64


@dataclass(frozen=True)
class CoonsWarp:
    """Холст Кунса для surya: области страниц в px кадра, расклад холста и масштаб входа сети.

    Поля:
        patches: Области левой и правой страницы (``rectify.coons.Patch``) в px кадра; у кадра с одной страницей
            (обложка, крышка, страница без полоски соседней) — одна область, корешка нет (``layout.single``).
        layout: Расклад холста в px «масштаба кадра» (длины по бумаге — как в кадре).
        scale: Множитель холста под surya (≤ 1; длинная сторона → ``SURYA_MAX_SIDE``).
        margin_px: Поле холста вокруг прямоугольников страниц, px кадра.
    """

    patches: tuple[Patch, ...]
    layout: Layout
    scale: float
    margin_px: float

    @property
    def small_size(self) -> tuple[int, int]:
        """Размер холста на входе surya ``(W, H)``."""
        return scaled_size(self.layout, self.scale)


def plan_coons(
    frames: list[PageFrame] | None,
    seg_scale: tuple[float, float],
    shape: tuple[int, int],
    px_per_mm: float,
    margin_mm: float = CANVAS_MARGIN_MM,
    tangent_mm: float = TANGENT_MM,
) -> tuple[CoonsWarp | None, FramesReason]:
    """Холст Кунса по подгоночным областям страниц: двух (разворот) или одной (кадр с одной страницей).

    Args:
        frames: Области левой и правой страницы или одна область (px сегментации), или None.
        seg_scale: Масштаб сегментации относительно кадра по X и Y (``seg / кадр``).
        shape: Размер кадра ``(H, W)``.
        px_per_mm: Плотность кадра, px/мм бумаги.
        margin_mm: Поле холста вокруг прямоугольников страниц, мм бумаги.
        tangent_mm: Отрезок у конца стороны для касательной продолжения, мм бумаги (как у выхода страниц).

    Returns:
        ``(холст, причина)``: холст или None и причина отката (``FramesReason.OK`` — холст построен).
    """
    if frames is None or len(frames) not in (1, 2):
        return None, FramesReason.NOT_TWO_PAGES
    h, w = shape
    # Рамки вырезки для surya ещё нет: «итог» — сам кадр, перевод — px сегментации → px кадра.
    geometry = SpreadGeometry(frames, (0, 0, w, h), seg_scale)
    tangent_px = tangent_mm * px_per_mm
    try:
        # Левая страница (и единственная) — внешняя сторона слева, правая — справа.
        patches = tuple(page_patch(f, i == 0, geometry.to_output, tangent_px) for i, f in enumerate(frames))
    except ValueError:
        # Сторона области выродилась в точку.
        return None, FramesReason.DEGENERATE
    margin_px = margin_mm * px_per_mm
    layout = plan_layout(patches, shape, margin_px=margin_px)
    if min(layout.width, layout.height) < 2 or min(r.width for r in layout.rects) <= 0:
        return None, FramesReason.DEGENERATE
    if not check_patches(patches, layout, px_per_mm):
        return None, FramesReason.DEGENERATE
    scale = min(1.0, SURYA_MAX_SIDE / max(layout.width, layout.height))
    return CoonsWarp(patches, layout, scale, margin_px), FramesReason.OK


def check_patches(patches: tuple[Patch, ...], layout: Layout, px_per_mm: float) -> bool:
    """Годна ли поверхность Кунса для surya: без складок, с разумной площадью и простым контуром областей.

    Проверки по каждой странице: (а) на сетке шагом ``JACOBIAN_STEP_MM`` по её части холста (с полями) определитель
    якобиана конечных разностей положителен (нет складки, ориентация не перевёрнута); (б) площадь прямоугольника
    страницы относится к площади области в пределах ``[1/MAX_AREA_RATIO, MAX_AREA_RATIO]``; (в) контур области без
    самопересечений.

    Args:
        patches: Области левой и правой страницы (или одна), px кадра.
        layout: Расклад холста.
        px_per_mm: Плотность кадра, px/мм.

    Returns:
        True — поверхность годна.
    """
    step = max(JACOBIAN_STEP_MM * px_per_mm, 1.0)
    # Часть холста каждой страницы: у одной страницы — весь холст.
    bounds = ((0.0, float(layout.width)),) if layout.single else (
        (0.0, layout.spine_x), (layout.spine_x, float(layout.width)))  # fmt: skip
    for patch, rect, (lo, hi) in zip(patches, layout.rects, bounds):
        if hi - lo < 1.0:
            return False
        # Узлы сетки по части холста страницы, концы включительно.
        cols = np.linspace(lo, hi, max(int(np.ceil((hi - lo) / step)), 1) + 1)
        rows = np.linspace(0.0, float(layout.height), max(int(np.ceil(layout.height / step)), 1) + 1)
        u, v = rect_fractions(rect, cols, rows)
        gx, gy = patch.grid(u, v)
        gx, gy = gx.astype(np.float64), gy.astype(np.float64)
        # Конечные разности вдоль столбцов (по X холста) и строк (по Y холста) на общей части ячеек.
        xc, yc = np.diff(gx, axis=1)[:-1, :], np.diff(gy, axis=1)[:-1, :]
        xr, yr = np.diff(gx, axis=0)[:, :-1], np.diff(gy, axis=0)[:, :-1]
        det = xc * yr - xr * yc
        if not np.isfinite(det).all() or float(det.min()) <= 0:
            return False
        outline = patch_outline(patch)
        area = abs(float(cv2.contourArea(outline.astype(np.float32).reshape(-1, 1, 2))))
        ratio = rect.width * rect.height / max(area, 1.0)
        if not 1 / MAX_AREA_RATIO <= ratio <= MAX_AREA_RATIO:
            return False
        if not is_simple(outline):
            return False
    return True


def resample_closed(points: np.ndarray, n: int) -> np.ndarray:
    """Замкнутый контур, перевыбранный ``n`` точками равномерно по длине дуги.

    Args:
        points: Вершины (N, 2) по порядку обхода (последняя соединяется с первой).
        n: Число точек выхода.

    Returns:
        Точки (n, 2); при вырожденном контуре — исходные вершины.
    """
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    closed = np.vstack([pts, pts[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    arc = np.r_[0.0, np.cumsum(seg)]
    if arc[-1] <= 0:
        return pts
    d = np.linspace(0.0, arc[-1], n, endpoint=False)
    return np.stack([np.interp(d, arc, closed[:, 0]), np.interp(d, arc, closed[:, 1])], axis=1)


def is_simple(outline: np.ndarray) -> bool:
    """Простой ли замкнутый контур (несмежные рёбра не пересекаются собственно).

    Контур перевыбирается ``OUTLINE_SAMPLES`` точками; проверка — знаки векторных произведений по всем парам
    несмежных рёбер разом.

    Args:
        outline: Вершины контура (N, 2).

    Returns:
        True — самопересечений нет.
    """
    a = resample_closed(outline, OUTLINE_SAMPLES)
    b = np.roll(a, -1, axis=0)
    n = len(a)
    i, j = np.triu_indices(n, k=2)
    # Первое и последнее рёбра смежны через замыкание.
    keep = ~((i == 0) & (j == n - 1))
    i, j = i[keep], j[keep]
    di, dj = b[i] - a[i], b[j] - a[j]
    d1 = di[:, 0] * (a[j, 1] - a[i, 1]) - di[:, 1] * (a[j, 0] - a[i, 0])
    d2 = di[:, 0] * (b[j, 1] - a[i, 1]) - di[:, 1] * (b[j, 0] - a[i, 0])
    d3 = dj[:, 0] * (a[i, 1] - a[j, 1]) - dj[:, 1] * (a[i, 0] - a[j, 0])
    d4 = dj[:, 0] * (b[i, 1] - a[j, 1]) - dj[:, 1] * (b[i, 0] - a[j, 0])
    return not bool(((d1 * d2 < 0) & (d3 * d4 < 0)).any())


def warp_coons(bgr: np.ndarray, warp: CoonsWarp) -> np.ndarray:
    """Холст Кунса на масштабе входа surya; за пределами кадра — чёрный (как ``borderValue=0`` у keystone).

    Кадр уменьшается ``INTER_AREA`` в ``warp.scale`` раз, карта считается сразу на сетке малого холста
    (``rectify_maps_scaled``) и переводится в px малого кадра; выборка — билинейная.

    Args:
        bgr: Кадр BGR (полное разрешение).
        warp: Холст Кунса.

    Returns:
        Холст BGR размера ``warp.small_size``.
    """
    map_x, map_y = rectify_maps_scaled(warp.patches, warp.layout, warp.scale)
    fh, fw = bgr.shape[:2]
    if warp.scale < 1.0:
        sw, sh = max(1, int(round(fw * warp.scale))), max(1, int(round(fh * warp.scale)))
        small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        sw, sh, small = fw, fh, bgr
    # Центры пикселей: точка кадра X → малый кадр (X + 0.5)·s − 0.5.
    sx, sy = sw / fw, sh / fh
    mx = ((map_x + 0.5) * sx - 0.5).astype(np.float32)
    my = ((map_y + 0.5) * sy - 0.5).astype(np.float32)
    return cv2.remap(small, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def from_surya(polys: list[np.ndarray], warp: CoonsWarp) -> list[np.ndarray]:
    """Многоугольники surya в px малого холста (непрерывные координаты края) → px холста масштаба кадра.

    Args:
        polys: Многоугольники (N, 2) на малом холсте.
        warp: Холст Кунса.

    Returns:
        Многоугольники (N, 2) float64 в координатах центров пикселей холста ``warp.layout``.
    """
    return [np.asarray(p, np.float64).reshape(-1, 2) / warp.scale - 0.5 for p in polys]


def to_surya(polys: list[np.ndarray], warp: CoonsWarp) -> list[np.ndarray]:
    """Обратное к :func:`from_surya`: px холста масштаба кадра → px малого холста (для рисования на нём)."""
    return [(np.asarray(p, np.float64).reshape(-1, 2) + 0.5) * warp.scale for p in polys]


def spine_copy(point: np.ndarray, spine_x: float, left: bool) -> np.ndarray:
    """Точка на линии корешка, отнесённая к странице: левая — X чуть меньше ``spine_x``, правая — ровно ``spine_x``.

    ``coons_at`` относит точку к левой странице при ``x < spine_x``, поэтому копии для двух карт различаются на одно
    представимое число double.

    Args:
        point: Точка пересечения ребра с линией корешка (2,).
        spine_x: X линии корешка на холсте.
        left: Копия для левой страницы.

    Returns:
        Точка (2,).
    """
    x = float(np.nextafter(spine_x, -np.inf)) if left else float(spine_x)
    return np.array([x, float(point[1])])


def densify(poly: np.ndarray, step_px: float, spine_x: float | None) -> np.ndarray:
    """Сгущение замкнутого многоугольника по длине дуги; ребро через линию корешка — с двойной точкой на ней.

    На каждом ребре точки ставятся через ``step_px`` (ребро делится на ``ceil(длина / step_px)`` равных частей),
    исходные вершины сохраняются. Ребро, пересекающее ``x = spine_x``, получает точку пересечения дважды: одна копия
    пойдёт через карту страницы начала ребра, другая — через карту страницы конца (:func:`spine_copy`). Так блок через
    корешок получает на кадре честный излом на линии корешка, а не хорду через неё.

    Args:
        poly: Вершины (N, 2) по порядку обхода.
        step_px: Наибольший шаг между соседними точками, px.
        spine_x: X линии корешка или None (без разреза).

    Returns:
        Точки (M, 2) float64 по порядку обхода.
    """
    pts = np.asarray(poly, np.float64).reshape(-1, 2)
    step = max(float(step_px), 1e-6)
    out: list[np.ndarray] = []
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        crosses = spine_x is not None and (a[0] - spine_x) * (b[0] - spine_x) < 0
        if crosses:
            t = (spine_x - a[0]) / (b[0] - a[0])
            hit = a + (b - a) * t
            # Копия у начала — для страницы ``a``, у конца — для страницы ``b``.
            pieces = [(a, spine_copy(hit, spine_x, a[0] < spine_x)), (spine_copy(hit, spine_x, b[0] < spine_x), b)]
        else:
            pieces = [(a, b)]
        for k, (s, e) in enumerate(pieces):
            parts = max(int(np.ceil(float(np.hypot(*(e - s))) / step)), 1)
            frac = np.arange(parts, dtype=np.float64)[:, None] / parts
            out.append(s + (e - s) * frac)
            if crosses and k == 0:
                # Конец первого куска (копия для страницы начала) — отдельной точкой: дальше идёт вторая копия.
                out.append(e[None, :])
    return np.vstack(out) if out else np.zeros((0, 2))


def unwarp_coons(polys: list[np.ndarray], warp: CoonsWarp, px_per_mm: float) -> list[np.ndarray]:
    """Многоугольники холста (px масштаба кадра) → кадр: сгущение, точная карта Кунса в каждой точке, упрощение.

    Args:
        polys: Многоугольники (N, 2) на холсте ``warp.layout``.
        warp: Холст Кунса.
        px_per_mm: Плотность кадра, px/мм (шаг сгущения и допуск упрощения — в мм).

    Returns:
        Многоугольники (M, 2) float32 в px кадра.
    """
    step = max(DENSIFY_MM * px_per_mm, 1.0)
    eps = SIMPLIFY_MM * px_per_mm
    out = []
    for poly in polys:
        # У одной страницы разреза по корешку нет: двойные точки на «корешке» (правом краю холста) не нужны.
        dense = densify(poly, step, None if warp.layout.single else warp.layout.spine_x)
        moved = coons_at(warp.patches, warp.layout, dense[:, 0], dense[:, 1])
        simple = cv2.approxPolyDP(moved.astype(np.float32).reshape(-1, 1, 2), eps, True)
        out.append(simple.reshape(-1, 2).astype(np.float32))
    return out


def page_points(warp: CoonsWarp, page: int, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Точки кадра по карте одной страницы (без разреза по корешку) для точек холста.

    Args:
        warp: Холст Кунса.
        page: 0 — левая страница, 1 — правая.
        xs: X точек холста (N,).
        ys: Y точек холста (N,).

    Returns:
        Точки (N, 2) px кадра.
    """
    u, v = rect_fractions(warp.layout.rects[page], np.asarray(xs, np.float64), np.asarray(ys, np.float64))
    return warp.patches[page].points(u, v)


def initial_guess(points: np.ndarray, warp: CoonsWarp) -> np.ndarray:
    """Начальные точки холста для обратного переноса: ближайший узел грубой сетки холста по их образам на кадре.

    Args:
        points: Точки кадра (N, 2).
        warp: Холст Кунса.

    Returns:
        Точки холста (N, 2).
    """
    layout = warp.layout
    step = max(layout.width, layout.height) / INVERSE_GRID_NODES
    gx = np.arange(0.0, layout.width + step, step)
    gy = np.arange(0.0, layout.height + step, step)
    cx, cy = np.meshgrid(gx, gy)
    nodes = np.stack([cx.ravel(), cy.ravel()], axis=1)
    images = coons_at(warp.patches, layout, nodes[:, 0], nodes[:, 1]).astype(np.float32)
    out = np.empty((len(points), 2))
    for k in range(0, len(points), INVERSE_CHUNK):
        chunk = points[k : k + INVERSE_CHUNK].astype(np.float32)
        d2 = ((chunk[:, None, :] - images[None, :, :]) ** 2).sum(axis=2)
        out[k : k + INVERSE_CHUNK] = nodes[np.argmin(d2, axis=1)]
    return out


def newton_page(points: np.ndarray, start: np.ndarray, warp: CoonsWarp, page: int) -> tuple[np.ndarray, np.ndarray]:
    """Обращение карты одной страницы методом Ньютона с якобианом конечных разностей.

    Args:
        points: Точки кадра (N, 2) — цель.
        start: Начальные точки холста (N, 2).
        warp: Холст Кунса.
        page: Страница (0 — левая, 1 — правая).

    Returns:
        ``(точки холста (N, 2), невязка (N,) px кадра)``.
    """
    x = start.astype(np.float64).copy()
    h = NEWTON_STEP_PX
    for _ in range(NEWTON_ITERATIONS):
        f = page_points(warp, page, x[:, 0], x[:, 1]) - points
        fx = (page_points(warp, page, x[:, 0] + h, x[:, 1]) - points - f) / h
        fy = (page_points(warp, page, x[:, 0], x[:, 1] + h) - points - f) / h
        det = fx[:, 0] * fy[:, 1] - fy[:, 0] * fx[:, 1]
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        # Решение 2×2: J · δ = f, J = [[fx₀, fy₀], [fx₁, fy₁]].
        dx = (f[:, 0] * fy[:, 1] - fy[:, 0] * f[:, 1]) / det
        dy = (fx[:, 0] * f[:, 1] - f[:, 0] * fx[:, 1]) / det
        x[:, 0] -= dx
        x[:, 1] -= dy
    residual = np.hypot(*(page_points(warp, page, x[:, 0], x[:, 1]) - points).T)
    return x, residual


def canvas_of(points: np.ndarray, warp: CoonsWarp) -> np.ndarray:
    """Точки кадра → холст (обращение карты Кунса): Ньютон по обеим страницам, выбор той, на чьей стороне корешка решение.

    Args:
        points: Точки кадра (N, 2).
        warp: Холст Кунса.

    Returns:
        Точки холста (N, 2).
    """
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    if not len(pts):
        return pts
    start = initial_guess(pts, warp)
    left, res_left = newton_page(pts, start, warp, 0)
    if warp.layout.single:
        return left
    right, res_right = newton_page(pts, start, warp, 1)
    spine = warp.layout.spine_x
    ok_left, ok_right = left[:, 0] < spine, right[:, 0] >= spine
    # Решение на своей стороне корешка; оба годны или оба нет — меньшая невязка.
    take_left = (ok_left & ~ok_right) | ((ok_left == ok_right) & (res_left <= res_right))
    return np.where(take_left[:, None], left, right)


def warp_polygons_coons(polys: list[np.ndarray], warp: CoonsWarp) -> list[np.ndarray]:
    """Многоугольники кадра → холст (обратная карта; для оверлеев и проверок).

    Args:
        polys: Многоугольники (N, 2) px кадра.
        warp: Холст Кунса.

    Returns:
        Многоугольники (N, 2) float64 на холсте ``warp.layout``.
    """
    return [canvas_of(p, warp) for p in polys]


def frames_on_frame(frames: list[PageFrame] | None, seg_scale: tuple[float, float]) -> list[np.ndarray]:
    """Контуры подгоночных областей в px кадра (для оверлея).

    Args:
        frames: Области (px сегментации) или None.
        seg_scale: Масштаб сегментации относительно кадра по X и Y.

    Returns:
        Контуры (N, 2) px кадра; пусто без областей.
    """
    if not frames:
        return []
    return [f.polygon() / np.asarray(seg_scale, np.float64) for f in frames]


def canvas_record(warp: CoonsWarp) -> dict:
    """Параметры холста для ``layout.json``: размер, линия корешка (None у одной страницы), прямоугольники страниц, поле и масштаб surya.

    Args:
        warp: Холст Кунса.

    Returns:
        Словарь, пригодный для JSON.
    """
    layout = warp.layout
    return {
        "width": int(layout.width),
        "height": int(layout.height),
        # У одной страницы корешка нет.
        "spine_x": None if layout.single else round(float(layout.spine_x), 3),
        "rects": [[round(float(v), 3) for v in (r.x0, r.y0, r.x1, r.y1)] for r in layout.rects],
        "scale": float(warp.scale),
        "margin_px": round(float(warp.margin_px), 3),
    }
