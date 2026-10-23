"""Подгоночные области страниц разворота (для оверлеев и будущего выпрямления): внешняя сторона, верх и низ — сплайном с узлами на изгибах (``page_boundary_adaptive``), внутренняя — по линии корешка; у полосы-вклейки верх и низ — по маске только до вспомогательных линий, через полосу — отрезок между страницами."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from ocr_utils.scan_cropping.v2.gutter_strip import Strip
from ocr_utils.scan_cropping.v2.page_boundary import CURVE_SAMPLES, EXTEND, SPINE_ZONE_MM
from ocr_utils.scan_cropping.v2.page_boundary_adaptive import fit_boundary_adaptive
from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM, PageQuad
from ocr_utils.scan_cropping.v2.page_sides import Side
from ocr_utils.scan_cropping.v2.spread_lines import Line
from ocr_utils.scan_cropping.v2.trapezoid import fit_enclosing_quad

# Зона полосы для верха и низа шире полосы на столько в сторону страницы, мм: у вспомогательной линии маска страницы
# кончается вертикальной кромкой, и её точки у угла попадают в профиль верха — сплайн нырял за ними на 3–4 мм на
# последних 5–8 мм (ПХ 1975/09 IMG_0021, верх левой страницы).
STRIP_GUARD_MM = 6.0


class StripJoin(StrEnum):
    """Как верх и низ страниц идут через полосу-вклейку до линии корешка."""

    # Отрезок от конца верха (низа) левой страницы на её вспомогательной линии к концу правой; угол — на корешке.
    SEGMENT = "segment"
    # Каждая страница — своя касательная от вспомогательной линии до линии корешка (без излома; углы двух страниц на
    # корешке могут разойтись по высоте).
    TANGENT = "tangent"


@dataclass(frozen=True)
class PageFrame:
    """Подгоночная область страницы.

    Поля:
        top: Верх (N, 2) от внешнего угла к углу на корешке, px сегментации.
        bottom: Низ (N, 2) от внешнего угла к углу на корешке.
        far: Внешняя сторона (N, 2) сверху вниз.
        spine: Внутренняя сторона — отрезок линии корешка ``[[x, y_верх], [x, y_низ]]``.
        knots: Добавленные узлы сплайна на изгибах верха и низа (K, 2).
    """

    top: np.ndarray
    bottom: np.ndarray
    far: np.ndarray
    spine: np.ndarray
    knots: np.ndarray

    def polygon(self) -> np.ndarray:
        """Замкнутый контур: верх (внешний угол → корешок), сторона у корешка, низ обратно, внешняя сторона обратно."""
        return np.vstack([self.top, self.spine[1:], self.bottom[::-1][1:], self.far[::-1][1:-1]])


def line_points(line: Line, y0: float, y1: float, step: float = 2.0) -> np.ndarray:
    """Точки прямой ``x(y)`` с шагом ``step`` px на ``[y0, y1]`` (N, 2)."""
    ys = np.arange(y0, y1 + step, step)
    return np.stack([line.x_at(ys), ys], axis=1)


def beyond(shape: tuple[int, int], limit: Line, page_left: bool) -> np.ndarray:
    """Пиксели за прямой со стороны корешка: у левой страницы — правее прямой, у правой — левее."""
    h, w = shape
    xs = limit.x_at(np.arange(h, dtype=np.float64))[:, None]
    cols = np.arange(w)[None, :]
    return (cols > xs) if page_left else (cols < xs)


def strip_zone(shape: tuple[int, int], strip: Strip) -> np.ndarray:
    """Зона полосы-вклейки: между её вспомогательными линиями во всю высоту кадра."""
    h, w = shape
    yy = np.arange(h, dtype=np.float64)
    cols = np.arange(w)[None, :]
    return (cols >= strip.left.x_at(yy)[:, None]) & (cols <= strip.right.x_at(yy)[:, None])


def cut_at(curve_points: np.ndarray, limit: Line, outer: np.ndarray, keep_left: bool) -> np.ndarray:
    """Кусок продлённой кривой верха или низа от внешнего угла до пересечения с прямой ``limit``.

    Args:
        curve_points: Точки продлённой кривой (``SideCurve.points`` на ``[-EXTEND, 1 + EXTEND]``).
        limit: Прямая конца (линия корешка или вспомогательная линия полосы).
        outer: Внешний угол рамки страницы.
        keep_left: Страница левее прямой (левая страница).

    Returns:
        Ломаная (N, 2) от ``outer`` до точки на ``limit`` (последняя точка — на прямой).
    """
    pts = curve_points[np.argsort(curve_points[:, 0])]
    if not keep_left:
        pts = pts[::-1]
    # Сторона прямой: положительно — внутри страницы.
    side = (limit.x_at(pts[:, 1]) - pts[:, 0]) * (1 if keep_left else -1)
    start = int(np.argmin(np.abs(pts[:, 0] - outer[0])))
    inside = np.flatnonzero(side[start:] <= 0)
    if not len(inside):
        end = pts[-1]
        return np.vstack([outer, pts[start:], [[limit.x_at(end[1]), end[1]]]])
    k = start + int(inside[0])
    a, b = pts[k - 1], pts[k]
    sa, sb = side[k - 1], side[k]
    hit = a + (b - a) * (sa / (sa - sb)) if sa != sb else b
    body = pts[start:k]
    return np.vstack([outer, body, hit])


def spine_hit(p: np.ndarray, q: np.ndarray, spine: Line) -> np.ndarray:
    """Точка отрезка ``p → q`` на линии корешка (отрезок пересекает её; иначе — проекция середины на прямую)."""
    fp, fq = spine.x_at(p[1]) - p[0], spine.x_at(q[1]) - q[0]
    if fp != fq and fp * fq <= 0:
        return p + (q - p) * (fp / (fp - fq))
    mid = (p + q) / 2
    return np.array([spine.x_at(mid[1]), mid[1]])


def spread_frames(
    pages: list[PageQuad],
    spine: Line,
    strip: Strip | None,
    paint: np.ndarray,
    core_zone: np.ndarray,
    texts: list[np.ndarray],
    join: StripJoin = StripJoin.TANGENT,
) -> list[PageFrame]:
    """Подгоночные области двух страниц разворота.

    Внешняя сторона, верх и низ — ``fit_boundary_adaptive`` (сплайн с 1–2 узлами на изгибах; сдвиг наружу только
    текстом), внутренняя сторона — линия корешка. Без полосы верх и низ продлеваются до линии корешка. С полосой подгонка идёт по маске и трапеции страницы без зоны полосы, а точки контура
    за вспомогательной линией негодны (неплотно приклеенная полоса даёт там всплески края), верх и низ кончаются на
    ней, а через полосу идут отрезком от верха (низа) левой страницы у левой линии к верху (низу) правой у
    правой; угол на корешке — точка этого отрезка на линии корешка.

    Args:
        pages: Две страницы с трапециями слева направо.
        spine: Линия корешка (``gutter_strip.spine_line``).
        strip: Полоса или None.
        paint: Зона закраса пальцев на масштабе сегментации (точки контура там негодны).
        core_zone: Ядро пальца с припуском (точки верха и низа там негодны).
        texts: Вершины текста по странице (N, 2) px сегментации.
        join: Как верх и низ идут через полосу до линии корешка: касательная своей страницы (по умолчанию, решение
            2026-10-22: без излома у полосы; ступенька углов на корешке — ПХ 1975/05 IMG_0021: 1.35 мм верх, 0.52 мм низ)
            или отрезок между страницами.

    Returns:
        Области левой и правой страницы.
    """
    frames_raw = []
    t_ext = np.linspace(-EXTEND, 1 + EXTEND, CURVE_SAMPLES)
    for page, spine_right, text in ((pages[0], True, texts[0]), (pages[1], False, texts[1])):
        limit = spine if strip is None else (strip.left if spine_right else strip.right)
        mask, quad = page.mask, page.quad
        exclude, edge_exclude = paint, core_zone
        if strip is not None:
            # Подгонка — по странице без зоны полосы (решение 2026-10-22: зона полосы изымается из бумаги только для
            # подгоночных областей; бумага кадра, вырезка и закрас её включают): маска и трапеция страницы без полосы.
            mask = page.mask & ~strip_zone(page.mask.shape, strip)
            quad = fit_enclosing_quad(mask)
            if quad is None:
                mask, quad = page.mask, page.quad
            # Точки контура в зоне полосы и ``STRIP_GUARD_MM`` перед ней негодны, как под пальцем (кромка маски у
            # полосы и её закругление), а зоны сплайна у корешка нет: верх и низ строятся так, будто в зоне полосы
            # точек нет вовсе, и дальше идут касательной.
            guarded = Line(limit.x0 + (-1 if spine_right else 1) * STRIP_GUARD_MM * SEG_PX_PER_MM, limit.y0, limit.k,
                           limit.y_top, limit.y_bot)  # fmt: skip
            far = beyond(page.mask.shape, guarded, page_left=spine_right)
            exclude, edge_exclude = paint | far, core_zone | far
        ys = quad[:, 1]
        fold = line_points(limit, float(ys.min()), float(ys.max()))
        boundary, fits = fit_boundary_adaptive(mask, quad, spine_right, SEG_PX_PER_MM, exclude, text,
                                               fold_points=fold, edge_exclude=edge_exclude,
                                               spine_zone_mm=None if strip is not None else SPINE_ZONE_MM)  # fmt: skip
        frames_raw.append((boundary, fits, spine_right, limit))
    out_parts = []
    for boundary, fits, spine_right, limit in frames_raw:
        outer_top = boundary.top[0] if spine_right else boundary.top[-1]
        outer_bot = boundary.bottom[0] if spine_right else boundary.bottom[-1]
        # Конец верха и низа: вспомогательная линия (через полосу — отрезок) или сразу линия корешка (касательная).
        end = spine if strip is None or join is StripJoin.TANGENT else limit
        top = cut_at(boundary.curves[Side.TOP].points(t_ext), end, outer_top, spine_right)
        bottom = cut_at(boundary.curves[Side.BOTTOM].points(t_ext), end, outer_bot, spine_right)
        far = boundary.left if spine_right else boundary.right
        knots = [boundary.curves[s].points(np.asarray(f.added, np.float64)) for s, f in fits.items() if len(f.added)]
        out_parts.append([top, bottom, far, np.vstack(knots) if knots else np.zeros((0, 2))])
    if strip is not None and join is StripJoin.SEGMENT:
        # Через полосу — отрезок между концами верха (низа) двух страниц на вспомогательных линиях; угол — на корешке.
        for i in (0, 1):  # верх, низ
            hit = spine_hit(out_parts[0][i][-1], out_parts[1][i][-1], spine)
            out_parts[0][i] = np.vstack([out_parts[0][i], hit])
            out_parts[1][i] = np.vstack([out_parts[1][i], hit])
    frames = []
    for top, bottom, far, knots in out_parts:
        spine_side = np.array([top[-1], bottom[-1]])
        frames.append(PageFrame(top, bottom, far, spine_side, knots))
    return frames
