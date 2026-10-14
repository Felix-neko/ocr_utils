"""Склейка страниц разворота в маске бумаги: мостик между массивными компонентами пропорционально их массе.

SAM 3 выдаёт каждую страницу отдельным экземпляром, и между страницами у корешка остаются щель и выемки. Наружу
(обложка, торец, коврик) маска расти не должна, поэтому все варианты добавляют пиксели только там, где близко две
страницы сразу, и только внутри маски книги.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from scipy import ndimage

from ocr_utils.scan_cropping.v2.book import disk
from ocr_utils.scan_cropping.v2.glue import (  # noqa: F401 — ядро, ре-экспорт для стенда
    PAGE_MIN_SHARE,
    SPINE_SEARCH,
    SPINE_SMOOTH,
    W_MAX,
    GlueResult,
    close_pairs,
    hull_bridge_add,
    outside_distance,
    pages_of,
    pair_band,
    split_pages,
)

# Метаболы: радиус влияния r_i = METABALL_K·√A_i, порог суммарного поля.
METABALL_K = 0.14
METABALL_TAU = 1.0
# Мостик-эллипс: d_A + d_B ≤ min(W_MAX, FERMAT_C·√min(A_A, A_B)).
FERMAT_C = 0.12
# Адаптивное закрытие: радиус ADAPTIVE_K·√min(A_A, A_B), не больше W_MAX/2.
ADAPTIVE_K = 0.06
# Пиксель склейки лежит между страницами, если ближайшие точки страниц видны под углом не меньше этого, градусы.
BETWEEN_MIN_ANGLE = 90.0
# Сглаживание краёв страниц перед оценкой угла, px.
BETWEEN_SMOOTH_PX = 9
# Открытие добавленного склейкой, px: срезает тонкие клочья у края.
ADD_OPEN_PX = 3
# Заливка вогнутостей после склейки, px.
CONCAVE_DEPTH = 40


class GlueMethod(str, Enum):
    """Способ склейки страниц."""

    METABALLS = "metaballs"
    FERMAT = "fermat"
    HULL_BRIDGE = "hull_bridge"
    ADAPTIVE_CLOSE = "adaptive_close"










def nearest_vectors(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Вектор от каждого пикселя к ближайшему пикселю маски.

    Args:
        mask: Булева маска.

    Returns:
        ``(dx, dy)`` float32 (внутри маски — нули).
    """
    h, w = mask.shape
    _, labels = cv2.distanceTransformWithLabels((~mask).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(mask)
    # Метки DIST_LABEL_PIXEL нумеруют нулевые пиксели входа (пиксели маски) в порядке обхода по строкам.
    lut_x = np.zeros(labels.max() + 1, np.float32)
    lut_y = np.zeros(labels.max() + 1, np.float32)
    lut_x[labels[ys, xs]] = xs
    lut_y[labels[ys, xs]] = ys
    gy, gx = np.mgrid[0:h, 0:w]
    return lut_x[labels] - gx, lut_y[labels] - gy


def between(pages: list[np.ndarray], i: int, j: int, min_angle: float = BETWEEN_MIN_ANGLE) -> np.ndarray:
    """Пиксели, лежащие МЕЖДУ страницами i и j: ближайшие точки страниц видны под углом не меньше ``min_angle``.

    Над и под корешком (там, где сходятся верхние или нижние края страниц) обе ближайшие точки лежат по одну
    сторону — угол мал, и туда (торец, край крышки) склейка не заходит.

    Args:
        pages: Маски страниц.
        i: Первая страница.
        j: Вторая страница.
        min_angle: Минимальный угол, градусы.

    Returns:
        Булева маска.
    """
    # Края страниц сглаживаются открытием: на зубчатом краю ближайшая точка прыгает, и угол врёт.
    smooth = [cv2.morphologyEx(pages[k].astype(np.uint8), cv2.MORPH_OPEN, disk(BETWEEN_SMOOTH_PX)) > 0 for k in (i, j)]
    ax, ay = nearest_vectors(smooth[0] if smooth[0].any() else pages[i])
    bx, by = nearest_vectors(smooth[1] if smooth[1].any() else pages[j])
    na = np.hypot(ax, ay) + 1e-6
    nb = np.hypot(bx, by) + 1e-6
    cos = (ax * bx + ay * by) / (na * nb)
    return cos <= np.cos(np.radians(min_angle))


def bridges_only(add: np.ndarray, pages: list[np.ndarray]) -> np.ndarray:
    """Куски добавленного, касающиеся хотя бы двух страниц (мостики), без «наростов» на одной странице.

    Args:
        add: Добавленное склейкой.
        pages: Маски страниц.

    Returns:
        Булева маска.
    """
    count, labels = cv2.connectedComponents(add.astype(np.uint8))
    if count <= 1:
        return add
    touches = np.zeros(count, np.int32)
    for p in pages:
        ring = (cv2.dilate(p.astype(np.uint8), disk(2)) > 0) & ~p
        touches[np.unique(labels[ring & add])] += 1
    keep = touches >= 2
    keep[0] = False
    return keep[labels]






def glue(
    paper: np.ndarray,
    book: np.ndarray,
    method: GlueMethod,
    concave: bool = False,
    hulls: np.ndarray | None = None,
    instances: np.ndarray | None = None,
    between_filter: bool = True,
) -> GlueResult:
    """Склейка страниц маски бумаги выбранным способом.

    Args:
        paper: Маска бумаги (``sam3_1288_plain``).
        book: Маска книги.
        method: Способ.
        concave: Залить после склейки неглубокие вогнутости (``paper.fill_concavities``).
        hulls: Своя печать (разделение слитых страниц по корешку, если нет экземпляров) или None.
        instances: Карта экземпляров-страниц SAM 3 или None.
        between_filter: Фильтр «только между страницами» (угол к ближайшим точкам страниц, мостики, касающиеся
            обеих страниц). False — прежнее поведение для сравнения: добавлять можно во всей полосе у пары страниц
            (в том числе над и под корешком), вогнутости заливаются по всей маске.

    Returns:
        Маска бумаги, добавленное, полоса между страницами.
    """
    from research.bound_volumes.paper import fill_concavities

    pages = split_pages(paper, book, hulls, instances)
    dists = [outside_distance(p) for p in pages]
    areas = [float(p.sum()) for p in pages]
    pairs = close_pairs(dists, pages)
    add = np.zeros(paper.shape, bool)
    band = np.zeros(paper.shape, bool)
    for i, j in pairs:
        band |= pair_band(dists, i, j) & (between(pages, i, j) if between_filter else True)
    if method == GlueMethod.METABALLS and pairs:
        # Поле метаболов: вклад страницы спадает от 1 на её краю до 0 на радиусе влияния r_i ∝ √A_i.
        field = np.zeros(paper.shape, np.float32)
        count = np.zeros(paper.shape, np.uint8)
        for d, a in zip(dists, areas):
            r = METABALL_K * np.sqrt(a)
            f = np.clip(1 - d / r, 0, 1) ** 2
            field += f
            count += (f > 0).astype(np.uint8)
        add = (field >= METABALL_TAU) & (count >= 2)
    elif method == GlueMethod.FERMAT:
        for i, j in pairs:
            w = min(W_MAX, FERMAT_C * np.sqrt(min(areas[i], areas[j])))
            add |= dists[i] + dists[j] <= w
    elif method == GlueMethod.HULL_BRIDGE and pairs:
        add |= hull_bridge_add(pages, dists, pairs)
    elif method == GlueMethod.ADAPTIVE_CLOSE:
        for i, j in pairs:
            r = int(min(W_MAX // 2, ADAPTIVE_K * np.sqrt(min(areas[i], areas[j]))))
            pad = r + 2
            u = np.pad((pages[i] | pages[j]).astype(np.uint8), pad)
            closed = cv2.morphologyEx(u, cv2.MORPH_CLOSE, disk(r))[pad:-pad, pad:-pad] > 0
            add |= closed & pair_band(dists, i, j)
    if not between_filter:
        # Прежнее поведение (до фильтра): добавленное внутри книги, дыры, запертые мостиками, — тоже бумага,
        # вогнутости заливаются по всей маске.
        out = (paper | add) & book
        out = ndimage.binary_fill_holes(out) & book
        if concave:
            out = fill_concavities(out, CONCAVE_DEPTH) & book
        return GlueResult(out, out & ~paper, band)
    # Добавлять можно только между страницами (не над и не под корешком — там торец и край крышки).
    add &= band & book & ~paper
    out = paper | add
    # Дыры, запертые мостиками (щель с обеих сторон закрыта), — тоже бумага, но тоже только между страницами.
    add |= ndimage.binary_fill_holes(out) & band & ~paper
    # Тонкие клочья у внешнего края (угол на зубчатом краю случайно вышел большим) срезаются открытием; остаются
    # только куски добавленного, касающиеся обеих страниц, — настоящие мостики.
    add = cv2.morphologyEx(add.astype(np.uint8), cv2.MORPH_OPEN, disk(ADD_OPEN_PX)) > 0
    add = bridges_only(add, pages)
    out = paper | add
    if concave:
        # Вогнутости заливаются только в полосе между страницами (у корешка): внешние края не трогаем.
        out |= fill_concavities(out, CONCAVE_DEPTH) & book & band
    return GlueResult(out, out & ~paper, band)
