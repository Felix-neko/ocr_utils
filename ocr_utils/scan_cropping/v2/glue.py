"""Склейка страниц разворота в маске бумаги (``hull_bridge`` без фильтра): щель и выемки у корешка становятся бумагой."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

# Страница — компонента маски бумаги не меньше этой доли маски книги.
PAGE_MIN_SHARE = 0.02
# Предельная ширина щели между страницами, которую склеиваем, px масштаба сегментации (~12 мм).
W_MAX = 80
# Поиск корешка: доли ширины книги и окно сглаживания профиля печати по столбцам, px.
SPINE_SEARCH = (0.2, 0.8)
SPINE_SMOOTH = 15


@dataclass(frozen=True)
class GlueResult:
    """Итог склейки: маска бумаги, добавленное и полоса между страницами (зона, где склейка допустима)."""

    paper: np.ndarray
    added: np.ndarray
    band: np.ndarray


def pages_of(paper: np.ndarray, book: np.ndarray) -> list[np.ndarray]:
    """Компоненты-страницы маски бумаги.

    Args:
        paper: Маска бумаги.
        book: Маска книги.

    Returns:
        Список булевых масок страниц.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(paper.astype(np.uint8), connectivity=8)
    min_area = PAGE_MIN_SHARE * max(int(book.sum()), 1)
    return [labels == i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def outside_distance(mask: np.ndarray) -> np.ndarray:
    """Расстояние до маски (0 внутри).

    Args:
        mask: Булева маска.

    Returns:
        Карта float32.
    """
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)


def split_pages(
    paper: np.ndarray, book: np.ndarray, hulls: np.ndarray | None = None, instances: np.ndarray | None = None
) -> list[np.ndarray]:
    """Страницы разворота, даже если SAM 3 и закрытие слили их по корешку в одну компоненту.

    Если в карте экземпляров SAM 3 не меньше двух страниц, пиксели бумаги раздаются ближайшему экземпляру. Иначе —
    ближайшей печати левой или правой страницы относительно корешка (самый широкий провал печати по столбцам в
    средней части книги). Без двух страниц печати — компоненты маски.

    Args:
        paper: Маска бумаги.
        book: Маска книги.
        hulls: Своя печать (защита глифов) или None.
        instances: Карта меток экземпляров-страниц SAM 3 или None.

    Returns:
        Маски страниц.
    """
    if instances is not None:
        ids = [i for i in np.unique(instances) if i > 0]
        if len(ids) >= 2:
            dists = np.stack([outside_distance(instances == i) for i in ids])
            owner = dists.argmin(axis=0)
            return [paper & (owner == k) for k in range(len(ids))]
    if hulls is not None and hulls.any():
        # Корешок — самый широкий провал печати по столбцам в средней части книги (а не середина книги: при
        # поднятой странице одна страница в кадре вдвое шире другой). Ядра страниц — печать слева и справа от него.
        xs = np.nonzero(book.any(axis=0))[0]
        x0, x1 = xs[0], xs[-1]
        lo, hi = int(x0 + SPINE_SEARCH[0] * (x1 - x0)), int(x0 + SPINE_SEARCH[1] * (x1 - x0))
        profile = np.convolve(hulls[:, lo:hi].sum(axis=0).astype(float), np.ones(SPINE_SMOOTH) / SPINE_SMOOTH, "same")
        spine = lo + int(np.argmin(profile))
        cols = np.arange(book.shape[1])[None, :]
        cores = [hulls & (cols < spine), hulls & (cols >= spine)]
        if all(c.sum() >= 1000 for c in cores):
            dists = np.stack([outside_distance(c) for c in cores])
            owner = dists.argmin(axis=0)
            return [paper & (owner == i) for i in range(2)]
    return pages_of(paper, book)


def pair_band(dists: list[np.ndarray], i: int, j: int, width: int = W_MAX) -> np.ndarray:
    """Полоса между страницами i и j: близко к обеим и примерно посередине.

    Args:
        dists: Расстояния до страниц.
        i: Первая страница.
        j: Вторая страница.
        width: Предельная ширина щели, px.

    Returns:
        Булева маска полосы.
    """
    a, b = dists[i], dists[j]
    return (a <= width) & (b <= width) & (np.abs(a - b) <= width)


def close_pairs(dists: list[np.ndarray], pages: list[np.ndarray], width: int = W_MAX) -> list[tuple[int, int]]:
    """Пары страниц, между которыми щель не шире ``width``.

    Args:
        dists: Расстояния до страниц.
        pages: Маски страниц.
        width: Предельная ширина, px.

    Returns:
        Пары индексов.
    """
    return [(i, j) for i in range(len(pages)) for j in range(i + 1, len(pages))
            if pages[j].any() and dists[i][pages[j]].min() <= width]  # fmt: skip


def hull_bridge_add(pages: list[np.ndarray], dists: list[np.ndarray], pairs: list[tuple[int, int]]) -> np.ndarray:
    """Мостик выпуклых оболочек: часть общей оболочки пары страниц, не накрытая оболочкой ни одной из них, в полосе.

    Args:
        pages: Маски страниц.
        dists: Расстояния до страниц.
        pairs: Близкие пары страниц (:func:`close_pairs`).

    Returns:
        Булева маска добавленного.
    """
    shape = pages[0].shape if pages else (0, 0)
    add = np.zeros(shape, bool)
    for i, j in pairs:
        hulls = []
        for m in (pages[i], pages[j], pages[i] | pages[j]):
            h = np.zeros(shape, np.uint8)
            cv2.fillPoly(h, [cv2.convexHull(cv2.findNonZero(m.astype(np.uint8)))], 1)
            hulls.append(h > 0)
        add |= hulls[2] & ~hulls[0] & ~hulls[1] & pair_band(dists, i, j)
    return add


def glue_hull_bridge(
    paper: np.ndarray, book: np.ndarray, hulls: np.ndarray | None = None, instances: np.ndarray | None = None
) -> GlueResult:
    """Склейка страниц мостиком выпуклых оболочек без фильтра «только между страницами» (``glue_hull_bridge_nofilter``).

    Добавленное ограничено книгой; дыры, запертые мостиками, тоже становятся бумагой. Что из добавленного у края
    книги на самом деле торец или обложка, решает уточнение (``glue_refine``).

    Args:
        paper: Маска бумаги до склейки.
        book: Маска книги.
        hulls: Своя печать (разделение слитых страниц по корешку, если нет экземпляров) или None.
        instances: Карта экземпляров-страниц SAM 3 или None.

    Returns:
        Маска бумаги, добавленное, полоса между страницами.
    """
    pages = split_pages(paper, book, hulls, instances)
    dists = [outside_distance(p) for p in pages]
    pairs = close_pairs(dists, pages)
    band = np.zeros(paper.shape, bool)
    for i, j in pairs:
        band |= pair_band(dists, i, j)
    add = hull_bridge_add(pages, dists, pairs) if pairs else np.zeros(paper.shape, bool)
    out = (paper | add) & book
    out = ndimage.binary_fill_holes(out) & book
    return GlueResult(out, out & ~paper, band)
