"""Грубая маска книги из маски фона (закрытие дыр, срез выступов) и выпуклые оболочки своей печати по страницам."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from ocr_utils.scan_cropping.v2.background import book_from_background

# Радиус закрытия дыр, px масштаба сегментации (~6.9 px/мм).
CLOSE_R = 30
# Радиус открытия при поиске выступов, px: выступ — то, что держится на перешейке уже 2·R.
OPEN_R = 60
# Выступ отбрасывается, только если его площадь не меньше этого, px² (мелочь у края не трогаем).
MIN_PROTRUSION_PX = 3000
# Ложная «печать» на торце стопки: узкий сгусток вдоль боковой кромки (линии между листами защита глифов принимает за
# строки, 04/IMG_1132). Отбрасываются сгустки с удлинением не меньше EDGE_LINE_ASPECT и длинной осью не дальше
# EDGE_LINE_DEG от вертикали.
EDGE_LINE_ASPECT = 4.0
EDGE_LINE_DEG = 30.0
# Радиус открытия фона перед построением книги, px: нити фона уже 2·R (≤ ~1.5 мм) внутри «не фона» — линейки печати,
# до которых заливка фона дотекла от коврика через тень у корешка, а не коврик. Без этого верхняя линия рамки страницы
# отрезала поле над ней, и поле срезалось как выступ (РИСК 1991/05 IMG_0663: линия 6 px, поле 15 мм).
BG_THIN_R = 5


def disk(r: int) -> np.ndarray:
    """Диск радиуса ``r`` для морфологии.

    Args:
        r: Радиус, px.

    Returns:
        Ядро uint8.
    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def close_holes(book: np.ndarray, hulls: np.ndarray, r: int = CLOSE_R) -> np.ndarray:
    """Закрытие диском ``r`` (с полем, чтобы край кадра не мешал), заливка дыр, плюс оболочки своей печати.

    Args:
        book: Маска книги.
        hulls: Выпуклые оболочки своей печати по страницам.
        r: Радиус закрытия, px.

    Returns:
        Маска книги без дыр.
    """
    pad = r + 2
    padded = np.pad((book | hulls).astype(np.uint8), pad, mode="edge")
    closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, disk(r))[pad:-pad, pad:-pad] > 0
    return ndimage.binary_fill_holes(closed | hulls)


def cut_protrusions(book: np.ndarray, hulls: np.ndarray, r: int = OPEN_R) -> np.ndarray:
    """Срез выступов: части маски, исчезающие при открытии диском ``r``, без своей печати и не меньше порога площади.

    Открытие убирает всё, что уже 2·r, в том числе узкий обрез вдоль края книги; поэтому снимается не вся разность, а
    только компоненты разности, которые (1) крупнее :data:`MIN_PROTRUSION_PX` и (2) далеко выдаются наружу от открытой
    маски — дальше ``r`` px (узкая полоса обреза выдаётся меньше).

    Args:
        book: Маска книги.
        hulls: Оболочки своей печати (части с печатью не срезаются).
        r: Радиус открытия, px.

    Returns:
        Маска книги без выступов.
    """
    pad = r + 2
    padded = np.pad(book.astype(np.uint8), pad, mode="constant")
    opened = cv2.morphologyEx(padded, cv2.MORPH_OPEN, disk(r))[pad:-pad, pad:-pad] > 0
    # Крупнейшая компонента открытой маски (и все с печатью) — ядро книги.
    labels, count = ndimage.label(opened)
    if count == 0:
        return book
    sizes = ndimage.sum(opened, labels, index=range(1, count + 1))
    core_ids = {1 + int(np.argmax(sizes))} | set(np.unique(labels[hulls & opened]).tolist()) - {0}
    core = np.isin(labels, list(core_ids))
    rest = book & ~core
    dist = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
    labels, count = ndimage.label(rest)
    out = book.copy()
    for i in range(1, count + 1):
        part = labels == i
        if part.sum() < MIN_PROTRUSION_PX or (part & hulls).any():
            continue
        if dist[part].max() > r:
            out &= ~part
    return ndimage.binary_fill_holes(out)


def book_close_open(bg: np.ndarray, hulls: np.ndarray) -> np.ndarray:
    """Грубая маска книги: «не фон», срез выступов, затем закрытие дыр с оболочками своей печати.

    Выступы срезаются до закрытия: иначе закрытие заливает щель между книгой и посторонним предметом (04/IMG_1129).
    Тонкие нити фона (линейки печати, до которых дотекла заливка) перед этим возвращаются книге (``drop_thin_bg``).
    Руки не вычитаются: кадр уже без пальцев.

    Args:
        bg: Маска фона.
        hulls: Оболочки своей печати (``content_hulls(content, ~bg)``).

    Returns:
        Булева маска книги.
    """
    base = book_from_background(drop_thin_bg(bg), np.zeros_like(bg))
    return close_holes(cut_protrusions(base, hulls), hulls)


def drop_thin_bg(bg: np.ndarray, r: int = BG_THIN_R) -> np.ndarray:
    """Фон без тонких нитей: открытие диском ``r`` (с полем по краю кадра, чтобы фон у рамки не подъедался).

    Args:
        bg: Маска фона.
        r: Радиус открытия, px.

    Returns:
        Маска фона без нитей уже ``2·r``.
    """
    pad = r + 2
    padded = np.pad(bg.astype(np.uint8), pad, mode="edge")
    return cv2.morphologyEx(padded, cv2.MORPH_OPEN, disk(r))[pad:-pad, pad:-pad] > 0


def drop_edge_lines(content: np.ndarray) -> np.ndarray:
    """Убирает из своей печати узкие почти вертикальные сгустки — линии торца стопки.

    Пиксели компонент берутся одной сортировкой меток (а не маской на каждую компоненту), иначе на тысячах знаков
    функция работает минутами.

    Args:
        content: Своя печать (bool).

    Returns:
        Печать без таких сгустков (bool).
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(content.astype(np.uint8), connectivity=8)
    keep = np.ones(count, bool)
    # Кандидаты — только вытянутые по рамке компоненты (высота ≥ ширины): остальные заведомо не вертикальные линии.
    cand = np.nonzero((stats[:, cv2.CC_STAT_HEIGHT] >= EDGE_LINE_ASPECT * 0.5 * stats[:, cv2.CC_STAT_WIDTH])
                      & (stats[:, cv2.CC_STAT_AREA] >= 5))[0]  # fmt: skip
    cand = cand[cand > 0]
    if len(cand) == 0:
        return content
    flat = labels.ravel()
    order = np.argsort(flat, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(flat, minlength=count))]
    w = labels.shape[1]
    for i in cand:
        pix = order[bounds[i] : bounds[i + 1]]
        (_, _), (rw, rh), ang = cv2.minAreaRect(np.c_[pix % w, pix // w].astype(np.float32))
        long_side, short_side = max(rw, rh), max(min(rw, rh), 1.0)
        # Угол длинной оси от вертикали.
        axis = ang if rw >= rh else ang + 90
        from_vertical = abs(((axis % 180) - 90))
        if long_side / short_side >= EDGE_LINE_ASPECT and from_vertical <= EDGE_LINE_DEG:
            keep[i] = False
    return keep[labels] & content


def content_hulls(content: np.ndarray, book: np.ndarray) -> np.ndarray:
    """Выпуклая оболочка своей печати по каждой странице (левая и правая половины книги): накрывает линейки таблиц
    и колонтитулов, которых нет в защите глифов.

    Args:
        content: Своя печать (bool).
        book: Маска книги (bool).

    Returns:
        Маска оболочек (bool).
    """
    content = drop_edge_lines(content)
    xs = np.nonzero(book.any(axis=0))[0]
    out = np.zeros(content.shape, np.uint8)
    if len(xs) == 0:
        return out > 0
    mid = int((xs[0] + xs[-1]) / 2)
    for part in (slice(0, mid), slice(mid, content.shape[1])):
        sub = np.zeros_like(content)
        sub[:, part] = content[:, part] & book[:, part]
        ys, xx = np.nonzero(sub)
        if len(ys) < 50:
            continue
        hull = cv2.convexHull(np.c_[xx, ys].astype(np.int32))
        cv2.fillPoly(out, [hull], 1)
    return out > 0
