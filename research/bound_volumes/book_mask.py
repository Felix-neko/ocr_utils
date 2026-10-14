"""Постобработка маски книги «не фон»: закрытие дыр с обязательной своей печатью и срез выступов на узких перешейках."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from scipy import ndimage

from ocr_utils.scan_cropping.v2.book import (  # noqa: F401 — ядро, ре-экспорт для стенда
    CLOSE_R,
    OPEN_R,
    MIN_PROTRUSION_PX,
    disk,
    close_holes,
    cut_protrusions,
)

# Для варианта hull_guard: выступ — часть дальше этого от оболочки печати, px.
HULL_GUARD_D = 120


class PostVariant(str, Enum):
    """Вариант постобработки маски книги."""

    # Как было: не фон − руки.
    RAW = "raw"
    # Не фон, руки не вычитаются (кадр после закраса пальцев).
    NO_HANDS = "no_hands"
    # NO_HANDS + закрытие CLOSE_R + заливка дыр + оболочки своей печати.
    CLOSE = "close"
    # CLOSE + срез выступов открытием OPEN_R с восстановлением.
    CLOSE_OPEN = "close_open"
    # CLOSE + срез частей дальше HULL_GUARD_D от оболочки печати, держащихся на перешейке.
    HULL_GUARD = "hull_guard"


@dataclass(frozen=True)
class PostResult:
    """Итог постобработки.

    Поля:
        book: Маска книги после постобработки.
        added: Добавленное закрытием, заливкой и оболочкой печати.
        removed: Снятое как выступ.
    """

    book: np.ndarray
    added: np.ndarray
    removed: np.ndarray








def hull_guard(book: np.ndarray, hulls: np.ndarray, d: int = HULL_GUARD_D) -> np.ndarray:
    """Срез частей маски дальше ``d`` от оболочек печати (и того, что с ними связано только через эту даль).

    Args:
        book: Маска книги.
        hulls: Оболочки своей печати.
        d: Предельное расстояние, px.

    Returns:
        Маска книги.
    """
    if not hulls.any():
        return book
    dist = cv2.distanceTransform((~hulls).astype(np.uint8), cv2.DIST_L2, 5)
    near = book & (dist <= d)
    far = book & ~near
    labels, count = ndimage.label(far)
    out = near.copy()
    # Далёкие части оставляем, если они прилегают к ближней зоне широко (обрез и обложка вдоль всей стороны).
    ring = cv2.dilate(near.astype(np.uint8), disk(2)) > 0
    for i in range(1, count + 1):
        part = labels == i
        contact = (cv2.dilate(part.astype(np.uint8), disk(2)) > 0) & ring & near
        if contact.sum() >= 0.5 * np.sqrt(part.sum()) * 4:
            out |= part
    return ndimage.binary_fill_holes(out)


def postprocess(bg: np.ndarray, hands: np.ndarray, hulls: np.ndarray, variant: PostVariant) -> PostResult:
    """Маска книги по маске фона в заданном варианте постобработки.

    Args:
        bg: Маска фона.
        hands: Маска рук (используется только в варианте RAW).
        hulls: Оболочки своей печати.
        variant: Вариант.

    Returns:
        Маска книги и разности с исходной «не фон».
    """
    from research.bound_volumes.background import book_from_background

    base = book_from_background(bg, np.zeros_like(bg))
    if variant == PostVariant.RAW:
        book = book_from_background(bg, hands)
        return PostResult(book, book & ~base, base & ~book)
    if variant == PostVariant.NO_HANDS:
        return PostResult(base, np.zeros_like(base), np.zeros_like(base))
    if variant == PostVariant.CLOSE:
        book = close_holes(base, hulls)
    elif variant == PostVariant.CLOSE_OPEN:
        # Выступы срезаются до закрытия: иначе закрытие заливает щель между книгой и мышью (04/IMG_1129).
        book = close_holes(cut_protrusions(base, hulls), hulls)
    else:
        book = close_holes(hull_guard(base, hulls), hulls)
    return PostResult(book, book & ~base, base & ~book)
