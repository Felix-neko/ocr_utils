"""Тесты уточнения зоны склейки (``research.bound_volumes.glue_refine``) на синтетике."""

import cv2
import numpy as np
import pytest

from research.bound_volumes.glue_refine import RefineMethod, refine


def scene() -> tuple:
    """Две страницы с текстом, щель 30 px между ними, у верхнего края щели — тёмно-синий «торец обложки».

    Returns:
        ``(bgr, книга, бумага до склейки, бумага после склейки, оболочки печати, страницы)``.
    """
    h, w = 500, 800
    bgr = np.full((h, w, 3), 15, np.uint8)  # коврик
    book = np.zeros((h, w), bool)
    book[40:460, 60:740] = True
    bgr[book] = (60, 40, 30)  # торец, обложка: тёмный синеватый
    left = np.zeros((h, w), bool)
    left[60:440, 80:385] = True
    right = np.zeros((h, w), bool)
    right[60:440, 415:720] = True
    paper_col = (200, 225, 235)
    bgr[left | right] = paper_col
    gap = np.zeros((h, w), bool)
    gap[60:440, 385:415] = True
    bgr[gap] = paper_col  # щель — бумага (складка у корешка)
    bgr[60:90, 385:415] = (90, 50, 30)  # у верхнего края щели — кусок обложки
    hulls = np.zeros((h, w), bool)
    for x0 in (100, 435):
        for y in range(100, 420, 20):
            cv2.line(bgr, (x0, y), (x0 + 260, y), (30, 30, 30), 2)
        hulls[95:425, x0 - 5 : x0 + 265] = True
    base = left | right
    glued = base | gap
    return bgr, book, base, glued, hulls, [left, right]


@pytest.mark.parametrize("method", list(RefineMethod))
def test_cover_piece_cut_paper_kept(method: RefineMethod) -> None:
    """Кусок обложки у края в зоне склейки срезается, бумага щели ниже остаётся."""
    bgr, book, base, glued, hulls, pages = scene()
    r = refine(bgr, book, base, glued, hulls, pages, method, max_depth=70)
    assert not r.paper[70, 400]
    assert r.paper[300, 400]


def test_deeper_than_band_untouched() -> None:
    """Глубже D от фона ничего не срезается, даже если цвет чужой."""
    bgr, book, base, glued, hulls, pages = scene()
    bgr[250:270, 390:410] = (90, 50, 30)
    r = refine(bgr, book, base, glued, hulls, pages, RefineMethod.COLOUR, max_depth=70)
    assert r.paper[260, 400]
