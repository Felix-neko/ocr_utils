"""Тесты вето фона scan_cropping v2 на синтетике: фон только уменьшается, книга возвращает залив у края листа."""

import numpy as np

from ocr_utils.scan_cropping.v2.bg_veto import veto_background, veto_book
from ocr_utils.scan_cropping.v2.book import book_close_open, content_hulls


def frame_with_bay() -> tuple[np.ndarray, dict, np.ndarray]:
    """Кадр 400×600: тёмный коврик, светлая страница с печатью и залив фона у верхнего края (рукописный номер).

    Returns:
        ``(кадр BGR, маски шага «фон и книга», сырая страница SAM 3)``.
    """
    img = np.full((400, 600, 3), 20, np.uint8)
    img[60:360, 80:520] = 230
    page = np.zeros((400, 600), bool)
    page[60:360, 80:520] = True
    # Своя печать — строки в середине страницы.
    content = np.zeros_like(page)
    for y in range(160, 320, 20):
        content[y : y + 8, 120:480] = True
    # Фон flood: коврик плюс залив 100×80 px от верхнего края страницы вниз.
    bg = ~page
    bg[60:140, 200:300] = True
    book = book_close_open(bg, content_hulls(content, ~bg))
    return img, dict(content=content, bg=bg, book=book, hulls=content_hulls(content, book)), page


def test_veto_only_shrinks_background() -> None:
    """Вето не добавляет фон и отменяет его только там, где SAM 3 — страница, а SAM 2 — не фон."""
    bg = np.zeros((10, 10), bool)
    bg[:, :6] = True
    sam3 = np.zeros_like(bg)
    sam3[:, 3:] = True
    sam2 = np.zeros_like(bg)
    sam2[:, :4] = True
    out = veto_background(bg, sam3, sam2)
    assert not (out & ~bg).any()
    # Отменены столбцы 4–5: страница SAM 3 и не фон SAM 2; столбец 3 — фон и у SAM 2, остаётся.
    assert out[:, :4].all() and not out[:, 4:].any()


def test_veto_book_returns_bay_and_keeps_mat() -> None:
    """Залив у края страницы возвращается книге; коврик, который SAM 2 считает фоном, остаётся фоном."""
    img, book, page = frame_with_bay()
    assert not book["book"][60:140, 200:300].all()
    sam2_bg = ~page
    vetoed = veto_book(img, book, page, sam2_bg)
    assert vetoed["book"][60:140, 200:300].all()
    assert vetoed["bg"][:40].all()
    assert set(vetoed) == set(book)


def test_veto_book_keeps_background_where_sam2_disagrees() -> None:
    """SAM 2 считает залив фоном — вето его не трогает."""
    img, book, page = frame_with_bay()
    sam2_bg = book["bg"].copy()
    vetoed = veto_book(img, book, page, sam2_bg)
    assert (vetoed["bg"] == book["bg"]).all()
