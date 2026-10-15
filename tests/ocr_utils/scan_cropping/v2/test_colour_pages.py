"""Тесты цветных страниц (синтетика): уровни цветности; оранжевая страница с кремовым полем — верхнее и нижнее поле в бумагу по палитре, красная ткань крышки и бело-голубой картон за тенью кромки — нет, вбок за сторону трапеции рост не идёт."""

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.paper import smooth_lab
from ocr_utils.scan_cropping.v2.colour_pages import Colour, chroma_of, edge_light, grow_by_palette, page_colour

H, W = 700, 700
CREAM = (200, 228, 240)
ORANGE = (40, 140, 240)
BROWN = (40, 60, 90)
CLOTH = (40, 40, 180)  # красная ткань крышки
BOARD = (245, 238, 230)  # бело-голубой картон


def orange_page() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Страница 150..600 × 150..550: кремовое поле 30 px, оранжевая плашка с коричневым текстом и кремовыми буквами;
    тень кромки листа; слева за кромкой красная ткань, сверху — бело-голубой картон; маска — только по плашке.

    Returns:
        ``(картинка, маска страницы, трапеция, книга, фон)``.
    """
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (20, 20, 20)
    img[100:650, 100:650] = BOARD
    img[100:650, 100:150] = CLOTH
    img[150:600, 150:550] = CREAM
    img[150:600, 150:152] = (150, 170, 180)  # тень кромки слева
    img[150:152, 150:550] = (150, 170, 180)  # тень кромки сверху
    img[180:570, 180:520] = ORANGE
    for y in range(230, 540, 30):
        cv2.putText(img, "TEXT text", (200, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BROWN, 2)
        cv2.putText(img, "white", (380, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CREAM, 2)
    img[560:598, 152:548] = (np.array(CREAM) * 0.92).astype(np.uint8)  # поле в тени у нижней кромки (~20 уровней L)
    page = np.zeros((H, W), bool)
    page[180:570, 180:520] = True
    quad = np.array([[180, 180], [520, 180], [520, 570], [180, 570]], np.float64)
    book = np.zeros((H, W), bool)
    book[100:650, 100:650] = True
    bg = ~book
    return img, page, quad, book, bg


def test_levels() -> None:
    """Оранжевая страница — целиком цветная; белая с одной плашкой — цветные элементы; чистая — обычная."""
    img, page, *_ = orange_page()
    assert page_colour(chroma_of(img), page).level is Colour.FULL
    white = np.full((200, 200, 3), CREAM, np.uint8)
    mask = np.ones((200, 200), bool)
    assert page_colour(chroma_of(white), mask).level is Colour.PLAIN
    white[20:40, 20:60] = ORANGE  # плашка 2 % страницы
    white[100:120, 20:80] = ORANGE
    assert page_colour(chroma_of(white), mask).level is Colour.ELEMENTS


def test_grow_takes_margin_not_cloth_or_board() -> None:
    """Верхнее и нижнее поле (в том числе в тени) — в бумагу; красная ткань и картон за кромкой — нет; вбок — нет."""
    img, page, quad, book, bg = orange_page()
    grown = grow_by_palette(smooth_lab(img), edge_light(img), page, quad, page, book, bg)
    assert grown[160:175, 200:500].mean() > 0.9  # верхнее поле
    # Боковые стороны — не палитра (за ними бывает белый торец блока): левее стороны трапеции + 2 мм рост не идёт.
    assert not grown[:, :160].any()
    assert grown[575:595, 200:500].mean() > 0.9  # нижнее поле в тени
    assert not grown[:, :148].any()  # ткань крышки и картон левее кромки
    assert not grown[:148, :].any()  # картон выше кромки
