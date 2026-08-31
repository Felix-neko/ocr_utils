"""Синтетические полосы для тестов детекции: растровая сетка, штрих, текст, плашка.

ПОЧЕМУ НЕ ШУМ. Прошлые фикстуры изображали полутон случайным шумом в средних тонах — на
копии 1/4 этого хватало, потому что детектор смотрел только на яркость. Новый детектор
смотрит на РАЗМЕР пятен краски, и шум для него не полутон, а сплошная каша. Полутоновую
печать надо рисовать тем, чем она и является: регулярной сеткой мелких точек.
"""

import cv2
import numpy as np

PAPER = 245
INK = 30


def paper(shape: tuple[int, int]) -> np.ndarray:
    """Чистая бумага."""
    return np.full(shape, PAPER, np.uint8)


def screen(shape: tuple[int, int], pitch: int = 8, radius: int = 2) -> np.ndarray:
    """Растровая сетка: точки на регулярной решётке — так печатается полутон.

    ``pitch`` — шаг сетки в пикселях. При 600 dpi настоящая сетка журнала это 5-10 px.
    """
    image = paper(shape)
    for y in range(pitch, shape[0] - pitch, pitch):
        for x in range(pitch, shape[1] - pitch, pitch):
            cv2.circle(image, (x, y), radius, INK, -1)
    return image


def with_screen(page: np.ndarray, box: tuple[int, int, int, int], pitch: int = 8, radius: int = 2) -> np.ndarray:
    """Вклеивает растровое пятно в полосу; ``box`` — ``(x1, y1, x2, y2)``."""
    x1, y1, x2, y2 = box
    page[y1:y2, x1:x2] = screen((y2 - y1, x2 - x1), pitch, radius)
    return page


def line_art(shape: tuple[int, int], step: int = 24, thickness: int = 4) -> np.ndarray:
    """Штриховой рисунок: длинные связные штрихи вместо точек.

    Плотность краски примерно та же, что у растровой сетки, — разделять детектор обязан
    именно по размеру связных пятен, а не по количеству чёрного.
    """
    image = paper(shape)
    for y in range(-shape[1], shape[0] + shape[1], step):
        cv2.line(image, (0, y), (shape[1] - 1, y + shape[1]), INK, thickness)
    return image


def text_page(
    shape: tuple[int, int],
    line_step: int = 30,
    glyph_w: int = 7,
    glyph_h: int = 15,
    char_step: int = 13,
    margin: int = 60,
) -> np.ndarray:
    """Полоса сплошного текста: строки из отдельных «букв».

    Умолчания — размеры буквы на КОПИИ 1/4 кадра 600 dpi, то есть в том масштабе, где
    работает детектор обложки. Буквы обязаны быть раздельными: на слипшемся в сплошную
    заливку «тексте» размыкание ничего не убирает, и полоса выглядит как плашка обложки.
    """
    image = paper(shape)
    for y in range(margin, shape[0] - margin - glyph_h, line_step):
        for x in range(margin, shape[1] - margin - glyph_w, char_step):
            image[y : y + glyph_h, x : x + glyph_w] = INK
    return image


def solid_plate(shape: tuple[int, int], box: tuple[int, int, int, int], level: int = 75) -> np.ndarray:
    """Обложка-плашка: крупная сплошная краска и почти никакого текста."""
    image = paper(shape)
    x1, y1, x2, y2 = box
    image[y1:y2, x1:x2] = level
    return image
