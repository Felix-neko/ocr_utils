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


def screen(shape: tuple[int, int], pitch: int = 8, radius: int = 2, seed: int = 0, blur: float = 0.6) -> np.ndarray:
    """Растровая сетка: точки на решётке с дрожанием — так печатается полутон.

    ``pitch`` — шаг сетки в пикселях. При 600 dpi настоящая сетка журнала это 5-10 px.

    ДРОЖАНИЕ ОБЯЗАТЕЛЬНО, и это не украшение. Идеальная решётка выстраивает точки в ровные
    строки по линиям развёртки, и признак «отточия» (``dots.leader_dot_features``) видит в
    ней ту же периодичность строк, что у колонки точек в оглавлении: замер по идеальной
    решётке дал пустых строк 0.47 при периодичности 0.87, то есть выше обоих порогов.
    Настоящая растровая сетка так не выглядит — она повёрнута к развёртке, размыта камерой,
    и размер точки гуляет с тоном; на 65 настоящих областях периодичность не превысила 0.51.
    Без дрожания фикстура изображала бы не полутон, а как раз то, что от него отличают.

    РАЗМЫТИЕ ТОЖЕ ОБЯЗАТЕЛЬНО. Резкая решётка двухцветна — в ней ровно два уровня яркости, —
    а настоящий скан размыт оптикой, и между краской и бумагой лежит вся шкала полутонов.
    Именно на ней держится ``detection.tone``: у резкой фикстуры доля средних тонов 0.000 и
    энтропия 0.83, то есть она выглядит штриховым рисунком; при sigma 0.6 — 0.373 и 4.29,
    как у настоящей фотографии. Больше 0.6 брать нельзя: при 0.8 точки смыкаются, и
    ``adaptiveThreshold`` перестаёт видеть отдельные пятна вовсе.
    """
    image = paper(shape)
    rng = np.random.default_rng(seed)
    for y in range(pitch, shape[0] - pitch, pitch):
        for x in range(pitch, shape[1] - pitch, pitch):
            dy, dx = rng.integers(-1, 2, 2)
            cv2.circle(image, (int(x + dx), int(y + dy)), radius, INK, -1)
    return cv2.GaussianBlur(image, (0, 0), blur) if blur else image


def with_screen(
    page: np.ndarray, box: tuple[int, int, int, int], pitch: int = 8, radius: int = 2, seed: int = 0, blur: float = 0.6
) -> np.ndarray:
    """Вклеивает растровое пятно в полосу; ``box`` — ``(x1, y1, x2, y2)``."""
    x1, y1, x2, y2 = box
    page[y1:y2, x1:x2] = screen((y2 - y1, x2 - x1), pitch, radius, seed, blur)
    return page


def dot_leaders(shape: tuple[int, int], line_step: int = 55, dot_step: int = 30, radius: int = 3) -> np.ndarray:
    """Колонка отточий, как в оглавлении: точки по базовым линиям текста.

    Между строками — бумага; именно этим отточия и отличаются от растровой сетки, которая
    заполняет площадь сплошь (см. ``dots.leader_dot_features``).
    """
    image = paper(shape)
    for y in range(line_step, shape[0] - line_step, line_step):
        for x in range(dot_step, shape[1] - dot_step, dot_step):
            cv2.circle(image, (x, y), radius, INK, -1)
    return image


def line_art(
    shape: tuple[int, int], step: int = 24, thickness: int = 4, seed: int = 0, blur: float = 0.6
) -> np.ndarray:
    """Штриховой рисунок: длинные связные штрихи вместо точек.

    Плотность краски примерно та же, что у растровой сетки, — разделять детектор обязан
    именно по размеру связных пятен, а не по количеству чёрного.

    ШТРИХИ НЕРЕГУЛЯРНЫ, и это существенно. Идеальная решётка параллельных линий одинаковой
    толщины — это дифракционная решётка, и в спектре она даёт пик мощнее любого растра: замер
    по такой фикстуре дал выступ 172 при 1.1..2.2 у настоящих чертежей из пака. Признак
    ``tone.screen_peak`` на ней срабатывал бы наоборот. Настоящий рисунок так не выглядит: у
    него гуляют и шаг, и толщина, и направление.

    Размытие — то же, что у ``screen``: скан не бывает резким. У штриха оно поднимает долю
    средних тонов с 0.000 до 0.103 (у растра — до 0.379), то есть разделение сохраняет.
    """
    image = paper(shape)
    rng = np.random.default_rng(seed)
    y = float(-shape[1])
    while y < shape[0] + shape[1]:
        width = int(rng.integers(max(2, thickness - 1), thickness + 2))
        cv2.line(image, (0, int(y)), (shape[1] - 1, int(y + shape[1] * rng.uniform(0.7, 1.3))), INK, width)
        y += float(rng.integers(max(6, step - 8), step + 9))
    return cv2.GaussianBlur(image, (0, 0), blur) if blur else image


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
