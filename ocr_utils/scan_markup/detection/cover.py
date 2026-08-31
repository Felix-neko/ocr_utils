"""Обложка выпуска: первая полоса и детектор сплошной плашки.

ГДЕ ОБЛОЖКИ ЛЕЖАТ. У выпуска этого журнала их четыре — полосы 0 и 1 (обложки 1 и 2) и две
последние (обложки 3 и 4). Из них предсказуема ровно одна: полоса 0 всегда занята цветным
изображением во весь кадр. Остальные три бывают чем угодно — текстом, объявлением, парой
ч/б фотографий с подписями (1969/12 IMG_0150_2R), штриховым рисунком, — и обрабатываются
обычной детекцией.

ПОЧЕМУ ЭВРИСТИКА ЗАПЕРТА НА ПОЛОСУ 0. Признак «мало текстовой краски, много сплошной»
одинаково верен и для плашки обложки, и для внутренней полосы, целиком занятой двумя
фотографиями с подписями. Прошлая версия детектора звала :func:`is_cover_page` на каждой
полосе и на паке-1 объявила обложками две внутренних, накрыв обе фотографии одним
прямоугольником во весь кадр. Поэтому теперь функция вызывается ТОЛЬКО из
:func:`cover_decision`, и только когда полоса первая в выпуске: ограничение живёт в графе
вызовов, а не в договорённости.
"""

import cv2
import numpy as np

from ocr_utils.background_smoothing.processing import HALFTONE_OPEN_PX, global_threshold
from ocr_utils.scan_markup.detection.boxes import MIN_REGION_SIDE_PX

# Ядро смыкания маски краски на копии 1/4. То же значение и по той же причине, что
# ``background_smoothing.layout.RASTER_CLOSE_PX``: сводит разрывы одной плашки в одно пятно.
INK_CLOSE_PX = 31

# Минимальная площадь пятна краски на копии 1/4, ниже которой это не плашка, а крапина.
INK_MIN_AREA_PX = 900

# Признак «это обложка, а не полоса с чертежом» — почти полное отсутствие текста. Замер по
# 1966/03 (97 полос), доля «текстовой» краски (вся краска минус то, что уцелело после
# размыкания):
#
#     обложка IMG_0104_2R                       0.011
#     полоса с таблицей и плашками IMG_0121_1L  0.133
#     все 97 текстовых полос                    0.089 .. 0.197
#
# Порог 0.04 лежит посреди пустого промежутка, так что разделение устойчивое.
COVER_MAX_TEXT_FRAC = 0.04

# Доля сплошной краски, ниже которой полоса на обложку не претендует, даже будучи пустой:
# без этого чистый шмуцтитул с одной строчкой заголовка ушёл бы в растр целиком.
COVER_MIN_INK_FRAC = 0.05


def ink_components(gray: np.ndarray, min_area: int = INK_MIN_AREA_PX) -> tuple[list[tuple], float]:
    """Крупные сплошные пятна краски на копии 1/4 и доля «текстовой» краски вокруг них.

    Плашка обложки в средние тона не попадает вовсе (замер: плашка обложки 1966/03 —
    серого 74..79 при нижней границе средних тонов 100), поэтому ищется всё, что темнее
    бумаги. Текст отсеивается размыканием: штрих основного текста при 600 dpi — единицы
    точек копии 1/4, плашка остаётся целиком.

    Вторым значением возвращается доля краски, которую размыкание УБРАЛО, — то есть оценка
    того, много ли на полосе текста. По ней :func:`is_cover_page` отличает обложку от
    полосы с чертежом.
    """
    ink = (gray < global_threshold(gray)).astype(np.uint8)
    opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((HALFTONE_OPEN_PX,) * 2, np.uint8))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((INK_CLOSE_PX,) * 2, np.uint8))

    text_frac = float(np.count_nonzero(ink.astype(bool) & ~closed.astype(bool))) / float(ink.size)

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(closed, 8)
    boxes = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area >= min_area:
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes, text_frac


def is_cover_page(
    boxes: list[tuple[int, int, int, int]],
    text_frac: float,
    width: int,
    height: int,
    max_text_frac: float = COVER_MAX_TEXT_FRAC,
    min_ink_frac: float = COVER_MIN_INK_FRAC,
) -> bool:
    """Похожа ли полоса на обложку: есть крупная сплошная краска и почти нет текста.

    Вызывать только через :func:`cover_decision` — мотивировка в докстринге модуля.
    """
    if text_frac > max_text_frac:
        return False
    ink_area = sum((box[2] - box[0]) * (box[3] - box[1]) for box in boxes)
    return ink_area >= min_ink_frac * width * height


def cover_region(width: int, height: int) -> tuple[int, int, int, int]:
    """Область «вся полоса» — то, чем помечается обложка.

    Обложку недостаточно обвести по плашке: вокруг неё идёт фактура бумаги, рисунок и
    выворотные элементы, которые бинаризация испортит ровно так же.
    """
    return 0, 0, width, height


def cover_decision(
    order_index: int,
    work_gray: np.ndarray,
    first_page_is_cover: bool,
    max_text_frac: float = COVER_MAX_TEXT_FRAC,
    min_ink_frac: float = COVER_MIN_INK_FRAC,
) -> bool:
    """Считать ли эту полосу обложкой целиком. ``work_gray`` — серая копия 1/4.

    Полоса не первая в выпуске — не обложка, и пиксели даже не смотрим. Первая полоса при
    ``first_page_is_cover`` объявляется обложкой сразу; без флага решает эвристика сплошной
    плашки — она нужна, потому что первую полосу можно обрабатывать и без структурного
    допущения, например прогоняя ``detect`` по одному выпуску из середины пака.
    """
    if order_index != 0:
        return False
    if first_page_is_cover:
        return True
    height, width = work_gray.shape[:2]
    boxes, text_frac = ink_components(work_gray)
    return is_cover_page(boxes, text_frac, width, height, max_text_frac, min_ink_frac)


def lineart_ink_frac(work_gray: np.ndarray) -> float:
    """Доля площади полосы, занятая краской. Признак «полоса — сплошной рисунок».

    Считается по той же копии 1/4 и тем же порогом бумаги, что у :func:`ink_components`, но
    БЕЗ размыкания: у штрихового рисунка вся краска и есть содержимое, и убирать из неё
    тонкие штрихи — значит стереть сам рисунок. Отдельно от ``dots``, потому что вопрос
    другой: не «полутоновая ли это печать», а «занят ли рисунком весь кадр».
    """
    return float(np.count_nonzero(work_gray < global_threshold(work_gray))) / float(work_gray.size)


def lineart_side_ok(width: int, height: int) -> bool:
    """Не выродилась ли полоса в лоскут — страховка от кадра-полоски."""
    return width >= MIN_REGION_SIDE_PX and height >= MIN_REGION_SIDE_PX
