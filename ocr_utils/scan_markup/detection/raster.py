"""Предварительный поиск растровых (полутоновых) областей на полосе.

ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ ``background_smoothing.layout``. Там задача была ЗАЩИТИТЬ уже
найденные блоки Surya от сглаживания фона, поэтому связные растровые компоненты брались
только те, что граничат с блоком: остальное всё равно попадало под страховку
``has_halftone``, браковавшую кадр целиком. Здесь задача обратная — НАЙТИ всё, в том числе
то, что Surya пропустила: растровую обложку без всякой вёрстки, вклейку во всю полосу,
фотографию, которую разметчик блока не получила. Поэтому компоненты берутся ВСЕ, а блоки
Surya к ним добавляются, а не наоборот.

Обе половины нужны и по отдельности слабы:

* Surya обводит визуальный блок и на смонтированной полосе прихватывает пустую бумагу
  вокруг, а край самой фотографии иногда срезает;
* компонента средних тонов обводит только то, что попало в диапазон 100..225, и на снимке
  со светлым небом заметно уже самого снимка (замер из ``layout.py``, 1968/01 IMG_0015_2R:
  блок y 1089..2926, компонента y 1376..2908).

Поэтому пересекающиеся прямоугольники ОБЪЕДИНЯЮТСЯ, и объединение никогда не сжимает
блок. Всё найденное всё равно правится руками в CVAT — задача этого модуля не угадать
границу с точностью до пикселя, а не дать разметчику рисовать с нуля.
"""

import logging

import cv2
import numpy as np

from ocr_utils.background_smoothing.layout import RASTER_CLOSE_PX, RASTER_MIN_AREA_PX
from ocr_utils.background_smoothing.processing import (
    HALFTONE_DOWNSCALE,
    HALFTONE_HI,
    HALFTONE_LO,
    HALFTONE_OPEN_PX,
    global_threshold,
)

logger = logging.getLogger(__name__)

# Зазор (в пикселях РАБОЧЕЙ копии 1/4), при котором два прямоугольника считаются одной
# областью. Разворот фотографии в вёрстке часто распадается на два-три пятна средних
# тонов с белым просветом между ними; отдавать их разметчику по отдельности — заставлять
# его вручную сливать то, что и так одна картинка.
MERGE_GAP_PX = 12

# Минимальная доля площади полосы, ниже которой область не считается иллюстрацией.
# Полоса 3492x6051 при 600 dpi — это 14.8x25.6 см, то есть 379 см^2; 0.2% от неё — около
# 0.9x0.9 см. Всё, что мельче, на полосе журнала — виньетка, буквица или грязь, а не
# растровая картинка. Порог строже, чем RASTER_MIN_AREA_PX (0.5x0.5 см), и работает
# поверх него: тот отсеивает компоненты, этот — уже слитые области.
MIN_REGION_FRAC = 0.002

# Доля площади полосы, с которой область считается «во всю страницу» (обложка, вклейка).
# Такая полоса целиком уходит в PDF без распрямления строк.
FULL_PAGE_FRAC = 0.75

# Минимальная сторона области (в пикселях РАБОЧЕЙ копии). Смыкание ядром RASTER_CLOSE_PX
# у самой рамки кадра дотягивает всё, что подошло к краю ближе половины ядра, до края
# (эффект описан в layout.py), и тень разворота превращается в лоскут во всю высоту
# полосы шириной в десяток точек. Иллюстраций такой формы не бывает: 40 точек копии — это
# ~0.7 см оригинала, тоньше не бывает даже узкая колонка с фотографией.
MIN_REGION_SIDE_PX = 40

# --- Обложки -----------------------------------------------------------------
# Обложка этого журнала — не фотография, а СПЛОШНАЯ КРАСКА: цветная плашка с выворотным
# шрифтом (замер, 1966/03 IMG_0104_2R: плашка серого 74..79). Детектор средних тонов её
# не видит принципиально — она темнее HALFTONE_LO. А бинаризовать её нельзя: выворотный
# шрифт превратится в кашу. Поэтому ищется второй признак — крупные сплошные пятна краски
# любой яркости, темнее порога бумаги.
#
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


def halftone_components(gray: np.ndarray, min_area: int = RASTER_MIN_AREA_PX) -> list[tuple[int, int, int, int]]:
    """Связные области средних тонов на РАБОЧЕЙ копии; прямоугольники ``(x1, y1, x2, y2)``.

    Та же маска и та же морфология, что у ``processing.has_halftone`` и
    ``layout.raster_regions``: диапазон ``HALFTONE_LO..HALFTONE_HI`` -> размыкание ядром
    ``HALFTONE_OPEN_PX`` (убирает серые каймы букв, у растровой печати пятна остаются) ->
    смыкание ядром ``RASTER_CLOSE_PX`` (сводит зерно одной фотографии в одну компоненту).

    Отличие от ``layout.raster_regions`` — возвращаются все компоненты площадью не меньше
    ``min_area``, без привязки к блокам Surya.
    """
    mid = ((gray > HALFTONE_LO) & (gray < HALFTONE_HI)).astype(np.uint8)
    mid = cv2.morphologyEx(mid, cv2.MORPH_OPEN, np.ones((HALFTONE_OPEN_PX,) * 2, np.uint8))
    mid = cv2.morphologyEx(mid, cv2.MORPH_CLOSE, np.ones((RASTER_CLOSE_PX,) * 2, np.uint8))

    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mid, 8)
    boxes = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area >= min_area:
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


def ink_components(
    gray: np.ndarray, min_area: int = RASTER_MIN_AREA_PX
) -> tuple[list[tuple[int, int, int, int]], float]:
    """Крупные сплошные пятна краски и доля «текстовой» краски вокруг них.

    Второй детектор, дополняющий :func:`halftone_components`. Тот ищет средние тона и
    находит полутоновую ФОТОГРАФИЮ; этот ищет всё, что темнее бумаги, и находит СПЛОШНУЮ
    ПЛАШКУ — цветной фон обложки с выворотным шрифтом, который в средние тона не попадает
    вовсе (замер: плашка обложки 1966/03 — серого 74..79 при HALFTONE_LO = 100).

    Текст отсеивается тем же размыканием ядром ``HALFTONE_OPEN_PX``, что и в детекторе
    полутонов: штрих основного текста при 600 dpi — единицы точек копии 1/4, плашка
    остаётся целиком.

    Вторым значением возвращается доля краски, которую размыкание УБРАЛО, — то есть
    оценка того, много ли на полосе текста. По ней :func:`is_cover_page` отличает обложку
    от полосы с чертежом.
    """
    ink = (gray < global_threshold(gray)).astype(np.uint8)
    opened = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((HALFTONE_OPEN_PX,) * 2, np.uint8))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((RASTER_CLOSE_PX,) * 2, np.uint8))

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
    """Обложка ли это: есть крупная сплошная краска и почти нет текста.

    Обложку недостаточно обвести по плашке — бинаризовать нельзя всю полосу целиком
    (вокруг плашки идёт фактура бумаги, рисунок и выворотные элементы), поэтому такая
    полоса помечается одной областью во весь кадр.
    """
    if text_frac > max_text_frac:
        return False
    ink_area = sum((box[2] - box[0]) * (box[3] - box[1]) for box in boxes)
    return ink_area >= min_ink_frac * width * height


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    """Пересекаются ли прямоугольники, если каждый раздуть на ``gap``."""
    return not (a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1])


def merge_boxes(boxes: list[tuple[int, int, int, int]], gap: int = MERGE_GAP_PX) -> list[tuple[int, int, int, int]]:
    """Сливает прямоугольники, пересекающиеся или отстоящие меньше чем на ``gap``.

    Слияние повторяется до неподвижной точки: объединение двух прямоугольников может
    дотянуться до третьего, который поодиночке не доставал ни до одного из них.
    """
    result = [tuple(map(int, box)) for box in boxes]
    changed = True
    while changed:
        changed = False
        merged: list[tuple[int, int, int, int]] = []
        for box in result:
            for index, other in enumerate(merged):
                if _overlaps(box, other, gap):
                    merged[index] = (
                        min(box[0], other[0]),
                        min(box[1], other[1]),
                        max(box[2], other[2]),
                        max(box[3], other[3]),
                    )
                    changed = True
                    break
            else:
                merged.append(box)
        result = merged
    return result


def polygons_to_boxes(polygons: list[np.ndarray]) -> list[tuple[int, int, int, int]]:
    """Полигоны Surya (4 точки) -> охватывающие прямоугольники."""
    boxes = []
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        boxes.append(
            (
                int(np.floor(points[:, 0].min())),
                int(np.floor(points[:, 1].min())),
                int(np.ceil(points[:, 0].max())),
                int(np.ceil(points[:, 1].max())),
            )
        )
    return boxes


def find_raster_boxes(
    work_bgr: np.ndarray,
    work_gray: np.ndarray,
    detector=None,
    min_region_frac: float = MIN_REGION_FRAC,
    merge_gap: int = MERGE_GAP_PX,
    min_side: int = MIN_REGION_SIDE_PX,
    cover_max_text_frac: float = COVER_MAX_TEXT_FRAC,
    cover_min_ink_frac: float = COVER_MIN_INK_FRAC,
) -> tuple[list[tuple[int, int, int, int]], bool]:
    """Растровые области на РАБОЧЕЙ копии полосы и признак «это обложка».

    Возвращает ``(прямоугольники (x1, y1, x2, y2), cover)``. При ``cover=True``
    прямоугольник ровно один — во весь кадр.

    ``detector`` — ``background_smoothing.layout.LayoutDetector`` или ``None`` (тогда
    работают только два пиксельных детектора; так гоняются тесты и так можно обойтись без
    GPU, ценой пропущенных светлых фотографий).

    Складываются три источника:

    * компоненты средних тонов — полутоновая фотография;
    * компоненты сплошной краски — плашка обложки, которую первый детектор не видит;
    * блоки ``Picture`` из Surya, уже отфильтрованные ``is_raster_block`` (чертежи и схемы
      отсеиваются).

    Дальше слияние пересекающихся, отсев тонких лоскутов от тени разворота и отсев мелочи
    по доле площади.
    """
    height, width = work_gray.shape[:2]

    ink_boxes, text_frac = ink_components(work_gray)
    if is_cover_page(ink_boxes, text_frac, width, height, cover_max_text_frac, cover_min_ink_frac):
        return [(0, 0, width, height)], True

    boxes = halftone_components(work_gray) + ink_boxes
    if detector is not None:
        boxes += polygons_to_boxes(detector.picture_polygons(work_bgr, work_gray))

    min_area = min_region_frac * width * height
    kept = []
    for box in merge_boxes(boxes, merge_gap):
        box_w, box_h = box[2] - box[0], box[3] - box[1]
        if box_w >= min_side and box_h >= min_side and box_w * box_h >= min_area:
            kept.append(box)
    return kept, False


def is_full_page(box: tuple[int, int, int, int], width: int, height: int, frac: float = FULL_PAGE_FRAC) -> bool:
    """Занимает ли область почти всю полосу — то есть обложку или вклейку.

    Отдельной ветки под обложки не нужно: у растровой обложки одна компонента средних
    тонов почти во весь кадр, и её находит уже :func:`halftone_components`, даже когда
    Surya не видит на полосе никакой вёрстки.
    """
    return (box[2] - box[0]) * (box[3] - box[1]) >= frac * width * height


def scale_box(box: tuple[int, int, int, int], scale: int = HALFTONE_DOWNSCALE) -> tuple[int, int, int, int]:
    """Прямоугольник рабочей копии -> координаты оригинала."""
    return box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale
