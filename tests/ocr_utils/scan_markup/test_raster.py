"""Поиск растровых областей: слияние, отсев мелочи, признак «во всю полосу»."""

import cv2
import numpy as np
import pytest

from ocr_utils.scan_markup.detection.raster import (
    find_raster_boxes,
    halftone_components,
    ink_components,
    is_cover_page,
    is_full_page,
    merge_boxes,
    polygons_to_boxes,
)


class _StubDetector:
    """Дублёр LayoutDetector: отдаёт заданные полигоны, не трогая GPU и Surya."""

    def __init__(self, polygons) -> None:
        self._polygons = polygons

    def picture_polygons(self, bgr, gray=None):
        return self._polygons


def _halftone(shape) -> np.ndarray:
    """Полутоновое пятно: шум в средних тонах, тот самый признак растровой печати."""
    rng = np.random.default_rng(1)
    return rng.integers(120, 210, shape, dtype=np.uint8)


def _page_with_photo(box=(60, 80, 260, 300), size=(500, 400)) -> np.ndarray:
    """Серая полоса: белая бумага, чёрные «строки» и одно полутоновое пятно."""
    gray = np.full(size, 245, np.uint8)
    for y in range(320, 480, 12):
        gray[y : y + 4, 40:360] = 20  # текст: тёмный, в средние тона не попадает
    x1, y1, x2, y2 = box
    gray[y1:y2, x1:x2] = _halftone((y2 - y1, x2 - x1))
    return gray


def _text_page(size=(1400, 900), lines=range(100, 1300, 30)) -> np.ndarray:
    """Полоса сплошного текста в масштабе рабочей копии кадра 600 dpi."""
    gray = np.full(size, 245, np.uint8)
    for y in lines:
        gray[y : y + 10, 100 : size[1] - 100] = 20
    return gray


def test_merge_joins_through_a_third_box() -> None:
    """Слияние идёт до неподвижной точки: объединение двух может дотянуться до третьего."""
    assert merge_boxes([(0, 0, 10, 10), (20, 0, 30, 10), (9, 0, 21, 10)], gap=0) == [(0, 0, 30, 10)]


def test_merge_keeps_distant_boxes_apart() -> None:
    """Далёкие друг от друга области остаются раздельными."""
    assert len(merge_boxes([(0, 0, 10, 10), (100, 100, 110, 110)], gap=12)) == 2


def test_halftone_components_find_the_photo_and_ignore_text() -> None:
    """Компонента средних тонов находит фотографию; строки текста в неё не попадают.

    У текста серое сидит тонкой каймой по краям букв, и размыкание её убирает — на этом
    и держится весь детектор (см. processing.has_halftone).
    """
    boxes = halftone_components(_page_with_photo())
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]
    assert 40 <= x1 <= 80 and 60 <= y1 <= 100
    assert 240 <= x2 <= 280 and 280 <= y2 <= 320


def test_surya_block_and_component_are_merged_not_duplicated() -> None:
    """Блок Surya и найденная под ним компонента дают ОДНУ область, а не две.

    Блок обводит вёрстку и прихватывает бумагу вокруг, компонента — только средние тона;
    разметчику нужен их союз, а не два вложенных прямоугольника.
    """
    gray = _page_with_photo()
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    block = np.array([[50, 70], [270, 70], [270, 310], [50, 310]], np.float32)

    boxes, cover = find_raster_boxes(bgr, gray, _StubDetector([block]))
    assert not cover
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]
    assert x1 <= 50 and y1 <= 70 and x2 >= 270 and y2 >= 310


def test_detector_is_optional() -> None:
    """Без Surya детектор работает по одним полутонам — так гоняются тесты и прогон без GPU."""
    gray = _page_with_photo()
    assert len(find_raster_boxes(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), gray, None)[0]) == 1


@pytest.mark.parametrize("side,kept", [(40, False), (80, True)])
def test_small_specks_are_dropped_by_area_fraction(side: int, kept: bool) -> None:
    """Крапина мельче min_region_frac иллюстрацией не считается, крупное пятно — считается.

    Размеры полосы — реальные для рабочей копии 1/4 кадра 600 dpi (1513x873), иначе порог
    в долях площади проверялся бы не на том масштабе: 0.2% от такой полосы — это ~51x51
    точки копии, то есть около 0.9x0.9 см оригинала.
    """
    gray = _text_page()
    gray[700 : 700 + side, 400 : 400 + side] = _halftone((side, side))
    boxes, _cover = find_raster_boxes(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), gray, None)
    assert bool(boxes) is kept


def test_photo_covering_whole_page_is_full_page() -> None:
    """Полутоновая вклейка во весь кадр находится и без разметки вёрстки."""
    gray = _halftone((500, 400))
    boxes, _cover = find_raster_boxes(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), gray, None)
    assert len(boxes) == 1
    assert is_full_page(boxes[0], 400, 500)


def test_solid_ink_cover_is_detected_whole_page() -> None:
    """Обложка-плашка помечается областью во ВЕСЬ кадр, а не по границе плашки.

    Плашка темнее HALFTONE_LO, поэтому детектор полутонов её не видит вовсе; находит её
    детектор сплошной краски. Обвести только плашку мало: вокруг неё идут выворотные
    элементы и фактура бумаги, которые бинаризация тоже испортит.
    """
    gray = np.full((1400, 900), 245, np.uint8)
    gray[100:500, 50:850] = 75  # цветная плашка с выворотным шрифтом
    gray[1200:1230, 700:850] = 20  # номер выпуска — немного текста внизу

    boxes, cover = find_raster_boxes(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), gray, None)
    assert cover
    assert boxes == [(0, 0, 900, 1400)]
    assert is_full_page(boxes[0], 900, 1400)


def test_text_page_with_black_blocks_is_not_a_cover() -> None:
    """Полоса с чертежом или таблицей на обложку не претендует: текста на ней много.

    Замер по 1966/03: у обложки доля текстовой краски 0.011, у полосы с плашками 0.133,
    у всех текстовых полос 0.089..0.197.
    """
    gray = _text_page()
    gray[200:400, 200:500] = 20  # чёрные плашки схемы посреди текста

    boxes, cover = find_raster_boxes(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), gray, None)
    assert not cover
    assert boxes and boxes != [(0, 0, 900, 1400)]


def test_ink_components_report_text_fraction() -> None:
    """Доля текстовой краски у полосы текста высокая, у полосы с плашкой — низкая."""
    _boxes, text_page_frac = ink_components(_text_page())
    solid = np.full((1400, 900), 245, np.uint8)
    solid[100:500, 50:850] = 75
    _boxes, cover_frac = ink_components(solid)
    assert text_page_frac > 0.05 > cover_frac


def test_is_cover_page_needs_enough_ink() -> None:
    """Пустой шмуцтитул с одной строкой в растр целиком не уходит."""
    assert not is_cover_page([(0, 0, 50, 50)], text_frac=0.001, width=900, height=1400)
    assert is_cover_page([(0, 0, 800, 400)], text_frac=0.001, width=900, height=1400)


def test_thin_gutter_strip_is_dropped() -> None:
    """Лоскут от тени разворота вдоль края отбрасывается по минимальной стороне.

    Смыкание у рамки кадра дотягивает подошедшее к краю до самого края, и тень разворота
    превращается в полосу во всю высоту шириной в десяток точек.
    """
    gray = _text_page()
    gray[:, :12] = 60  # тёмная кромка вдоль левого края
    boxes, _cover = find_raster_boxes(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), gray, None)
    assert all(box[2] - box[0] >= 40 for box in boxes)


@pytest.mark.parametrize("box,expected", [((0, 0, 380, 480), True), ((0, 0, 200, 250), False)])
def test_is_full_page_threshold(box, expected) -> None:
    """Порог «во всю полосу» — доля площади, а не касание краёв."""
    assert is_full_page(box, 400, 500) is expected


def test_polygons_to_boxes_takes_the_bounding_rect() -> None:
    """Полигон Surya сводится к охватывающему прямоугольнику с округлением наружу."""
    polygon = np.array([[10.4, 20.6], [99.2, 20.6], [99.2, 80.1], [10.4, 80.1]], np.float32)
    assert polygons_to_boxes([polygon]) == [(10, 20, 100, 81)]
