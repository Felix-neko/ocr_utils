"""Алгебра прямоугольников разметки: слияние, отсев мелочи, пересчёт координат."""

import numpy as np
import pytest

from ocr_utils.scan_markup.detection.boxes import (
    clamp_box,
    is_full_page,
    keep_significant,
    merge_boxes,
    polygons_to_boxes,
    upscale_box,
)


def test_merge_joins_through_a_third_box() -> None:
    """Слияние идёт до неподвижной точки: объединение двух может дотянуться до третьего."""
    assert merge_boxes([(0, 0, 10, 10), (20, 0, 30, 10), (9, 0, 21, 10)], gap=0) == [(0, 0, 30, 10)]


def test_merge_keeps_distant_boxes_apart() -> None:
    """Далёкие друг от друга области остаются раздельными."""
    assert len(merge_boxes([(0, 0, 10, 10), (100, 100, 110, 110)], gap=12)) == 2


@pytest.mark.parametrize("side,kept", [(150, False), (400, True)])
def test_small_specks_are_dropped(side: int, kept: bool) -> None:
    """Крапина отбрасывается и по минимальной стороне, и по доле площади полосы.

    Размеры полосы — реальные для кадра 600 dpi. Оба порога нужны: 176x272 (пятно от пальца
    у края, 1967/05 IMG_0060_2R) проходит по площади, но не проходит по стороне.
    """
    boxes = keep_significant([(100, 100, 100 + side, 100 + side)], 3492, 6051)
    assert bool(boxes) is kept


def test_thin_strip_is_dropped_by_side() -> None:
    """Лоскут от тени разворота вдоль края отбрасывается, хотя площадь у него большая."""
    assert keep_significant([(0, 0, 60, 6000)], 3492, 6051) == []


@pytest.mark.parametrize("box,expected", [((0, 0, 3400, 5900), True), ((0, 0, 1000, 1000), False)])
def test_is_full_page_threshold(box, expected) -> None:
    """Порог «во всю полосу» — доля площади, а не касание краёв."""
    assert is_full_page(box, 3492, 6051) is expected


def test_polygons_to_boxes_takes_the_bounding_rect() -> None:
    """Полигон Surya сводится к охватывающему прямоугольнику с округлением наружу."""
    polygon = np.array([[10.4, 20.6], [99.2, 20.6], [99.2, 80.1], [10.4, 80.1]], np.float32)
    assert polygons_to_boxes([polygon]) == [(10, 20, 100, 81)]


def test_upscale_and_clamp() -> None:
    """Подъём координат копии до оригинала и загон вылезшего края обратно в кадр."""
    assert upscale_box((10, 20, 30, 40), 4) == (40, 80, 120, 160)
    assert clamp_box((40, 80, 120, 160), 100, 150) == (40, 80, 100, 150)
