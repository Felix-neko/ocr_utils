"""Повёрнутая полоса в сборщике промежуточных PDF.

``pages.width/height`` описывают ОРИГИНАЛ, а файлы на диске у такой полосы повёрнуты. Здесь
проверяется, что сборщик знает про это в обоих местах: и в проверке размера, и в координатах
врезок — иначе иллюстрация легла бы на совершенно другое место полосы.
"""

from __future__ import annotations

import numpy as np
import pytest

from ocr_utils.pdf_utils.intermediate_pdfs import PagePlan, PicturePlan
from ocr_utils.db.models import KIND_GRAYSCALE
from ocr_utils.scan_markup.rotation import rotate_box, rotate_cw, rotate_size

WIDTH, HEIGHT = 3420, 6071


def plan(rotate: int, pictures=()) -> PagePlan:
    return PagePlan(
        page_id=1,
        original_rel_path="1967/01/a.tif",
        sharpened_rel_path="1967/01/a.jpg",
        width=WIDTH,
        height=HEIGHT,
        dpi=600,
        pictures=tuple(pictures),
        full_pdf_page_idx=0,
        pages_with_pics_only_pdf_page_idx=None,
        rotate_cw=rotate,
    )


@pytest.mark.parametrize(
    "rotate,expected", [(0, (WIDTH, HEIGHT)), (90, (HEIGHT, WIDTH)), (180, (WIDTH, HEIGHT)), (270, (HEIGHT, WIDTH))]
)
def test_expected_file_size_follows_the_rotation(rotate, expected):
    assert plan(rotate).file_size == expected


def test_pictures_are_untouched_without_rotation():
    picture = PicturePlan(10, 20, 30, 40, KIND_GRAYSCALE)
    assert plan(0, [picture]).placed_pictures() == (picture,)


@pytest.mark.parametrize("rotate", [90, 180, 270])
def test_pictures_move_with_the_page(rotate):
    """Проверка на пикселях: пересчитанный прямоугольник обязан накрыть то же содержимое."""
    picture = PicturePlan(300, 900, 1200, 2400, KIND_GRAYSCALE)
    placed = plan(rotate, [picture]).placed_pictures()[0]

    marked = np.zeros((HEIGHT, WIDTH), np.uint8)
    marked[picture.y1 : picture.y2, picture.x1 : picture.x2] = 255
    turned = rotate_cw(marked, rotate)
    check = np.zeros_like(turned)
    check[placed.y1 : placed.y2, placed.x1 : placed.x2] = 255
    assert np.array_equal(turned, check)
    assert (placed.x2 - placed.x1, placed.y2 - placed.y1) == rotate_size(
        picture.x2 - picture.x1, picture.y2 - picture.y1, rotate
    )


def test_a_full_page_picture_stays_full_page_after_rotation():
    """Полосную иллюстрацию сборщик кладёт подложкой, а не врезкой, — признак не должен теряться."""
    covering = PicturePlan(0, 0, WIDTH, HEIGHT, KIND_GRAYSCALE)
    placed = plan(90, [covering]).placed_pictures()[0]
    assert placed.covers(*plan(90).file_size)


def test_box_rotation_is_reversible():
    box = (11, 22, 333, 444)
    turned = rotate_box(box, WIDTH, HEIGHT, 90)
    back = rotate_box(turned, *rotate_size(WIDTH, HEIGHT, 90), 270)
    assert back == box
