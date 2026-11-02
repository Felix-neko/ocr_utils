"""Тесты своей стороны страниц на итоге (``fold.own_sides``) и закраса чужого в вырезке разбора (``fold.own_view``): полоса повтора концов строк у корешка и соседняя страница за корешком не попадают в разбор."""

import numpy as np

from ocr_utils.scan_cropping.v2.fill import FillMethod
from ocr_utils.scan_cropping.v2.rectify.coons import Layout, PageRect
from ocr_utils.scan_cropping.v2.rectify.fold import own_sides, own_view


def layout(single: bool = False) -> Layout:
    """Холст 200×100: страницы 10..100 и 100..190 по X, 10..90 по Y, корешок на X = 100 (или одна страница)."""
    if single:
        return Layout(200, 100, (PageRect(10, 10, 190, 90),), 100.0)
    return Layout(200, 100, (PageRect(10, 10, 100, 90), PageRect(100, 10, 190, 90)), 100.0)


def fold_field() -> tuple[np.ndarray, np.ndarray]:
    """Поле со складкой у корешка: левая страница сдвинута влево (+8 px), правая вправо (−12 px), скачок на X 92..94."""
    cols = np.arange(200, dtype=np.float32)
    row = np.where(cols < 92, 8.0, np.where(cols < 94, 8.0 - 10.0 * (cols - 91), -12.0)).astype(np.float32)
    dx = np.repeat(row[None, :], 100, axis=0)
    return dx, np.zeros_like(dx)


def test_own_sides_drop_fold_but_keep_margins():
    left, right = own_sides(layout(), fold_field())
    # Полоса повтора (источник — уже показанные концы строк левой страницы) — ничья.
    assert not left[50, 92:111].any() and not right[50, 92:112].any()
    # Поля за прямоугольником страницы — свои: ограничивает только корешок.
    assert left[5, 3] and right[95, 199]


def test_own_sides_single_page_is_none():
    zero = np.zeros((100, 200), np.float32)
    assert own_sides(layout(single=True), (zero, zero)) is None


def test_own_view_paints_foreign_ink():
    left, right = own_sides(layout(), fold_field())
    image = np.full((100, 200, 3), 235, np.uint8)
    image[40:46, 96:108] = 20  # повторённая буква левой страницы в полосе складки
    image[40:46, 130:140] = 20  # своя буква правой страницы
    box = (100, 0, 200, 100)
    view = own_view(image, right, box, FillMethod.PULL_PUSH, 300.0 / 25.4, 0.0)
    # Чужая краска закрашена бумагой, своя — нетронута; вне вырезки и исходная картинка — без изменений.
    assert view[42, 104].min() > 150
    assert (view[40:46, 130:140] == 20).all()
    assert (image[40:46, 96:108] == 20).all() and (view[:, :100] == image[:, :100]).all()


def test_own_view_without_foreign_returns_same_image():
    zero = np.zeros((100, 200), np.float32)
    left, right = own_sides(layout(), (zero, zero))
    image = np.full((100, 200, 3), 200, np.uint8)
    assert own_view(image, right, (100, 0, 200, 100), FillMethod.PULL_PUSH, 11.8, 0.0) is image
