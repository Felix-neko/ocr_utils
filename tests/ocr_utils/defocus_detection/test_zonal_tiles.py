"""Зональная карта по сетке тайлов: находит провал там, где он есть, и молчит там, где нет."""

import cv2
import numpy as np
import pytest

from ocr_utils.defocus_detection.lines.measure import measure_lines
from ocr_utils.defocus_detection.lines.regions import LineRegion
from ocr_utils.defocus_detection.lines.zonal_tiles import tile_zonal
from ocr_utils.defocus_detection.metrics import ALGORITHMS

from .pages import blur, draw_text_lines

EDGE_WIDTH = ALGORITHMS["edge_width"]
SIZE = 1200


def page_with_lines(slant: float = 0.0, seed: int = 5):
    """Синтетическая полоса в пять колонок с полигонами строк по всей её площади.

    Args:
        slant: Наклон строк как тангенс.
        seed: Зерно генератора.

    Returns:
        Кортеж (кадр, список полигонов).
    """
    return draw_text_lines(height=SIZE, width=SIZE, stroke=4, line_height=26, columns=5, slant=slant, seed=seed)


def zonal(image, polygons, n=3):
    """Строит зональную карту по кадру и готовым полигонам строк.

    Args:
        image: Полутоновый кадр.
        polygons: Полигоны строк.
        n: Сторона сетки.

    Returns:
        ``TileZonalResult`` или None.
    """
    measurements = measure_lines(
        image, [LineRegion(polygon=p) for p in polygons], EDGE_WIDTH, n_tiles=n, corridor=(0.0, 100.0)
    )
    return tile_zonal(measurements, n=n)


def blur_corner(image, sigma, right=True, bottom=True):
    """Размывает один угол кадра с плавным переходом.

    Args:
        image: Полутоновый кадр.
        sigma: Сигма размытия в углу.
        right: Правый угол (иначе левый).
        bottom: Нижний угол (иначе верхний).

    Returns:
        Кадр с размытым углом.
    """
    h, w = image.shape
    ys = np.linspace(0.0, 1.0, h)[:, None]
    xs = np.linspace(0.0, 1.0, w)[None, :]
    weight = (ys if bottom else 1.0 - ys) * (xs if right else 1.0 - xs)
    weight = np.clip((weight - 0.15) / 0.5, 0.0, 1.0)
    blurred = cv2.GaussianBlur(image, (0, 0), sigma).astype(np.float64)
    return np.clip(image * (1.0 - weight) + blurred * weight, 0, 255).astype(np.uint8)


def test_flat_page_shows_no_zonal_defocus():
    """Ровно снятая полоса не должна давать зонального сигнала."""
    page, polygons = page_with_lines()
    result = zonal(blur(page, 1.0), polygons)
    assert result is not None
    assert result.drop < 0.25


def test_uniform_blur_is_not_a_zone():
    """Равномерно мягкий кадр — это плохой общий фокус, а не зональный расфокус.

    Ровно тот случай, ради которого зональная метрика относительна: она сравнивает
    части кадра друг с другом, а не с абсолютным порогом.
    """
    page, polygons = page_with_lines()
    soft = zonal(blur(page, 2.0), polygons)
    sharp = zonal(blur(page, 1.0), polygons)
    assert soft is not None and sharp is not None
    assert soft.drop == pytest.approx(sharp.drop, abs=0.3)


@pytest.mark.parametrize(
    ("right", "bottom", "expected"),
    [
        (True, True, "правый нижний угол"),
        (False, True, "левый нижний угол"),
        (True, False, "правый верхний угол"),
        (False, False, "левый верхний угол"),
    ],
)
def test_blurred_corner_is_found_and_named(right, bottom, expected):
    """Провал резкости в углу должен попасть ровно в тот тайл и называться правильно."""
    page, polygons = page_with_lines()
    result = zonal(blur_corner(blur(page, 0.8), 2.6, right=right, bottom=bottom), polygons)

    assert result is not None
    assert result.drop > 0.3, "перепад обязан быть заметным"
    iy, ix = result.worst
    assert (ix == result.n - 1) == right, "худший тайл не на той стороне по горизонтали"
    assert (iy == result.n - 1) == bottom, "худший тайл не на той стороне по вертикали"
    assert result.where().startswith(expected)


def test_slanted_lines_do_not_fake_a_zone():
    """ТРАПЕЦИЯ НЕ ДОЛЖНА ВЫГЛЯДЕТЬ КАК РАСФОКУС.

    Наклон строк меняется по кадру, и любая нарезка, прихватывающая бумагу или
    пересчитывающая пиксели, дала бы плавный градиент σ — идеально похожий на
    оптический завал плоскости и потому особенно коварный.
    """
    page, polygons = page_with_lines(slant=0.05)
    result = zonal(blur(page, 1.0), polygons)
    assert result is not None
    assert result.drop < 0.25


def test_empty_page_gives_no_zonal_map():
    """Кадр без текста (обложка подшивки) не даёт карты, а не даёт мусор."""
    page, _ = page_with_lines()
    measurements = measure_lines(page, [], EDGE_WIDTH, n_tiles=3)
    assert tile_zonal(measurements) is None


def test_sparse_tiles_are_dropped_not_guessed():
    """Тайл с горсткой строк не участвует: медиана по нему шумит сильнее эффекта."""
    page, polygons = page_with_lines()
    # Оставляем строки только в верхней трети кадра — заполнится один ряд тайлов.
    top_only = [p for p in polygons if p[:, 1].max() < SIZE / 3]
    measurements = measure_lines(
        page, [LineRegion(polygon=p) for p in top_only], EDGE_WIDTH, n_tiles=3, corridor=(0.0, 100.0)
    )
    result = tile_zonal(measurements, n=3)
    if result is not None:
        assert np.isnan(result.sharpness[2]).all(), "пустой нижний ряд не должен получать балл"


@pytest.mark.parametrize("n", [3, 4])
def test_grid_side_is_configurable(n):
    """Сетка задаётся стороной: 3 — это 3x3, 4 — 4x4."""
    page, polygons = page_with_lines()
    result = zonal(blur(page, 1.0), polygons, n=n)
    assert result is not None
    assert result.sharpness.shape == (n, n)
    assert result.n == n
