"""Детектор полутоновой печати: растровая сетка против штриха и текста."""

import numpy as np
import pytest

from ocr_utils.scan_markup.detection.dots import (
    REFERENCE_DPI,
    cell_maps,
    dot_fraction,
    ink_components,
    is_dot,
    params_for_dpi,
    screen_boxes,
)
from tests.ocr_utils.scan_markup import synthetic

SIZE = (1800, 1200)  # полоса 300 dpi: настоящие 21 Мп тест гонять незачем
DPI = 300


def _params():
    return params_for_dpi(DPI)


def _text():
    """Текст в масштабе ПОЛНОГО кадра 300 dpi: буква вдвое крупнее, чем на копии 1/4."""
    return synthetic.text_page(SIZE, line_step=60, glyph_w=14, glyph_h=30, char_step=26)


def test_screen_is_found_and_line_art_is_not() -> None:
    """Главное разделение: сетка точек — растр, штрих той же плотности — нет.

    Это ровно тот дефект, ради которого детектор переписан: 31 полоса пака-1 со штриховыми
    виньетками и эмблемами рубрик уезжала в растр, потому что на копии 1/4 плотная штриховка
    неотличима от полутона по яркости.
    """
    params = _params()
    assert screen_boxes(synthetic.screen(SIZE, pitch=4, radius=1), params)[0]
    assert not screen_boxes(synthetic.line_art(SIZE), params)[0]


def test_text_page_gives_nothing() -> None:
    """Полоса сплошного текста растровых областей не даёт: буквы крупнее точки растра."""
    assert not screen_boxes(_text(), _params())[0]


def test_blank_page_gives_nothing() -> None:
    """Чистая бумага: пятен нет вовсе, доля точечных ничего не значит."""
    assert not screen_boxes(synthetic.paper(SIZE), _params())[0]


def test_photo_box_lands_on_the_screen_patch() -> None:
    """Найденный прямоугольник совпадает с вклеенным растровым пятном с точностью до клетки.

    Точнее клетки ответа и не бывает: границей области служит граница клетки, а пограничная
    клетка наполовину занята текстом вокруг и растровой не считается. Поэтому проверяется не
    совпадение по пикселю, а что рамка накрывает почти всё пятно и почти не вылезает наружу.
    """
    patch = (200, 200, 800, 900)
    page = synthetic.with_screen(_text(), patch, pitch=4, radius=1)
    boxes = screen_boxes(page, _params())[0]
    assert len(boxes) == 1

    cell = _params().cell_px
    for got, want in zip(boxes[0], patch):
        assert abs(got - want) <= 2 * cell

    x1, y1, x2, y2 = boxes[0]
    found_area = (x2 - x1) * (y2 - y1)
    patch_area = (patch[2] - patch[0]) * (patch[3] - patch[1])
    assert found_area > 0.6 * patch_area


def test_light_patch_stays_one_region() -> None:
    """Светлый полутон (редкие мелкие точки) остаётся ОДНОЙ областью, а не рассыпается.

    Прошлый детектор ловил только яркость 100..225, и светлая фотография распадалась на
    куски: 1969/01 IMG_0030_2R давал четыре, 1970/01 IMG_0018_1L — два.
    """
    page = synthetic.with_screen(synthetic.paper(SIZE), (200, 200, 900, 1000), pitch=5, radius=1)
    assert len(screen_boxes(page, _params())[0]) == 1


@pytest.mark.parametrize("dpi", [300, 450, 600])
def test_thresholds_scale_with_dpi(dpi: int) -> None:
    """Одна и та же ФИЗИЧЕСКАЯ картинка, снятая в разном разрешении, даёт ту же область.

    Это и есть проверка, что пороги пересчитываются: паки бывают 300, 450 и 600 dpi, а все
    константы заданы при 600.
    """
    scale = dpi / REFERENCE_DPI
    size = (round(1800 * scale * 2), round(1200 * scale * 2))
    box = tuple(round(value * scale * 2) for value in (200, 200, 900, 1000))
    page = synthetic.with_screen(
        synthetic.paper(size), box, pitch=max(3, round(8 * scale)), radius=max(1, round(2 * scale))
    )

    boxes = screen_boxes(page, params_for_dpi(dpi))[0]
    assert len(boxes) == 1
    # Промах не больше двух клеток при ЛЮБОМ разрешении — значит пороги пересчитались.
    # В абсолютных пикселях допуск растёт вместе с dpi, в физических долях кадра он один.
    for got, want in zip(boxes[0], box):
        assert abs(got - want) <= 2 * params_for_dpi(dpi).cell_px


def test_is_dot_rejects_long_thin_components() -> None:
    """Пятно площадью с точку, но длинное, точкой не считается: это обрывок штриха."""
    params = _params()
    image = synthetic.paper((200, 200))
    image[100:103, 20:180] = synthetic.INK  # штрих 160x3: площадь мала, сторона велика
    stats, _centroids, _mask = ink_components(image, params)
    assert stats.size and not is_dot(stats, params).all()


def test_dot_fraction_is_high_inside_a_screen() -> None:
    """Доля точечных пятен внутри растрового пятна близка к единице — она же идёт в базу."""
    params = _params()
    page = synthetic.with_screen(synthetic.paper(SIZE), (200, 200, 900, 1000), pitch=4, radius=1)
    stats, centroids, _mask = ink_components(page, params)
    maps = cell_maps(stats, centroids, page.shape[:2], params)
    assert dot_fraction(maps, (200, 200, 900, 1000)) > 0.9


def test_cell_maps_survive_a_page_without_ink() -> None:
    """Пустой кадр не должен ронять расчёт карт."""
    params = _params()
    maps = cell_maps(np.empty((0, 5), np.int32), np.empty((0, 2)), SIZE, params)
    assert maps.dots.sum() == 0 and maps.total.sum() == 0
