"""Проверка разбора полосы на зоны замера."""

import numpy as np
import pytest

from ocr_utils.show_through_detection.zones import NO_MARGIN, NO_TEXT, build_zones, scaled, scaled_float
from tests.ocr_utils.show_through_detection.pages import MARGIN_X, draw_page, expose, scan

NOISE = 1.5


def test_margins_and_text_block_do_not_overlap() -> None:
    """Опорные поля и зона замера обязаны быть непересекающимися множествами.

    Иначе основная метрика делила бы величину саму на себя и на любой полосе давала
    бы единицу.
    """
    zones = build_zones(expose(scan(draw_page()), noise=NOISE))
    assert not (zones.gap & zones.margin).any()


def test_margin_is_found_on_the_sides_of_the_text_block() -> None:
    """Поля должны находиться там, где они нарисованы, — по краям полосы."""
    page = expose(scan(draw_page()), noise=NOISE)
    zones = build_zones(page)
    width = page.shape[1]
    left_edge = zones.margin[:, : int(width * MARGIN_X / 2)].sum()
    centre = zones.margin[:, int(width * 0.4) : int(width * 0.6)].sum()
    assert left_edge > 0, "на левом поле не нашлось опорных пикселей"
    assert left_edge > centre, "поля найдены в середине наборной полосы, а не по краям"


def test_gap_mask_is_not_empty_at_realistic_line_spacing() -> None:
    """Межстрочья не должны съедаться раздутием краски при обычном интерлиньяже.

    Ровно на этом спотыкается работа в уменьшенном разрешении: там раздутие краски
    закрывает промежуток целиком, и метрика начинает мерить поля вместо межстрочий.
    """
    zones = build_zones(expose(scan(draw_page(line_height=26, stroke=3)), noise=NOISE))
    assert zones.gap.sum() > 0.01 * zones.gap.size


def test_blank_page_is_reported_as_having_no_text() -> None:
    """Вырожденная полоса не роняет разбор, а объясняет себя."""
    zones = build_zones(np.full((1400, 1000), 238, np.uint8))
    assert zones.problem == NO_TEXT
    assert not zones.usable


def test_page_without_margins_is_measurable_but_noted() -> None:
    """Полоса под обрез измерима, но без опоры — и это должно быть сказано, а не скрыто.

    Такие полосы (таблица во всю ширину, набор под обрез) нельзя молча выбрасывать:
    просвет на них требует пересъёмки ровно так же, просто считать его будет запасная
    метрика.
    """
    page = draw_page()
    # Обрезаем поля: остаётся один набор.
    x0, x1 = int(page.shape[1] * MARGIN_X), int(page.shape[1] * (1 - MARGIN_X))
    y0, y1 = int(page.shape[0] * 0.09), int(page.shape[0] * 0.91)
    zones = build_zones(expose(scan(page)[y0:y1, x0:x1], noise=NOISE))
    assert zones.usable, "полоса под обрез измерима"
    assert zones.note == NO_MARGIN
    assert not zones.has_margin


def test_otsu_level_matches_real_scans() -> None:
    """Порог Оцу синтетики должен попадать туда же, где он на настоящих сканах.

    Замерено по паку: 0.65–0.66 в долях уровня бумаги. Если синтетика уедет от этого
    значения, ``ghost_ink`` на ней проверять бессмысленно — она моделирует именно порог.
    """
    zones = build_zones(expose(scan(draw_page()), noise=NOISE))
    assert 0.55 < zones.otsu < 0.75, f"порог Оцу синтетики уехал: {zones.otsu:.3f}"


def test_paper_level_is_about_one_after_normalisation() -> None:
    """Нормировка обязана приводить бумагу к единице при любой экспозиции.

    Это и есть ответ на «в разных выпусках разный цвет бумаги и освещение»: дальше
    по конвейеру никто про яркость уже не думает.
    """
    for gain, offset in ((1.0, 0.0), (0.7, 30.0), (1.15, -20.0)):
        zones = build_zones(expose(scan(draw_page()), gain, offset, NOISE))
        assert 0.95 < zones.paper <= 1.0, f"бумага уехала при gain={gain}, offset={offset}: {zones.paper}"


@pytest.mark.parametrize("height,expected", [(2800, 5), (1400, 2), (5600, 10)])
def test_sizes_scale_with_frame_height(height: int, expected: int) -> None:
    """Размеры, названные для эталонной высоты, обязаны пересчитываться под кадр."""
    assert scaled(height, 5) == expected
    assert scaled_float(height, 5) == pytest.approx(5 * height / 2800.0)


def test_sizes_never_collapse_to_zero() -> None:
    """На крошечном кадре ядро морфологии всё равно должно остаться осмысленным."""
    assert scaled(100, 1) >= 1
