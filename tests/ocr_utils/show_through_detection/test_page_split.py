"""Проверка деления кадра-разворота на полосы."""

import numpy as np

from ocr_utils.show_through_detection.page_split import LEFT, RIGHT, WHOLE, find_gutter, split_spread
from tests.ocr_utils.show_through_detection.pages import draw_page, expose, spread


def make_spread(left_width: int = 1000, right_width: int = 1000, gutter: int = 90) -> np.ndarray:
    """Синтетический разворот из двух полос.

    Args:
        left_width: Ширина левой полосы.
        right_width: Ширина правой полосы.
        gutter: Ширина корешкового провала.

    Returns:
        Полутоновый кадр-разворот.
    """
    left = draw_page(1400, left_width, seed=1)
    right = draw_page(1400, right_width, seed=2)
    return expose(spread(left, right, gutter), noise=1.5)


def test_gutter_is_found_at_the_centre() -> None:
    """Симметричный разворот: корешок посередине."""
    page = make_spread()
    found = find_gutter(page)
    assert found.confident
    assert abs(found.gutter - page.shape[1] // 2) < 60


def test_gutter_is_found_when_it_is_off_centre() -> None:
    """Корешок гуляет по кадру от съёмки к съёмке — деление пополам этого не ловит.

    Именно ради этого случая корешок и ищется: при неравных полосах середина кадра
    уезжает в наборную полосу, и половина текста соседней страницы попала бы в замер
    как «своя».
    """
    page = make_spread(left_width=800, right_width=1200)
    found = find_gutter(page)
    assert found.confident
    expected = 800 + 45
    assert abs(found.gutter - expected) < 80, f"корешок найден в {found.gutter}, ожидался около {expected}"
    assert abs(found.gutter - page.shape[1] // 2) > 100, "тест бессмыслен, если корешок и так в центре"


def test_spread_is_split_into_two_halves() -> None:
    """Разворот режется на две полосы, обе непустые."""
    halves = split_spread(make_spread())
    assert [side for side, _ in halves] == [LEFT, RIGHT]
    assert all(half.shape[1] > 300 for _, half in halves)


def test_single_page_is_not_split() -> None:
    """Одиночная страница (обложка, вклейка) — это одна полоса, а не две половинки.

    Разрезав её пополам, мы получили бы две полуполосы с оборванным набором и
    двумя бессмысленными баллами вместо одного осмысленного.
    """
    page = expose(draw_page(1400, 1000), noise=1.5)
    halves = split_spread(page)
    assert [side for side, _ in halves] == [WHOLE]
    assert halves[0][1].shape == page.shape


def test_blank_frame_falls_back_to_the_middle() -> None:
    """Без выраженного корешка деление берётся пополам, а не падает."""
    blank = np.full((1400, 3000), 238, np.uint8)
    found = find_gutter(blank)
    assert not found.confident
    assert found.gutter == 1500
