"""Обложка выпуска: первая полоса и запертая на неё эвристика сплошной плашки."""

from ocr_utils.scan_markup.detection.cover import (
    cover_decision,
    cover_region,
    ink_components,
    is_cover_page,
    lineart_ink_frac,
)
from tests.ocr_utils.scan_markup import synthetic

SIZE = (1400, 900)  # копия 1/4 кадра 600 dpi — масштаб, на котором откалиброван детектор


def _plate():
    return synthetic.solid_plate(SIZE, (50, 100, 850, 500))


def test_ink_components_report_text_fraction() -> None:
    """Доля текстовой краски у полосы текста высокая, у полосы с плашкой — низкая."""
    _boxes, text_page_frac = ink_components(synthetic.text_page(SIZE))
    _boxes, cover_frac = ink_components(_plate())
    assert text_page_frac > 0.05 > cover_frac


def test_is_cover_page_needs_enough_ink() -> None:
    """Пустой шмуцтитул с одной строкой в растр целиком не уходит."""
    assert not is_cover_page([(0, 0, 50, 50)], text_frac=0.001, width=900, height=1400)
    assert is_cover_page([(0, 0, 800, 400)], text_frac=0.001, width=900, height=1400)


def test_cover_decision_only_fires_on_the_first_page() -> None:
    """Внутренняя полоса обложкой не объявляется, даже если выглядит ровно как плашка.

    Это и есть починка дефекта «две картинки детектированы как одна большая»: 1969/11
    IMG_0093_2R (полоса 84 из 99) и 1969/12 IMG_0150_2R (полоса 98 из 99) — две фотографии
    с подписями, и прошлая версия накрывала их одним прямоугольником во весь кадр.
    """
    plate = _plate()
    assert cover_decision(0, plate, first_page_is_cover=False)
    assert not cover_decision(84, plate, first_page_is_cover=False)
    assert not cover_decision(98, plate, first_page_is_cover=True)


def test_first_page_is_cover_does_not_look_at_pixels() -> None:
    """С флагом первая полоса — обложка, каким бы ни было её содержимое."""
    assert cover_decision(0, synthetic.text_page(SIZE), first_page_is_cover=True)
    assert not cover_decision(0, synthetic.text_page(SIZE), first_page_is_cover=False)


def test_cover_region_is_the_whole_frame() -> None:
    """Обложка помечается кадром целиком, а не границей плашки."""
    assert cover_region(3492, 6051) == (0, 0, 3492, 6051)


def test_lineart_ink_frac_separates_a_drawing_from_a_text_page() -> None:
    """Сплошной рисунок покрывает краской заметно больше полосы, чем текст.

    По этому числу отбирается кандидат в «цветной штриховой рисунок во всю полосу»
    (1970/04 IMG_0052_2R — обложка 3, целиком синий рисунок).
    """
    drawing = lineart_ink_frac(synthetic.line_art(SIZE, step=6, thickness=3))
    text = lineart_ink_frac(synthetic.text_page(SIZE))
    assert drawing > text
