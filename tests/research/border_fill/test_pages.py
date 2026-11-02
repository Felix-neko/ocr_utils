"""Тесты страниц второго преобразования в стенде закраса: маски краски опоры и закрас вариантов без краски."""

import numpy as np

from research.border_fill.pages import PageVariant, fill_page, sauvola_ink


def page_with_text_at_edge() -> tuple[np.ndarray, np.ndarray]:
    """Страница 400×300 px (бумага 230) с чёрными штрихами у правого края области; правые 40 px — вне области."""
    image = np.full((400, 300, 3), 230, np.uint8)
    for y in range(40, 360, 30):
        image[y : y + 8, 240:258] = 20  # «буквы» вплотную к краю области
    image[:, 260:] = 120  # вне области: тёмный фон
    paper = np.zeros((400, 300), bool)
    paper[:, :260] = True
    return image, paper


def test_sauvola_ink_marks_strokes_not_paper():
    image, _ = page_with_text_at_edge()
    ink = sauvola_ink(image, 4.0)
    assert ink[44, 248] and not ink[200, 100]


def test_variants_fill_with_paper_not_ink():
    image, paper = page_with_text_at_edge()
    for variant in (PageVariant.BASE, PageVariant.SAUVOLA, PageVariant.DARK, PageVariant.STRICT):
        out = fill_page(image, paper, 13.8, variant)
        # Закраска вне области — цвет бумаги, а не штрихов и не фона.
        assert np.abs(out[:, 280:].astype(float) - 230).mean() < 15, variant
        # Область страницы не тронута.
        assert (out[paper] == image[paper]).all()
