"""Тесты растяжки уровней: страница с краской растягивается как раньше, пустая бумага с потолком чёрной точки — мягко."""

import numpy as np

from ocr_utils.scan_cropping.levels import LEVELS_MAX_BLACK, compensate_levels


def blank_paper() -> np.ndarray:
    """Пустая пожелтевшая бумага с фактурой: серое 200…245, синий канал ниже остальных."""
    rng = np.random.default_rng(0)
    base = rng.uniform(200, 245, (300, 400)).astype(np.float32)
    return np.stack([base - 15, base, base + 5], axis=2).clip(0, 255).astype(np.uint8)


def test_cap_keeps_blank_paper_mild() -> None:
    """Без потолка пустая бумага растягивается на весь диапазон, с потолком — контраст растёт меньше чем в 1.5 раза."""
    img = blank_paper()
    spread_in = float(np.ptp(img[..., 1].astype(int)))
    loose = compensate_levels(img)
    capped = compensate_levels(img, max_black=LEVELS_MAX_BLACK)
    assert np.ptp(loose[..., 1].astype(int)) > 3 * spread_in
    assert np.ptp(capped[..., 1].astype(int)) < 1.5 * spread_in


def test_cap_does_not_change_page_with_ink() -> None:
    """У страницы с краской чёрная точка ниже потолка — итог тот же, что без потолка."""
    img = blank_paper()
    img[50:70, 20:380] = 25  # строка краски
    img[120:140, 20:380] = 30
    assert np.array_equal(compensate_levels(img), compensate_levels(img, max_black=LEVELS_MAX_BLACK))
