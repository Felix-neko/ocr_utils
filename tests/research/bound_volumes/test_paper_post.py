"""Тесты доводки маски бумаги SAM 3: заливка вогнутостей и включение краски только внутри оболочки страницы."""

import numpy as np

from research.bound_volumes.paper import fill_concavities, page_hulls


def test_shallow_notch_filled_deep_cut_kept() -> None:
    """Неглубокая выемка у края заливается, глубокий вырез (разрыв у корешка) остаётся."""
    mask = np.zeros((400, 600), bool)
    mask[50:350, 50:550] = True
    mask[50:80, 280:320] = False  # выемка глубиной 30 px у верхнего края
    mask[200:350, 290:310] = False  # вырез глубиной 150 px снизу
    out = fill_concavities(mask, depth=40)
    assert out[60, 300]
    assert not out[300, 300]


def test_page_hulls_skip_small_components() -> None:
    """Оболочки строятся только по крупным компонентам; мелкая соринка снаружи страницы оболочки не раздувает."""
    mask = np.zeros((400, 600), bool)
    mask[50:350, 50:300] = True
    mask[5:9, 590:594] = True
    hull = page_hulls(mask, min_area=1000)
    assert hull[200, 200]
    assert not hull[7, 592]
    assert not hull[30, 400]
