"""Тесты признака смаза: двоение в зоне синтетической страницы даёт пик кепстра на векторе сдвига, резкая — нет."""

import cv2
import numpy as np

from research.spread_defects.motion import TILE, motion_score, motion_tiles


def _page(rows: int = 9, cols: int = 9, seed: int = 0) -> np.ndarray:
    """Синтетическая полоса текста: кремовая бумага, строки случайных латинских слов чёрным, ~450 dpi."""
    rng = np.random.default_rng(seed)
    height, width = rows * TILE, cols * TILE
    page = np.full((height, width, 3), (238, 232, 215), np.uint8)
    letters = "abcdefghijklmnopqrstuvwxyz"
    for y in range(60, height - 20, 62):
        x = 10
        while x < width - 80:
            word = "".join(rng.choice(list(letters), rng.integers(3, 9)))
            cv2.putText(page, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3, cv2.LINE_AA)
            x += 30 * len(word) + 30
    return page


def _double(image: np.ndarray, shift: tuple[int, int], weight: float = 0.45) -> np.ndarray:
    """Двойная экспозиция: изображение и его копия со сдвигом ``shift`` (dx, dy) с весом ``weight``."""
    moved = np.roll(image, shift=(shift[1], shift[0]), axis=(0, 1))
    return ((1 - weight) * image.astype(np.float32) + weight * moved.astype(np.float32)).astype(np.uint8)


def test_sharp_page_has_low_score():
    """Резкая страница: зональная оценка низкая."""
    score = motion_score(motion_tiles(_page()))
    assert score.interior > 10
    assert score.agree_peak < 7


def test_zonal_double_image_found_with_shift():
    """Двоение (5, 0) в правой половине: высокая оценка, пик в правой половине, сдвиг (5, 0)."""
    page = _page()
    half = page.shape[1] // 2
    page[:, half:] = _double(page[:, half:], (5, 0))
    score = motion_score(motion_tiles(page))
    assert score.agree_peak > 9
    assert score.col * TILE >= half - TILE
    assert (abs(score.shift_dx), abs(score.shift_dy)) == (5, 0)
