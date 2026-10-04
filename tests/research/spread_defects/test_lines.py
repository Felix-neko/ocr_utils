"""Тесты сборки строк у сгиба: строки находятся, концевая буква — у сгиба, опорные — дальше."""

import cv2
import numpy as np

from research.spread_defects.lines import LEFT, RIGHT, find_glyphs, side_lines


def _spread() -> tuple[np.ndarray, np.ndarray]:
    """Разворот 2400×4000: строки слов на обеих полосах, сгиб вертикален посередине (x = 2000)."""
    rng = np.random.default_rng(3)
    gray = np.full((2400, 4000), 235, np.uint8)
    for y in range(300, 2100, 62):
        for x_start, x_end in ((300, 1960), (2040, 3700)):
            x = x_start
            while x < x_end - 120:
                word = "".join(rng.choice(list("abcdefghknopqrsuvxz"), rng.integers(3, 8)))
                cv2.putText(gray, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 20, 3, cv2.LINE_AA)
                x += 28 * len(word) + 30
    return gray.astype(np.float32), np.array([0.0, 2000.0])


def test_lines_found_on_both_sides():
    """На каждой полосе — строки набора, концевые буквы ближе опорных к сгибу."""
    gray, fold = _spread()
    glyphs = find_glyphs(gray)
    for side in (LEFT, RIGHT):
        lines = side_lines(glyphs, fold, side)
        assert len(lines) >= 20
        line = lines[len(lines) // 2]
        ends_x = glyphs.cx[line.ends]
        refs_x = glyphs.cx[line.refs]
        assert np.all(np.abs(ends_x - 2000).max() < np.abs(refs_x - 2000).min())
