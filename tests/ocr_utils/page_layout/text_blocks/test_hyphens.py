"""Дефисы в краске текста (``text_blocks.hyphens``): низкая широкая чёрточка у буквы — дефис, точка и пыль — нет."""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.orientation.detectors.ink_axis import glyph_mask
from ocr_utils.page_layout.text_blocks.hyphens import hyphens_mask

DPI = 150.0


def _page() -> np.ndarray:
    """Рабочая копия 150 dpi: «слово» 8 px высотой, за ним дефис, точка, одинокая чёрточка вдали."""
    work = np.full((60, 200), 255, dtype=np.uint8)
    work[20:28, 20:60] = 0  # слово: буквы высотой 8 px (≈ 1.4 мм, строчная корпуса)
    work[23:26, 61:66] = 0  # дефис 5 × 3 px вплотную справа, на уровне середины слова
    work[25:28, 90:93] = 0  # точка 3 × 3 px — не шире своей высоты
    work[20:28, 100:140] = 0  # второе слово
    work[40:43, 170:175] = 0  # чёрточка 5 × 3 px без буквы рядом — пыль или обрывок черты
    return work


def test_hyphen_next_to_a_letter_is_found():
    """Дефис у буквы попадает в маску; маска глифов его не берёт (он ниже 5 px)."""
    work = _page()
    glyphs = glyph_mask(work)
    assert not glyphs[23:26, 61:66].any()
    found = hyphens_mask(work, glyphs, DPI)
    assert found[23:26, 61:66].all()


def test_dot_and_lonely_dash_are_not_hyphens():
    """Точка не шире своей высоты, чёрточка вдали от букв — не дефисы; сами буквы тоже не дефисы."""
    work = _page()
    found = hyphens_mask(work, glyph_mask(work), DPI)
    assert not found[25:28, 90:93].any()
    assert not found[40:43, 170:175].any()
    assert not found[20:28, 20:60].any() and not found[20:28, 100:140].any()
    assert int((found > 0).sum()) == 15  # только дефис 5 × 3


def test_hyphen_with_a_faint_rim_is_found():
    """Дефис с прилипшим бледным краем: бокс 4 × 4, но средняя толщина 2.5 px — всё равно дефис.

    Так выглядел «организа-» на 1973/07 с.88: по отношению сторон БОКСА (1.0) он не проходил.
    """
    work = np.full((60, 120), 255, dtype=np.uint8)
    work[20:28, 20:60] = 0  # слово
    work[23:25, 62:66] = 0  # тело дефиса 4 × 2
    work[22, 63] = 0  # прилипшие пиксели бледного края сверху и снизу: бокс 4 × 4, площадь 10,
    work[25, 64] = 0  # заливка 0.62 — как у «организа-» на 1973/07 с.88
    found = hyphens_mask(work, glyph_mask(work), DPI)
    assert found[23:25, 62:66].all()
