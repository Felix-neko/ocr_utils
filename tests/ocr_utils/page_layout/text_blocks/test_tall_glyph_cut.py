"""Резка высоких глифов (``segment._cut_tall_glyphs``) по способам ``TallGlyphCut``: сросшиеся буквы двух строк режутся, целая высокая буква в строке — только по высоте (``HEIGHT``), но не по геометрии двух рядов."""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.text_blocks import segment
from ocr_utils.page_layout.text_blocks.glyph_merge import merged_components
from ocr_utils.page_layout.text_blocks.segment import SCALES, TallGlyphCut, _cut_tall_glyphs

# Буква-прямоугольник синтетики: высота, ширина и просвет между буквами ряда (px рабочей копии 150 dpi).
LETTER_H, LETTER_W, LETTER_GAP = 12, 8, 3
# Шаг строк (px): около 1.9 высоты буквы, как у корпуса пака-1.
ROW_STEP = 23


def _row(mask: np.ndarray, top: int, x0: int, x1: int) -> None:
    """Нарисовать ряд букв-прямоугольников от ``x0`` до ``x1`` с верхом ``top``."""
    for x in range(x0, x1 - LETTER_W, LETTER_W + LETTER_GAP):
        mask[top : top + LETTER_H, x : x + LETTER_W] = 255


def _two_rows_bridged() -> np.ndarray:
    """Два ряда букв с шагом ``ROW_STEP``; буква верхнего ряда тонкой перемычкой срослась с буквой нижнего."""
    mask = np.zeros((140, 400), dtype=np.uint8)
    for top in (20, 20 + ROW_STEP):
        _row(mask, top, 10, 390)
    # Буквы ряда стоят с шагом 11 px с x = 10: буква x 153..161 → перемычка в её середине.
    mask[20 + LETTER_H : 20 + ROW_STEP, 153:155] = 255
    return mask


def _word_with_descender() -> np.ndarray:
    """Один ряд букв; одна буква («Д» с выносными) вдвое выше: тело 12 px, тонкая шейка, низ 10 px."""
    mask = np.zeros((100, 400), dtype=np.uint8)
    _row(mask, 30, 10, 390)
    x = 153
    mask[30 + LETTER_H : 30 + LETTER_H + 2, x + 3 : x + 5] = 255  # шейка 2 px — перемычка
    mask[30 + LETTER_H + 2 : 30 + LETTER_H + 12, x : x + LETTER_W] = 255  # выносной низ
    return mask


def _parts(mask: np.ndarray) -> int:
    """Сколько букв (склеенных компонент) в маске."""
    return merged_components(mask)[0]


def test_two_row_glyph_is_cut_by_every_cutting_mode(monkeypatch):
    """Перемычка между буквами двух строк режется и по высоте, и по геометрии двух рядов."""
    mask = _two_rows_bridged()
    for mode in (TallGlyphCut.HEIGHT, TallGlyphCut.TWO_LEVELS):
        monkeypatch.setattr(segment, "DEFAULT_TALL_GLYPH_CUT", mode)
        assert _parts(_cut_tall_glyphs(mask, SCALES[0])) == _parts(mask) + 1


def test_descender_letter_is_cut_only_by_height(monkeypatch):
    """Высокая буква в одной строке: по высоте режется (прежняя ошибка), по геометрии двух рядов — нет."""
    mask = _word_with_descender()
    monkeypatch.setattr(segment, "DEFAULT_TALL_GLYPH_CUT", TallGlyphCut.HEIGHT)
    assert not np.array_equal(_cut_tall_glyphs(mask, SCALES[0]), mask)
    monkeypatch.setattr(segment, "DEFAULT_TALL_GLYPH_CUT", TallGlyphCut.TWO_LEVELS)
    assert np.array_equal(_cut_tall_glyphs(mask, SCALES[0]), mask)


def test_descender_letter_above_next_row_is_not_cut(monkeypatch):
    """Высокая буква над следующей строкой (не касаясь её): соседи нижней половинки — на уровне самой буквы."""
    mask = np.zeros((140, 400), dtype=np.uint8)
    _row(mask, 20, 10, 390)
    _row(mask, 20 + ROW_STEP + 8, 10, 390)
    x = 153
    mask[20 + LETTER_H : 20 + LETTER_H + 2, x + 3 : x + 5] = 255
    mask[20 + LETTER_H + 2 : 20 + LETTER_H + 12, x : x + LETTER_W] = 255
    monkeypatch.setattr(segment, "DEFAULT_TALL_GLYPH_CUT", TallGlyphCut.TWO_LEVELS)
    assert np.array_equal(_cut_tall_glyphs(mask, SCALES[0]), mask)


def test_none_mode_never_cuts(monkeypatch):
    """Способ ``NONE`` не режет ничего, даже настоящую перемычку двух строк."""
    monkeypatch.setattr(segment, "DEFAULT_TALL_GLYPH_CUT", TallGlyphCut.NONE)
    mask = _two_rows_bridged()
    assert np.array_equal(_cut_tall_glyphs(mask, SCALES[0]), mask)
