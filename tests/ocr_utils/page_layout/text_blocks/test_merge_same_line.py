"""Тесты балла набора кусков одной строки (``regroup._line_style_distance``): короткий хвост без замеренных глифов («143 с.—») сверяется со строкой только по жирности, а длинный кусок — по-прежнему и по кеглю."""

from ocr_utils.page_layout.text_blocks.blocks import Row
from ocr_utils.page_layout.text_blocks.regroup import _line_style_distance


def row(x0: float, x1: float, glyph_h: float, weight: float, height: float = 11.0) -> Row:
    """Кусок строки на высоте 100 px рабочей копии 150 dpi (5.9 px/мм)."""
    return Row(y=100.0, height=height, x0=x0, x1=x1, axes=(), glyph_h=glyph_h, glyph_w=5.0 if glyph_h else 0.0,
               weight=weight)  # fmt: skip


def test_short_unmeasured_tail_joins_its_line():
    # ХиП 1991/01 IMG_1030 R: строка 5.9–99.7 мм (кегль 1.44 мм ≈ 8.5 px), хвост «с.—» 101.9–106.3 мм без глифов,
    # высота ряда 1.52 мм (9 px): запасной кегль давал балл 0.53 ≥ 0.5, и хвост оставался отдельным рядом.
    line = row(35.0, 588.0, glyph_h=8.5, weight=2.16)
    tail = row(601.0, 627.0, glyph_h=0.0, weight=2.34, height=9.0)
    assert _line_style_distance(line, tail) < 0.5


def test_tail_of_other_weight_stays_apart():
    # Жирность у хвоста вдвое больше — это другой набор, хвост не сливается.
    line = row(35.0, 588.0, glyph_h=8.5, weight=2.0)
    tail = row(601.0, 627.0, glyph_h=0.0, weight=4.0, height=9.0)
    assert _line_style_distance(line, tail) >= 0.5


def test_long_piece_keeps_size_check():
    # Длинный кусок без глифов (слипшийся жирный заголовок) — прежний балл по запасному кеглю.
    line = row(35.0, 300.0, glyph_h=8.5, weight=2.0)
    heading = row(310.0, 560.0, glyph_h=0.0, weight=2.0, height=30.0)
    assert _line_style_distance(line, heading) >= 0.5
