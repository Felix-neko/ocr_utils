"""Разрез ряда по набору «кегль + жирность» (``blocks.RowStyleSplit``) и продолжение висячего списка в правиле подписи (``blocks.HangingList``)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ocr_utils.page_layout.text_blocks import blocks
from ocr_utils.page_layout.text_blocks.blocks import HangingList, Row, RowStyleSplit, _split_by_measure, _weak_overlap

DPI = 150.0
# Рендер вдвое крупнее рабочей копии (300 dpi против 150).
K = 2.0


def _glyph_axis(x0: float, x1: float, cy: float, kegl: float, pitch: float) -> SimpleNamespace:
    """Заглушка оси: буквы-боксы высотой ``kegl`` с шагом ``pitch`` от ``x0`` до ``x1`` (px рабочей копии)."""
    starts = np.arange(x0, x1 - pitch / 2, pitch)
    glyphs = np.stack(
        [starts, np.full_like(starts, cy - kegl / 2), starts + 0.7 * pitch, np.full_like(starts, cy + kegl / 2)], 1
    )
    return SimpleNamespace(x0=x0, x1=x1, cy=cy, height=kegl * 1.3, glyphs=glyphs, dpi=DPI)


def _draw(ink: np.ndarray, axis: SimpleNamespace, stroke: int) -> None:
    """Нарисовать на рендере буквы оси двумя вертикальными штрихами толщиной ``stroke`` px рендера."""
    for gx0, gy0, gx1, gy1 in axis.glyphs:
        y0, y1 = int(gy0 * K), int(gy1 * K)
        for x in (int(gx0 * K), int(gx1 * K) - stroke):
            ink[y0:y1, x : x + stroke] = 255


def _caption_and_headline() -> tuple[list[SimpleNamespace], np.ndarray]:
    """Ряд «подпись | заголовок» как 1969/05 IMG_0072_2R: кегль 14 светлый и 19 жирный через 63 px пустоты."""
    caption = _glyph_axis(90.0, 262.0, 128.0, 14.0, 12.0)
    headline = _glyph_axis(325.0, 732.0, 128.0, 19.0, 16.0)
    ink = np.zeros((400, 1600), dtype=np.uint8)
    _draw(ink, caption, 4)
    _draw(ink, headline, 10)
    return [caption, headline], ink


def test_style_split_cuts_light_caption_from_bold_headline(monkeypatch):
    """Кегль различается в 1.36 раза (ниже ``ROW_GLYPH_RATIO``), но жирность — вдвое: режет только ``STYLE``."""
    group, ink = _caption_and_headline()
    assert len(_split_by_measure(group, ink, K, DPI, by_boxes=True, style=False)) == 1
    assert len(_split_by_measure(group, ink, K, DPI, by_boxes=True, style=True)) == 2


def test_style_split_keeps_same_weight_pieces():
    """Те же кегли одной жирности: балл набора ниже порога, ряд не режется."""
    group, _ = _caption_and_headline()
    ink = np.zeros((400, 1600), dtype=np.uint8)
    for axis in group:
        _draw(ink, axis, 6)
    assert len(_split_by_measure(group, ink, K, DPI, by_boxes=True, style=True)) == 1


def test_default_row_style_split_is_known_mode():
    """Переключатель по умолчанию — один из способов :class:`RowStyleSplit`."""
    assert isinstance(blocks.DEFAULT_ROW_STYLE_SPLIT, RowStyleSplit)


def _row(y: float, x0: float, x1: float, axes: tuple) -> Row:
    """Ряд с краями и осями (остальные поля правилу подписи не нужны)."""
    return Row(y=y, height=16.0, x0=x0, x1=x1, axes=axes, glyph_h=11.0, glyph_w=9.0, weight=1.9)


def _label_row() -> Row:
    """«Q — объем поставки…»: метка 105–119, тире в маску глифов не вошло, текст с 157 (1975/02 IMG_0066_1L)."""
    label = SimpleNamespace(x0=104.0, x1=578.0, dpi=DPI, glyphs=np.array([[105.0, 1040.0, 119.0, 1061.0]]))
    text = _glyph_axis(157.0, 578.0, 1048.0, 11.0, 11.0)
    axis = SimpleNamespace(x0=104.0, x1=578.0, dpi=DPI, glyphs=np.concatenate([label.glyphs, text.glyphs]))
    return _row(1048.0, 104.0, 578.0, (axis,))


def test_hanging_list_continuation_stays_in_block(monkeypatch):
    """Строка «поставки, руб.;» под текстом после метки «Q —»: прежнее правило режет, ``ALIGN`` — нет."""
    long_row = _label_row()
    short = _row(1025.0, 156.0, 299.0, (_glyph_axis(156.0, 299.0, 1025.0, 11.0, 11.0),))
    monkeypatch.setattr(blocks, "DEFAULT_HANGING_LIST", HangingList.OFF)
    assert _weak_overlap(short, long_row, 726.0)
    monkeypatch.setattr(blocks, "DEFAULT_HANGING_LIST", HangingList.ALIGN)
    assert not _weak_overlap(short, long_row, 726.0)


def test_signature_off_hanging_text_is_still_split(monkeypatch):
    """Короткий ряд, начатый не под словом за меткой (подпись правее), режется и при ``ALIGN``."""
    long_row = _label_row()
    short = _row(1025.0, 380.0, 500.0, (_glyph_axis(380.0, 500.0, 1025.0, 11.0, 11.0),))
    monkeypatch.setattr(blocks, "DEFAULT_HANGING_LIST", HangingList.ALIGN)
    assert _weak_overlap(short, long_row, 726.0)
