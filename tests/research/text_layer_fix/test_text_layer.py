"""Разбор потока содержимого: формы спанов, привязка глифов, дерево структуры."""

from __future__ import annotations

import fitz
import pikepdf
import pytest

from research.text_layer_fix.text_layer import SpanShape, load_layer, multiply, tokenize
from tests.research.text_layer_fix.synthetic import finereader_like_pdf, pymupdf_text_pdf


def test_tokenize_strings_and_dicts() -> None:
    tokens = tokenize(rb"/Span <</MCID 3>> BDC BT (a\)b) Tj <41 42> Tj [(x) -20 (y)] TJ ET EMC")
    kinds = [t.kind for t in tokens]
    assert kinds[:5] == ["name", "dict_open", "name", "number", "dict_close"]
    strings = [t.value for t in tokens if t.kind == "string"]
    assert strings == [b"a)b", b"AB", b"x", b"y"]
    assert [t.value for t in tokens if t.kind == "op"] == ["BDC", "BT", "Tj", "Tj", "TJ", "ET", "EMC"]


def test_multiply_matches_pdf_order() -> None:
    translate = (1.0, 0.0, 0.0, 1.0, 10.0, 20.0)
    scale = (2.0, 0.0, 0.0, 3.0, 0.0, 0.0)
    assert multiply(translate, scale) == (2.0, 0.0, 0.0, 3.0, 20.0, 60.0)


def test_finereader_like_stream(tmp_path) -> None:
    path, _ = finereader_like_pdf(tmp_path / "fr.pdf")
    doc = fitz.open(str(path))
    layer = load_layer(doc[0], pikepdf.open(str(path)))
    words = {w.mcid: w for w in layer.words if w.mcid is not None}
    assert set(words) == {0, 1, 2}
    assert layer.unmatched_glyphs == 0
    assert words[0].shape == SpanShape.PLAIN and words[0].text == "Таблица"
    assert words[1].shape == SpanShape.TD_ONLY and words[1].text == "и"
    assert words[2].shape == SpanShape.ROTATED and words[2].text == "Резервы"
    # Повёрнутое слово читается снизу вверх: направление (0, -1) в координатах fitz.
    dx, dy = words[2].direction
    assert abs(dx) < 1e-6 and dy == pytest.approx(-1.0)
    assert words[0].stretch == pytest.approx(1.2)
    assert words[0].render_mode == 3 and words[1].render_mode == 3
    # Сдвиг cm внутри первого спана: диапазон текста уже спана, спан начинается с /Span.
    start, end = words[0].text_ranges[0]
    raw = doc[0].read_contents()
    assert raw[start : start + 2] == b"BT" and raw[end - 2 : end] == b"ET"
    assert raw[words[0].span_range[0] :].startswith(b"/Span")
    assert b"cm" in raw[words[0].span_range[0] : start]
    # Рамка слова — из символов rawdict, лежит вокруг начала первого глифа.
    assert words[0].bbox.contains(fitz.Point(101, 600 - 500 - 3 - 1))


def test_pymupdf_text_uses_font_widths(tmp_path) -> None:
    path = pymupdf_text_pdf(tmp_path / "mu.pdf")
    doc = fitz.open(str(path))
    layer = load_layer(doc[0], pikepdf.open(str(path)))
    texts = sorted(w.text for w in layer.words if w.text.strip())
    assert layer.unmatched_glyphs == 0, "ширины из /W и /Widths должны совпасть с rawdict"
    assert "Снабжение 1966" in texts and "Hello world" in texts
    assert all(w.mcid is None for w in layer.words)
