"""Вердикты по словам: удалить, усечь, оставить, подозрительное."""

from __future__ import annotations

import fitz

from ocr_utils.scan_markup.table_detection.geometry import Box

from ocr_utils.text_layer_fix.classify import Verdict, classify_words, looks_like_junk
from ocr_utils.text_layer_fix.text_layer import Glyph, SpanShape, TextLayer, Word
from ocr_utils.text_layer_fix.zones import RotatedZone, ZoneKind


def _word(mcid: int, text: str, x0: float, y0: float, width: float, shape: SpanShape = SpanShape.PLAIN) -> Word:
    """Слово из глифов по 10 pt шириной, рамки в pt (страница 1 pt = 1 px при dpi 72)."""
    step = width / max(len(text), 1)
    glyphs = [
        Glyph(
            i,
            bytes([65 + i]),
            (x0 + i * step, y0),
            (x0 + i * step, y0),
            step,
            ch,
            fitz.Rect(x0 + i * step, y0 - 8, x0 + (i + 1) * step, y0 + 2),
        )
        for i, ch in enumerate(text)
    ]
    return Word(mcid, shape, "F0", 8.0, (1.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0, 0.0, 0.0), glyphs)


def test_verdicts_by_geometry() -> None:
    words = [
        _word(0, "XX", 105, 110, 20),  # целиком в зоне
        _word(1, "Итого", 10, 110, 40),  # далеко
        _word(2, "стоим", 80, 110, 60),  # задето: глифы 2 и 3 в зоне, 0, 1 и 4 — вне
        _word(3, "^", 96, 140, 8),  # россыпь рядом с зоной (2 px под ней)
        _word(4, "Резервы", 160, 110, 30, SpanShape.ROTATED),  # FineReader написал повёрнутым
    ]
    layer = TextLayer(0, fitz.Rect(0, 0, 300, 300), words)
    zones = [
        RotatedZone(Box(100, 100, 130, 130), ZoneKind.TABLE_CELL, 90, 1.0),
        RotatedZone(Box(150, 100, 200, 130), ZoneKind.LINE_ART_LABEL, 90, 1.0),
    ]
    verdicts = {v.mcid: v for v in classify_words(layer, zones, fitz.Matrix(1, 1), 72.0)}
    assert verdicts[0].verdict == Verdict.DELETE
    assert verdicts[1].verdict == Verdict.KEEP
    assert verdicts[2].verdict == Verdict.SANITIZE and verdicts[2].keep_glyphs == (0, 1, 4)
    assert verdicts[3].verdict == Verdict.SUSPECT and verdicts[3].dist_mm < 4
    assert verdicts[4].verdict == Verdict.KEEP_ROTATED and verdicts[4].zone_index == 1


def test_inactive_zone_keeps_words() -> None:
    words = [_word(0, "шт.", 105, 110, 20)]
    layer = TextLayer(0, fitz.Rect(0, 0, 300, 300), words)
    zone = RotatedZone(Box(100, 100, 130, 130), ZoneKind.TABLE_CELL, 0, 0.0, note="снята")
    verdicts = classify_words(layer, [zone], fitz.Matrix(1, 1), 72.0)
    assert verdicts[0].verdict == Verdict.KEEP


def test_junk_heuristic() -> None:
    assert looks_like_junk("^") and looks_like_junk("X") and looks_like_junk('i"o=')
    assert not looks_like_junk("1966") and not looks_like_junk("Итого") and not looks_like_junk("т/сут")


def test_unreadable_zone_keeps_genuine_words() -> None:
    words = [_word(0, "Наименование", 102, 110, 26), _word(1, "СЪСЧчГ", 102, 120, 26), _word(2, "1966", 102, 128, 20)]
    layer = TextLayer(0, fitz.Rect(0, 0, 300, 300), words)
    zone = RotatedZone(Box(100, 100, 130, 132), ZoneKind.TABLE_CELL, 90, 1.0)
    known = lambda text: text == "наименование"  # noqa: E731 — заглушка словаря
    verdicts = {v.mcid: v for v in classify_words(layer, [zone], fitz.Matrix(1, 1), 72.0, readable=set(), known=known)}
    assert verdicts[0].verdict == Verdict.SUSPECT and verdicts[2].verdict == Verdict.SUSPECT
    assert verdicts[1].verdict == Verdict.DELETE
    # Прочитанная зона: удаляется всё, включая настоящее слово (его заменит вставка).
    verdicts = {v.mcid: v for v in classify_words(layer, [zone], fitz.Matrix(1, 1), 72.0, readable={0}, known=known)}
    assert all(v.verdict == Verdict.DELETE for v in verdicts.values())
