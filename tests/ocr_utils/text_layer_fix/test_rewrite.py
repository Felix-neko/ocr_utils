"""Правка слоя: удаление и усечение слов, вставка повёрнутого текста, сверка."""

from __future__ import annotations

import fitz
import pikepdf
import pytest

from ocr_utils.text_layer_fix.raster import page_raster
from ocr_utils.text_layer_fix.rewrite import (
    Insert,
    InsertFont,
    apply_edits,
    fit_fontsize,
    rebuild_text_object,
    verify_page,
)
from ocr_utils.text_layer_fix.text_layer import load_layer, page_content
from tests.ocr_utils.text_layer_fix.synthetic import finereader_like_pdf


@pytest.fixture(scope="module")
def font() -> InsertFont:
    return InsertFont()


def _line_dirs(page: fitz.Page) -> dict[str, tuple[float, float]]:
    result = {}
    for block in page.get_text("dict", flags=0)["blocks"]:
        for line in block.get("lines", []):
            result["".join(s["text"] for s in line["spans"])] = tuple(round(v, 3) for v in line["dir"])
    return result


def test_blank_trim_insert_roundtrip(tmp_path, font) -> None:
    src, _ = finereader_like_pdf(tmp_path / "fr.pdf")
    doc0 = fitz.open(str(src))
    layer = load_layer(doc0[0], pikepdf.open(str(src)))
    words = {w.mcid: w for w in layer.words if w.mcid is not None}
    image = page_raster(doc0[0])
    raw = page_content(doc0[0])
    inserts = [
        Insert("Снизу вверх", fitz.Rect(300, 100, 312, 300), 90),
        Insert("Сверху вниз", fitz.Rect(320, 100, 332, 300), 270),
        Insert("Кверху ногами", fitz.Rect(100, 350, 250, 362), 180),
        Insert("Прямо\nвторая", fitz.Rect(100, 380, 250, 410), 0),
    ]
    doc = fitz.open(str(src))
    stats = apply_edits(doc, 0, raw, [words[0]], {id(words[2]): (words[2], (0, 1, 2))}, inserts, font)
    out = tmp_path / "out.pdf"
    doc.save(str(out))
    assert (stats.blanked, stats.trimmed, stats.inserted, stats.inserted_lines) == (1, 1, 4, 5)

    doc1 = fitz.open(str(out))
    page = doc1[0]
    assert len(page.get_contents()) == 1, "правки и вставки — в тот же единственный поток"
    # Слова, которых правки не касались, остались посимвольно на месте (проверка на сдвиг смещений).
    untouched = [w for w in words.values() if w.mcid == 1]
    after = load_layer(page, pikepdf.open(str(out)))
    for word in untouched:
        same = after.word_by_mcid(word.mcid)
        assert same is not None and same.text == word.text
        assert [g.origin for g in same.glyphs] == pytest.approx([g.origin for g in word.glyphs], abs=0.05)
    texts = {w[4] for w in page.get_text("words")}
    assert "Таблица" not in texts and "Рез" in texts and "и" in texts
    dirs = _line_dirs(page)
    assert dirs["Снизу вверх"] == (0.0, -1.0)
    assert dirs["Сверху вниз"] == (0.0, 1.0)
    assert dirs["Кверху ногами"] == (-1.0, 0.0)
    assert dirs["Прямо"] == (1.0, 0.0) and dirs["вторая"] == (1.0, 0.0)
    for insert in inserts:
        first = insert.text.split("\n")[0]
        assert any(hit.intersects(insert.rect) for hit in page.search_for(first)), first
    report = verify_page(doc0[0], page, [words[1]], [words[0]], inserts, image.main_xref)
    assert report.ok, report
    # Спан удалённого слова остался (пустой), а дерева структуры в синтетике нет — MCID на месте.
    assert b"/Span <</MCID 0>> BDC" in page.read_contents()
    # Второй вызов на той же странице не дублирует шрифт.
    assert sum(1 for f in page.get_fonts(full=True) if f[4] == "TLFnoto") == 1


def test_rebuild_keeps_positions(tmp_path) -> None:
    src, _ = finereader_like_pdf(tmp_path / "fr.pdf", with_image=False)
    doc0 = fitz.open(str(src))
    layer = load_layer(doc0[0], pikepdf.open(str(src)))
    word = layer.word_by_mcid(0)
    rebuilt = rebuild_text_object(word, (2, 3))
    assert rebuilt.startswith(b"BT") and rebuilt.endswith(b"ET") and rebuilt.count(b"Tj") == 2
    assert rebuild_text_object(word, ()) == b""
    doc = fitz.open(str(src))
    apply_edits(doc, 0, page_content(doc0[0]), [], {id(word): (word, (2, 3))}, [], None)
    out = tmp_path / "trim.pdf"
    doc.save(str(out))
    after = load_layer(fitz.open(str(out))[0], pikepdf.open(str(out)))
    kept = after.word_by_mcid(0)
    assert kept.text == "бл"
    assert kept.glyphs[0].origin == pytest.approx(word.glyphs[2].origin, abs=0.05)
    assert kept.glyphs[1].origin == pytest.approx(word.glyphs[3].origin, abs=0.05)


def test_fit_fontsize_bounds(font) -> None:
    assert fit_fontsize(font, ["Итого"], 200.0, 40.0) == 10.0
    assert fit_fontsize(font, ["Очень длинная надпись в узкой ячейке"], 30.0, 40.0) < 4.0
    assert fit_fontsize(font, ["а"], 1.0, 1.0) == 2.5
    two = fit_fontsize(font, ["Первая", "Вторая"], 200.0, 12.0)
    assert two == pytest.approx(12.0 / (1.15 * 2))
