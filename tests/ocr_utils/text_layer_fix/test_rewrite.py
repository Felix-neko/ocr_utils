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
    # «Прямо вторая» помещается в зону 150×30 pt одной строкой: перенос по словам не нужен.
    assert (stats.blanked, stats.trimmed, stats.inserted, stats.inserted_lines) == (1, 1, 4, 4)

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
    assert dirs["Прямо вторая"] == (1.0, 0.0)
    for insert in inserts:
        first = insert.text.split()[0]
        assert any(hit.intersects(insert.rect) for hit in page.search_for(first)), first
    report = verify_page(doc0[0], page, [words[1]], [words[0]], inserts, image.main_xref, font=font)
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


def test_wrap_lines_prefers_larger_font_with_word_wrap(font) -> None:
    from ocr_utils.text_layer_fix.rewrite import MAX_FONT_PT, MIN_FONT_PT, wrap_lines

    text = "Специфицированная норма на условный сутко-комплект"
    # Узкая боковая ячейка: длина 120 pt, толщина 40 pt — одной строкой влезло бы только петитом.
    lines, size = wrap_lines(font, text, 120.0, 40.0)
    assert len(lines) >= 2 and " ".join(lines) == text
    assert size > MIN_FONT_PT
    assert all(font.length(line, size) <= 120.0 + 1e-6 for line in lines)
    assert len(lines) * 1.15 * size <= 40.0 + 1e-6
    # Короткий текст в просторной зоне — одна строка максимальным кеглем.
    assert wrap_lines(font, "Всего", 200.0, 40.0) == (["Всего"], MAX_FONT_PT)
    # Слово длиннее зоны — своей строкой, кегль по толщине зоны (по длине его ужмёт растяжение).
    lines, size = wrap_lines(font, "Сверхдлинноесловобезпробелов", 10.0, 5.0)
    assert lines == ["Сверхдлинноесловобезпробелов"] and MIN_FONT_PT < size * 1.15 <= 5.0
    assert wrap_lines(font, "   ", 100.0, 10.0) == ([], 0.0)


def test_wrapped_insert_lines_are_searchable(tmp_path, font) -> None:
    src, _ = finereader_like_pdf(tmp_path / "fr.pdf")
    doc = fitz.open(str(src))
    raw = page_content(doc[0])
    insert = Insert("Специфицированная норма на условный сутко-комплект", fitz.Rect(300, 100, 340, 220), 90)
    stats = apply_edits(doc, 0, raw, [], {}, [insert], font)
    assert stats.inserted_lines >= 2
    out = tmp_path / "out.pdf"
    doc.save(str(out))
    page = fitz.open(str(out))[0]
    for word in ("Специфицированная", "условный", "сутко-комплект"):
        assert any(hit.intersects(insert.rect) for hit in page.search_for(word)), word


def test_refit_target_rules() -> None:
    from ocr_utils.text_layer_fix.rewrite import MIN_REFIT_PT, refit_target

    crop = fitz.Rect(50, 50, 350, 550)
    # Целиком внутри — не трогать.
    assert refit_target(fitz.Rect(100, 100, 150, 110), crop) is None
    # Частично снаружи — пересечение.
    assert refit_target(fitz.Rect(30, 100, 80, 110), crop) == fitz.Rect(50, 100, 80, 110)
    # Целиком снаружи слева — полоска у левого края, высота прежняя.
    outside = refit_target(fitz.Rect(10, 100, 40, 110), crop)
    assert outside == fitz.Rect(50, 100, 50 + MIN_REFIT_PT, 110)
    # Узкое пересечение расширяется внутрь до MIN_REFIT_PT.
    thin = refit_target(fitz.Rect(20, 100, 50.5, 110), crop)
    assert thin.x0 == 50 and thin.width == MIN_REFIT_PT
    # Снаружи по обеим осям (угол) — квадратик в углу.
    corner = refit_target(fitz.Rect(0, 0, 20, 20), crop)
    assert corner == fitz.Rect(50, 50, 50 + MIN_REFIT_PT, 50 + MIN_REFIT_PT)


def test_refit_word_lands_inside_crop(tmp_path, font) -> None:
    """Слова, вылезающие за обрезанную страницу, ужимаются внутрь и находятся поиском; остальные на месте."""
    from ocr_utils.text_layer_fix.fixer import apply_page_edits, plan_page_edits, verify_saved

    src, _ = finereader_like_pdf(tmp_path / "fr.pdf")
    doc0 = fitz.open(str(src))
    pdf = pikepdf.open(str(src))
    # Обрезка режет «Таблица» (x 100–150) пополам, «Резервы» (x 40–53) остаётся снаружи, «и» внутри.
    crop = fitz.Rect(125, 60, 380, 580)
    edits = plan_page_edits(doc0[0], pdf, {"words": [], "zones": [], "readings": {}}, crop)
    assert {w.text for w, _ in edits.refits.values()} == {"Таблица", "Резервы"}
    assert [w.text for w in edits.kept] == ["и"]
    doc = fitz.open(str(src))
    fix = apply_page_edits(doc, 0, edits, font)
    assert fix.refitted == 2
    doc[0].set_mediabox(crop * ~doc[0].transformation_matrix)  # set_mediabox ждёт координаты PDF
    out = tmp_path / "out.pdf"
    doc.save(str(out))
    saved = fitz.open(str(out))
    page = saved[0]
    assert abs(page.rect.width - crop.width) < 0.01 and abs(page.rect.height - crop.height) < 0.01
    report = verify_saved(doc0[0], page, edits, None, font, crop)
    assert report.ok, report
    words = {w[4]: fitz.Rect(w[:4]) for w in page.get_text("words")}
    assert "Таблица" in words and "Резервы" in words and "и" in words
    inside = page.rect + (-0.01, -0.01, 0.01, 0.01)  # допуск на округление матрицы
    assert inside.contains(words["Таблица"]) and inside.contains(words["Резервы"])
    # «Таблица» ужата в пересечение: начинается у левого края страницы.
    assert words["Таблица"].x0 < 1.0
