"""Маскирование мусора модели: чужие теги, невидимые символы, одиночные `*` и `_`; защищённые зоны; идемпотентность."""

from __future__ import annotations

import json

from ocr_utils.external_ocr_services.masking import MaskReport, sanitize
from ocr_utils.external_ocr_services.schema import MARKDOWN_TAGS, parse_json_text


def _clean(text: str) -> tuple[str, dict]:
    out, report = sanitize(text, MARKDOWN_TAGS)
    return out, report.as_dict()


def test_foreign_tags_become_entities_known_tags_stay():
    out, report = _clean('дре<a>зденским <span><unclear>е</unclear></span>х <b class="x">ж</b> <!-- к -->')
    assert (
        out
        == 'дре&lt;a&gt;зденским &lt;span&gt;<unclear>е</unclear>&lt;/span&gt;х &lt;b class="x"&gt;ж&lt;/b&gt; &lt;!-- к --&gt;'
    )
    assert report["tags"] == ["a", "span", "span", "b", "b", "!--"]
    kept = (
        "<rubric>*Р*</rubric>\n\n<author>**И. И.**</author>\n\n<supplied>к</supplied> <gap>▒▒</gap> <latex>$x$</latex>"
    )
    assert _clean(kept) == (kept, {})


def test_invisible_and_control_characters():
    out, report = _clean("теле­тайпа и\tтаб\r\nzw​sp﻿bompua\u0007")
    assert out == "телетайпа и таб\nzw▯sp▯bom▯pua▯"
    assert report == {
        "soft_hyphens": 1,
        "nbsp": 2,
        "controls": ["U+200B", "U+FEFF", "U+E000", "U+0007"],
        "controls_total": 5,
    }


def test_lone_stars_and_underscores_escaped_paired_markup_and_lists_kept():
    text = (
        "# О методах норм поставок*\n\n[^1]: * В порядке постановки вопроса.\n\n-2* и 3*\n\n"
        "**жирный** и *курсив* и **жирный с * внутри**\n\n* пункт\n* ещё * пункт\n\n* * *\n\nв лице ______ действующего\n\nк_п"
    )
    out, report = _clean(text)
    assert out == (
        "# О методах норм поставок\\*\n\n[^1]: \\* В порядке постановки вопроса.\n\n-2\\* и 3\\*\n\n"
        "**жирный** и *курсив* и **жирный с * внутри**\n\n* пункт\n* ещё \\* пункт\n\n* * *\n\n"
        "в лице \\_\\_\\_\\_\\_\\_ действующего\n\nк\\_п"
    )
    assert report == {"escaped_stars": 5, "escaped_underscores": 7}


def test_protected_zones_untouched():
    text = (
        '<table>\n<tr><td>a_b*</td><td rowspan="2">1*</td></tr>\n</table>\n\n'
        "<latex>$$S_i^t = \\sum_{r=1}^{q} a < b$$</latex>\n\n```\n[графика]\nнадпись: a_b *\n```\n\nкод `x_*y` и текст_с*"
    )
    out, report = _clean(text)
    assert out == text.replace("текст_с*", "текст\\_с\\*")
    assert report == {"escaped_stars": 1, "escaped_underscores": 1}


def test_sanitize_is_idempotent_and_wired_into_parse():
    body = "дре<a>зденским теле­тайпа поставок*. в лице ____ и <table><tr><td>x_y</td></tr></table>"
    once, report = _clean(body)
    twice, again = _clean(once)
    assert twice == once and again == {}
    result = parse_json_text(json.dumps({"content_markdown": body}, ensure_ascii=False))
    assert result.content_markdown == once and result.masked == report
    reread = parse_json_text(result.to_json())
    assert reread.content_markdown == once and reread.masked == {}, "повторный разбор чистого .json ничего не маскирует"
    assert MaskReport().as_dict() == {}


def test_recognize_page_reports_masked_in_meta(tmp_path):
    from pathlib import Path

    from ocr_utils.external_ocr_services.models import resolve
    from ocr_utils.external_ocr_services.ocr import PageJob, RunOptions, recognize_page
    from tests.ocr_utils.external_ocr_services.test_pipeline import ISSUE, FakeClient, _answer, _make_pages

    _make_pages(tmp_path / "in")
    rel = Path(ISSUE) / "IMG_0002.jpg"
    fake = FakeClient(lambda p: _answer(body="Текст дре<a>зденским теле­тайпа."))
    meta, result = recognize_page(
        fake, resolve("deepseek-v41-flash"), tmp_path / "in" / rel, PageJob(rel), tmp_path / "out", RunOptions()
    )
    assert result.content_markdown.rstrip() == "Текст дре&lt;a&gt;зденским телетайпа."
    assert meta["masked"] == {"tags": ["a"], "soft_hyphens": 1}
    assert "&lt;a&gt;" in (tmp_path / "out" / ISSUE / "IMG_0002.md").read_text(encoding="utf-8")
