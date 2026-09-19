"""Склейка разорванных переносов по словарю: правила, теги внутри слова, шаг в recognize_page и опция."""

import json
from pathlib import Path

import pytest

from ocr_utils.external_ocr_services.hyphen_join import (
    JoinRule,
    Morph,
    MorphBackend,
    default_morph,
    join_across_boundary,
    join_broken_hyphens,
    should_join,
)
from ocr_utils.external_ocr_services.models import resolve
from ocr_utils.external_ocr_services.ocr import PageJob, RunOptions, recognize_page
from tests.ocr_utils.external_ocr_services.test_pipeline import ISSUE, FakeClient, _answer, _make_pages


def test_rule_e_joins_breaks_and_keeps_compounds():
    morph = default_morph()
    assert morph.backend is MorphBackend.PYMORPHY3 and default_morph() is morph, "один анализатор на процесс"
    text = (
        "взять кре-диты и ва<supplied>л</supplied>ютных, материально-техническое, торгово-экономических, "
        "когда-нибудь, Мифи-нанц, три-четыре, миллио-нов"
    )
    out, report = join_broken_hyphens(text, morph)
    assert "кредиты" in out and "миллионов" in out
    for kept in ("материально-техническое", "торгово-экономических", "когда-нибудь", "Мифи-нанц", "три-четыре"):
        assert kept in out, kept
    assert "ва<supplied>л</supplied>ютных" in out, "слово без дефиса не трогается"
    assert report.joined == ["кре-диты", "миллио-нов"] and "торгово-экономических" in report.kept
    # Теги внутри слова сохраняются, заменяется только дефис.
    tagged, report = join_broken_hyphens("кре-<supplied>ди</supplied>ты", morph, JoinRule.A)
    assert tagged == "кре<supplied>ди</supplied>ты" and report.joined == ["кре-диты"]
    # Правило E против C: составное с неизвестной первой половиной на «о» не сливается только у E.
    assert should_join("торгово", "экономических", morph, JoinRule.C) is True
    assert should_join("торгово", "экономических", morph, JoinRule.E) is False
    assert (
        should_join("кре", "диты", morph, JoinRule.E) is True
        and should_join("Мифи", "нанц", morph, JoinRule.E) is False
    )


def test_tags_around_hyphen_and_line_break_inside_word():
    morph = default_morph()
    # Закрывающий тег между половиной и дефисом, тег вокруг первой половины, тег после дефиса:
    # заменяется только дефис, теги остаются на своих местах.
    cases = {
        "<supplied>кре</supplied>-диты": "<supplied>кре</supplied>диты",
        "кре-</supplied>диты": "кре</supplied>диты",
        "<unclear>кре-</unclear><supplied>ди</supplied>ты": "<unclear>кре</unclear><supplied>ди</supplied>ты",
        "за-\nдолженность": "задолженность",
        "торгово-\nэкономических": "торгово-\nэкономических",
    }
    for source, expected in cases.items():
        out, _ = join_broken_hyphens(source, morph)
        assert out == expected, source


def test_long_word_without_hyphen_is_linear():
    """37-буквенное слово без дефиса вешало прогон 1976/12: вложенный квантификатор в регулярке."""
    import time

    body = "Трест «Красноярскинструментподшипникснабсбыт» и Красноярскэлектроприборснабсбытсбытснабжениеснаб " * 3
    started = time.monotonic()
    out, report = join_broken_hyphens(body, default_morph())
    assert time.monotonic() - started < 1.0 and out == body and not report.joined
    assert join_across_boundary(body, "жения", default_morph()) is None


def test_join_across_boundary_keeps_tags_and_compounds():
    morph = default_morph()
    joined = join_across_boundary("форму <supplied>снаб-</supplied>", "жения (транзитную).", morph)
    assert joined is not None and joined.joined and joined.word == "снаб-жения"
    assert joined.text == "форму <supplied>снаб</supplied>жения (транзитную)."
    assert joined.text[joined.head_start :] == "жения (транзитную)."
    joined = join_across_boundary("форму снаб-  ", "<unclear>жения</unclear> и", morph)
    assert joined is not None and joined.text == "форму снаб<unclear>жения</unclear> и"
    assert joined.text[joined.head_start :] == "<unclear>жения</unclear> и"
    # Половина в дефисной цепочке: перед хвостом дефис, после головы дефис с продолжением.
    chained = join_across_boundary("проработанностью технико-техно-", "логического взаимодействия", morph)
    assert (
        chained is not None and chained.joined and chained.text.startswith("проработанностью технико-технологического ")
    )
    chained = join_across_boundary("стимулирования снаб-", "женческо-сбытовых организаций", morph)
    assert chained is not None and chained.joined and "снабженческо-сбытовых организаций" in chained.text
    assert join_across_boundary("связи военно-", "воздушные силы", morph).joined is False, "составное — дефис остаётся"
    compound = join_across_boundary("связи торгово-", "экономических стран.", morph)
    assert compound is not None and not compound.joined and compound.text == "связи торгово-экономических стран."
    # Не слово с переносом: нет дефиса, голова с прописной, дефис после цифры.
    assert join_across_boundary("связи торгово", "экономических", morph) is None
    assert join_across_boundary("связи снаб-", "Жения", morph) is None
    assert join_across_boundary("в 1966-", "1967 гг.", morph) is None


def test_mawo_backend_is_optional():
    pytest.importorskip("mawo_pymorphy3")
    out, report = join_broken_hyphens("кре-диты", Morph(MorphBackend.MAWO), JoinRule.A)
    assert out == "кредиты" and report.joined == ["кре-диты"]


def test_recognize_page_joins_hyphens_and_records_them(tmp_path):
    _make_pages(tmp_path / "in")
    rel = Path(ISSUE) / "IMG_0002.jpg"
    body = "# Заголовок\n\nМожно взять дополнительные кре-диты в международных банках и торгово-экономических связях."
    for join_hyphens, expected in ((True, ["кре-диты"]), (False, None)):
        fake = FakeClient(lambda p: _answer(body=body))
        out_dir = tmp_path / ("on" if join_hyphens else "off")
        meta, result = recognize_page(
            fake,
            resolve("deepseek-v41-flash"),
            tmp_path / "in" / rel,
            PageJob(rel),
            out_dir,
            RunOptions(join_hyphens=join_hyphens),
        )
        assert meta.get("hyphens_joined") == expected
        assert (
            "кредиты" in result.content_markdown
        ) is join_hyphens and "торгово-экономических" in result.content_markdown
        saved = json.loads((out_dir / ISSUE / "IMG_0002.json").read_text(encoding="utf-8"))
        assert ("кредиты" in saved["content_markdown"]) is join_hyphens
    assert RunOptions().join_hyphens is True
