"""Промпты по этапам: правило `#`, списки выпуска, нейтральная фраза, раскладка тайлов."""

from ocr_utils.external_ocr_services.prompts import DEFAULT_DAMAGE_NOTE, system_prompt, user_prompt


def test_toc_stage_prompt():
    text = system_prompt("toc", "журнал «МТС», 1975")
    assert "(this one: журнал «МТС», 1975)" in text
    assert "ONLY `#` heading allowed is the main heading" in text and '"toc": {"kind"' in text
    assert "KNOWN STRUCTURE" not in text and "<restored>" in text


def test_page_stage_prompt_with_and_without_lists():
    articles = [{"title": "Улучшать методы", "authors": ["И. Фетисов"]}, {"title": "Без автора", "authors": []}]
    with_lists = system_prompt("page", "", ["Консультация"], articles)
    assert "KNOWN STRUCTURE OF THIS ISSUE" in with_lists
    assert "«Улучшать методы» — И. Фетисов" in with_lists and "* «Без автора»\n" in with_lists
    assert "Rubrics: «Консультация»." in with_lists and "ONLY for a title from the list" in with_lists
    assert (
        "separate article with its own author" not in with_lists
        and "EVERY other heading is `## Подзаголовок`" in with_lists
    )
    assert (
        "starts_here" in with_lists and "running_header" in with_lists and "ONE short sentence in Russian" in with_lists
    )
    assert "directly before a `#` title" in with_lists
    assert "<latex>$$" in with_lists and "\\frac" in with_lists and "<latex>$V_{потр}$</latex>" in with_lists
    assert "<latex>" in system_prompt("toc")
    assert '"toc":' not in with_lists and "Check whether this page is itself a table of contents" in with_lists
    plain = system_prompt("page")
    assert "KNOWN STRUCTURE" not in plain and "Article title → `# Заголовок статьи`" in plain
    assert "starts_here" in plain and "continues_previous" not in system_prompt("page")
    assert (
        "continues_previous" not in with_lists
    ), "поле полосы убрано: оно подталкивало модель подписывать продолжение названием"
    assert '"article":' not in system_prompt("toc")
    assert "journal or a newspaper" in plain and "this one:" not in plain


def test_user_prompt_tiles_and_hint():
    two = user_prompt(2, 1, 2)
    assert "2 overlapping tiles: 1 column(s) × 2 row(s)" in two and "left column before" not in two
    assert DEFAULT_DAMAGE_NOTE in two and "Transcribe this page." in two
    four = user_prompt(4, 2, 2, "левый край срезан", "toc", "index")
    assert "the whole left column before the right one" in four and "ANNUAL INDEX" in four
    assert "левый край срезан" in four and DEFAULT_DAMAGE_NOTE not in four
    one = user_prompt(1, 1, 1, stage="toc")
    assert "tiles" not in one and "TABLE OF CONTENTS of the issue" in one
