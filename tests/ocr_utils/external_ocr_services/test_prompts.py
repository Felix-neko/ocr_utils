"""Промпты по этапам: правило `#`, списки выпуска, нейтральная фраза, раскладка тайлов."""

from ocr_utils.external_ocr_services.prompts import DEFAULT_DAMAGE_NOTE, system_prompt, user_prompt


def test_toc_stage_prompt():
    text = system_prompt("toc", "журнал «МТС», 1975")
    assert "(this one: журнал «МТС», 1975)" in text
    assert "ONLY `#` heading allowed is the main heading" in text and '"toc": {"kind"' in text
    assert "KNOWN STRUCTURE" not in text and "<supplied>" in text and "<gap>4</gap>" in text
    assert "▒" not in text and "at most 6" not in text, "число вместо повтора заполнителя"
    # v10: предварительная классификация, рубрики оглавления своим тегом и полностью; v14: блок <toc>
    # модель не пишет (тег искажался), его ставит код — в промптах тега нет вовсе.
    assert "PRELIMINARILY classified" in text and "<toc>" not in text and "</toc>" not in text
    assert "<toc>" not in user_prompt(1, 1, 1, stage="toc") and "<toc>" not in user_prompt(
        1, 1, 1, stage="toc", toc_kind="index"
    )
    assert "two are independent transcriptions" in text, "тело не сокращать из-за объекта toc (v13)"
    assert "<rubric-in-toc>*ОПЫТ РАБОТЫ ТЕРРИТОРИАЛЬНЫХ УПРАВЛЕНИЙ*</rubric-in-toc>" in text
    assert "never shortened" in text and "`<rubric>*ОПЫТ РАБОТЫ*</rubric>`" not in text
    for old in ("<restored>", "<fuzzy>", "<unknown/>", "[картинка"):
        assert old not in text, old
    # v15: картинки трёх видов — fenced-блок с видом первой строкой, без XML-обёртки и без «> »;
    # имена тегов без подчёркиваний; сноски — в теге.
    for piece in ("[блок-схема]", "[фотография]", "[графика]", "<footnote>[^1]:", "ось X", "ось Y", "```"):
        assert piece in text, piece
    for old in ("<schema>", "<photo>", "<line_art>", "<rubric_in_toc>", "> [", "block quote wrapped"):
        assert old not in text, old
    # Нет напечатанного номера страницы — не повреждение (v15).
    assert "NOT damage" in text and "null when no page number is printed" in text
    # v16: без «[неразборчиво]» (спорил с тегами), правило 1 отсылает к правилу 5, поля переименованы,
    # на смазанном слове — тег, а не подстановка; edge_words только по словам с достройкой/сомнением.
    assert "[неразборчиво]" not in text and "rule 5 tells you to restore them" in text
    assert '"is_damaged": boolean' in text and '"damage_description": string' in text
    for old in ('"damaged"', '"damage":', "never nest tags", "for EVERY line that touches"):
        assert old not in text, old
    assert "never a plausible substitute without a tag" in text and "read it as the WHOLE word" in text


def test_page_stage_prompt_with_and_without_lists():
    """Системный промпт этапа page не зависит от выпуска: список статей уходит в пользовательское сообщение."""
    articles = [{"title": "Улучшать методы", "authors": ["И. Фетисов"]}, {"title": "Без автора", "authors": []}]
    with_lists = system_prompt("page", "", has_list=True)
    assert "KNOWN STRUCTURE OF THIS ISSUE" in with_lists and "Улучшать методы" not in with_lists
    assert "ONLY for a title from the list" in with_lists and "EVERY other heading is `## Подзаголовок`" in with_lists
    assert (
        "starts_here" in with_lists and "running_header" in with_lists and "ONE short sentence in Russian" in with_lists
    )
    assert "directly before a `#` title" in with_lists
    assert "<latex>$$" in with_lists and "\\frac" in with_lists and "<latex>$V_{потр}$</latex>" in with_lists
    assert "<latex>" in system_prompt("toc")
    assert '"toc":' not in with_lists and "Check whether this page is itself a table of contents" in with_lists
    plain = system_prompt("page")
    assert "KNOWN STRUCTURE" not in plain and "Article title → `# Заголовок статьи`" in plain
    assert "starts_here" in plain and "continues_previous" not in plain and "continues_previous" not in with_lists
    assert '"article":' not in system_prompt("toc")
    # Список — в пользовательском сообщении, после константных блоков и перед раскладкой тайлов.
    user = user_prompt(2, 1, 2, rubrics=["Консультация"], articles=articles)
    assert "KNOWN STRUCTURE OF THIS ISSUE" in user
    assert "«Улучшать методы» — И. Фетисов" in user and "* «Без автора»\n" in user
    assert "Rubrics: «Консультация»." in user
    assert user.index(DEFAULT_DAMAGE_NOTE) < user.index("Transcribe this page.") < user.index("KNOWN STRUCTURE")
    assert user.index("KNOWN STRUCTURE") < user.index("2 overlapping tiles")
    assert "Apply rule 5 strictly" not in user, "третий повтор правила о повреждениях убран (v16)"
    assert "KNOWN STRUCTURE" not in user_prompt(2, 1, 2) and "KNOWN STRUCTURE" not in user_prompt(1, 1, 1, stage="toc")


def test_system_prompts_share_prefix_and_never_depend_on_issue():
    """Кэш префикса: toc и page совпадают до стадийного правила 6; page с списком и без — до правила 6 тоже."""
    toc, page, page_plain = (
        system_prompt("toc", "X"),
        system_prompt("page", "X", has_list=True),
        system_prompt("page", "X"),
    )
    shared = toc.index("6. Structure (Markdown):")
    assert toc[:shared] == page[:shared] == page_plain[:shared] and shared > 5000
    assert "5. Damaged text" in toc[:shared] and "3. Tables, illustrations" in toc[:shared]
    assert system_prompt("page", "X", has_list=True) == page, "текст детерминирован — иначе кэш не совпадёт"


def test_user_prompt_tiles_and_damage_note():
    two = user_prompt(2, 1, 2)
    assert "2 overlapping tiles: 1 column(s) × 2 row(s)" in two and "left column before" not in two
    assert DEFAULT_DAMAGE_NOTE in two and "Transcribe this page." in two and "which edge" in two
    four = user_prompt(4, 2, 2, stage="toc", toc_kind="index")
    assert "the whole left column before the right one" in four and "ANNUAL INDEX" in four
    one = user_prompt(1, 1, 1, stage="toc")
    assert "tiles" not in one and "TABLE OF CONTENTS of the issue" in one and "preliminarily classified" in one
    assert "<supplied>" in one and "<restored>" not in one
