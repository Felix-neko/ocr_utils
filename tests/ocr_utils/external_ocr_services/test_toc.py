"""Слияние полос оглавления, рубрики-продолжения, дедуп, списки для промпта и отпечаток."""

from ocr_utils.external_ocr_services.schema import TocArticle, TocPage, TocSection
from ocr_utils.external_ocr_services.toc import (
    from_dict,
    merge_pages,
    normalize_title,
    prompt_lists,
    to_dict,
    to_markdown,
    toc_hash,
)


def _article(title, *names, page=None, issue=None):
    return TocArticle(title, [{"name": name, "position": None} for name in names], page, issue)


def test_first_section_without_rubric_continues_previous_page():
    first = TocPage(
        "contents",
        sections=[
            TocSection(None, [_article("Передовая", page="1")]),
            TocSection("Опыт работы", [_article("Первые шаги", "В. Тычинин", page="3")]),
        ],
    )
    second = TocPage(
        "contents",
        continues_previous=True,
        sections=[
            TocSection(None, [_article("Второй шаг", "А. Иванов", page="7"), _article("Первые шаги", page="3")]),
            TocSection("Консультация", [_article("О нормах", page="9")]),
        ],
    )
    toc = merge_pages("contents", [("1975/12/a.jpg", first), ("1975/12/b.jpg", second)])
    assert [section.rubric for section in toc.sections] == [None, "Опыт работы", "Консультация"]
    assert [a.title for a in toc.sections[1].articles] == ["Первые шаги", "Второй шаг"], "продолжение + дедуп"
    assert toc.continuations == 1 and toc.pages == ["1975/12/a.jpg", "1975/12/b.jpg"]
    assert toc.rubrics == ["Опыт работы", "Консультация"]


def test_prompt_lists_and_hash_change_with_content():
    toc = merge_pages(
        "contents", [("a", TocPage("contents", sections=[TocSection("Р", [_article("Т", "И. Фетисов")])]))]
    )
    rubrics, articles = prompt_lists(toc)
    assert rubrics == ["Р"] and articles == [{"title": "Т", "authors": ["И. Фетисов"], "rubric": "Р"}]
    assert prompt_lists(None) == ([], [])
    base = toc_hash(rubrics, articles)
    assert base == toc_hash(rubrics, articles) and len(base) == 12
    assert base != toc_hash(rubrics, articles + [{"title": "Ещё", "authors": []}])
    assert toc_hash([], []) != base


def test_dict_markdown_roundtrip():
    index = merge_pages(
        "index",
        [
            (
                "1975/12/z.jpg",
                TocPage("index", sections=[TocSection("ЭКОНОМИКА", [_article("Т", "А", page="12", issue="3")])]),
            )
        ],
    )
    payload = to_dict({"index": index})
    assert payload["index"]["sections"][0]["articles"][0]["issue"] == "3"
    back = from_dict(payload)
    assert back["index"].articles[0].title == "Т" and back["index"].pages == ["1975/12/z.jpg"]
    text = to_markdown({"index": index})
    assert "# Указатель статей за год" in text and "## ЭКОНОМИКА" in text and "- А. Т — № 3, 12" in text
    assert "- Б. Солодун. Хозяйствовать — 3" in to_markdown(
        {
            "contents": merge_pages(
                "contents",
                [
                    (
                        "a",
                        TocPage(
                            "contents", sections=[TocSection(None, [_article("Хозяйствовать", "Б. Солодун", page="3")])]
                        ),
                    )
                ],
            )
        }
    )


def test_ensure_toc_block_wraps_entries_once():
    """Забытый <toc>: оборачивается отрезок от первого до последнего элемента; готовый тег не трогается."""
    from ocr_utils.external_ocr_services.toc import ensure_toc_block

    body = (
        "# Материально-техническое снабжение\n\nОРГАН ГОСКОМИТЕТА\n\n# СОДЕРЖАНИЕ\n\n"
        "<rubric-in-toc>*РЕШЕНИЯ СЪЕЗДА*</rubric-in-toc>\n\n"
        "<author>**Христораднов Ю.**</author>. Большие задачи — 3\n\n"
        "<rubric-in-toc>*ПРОБЛЕМЫ*</rubric-in-toc>\n\n"
        "- <author>**Колмаков С.**</author>. Система показателей — 11\n\n"
        "Редакционная коллегия: …\n"
    )
    out, wrapped = ensure_toc_block(body)
    assert wrapped
    assert out.split("\n\n")[3:5] == ["<toc>", "<rubric-in-toc>*РЕШЕНИЯ СЪЕЗДА*</rubric-in-toc>"]
    assert "Система показателей — 11\n\n</toc>\n\nРедакционная коллегия" in out
    assert ensure_toc_block(out) == (out, False)
    assert ensure_toc_block("Обычный текст.\n") == ("Обычный текст.\n", False)
    index = "- Иванов И. Название — № 3, 12\n\n- Петров П. Другое — № 4, 5\n"
    assert ensure_toc_block(index)[0].startswith("<toc>\n\n- Иванов")
    # Пункты подряд и `</toc>` приклеен к последнему без пустой строки — весь абзац внутри блока.
    glued = "# С\n\n- А — 1\n- Б — 2\n</toc>\n\nРедколлегия.\n"
    assert ensure_toc_block(glued)[0] == "# С\n\n<toc>\n\n- А — 1\n- Б — 2\n\n</toc>\n\nРедколлегия.\n"


def _page():
    from ocr_utils.external_ocr_services.schema import TocArticle, TocKind, TocPage, TocSection

    return TocPage(
        TocKind.CONTENTS,
        sections=[
            TocSection(
                None, [TocArticle("Первые шаги работы по-новому", [{"name": "В. Тычинин", "position": None}], "1")]
            ),
            TocSection(
                "ОПЫТ РАБОТЫ",
                [
                    TocArticle("Развивать связи", [{"name": "И. Комаровский"}, {"name": "М. Кругман"}], "37"),
                    TocArticle("Затраты и рентабельность", [{"name": "С. Финкель"}], "63"),
                ],
            ),
        ],
    )


def test_parse_toc_block_entries_and_rubrics():
    """Разбор блока: пункты с одним и двумя авторами, без автора, с номером выпуска и «с.», пункт без тире; рубрики — контекст."""
    from ocr_utils.external_ocr_services.toc import parse_toc_block

    body = (
        "# СОДЕРЖАНИЕ\n\n<toc>\n\n- <author>**В. Тычинин**</author>. Первые шаги — 1\n\n"
        "<rubric-in-toc>*ОПЫТ РАБОТЫ*</rubric-in-toc>\n\n"
        "- <author>**И. Комаровский, М. Кругман**</author>. Развивать связи — 37\n"
        "- Семинар по совершенствованию — № 3, 92\n- Указатель — № 12, с. 94\n- В указателе — 5, 49\n"
        "- Информация — №№ 1, 2, 4\n- Новые книги № 1, с. 28; № 2, с. 20\n\nЛишний абзац.\n\n</toc>\n\nРедколлегия.\n"
    )
    entries = parse_toc_block(body)
    assert [e.rubric for e in entries] == [None] + ["ОПЫТ РАБОТЫ"] * 6
    first = entries[0].article
    assert (first.title, first.page, first.issue, first.authors) == (
        "Первые шаги",
        "1",
        None,
        [{"name": "В. Тычинин", "position": None}],
    )
    assert [a["name"] for a in entries[1].article.authors] == ["И. Комаровский", "М. Кругман"] and entries[
        1
    ].article.page == "37"
    assert (entries[2].article.title, entries[2].article.issue, entries[2].article.page) == (
        "Семинар по совершенствованию",
        "3",
        "92",
    )
    assert (entries[3].article.issue, entries[3].article.page) == ("12", "94"), "«№ 12, с. 94»"
    assert (entries[4].article.issue, entries[4].article.page) == ("5", "49"), "указатель без «№»: выпуск, страница"
    assert (entries[5].article.issue, entries[5].article.page) == ("1", "4") and not entries[5].bare
    assert entries[6].bare and entries[6].article.title.startswith("Новые книги") and entries[6].article.page is None
    assert not any(e.bare for e in entries[:6]) and parse_toc_block("Обычный текст.\n") is None


def test_reconcile_toc_body_without_entries_is_rebuilt():
    """Тело с одними рубриками при 3 статьях в toc → все три «не было в теле», блок построен по toc."""
    from ocr_utils.external_ocr_services.toc import reconcile_toc

    lost = "# СОДЕРЖАНИЕ\n\n<toc>\n\n<rubric-in-toc>*ОПЫТ РАБОТЫ*</rubric-in-toc>\n\n</toc>\n\nРедколлегия.\n"
    body, page, check = reconcile_toc(lost, _page())
    assert check.rebuilt and len(check.missing_in_body) == 3 and check.missing_in_toc == []
    assert "в теле не было 3 статей из toc" in check.message() and "построен заново" in check.message()
    assert body.startswith(
        "# СОДЕРЖАНИЕ\n\n<toc>\n\n- <author>**В. Тычинин**</author>. Первые шаги работы по-новому — 1\n\n"
    )
    assert (
        "<rubric-in-toc>*ОПЫТ РАБОТЫ*</rubric-in-toc>\n\n- <author>**И. Комаровский, М. Кругман**</author>. Развивать связи — 37\n"
        in body
    )
    assert body.endswith("— 63\n\n</toc>\n\nРедколлегия.\n") and sum(len(s.articles) for s in page.sections) == 3


def test_reconcile_toc_extra_body_entry_goes_to_toc_section():
    """Статья только в теле под рубрикой → добавлена в секцию toc с той же рубрикой; блок перестроен."""
    from ocr_utils.external_ocr_services.toc import reconcile_toc

    body = (
        "<toc>\n\n- <author>**В. Тычинин**</author>. Первые шаги работы по-новому — 1\n\n"
        "<rubric-in-toc>*Опыт работы*</rubric-in-toc>\n\n- <author>**И. Комаровский, М. Кругман**</author>. Развивать связи — 37\n"
        "- <author>**С. Финкель**</author>. Затраты и рентабельность — 63\n"
        "- <author>**П. Шейн**</author>. Планирование потребности — 76\n\n</toc>\n"
    )
    out, page, check = reconcile_toc(body, _page())
    assert check.missing_in_body == [] and check.missing_in_toc == ["Планирование потребности"] and check.rebuilt
    # Пункт без «— где» присутствием в теле считается, но в объект как новая статья не идёт.
    bare_body = body.replace("</toc>", "- Заметка без номера страницы\n\n</toc>")
    _, page2, check2 = reconcile_toc(bare_body, _page())
    assert check2.missing_in_toc == ["Планирование потребности"] and sum(len(s.articles) for s in page2.sections) == 4
    added = page.sections[1].articles[-1]
    assert page.sections[1].rubric == "ОПЫТ РАБОТЫ" and added.title == "Планирование потребности" and added.page == "76"
    assert added.authors == [{"name": "П. Шейн", "position": None}] and "Планирование потребности — 76" in out
    assert "в toc добавлено 1 статей из тела" in check.message()


def test_reconcile_toc_matching_bodies_untouched():
    """Полное совпадение (регистр, перенос, без «- ») — ничего не меняется, сообщения нет."""
    from ocr_utils.external_ocr_services.toc import reconcile_toc

    body = (
        "<toc>\n\n<author>**В. Тычинин**</author>. первые шаги работы\nпо-новому — 1\n\n"
        "<rubric-in-toc>*ОПЫТ РАБОТЫ*</rubric-in-toc>\n\n- <author>**И. Комаровский, М. Кругман**</author>. Развивать связи — 37\n\n"
        "- <author>**С. Финкель**</author>. ЗАТРАТЫ И РЕНТАБЕЛЬНОСТЬ — 63\n\n</toc>\n"
    )
    out, page, check = reconcile_toc(body, _page())
    assert out == body and check.message() is None and not check.rebuilt
    assert check.as_dict() == {"missing_in_body": [], "missing_in_toc": [], "rebuilt": False}


def test_dedup_ignores_case_and_damage_tags():
    """Одна и та же статья с двух полос (тайлы перекрываются): регистр и теги <supplied> в названии не мешают дедупу."""
    assert normalize_title("Выгоден прокат техни<supplied>ки</supplied>") == normalize_title("ВЫГОДЕН ПРОКАТ ТЕХНИКИ")
    assert normalize_title("Товар<gap>▒▒</gap> и деньги") == "товар и деньги"
    first = TocPage("contents", sections=[TocSection("Почта", [_article("Выгоден прокат техники", page="91")])])
    second = TocPage(
        "contents",
        continues_previous=True,
        sections=[TocSection(None, [_article("Выгоден прокат техни<supplied>ки</supplied>", page="91")])],
    )
    toc = merge_pages("contents", [("a.jpg", first), ("b.jpg", second)])
    assert [a.title for a in toc.articles] == ["Выгоден прокат техники"]
