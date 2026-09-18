"""Слияние полос оглавления, рубрики-продолжения, дедуп, списки для промпта и отпечаток."""

from ocr_utils.external_ocr_services.schema import TocArticle, TocPage, TocSection
from ocr_utils.external_ocr_services.toc import from_dict, merge_pages, prompt_lists, to_dict, to_markdown, toc_hash


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
        "<rubric_in_toc>*РЕШЕНИЯ СЪЕЗДА*</rubric_in_toc>\n\n"
        "<author>**Христораднов Ю.**</author>. Большие задачи — 3\n\n"
        "<rubric_in_toc>*ПРОБЛЕМЫ*</rubric_in_toc>\n\n"
        "- <author>**Колмаков С.**</author>. Система показателей — 11\n\n"
        "Редакционная коллегия: …\n"
    )
    out, wrapped = ensure_toc_block(body)
    assert wrapped
    assert out.split("\n\n")[3:5] == ["<toc>", "<rubric_in_toc>*РЕШЕНИЯ СЪЕЗДА*</rubric_in_toc>"]
    assert "Система показателей — 11\n\n</toc>\n\nРедакционная коллегия" in out
    assert ensure_toc_block(out) == (out, False)
    assert ensure_toc_block("Обычный текст.\n") == ("Обычный текст.\n", False)
    index = "- Иванов И. Название — № 3, 12\n\n- Петров П. Другое — № 4, 5\n"
    assert ensure_toc_block(index)[0].startswith("<toc>\n\n- Иванов")


def test_ensure_toc_entries_rebuilds_lost_list():
    """Тело с одними рубриками при 3 статьях в toc → блок <toc> строится заново по объекту; полный список не трогается."""
    from ocr_utils.external_ocr_services.schema import TocArticle, TocKind, TocPage, TocSection
    from ocr_utils.external_ocr_services.toc import ensure_toc_entries, render_toc_block

    page = TocPage(
        TocKind.INDEX,
        sections=[
            TocSection(None, [TocArticle("Первая", [{"name": "И. Иванов", "position": None}], "5", "3")]),
            TocSection(
                "ОПЫТ", [TocArticle("Вторая", [], "9", "3"), TocArticle("Третья", [{"name": "П. Петров"}], None, None)]
            ),
        ],
    )
    lost = "# УКАЗАТЕЛЬ\n\n<toc>\n\n<rubric_in_toc>*ОПЫТ*</rubric_in_toc>\n\n</toc>\n\nРедколлегия.\n"
    out, rebuilt = ensure_toc_entries(lost, page)
    assert rebuilt and out.startswith("# УКАЗАТЕЛЬ\n\n<toc>\n\n- <author>**И. Иванов**</author>. Первая — № 3, 5\n\n")
    assert (
        "<rubric_in_toc>*ОПЫТ*</rubric_in_toc>\n\n- Вторая — № 3, 9\n- <author>**П. Петров**</author>. Третья\n\n</toc>\n\nРедколлегия."
        in out
    )
    full = "<toc>\n\n- Первая — № 3, 5\n\n- Вторая — № 3, 9\n\n</toc>\n"
    assert ensure_toc_entries(full, page) == (full, False), "две из трёх — больше половины, список цел"
    assert render_toc_block(page).count("\n- ") == 3
