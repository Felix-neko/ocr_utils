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
