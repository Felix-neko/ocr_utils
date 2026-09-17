"""Схема по этапу и терпимый разбор ответа: оглавление, toc_kind, поля повреждений, мусор вокруг JSON."""

import json

import pytest

from ocr_utils.external_ocr_services.render import to_markdown
from ocr_utils.external_ocr_services.schema import ParseError, json_schema, parse_json_text, tag_counts

PAGE = {
    "damage": "",
    "page_number": 12,
    "rubric": "Консультация",
    "title": "О нормах",
    "title_in_list": True,
    "authors": [
        {"name": "И. Фетисов", "position": " начальник ", "article": "starts_here", "printed": "below_title"},
        {"name": "", "position": None},
        {"name": "Б. Мительман", "position": None, "article": "чушь", "printed": None},
    ],
    "continues_previous": True,  # старое поле полосы: разбор его игнорирует
    "running_header": None,
    "running_footer": "МТС",
    "toc_kind": "none",
    "content_markdown": "# О нормах\n\nП р и м е ч а н и е. Текст <fuzzy>и</fuzzy> ещё <unknown/>.",
    "restored": [],
    "fuzzy": ["<fuzzy>и</fuzzy>"],
    "unknown": 1,
    "edge_words": [{"seen": "ещ", "full": "ещё", "kind": "fuzzy"}],
    "notes": "",
}


def test_schema_is_strict_and_stage_specific():
    page = json_schema("page")
    assert page["additionalProperties"] is False and page["required"] == list(page["properties"])
    assert "toc" not in page["properties"] and page["properties"]["toc_kind"]["enum"] == ["none", "contents", "index"]
    author = page["properties"]["authors"]["items"]
    assert author["required"] == ["name", "position", "article", "printed"]
    assert "continues_previous" not in page["properties"], "поле полосы убрано"
    toc = json_schema("toc")
    assert "toc" in toc["properties"] and toc["required"] == list(toc["properties"]) and "toc" in toc["required"]
    with pytest.raises(ValueError):
        json_schema("x")


def test_parse_page_fields_and_unspace():
    result = parse_json_text(json.dumps(PAGE, ensure_ascii=False))
    assert result.page_number == "12" and result.title_in_list is True and result.toc_kind == "none"
    assert result.authors == [
        {"name": "И. Фетисов", "position": "начальник", "article": "starts_here", "printed": "below_title"},
        {"name": "Б. Мительман", "position": None, "article": None, "printed": None},
    ]
    assert not hasattr(result, "continues_previous")
    assert "*Примечание*." in result.content_markdown and result.toc is None
    assert tag_counts(result.content_markdown) == {"restored": 0, "fuzzy": 1, "unknown": 1}
    md = to_markdown(result)
    assert md.startswith('---\npage_number: "12"') and 'toc_kind: "none"' in md and 'title: "О нормах"' in md
    assert "continues_previous" not in md


def test_parse_toc_stage_with_fence_and_trailing_junk():
    payload = dict(PAGE, toc_kind="contents")
    payload["toc"] = {
        "kind": "contents",
        "continues_previous": True,
        "sections": [
            {
                "rubric": None,
                "articles": [{"title": "Т", "authors": [{"name": "А", "position": None}], "page": 5, "issue": None}],
            },
            {"rubric": "Р", "articles": [{"title": "", "authors": []}]},
        ],
    }
    text = "Вот результат:\n```json\n" + json.dumps(payload, ensure_ascii=False) + '\n```\n{"type": "json_object"}'
    result = parse_json_text(text, "toc")
    assert result.toc.kind == "contents" and result.toc.continues_previous is True
    assert [s.rubric for s in result.toc.sections] == [None, "Р"]
    assert result.toc.sections[0].articles[0].page == "5" and result.toc.sections[1].articles == []
    assert result.toc.sections[0].articles[0].authors == [{"name": "А", "position": None}], "в оглавлении без привязки"
    # Наш собственный .json читается тем же разбором.
    again = parse_json_text(result.to_json(), "toc")
    assert again.toc.sections[0].articles[0].title == "Т"


def test_old_is_toc_field_and_bad_json():
    result = parse_json_text('{"content_markdown": "x", "is_toc": true}')
    assert result.toc_kind == "contents"
    for text in ("", "не json", '{"page_number": "1"}'):
        with pytest.raises(ParseError):
            parse_json_text(text)
