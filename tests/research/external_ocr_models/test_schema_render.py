"""Разбор ответа модели и обратная раскладка в markdown."""

import pytest

from research.external_ocr_models.render import to_markdown
from research.external_ocr_models.schema import ParseError, parse_json_text, parse_markdown_text, unspace_letters

GOOD = '{"page_number": "12", "running_header": null, "running_footer": "МТС № 3", "is_toc": false, "content_markdown": "# Заголовок\\n\\nТекст.", "notes": ""}'


def test_plain_json():
    result = parse_json_text(GOOD)
    assert result.page_number == "12"
    assert result.running_header is None
    assert result.running_footer == "МТС № 3"
    assert result.content_markdown.startswith("# Заголовок")


def test_fenced_json_with_preamble():
    text = "Вот результат:\n```json\n" + GOOD + "\n```\nГотово."
    assert parse_json_text(text).page_number == "12"


def test_trailing_junk_after_json_is_ignored():
    assert parse_json_text(GOOD + '\n{"type": "json_object"}').page_number == "12"
    assert parse_json_text(GOOD + "\n" + GOOD).content_markdown.startswith("# Заголовок")


def test_number_page_number_becomes_text():
    assert parse_json_text(GOOD.replace('"12"', "12")).page_number == "12"


@pytest.mark.parametrize("text", ["", "   ", "не json", '{"page_number": "1"}', '{"content_markdown": 5}'])
def test_bad_json_raises(text):
    with pytest.raises(ParseError):
        parse_json_text(text)


def test_markdown_front_matter():
    text = '---\npage_number: "7"\nrunning_header: null\nrunning_footer: "низ"\nis_toc: true\nnotes: ""\n---\n# Заголовок\n\nТело\n'
    result = parse_markdown_text(text)
    assert result.page_number == "7"
    assert result.running_header is None
    assert result.running_footer == "низ"
    assert result.is_toc is True
    assert result.content_markdown == "# Заголовок\n\nТело"


def test_markdown_without_front_matter_keeps_body():
    result = parse_markdown_text("просто текст")
    assert result.content_markdown == "просто текст"
    assert "без YAML" in result.notes


def test_render_roundtrip():
    result = parse_json_text(GOOD)
    md = to_markdown(result)
    assert md.startswith('---\npage_number: "12"\nrunning_header: null\n')
    assert md.endswith("# Заголовок\n\nТекст.\n")
    again = parse_markdown_text(md)
    assert again.page_number == "12" and again.running_footer == "МТС № 3" and again.is_toc is False


def test_unspace_letters():
    assert unspace_letters("П р и м е ч а н и е. Емкости складов") == "*Примечание*. Емкости складов"
    assert unspace_letters("см. т а б л. 2 и текст") == "см. *табл*. 2 и текст"
    assert unspace_letters("**П р и м е ч а н и е**") == "**Примечание**"
    # обычный текст с короткими словами не трогаем
    assert unspace_letters("в 3 раза и на 5 т") == "в 3 раза и на 5 т"
    assert unspace_letters("а б в") == "*абв*"
    assert unspace_letters("и т. д.") == "и т. д."


def test_parse_applies_unspace():
    text = GOOD.replace("Текст.", "П р и м е ч а н и е. Текст.")
    assert parse_json_text(text).content_markdown.endswith("*Примечание*. Текст.")


def test_restored_field_and_schema():
    from research.external_ocr_models.schema import json_schema

    text = GOOD.replace('"notes": ""', '"restored": ["района", " "], "notes": ""')
    assert parse_json_text(text).restored == ["района"]
    assert parse_json_text(GOOD).restored == []
    assert "restored" not in json_schema()["properties"] and "restored" in json_schema(True)["required"]
