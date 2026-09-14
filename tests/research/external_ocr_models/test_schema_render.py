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


def test_damage_fields_schema_and_tags():
    from research.external_ocr_models.schema import json_schema, tag_counts, unbalanced_tags

    text = GOOD.replace(
        '"notes": ""',
        '"restored": ["<restored>м</restored>ного", " "], "fuzzy": ["пр<fuzzy>е</fuzzy>д"], "unknown": 2, "damage": "правый край срезан", "notes": ""',
    )
    result = parse_json_text(text)
    assert result.restored == ["<restored>м</restored>ного"] and result.fuzzy == ["пр<fuzzy>е</fuzzy>д"]
    assert result.unknown == 2 and result.damage == "правый край срезан"
    assert parse_json_text(GOOD).restored == [] and parse_json_text(GOOD).unknown == 0
    assert "restored" not in json_schema()["properties"]
    assert {"restored", "fuzzy", "unknown", "damage", "edge_words"} <= set(json_schema(True)["required"])
    assert list(json_schema(True)["properties"])[0] == "damage"  # модель описывает повреждение до текста
    body = "<restored>м</restored>ного пр<fuzzy>е</fuzzy>д <unknown/> и <unknown />, а тут <fuzzy>без пары"
    assert tag_counts(body) == {"restored": 1, "fuzzy": 1, "unknown": 2}
    assert unbalanced_tags(body) == ["fuzzy"]
    assert unspace_letters("<restored>м</restored>ного П р и м") == "<restored>м</restored>ного *Прим*"


def test_tags_from_edge_words():
    from research.external_ocr_models.schema import tags_from_edge_words

    body = "нужно нефтепромысловое оборудование. Часть в Азербайджане. Уже <restored>ре</restored>сурсов много; неясно слово."
    words = [
        {"seen": "о", "full": "оборудование", "kind": "hidden"},
        {"seen": "Азербайджа", "full": "Азербайджане", "kind": "fuzzy"},
        {"seen": "сурсов", "full": "ресурсов", "kind": "hidden"},  # уже помечено — не трогать
        {"seen": "сло", "full": "слово", "kind": "unknown"},
        {"seen": "x", "full": "нет такого", "kind": "hidden"},
        {"seen": "мно-", "full": "много", "kind": "hidden"},  # перенос — не срез
    ]
    out, inserted = tags_from_edge_words(body, words)
    assert inserted == 3
    assert "о<restored>борудование</restored>" in out and "Азербайджа<fuzzy>не</fuzzy>" in out
    assert out.count("<restored>ре</restored>сурсов") == 1 and "сло<unknown/>" in out


def test_structure_fields_and_tags():
    from research.external_ocr_models.evaluate import normalize, structure

    text = GOOD.replace(
        '"is_toc": false,',
        '"is_toc": false, "rubric": "Опыт работы", "title": "Заголовок", "authors": [{"name": "И. Фетисов", "position": "начальник"}, {"name": " ", "position": null}, {"name": "Наш корр.", "position": null}],',
    )
    result = parse_json_text(text)
    assert result.rubric == "Опыт работы" and result.title == "Заголовок"
    assert result.authors == [{"name": "И. Фетисов", "position": "начальник"}, {"name": "Наш корр.", "position": None}]
    assert parse_json_text(GOOD).authors == [] and parse_json_text(GOOD).rubric is None
    body = "<rubric>*ОПЫТ*</rubric>\n\n# Заголовок\n\n<author>**И. Фетисов,**</author>\n<position>*начальник*</position>\n\n**Просто жирное.**\n\nТекст."
    counts = structure(body)
    assert (counts.rubrics, counts.authors, counts.positions, counts.h1, counts.h3) == (1, 1, 1, 1, 0)
    assert normalize(body) == "опыт заголовок и. фетисов, начальник просто жирное. текст."
