"""Схема по этапу и терпимый разбор ответа: оглавление, toc_kind, поля повреждений, мусор вокруг JSON."""

import json

import pytest

from ocr_utils.external_ocr_services.render import to_markdown
from ocr_utils.external_ocr_services.schema import (
    DamageTag,
    ParseError,
    Stage,
    TocKind,
    json_schema,
    parse_json_text,
    tag_counts,
    tags_from_edge_words,
    unbalanced_tags,
)

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
    "content_markdown": "# О нормах\n\nП р и м е ч а н и е. Текст <unclear>и</unclear> ещё <gap>▒▒</gap>.",
    "supplied": [],
    "unclear": ["<unclear>и</unclear>"],
    "gap": 1,
    "edge_words": [{"seen": "ещ", "full": "ещё", "kind": "unclear"}],
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
    assert tag_counts(result.content_markdown) == {"supplied": 0, "unclear": 1, "gap": 1}
    assert result.unclear == ["<unclear>и</unclear>"] and result.gap == 1
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


def test_enums_serialise_as_plain_strings():
    """StrEnum в результате и meta уходят в JSON своими значениями — формат файлов не меняется."""
    result = parse_json_text('{"content_markdown": "<supplied>а</supplied>б<gap>▒</gap>", "toc_kind": "index"}', "toc")
    assert result.toc_kind is TocKind.INDEX and result.toc_kind == "index"
    assert json.loads(result.to_json())["toc_kind"] == "index"
    counts = tag_counts(result.content_markdown)
    assert counts[DamageTag.SUPPLIED] == 1 and json.loads(json.dumps(counts)) == {"supplied": 1, "unclear": 0, "gap": 1}
    assert json_schema(Stage.TOC)["properties"]["toc_kind"]["enum"] == ["none", "contents", "index"]
    assert json_schema("page") == json_schema(Stage.PAGE)


def test_legacy_damage_tags_and_fields_are_modernised():
    """Ответ по старой схеме (до v10): теги и поля переводятся в TEI, <unknown/> — в <gap> с заполнителем."""
    legacy = {
        "content_markdown": "снабже<restored>ния</restored> пр<fuzzy>е</fuzzy>д <unknown/>ности",
        "restored": ["снабже<restored>ния</restored>"],
        "fuzzy": ["пр<fuzzy>е</fuzzy>д"],
        "unknown": 1,
        "edge_words": [{"seen": "ности", "full": "<unknown/>ности", "kind": "unknown"}],
    }
    result = parse_json_text(json.dumps(legacy, ensure_ascii=False))
    assert result.content_markdown == "снабже<supplied>ния</supplied> пр<unclear>е</unclear>д <gap>▒▒▒</gap>ности"
    assert result.supplied == ["снабже<restored>ния</restored>"] and result.unclear and result.gap == 1
    assert tag_counts(result.content_markdown) == {"supplied": 1, "unclear": 1, "gap": 1}
    assert unbalanced_tags("<gap>▒</gap> <supplied>а") == ["supplied"]


def test_tags_from_edge_words_gap_filler_width():
    """Пропуск из edge_words: ширина — по заполнителям модели, иначе по разнице длин, иначе три; сторона — как у модели."""
    body = "Слово предпри и ности и конец."
    edge = [
        {"seen": "предпри", "full": "предпри<gap>▒▒▒▒</gap>", "kind": "gap"},
        {"seen": "ности", "full": "<gap>▒▒▒</gap>ности", "kind": "gap"},
        {"seen": "кон", "full": "конец", "kind": "hidden"},
    ]
    out, inserted = tags_from_edge_words(body, edge)
    assert inserted == 3
    assert "предпри<gap>▒▒▒▒</gap>" in out and "<gap>▒▒▒</gap>ности" in out and "кон<supplied>ец</supplied>" in out


def test_gap_number_expanded_by_code():
    """Модель пишет <gap>N</gap>; разбор ставит заполнители (N < 16) или «[N symbols]»; мусор и ряды «▒» — по правилу."""
    from ocr_utils.external_ocr_services.schema import GAP_COUNT_MAX, expand_gaps, gap_width, is_gap_runaway

    assert expand_gaps("а<gap>4</gap>б") == "а<gap>▒▒▒▒</gap>б"
    assert expand_gaps("<gap>15</gap>") == "<gap>" + "▒" * 15 + "</gap>"
    assert (
        expand_gaps("<gap>16</gap>") == "<gap>[16 symbols]</gap>"
        and expand_gaps("<gap>200</gap>") == "<gap>[200 symbols]</gap>"
    )
    assert (
        expand_gaps("<gap>▒▒</gap>") == "<gap>▒▒</gap>"
        and expand_gaps("<gap>" + "▒" * 30 + "</gap>") == "<gap>[30 symbols]</gap>"
    )
    assert expand_gaps("<gap>abc</gap>") == "<gap>▒▒▒</gap>" and expand_gaps("<gap></gap>") == "<gap>▒▒▒</gap>"
    assert expand_gaps("<gap>0</gap>") == "<gap>▒</gap>" and gap_width("99999") == GAP_COUNT_MAX
    assert expand_gaps("<gap>[7 symbols]</gap>") == "<gap>▒▒▒▒▒▒▒</gap>", "уже раскрытое читается обратно"
    result = parse_json_text('{"content_markdown": "предпри<gap>4</gap> и <gap>80</gap>", "gap": 2}')
    assert result.content_markdown == "предпри<gap>▒▒▒▒</gap> и <gap>[80 symbols]</gap>" and result.gap == 2
    assert tag_counts(result.content_markdown)["gap"] == 2
    # Старый формат с повтором символа: ряд от 20 «▒» в сыром ответе — сбой (обрезанный ответ).
    assert is_gap_runaway("x<gap>" + "▒" * 25) and not is_gap_runaway("<gap>▒▒▒▒▒▒</gap>")


def test_tags_from_edge_words_numeric_gap():
    """Пропуск из edge_words с числом в full: ширина из числа, сторона — как у модели."""
    body = "Слово предпри и ности."
    edge = [
        {"seen": "предпри", "full": "предпри<gap>4</gap>", "kind": "gap"},
        {"seen": "ности", "full": "<gap>20</gap>ности", "kind": "gap"},
    ]
    out, inserted = tags_from_edge_words(body, edge)
    assert inserted == 2 and "предпри<gap>▒▒▒▒</gap>" in out and "<gap>[20 symbols]</gap>ности" in out
