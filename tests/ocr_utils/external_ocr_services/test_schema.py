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
    drop_duplicate_supplied,
    strip_hyphen_supplied,
    tags_from_edge_words,
    unbalanced_tags,
)

PAGE = {
    "is_damaged": False,
    "damage_description": "",
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


def test_tags_from_edge_words_gaps_stay_out_and_empty_entries_counted():
    """Пропуски из edge_words в тело не переносятся (в теле слово целое — «в первую очередь»); hidden/unclear —
    как раньше; записи «как видно = как написано» без тегов считаются пустыми."""
    body = "Слово предпри и ности и конец, в первую очередь ещё."
    edge = [
        {"seen": "предпри", "full": "предпри<gap>▒▒▒▒</gap>", "kind": "gap"},
        {"seen": "ности", "full": "<gap>▒▒▒</gap>ности", "kind": "gap"},
        {"seen": "первую", "full": "первую<gap>3</gap>", "kind": "gap"},
        {"seen": "кон", "full": "конец", "kind": "hidden"},
        {"seen": "ещ", "full": "ещё", "kind": "unclear"},
        {"seen": "Слово", "full": "Слово", "kind": "hidden"},
        {"seen": "и", "full": "и", "kind": "hidden"},
    ]
    out, report = tags_from_edge_words(body, edge)
    assert report.inserted == 2 and report.empty == 2
    assert "кон<supplied>ец</supplied>" in out and "ещ<unclear>ё</unclear>" in out
    assert "<gap>" not in out and "в первую очередь" in out


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


def test_strip_hyphen_supplied_and_hyphenated_entries_skipped():
    """Продолжение переноса, объявленное восстановленным («ва-» → «ва<supplied>л</supplied>ютных»): тег снимается,
    буквы остаются; такая запись и тегов в тело не добавляет."""
    body = "Курс ва<supplied>л</supplied>ютных операций и кре<supplied>ди</supplied>ты, а снабже<supplied>ния</supplied> нет."
    edge = [
        {"seen": "ва-", "full": "ва<supplied>л</supplied>ютных", "kind": "hidden"},
        {"seen": "кре-", "full": "кре<supplied>ди</supplied>ты", "kind": "hidden"},
        {"seen": "снабже", "full": "снабже<supplied>ния</supplied>", "kind": "hidden"},
        {"seen": "нет-", "full": "нет", "kind": "hidden"},
    ]
    out, stripped = strip_hyphen_supplied(body, edge)
    assert stripped == 2 and "Курс валютных операций и кредиты," in out
    assert "снабже<supplied>ния</supplied>" in out, "честная достройка без дефиса не трогается"
    out2, report = tags_from_edge_words("Курс валютных операций.", edge[:1])
    assert report.inserted == 0 and out2 == "Курс валютных операций."


def test_drop_duplicate_supplied():
    """«трудностей <supplied>стей</supplied>» — слово целое, хвост в теге лишний; чужой хвост остаётся."""
    body = "не без трудностей <supplied>стей</supplied> сформирован парк на базе трактора <supplied>ра</supplied> К-701"
    out, dropped = drop_duplicate_supplied(body)
    assert dropped == 2 and out == "не без трудностей сформирован парк на базе трактора К-701"
    keep = "пункта <supplied>проката</supplied> и Ру<supplied>ко</supplied>водители"
    assert drop_duplicate_supplied(keep) == (keep, 0)


def test_messages_field_roundtrip_and_header():
    """messages — не из ответа модели (пусто), но перечитывается из .json и попадает в шапку .md."""
    result = parse_json_text('{"content_markdown": "Текст.", "messages": ["не поле модели"]}')
    assert result.messages == ["не поле модели"]
    result.messages.append("оглавление: блок <toc> построен заново по toc")
    assert json.loads(result.to_json())["messages"] == result.messages
    md = to_markdown(result)
    assert 'messages: ["не поле модели", "оглавление: блок <toc> построен заново по toc"]\n---\n' in md
    assert "messages:" not in to_markdown(parse_json_text('{"content_markdown": "Текст."}'))


def test_tag_homoglyphs_and_spaces_normalised():
    """Модель пишет <тoc>, <тоc>, < toc>: имена наших тегов приводятся к латинице без пробелов; чужие теги не трогаются."""
    from ocr_utils.external_ocr_services.schema import normalise_tags

    assert normalise_tags("<тoc>\n- а — 1\n</ toc >") == "<toc>\n- а — 1\n</toc>"
    assert normalise_tags("< toc>x<тоc>y<аuthor>**И**</аuthor>") == "<toc>x<toc>y<author>**И**</author>"
    assert (
        normalise_tags("<table><tr><td>1</td></tr></table> <неизвестный>")
        == "<table><tr><td>1</td></tr></table> <неизвестный>"
    )
    result = parse_json_text('{"content_markdown": "< toc>\\n- а — 1\\n</toc>"}')
    assert result.content_markdown.startswith("<toc>\n")


def test_legacy_tag_names_and_hyphen_tags_normalised():
    """v15: <rubric_in_toc> старых ответов → <rubric-in-toc>; имя с дефисом узнаётся и чистится от омоглифов."""
    from ocr_utils.external_ocr_services.schema import normalise_tags

    assert normalise_tags("<rubric_in_toc>*А*</rubric_in_toc>") == "<rubric-in-toc>*А*</rubric-in-toc>"
    assert normalise_tags("< rubric-in-toc >*А*</ rubriс-in-toc>") == "<rubric-in-toc>*А*</rubric-in-toc>"
    result = parse_json_text('{"content_markdown": "<rubric_in_toc>*А*</rubric_in_toc>"}')
    assert result.content_markdown == "<rubric-in-toc>*А*</rubric-in-toc>"


LEGACY_ILLUSTRATIONS = """Текст.

<photo>
> [фотография: портрет мужчины в костюме]
> подпись: И. Иванов
</photo>

<schema>
> [блок-схема]
> Заявка → Склад
> надпись: план
</schema>

<line_art>
> [графика: график роста]
> ось X: годы, 1960, 1965
</line_art>

Дальше."""


def test_legacy_illustration_blocks_become_fenced_blocks():
    """v15: старые обёртки <photo>/<schema>/<line_art> с block quote → fenced-блок с видом первой строкой."""
    from ocr_utils.external_ocr_services.schema import (
        IllustrationKind,
        count_illustrations,
        modernise_illustrations,
        unbalanced_fences,
    )

    out = modernise_illustrations(LEGACY_ILLUSTRATIONS)
    assert (
        out == "Текст.\n\n```\n[фотография]\nпортрет мужчины в костюме\nподпись: И. Иванов\n```\n\n"
        "```\n[блок-схема]\nЗаявка → Склад\nнадпись: план\n```\n\n"
        "```\n[графика]\nграфик роста\nось X: годы, 1960, 1965\n```\n\nДальше."
    )
    assert modernise_illustrations(out) == out, "идемпотентно"
    assert count_illustrations(out) == {
        IllustrationKind.PHOTO: 1,
        IllustrationKind.SCHEMA: 1,
        IllustrationKind.LINE_ART: 1,
    }
    assert not unbalanced_fences(out) and unbalanced_fences(out + "\n```\n[графика]\nоборван")
    assert [kind.key for kind in IllustrationKind] == ["photo", "schema", "line-art"]
    # Конвертация идёт при любом разборе — старые .json читаются в новом виде.
    result = parse_json_text(json.dumps({"content_markdown": LEGACY_ILLUSTRATIONS}, ensure_ascii=False))
    assert "<photo>" not in result.content_markdown and "[фотография]" in result.content_markdown
