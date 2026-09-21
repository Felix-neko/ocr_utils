"""Сверка `#` и `<rubric>` выпуска с оглавлением при сборке: фантомы, восстановление из колонтитула, полосы оглавления."""

from __future__ import annotations

import json
from pathlib import Path

from ocr_utils.external_ocr_services.assemble import assemble_issue, issue_pages
from ocr_utils.external_ocr_services.schema import PageResult, Stage, TocArticle, TocKind, TocPage, TocSection
from ocr_utils.external_ocr_services.toc import merge_pages, to_dict

ISSUE = "1974/05"


def _page(out_dir: Path, stem: str, body: str, page_number: str | None = None, **fields) -> None:
    """Полоса на диске: .json как пишет recognize_page (лишние поля PageResult — через ``fields``) и meta без ошибки."""
    base = out_dir / ISSUE / stem
    base.parent.mkdir(parents=True, exist_ok=True)
    stage = fields.pop("stage", Stage.PAGE)
    result = PageResult(content_markdown=body, page_number=page_number, **fields)
    base.with_suffix(".json").write_text(result.to_json(), encoding="utf-8")
    base.with_suffix(".meta.json").write_text(
        json.dumps({"page": f"{ISSUE}/{stem}.jpg", "stage": stage.value, "error": None, "parse_error": None}),
        encoding="utf-8",
    )


def _toc(out_dir: Path, sections: list[tuple[str | None, list[tuple[str, str | None]]]]) -> None:
    """toc.json выпуска: секции ``(рубрика, [(название, страница)])``; id присваивает слияние."""
    page = TocPage(
        TocKind.CONTENTS,
        sections=[
            TocSection(rubric, [TocArticle(title, [], page) for title, page in items]) for rubric, items in sections
        ],
    )
    toc = merge_pages(TocKind.CONTENTS, [(f"{ISSUE}/toc.jpg", page)])
    (out_dir / ISSUE).mkdir(parents=True, exist_ok=True)
    (out_dir / ISSUE / "toc.json").write_text(
        json.dumps(to_dict({TocKind.CONTENTS: toc}), ensure_ascii=False), encoding="utf-8"
    )


def _assemble(out_dir: Path):
    return assemble_issue(out_dir, ISSUE, issue_pages(out_dir, ISSUE))


def test_phantom_headings_deleted_or_demoted_by_toc_page(tmp_path):
    """Настоящий `#` — на странице из оглавления; фантом сверху продолжения удаляется, напечатанный лозунг понижается."""
    _toc(tmp_path, [(None, [("Первая статья", "3"), ("Вторая статья", "6")])])
    # Обложка с лозунгом-названием статьи под фото (без текста статьи) — напечатано, понизить.
    _page(tmp_path, "IMG_0001", "```\n[фотография]\nздание\n```\n\n# Вторая статья")
    _page(tmp_path, "IMG_0002", "# Первая статья\n\nТекст первой статьи.", None)
    _page(tmp_path, "IMG_0003", "Продолжение первой.", "4")
    # Продолжение с выдуманным из списка `#` сверху и текстом ниже — удалить.
    _page(tmp_path, "IMG_0004", "# Первая статья\n\nЕщё продолжение первой.", "5")
    _page(tmp_path, "IMG_0005", "# Вторая статья\n\nТекст второй.", "6")
    # Колонтитул с названием статьи, переписанный как `#`, — удалить.
    _page(tmp_path, "IMG_0006", "# Вторая статья\n\nКонец второй.", "7", running_header="Вторая статья")
    assembly = _assemble(tmp_path)
    text = assembly.text
    assert text.count("# Первая статья") == 1 and text.count("\n# Вторая статья") == 1
    assert "## Вторая статья\n\n<!-- article A1 -->\n\n# Первая статья" in text
    assert "Продолжение первой.\n\nЕщё продолжение первой." in text, "фантом сверху продолжения удалён"
    assert "<!-- article A2 -->\n\n# Вторая статья\n\nТекст второй.\n\nКонец второй." in text
    report = assembly.reconcile
    assert [(p["page"].rsplit("/", 1)[1], p["action"]) for p in report.phantom_headings] == [
        ("IMG_0001", "demoted"),
        ("IMG_0004", "deleted"),
        ("IMG_0006", "deleted"),
    ]
    assert [(a["id"], a["status"], a["page_number"]) for a in report.articles] == [
        ("A1", "found", 3),
        ("A2", "found", 6),
    ]
    # Смещения статей в sidecar указывают на `#`; страница 3 не напечатана — выведена по соседям.
    for entry in report.articles:
        assert text[entry["offset"] :].startswith("# ")
        assert text.count("\n", 0, entry["offset"]) + 1 == entry["line"]
    assert assembly.pages[1].page_number is None and assembly.pages[1].suggested_page_number == 3
    assert assembly.pages[1].page_number_source == "suggested"
    assert assembly.as_dict()["phantom_headings"] == report.phantom_headings
    # Начало полосы с удалённым `#` — на следующем блоке.
    assert text[assembly.pages[3].offset :].startswith("Ещё продолжение первой.")


def test_heading_and_rubric_restored_from_running_header_ids(tmp_path):
    """v20: статья без `#`, но с id в колонтитуле, получает заголовок; рубрика, ушедшая в колонтитул, — тег."""
    _toc(tmp_path, [("Опыт работы", [("Статья без заголовка", "10")]), (None, [("Обычная", "12")])])
    _page(
        tmp_path,
        "IMG_0001",
        "<marker>*девиз*</marker>\n\nТекст статьи, заголовок которой модель приняла за колонтитул.",
        "10",
        running_header="Статья без заголовка 10",
        running_header_article_id="A1",
        running_footer="Опыт работы",
        running_footer_rubric_id="R1",
    )
    _page(tmp_path, "IMG_0002", "Продолжение.", "11", running_header="Опыт работы", running_header_rubric_id="R1")
    _page(tmp_path, "IMG_0003", "# Обычная\n\nТекст.", "12", headings=[{"text": "Обычная", "article_id": "A2"}])
    assembly = _assemble(tmp_path)
    text = assembly.text
    assert (
        "<marker>*девиз*</marker>\n\n<!-- rubric R1 -->\n\n<rubric>*Опыт работы*</rubric>\n\n"
        "<!-- article A1 -->\n\n# Статья без заголовка\n\nТекст статьи" in text
    ), "рубрика и заголовок вставлены после маркера, без номера страницы из колонтитула"
    assert "<!-- article A2 -->\n\n# Обычная" in text
    report = assembly.reconcile
    assert [(a["id"], a["status"]) for a in report.articles] == [("A1", "restored"), ("A2", "found")]
    assert [(r["id"], r["status"]) for r in report.rubrics] == [("R1", "restored")]
    assert text[report.articles[0]["offset"] :].startswith("# Статья без заголовка")
    assert text[report.rubrics[0]["offset"] :].startswith("<rubric>*Опыт работы*")


def test_phantom_rubrics_and_wrong_model_ids(tmp_path):
    """Лишний `<rubric>` с текстом колонтитула удаляется, напечатанный повтор — `<marker>`; чужой id модели не мешает."""
    _toc(tmp_path, [("Резервы", [("Первая", "1"), ("Вторая", "3")]), ("Письма", [("Третья", "5")])])
    _page(tmp_path, "IMG_0001", "<rubric>*РЕЗЕРВЫ*</rubric>\n\n# Первая\n\nТекст.", "1")
    # Модель дала рубрике чужой id (R2) — текст говорит «Резервы», это колонтитул → удалить.
    _page(
        tmp_path,
        "IMG_0002",
        "<rubric>*Резервы*</rubric>\n\nПродолжение первой.",
        "2",
        running_header="Резервы 2",
        rubrics=[{"text": "Резервы", "rubric_id": "R2"}],
    )
    _page(tmp_path, "IMG_0003", "# Вторая\n\nТекст.\n\n<rubric>*РЕЗЕРВЫ*</rubric>\n\nНапечатанный повтор ниже.", "3")
    _page(tmp_path, "IMG_0005", "<rubric>*ПИСЬМА*</rubric>\n\n# Третья\n\nТекст.", "5")
    assembly = _assemble(tmp_path)
    text = assembly.text
    assert text.count("<rubric>") == 2
    assert "<!-- rubric R1 -->\n\n<rubric>*РЕЗЕРВЫ*</rubric>\n\n<!-- article A1 -->\n\n# Первая" in text
    assert "Текст.\n\nПродолжение первой." in text, "рубрика-колонтитул удалена"
    assert "Текст.\n\n<marker>*РЕЗЕРВЫ*</marker>\n\nНапечатанный повтор" in text
    assert "<!-- rubric R2 -->\n\n<rubric>*ПИСЬМА*</rubric>\n\n<!-- article A3 -->\n\n# Третья" in text
    assert [(p["page"].rsplit("/", 1)[1], p["action"]) for p in assembly.reconcile.phantom_rubrics] == [
        ("IMG_0002", "deleted"),
        ("IMG_0003", "marker"),
    ]
    assert not assembly.repeated_rubrics


def test_toc_pages_keep_one_list_heading(tmp_path):
    """На полосе оглавления `#` не заголовок списка → `##`; вторая подряд полоса того же вида теряет `# СОДЕРЖАНИЕ`."""
    _toc(tmp_path, [(None, [("Статья", "3")])])
    _page(
        tmp_path,
        "IMG_0001",
        "# Материально-техническое снабжение\n\n## СОДЕРЖАНИЕ\n\n<toc>\n\n- Статья — 3\n\n</toc>",
        stage=Stage.TOC,
        toc_kind=TocKind.CONTENTS,
        toc=TocPage(TocKind.CONTENTS, False, []),
    )
    _page(
        tmp_path,
        "IMG_0002",
        "# СОДЕРЖАНИЕ\n\n**РЕДКОЛЛЕГИЯ**\n\n<toc>\n\n- Ещё — 5\n\n</toc>",
        stage=Stage.TOC,
        toc_kind=TocKind.CONTENTS,
        toc=TocPage(TocKind.CONTENTS, False, []),
    )
    _page(tmp_path, "IMG_0003", "# Статья\n\nТекст.", "3")
    assembly = _assemble(tmp_path)
    text = assembly.text
    assert text.startswith("---\n") and "\n## Материально-техническое снабжение\n\n## СОДЕРЖАНИЕ" in text
    assert "</toc>\n\n**РЕДКОЛЛЕГИЯ**" in text and text.count("# СОДЕРЖАНИЕ") == 1
    assert [(t["page"].rsplit("/", 1)[1], t["action"]) for t in assembly.reconcile.toc_page_headings] == [
        ("IMG_0001", "demoted"),
        ("IMG_0002", "deleted"),
    ]
    assert "<!-- article A1 -->\n\n# Статья" in text


def test_without_toc_json_nothing_is_reconciled(tmp_path):
    _page(tmp_path, "IMG_0001", "# Статья\n\nТекст.", "3")
    _page(tmp_path, "IMG_0002", "# Статья\n\nЕщё.", "4")
    assembly = _assemble(tmp_path)
    assert assembly.text.count("# Статья") == 2 and "<!--" not in assembly.text
    assert assembly.reconcile.articles == [] and assembly.reconcile.phantom_headings == []


def test_heading_not_restored_from_rubric_only_header(tmp_path):
    """Id статьи у колонтитула с одной лишь рубрикой (единственная статья раздела) заголовок не восстанавливает."""
    _toc(tmp_path, [("Экономическое образование кадров", [("НОТ на предприятиях", "73")])])
    _page(tmp_path, "IMG_0001", "Текст лекции без заголовка.", "73")
    _page(
        tmp_path,
        "IMG_0002",
        "Продолжение лекции.",
        "74",
        running_header="Экономическое образование кадров 74",
        running_header_article_id="A1",
        running_header_rubric_id="R1",
    )
    assembly = _assemble(tmp_path)
    assert "# " not in assembly.text and "<rubric>" not in assembly.text
    assert [(a["id"], a["status"]) for a in assembly.reconcile.articles] == [("A1", "missing")]
    assert [(r["id"], r["status"]) for r in assembly.reconcile.rubrics] == [("R1", "missing")]


def test_heading_restored_from_bold_topic_and_rubric_from_marker(tmp_path):
    """Лекция: название набрано «**Тема: …**» под названием курса, шапка раздела — маркером; на странице по оглавлению оба восстанавливаются."""
    _toc(tmp_path, [("Экономическое образование кадров", [("Технические средства управления", "75")])])
    _page(
        tmp_path,
        "IMG_0001",
        "<marker>*ЭКОНОМИЧЕСКОЕ ОБРАЗОВАНИЕ КАДРОВ*</marker>\n\n## ОСНОВЫ ЭКОНОМИКИ И УПРАВЛЕНИЯ\n\n"
        "**Тема: Технические средства управления**\n\nСовершенствование системы.",
        None,
    )
    _page(tmp_path, "IMG_0002", "Продолжение.\n\n**Тема: Технические средства управления** — повтор в тексте.", "76")
    _page(tmp_path, "IMG_0003", "Ещё.", "77")
    # Курсив, подчёркивание и `###` тоже годятся в название — на своей странице.
    _toc(tmp_path, [(None, [("Технические средства управления", "75"), ("Курсивная", "76"), ("Третья", "77")])])
    _page(tmp_path, "IMG_0002", "Продолжение.\n\n*Курсивная*\n\nТекст.", "76")
    _page(tmp_path, "IMG_0003", "### Третья\n\nЕщё.", "77")
    assembly = _assemble(tmp_path)
    assert "<!-- article A2 -->\n\n# Курсивная\n\nТекст." in assembly.text
    assert "<!-- article A3 -->\n\n# Третья\n\nЕщё." in assembly.text
    _toc(tmp_path, [("Экономическое образование кадров", [("Технические средства управления", "75")])])
    _page(tmp_path, "IMG_0002", "Продолжение.\n\n**Тема: Технические средства управления** — повтор в тексте.", "76")
    _page(tmp_path, "IMG_0003", "Ещё.", "77")
    assembly = _assemble(tmp_path)
    text = assembly.text
    assert (
        "<!-- rubric R1 -->\n\n<rubric>*ЭКОНОМИЧЕСКОЕ ОБРАЗОВАНИЕ КАДРОВ*</rubric>\n\n## ОСНОВЫ ЭКОНОМИКИ И УПРАВЛЕНИЯ\n\n"
        "<!-- article A1 -->\n\n# Тема: Технические средства управления\n\nСовершенствование" in text
    ), "заголовок как напечатан, с пометкой «Тема:»"
    assert text.count("# Тема: Технические") == 1, "повтор на другой странице не тронут"
    assert [(a["id"], a["status"], a["page_number"]) for a in assembly.reconcile.articles] == [("A1", "restored", 75)]
    assert [(r["id"], r["status"]) for r in assembly.reconcile.rubrics] == [("R1", "restored")]


def test_toc_list_heading_is_not_a_candidate_for_a_lecture_title_and_body_restore_skips_headings(tmp_path):
    """`# СОДЕРЖАНИЕ` не становится статьёй «… Содержание …»; лекция «Тема 11.» восстанавливается; чужой `#` не трогается."""
    _toc(
        tmp_path,
        [
            (None, [("Первая", "1"), ("Планирование — центральное звено. Содержание планов", "5")]),
            (None, [("О разработке и внедрении систе", "7"), ("мы обеспечения объектов", "7")]),
        ],
    )
    _page(
        tmp_path,
        "IMG_0001",
        "# СОДЕРЖАНИЕ\n\n<toc>\n\n- Первая — 1\n\n</toc>",
        stage=Stage.TOC,
        toc_kind=TocKind.CONTENTS,
        toc=TocPage(TocKind.CONTENTS, False, []),
    )
    _page(tmp_path, "IMG_0002", "# Первая\n\nТекст.", "1")
    _page(tmp_path, "IMG_0003", "## Тема 11. Планирование — центральное звено. Содержание планов\n\nЛекция.", "5")
    _page(tmp_path, "IMG_0004", "# О разработке и внедрении системы обеспечения объектов\n\nТекст.", "7")
    assembly = _assemble(tmp_path)
    text = assembly.text
    assert text.count("# СОДЕРЖАНИЕ") == 1 and "<!-- article A2 -->\n\n# СОДЕРЖАНИЕ" not in text
    assert "<!-- article A2 -->\n\n# Тема 11. Планирование — центральное звено. Содержание планов\n\nЛекция." in text
    assert text.count("# О разработке") == 1 and "# # О разработке" not in text
    statuses = {a["id"]: a["status"] for a in assembly.reconcile.articles}
    assert statuses == {"A1": "found", "A2": "restored", "A3": "found", "A4": "skipped"}, "хвост названия — skipped"


def test_loose_title_match_only_in_toc_page_window_and_marker_section_becomes_heading(tmp_path):
    """Похожесть 0.6–0.75 принимается только на странице статьи (±1) и без второго близкого кандидата; маркер-раздел → `#`."""
    _toc(tmp_path, [(None, [("На главном направлении", "3"), ("Информация", "6"), ("Двоякая роль", "8")])])
    _page(tmp_path, "IMG_0001", "## НА ОДНОМ ИЗ ГЛАВНЫХ НАПРАВЛЕНИЙ\n\nПовтор не на своей странице.", "1")
    _page(tmp_path, "IMG_0003", "<marker>*АСУ МТС*</marker>\n\n## НА ОДНОМ ИЗ ГЛАВНЫХ НАПРАВЛЕНИЙ\n\nТекст.", "3")
    _page(tmp_path, "IMG_0006", "<marker>*ИНФОРМАЦИЯ*</marker>\n\n## У нас в гостях\n\nЗаметка.", "6")
    # Два кандидата почти одинаковой похожести — неоднозначно, не восстанавливать.
    _page(tmp_path, "IMG_0008", "## Двоякая позиция\n\nТекст.\n\n## Двоякая политика\n\nЕщё.", "8")
    assembly = _assemble(tmp_path)
    text = assembly.text
    assert "<!-- article A1 -->\n\n# НА ОДНОМ ИЗ ГЛАВНЫХ НАПРАВЛЕНИЙ\n\nТекст." in text
    assert text.count("\n# НА ОДНОМ") == 1 and "## НА ОДНОМ ИЗ ГЛАВНЫХ НАПРАВЛЕНИЙ\n\nПовтор" in text
    assert "<!-- article A2 -->\n\n# ИНФОРМАЦИЯ\n\n## У нас в гостях" in text and "<marker>*ИНФОРМАЦИЯ*" not in text
    assert "\n# Двоякая" not in text, "два близких кандидата — не восстанавливать"
    by_id = {a["id"]: a for a in assembly.reconcile.articles}
    assert (by_id["A1"]["status"], by_id["A1"]["source"]) == ("restored", "body_loose")
    assert (by_id["A2"]["status"], by_id["A2"]["source"]) == ("restored", "marker")
    assert by_id["A3"]["status"] == "missing"
