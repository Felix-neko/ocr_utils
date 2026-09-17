"""Пост-обработка markdown: `#` только из списка, авторы при своей статье, рубрика перед `#`, идемпотентность."""

from ocr_utils.external_ocr_services.structure import (
    wrap_bare_math,
    drop_leaked_titles,
    place_rubrics,
    StructureReport,
    apply,
    demote_unlisted_headings,
    place_authors,
    rubric_from_heading,
    title_matches,
)

TITLES = [
    "Почему возникают сверхнормативные запасы?",
    "Фильм о снабжении",
    "Улучшать качество и ассортимент химических волокон",
]


def test_title_matches_tolerates_case_wrapping_and_small_changes():
    assert title_matches("ПОЧЕМУ ВОЗНИКАЮТ СВЕРХНОРМАТИВНЫЕ ЗАПАСЫ", TITLES)
    assert title_matches("Фильм о снабжении", TITLES)
    assert title_matches("Улучшать качество и ассортимент химволокон", TITLES)  # ratio
    assert not title_matches("ФОРМИРОВАНИЕ И РЕАЛИЗАЦИЯ ПЛАНА СНАБЖЕНИЯ", TITLES)
    assert not title_matches("", TITLES) and not title_matches("Фильм", [])


def test_demote_unlisted_headings_only_with_list():
    body = "# ФОРМИРОВАНИЕ И РЕАЛИЗАЦИЯ ПЛАНА СНАБЖЕНИЯ\n\nТекст.\n\n# Фильм о снабжении\n\nЕщё.\n"
    report = StructureReport()
    out = demote_unlisted_headings(body, TITLES, report)
    assert out.startswith("## ФОРМИРОВАНИЕ") and "\n# Фильм о снабжении\n" in out
    assert report.demoted_headings == ["ФОРМИРОВАНИЕ И РЕАЛИЗАЦИЯ ПЛАНА СНАБЖЕНИЯ"] and report.title_in_list is True
    assert demote_unlisted_headings(body, [], StructureReport()) == body, "без списка ничего не трогаем"
    only_bad = StructureReport()
    demote_unlisted_headings("# Чужой\n\nТекст.\n", TITLES, only_bad)
    assert only_bad.title_in_list is False


def _author(name, article=None, printed=None, position=None):
    return {"name": name, "position": position, "article": article, "printed": printed}


def test_starts_here_author_moves_after_heading_through_rubric():
    body = (
        "Конец предыдущей статьи.\n\n<author>**Ф. АНИСИМОВ,**</author>\n\n<position>*управляющий базой*</position>\n\n"
        "<author>**С. Демидов**</author>\n\n<rubric>*Консультация*</rubric>\n\n# Заголовок\n\nПервый абзац.\n"
    )
    authors = [_author("Ф. Анисимов", "ends_here", "end_of_text"), _author("С. Демидов", "starts_here", "above_title")]
    report = StructureReport()
    out = place_authors(body, authors, report)
    paragraphs = out.strip().split("\n\n")
    assert paragraphs == [
        "Конец предыдущей статьи.",
        "<author>**Ф. АНИСИМОВ,**</author>",
        "<position>*управляющий базой*</position>",
        "<rubric>*Консультация*</rubric>",
        "# Заголовок",
        "<author>**С. Демидов**</author>",
        "Первый абзац.",
    ]
    assert report.moved_authors == ["С. Демидов"]
    assert place_authors(out, authors, StructureReport()) == out, "идемпотентно"


def test_first_paragraph_author_moves_even_without_mark_but_not_when_ends_here():
    body = "<author>**С. Демидов**</author>\n\n# Восстановление\n\nТекст.\n"
    out = place_authors(body, [_author("С. Демидов")], StructureReport())
    assert out.startswith("# Восстановление\n\n<author>**С. Демидов**</author>\n\nТекст.")
    # Модель явно сказала «подпись прошлой статьи» — верим.
    same = place_authors(body, [_author("С. Демидов", "ends_here", "end_of_text")], StructureReport())
    assert same == body
    # Автор не из JSON вовсе — по положению.
    assert place_authors(body, [], StructureReport()).startswith("# Восстановление\n\n<author>")


def test_running_header_author_on_continuation_page_is_dropped():
    body = "<author>**А. Галицкий**</author>\n\nвсемерной экономии топлива.\n\n<author>**Кто-то**</author>\n"
    report = StructureReport()
    out = place_authors(body, [_author("А. Галицкий", "continues", "running_header")], report)
    assert out == "всемерной экономии топлива.\n\n<author>**Кто-то**</author>\n" and report.dropped_authors == [
        "А. Галицкий"
    ]


def test_two_end_signatures_stay_in_place_and_list_items_untouched():
    body = (
        "Конец первой.\n\n<author>**М. БАКАНОВ,**</author>\n\n<position>*начальник отдела тары*</position>\n\n"
        "# Фильм о снабжении\n\nТекст.\n\n<author>**И. ДУБОВСКИЙ**</author>\n"
    )
    authors = [_author("М. Баканов", "ends_here", "end_of_text"), _author("И. Дубовский", "ends_here", "end_of_text")]
    assert place_authors(body, authors, StructureReport()) == body
    toc = "- <author>**Л. Курский**</author>. Снабжение — 5\n\n# СОДЕРЖАНИЕ\n"
    assert place_authors(toc, [], StructureReport()) == toc


def test_rubric_from_heading_before_h1():
    body = "Текст.\n\n## Читатель предлагает...\n\n# Укрепить дисциплину\n\n## Обычный подзаголовок\n\nЕщё.\n"
    report = StructureReport()
    out = rubric_from_heading(body, ["Читатель предлагает"], report)
    assert "<rubric>*Читатель предлагает...*</rubric>\n\n# Укрепить" in out and "## Обычный подзаголовок" in out
    assert report.rubrics_from_headings == ["Читатель предлагает..."]
    assert rubric_from_heading(body, ["Консультация"], StructureReport()) == body


def test_apply_all_and_report():
    body = "<author>**В. Э. Дымшиц,**</author>\n\n<position>*Председатель*</position>\n\n# Материально-техническое снабжение — на уровень современных задач\n\n# Чужой заголовок\n\nТекст.\n"
    out, report = apply(
        body,
        ["Материально-техническое снабжение — на уровень современных задач"],
        [],
        [_author("В. Э. Дымшиц", "starts_here", "beside_title", "Председатель")],
    )
    assert out.split("\n\n")[:3] == [
        "# Материально-техническое снабжение — на уровень современных задач",
        "<author>**В. Э. Дымшиц,**</author>",
        "<position>*Председатель*</position>",
    ]
    assert "## Чужой заголовок" in out
    assert report.as_dict() == {
        "demoted_headings": ["Чужой заголовок"],
        "moved_authors": ["В. Э. Дымшиц"],
        "dropped_authors": [],
        "rubrics_from_headings": [],
        "moved_rubrics": [],
        "rubrics_from_toc": [],
        "rubrics_replaced": [],
        "markers_from_rubrics": [],
        "markers": 0,
        "dropped_headings": [],
        "wrapped_math": 0,
    }
    assert report.title_in_list is True


def test_rubric_is_recognised_before_heading_gets_demoted():
    body = "## Читатель предлагает...\n\n# Укрепить дисциплину\n\nТекст.\n"
    out, report = apply(body, ["Другая статья"], ["Читатель предлагает"], [])
    # Рубрика из списка перед пониженным заголовком: `#` статьи этой рубрики на полосе нет — остаётся маркером.
    assert out.startswith("<marker>*Читатель предлагает...*</marker>\n\n## Укрепить дисциплину")
    assert report.rubrics_from_headings == ["Читатель предлагает..."] and report.demoted_headings == [
        "Укрепить дисциплину"
    ]
    assert report.markers_from_rubrics == ["Читатель предлагает..."] and report.markers == 1


ARTICLES = [
    {"title": "Как бороться с рыночной стихией?", "authors": [], "rubric": "Экономика и право"},
    {"title": "Собственность и оплата труда", "authors": ["В. Ракоти"], "rubric": "Проблемы и суждения"},
    {"title": "Без рубрики", "authors": [], "rubric": None},
]
RUBRICS = ["Экономика и право", "Проблемы и суждения"]


def test_rubric_marker_at_page_bottom_moves_before_its_heading():
    body = "# Как бороться с рыночной стихией?\n\nТекст.\n\nЕщё текст.\n\n<rubric>*Экономика и право*</rubric>\n"
    report = StructureReport()
    out = place_rubrics(body, ARTICLES, report, RUBRICS)
    assert out == "<rubric>*Экономика и право*</rubric>\n\n# Как бороться с рыночной стихией?\n\nТекст.\n\nЕщё текст.\n"
    assert report.moved_rubrics == ["Экономика и право"] and not report.rubrics_from_toc
    assert place_rubrics(out, ARTICLES, StructureReport(), RUBRICS) == out, "идемпотентно"


def test_rubric_inserted_from_toc_and_printed_variant_replaced():
    body = "# Собственность и оплата труда\n\nТекст.\n"
    report = StructureReport()
    out = place_rubrics(body, ARTICLES, report, RUBRICS)
    assert out.startswith("<rubric>*Проблемы и суждения*</rubric>\n\n# Собственность")
    assert report.rubrics_from_toc == ["Проблемы и суждения"]
    # Напечатанный маркер не той рубрики: остаётся маркером на месте, а рубрика статьи приходит из оглавления.
    printed = "<rubric>*Проблемы и суждения*</rubric>\n\n# Как бороться с рыночной стихией?\n\nТекст.\n"
    report = StructureReport()
    out = place_rubrics(printed, ARTICLES, report, RUBRICS)
    assert out.split("\n\n")[:3] == [
        "<marker>*Проблемы и суждения*</marker>",
        "<rubric>*Экономика и право*</rubric>",
        "# Как бороться с рыночной стихией?",
    ]
    assert report.markers_from_rubrics == ["Проблемы и суждения"] and report.rubrics_from_toc == ["Экономика и право"]


def test_rubric_on_continuation_page_and_unknown_rubric_become_markers():
    body = "Продолжение статьи.\n\n<rubric>*Экономика и право*</rubric>\n\nЕщё абзац.\n"
    report = StructureReport()
    out = place_rubrics(body, ARTICLES, report, RUBRICS)
    assert out == "Продолжение статьи.\n\n<marker>*Экономика и право*</marker>\n\nЕщё абзац.\n"
    assert report.markers_from_rubrics == ["Экономика и право"]
    unknown = "<rubric>*Рынок: реалии и надежды*</rubric>\n\n# Без рубрики\n\nТекст.\n"
    out = place_rubrics(unknown, ARTICLES, StructureReport(), RUBRICS)
    assert out.startswith(
        "<marker>*Рынок: реалии и надежды*</marker>\n\n# Без рубрики"
    ), "не из оглавления — маркер и перед #"
    assert place_rubrics(unknown, [], StructureReport(), RUBRICS) == unknown, "без списка статей не трогаем"
    marker = "<marker>*Девиз*</marker>\n\n# Без рубрики\n\nТекст.\n"
    assert place_rubrics(marker, ARTICLES, StructureReport(), RUBRICS) == marker, "маркер модели не трогаем"


def test_two_articles_on_page_get_their_own_rubrics():
    body = (
        "# Как бороться с рыночной стихией?\n\nТекст.\n\n<rubric>*Экономика и право*</rubric>\n\n"
        "# Собственность и оплата труда\n\nТекст 2.\n"
    )
    report = StructureReport()
    out = place_rubrics(body, ARTICLES, report, RUBRICS)
    parts = out.split("\n\n")
    assert parts[0] == "<rubric>*Экономика и право*</rubric>" and parts[1].startswith("# Как бороться")
    assert parts[3] == "<rubric>*Проблемы и суждения*</rubric>" and parts[4].startswith("# Собственность")
    assert report.moved_rubrics == ["Экономика и право"] and report.rubrics_from_toc == ["Проблемы и суждения"]


def test_marker_with_toc_rubric_text_becomes_the_rubric():
    body = "# Собственность и оплата труда\n\nТекст.\n\n<marker>*Проблемы и суждения*</marker>\n"
    report = StructureReport()
    out = place_rubrics(body, ARTICLES, report, RUBRICS)
    assert out == "<rubric>*Проблемы и суждения*</rubric>\n\n# Собственность и оплата труда\n\nТекст.\n"
    assert report.moved_rubrics == ["Проблемы и суждения"] and not report.rubrics_from_toc and "<marker>" not in out


def test_leaked_title_on_continuation_page_is_dropped():
    titles = ["Как бороться с рыночной стихией?"]
    body = "<rubric>*Экономика и право*</rubric>\n\n## Как бороться с рыночной стихией? Взгляд с Петровки\n\nсегодня пришли к выводу.\n"
    report = StructureReport()
    out = drop_leaked_titles(body, titles, report)
    assert out == "<rubric>*Экономика и право*</rubric>\n\nсегодня пришли к выводу.\n" and report.dropped_headings
    keep = "Текст.\n\n## Как бороться с рыночной стихией?\n\nЕщё.\n"
    assert drop_leaked_titles(keep, titles, StructureReport()) == keep, "не в начале страницы — не трогаем"
    with_h1 = "## Как бороться с рыночной стихией?\n\n# Другая статья\n"
    assert drop_leaked_titles(with_h1, titles, StructureReport()) == with_h1, "на странице есть # — не продолжение"
    assert drop_leaked_titles(body, [], StructureReport()) == body


def test_bare_dollar_math_is_wrapped_and_tagged_math_untouched():
    body = "где $a_i$ — ресурсы; <latex>$b_j$</latex> — потребность.\n\n$$\\sum_i x_i = 1$$\n\nЦена 5 $ за тонну.\n"
    report = StructureReport()
    out = wrap_bare_math(body, report)
    assert out.startswith("где <latex>$a_i$</latex> — ресурсы; <latex>$b_j$</latex> — потребность.")
    assert "<latex>$$\\sum_i x_i = 1$$</latex>" in out and "Цена 5 $ за тонну." in out
    assert report.wrapped_math == 2 and wrap_bare_math(out, StructureReport()) == out
