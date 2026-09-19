"""Раскладка md полосы: переводы строк у тегов автора и должности (вьюер markdown склеивал строки)."""

from __future__ import annotations

from ocr_utils.external_ocr_services.render import space_author_tags, to_markdown
from ocr_utils.external_ocr_services.schema import PageResult


def test_space_author_tags_doubles_single_newlines_only():
    body = (
        "# Заголовок\n"
        "<author>**С. КОЛМАКОВ**</author>\n"
        "<position>*начальник управления*</position>\n"
        "<author>**В. ОДЕСС**</author>\n"
        "<position>*зам. начальника*</position>\n"
        "Текст."
    )
    out = space_author_tags(body)
    assert out == (
        "# Заголовок\n\n"
        "<author>**С. КОЛМАКОВ**</author>\n\n"
        "<position>*начальник управления*</position>\n\n"
        "<author>**В. ОДЕСС**</author>\n\n"
        "<position>*зам. начальника*</position>\n\n"
        "Текст."
    )
    assert space_author_tags(out) == out, "идемпотентно: удвоенные переводы не удваиваются снова"
    # Двойной перевод не трогается; пробел и другие символы рядом с тегом — тоже.
    assert space_author_tags("а\n\n<author>б</author>\n\nв") == "а\n\n<author>б</author>\n\nв"
    assert (
        space_author_tags("- <author>б</author> в\n<position>г</position>")
        == "- <author>б</author> в\n\n<position>г</position>"
    )
    assert space_author_tags("<author>б</author>") == "<author>б</author>"
    assert space_author_tags("<rubric>*Р*</rubric>\n# З") == "<rubric>*Р*</rubric>\n# З", "другие теги не в счёт"


def test_to_markdown_applies_spacing():
    result = PageResult(content_markdown="# З\n<author>**И. Иванов**</author>\n<position>*инженер*</position>\nТекст.")
    md = to_markdown(result)
    assert md.endswith("# З\n\n<author>**И. Иванов**</author>\n\n<position>*инженер*</position>\n\nТекст.\n")
