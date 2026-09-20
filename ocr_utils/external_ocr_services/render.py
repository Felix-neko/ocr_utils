"""Разложить PageResult в .md: YAML-шапка с полями полосы и тело в markdown; переводы строк у тегов автора."""

from __future__ import annotations

import json
import re

from ocr_utils.external_ocr_services.schema import PageResult, StructureTag

# Теги автора и должности: подряд идущие строки `<author>…</author>` / `<position>…</position>`,
# разделённые одинарным переводом строки, вьюер markdown склеивает в одну строку.
_AUTHOR_TAG = re.compile(rf"</?(?:{StructureTag.AUTHOR}|{StructureTag.POSITION})>")


def space_author_tags(text: str) -> str:
    """Удвоить одинарный перевод строки перед `<author>`/`<position>` и после `</author>`/`</position>`.

    Один проход слева направо по уже изменённому тексту: двойной перевод строки не трогается, и
    если удвоение у предыдущего тега сделало перевод у соседнего двойным, второй раз он не
    удваивается; любой другой символ рядом с тегом (в том числе пробел) не трогается. Идемпотентно.

    Args:
        text: Тело полосы или блок выпуска в markdown.

    Returns:
        Текст с пустой строкой между соседними тегами автора и должности.
    """
    parts: list[str] = []
    position = 0
    for match in _AUTHOR_TAG.finditer(text):
        parts.append(text[position : match.start()])
        current = "".join(parts)  # что уже собрано — чтобы видеть удвоение у предыдущего тега
        if not match.group(0).startswith("</") and current.endswith("\n") and not current.endswith("\n\n"):
            parts.append("\n")
        parts.append(match.group(0))
        position = match.end()
        if (
            match.group(0).startswith("</")
            and text.startswith("\n", position)
            and not text.startswith("\n\n", position)
        ):
            parts.append("\n")
    parts.append(text[position:])
    return "".join(parts)


def _yaml_value(value: object) -> str:
    """Скаляр для YAML-шапки: ``null``, ``true``/``false`` или строка в JSON-кавычках.

    JSON-строка — валидный YAML-скаляр, а экранирование кавычек и переносов достаётся даром.

    Args:
        value: Значение поля ``PageResult`` (строка, число, bool, StrEnum или ``None``).
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(str(value), ensure_ascii=False)


def _ref_line(ref: dict, id_key: str) -> str:
    """Запись ``headings``/``rubrics`` одной строкой шапки: «A3: Название» или «?: Название» без id.

    Args:
        ref: ``{"text", id_key}``.
        id_key: ``article_id`` / ``rubric_id``.
    """
    return f"{ref.get(id_key) or '?'}: {ref.get('text', '')}"


def to_markdown(result: PageResult) -> str:
    """Текст файла ``.md`` полосы: шапка с полями, затем тело; в конце ровно один перевод строки.

    Args:
        result: Разобранный и доведённый пост-обработкой ответ модели.
    """
    # В шапку — поля полосы, которые нужны при сборке выпуска и просмотре глазами; поля
    # повреждений и авторов с привязкой остаются в .json.
    head = [
        "---",
        f"page_number: {_yaml_value(result.page_number)}",
        f"running_header: {_yaml_value(result.running_header)}",
        f"running_footer: {_yaml_value(result.running_footer)}",
        f"toc_kind: {_yaml_value(result.toc_kind)}",
        f"rubric: {_yaml_value(result.rubric)}",
        f"title: {_yaml_value(result.title)}",
        f"notes: {_yaml_value(result.notes)}",
    ]
    # Id статей у `#` и рубрик у `<rubric>` (v20) — JSON-списки строк «A3: Название», они же валидный YAML.
    if result.headings:
        head.append(
            f"headings: {json.dumps([_ref_line(h, 'article_id') for h in result.headings], ensure_ascii=False)}"
        )
    if result.rubrics:
        head.append(f"rubrics: {json.dumps([_ref_line(r, 'rubric_id') for r in result.rubrics], ensure_ascii=False)}")
    for key in (
        "running_header_article_id",
        "running_header_rubric_id",
        "running_footer_article_id",
        "running_footer_rubric_id",
    ):
        if getattr(result, key):
            head.append(f"{key}: {_yaml_value(getattr(result, key))}")
    # Замечания пост-обработки — только если есть: JSON-список строк, он же валидный YAML.
    if result.messages:
        head.append(f"messages: {json.dumps(result.messages, ensure_ascii=False)}")
    head += ["---", ""]
    return "\n".join(head) + space_author_tags(result.content_markdown.rstrip()) + "\n"
