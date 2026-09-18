"""Разложить PageResult в .md: YAML-шапка с полями полосы и тело в markdown."""

from __future__ import annotations

import json

from ocr_utils.external_ocr_services.schema import PageResult


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
        "---",
        "",
    ]
    return "\n".join(head) + result.content_markdown.rstrip() + "\n"
