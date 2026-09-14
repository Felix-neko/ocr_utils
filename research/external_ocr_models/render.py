"""Разложить PageResult в .md: YAML-шапка с полями страницы и тело в markdown."""

from __future__ import annotations

import json

from research.external_ocr_models.schema import PageResult


def _yaml_value(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(str(value), ensure_ascii=False)


def to_markdown(result: PageResult) -> str:
    head = [
        "---",
        f"page_number: {_yaml_value(result.page_number)}",
        f"running_header: {_yaml_value(result.running_header)}",
        f"running_footer: {_yaml_value(result.running_footer)}",
        f"is_toc: {_yaml_value(result.is_toc)}",
        f"notes: {_yaml_value(result.notes)}",
        "---",
        "",
    ]
    return "\n".join(head) + result.content_markdown.rstrip() + "\n"
