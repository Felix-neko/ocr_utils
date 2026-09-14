"""Ответ модели на одну полосу: поля, JSON-схема и терпимый разбор.

Разбор терпимый не от хорошей жизни: модели оборачивают JSON в ```-ограждения, дописывают
«Вот результат:» перед ним, а на длинных строках с таблицами ломают экранирование. Что
можно вытащить — вытаскиваем, остальное считаем сбоем разбора и сохраняем сырой текст.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

FIELDS = ("page_number", "running_header", "running_footer", "is_toc", "content_markdown", "notes")


class ParseError(ValueError):
    """Ответ модели не удалось привести к PageResult."""


@dataclass
class PageResult:
    content_markdown: str
    page_number: str | None = None
    running_header: str | None = None
    running_footer: str | None = None
    is_toc: bool = False
    notes: str = ""
    restored: list[str] = field(default_factory=list)  # слова с достроенными буквами (режим restore)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)


# Строгая схема для response_format=json_schema. additionalProperties=false и полный
# required — требование strict-режима у OpenAI-совместимых провайдеров.
def json_schema(restore: bool = False) -> dict:
    """Схема ответа; в режиме restore — с полем restored (strict требует полного required)."""
    schema = json.loads(json.dumps(JSON_SCHEMA))
    if restore:
        schema["properties"]["restored"] = {
            "type": "array",
            "items": {"type": "string"},
            "description": "Every word containing reconstructed characters; empty if none.",
        }
        schema["required"].append("restored")
    return schema


JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "page_number": {
            "type": ["string", "null"],
            "description": 'Page number as printed on the page, e.g. "12"; null if none is printed.',
        },
        "running_header": {
            "type": ["string", "null"],
            "description": "Running header text printed in the top margin (journal name, issue, rubric); null if none.",
        },
        "running_footer": {
            "type": ["string", "null"],
            "description": "Running footer text printed in the bottom margin; null if none.",
        },
        "is_toc": {"type": "boolean", "description": "True if this page is the issue's table of contents."},
        "content_markdown": {"type": "string", "description": "Full body of the page as Markdown, per the rules."},
        "notes": {"type": "string", "description": "Uncertainties: unreadable areas, doubts. Empty string if none."},
    },
    "required": list(FIELDS),
    "additionalProperties": False,
}

# Разрядка: «П р и м е ч а н и е» — три и более букв через одиночные пробелы. Модели и
# особенно локальные движки переносят её буквально; читать такое неудобно, а для поиска
# и слияния с FineReader слово должно быть словом.
_SPACED_WORD = re.compile(r"(?<![А-ЯЁа-яёA-Za-z])([А-ЯЁа-яёA-Za-z](?: [А-ЯЁа-яёA-Za-z]){2,})(?![А-ЯЁа-яёA-Za-z])")


def unspace_letters(text: str) -> str:
    """Склеить слова, набранные вразрядку, и выделить их курсивом: «П р и м е ч а н и е» → «*Примечание*».

    Уже курсивные или жирные («*П р и м е ч а н и е*») просто склеиваются.
    """

    def join(match: re.Match) -> str:
        word = match.group(1).replace(" ", "")
        start = match.start()
        before = text[max(0, start - 2) : start]
        if before.endswith(("*", "_")):
            return word
        return f"*{word}*"

    return _SPACED_WORD.sub(join, text)


_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def _strip_fence(text: str) -> str:
    match = _FENCE.match(text)
    return match.group(1) if match else text


def _coerce(payload: dict) -> PageResult:
    if not isinstance(payload, dict) or "content_markdown" not in payload:
        raise ParseError("в JSON нет поля content_markdown")
    body = payload.get("content_markdown")
    if not isinstance(body, str):
        raise ParseError("content_markdown не строка")

    def text_or_none(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    return PageResult(
        content_markdown=unspace_letters(body),
        page_number=text_or_none(payload.get("page_number")),
        running_header=text_or_none(payload.get("running_header")),
        running_footer=text_or_none(payload.get("running_footer")),
        is_toc=bool(payload.get("is_toc", False)),
        notes=str(payload.get("notes") or ""),
        restored=[str(item) for item in (payload.get("restored") or []) if str(item).strip()],
    )


def parse_json_text(text: str) -> PageResult:
    """JSON из ответа: как есть, без ограждений или первый ``{...}`` в тексте."""
    if not text or not text.strip():
        raise ParseError("пустой ответ")
    candidates = [text, _strip_fence(text)]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
        # Первый объект, что бы ни шло следом: модели дописывают после JSON второй объект,
        # эхо response_format или комментарий.
        candidates.append(text[start:])
    last_error: Exception | None = None
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            payload, _ = decoder.raw_decode(candidate.lstrip())
            return _coerce(payload)
        except (json.JSONDecodeError, ParseError) as error:
            last_error = error
    raise ParseError(f"невалидный JSON: {last_error}")


_FRONT_MATTER = re.compile(r"^\s*---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
_YAML_LINE = re.compile(r"^([a-z_]+):\s*(.*)$")


def _yaml_scalar(raw: str) -> object:
    value = raw.strip()
    if value in ("", "null", "~", "None"):
        return None
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def parse_markdown_text(text: str) -> PageResult:
    """Markdown с YAML-шапкой (режим --output-mode markdown). Шапка плоская: ключ: значение."""
    if not text or not text.strip():
        raise ParseError("пустой ответ")
    text = _strip_fence(text)
    match = _FRONT_MATTER.match(text)
    if not match:
        # Без шапки — весь текст считаем телом: это хуже, чем сбой, но не потеря страницы.
        return PageResult(content_markdown=text.strip(), notes="ответ без YAML-шапки")
    fields: dict[str, object] = {}
    for line in match.group(1).splitlines():
        pair = _YAML_LINE.match(line.strip())
        if pair:
            fields[pair.group(1)] = _yaml_scalar(pair.group(2))
    fields["content_markdown"] = match.group(2).strip()
    return _coerce(fields)
