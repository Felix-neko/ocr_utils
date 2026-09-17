"""Ответ модели на одну полосу: поля, JSON-схема по этапу и терпимый разбор.

Разбор терпимый не от хорошей жизни: модели оборачивают JSON в ```-ограждения, дописывают
«Вот результат:» перед ним, а после — эхо ``response_format``. Что можно вытащить —
вытаскиваем, остальное считаем сбоем разбора и сохраняем сырой текст.

Два этапа — две схемы. ``page`` (обычная полоса): текст, структура статьи, признак «а не
оглавление ли это» (``toc_kind``) и поля повреждений. ``toc`` (полоса оглавления или указателя):
то же плюс структурированное оглавление ``toc`` — секции по рубрикам со статьями.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

# Виды полосы по мнению модели; совпадают с видами детектора ``scan_markup.toc``.
TOC_KINDS = ("none", "contents", "index")
STAGES = ("page", "toc")

# Чья это подпись: автор статьи, начинающейся на полосе; подпись под концом статьи; статья идёт
# через полосу (имя в колонтитуле или повтор). По этому полю пост-обработка (``structure``)
# ставит ``<author>`` после ``#`` или оставляет под последним абзацем.
AUTHOR_ARTICLE = ("starts_here", "ends_here", "continues")
# Где имя напечатано: над заголовком, рядом с ним, под ним, в конце текста, в колонтитуле.
AUTHOR_PRINTED = ("above_title", "beside_title", "below_title", "end_of_text", "running_header")


class ParseError(ValueError):
    """Ответ модели не удалось привести к PageResult."""


@dataclass
class TocArticle:
    """Статья в оглавлении: название, авторы, номер страницы и (в указателе) номер выпуска."""

    title: str
    authors: list[dict] = field(default_factory=list)  # [{"name": …, "position": … | None}]
    page: str | None = None
    issue: str | None = None


@dataclass
class TocSection:
    """Секция оглавления: рубрика (``None`` — без рубрики) и её статьи по порядку."""

    rubric: str | None
    articles: list[TocArticle] = field(default_factory=list)


@dataclass
class TocPage:
    """Структурированное оглавление ОДНОЙ полосы; слияние полос выпуска — в ``toc.py``."""

    kind: str  # contents | index
    continues_previous: bool = False  # полоса начинается продолжением списка без заголовка
    sections: list[TocSection] = field(default_factory=list)


@dataclass
class PageResult:
    content_markdown: str
    page_number: str | None = None
    running_header: str | None = None
    running_footer: str | None = None
    toc_kind: str = "none"
    notes: str = ""
    # Полоса начинается серединой статьи с предыдущей полосы — для сборки выпуска.
    continues_previous: bool = False
    # Структура статьи: рубрика, заголовок статьи, начинающейся на полосе, авторы с должностями —
    # дубль того, что в теле стоит в тегах <rubric>/<author>/<position>. ``title_in_list`` —
    # совпал ли заголовок с переданным списком статей (None — списка не было).
    rubric: str | None = None
    title: str | None = None
    title_in_list: bool | None = None
    # [{"name", "position", "article": AUTHOR_ARTICLE | None, "printed": AUTHOR_PRINTED | None}]
    authors: list[dict] = field(default_factory=list)
    # Повреждения: булев вердикт (надёжный триггер второго прохода: в тексте ``damage`` модель
    # пишет «повреждений не обнаружено» на 4 из 5 чистых полос), описание глазами модели, слова с
    # <restored> и <fuzzy>, число <unknown/>, записи по повреждённым строкам.
    damaged: bool = False
    damage: str = ""
    restored: list[str] = field(default_factory=list)
    fuzzy: list[str] = field(default_factory=list)
    unknown: int = 0
    edge_words: list[dict] = field(default_factory=list)  # [{seen, full, kind}]
    # Только у этапа toc.
    toc: TocPage | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)


_AUTHOR_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "position": {"type": ["string", "null"]}},
    "required": ["name", "position"],
    "additionalProperties": False,
}

# Автор на обычной полосе: плюс привязка к статье и место печати.
_PAGE_AUTHOR_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "position": {"type": ["string", "null"]},
        "article": {
            "type": "string",
            "enum": list(AUTHOR_ARTICLE),
            "description": "starts_here — author of the article that starts on this page; ends_here — signature under "
            "the end of an article; continues — the article runs across this page (name in the running header).",
        },
        "printed": {"type": "string", "enum": list(AUTHOR_PRINTED), "description": "Where the name is printed."},
    },
    "required": ["name", "position", "article", "printed"],
    "additionalProperties": False,
}

_TOC_SCHEMA = {
    "type": "object",
    "description": "Structured table of contents of THIS page only.",
    "properties": {
        "kind": {"type": "string", "enum": ["contents", "index"]},
        "continues_previous": {
            "type": "boolean",
            "description": "True if the page starts in the middle of the list, without the TOC heading.",
        },
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rubric": {
                        "type": ["string", "null"],
                        "description": "Rubric heading; null for entries without one.",
                    },
                    "articles": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "authors": {"type": "array", "items": _AUTHOR_SCHEMA},
                                "page": {"type": ["string", "null"]},
                                "issue": {
                                    "type": ["string", "null"],
                                    "description": "Issue number in an annual index.",
                                },
                            },
                            "required": ["title", "authors", "page", "issue"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["rubric", "articles"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["kind", "continues_previous", "sections"],
    "additionalProperties": False,
}


def json_schema(stage: str = "page") -> dict:
    """Строгая схема ответа этапа: additionalProperties=false и полный required, как требует strict-режим."""
    if stage not in STAGES:
        raise ValueError(f"неизвестный этап {stage!r}")
    properties: dict = {
        "damaged": {
            "type": "boolean",
            "description": "True only if some letters on the page are hidden, cut off, blurred, squashed, smeared or "
            "washed out; false for a cleanly printed page.",
        },
        "damage": {"type": "string", "description": "What damage is visible on the page; empty string if none."},
        "page_number": {"type": ["string", "null"], "description": 'Page number as printed, e.g. "12"; null if none.'},
        "rubric": {"type": ["string", "null"], "description": "Rubric printed above the title, without tags."},
        "title": {"type": ["string", "null"], "description": "Title of the article starting on this page."},
        "title_in_list": {
            "type": ["boolean", "null"],
            "description": "Whether the title matches an article from the given list; null if no list was given.",
        },
        "authors": {"type": "array", "items": _PAGE_AUTHOR_SCHEMA, "description": "Every author named on the page."},
        "continues_previous": {
            "type": "boolean",
            "description": "True if the page starts in the middle of an article that began on a previous page.",
        },
        "running_header": {"type": ["string", "null"]},
        "running_footer": {"type": ["string", "null"]},
        "toc_kind": {
            "type": "string",
            "enum": list(TOC_KINDS),
            "description": "contents — the issue's table of contents; index — annual index of articles; none otherwise.",
        },
        "content_markdown": {"type": "string", "description": "Full body of the page as Markdown, per the rules."},
        "restored": {"type": "array", "items": {"type": "string"}},
        "fuzzy": {"type": "array", "items": {"type": "string"}},
        "unknown": {"type": "integer"},
        "edge_words": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "seen": {"type": "string"},
                    "full": {"type": "string"},
                    "kind": {"type": "string", "enum": ["hidden", "fuzzy", "unknown"]},
                },
                "required": ["seen", "full", "kind"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string", "description": "Uncertainties; empty string if none."},
    }
    if stage == "toc":
        properties["toc"] = json.loads(json.dumps(_TOC_SCHEMA))
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


# Разрядка: «П р и м е ч а н и е» — три и более букв через одиночные пробелы. Модели переносят её
# буквально; читать такое неудобно, а для поиска слово должно быть словом.
_SPACED_WORD = re.compile(r"(?<![А-ЯЁа-яёA-Za-z])([А-ЯЁа-яёA-Za-z](?: [А-ЯЁа-яёA-Za-z]){2,})(?![А-ЯЁа-яёA-Za-z])")


def unspace_letters(text: str) -> str:
    """Склеить слова, набранные вразрядку, и выделить их курсивом: «П р и м е ч а н и е» → «*Примечание*»."""

    def join(match: re.Match) -> str:
        word = match.group(1).replace(" ", "")
        before = text[max(0, match.start() - 2) : match.start()]
        if before.endswith(("*", "_")):
            return word
        return f"*{word}*"

    return _SPACED_WORD.sub(join, text)


DAMAGE_TAGS = ("restored", "fuzzy")
# Теги структуры: рубрика статьи (только из оглавления), автор, должность и маркер — текстовый
# элемент вне потока статьи (маркер раздела в углу, девиз, рубрика не из оглавления).
STRUCTURE_TAGS = ("rubric", "author", "position", "marker")
_UNKNOWN = re.compile(r"<unknown\s*/>")


def tag_counts(text: str) -> dict[str, int]:
    """Сколько в тексте пар <restored>, <fuzzy> и маркеров <unknown/>."""
    counts = {name: len(re.findall(rf"<{name}>.*?</{name}>", text, re.DOTALL)) for name in DAMAGE_TAGS}
    counts["unknown"] = len(_UNKNOWN.findall(text))
    return counts


def unbalanced_tags(text: str) -> list[str]:
    """Имена тегов, у которых число открывающих и закрывающих не совпало."""
    return [name for name in DAMAGE_TAGS if text.count(f"<{name}>") != text.count(f"</{name}>")]


_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def _strip_fence(text: str) -> str:
    match = _FENCE.match(text)
    return match.group(1) if match else text


def _text_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _enum_or_none(value: object, allowed: tuple[str, ...]) -> str | None:
    text = str(value or "").strip().lower()
    return text if text in allowed else None


def _authors(items: object, with_article: bool = False) -> list[dict]:
    """Авторы; на обычной полосе — с привязкой к статье и местом печати (терпимо к их отсутствию)."""
    authors = []
    for item in items or []:
        if not isinstance(item, dict) or not str(item.get("name") or "").strip():
            continue
        author = {"name": str(item.get("name")).strip(), "position": _text_or_none(item.get("position"))}
        if with_article:
            author["article"] = _enum_or_none(item.get("article"), AUTHOR_ARTICLE)
            author["printed"] = _enum_or_none(item.get("printed"), AUTHOR_PRINTED)
        authors.append(author)
    return authors


# «Повреждений не обнаружено» и его варианты: модель пишет это в ``damage`` вместо пустой строки.
# Смотрим только на первое предложение: «обрезки нет, но размыт край» — уже повреждение.
_NO_DAMAGE = re.compile(
    r"не\s+обнаружен|нет\s+повреждени|повреждени[йя]\s+нет|отсутству|без\s+повреждени|"
    r"no\s+(visible\s+)?damage|page\s+is\s+clean|cleanly\s+printed|^\s*(none|нет)\s*$",
    re.IGNORECASE,
)


def is_no_damage_text(text: str) -> bool:
    """Пустое описание или явное «повреждений нет» в первом предложении."""
    first = re.split(r"[.;]", text.strip(), maxsplit=1)[0]
    return not first.strip() or _NO_DAMAGE.search(first) is not None


def _damaged(payload: dict) -> bool:
    """Булев вердикт; если модель его не дала — по тексту описания."""
    value = payload.get("damaged")
    if isinstance(value, bool):
        return value
    return not is_no_damage_text(str(payload.get("damage") or ""))


def _coerce_toc(payload: object) -> TocPage | None:
    if not isinstance(payload, dict):
        return None
    kind = str(payload.get("kind") or "contents").strip().lower()
    if kind not in ("contents", "index"):
        kind = "contents"
    sections: list[TocSection] = []
    for raw in payload.get("sections") or []:
        if not isinstance(raw, dict):
            continue
        articles = [
            TocArticle(
                title=str(item.get("title")).strip(),
                authors=_authors(item.get("authors")),
                page=_text_or_none(item.get("page")),
                issue=_text_or_none(item.get("issue")),
            )
            for item in (raw.get("articles") or [])
            if isinstance(item, dict) and str(item.get("title") or "").strip()
        ]
        sections.append(TocSection(rubric=_text_or_none(raw.get("rubric")), articles=articles))
    return TocPage(kind=kind, continues_previous=bool(payload.get("continues_previous", False)), sections=sections)


def _coerce(payload: dict, stage: str) -> PageResult:
    if not isinstance(payload, dict) or "content_markdown" not in payload:
        raise ParseError("в JSON нет поля content_markdown")
    body = payload.get("content_markdown")
    if not isinstance(body, str):
        raise ParseError("content_markdown не строка")
    toc_kind = str(payload.get("toc_kind") or "").strip().lower()
    if toc_kind not in TOC_KINDS:
        # Старое булево поле стенда — на случай, если модель ответила по памяти.
        toc_kind = "contents" if payload.get("is_toc") else "none"
    title_in_list = payload.get("title_in_list")
    return PageResult(
        content_markdown=unspace_letters(body),
        page_number=_text_or_none(payload.get("page_number")),
        running_header=_text_or_none(payload.get("running_header")),
        running_footer=_text_or_none(payload.get("running_footer")),
        toc_kind=toc_kind,
        notes=str(payload.get("notes") or ""),
        rubric=_text_or_none(payload.get("rubric")),
        title=_text_or_none(payload.get("title")),
        title_in_list=None if title_in_list is None else bool(title_in_list),
        continues_previous=bool(payload.get("continues_previous", False)),
        authors=_authors(payload.get("authors"), with_article=True),
        damaged=_damaged(payload),
        damage=str(payload.get("damage") or ""),
        restored=[str(item) for item in (payload.get("restored") or []) if str(item).strip()],
        fuzzy=[str(item) for item in (payload.get("fuzzy") or []) if str(item).strip()],
        unknown=int(payload.get("unknown") or 0),
        edge_words=[item for item in (payload.get("edge_words") or []) if isinstance(item, dict) and item.get("full")],
        toc=_coerce_toc(payload.get("toc")) if stage == "toc" else None,
    )


def parse_json_text(text: str, stage: str = "page") -> PageResult:
    """JSON из ответа: как есть, без ограждений или первый ``{...}`` в тексте."""
    if not text or not text.strip():
        raise ParseError("пустой ответ")
    candidates = [text, _strip_fence(text)]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
        # Первый объект, что бы ни шло следом: модели дописывают после JSON второй объект или эхо.
        candidates.append(text[start:])
    last_error: Exception | None = None
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            payload, _ = decoder.raw_decode(candidate.lstrip())
            return _coerce(payload, stage)
        except (json.JSONDecodeError, ParseError) as error:
            last_error = error
    raise ParseError(f"невалидный JSON: {last_error}")


_TAG_STRIP = re.compile(r"</?(restored|fuzzy)>|<unknown\s*/>")


def tags_from_edge_words(body: str, edge_words: list[dict]) -> tuple[str, int]:
    """Расставить теги по списку ``edge_words`` там, где модель в тексте их не поставила.

    Модель охотнее заполняет список повреждённых строк, чем ставит теги в тексте. Для записи с
    ``kind=hidden`` невидимая часть — это ``full`` минус видимый фрагмент ``seen`` (с начала или с
    конца слова); первое вхождение голого ``full`` заменяется на слово с ``<restored>``. ``fuzzy``
    — то же с ``<fuzzy>``; ``unknown`` — ``seen`` + ``<unknown/>``. Возвращает текст и число вставок.
    """
    inserted = 0
    for item in edge_words:
        full = _TAG_STRIP.sub("", str(item.get("full") or "")).strip()
        seen = _TAG_STRIP.sub("", str(item.get("seen") or "")).strip()
        kind = str(item.get("kind") or "hidden")
        # «адми-» → «административные» — обычный перенос, а не срез.
        if not full or full not in body or seen.endswith(("-", "­")):
            continue
        tagged = None
        if kind == "unknown":
            tagged = (seen + "<unknown/>") if full.startswith(seen) else ("<unknown/>" + seen)
        elif seen and full.startswith(seen) and len(full) > len(seen):
            tag = "restored" if kind == "hidden" else "fuzzy"
            tagged = f"{seen}<{tag}>{full[len(seen):]}</{tag}>"
        elif seen and full.endswith(seen) and len(full) > len(seen):
            tag = "restored" if kind == "hidden" else "fuzzy"
            tagged = f"<{tag}>{full[: len(full) - len(seen)]}</{tag}>{seen}"
        elif kind == "fuzzy":
            tagged = f"<fuzzy>{full}</fuzzy>"
        if not tagged:
            continue
        pattern = re.compile(r"(?<![\w<>/])" + re.escape(full) + r"(?![\w<])")
        match = pattern.search(body)
        if match is None:
            continue
        before = body[max(0, match.start() - 12) : match.start()]
        if "<restored>" in before or "<fuzzy>" in before or "<unknown" in before:
            continue
        body = body[: match.start()] + tagged + body[match.end() :]
        inserted += 1
    return body, inserted
