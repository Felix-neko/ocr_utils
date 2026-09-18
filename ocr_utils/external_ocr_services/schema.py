"""Ответ модели на одну полосу: поля, JSON-схема по этапу и терпимый разбор.

Разбор терпимый не от хорошей жизни: модели оборачивают JSON в ```-ограждения, дописывают
«Вот результат:» перед ним, а после — эхо ``response_format``. Что можно вытащить —
вытаскиваем, остальное считаем сбоем разбора и сохраняем сырой текст.

Два этапа — две схемы. ``page`` (обычная полоса): текст, структура статьи, признак «а не
оглавление ли это» (``toc_kind``) и поля повреждений. ``toc`` (полоса оглавления или указателя):
то же плюс структурированное оглавление ``toc`` — секции по рубрикам со статьями.

Закрытые наборы значений (этап, вид полосы, привязка автора, теги) — ``StrEnum``: в коде они
сравниваются по имени, а в JSON-схему, ответ модели и файлы выхода уходят своими строковыми
значениями, так что формат файлов от этого не меняется.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum


class Stage(StrEnum):
    """Этап распознавания полосы: ``TOC`` — полоса оглавления/указателя (извлечь структуру),
    ``PAGE`` — обычная полоса (текст со списком статей выпуска в промпте)."""

    PAGE = "page"
    TOC = "toc"


class TocKind(StrEnum):
    """Вид полосы по мнению модели или базы; совпадает с видами детектора ``scan_markup.toc``.

    ``NONE`` — обычная полоса; ``CONTENTS`` — «Содержание» выпуска; ``INDEX`` — годовой указатель
    статей (декабрьские номера).
    """

    NONE = "none"
    CONTENTS = "contents"
    INDEX = "index"


# Виды, у которых есть структурированное оглавление (этап toc); ``NONE`` — не оглавление.
TOC_KINDS_WITH_TOC = (TocKind.CONTENTS, TocKind.INDEX)


class AuthorArticle(StrEnum):
    """Чья это подпись на обычной полосе; по ней пост-обработка (``structure``) ставит ``<author>``.

    ``STARTS_HERE`` — автор статьи, начинающейся на полосе (блок идёт сразу после ``#``);
    ``ENDS_HERE`` — подпись под концом статьи (остаётся под последним абзацем);
    ``CONTINUES`` — статья идёт через полосу, имя в колонтитуле или повторе (из тела убирается).
    """

    STARTS_HERE = "starts_here"
    ENDS_HERE = "ends_here"
    CONTINUES = "continues"


class AuthorPrinted(StrEnum):
    """Где на полосе напечатано имя автора: над заголовком, рядом, под ним, в конце текста, в колонтитуле."""

    ABOVE_TITLE = "above_title"
    BESIDE_TITLE = "beside_title"
    BELOW_TITLE = "below_title"
    END_OF_TEXT = "end_of_text"
    RUNNING_HEADER = "running_header"


class EdgeKind(StrEnum):
    """Вид повреждения строки в ``edge_words``: буквы скрыты/срезаны (``HIDDEN``), видны, но
    ненадёжны (``FUZZY``), не восстановимы (``UNKNOWN``)."""

    HIDDEN = "hidden"
    FUZZY = "fuzzy"
    UNKNOWN = "unknown"


class DamageTag(StrEnum):
    """Теги повреждений в тексте: парные ``<restored>…</restored>`` и ``<fuzzy>…</fuzzy>``,
    одиночный ``<unknown/>``."""

    RESTORED = "restored"
    FUZZY = "fuzzy"
    UNKNOWN = "unknown"


# Парные теги повреждений (у ``<unknown/>`` закрывающего нет).
PAIRED_DAMAGE_TAGS = (DamageTag.RESTORED, DamageTag.FUZZY)


class StructureTag(StrEnum):
    """Теги структуры: рубрика статьи (только из оглавления), автор, должность и маркер — текстовый
    элемент вне потока статьи (маркер раздела в углу, девиз, рубрика не из оглавления)."""

    RUBRIC = "rubric"
    AUTHOR = "author"
    POSITION = "position"
    MARKER = "marker"


# Формулы: LaTeX внутри тега, номер формулы снаружи текстом.
FORMULA_TAG = "latex"


class ParseError(ValueError):
    """Ответ модели не удалось привести к PageResult."""


@dataclass
class TocArticle:
    """Статья в оглавлении: название, авторы, номер страницы и (в указателе) номер выпуска.

    Args:
        title: Название статьи как напечатано в оглавлении.
        authors: ``[{"name": …, "position": … | None}]`` — авторы с должностями, если указаны.
        page: Номер страницы строкой (как напечатан); ``None`` — не указан.
        issue: Номер выпуска — только в годовом указателе; ``None`` в «Содержании».
    """

    title: str
    authors: list[dict] = field(default_factory=list)
    page: str | None = None
    issue: str | None = None


@dataclass
class TocSection:
    """Секция оглавления: рубрика (``None`` — без рубрики) и её статьи по порядку.

    Args:
        rubric: Заголовок рубрики над группой статей; ``None`` у статей вне рубрик.
        articles: Статьи секции в порядке печати.
    """

    rubric: str | None
    articles: list[TocArticle] = field(default_factory=list)


@dataclass
class TocPage:
    """Структурированное оглавление ОДНОЙ полосы; слияние полос выпуска — в ``toc.py``.

    Args:
        kind: Вид оглавления — ``CONTENTS`` или ``INDEX``.
        continues_previous: Полоса начинается продолжением списка без заголовка оглавления
            (модель ставит флаг нестрого; слияние учитывает его, но не полагается на него).
        sections: Секции полосы по порядку.
    """

    kind: TocKind
    continues_previous: bool = False
    sections: list[TocSection] = field(default_factory=list)


@dataclass
class PageResult:
    """Разобранный ответ модели на одну полосу — то, что уходит в ``.json`` и ``.md``.

    Args:
        content_markdown: Тело полосы в markdown по правилам промпта (``#`` только у статей из
            оглавления, теги ``<rubric>``/``<author>``/``<position>``/``<marker>``, теги повреждений).
        page_number: Номер страницы как напечатан; ``None`` — не напечатан.
        running_header: Колонтитул сверху (имя автора или название статьи-продолжения); ``None`` — нет.
        running_footer: Колонтитул снизу; ``None`` — нет.
        toc_kind: Что это за полоса по мнению модели: обычная, «Содержание», указатель. На обычной
            полосе не ``NONE`` — сигнал fallback «оглавление вне базы».
        notes: Сомнения модели свободным текстом; пусто — нет.
        rubric: Рубрика над заголовком статьи (без тегов); ``None`` — нет.
        title: Название статьи, начинающейся на полосе; ``None`` — полоса-продолжение.
        title_in_list: Совпал ли заголовок со списком статей в промпте; ``None`` — списка не было.
        authors: ``[{"name", "position", "article": AuthorArticle | None, "printed": AuthorPrinted | None}]``
            — все имена авторов на полосе с привязкой к статье и местом печати.
        damaged: Булев вердикт «на полосе есть повреждённые буквы» — надёжный триггер второго
            прохода (в тексте ``damage`` модель пишет «повреждений не обнаружено» на 4 из 5 чистых).
        damage: Описание повреждений глазами модели (по-русски); пусто — нет.
        restored: Слова, восстановленные по контексту (то, что стоит в ``<restored>``).
        fuzzy: Слова с ненадёжными буквами (``<fuzzy>``).
        unknown: Сколько мест не восстановить (``<unknown/>``).
        edge_words: ``[{"seen", "full", "kind": EdgeKind}]`` — записи по повреждённым строкам:
            что видно, полное слово, вид повреждения; по ним ставятся теги, если модель их не поставила.
        toc: Структурированное оглавление полосы — только у этапа ``TOC``, иначе ``None``.
    """

    content_markdown: str
    page_number: str | None = None
    running_header: str | None = None
    running_footer: str | None = None
    toc_kind: TocKind = TocKind.NONE
    notes: str = ""
    rubric: str | None = None
    title: str | None = None
    title_in_list: bool | None = None
    authors: list[dict] = field(default_factory=list)
    damaged: bool = False
    damage: str = ""
    restored: list[str] = field(default_factory=list)
    fuzzy: list[str] = field(default_factory=list)
    unknown: int = 0
    edge_words: list[dict] = field(default_factory=list)
    toc: TocPage | None = None

    def to_json(self) -> str:
        """Все поля как JSON (StrEnum сериализуются своими строковыми значениями); indent=1 — читаемый diff."""
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)


# Автор в оглавлении: имя и должность.
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
            "enum": [item.value for item in AuthorArticle],
            "description": "starts_here — author of the article that starts on this page; ends_here — signature under "
            "the end of an article; continues — the article runs across this page (name in the running header).",
        },
        "printed": {
            "type": "string",
            "enum": [item.value for item in AuthorPrinted],
            "description": "Where the name is printed.",
        },
    },
    "required": ["name", "position", "article", "printed"],
    "additionalProperties": False,
}

# Оглавление одной полосы: вид, признак продолжения, секции по рубрикам со статьями.
_TOC_SCHEMA = {
    "type": "object",
    "description": "Structured table of contents of THIS page only.",
    "properties": {
        "kind": {"type": "string", "enum": [item.value for item in TOC_KINDS_WITH_TOC]},
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


def json_schema(stage: Stage = Stage.PAGE) -> dict:
    """Строгая схема ответа этапа: additionalProperties=false и полный required, как требует strict-режим.

    Args:
        stage: ``PAGE`` — поля обычной полосы; ``TOC`` — те же плюс объект ``toc``.

    Returns:
        JSON-схема объекта ответа (``type: object`` с ``properties``/``required``), годная и для
        ``response_format`` строгого режима, и для описания в промпте.
    """
    stage = Stage(stage)  # строка из старого вызова → член перечисления; чужое значение — ValueError
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
        "running_header": {"type": ["string", "null"]},
        "running_footer": {"type": ["string", "null"]},
        "toc_kind": {
            "type": "string",
            "enum": [item.value for item in TocKind],
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
                    "kind": {"type": "string", "enum": [item.value for item in EdgeKind]},
                },
                "required": ["seen", "full", "kind"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string", "description": "Uncertainties; empty string if none."},
    }
    if stage is Stage.TOC:
        properties["toc"] = json.loads(json.dumps(_TOC_SCHEMA))  # копия: схема-константа не должна мутировать
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


# Разрядка: «П р и м е ч а н и е» — три и более букв через одиночные пробелы. Модели переносят её
# буквально; читать такое неудобно, а для поиска слово должно быть словом.
_SPACED_WORD = re.compile(r"(?<![А-ЯЁа-яёA-Za-z])([А-ЯЁа-яёA-Za-z](?: [А-ЯЁа-яёA-Za-z]){2,})(?![А-ЯЁа-яёA-Za-z])")


def _join_spaced_word(match: re.Match) -> str:
    """Замена для одного слова вразрядку: буквы склеены, курсив — если слово ещё не выделено.

    Args:
        match: Совпадение ``_SPACED_WORD``; исходная строка — ``match.string``.

    Returns:
        ``*Слово*`` или просто ``Слово``, если перед ним уже стоит ``*`` или ``_``.
    """
    word = match.group(1).replace(" ", "")
    # Слово уже внутри курсива/жирного — второй раз звёздочки не ставим.
    before = match.string[max(0, match.start() - 2) : match.start()]
    if before.endswith(("*", "_")):
        return word
    return f"*{word}*"


def unspace_letters(text: str) -> str:
    """Склеить слова, набранные вразрядку, и выделить их курсивом: «П р и м е ч а н и е» → «*Примечание*».

    Args:
        text: Тело полосы в markdown.

    Returns:
        Тот же текст со склеенными словами; остальное без изменений.
    """
    return _SPACED_WORD.sub(_join_spaced_word, text)


_UNKNOWN = re.compile(r"<unknown\s*/>")


def tag_counts(text: str) -> dict[DamageTag, int]:
    """Сколько в тексте пар ``<restored>``, ``<fuzzy>`` и маркеров ``<unknown/>``.

    Args:
        text: Тело полосы в markdown с тегами.

    Returns:
        ``{DamageTag: число}`` по всем трём тегам, нули включительно.
    """
    counts = {tag: len(re.findall(rf"<{tag}>.*?</{tag}>", text, re.DOTALL)) for tag in PAIRED_DAMAGE_TAGS}
    counts[DamageTag.UNKNOWN] = len(_UNKNOWN.findall(text))
    return counts


def unbalanced_tags(text: str) -> list[DamageTag]:
    """Парные теги повреждений, у которых число открывающих и закрывающих не совпало.

    Args:
        text: Тело полосы в markdown с тегами.

    Returns:
        Теги с непарными скобками; пустой список — разметка цела.
    """
    return [tag for tag in PAIRED_DAMAGE_TAGS if text.count(f"<{tag}>") != text.count(f"</{tag}>")]


_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def _strip_fence(text: str) -> str:
    """Снять ```-ограждение вокруг всего ответа, если оно есть; иначе текст как есть."""
    match = _FENCE.match(text)
    return match.group(1) if match else text


def _text_or_none(value: object) -> str | None:
    """Строка без краевых пробелов или ``None`` для пустого/отсутствующего значения."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _enum_or_none(value: object, allowed: type[StrEnum]) -> StrEnum | None:
    """Значение из ответа → член перечисления ``allowed`` (без регистра); чужое значение → ``None``."""
    text = str(value or "").strip().lower()
    try:
        return allowed(text)
    except ValueError:
        return None


def _authors(items: object, with_article: bool = False) -> list[dict]:
    """Авторы из ответа; на обычной полосе — с привязкой к статье и местом печати (терпимо к их отсутствию).

    Args:
        items: Значение поля ``authors`` ответа (список словарей или что-то не то).
        with_article: Разбирать ли поля ``article`` / ``printed`` (только у этапа ``PAGE``).
    """
    authors = []
    for item in items or []:
        # Запись без имени бесполезна — пропускаем; должность необязательна.
        if not isinstance(item, dict) or not str(item.get("name") or "").strip():
            continue
        author = {"name": str(item.get("name")).strip(), "position": _text_or_none(item.get("position"))}
        if with_article:
            author["article"] = _enum_or_none(item.get("article"), AuthorArticle)
            author["printed"] = _enum_or_none(item.get("printed"), AuthorPrinted)
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
    """Пустое описание или явное «повреждений нет» в первом предложении.

    Args:
        text: Поле ``damage`` ответа.
    """
    first = re.split(r"[.;]", text.strip(), maxsplit=1)[0]
    return not first.strip() or _NO_DAMAGE.search(first) is not None


def _damaged(payload: dict) -> bool:
    """Булев вердикт ``damaged``; если модель его не дала — по тексту описания (старые ответы).

    Args:
        payload: Разобранный JSON ответа.
    """
    value = payload.get("damaged")
    if isinstance(value, bool):
        return value
    return not is_no_damage_text(str(payload.get("damage") or ""))


def _coerce_toc(payload: object) -> TocPage | None:
    """Объект ``toc`` ответа → ``TocPage``; не словарь → ``None``. Статьи без названия пропускаются.

    Args:
        payload: Значение поля ``toc`` ответа этапа ``TOC``.
    """
    if not isinstance(payload, dict):
        return None
    # Неизвестный или пустой вид — считаем «Содержанием»: это самый частый случай.
    kind = _enum_or_none(payload.get("kind") or TocKind.CONTENTS, TocKind)
    if kind not in TOC_KINDS_WITH_TOC:
        kind = TocKind.CONTENTS
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


def _coerce(payload: dict, stage: Stage) -> PageResult:
    """Разобранный JSON → ``PageResult`` с приведением типов; без ``content_markdown`` — ``ParseError``.

    Args:
        payload: Объект из ответа модели (уже ``json.loads``).
        stage: Этап: у ``TOC`` дополнительно разбирается объект ``toc``.
    """
    if not isinstance(payload, dict) or "content_markdown" not in payload:
        raise ParseError("в JSON нет поля content_markdown")
    body = payload.get("content_markdown")
    if not isinstance(body, str):
        raise ParseError("content_markdown не строка")
    toc_kind = _enum_or_none(payload.get("toc_kind"), TocKind)
    if toc_kind is None:
        # Старое булево поле стенда — на случай, если модель ответила по памяти.
        toc_kind = TocKind.CONTENTS if payload.get("is_toc") else TocKind.NONE
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
        authors=_authors(payload.get("authors"), with_article=True),
        damaged=_damaged(payload),
        damage=str(payload.get("damage") or ""),
        restored=[str(item) for item in (payload.get("restored") or []) if str(item).strip()],
        fuzzy=[str(item) for item in (payload.get("fuzzy") or []) if str(item).strip()],
        unknown=int(payload.get("unknown") or 0),
        # Запись без полного слова бесполезна для расстановки тегов — пропускаем.
        edge_words=[item for item in (payload.get("edge_words") or []) if isinstance(item, dict) and item.get("full")],
        toc=_coerce_toc(payload.get("toc")) if Stage(stage) is Stage.TOC else None,
    )


def parse_json_text(text: str, stage: Stage = Stage.PAGE) -> PageResult:
    """JSON из ответа: как есть, без ограждений или первый ``{...}`` в тексте.

    Args:
        text: Сырой текст ответа модели (или содержимое сохранённого ``.json``).
        stage: Этап, по схеме которого разбирать (у ``TOC`` есть объект ``toc``).

    Returns:
        ``PageResult`` с приведёнными типами; ни один кандидат не разобрался — ``ParseError``
        с текстом последней ошибки.
    """
    if not text or not text.strip():
        raise ParseError("пустой ответ")
    # Кандидаты от самого строгого к самому терпимому; первый разобравшийся побеждает.
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

    Args:
        body: Тело полосы в markdown.
        edge_words: Записи ``{"seen", "full", "kind"}`` из ответа модели.

    Returns:
        ``(тело с расставленными тегами, число вставок)``; число уходит в meta
        (``tags_from_edge_words``), чтобы видеть, сколько тегов поставил код, а не модель.
    """
    inserted = 0
    for item in edge_words:
        # Теги внутри самих записей (модель иногда копирует их из текста) снимаем — ищем голые слова.
        full = _TAG_STRIP.sub("", str(item.get("full") or "")).strip()
        seen = _TAG_STRIP.sub("", str(item.get("seen") or "")).strip()
        kind = _enum_or_none(item.get("kind") or EdgeKind.HIDDEN, EdgeKind) or EdgeKind.HIDDEN
        # «адми-» → «административные» — обычный перенос, а не срез.
        if not full or full not in body or seen.endswith(("-", "­")):
            continue
        tagged = None
        if kind is EdgeKind.UNKNOWN:
            tagged = (seen + "<unknown/>") if full.startswith(seen) else ("<unknown/>" + seen)
        elif seen and full.startswith(seen) and len(full) > len(seen):
            # Видно начало слова — скрыт хвост.
            tag = DamageTag.RESTORED if kind is EdgeKind.HIDDEN else DamageTag.FUZZY
            tagged = f"{seen}<{tag}>{full[len(seen):]}</{tag}>"
        elif seen and full.endswith(seen) and len(full) > len(seen):
            # Видно конец слова — скрыто начало.
            tag = DamageTag.RESTORED if kind is EdgeKind.HIDDEN else DamageTag.FUZZY
            tagged = f"<{tag}>{full[: len(full) - len(seen)]}</{tag}>{seen}"
        elif kind is EdgeKind.FUZZY:
            tagged = f"<fuzzy>{full}</fuzzy>"
        if not tagged:
            continue
        # Целое слово, не внутри другого и не внутри тега.
        pattern = re.compile(r"(?<![\w<>/])" + re.escape(full) + r"(?![\w<])")
        match = pattern.search(body)
        if match is None:
            continue
        # Уже помечено моделью — не дублируем.
        before = body[max(0, match.start() - 12) : match.start()]
        if "<restored>" in before or "<fuzzy>" in before or "<unknown" in before:
            continue
        body = body[: match.start()] + tagged + body[match.end() :]
        inserted += 1
    return body, inserted
