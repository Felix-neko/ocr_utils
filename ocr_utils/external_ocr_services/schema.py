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
from ocr_utils.external_ocr_services.masking import HTML_TAGS, sanitize


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
    """Вид повреждения строки в ``edge_words``: буквы скрыты/срезаны и восстановлены (``HIDDEN``),
    видны, но ненадёжны (``UNCLEAR``), утрачены без восстановления (``GAP``)."""

    HIDDEN = "hidden"
    UNCLEAR = "unclear"
    GAP = "gap"


class DamageTag(StrEnum):
    """Теги повреждений в тексте по TEI: ``<supplied>`` — буквы не видны, вписаны по контексту;
    ``<unclear>`` — видны, но прочитаны с сомнением; ``<gap>`` — утрачены, внутри по одному
    ``GAP_FILLER`` на букву. Все три парные."""

    SUPPLIED = "supplied"
    UNCLEAR = "unclear"
    GAP = "gap"


# Все теги повреждений парные (``<gap>`` тоже: внутри заполнитель по числу утраченных букв).
PAIRED_DAMAGE_TAGS = tuple(DamageTag)
# Символ на одну утраченную букву внутри ``<gap>``: в журналах не встречается, виден в любом шрифте.
GAP_FILLER = "▒"
# Модель пишет в ``<gap>`` только ЧИСЛО утраченных букв (``<gap>4</gap>``): повтор одного символа
# при temperature 0 зацикливался до потолка токенов (IMG_0008_L: 15 тыс. «▒»), а число — 1–3
# токена, повторять нечего. Заполнитель расставляет разбор: меньше GAP_INLINE_MAX букв — по «▒»
# на букву, иначе внутри тега пишется ``[N symbols]``; число сверх GAP_COUNT_MAX режется.
GAP_INLINE_MAX = 16
GAP_COUNT_MAX = 999
GAP_DEFAULT_WIDTH = 3  # когда длина неизвестна (старый <unknown/>, мусор в теге)
# Старые имена тегов и полей (промпты до v10) — чтобы читать прежние .json и ответы по памяти модели.
LEGACY_DAMAGE_NAMES = {"restored": DamageTag.SUPPLIED, "fuzzy": DamageTag.UNCLEAR, "unknown": DamageTag.GAP}


class StructureTag(StrEnum):
    """Теги структуры: рубрика статьи (только из оглавления), рубрика внутри оглавления, автор,
    должность и маркер — текстовый элемент вне потока статьи (маркер раздела в углу, девиз, рубрика
    не из оглавления)."""

    RUBRIC = "rubric"
    RUBRIC_IN_TOC = (
        "rubricintoc"  # слитно: имена с «_» и «-» вьюер markdown в PyCharm не прячет (v15 — дефис, v18 — слитно)
    )
    AUTHOR = "author"
    POSITION = "position"
    MARKER = "marker"


class BlockTag(StrEnum):
    """Теги-обёртки блоков: список оглавления и сноска; считаются в meta (``blocks``)."""

    TOC = "toc"
    FOOTNOTE = "footnote"


class IllustrationKind(StrEnum):
    """Вид иллюстрации — первая строка fenced-блока в квадратных скобках: ``[фотография]``,
    ``[блок-схема]``, ``[графика]``; в meta (``blocks``) считаются под ключами :attr:`key`."""

    PHOTO = "фотография"
    SCHEMA = "блок-схема"
    LINE_ART = "графика"

    @property
    def key(self) -> str:
        """Ключ в meta ``blocks``: имя члена строчными через дефис (``photo``, ``schema``, ``line-art``)."""
        return self.name.lower().replace("_", "-")


# Старые XML-обёртки иллюстраций (промпты v10–v14): тег по виду вокруг block quote. Читаются из
# прежних .json и «по памяти» модели и переводятся в fenced-блок (``modernize_illustrations``).
LEGACY_ILLUSTRATION_TAGS = {
    "schema": IllustrationKind.SCHEMA,
    "photo": IllustrationKind.PHOTO,
    "line_art": IllustrationKind.LINE_ART,
}
# Старые имена тегов структуры (до v15 с подчёркиванием, v15–v17 через дефис) → новое слитное.
LEGACY_TAG_NAMES = {
    "rubric_in_toc": StructureTag.RUBRIC_IN_TOC.value,
    "rubric-in-toc": StructureTag.RUBRIC_IN_TOC.value,
}


# Формулы: LaTeX внутри тега, номер формулы снаружи текстом.
FORMULA_TAG = "latex"


class ParseError(ValueError):
    """Ответ модели не удалось привести к PageResult."""


# Префиксы id статей и рубрик в слитом оглавлении выпуска: «A3», «R2». Буква, а не число, чтобы
# модель не путала id с номером страницы или пункта; разные буквы — чтобы id статьи и рубрики не
# совпадали. Кириллические «А»/«Р» в ответе приводятся к латинице при разборе (``normalize_ref``).
ARTICLE_ID_PREFIX = "A"
RUBRIC_ID_PREFIX = "R"


@dataclass
class TocArticle:
    """Статья в оглавлении: название, авторы, номер страницы и (в указателе) номер выпуска.

    Args:
        title: Название статьи как напечатано в оглавлении.
        authors: ``[{"name": …, "position": … | None}]`` — авторы с должностями, если указаны.
        page: Номер страницы строкой (как напечатан); ``None`` — не указан.
        issue: Номер выпуска — только в годовом указателе; ``None`` в «Содержании».
        id: Id статьи в слитом оглавлении выпуска («A3»); у статьи одной полосы до слияния — ``None``.
    """

    title: str
    authors: list[dict] = field(default_factory=list)
    page: str | None = None
    issue: str | None = None
    id: str | None = None


@dataclass
class TocSection:
    """Секция оглавления: рубрика (``None`` — без рубрики) и её статьи по порядку.

    Args:
        rubric: Заголовок рубрики над группой статей; ``None`` у статей вне рубрик.
        articles: Статьи секции в порядке печати.
        id: Id рубрики в слитом оглавлении выпуска («R2»); одинаковые рубрики делят id; ``None`` —
            секция без рубрики или до слияния.
    """

    rubric: str | None
    articles: list[TocArticle] = field(default_factory=list)
    id: str | None = None


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
        rubric: Первая рубрика полосы (без тегов); ``None`` — нет. С v20 модель поле не заполняет —
            выводится из ``rubrics`` для шапки ``.md`` и сводки.
        title: Первый `#` полосы; ``None`` — полоса-продолжение. С v20 выводится из ``headings``.
        title_in_list: Совпал ли хоть один `#` со списком статей в промпте (считает код); ``None`` — списка не было.
        authors: ``[{"name", "position", "article": AuthorArticle | None, "printed": AuthorPrinted | None}]``
            — все имена авторов на полосе с привязкой к статье и местом печати.
        is_damaged: Булев вердикт «на полосе есть повреждённые буквы» (в описании модель пишет
            «повреждений не обнаружено» на 4 из 5 чистых полос, поэтому вердикт — отдельное поле);
            триггер второго прохода, если тот включён. В файлах до v16 — ``damaged``.
        damage_description: Описание повреждений глазами модели (по-русски, одно предложение);
            пусто — нет. В файлах до v16 — ``damage``.
        supplied: Слова с буквами, вписанными по контексту (то, что стоит в ``<supplied>``).
        unclear: Слова с буквами, прочитанными с сомнением (``<unclear>``).
        gap: Сколько мест утрачено без восстановления (тегов ``<gap>``).
        edge_words: ``[{"seen", "full", "kind": EdgeKind}]`` — записи по повреждённым строкам:
            что видно, полное слово, вид повреждения; по ним ставятся теги, если модель их не поставила.
        toc: Структурированное оглавление полосы — только у этапа ``TOC``, иначе ``None``.
        messages: Замечания пост-обработки к полосе (что достроено, какие расхождения найдены) —
            не из ответа модели; нужны, чтобы при прогоне всего пака видеть проблемные полосы.
        headings: ``[{"text", "article_id"}]`` — каждый ``#`` полосы по порядку и id статьи из списка в
            промпте («A3»; ``None`` — модель id не дала или он не из списка). После доводки структуры
            список переписывается кодом ровно по строкам тела — по нему сборка выпуска сверяет
            заголовки с оглавлением. В файлах до v20 поля нет: заполняется по ``title``.
        rubrics: ``[{"text", "rubric_id"}]`` — то же для тегов ``<rubric>`` («R2»).
        running_header_article_id: Id статьи из списка, если в верхнем колонтитуле её название; иначе ``None``.
        running_header_rubric_id: Id рубрики из списка, если в верхнем колонтитуле её название.
        running_footer_article_id: То же для нижнего колонтитула.
        running_footer_rubric_id: То же для нижнего колонтитула.
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
    is_damaged: bool = False
    damage_description: str = ""
    supplied: list[str] = field(default_factory=list)
    unclear: list[str] = field(default_factory=list)
    gap: int = 0
    edge_words: list[dict] = field(default_factory=list)
    toc: TocPage | None = None
    messages: list[str] = field(default_factory=list)
    # Что замаскировано в теле при разборе (masking.MaskReport.as_dict): чужие теги, невидимые символы,
    # экранированные `*`/`_`; пусто — тело было чистым. Уходит в meta и summary.csv.
    masked: dict = field(default_factory=dict)
    # Сколько битых `\`-escape починено при разборе (`\(` → `\\(`, `\н` → `\\н`, `\u` без hex): модель
    # экранирует markdown внутри JSON-строки, и строгий разбор падал бы на всём ответе.
    repaired_escapes: int = 0
    headings: list[dict] = field(default_factory=list)
    rubrics: list[dict] = field(default_factory=list)
    running_header_article_id: str | None = None
    running_header_rubric_id: str | None = None
    running_footer_article_id: str | None = None
    running_footer_rubric_id: str | None = None

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


def _ref_schema(id_key: str) -> dict:
    """Запись `headings`/`rubrics`: текст как написан и id из списка выпуска (``null`` — не из списка).

    Args:
        id_key: Имя поля id — ``article_id`` или ``rubric_id``.
    """
    return {
        "type": "object",
        "properties": {"text": {"type": "string"}, id_key: {"type": ["string", "null"]}},
        "required": ["text", id_key],
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
        "is_damaged": {
            "type": "boolean",
            "description": "True only if some letters on the page are hidden, cut off, blurred, squashed, smeared or "
            "washed out; false for a cleanly printed page.",
        },
        "damage_description": {
            "type": "string",
            "description": "One short sentence in Russian: which edge or area is damaged and how; empty string if none.",
        },
        "page_number": {"type": ["string", "null"], "description": 'Page number as printed, e.g. "12"; null if none.'},
        "headings": {
            "type": "array",
            "items": _ref_schema("article_id"),
            "description": "One entry per `#` heading in content_markdown, in reading order, with the id of the "
            "matching article from the given list (null if no list).",
        },
        "rubrics": {
            "type": "array",
            "items": _ref_schema("rubric_id"),
            "description": "One entry per <rubric> paragraph in content_markdown with the id of the rubric from the list.",
        },
        "authors": {"type": "array", "items": _PAGE_AUTHOR_SCHEMA, "description": "Every author named on the page."},
        "running_header": {"type": ["string", "null"]},
        "running_header_article_id": {
            "type": ["string", "null"],
            "description": "Id of the listed article whose title the running header names; else null.",
        },
        "running_header_rubric_id": {
            "type": ["string", "null"],
            "description": "Id of the listed rubric the running header names; else null.",
        },
        "running_footer": {"type": ["string", "null"]},
        "running_footer_article_id": {"type": ["string", "null"]},
        "running_footer_rubric_id": {"type": ["string", "null"]},
        "toc_kind": {
            "type": "string",
            "enum": [item.value for item in TocKind],
            "description": "contents — the issue's table of contents; index — annual index of articles; none otherwise.",
        },
        "content_markdown": {"type": "string", "description": "Full body of the page as Markdown, per the rules."},
        "supplied": {"type": "array", "items": {"type": "string"}},
        "unclear": {"type": "array", "items": {"type": "string"}},
        "gap": {"type": "integer"},
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


def tag_counts(text: str) -> dict[DamageTag, int]:
    """Сколько в тексте пар ``<supplied>``, ``<unclear>`` и ``<gap>``.

    Args:
        text: Тело полосы в markdown с тегами.

    Returns:
        ``{DamageTag: число}`` по всем трём тегам, нули включительно.
    """
    return {tag: len(re.findall(rf"<{tag}>.*?</{tag}>", text, re.DOTALL)) for tag in PAIRED_DAMAGE_TAGS}


def unbalanced_tags(text: str) -> list[DamageTag]:
    """Парные теги повреждений, у которых число открывающих и закрывающих не совпало.

    Args:
        text: Тело полосы в markdown с тегами.

    Returns:
        Теги с непарными скобками; пустой список — разметка цела.
    """
    return [tag for tag in PAIRED_DAMAGE_TAGS if text.count(f"<{tag}>") != text.count(f"</{tag}>")]


# Fenced-блок иллюстрации: строка ``` (возможно с языком), первая строка внутри — вид в квадратных
# скобках, дальше описание и надписи, закрывающая ```; всё на своих строках.
_ILLUSTRATION_BLOCK = re.compile(
    r"^[ \t]*```[^\n]*\n[ \t]*\[(?P<kind>[^\]\n]+)\][^\n]*\n.*?^[ \t]*```[ \t]*$", re.M | re.S
)
_FENCE_LINE = re.compile(r"^[ \t]*```", re.M)


def count_illustrations(text: str) -> dict[IllustrationKind, int]:
    """Сколько в теле fenced-блоков каждого вида (по первой строке ``[фотография]`` и т. п.).

    Args:
        text: Тело полосы в markdown.

    Returns:
        ``{IllustrationKind: число}`` по всем трём видам, нули включительно; блок с неизвестным
        видом в первой строке не считается.
    """
    counts = {kind: 0 for kind in IllustrationKind}
    for match in _ILLUSTRATION_BLOCK.finditer(text):
        kind = _enum_or_none(match.group("kind").strip().lower(), IllustrationKind)
        if kind is not None:
            counts[kind] += 1
    return counts


def unbalanced_fences(text: str) -> bool:
    """Нечётное число строк с ``` — незакрытый fenced-блок, дальше всё тело уедет «в цитату».

    Args:
        text: Тело полосы в markdown.
    """
    return len(_FENCE_LINE.findall(text)) % 2 == 1


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
        text: Поле ``damage_description`` ответа (``damage`` в файлах до v16).
    """
    first = re.split(r"[.;]", text.strip(), maxsplit=1)[0]
    return not first.strip() or _NO_DAMAGE.search(first) is not None


def _damage_description(payload: dict) -> str:
    """Описание повреждения: поле ``damage_description``, иначе прежнее ``damage`` (файлы и ответы до v16).

    Args:
        payload: Разобранный JSON ответа.

    Returns:
        Текст описания; пустая строка, если ни одного из полей нет.
    """
    return str(payload.get("damage_description") or payload.get("damage") or "")


def _damaged(payload: dict) -> bool:
    """Булев вердикт ``is_damaged`` (до v16 — ``damaged``); если модель его не дала — по тексту описания.

    Args:
        payload: Разобранный JSON ответа.

    Returns:
        ``True``, если на полосе, по мнению модели, есть повреждённые буквы.
    """
    for key in ("is_damaged", "damaged"):
        value = payload.get(key)
        if isinstance(value, bool):
            return value
    return not is_no_damage_text(_damage_description(payload))


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
                id=normalize_ref(item.get("id"), ARTICLE_ID_PREFIX),
            )
            for item in (raw.get("articles") or [])
            if isinstance(item, dict) and str(item.get("title") or "").strip()
        ]
        sections.append(
            TocSection(
                rubric=_text_or_none(raw.get("rubric")),
                articles=articles,
                id=normalize_ref(raw.get("id"), RUBRIC_ID_PREFIX),
            )
        )
    return TocPage(kind=kind, continues_previous=bool(payload.get("continues_previous", False)), sections=sections)


def _word_list(payload: dict, key: str, legacy_key: str) -> list[str]:
    """Список слов из поля ответа; пустое новое поле — из старого имени (ответы до v10).

    Args:
        payload: Разобранный JSON ответа.
        key: Имя поля по текущей схеме (``supplied`` / ``unclear``).
        legacy_key: Прежнее имя (``restored`` / ``fuzzy``).

    Returns:
        Непустые строки из поля.
    """
    items = payload.get(key) or payload.get(legacy_key) or []
    return [str(item) for item in items if str(item).strip()]


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
    # Нормализация тегов и старых форматов, затем маскирование мусора (чужие теги, невидимые символы,
    # одиночные `*`/`_`) — всё, что читается через parse_json_text, чистое; повтор ничего не меняет.
    body = expand_gaps(modernize_illustrations(modernize_tags(normalize_tags(unspace_letters(body)))))
    body = superscript_footnotes(body)
    body, mask_report = sanitize(body, MARKDOWN_TAGS)
    headings = _refs(payload, "headings", "article_id", ARTICLE_ID_PREFIX, payload.get("title"))
    rubrics = _refs(payload, "rubrics", "rubric_id", RUBRIC_ID_PREFIX, payload.get("rubric"))
    return PageResult(
        content_markdown=body,
        masked=mask_report.as_dict(),
        page_number=_text_or_none(payload.get("page_number")),
        running_header=_text_or_none(payload.get("running_header")),
        running_footer=_text_or_none(payload.get("running_footer")),
        toc_kind=toc_kind,
        notes=str(payload.get("notes") or ""),
        # Старые поля (до v20) — из новых списков: первый `#` и первый `<rubric>` полосы; в файлах до v20 — как были.
        rubric=_text_or_none(payload.get("rubric")) or (rubrics[0]["text"] if rubrics else None),
        title=_text_or_none(payload.get("title")) or (headings[0]["text"] if headings else None),
        title_in_list=None if title_in_list is None else bool(title_in_list),
        authors=_authors(payload.get("authors"), with_article=True),
        is_damaged=_damaged(payload),
        damage_description=_damage_description(payload),
        supplied=_word_list(payload, "supplied", "restored"),
        unclear=_word_list(payload, "unclear", "fuzzy"),
        gap=int(payload.get("gap") or payload.get("unknown") or 0),
        # Запись без полного слова бесполезна для расстановки тегов — пропускаем.
        edge_words=[item for item in (payload.get("edge_words") or []) if isinstance(item, dict) and item.get("full")],
        toc=_coerce_toc(payload.get("toc")) if Stage(stage) is Stage.TOC else None,
        # Замечания пост-обработки: у ответа модели их нет, у перечитанного .json — сохраняются.
        messages=[str(item) for item in (payload.get("messages") or []) if str(item).strip()],
        headings=headings,
        rubrics=rubrics,
        running_header_article_id=normalize_ref(payload.get("running_header_article_id"), ARTICLE_ID_PREFIX),
        running_header_rubric_id=normalize_ref(payload.get("running_header_rubric_id"), RUBRIC_ID_PREFIX),
        running_footer_article_id=normalize_ref(payload.get("running_footer_article_id"), ARTICLE_ID_PREFIX),
        running_footer_rubric_id=normalize_ref(payload.get("running_footer_rubric_id"), RUBRIC_ID_PREFIX),
    )


# Id статьи или рубрики как пишет его модель: латинская или кириллическая буква, необязательные
# пробелы, дефис или «#» между буквой и номером, регистр любой: «A3», «а 3», «R-2», «Р2», «a#3».
_REF = re.compile(r"^\s*([AaRrАаРр])\s*[-#№]?\s*(\d{1,3})\s*$")
_CYRILLIC_REF_LETTERS = {"А": "A", "а": "A", "Р": "R", "р": "R"}


def normalize_ref(value: object, prefix: str) -> str | None:
    """Id статьи («A3») или рубрики («R2») из ответа модели или файла в каноническом виде.

    Модель пишет id нестрого: кириллическими буквами (омоглифы «А»/«Р»), строчными, с пробелом или
    дефисом; всё это приводится к «A3»/«R2». Id другого вида (буква не та, число вместо id, мусор)
    → ``None`` — сборка тогда сопоставляет по названию.

    Args:
        value: Значение поля из ответа: строка, число или ``None``.
        prefix: Ожидаемый префикс — ``ARTICLE_ID_PREFIX`` или ``RUBRIC_ID_PREFIX``.

    Returns:
        Канонический id или ``None``.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return None  # голое число — не id: модель могла написать номер страницы или пункта
    match = _REF.match(str(value))
    if match is None:
        return None
    letter = _CYRILLIC_REF_LETTERS.get(match.group(1), match.group(1).upper())
    if letter != prefix:
        return None
    return f"{prefix}{int(match.group(2))}"


def _refs(payload: dict, key: str, id_key: str, prefix: str, legacy_text: object) -> list[dict]:
    """Список ``[{"text", id_key}]`` из поля ответа; в файле до v20 — из старого поля с одним текстом.

    Args:
        payload: Разобранный JSON ответа.
        key: Имя поля списка (``headings`` / ``rubrics``).
        id_key: Имя поля id в записи (``article_id`` / ``rubric_id``).
        prefix: Префикс id для нормализации.
        legacy_text: Значение старого поля (``title`` / ``rubric``) — запись без id, если списка нет.

    Returns:
        Записи с непустым текстом; id нормализован или ``None``.
    """
    items = payload.get(key)
    if not isinstance(items, list):
        text = _text_or_none(legacy_text)
        return [{"text": superscript_footnotes(text), id_key: None}] if text else []
    refs = []
    for item in items:
        if isinstance(item, str):  # модель дала голый текст вместо объекта
            item = {"text": item}
        if not isinstance(item, dict):
            continue
        text = _text_or_none(item.get("text"))
        if text:
            refs.append({"text": superscript_footnotes(text), id_key: normalize_ref(item.get(id_key), prefix)})
    return refs


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
    last_error: Exception | None = None
    decoder = json.JSONDecoder()
    for candidate, repaired in _json_candidates(text):
        try:
            payload, _ = decoder.raw_decode(candidate.lstrip())
            result = _coerce(payload, stage)
        except (json.JSONDecodeError, ParseError) as error:
            last_error = error
            continue
        result.repaired_escapes = repaired
        return result
    raise ParseError(f"невалидный JSON: {last_error}")


# Битый escape в JSON-строке: `\` перед символом не из `"\/bfnrtu` (`\(`, `\_`, `\н`) или `\u` без
# четырёх hex-знаков. Чётное число `\` перед ним — уже экранированные, их не трогать.
_BAD_ESCAPE = re.compile(r'(?<!\\)((?:\\\\)*)\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})')


def _repair_escapes(text: str) -> tuple[str, int]:
    """Починить битые `\`-escape в тексте ответа: одиночный `\` перед не-escape → `\\`.

    DeepSeek изредка переносит экранирование markdown внутрь JSON-строки («\(разборных\)»,
    «\_») или пишет `\н` вместо `\n`; строгий JSON на этом падает целиком (3 полосы из 10 649 в
    прогоне пака 21.09.2026). Починенная строка читается как литеральный `\` — то, что модель и
    имела в виду.

    Args:
        text: Кандидат на JSON (весь ответ или его срез).

    Returns:
        ``(текст, сколько замен)``; без битых escape — исходный текст и 0.
    """
    return _BAD_ESCAPE.subn(r"\1\\\\", text)


def _json_candidates(text: str) -> list[tuple[str, int]]:
    """Кандидаты на JSON в тексте ответа, от самого строгого к самому терпимому.

    Args:
        text: Сырой текст ответа модели.

    Returns:
        ``[(строка, починено escape)]``: текст как есть, без ограждений ```json, первый ``{...}``
        целиком и всё от первого ``{`` до конца (модели дописывают после JSON второй объект или
        эхо); затем те же с починенными `\`-escape (:func:`_repair_escapes`), если было что чинить.
    """
    plain = [text, _strip_fence(text)]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        plain.append(text[start : end + 1])
        plain.append(text[start:])
    candidates = [(candidate, 0) for candidate in plain]
    for candidate in plain:
        repaired, count = _repair_escapes(candidate)
        if count:
            candidates.append((repaired, count))
    return candidates


def pretty_json(text: str) -> str | None:
    """Ответ модели, переформатированный для чтения: отступы в 4 пробела, кириллица без escape.

    Схема не проверяется — только синтаксис: файл нужен, чтобы глазами смотреть, что именно
    ответила модель, в том числе при сбое приведения к ``PageResult``.

    Args:
        text: Сырой текст ответа модели (возможно, с ограждением ```json или хвостом после объекта).

    Returns:
        Красиво отформатированный JSON первого разобравшегося кандидата (см. ``_json_candidates``)
        или ``None``, если JSON в тексте не нашлось.
    """
    if not text or not text.strip():
        return None
    decoder = json.JSONDecoder()
    for candidate, _ in _json_candidates(text):
        try:
            payload, _ = decoder.raw_decode(candidate.lstrip())
        except json.JSONDecodeError:
            continue
        return json.dumps(payload, ensure_ascii=False, indent=4)
    return None


# Теги повреждений (новые и прежние); ``<gap>`` снимается вместе с содержимым (число, заполнители,
# «[N symbols]») — чтобы искать в тексте голое слово.
_TAG_STRIP = re.compile(rf"<gap>.*?</gap>|</?(supplied|unclear|gap|restored|fuzzy)>|<unknown\s*/>|{GAP_FILLER}", re.S)
_LEGACY_TAG = re.compile(r"</?(restored|fuzzy)>|<unknown\s*/>")
# Содержимое ``<gap>`` как его пишет модель: число, ряд заполнителей (ответ по памяти v10) или мусор.
_GAP_TAG = re.compile(rf"<{DamageTag.GAP}>(.*?)</{DamageTag.GAP}>", re.S)
_GAP_LONG = re.compile(r"\[(\d+) symbols\]")


def gap_width(content: str) -> int:
    """Число утраченных букв по содержимому ``<gap>``.

    Args:
        content: Текст между ``<gap>`` и ``</gap>``: число («4»), ряд «▒», уже раскрытое
            «[N symbols]» или что-то иное.

    Returns:
        Число от 1 до ``GAP_COUNT_MAX``; для пустого или чужого содержимого — ``GAP_DEFAULT_WIDTH``.
    """
    text = content.strip()
    if text.isdigit():
        width = int(text)
    elif text and set(text) == {GAP_FILLER}:
        width = len(text)
    elif (match := _GAP_LONG.fullmatch(text)) is not None:
        width = int(match.group(1))
    else:
        width = GAP_DEFAULT_WIDTH
    return max(1, min(width, GAP_COUNT_MAX))


def gap_markup(width: int) -> str:
    """Тег ``<gap>`` для пропуска в ``width`` букв: заполнители или «[N symbols]» для длинных.

    Args:
        width: Число утраченных букв (уже в пределах ``GAP_COUNT_MAX``).

    Returns:
        ``<gap>▒▒▒▒</gap>`` при ``width < GAP_INLINE_MAX``, иначе ``<gap>[N symbols]</gap>``.
    """
    inner = GAP_FILLER * width if width < GAP_INLINE_MAX else f"[{width} symbols]"
    return f"<{DamageTag.GAP}>{inner}</{DamageTag.GAP}>"


def _expand_gap(match: re.Match) -> str:
    """Замена одного ``<gap>…</gap>`` на раскрытый вид (для ``expand_gaps``).

    Args:
        match: Совпадение ``_GAP_TAG``.

    Returns:
        Тег с заполнителями или «[N symbols]».
    """
    return gap_markup(gap_width(match.group(1)))


def expand_gaps(text: str) -> str:
    """Раскрыть числовые пропуски модели: ``<gap>4</gap>`` → ``<gap>▒▒▒▒</gap>``, ``<gap>200</gap>`` →
    ``<gap>[200 symbols]</gap>``; ряды «▒» из старых ответов приводятся к тому же правилу.

    Args:
        text: Тело полосы в markdown.

    Returns:
        Текст, где содержимое каждого ``<gap>`` — заполнители или «[N symbols]».
    """
    return _GAP_TAG.sub(_expand_gap, text)


def is_gap_runaway(text: str) -> bool:
    """Ответ «убежал»: модель по памяти старого формата зациклилась на «▒» и упёрлась в потолок токенов.

    Args:
        text: Сырой текст ответа модели.

    Returns:
        ``True``, если в тексте есть ряд из 20 и более ``▒`` подряд — такой ответ обрезан и не разберётся.
    """
    return GAP_FILLER * 20 in text


# Кириллические омоглифы латинских букв в именах тегов: модель пишет ``<тoc>`` / ``<тоc>`` / ``< toc>``
# (1966/03 с. 93: 48 из 60 ответов), и тег перестаёт быть тегом.
_HOMOGLYPHS = str.maketrans("аеорсухАЕОРСУХТт", "aeopcyxAEOPCYXTt")
# Белый список для маскирования: свои теги разметки плюс HTML таблиц; всё остальное в угловых скобках
# после нормализации — мусор модели (`<a>`, `<span>`), маскируется в `&lt;…&gt;`.
MARKDOWN_TAGS = (
    frozenset(tag.value for enum in (DamageTag, StructureTag, BlockTag) for tag in enum) | {FORMULA_TAG} | HTML_TAGS
)
_KNOWN_TAGS = (
    {tag.value for enum in (DamageTag, StructureTag, BlockTag) for tag in enum}
    | {FORMULA_TAG}
    | set(LEGACY_TAG_NAMES)
    | set(LEGACY_ILLUSTRATION_TAGS)
)
_TAG_LIKE = re.compile(r"<(\s*/?)\s*([A-Za-z_\-\u0400-\u04FF]+)\s*(/?)\s*>")


def _normalize_tag(match: re.Match) -> str:
    """Замена одного тега-кандидата: омоглифы → латиница, пробелы убраны; чужие имена не трогаются.

    Args:
        match: Совпадение ``_TAG_LIKE``: (закрывающий слеш, имя, самозакрывающий слеш).

    Returns:
        Тег в каноническом виде, если имя после замены омоглифов — один из наших тегов, иначе как было.
    """
    closing, name, selfclose = match.groups()
    latin = name.translate(_HOMOGLYPHS)
    if latin not in _KNOWN_TAGS:
        return match.group(0)
    # Старые имена (``rubric_in_toc``, ``rubric-in-toc``) → новое слитное.
    latin = LEGACY_TAG_NAMES.get(latin, latin)
    return f"<{'/' if closing.strip() else ''}{latin}{'/' if selfclose else ''}>"


def normalize_tags(text: str) -> str:
    """Привести написание наших тегов к каноническому: ``<тoc>``, ``< toc>``, ``</ toc >`` → ``<toc>`` / ``</toc>``.

    Args:
        text: Тело полосы в markdown.

    Returns:
        Текст, где имена известных тегов написаны латиницей без пробелов; остальной текст не меняется.
    """
    return _TAG_LIKE.sub(_normalize_tag, text)


def modernize_tags(text: str) -> str:
    """Теги прежних промптов в тексте → TEI: ``<restored>``→``<supplied>``, ``<fuzzy>``→``<unclear>``,
    ``<unknown/>``→``<gap>▒▒▒</gap>``; модель иногда пишет их по памяти.

    Args:
        text: Тело полосы в markdown.

    Returns:
        Текст только с тегами ``DamageTag``.
    """

    return _LEGACY_TAG.sub(_modern_tag, text)


def _modern_tag(match: re.Match) -> str:
    """Замена одного прежнего тега на новый (для ``modernize_tags``).

    Args:
        match: Совпадение ``_LEGACY_TAG``.

    Returns:
        Новый тег той же роли; ``<unknown/>`` → ``<gap>`` с тремя заполнителями (длина неизвестна).
    """
    old = match.group(0)
    if old.startswith("<unknown"):
        return gap_markup(GAP_DEFAULT_WIDTH)
    closing = old.startswith("</")
    new = LEGACY_DAMAGE_NAMES[match.group(1)]
    return f"</{new}>" if closing else f"<{new}>"


# Старая обёртка иллюстрации: ``<photo>`` на своей строке, внутри block quote (``> …``), ``</photo>``.
_LEGACY_ILLUSTRATION = re.compile(
    r"^[ \t]*<(?P<tag>schema|photo|line_art)>[ \t]*\n(?P<body>.*?)\n[ \t]*</(?P=tag)>[ \t]*$", re.M | re.S
)
# Первая строка старого блока: ``[фотография: описание]`` / ``[графика: описание]`` / ``[блок-схема]``.
_LEGACY_ILLUSTRATION_HEAD = re.compile(r"^\[(?P<kind>[^\]:]+?)(?::\s*(?P<caption>[^\]]*))?\]\s*$")


def _modern_illustration(match: re.Match) -> str:
    """Один старый блок иллюстрации → fenced-блок нового формата (для ``modernize_illustrations``).

    Args:
        match: Совпадение ``_LEGACY_ILLUSTRATION``: тег и тело block quote.

    Returns:
        Fenced-блок: первая строка — вид из тега (``[фотография]``), затем описание из старой первой
        строки (если было) отдельной строкой и остальные строки без ``> ``.
    """
    kind = LEGACY_ILLUSTRATION_TAGS[match.group("tag")]
    lines = []
    for raw in match.group("body").split("\n"):
        line = re.sub(r"^[ \t]*>[ \t]?", "", raw).rstrip()
        if line:
            lines.append(line)
    out = [f"[{kind}]"]
    # Старая первая строка «[фотография: описание]» → вид уже написан, описание — своей строкой.
    if lines and (head := _LEGACY_ILLUSTRATION_HEAD.match(lines[0])):
        caption = (head.group("caption") or "").strip()
        if caption:
            out.append(caption)
        lines = lines[1:]
    out.extend(lines)
    return "```\n" + "\n".join(out) + "\n```"


# Знак сноски — надстрочные цифры: ⁰¹²³⁴⁵⁶⁷⁸⁹ (в Unicode они не подряд).
SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUPERSCRIPT = str.maketrans("0123456789", SUPERSCRIPT_DIGITS)
# Определение `[^12]:` — в начале абзаца или сразу после `<footnote>` (снимается первым); всё
# остальное `[^12]` — ссылка в тексте, в том числе перед двоеточием предложения («запасов [^1]:»).
_FOOTNOTE_REF = re.compile(r"\[\^(\d{1,3})\]")
_FOOTNOTE_DEF = re.compile(rf"(^|<{BlockTag.FOOTNOTE}>)[ \t]*\[\^(\d{{1,3}})\]:[ \t]*", re.M)


def superscript_footnotes(text: str) -> str:
    """Знаки сносок `[^1]` → надстрочные цифры: в тексте «слово¹», в сноске «<footnote>¹ текст</footnote>».

    До промпта v21 модель писала сноски в разметке markdown (`[^1]`, `[^1]: текст`); вьюеры
    показывают её ссылкой, а в тексте для чтения и поиска нужен знак как напечатан. Разбор приводит
    к надстрочным цифрам и старые `.json`, и ответы по памяти прежнего формата; повтор ничего не
    меняет (`[^` в результате нет).

    Args:
        text: Тело полосы.

    Returns:
        Тело с надстрочными знаками сносок.
    """
    text = _FOOTNOTE_DEF.sub(lambda m: f"{m.group(1)}{m.group(2).translate(_SUPERSCRIPT)} ", text)
    return _FOOTNOTE_REF.sub(lambda m: m.group(1).translate(_SUPERSCRIPT), text)


def modernize_illustrations(text: str) -> str:
    """Иллюстрации прежних промптов (v10–v14: block quote в теге ``<schema>``/``<photo>``/``<line_art>``)
    → fenced-блок с видом первой строкой; тело без старых блоков не меняется.

    Args:
        text: Тело полосы в markdown.

    Returns:
        Текст, где каждая старая обёртка заменена fenced-блоком; идемпотентно.
    """
    return _LEGACY_ILLUSTRATION.sub(_modern_illustration, text)


def _edge_kind(value: object) -> EdgeKind:
    """Вид повреждения из записи ``edge_words``; прежние имена (``fuzzy``, ``unknown``) и мусор → по умолчанию.

    Args:
        value: Поле ``kind`` записи.

    Returns:
        Член ``EdgeKind``; неизвестное значение → ``HIDDEN`` (самый частый случай — корешок).
    """
    text = str(value or "").strip().lower()
    legacy = {"fuzzy": EdgeKind.UNCLEAR, "unknown": EdgeKind.GAP}
    return _enum_or_none(text, EdgeKind) or legacy.get(text, EdgeKind.HIDDEN)


@dataclass(frozen=True)
class EdgeWordsReport:
    """Что сделал код по списку ``edge_words``: сколько тегов вставил в тело и сколько записей отбросил как пустые."""

    inserted: int = 0  # вставлено тегов <supplied>/<unclear> в тело
    empty: int = 0  # записей, где «как видно» совпадает с «как написано» и тегов нет — мусор модели


def tags_from_edge_words(body: str, edge_words: list[dict]) -> tuple[str, EdgeWordsReport]:
    """Расставить теги по списку ``edge_words`` там, где модель в тексте их не поставила.

    Модель охотнее заполняет список повреждённых строк, чем ставит теги в тексте. Для записи с
    ``kind=hidden`` невидимая часть — это ``full`` минус видимый фрагмент ``seen`` (с начала или с
    конца слова); первое вхождение голого ``full`` заменяется на слово с ``<supplied>``. ``unclear``
    — то же с ``<unclear>``. Записи ``kind=gap`` в тело не переносятся: почти всегда это строка,
    оборванная переносом, а слово в теле целое — на МТС 1991/02 так появились 24 ложных пропуска
    вида «в первую<gap>▒▒▒</gap> очередь» (`reports/external_ocr_hyphenation_second_pass.md`);
    пропускам верим только из самого текста. Пустые записи (``seen == full`` без тегов — четверть
    списка) пропускаются и считаются.

    Args:
        body: Тело полосы в markdown.
        edge_words: Записи ``{"seen", "full", "kind"}`` из ответа модели.

    Returns:
        ``(тело с расставленными тегами, отчёт)``: в отчёте число вставок и число пустых записей —
        оба уходят в meta (``tags_from_edge_words``, ``edge_words_empty``), чтобы видеть, сколько
        тегов поставил код, а не модель, и сколько мусора в списке.
    """
    inserted = empty = 0
    for item in edge_words:
        # Теги и заполнители внутри самих записей (модель копирует их из текста) снимаем — ищем голые слова.
        raw_full = str(item.get("full") or "")
        full = _TAG_STRIP.sub("", raw_full).strip()
        seen = _TAG_STRIP.sub("", str(item.get("seen") or "")).strip()
        kind = _edge_kind(item.get("kind"))
        # Запись без разницы между «видно» и «написано» и без тегов ничего не говорит о повреждении.
        if seen == full and raw_full.strip() == full:
            empty += 1
            continue
        # «адми-» → «административные» — обычный перенос, а не срез; пропуски — только из тела.
        if not full or full not in body or seen.endswith(("-", "\u00ad")) or kind is EdgeKind.GAP:
            continue
        tag = DamageTag.SUPPLIED if kind is EdgeKind.HIDDEN else DamageTag.UNCLEAR
        if seen and full.startswith(seen) and len(full) > len(seen):
            # Видно начало слова — скрыт хвост.
            tagged = f"{seen}<{tag}>{full[len(seen):]}</{tag}>"
        elif seen and full.endswith(seen) and len(full) > len(seen):
            # Видно конец слова — скрыто начало.
            tagged = f"<{tag}>{full[: len(full) - len(seen)]}</{tag}>{seen}"
        elif kind is EdgeKind.UNCLEAR:
            tagged = f"<{tag}>{full}</{tag}>"
        else:
            continue
        # Целое слово, не внутри другого и не внутри тега.
        pattern = re.compile(r"(?<![\w<>/])" + re.escape(full) + r"(?![\w<])")
        match = pattern.search(body)
        if match is None:
            continue
        # Уже помечено моделью — не дублируем.
        before = body[max(0, match.start() - 12) : match.start()]
        if any(f"<{tag}>" in before for tag in DamageTag):
            continue
        body = body[: match.start()] + tagged + body[match.end() :]
        inserted += 1
    return body, EdgeWordsReport(inserted=inserted, empty=empty)


def _supplied_variants(word: str) -> re.Pattern:
    """Регулярка на слово ``word`` в теле, как бы модель ни расставила в нём ``<supplied>``.

    Args:
        word: Голое слово без тегов.

    Returns:
        Скомпилированный шаблон: между любыми двумя буквами слова допускаются открывающие и
        закрывающие теги ``<supplied>``; границы — не буква и не тег.
    """
    tag = rf"(?:</?{DamageTag.SUPPLIED}>)*"
    inner = tag.join(re.escape(char) for char in word)
    return re.compile(r"(?<![\w<>/])" + tag + inner + tag + r"(?![\w<])")


def strip_hyphen_supplied(body: str, edge_words: list[dict]) -> tuple[str, int]:
    """Снять ``<supplied>`` с продолжений переносов, которые модель сама объявила «восстановленными».

    Режим ответа на полосе с обрезанным краем: перенос «ва-» в конце строки и продолжение «лютных»
    в начале следующей модель читает как утрату и пишет «ва<supplied>л</supplied>ютных», а в
    ``edge_words`` честно показывает ``seen`` с дефисом на конце — по этому признаку тег и снимается
    (на с. 47 МТС 1991/02 таких 39 в одном ответе). Буквы не меняются, только разметка.

    Args:
        body: Тело полосы в markdown (теги уже приведены к TEI).
        edge_words: Записи ``{"seen", "full", "kind"}`` из ответа модели.

    Returns:
        ``(тело без ложных тегов, сколько слов очищено)``; число уходит в meta
        (``hyphen_supplied_stripped``).
    """
    stripped = 0
    for item in edge_words:
        seen = _TAG_STRIP.sub("", str(item.get("seen") or "")).strip()
        raw_full = str(item.get("full") or "")
        if not seen.endswith(("-", "\u00ad")) or f"<{DamageTag.SUPPLIED}>" not in raw_full:
            continue
        full = _TAG_STRIP.sub("", raw_full).strip()
        if not full:
            continue
        # В теле слово может стоять с тегом не там, где в записи: ищем любую расстановку.
        match = _supplied_variants(full).search(body)
        if match is None or f"<{DamageTag.SUPPLIED}>" not in match.group(0):
            continue
        body = body[: match.start()] + full + body[match.end() :]
        stripped += 1
    return body, stripped


# «трудностей <supplied>стей</supplied>»: слово уже целое, а следом — его же хвост в теге.
_DUPLICATE_SUPPLIED = re.compile(rf"(\w+)([ \u00a0]+)<{DamageTag.SUPPLIED}>(\w+)</{DamageTag.SUPPLIED}>")


def _drop_duplicate(match: re.Match) -> str:
    """Замена для ``_DUPLICATE_SUPPLIED``: убрать тег, если он повторяет конец предыдущего слова.

    Args:
        match: Совпадение «слово, пробелы, ``<supplied>хвост</supplied>``».

    Returns:
        Одно слово, если хвост совпал с его концом; иначе исходный текст без изменений.
    """
    word, spaces, tail = match.group(1), match.group(2), match.group(3)
    if word.lower().endswith(tail.lower()) and len(tail) < len(word):
        return word
    return match.group(0)


def drop_duplicate_supplied(body: str) -> tuple[str, int]:
    """Убрать дубли достройки: «трактора <supplied>ра</supplied>» → «трактора».

    Модель на обрезанном крае иногда пишет слово целиком и тут же — его хвост в ``<supplied>``
    отдельным словом (v15, с. 93 МТС 1991/02: «трудностей <supplied>стей</supplied>
    сформирован»). Тег с хвостом, совпадающим с концом предыдущего слова, удаляется вместе с
    пробелом.

    Args:
        body: Тело полосы в markdown.

    Returns:
        ``(тело без дублей, сколько убрано)``; число уходит в meta (``duplicate_supplied_dropped``).
    """
    # Считаем отдельно от замены, чтобы не заводить вложенную функцию-счётчик.
    dropped = sum(1 for match in _DUPLICATE_SUPPLIED.finditer(body) if _drop_duplicate(match) != match.group(0))
    return _DUPLICATE_SUPPLIED.sub(_drop_duplicate, body), dropped
