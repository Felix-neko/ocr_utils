"""Маскирование мусора в ответе модели: чужие теги, невидимые символы, одиночные `*` и `_`, ломающие markdown.

Откуда. Разбор 1196 полос (1966/03, 1968/12, 1976/12, 1974/01–12, 19.09.2026): чужие теги `<a>` ×1
(«дре<a>зденским», 1974/03), `<span>` ×2 вокруг `<unclear>`; мягкий перенос U+00AD ×50 внутри слов
(«теле\\xadтайпа», невидим, ломает поиск и словарь); одиночные `*` ~25 — маркер сноски в печати
(«поставок*», «-2*», «[^1]: * …»), которые markdown спаривает в курсив; длинные `____` пустых полей
в бланках договоров — markdown делает из них линейку; управляющих C0/C1, zero-width, BOM, PUA — 0.
`* * *` (90) — разделитель, легитимный markdown (`<hr>`), не трогается.

Политика (решения пользователя): ничего содержательного не удалять, а маскировать так, чтобы вьюер
показал текст как есть: чужой тег → `&lt;a&gt;`; мягкий перенос — единственное, что удаляется (на печати
знака нет, удаление восстанавливает слово); прочие управляющие и невидимые → «▯» с кодом в отчёте;
одиночные `*` и `_` вне парной разметки → `\\*`, `\\_`. Экранирование — только в markdown-зонах:
внутри HTML-блока `<table>…</table>`, формулы `<latex>…</latex>`, fenced-блока и инлайн-кода
обратный слеш показался бы буквально. Вызывается из ``schema._coerce`` после нормализации тегов,
поэтому `.json`, md полос и выпуск чистые, а повторная санация старого `.json` ничего не меняет
(идемпотентно). Отчёт — ``PageResult.masked`` → ``meta["masked"]`` → `summary.csv`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

PLACEHOLDER = "▯"  # U+25AF — видимая замена управляющего или невидимого символа
SOFT_HYPHEN = "\u00ad"
NBSP = "\u00a0"
MAX_CONTROL_CODES = 20  # сколько кодов писать в отчёт (дальше только счётчик)

# HTML таблиц, который модель пишет по промпту (`rowspan`/`colspan` — атрибуты `th`/`td`); полный белый
# список (свои теги разметки + эти) собирает schema.py (``MARKDOWN_TAGS``) и передаёт в ``sanitize`` —
# schema здесь не импортируется, чтобы не было кольца импортов.
HTML_TAGS = frozenset({"table", "thead", "tbody", "tr", "td", "th", "br", "sub", "sup"})
_TAG = re.compile(r"<(?P<close>/?)(?P<name>[A-Za-z][\w:-]*)(?P<attrs>\s[^<>]*)?(?P<self>/?)>|<!--.*?-->", re.S)
# Зоны, где markdown не разбирается (HTML-блок таблицы, формула, код): экранировать нельзя.
_PROTECTED = re.compile(r"<table\b.*?</table>|<latex>.*?</latex>|```.*?```|`[^`\n]*`", re.S | re.IGNORECASE)
_LIST_LINE = re.compile(r"^\s*\* ")  # пункт списка со звёздочкой
_RULE_LINE = re.compile(r"^\s*(?:\*\s*){3,}$")  # разделитель «* * *» (markdown: <hr>)
# Парная разметка: `**жирный**` и `*курсив*` (непустое содержимое, без пробела после открывающей).
_PAIRED = re.compile(r"\*\*(?=\S)(?:[^*]|\*(?!\*))+?(?<=\S)\*\*|\*(?=[^\s*])[^*\n]*?(?<=[^\s*])\*")
_LONE_STAR = re.compile(r"(?<!\\)\*")
_UNDERSCORE = re.compile(r"(?<!\\)_")


@dataclass
class MaskReport:
    """Что замаскировано на полосе — для meta и summary.csv; пустой отчёт → ``{}``."""

    tags: list[str] = field(default_factory=list)  # имена чужих тегов (по одному на вхождение)
    soft_hyphens: int = 0  # удалённых мягких переносов
    nbsp: int = 0  # неразрывных пробелов и табуляций, заменённых пробелом
    controls: list[str] = field(default_factory=list)  # коды U+XXXX замаскированных символов (до MAX_CONTROL_CODES)
    controls_total: int = 0  # всего замаскированных символов
    escaped_stars: int = 0  # `*` → `\\*`
    escaped_underscores: int = 0  # `_` → `\\_`

    def as_dict(self) -> dict:
        """Только непустые поля — чтобы в meta и сводке чистая полоса выглядела как ``{}``."""
        return {key: value for key, value in vars(self).items() if value}


def mask_characters(text: str, report: MaskReport) -> str:
    """Символы: мягкий перенос удалить, NBSP и табуляция → пробел, `\\r` убрать, прочие управляющие → «▯».

    Args:
        text: Тело полосы.
        report: Отчёт — счётчики и коды.
    """
    out: list[str] = []
    for char in text:
        if char == SOFT_HYPHEN:
            report.soft_hyphens += 1
            continue
        if char in (NBSP, "\t"):
            report.nbsp += 1
            out.append(" ")
            continue
        if char == "\r":
            report.controls_total += 1
            continue
        if char == "\n":
            out.append(char)
            continue
        category = unicodedata.category(char)
        if category in ("Cc", "Cf", "Co", "Cn") or char in ("\u2028", "\u2029"):
            report.controls_total += 1
            if len(report.controls) < MAX_CONTROL_CODES:
                report.controls.append(f"U+{ord(char):04X}")
            out.append(PLACEHOLDER)
            continue
        out.append(char)
    return "".join(out)


class _Masker:
    """Замены для ``re.sub`` с учётом в отчёте (вместо вложенных функций).

    Args:
        report: Отчёт, в который пишутся счётчики.
        known_tags: Белый список имён тегов (в нижнем регистре).
    """

    def __init__(self, report: MaskReport, known_tags: frozenset[str]):
        self.report = report
        self.known_tags = known_tags

    def tag(self, match: re.Match) -> str:
        """Один тег-кандидат: свой — как есть, чужой (и HTML-комментарий) — со скобками-сущностями."""
        name = match.group("name")
        if name is not None and name.lower() in self.known_tags:
            return match.group(0)
        self.report.tags.append(name.lower() if name is not None else "!--")
        return match.group(0).replace("<", "&lt;").replace(">", "&gt;")

    def star(self, match: re.Match) -> str:
        """Одиночная `*` → `\\*`."""
        self.report.escaped_stars += 1
        return "\\*"

    def underscore(self, match: re.Match) -> str:
        """`_` → `\\_`."""
        self.report.escaped_underscores += 1
        return "\\_"


def mask_tags(text: str, masker: _Masker) -> str:
    """Чужие теги → `&lt;…&gt;`: вьюер показывает их буквально и не считает разметкой.

    Args:
        text: Тело полосы после ``normalize_tags``/``modernize_tags`` (омоглифы и старые имена уже приведены).
        masker: Замены с отчётом и белым списком.
    """
    return _TAG.sub(masker.tag, text)


def _escape_line(line: str, masker: _Masker) -> str:
    """Экранировать одиночные `*` и все `_` в одной строке markdown-зоны.

    Args:
        line: Строка вне таблиц, формул и кода.
        masker: Замены с отчётом.
    """
    if _RULE_LINE.match(line):
        return line
    prefix = ""
    if _LIST_LINE.match(line):  # маркер списка — не звёздочка текста
        split = line.index("* ") + 2
        prefix, line = line[:split], line[split:]
    # Парная разметка вырезается из рассмотрения, одиночные звёздочки между парами экранируются.
    pieces: list[str] = []
    position = 0
    for match in _PAIRED.finditer(line):
        pieces.append(_LONE_STAR.sub(masker.star, line[position : match.start()]))
        pieces.append(match.group(0))
        position = match.end()
    pieces.append(_LONE_STAR.sub(masker.star, line[position:]))
    return prefix + _UNDERSCORE.sub(masker.underscore, "".join(pieces))


def escape_markdown(text: str, masker: _Masker) -> str:
    """Экранировать `*` и `_` в markdown-зонах; таблицы, формулы, fenced-блоки и инлайн-код не трогаются.

    Args:
        text: Тело полосы.
        masker: Замены с отчётом.
    """
    pieces: list[str] = []
    position = 0
    for match in _PROTECTED.finditer(text):
        pieces.append(_escape_zone(text[position : match.start()], masker))
        pieces.append(match.group(0))
        position = match.end()
    pieces.append(_escape_zone(text[position:], masker))
    return "".join(pieces)


def _escape_zone(zone: str, masker: _Masker) -> str:
    """Построчное экранирование куска текста вне защищённых зон.

    Args:
        zone: Кусок markdown.
        masker: Замены с отчётом.
    """
    return "\n".join(_escape_line(line, masker) for line in zone.split("\n"))


def sanitize(text: str, known_tags: frozenset[str]) -> tuple[str, MaskReport]:
    """Все шаги по порядку: символы → чужие теги → экранирование. Идемпотентно.

    Args:
        text: Тело полосы после нормализации тегов.
        known_tags: Белый список имён тегов (свои теги разметки + ``HTML_TAGS``), в нижнем регистре.

    Returns:
        ``(чистое тело, отчёт)``.
    """
    report = MaskReport()
    masker = _Masker(report, known_tags)
    text = mask_characters(text, report)
    text = mask_tags(text, masker)
    text = escape_markdown(text, masker)
    return text, report
