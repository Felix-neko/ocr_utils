"""Сборка выпуска в один markdown: полосы по порядку, переносы и оборванные абзацы сшиты через границу полос.

Вход — готовые ``имя.json`` полос выпуска (у полосы оглавления — ответ этапа ``toc`` с блоком ``<toc>``,
у понижённой — этапа ``page``). Тела полос идут подряд без маркеров границ; привязка к полосам —
в sidecar ``{выпуск}.pages.json`` (смещение первого символа каждой полосы в тексте, список склеек,
пропущенные полосы). Полоса без результата оставляет в тексте HTML-комментарий — единственное
исключение из «без маркеров», иначе пропуск невидим.

Граница двух соседних полос A и B. Хвост — последний абзац A без замыкающих сносок и маркеров,
голова — первый абзац B; оба должны быть простым текстом (не заголовок, не тег-блок, не таблица,
не иллюстрация, не список, не строка внутри ``<toc>``). Тогда, по порядку:

1. хвост кончается половиной слова с дефисом, голова — строчное продолжение → ``hyphen_join.join_across_boundary``:
   дефис-перенос убран (по словарю pymorphy3, теги повреждений на местах) или составное слово
   оставлено с дефисом — один абзац;
2. иначе хвост не кончается знаком конца предложения, а голова начинается со строчной буквы, цифры
   или знака продолжения (``—``, ``,``, ``(``, ``;``) → один абзац через пробел;
3. иначе — отдельные абзацы.

Сноски A, отодвинутые с хвоста, идут сразу после сшитого абзаца: сноска следует за абзацем со ссылкой.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from ocr_utils.external_ocr_services import PROMPT_VERSION
from ocr_utils.external_ocr_services.hyphen_join import JoinRule, Morph, default_morph, join_across_boundary
from ocr_utils.external_ocr_services.ocr import output_paths, read_meta
from ocr_utils.external_ocr_services.render import _yaml_value
from ocr_utils.external_ocr_services.schema import BlockTag, DamageTag, ParseError, Stage, StructureTag, parse_json_text
from ocr_utils.external_ocr_services.structure import join_paragraphs, paragraphs_of

logger = logging.getLogger(__name__)

ISSUE_MD_SUFFIX = ".md"
ISSUE_PAGES_SUFFIX = ".pages.json"

# Теги повреждений могут стоять в начале и в конце простого абзаца — их снимаем перед проверками.
_DAMAGE_TAGS = re.compile(rf"</?(?:{'|'.join(tag.value for tag in DamageTag)})>")
# Абзац, начинающийся с тега, — не простой текст, если тег не тег повреждения: `<rubric>`, `<author>`,
# `<marker>`, `<footnote>`, `<toc>`, `<table>`, `<latex>`, HTML-комментарий.
_LEADING_TAG = re.compile(r"^<(?P<name>[a-z][\w-]*)")
_DAMAGE_TAG_NAMES = frozenset(tag.value for tag in DamageTag)
# Начала абзацев, которые не сшиваются: заголовки, цитаты, таблицы в markdown, fenced-блоки, списки.
_NOT_PLAIN_START = re.compile(r"^(?:#|>|\||```|[-*+] |\d+[.)] |\[\^)")
# Конец предложения (после снятия закрывающих тегов): точка, вопрос, восклицание, многоточие,
# курсив/жирный (подпись «*И. Иванов*»). Кавычка, скобка, двоеточие и точка с запятой концом не
# считаются: со строчной головой это середина предложения.
_SENTENCE_END = re.compile(r"[.!?…*]\s*$")
# Точка после сокращения — не конец предложения: «35 тыс.» + «автомашин».
_ABBREVIATION_END = re.compile(
    r"(?:^|[\s(«])(?:тыс|млн|млрд|руб|коп|гг?|т|кг|км|м|см|мм|стр|с|шт|экз|др|пр|напр|проц|ул|им|обл|р-н|п|см)\.\s*$"
)
# Голова, продолжающая оборванное предложение: строчная буква, цифра или знак продолжения.
_CONTINUATION_START = re.compile(r"^[а-яёa-z0-9—,(;]")
# Последнее слово хвоста и первое голое слово головы — для поиска повторённого хвоста слова
# («…сокращением» + «нием эксплуатационных»: модель дописала слово на первой полосе целиком).
_LAST_WORD = re.compile(r"([а-яёА-ЯЁ]{3,})\s*$")
_FIRST_WORD = re.compile(r"^([а-яё]{2,})(?![\w-])")
_TOC_OPEN = re.compile(rf"<{BlockTag.TOC}>")
_TOC_CLOSE = re.compile(rf"</{BlockTag.TOC}>")
_TABLE_OPEN = re.compile(r"<table\b", re.IGNORECASE)
_TABLE_CLOSE = re.compile(r"</table>", re.IGNORECASE)
# «Плавающие» блоки — вне потока текста, текст обтекает их и продолжается на следующей полосе:
# сноски (в теге или голые `[^1]:`), маркеры, иллюстрации (fenced-блок или старая цитата
# `> [картинка…]`), таблицы (HTML и markdown), их подписи («Рис. 2.», «Таблица 3») и примечания
# к таблицам («*Примечание.* …»).
_FLOATING_START = re.compile(
    rf"^(?:<{BlockTag.FOOTNOTE}>|\[\^\w+\]:|<{StructureTag.MARKER}>|```|> \[|<table\b|\||"
    r"Рис(?:\.|\s)|Фиг\.|Табл(?:\.|ица\b)|Схема\s*\d|График\s*\d|\*?Примечани[ея])",
    re.IGNORECASE,
)
# Номер таблицы или рисунка отдельной строкой («Таблица 3»): следующий абзац — её название, тоже
# плавающий (в печати он стоит над таблицей, а текст статьи продолжается после неё).
_CAPTION_NUMBER = re.compile(r"^(?:Таблица|Табл\.|Рис\.|Рис|Фиг\.)\s*\d+[.:]?\s*$", re.IGNORECASE)


class JoinKind(StrEnum):
    """Что произошло на границе полос."""

    HYPHEN = "hyphen"  # дефис-перенос убран, слово склеено по словарю
    COMPOUND = "compound"  # дефис на границе — составное слово, оставлен, абзацы сшиты без пробела
    DUPLICATE = "duplicate"  # модель дописала слово на предыдущей полосе целиком, а на следующей повторила его хвост — хвост убран
    PARAGRAPH = "paragraph"  # оборванное предложение сшито через пробел


@dataclass
class PageEntry:
    """Полоса в sidecar: откуда взята и где начинается в тексте выпуска."""

    file: str  # путь полосы относительно out-dir без суффикса
    page_number: str | None  # напечатанный номер по ответу модели
    stage: str  # этап, ответ которого вошёл в текст (page / toc)
    offset: int  # смещение первого символа полосы в тексте выпуска (внутри сшитого абзаца — внутри него)
    line: int  # строка (с 1) того же места


@dataclass
class JoinRecord:
    """Склейка на границе двух полос — для sidecar и проверки глазами."""

    after: str  # полоса перед границей (file)
    before: str  # полоса после границы
    kind: JoinKind
    word: str | None = None  # склеенное слово «снаб-жения» у hyphen/compound


@dataclass
class IssueAssembly:
    """Итог сборки выпуска: текст, привязка полос, склейки, пропуски."""

    issue: str  # «год/выпуск»
    text: str  # содержимое файла выпуска (шапка + тело)
    pages: list[PageEntry] = field(default_factory=list)
    joins: list[JoinRecord] = field(default_factory=list)
    missing: list[dict] = field(default_factory=list)  # {"file", "reason"}

    def count(self, kind: JoinKind) -> int:
        """Сколько склеек данного вида.

        Args:
            kind: Вид склейки.
        """
        return sum(1 for join in self.joins if join.kind is kind)

    def as_dict(self) -> dict:
        """Sidecar ``{выпуск}.pages.json`` целиком."""
        return {
            "issue": self.issue,
            "assembled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "prompt_version": PROMPT_VERSION,
            "pages": [vars(page) for page in self.pages],
            "joins": [{**vars(join), "kind": join.kind.value} for join in self.joins],
            "missing": self.missing,
            "counts": {kind.value: self.count(kind) for kind in JoinKind},
        }


@dataclass
class _Block:
    """Абзац в собираемом тексте с пометкой, простой ли это текст (годится ли для сшивания)."""

    text: str
    plain: bool  # простой текст — годится для сшивания
    floating: bool = (
        False  # блок вне потока (сноска, иллюстрация, таблица, подпись) — при поиске хвоста и головы пропускается
    )


@dataclass
class _PageBody:
    """Полоса, прочитанная с диска: абзацы с пометками или причина пропуска."""

    file: str
    page_number: str | None = None
    stage: str = Stage.PAGE.value
    blocks: list[_Block] = field(default_factory=list)
    reason: str | None = None  # не None — полосы нет, в текст идёт комментарий


def _is_plain(paragraph: str) -> bool:
    """Простой текстовый абзац: не заголовок, не тег-блок, не таблица, не иллюстрация, не список.

    Args:
        paragraph: Абзац после ``paragraphs_of``.
    """
    head = paragraph.lstrip()
    if _NOT_PLAIN_START.match(head):
        return False
    tag = _LEADING_TAG.match(head)
    if tag is not None and tag.group("name") not in _DAMAGE_TAG_NAMES:
        return False
    if head.startswith("<!--"):
        return False
    tail = _DAMAGE_TAGS.sub("", paragraph.rstrip())
    # Замыкающие теги блоков и формул: абзац кончается блоком, сшивать с ним нельзя.
    return not re.search(r"</(?:toc|footnote|table|latex|rubric|author|position|marker)>\s*$", tail)


def _blocks_of(body: str) -> list[_Block]:
    """Абзацы полосы с пометкой «простой текст»; строки внутри ``<toc>`` и ``<table>`` простыми не считаются.

    Args:
        body: Тело полосы в markdown.
    """
    blocks: list[_Block] = []
    inside_toc = inside_table = after_caption_number = False
    for paragraph in paragraphs_of(body):
        opened_here = bool(_TOC_OPEN.search(paragraph)) or bool(_TABLE_OPEN.search(paragraph))
        plain = _is_plain(paragraph) and not inside_toc and not inside_table and not opened_here
        floating = inside_table or after_caption_number or bool(_FLOATING_START.match(paragraph.lstrip()))
        blocks.append(_Block(paragraph, plain, floating))
        after_caption_number = bool(_CAPTION_NUMBER.match(paragraph.strip()))
        # Состояние блоков — на следующий абзац: открылся и не закрылся в этом же абзаце.
        if _TOC_OPEN.search(paragraph):
            inside_toc = True
        if _TOC_CLOSE.search(paragraph):
            inside_toc = False
        if _TABLE_OPEN.search(paragraph):
            inside_table = True
        if _TABLE_CLOSE.search(paragraph):
            inside_table = False
    return blocks


def _read_page(out_dir: Path, rel: Path) -> _PageBody:
    """Полоса с диска: ``.json`` тем же разбором, что и ответ модели; нет или битая — причина.

    Args:
        out_dir: Корень выхода.
        rel: Путь полосы относительно корня (с любым суффиксом).
    """
    paths = output_paths(out_dir, rel)
    file = rel.with_suffix("").as_posix()
    meta = read_meta(out_dir, rel) or {}
    if not paths.json.is_file():
        return _PageBody(file, reason=meta.get("error") or meta.get("parse_error") or "нет .json")
    if meta.get("error") or meta.get("parse_error"):
        return _PageBody(file, reason=str(meta.get("error") or meta.get("parse_error")))
    stage = Stage(meta.get("stage") or Stage.PAGE)
    try:
        result = parse_json_text(paths.json.read_text(encoding="utf-8"), stage)
    except (ParseError, OSError) as error:
        return _PageBody(file, reason=f".json не читается: {error}")
    return _PageBody(file, result.page_number, stage.value, _blocks_of(result.content_markdown))


def _tail_index(blocks: list[_Block], first: int) -> int | None:
    """Индекс хвоста полосы среди её блоков в общем списке: последний простой абзац, не считая
    замыкающих сносок и маркеров; ``None`` — хвоста нет.

    Args:
        blocks: Общий список блоков выпуска.
        first: Индекс первого блока полосы.
    """
    index = len(blocks) - 1
    while index >= first and blocks[index].floating:
        index -= 1
    if index >= first and blocks[index].plain:
        return index
    return None


def _head_index(blocks: list[_Block]) -> int | None:
    """Индекс головы полосы: первый абзац, не считая плавающих блоков в начале; ``None`` — не простой текст.

    Args:
        blocks: Блоки полосы.
    """
    index = 0
    while index < len(blocks) and blocks[index].floating:
        index += 1
    if index < len(blocks) and blocks[index].plain:
        return index
    return None


def _try_join(
    tail: str, head: str, morph: Morph | None, rule: JoinRule, join_paragraphs_across: bool
) -> tuple[str, int, JoinKind, str | None] | None:
    """Сшить хвост и голову, если это одно слово или одно предложение.

    Args:
        tail: Хвост предыдущей полосы (простой абзац).
        head: Голова следующей (простой абзац).
        morph: Анализатор для переносов; ``None`` — слова с дефисом на границе не сшивать.
        rule: Правило склейки.
        join_paragraphs_across: Сшивать ли оборванные предложения через пробел.

    Returns:
        ``(сшитый абзац, смещение начала головы в нём, вид, слово)`` или ``None`` — не сшивать.
    """
    boundary = join_across_boundary(tail, head, morph, rule) if morph is not None else None
    if boundary is not None:
        kind = JoinKind.HYPHEN if boundary.joined else JoinKind.COMPOUND
        return boundary.text, boundary.head_start, kind, boundary.word
    if not join_paragraphs_across:
        return None
    tail_text = _DAMAGE_TAGS.sub("", tail.rstrip())
    head_text = _DAMAGE_TAGS.sub("", head.lstrip())
    # Модель дописала слово с переносом на первой полосе целиком («сокращением»), а на следующей
    # осталась вторая половина («нием»): половина, которой словарь не знает, но которой кончается
    # последнее слово хвоста, — дубль, убирается.
    last, fragment = _LAST_WORD.search(tail_text), _FIRST_WORD.match(head_text)
    if morph is not None and last and fragment and fragment.group(1) != last.group(1):
        word, half = last.group(1), fragment.group(1)
        if word.lower().endswith(half) and morph.known(word) and not morph.known(half):
            stem = tail.rstrip() + " "
            rest = head.lstrip()[fragment.end() :].lstrip()
            return stem + rest, len(stem), JoinKind.DUPLICATE, f"{word}+{half}"
    if not _CONTINUATION_START.match(head_text):
        return None
    if _SENTENCE_END.search(tail_text) and not _ABBREVIATION_END.search(tail_text):
        return None
    stem = tail.rstrip() + " "
    return stem + head.lstrip(), len(stem), JoinKind.PARAGRAPH, None


def assemble_issue(
    out_dir: Path,
    issue_key: str,
    page_rels: list[Path],
    *,
    join_hyphens: bool = True,
    join_paragraphs_across: bool = True,
    rule: JoinRule = JoinRule.E,
) -> IssueAssembly:
    """Собрать текст выпуска из готовых полос (в память; запись — :func:`write_issue`).

    Args:
        out_dir: Корень выхода с ``{год}/{выпуск}/{полоса}.json``.
        issue_key: «год/выпуск».
        page_rels: Полосы выпуска по порядку (пути относительно корня, суффикс любой).
        join_hyphens: Сшивать слова с дефисом на границе (иначе граница с дефисом — как без него).
        join_paragraphs_across: Сшивать оборванные предложения через пробел.
        rule: Правило склейки переносов.

    Returns:
        :class:`IssueAssembly`: текст файла, смещения полос, склейки, пропуски.
    """
    morph = default_morph() if join_hyphens else None  # словарь грузится один раз на процесс
    blocks: list[_Block] = []
    # Для каждой полосы — (индекс блока, смещение внутри него), чтобы после сборки посчитать offset.
    starts: list[tuple[int, int]] = []
    pages: list[PageEntry] = []
    joins: list[JoinRecord] = []
    missing: list[dict] = []
    previous: _PageBody | None = None
    previous_first = 0  # индекс первого блока предыдущей полосы
    for rel in page_rels:
        page = _read_page(out_dir, rel)
        if page.reason is not None:
            missing.append({"file": page.file, "reason": page.reason})
            page.blocks = [_Block(f"<!-- полоса {page.file} не распознана: {page.reason} -->", False)]
        first = len(blocks)
        start = (first, 0)
        tail_index = _tail_index(blocks, previous_first) if previous is not None and blocks else None
        head_index = _head_index(page.blocks)
        if tail_index is not None and head_index is not None:
            head = page.blocks[head_index].text
            joined = _try_join(blocks[tail_index].text, head, morph, rule, join_paragraphs_across)
            if joined is not None:
                text, head_start, kind, word = joined
                blocks[tail_index] = _Block(text, True)
                start = (tail_index, head_start)
                joins.append(JoinRecord(previous.file, page.file, kind, word))
                # Голова ушла в хвост; плавающие блоки перед ней (таблица, сноска) и всё остальное
                # идут следом за сшитым абзацем в прежнем порядке.
                page.blocks = page.blocks[:head_index] + page.blocks[head_index + 1 :]
        blocks.extend(page.blocks)
        starts.append(start)
        pages.append(PageEntry(page.file, page.page_number, page.stage, 0, 0))
        # Хвост следующей границы ищется от начала этой полосы; у сшитой полосы её начало — хвост
        # предыдущей, так что однобзацная полоса, целиком ушедшая в склейку, тоже даёт хвост.
        previous, previous_first = page, start[0]
    header = _header(issue_key, len(pages), len(missing))
    body = join_paragraphs([block.text for block in blocks]) if blocks else ""
    text = header + body
    # Смещения: накопленная длина абзацев до нужного плюс смещение внутри него.
    offsets = [len(header)]
    for block in blocks:
        offsets.append(offsets[-1] + len(block.text) + 2)  # «\n\n» между абзацами
    for entry, (index, inner) in zip(pages, starts):
        entry.offset = offsets[index] + inner if index < len(blocks) else len(text)
        entry.line = text.count("\n", 0, entry.offset) + 1
    return IssueAssembly(issue_key, text, pages, joins, missing)


def _header(issue_key: str, pages: int, missing: int) -> str:
    """YAML-шапка файла выпуска. Времени сборки здесь нет — файл воспроизводим байт в байт.

    Args:
        issue_key: «год/выпуск».
        pages: Полос в выпуске.
        missing: Из них без результата.
    """
    year, _, issue = issue_key.partition("/")
    lines = [
        "---",
        f"year: {_yaml_value(year)}",
        f"issue: {_yaml_value(issue)}",
        f"pages: {pages}",
        f"missing: {missing}",
        "---",
        "",
    ]
    return "\n".join(lines)


def issue_paths(out_dir: Path, issue_key: str) -> tuple[Path, Path]:
    """Где лежат файл выпуска и его sidecar: ``out_dir/{год}/{выпуск}.md`` и ``.pages.json``.

    Args:
        out_dir: Корень выхода.
        issue_key: «год/выпуск».
    """
    base = out_dir / issue_key
    return base.with_name(base.name + ISSUE_MD_SUFFIX), base.with_name(base.name + ISSUE_PAGES_SUFFIX)


def write_issue(out_dir: Path, assembly: IssueAssembly) -> Path:
    """Записать файл выпуска и sidecar.

    Args:
        out_dir: Корень выхода.
        assembly: Итог :func:`assemble_issue`.

    Returns:
        Путь файла выпуска.
    """
    md_path, pages_path = issue_paths(out_dir, assembly.issue)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(assembly.text, encoding="utf-8")
    pages_path.write_text(json.dumps(assembly.as_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    return md_path


def issue_pages(out_dir: Path, issue_key: str) -> list[Path]:
    """Полосы выпуска по готовым ``.json`` в папке выхода (для сборки без прогона).

    Args:
        out_dir: Корень выхода.
        issue_key: «год/выпуск».

    Returns:
        Пути относительно корня (с суффиксом ``.json``) по порядку имён; служебные ``.meta.json``,
        ``.toc.json`` и ``toc.json`` выпуска не в счёт.
    """
    issue_dir = out_dir / issue_key
    rels = []
    for path in sorted(issue_dir.glob("*.json")):
        if path.name == "toc.json" or path.name.endswith((".meta.json", ".toc.json")):
            continue
        rels.append(path.relative_to(out_dir))
    return rels


def list_issues(out_dir: Path, only_year: str | None = None, only_issue: str | None = None) -> list[str]:
    """Выпуски в папке выхода: ``{год}/{выпуск}`` с хотя бы одним ``.json`` полосы.

    Args:
        out_dir: Корень выхода.
        only_year: Только этот год.
        only_issue: Только этот выпуск.
    """
    keys = []
    for year_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        if only_year and year_dir.name != only_year:
            continue
        for issue_dir in sorted(p for p in year_dir.iterdir() if p.is_dir()):
            if only_issue and issue_dir.name != only_issue:
                continue
            key = f"{year_dir.name}/{issue_dir.name}"
            if issue_pages(out_dir, key):
                keys.append(key)
    return keys


def log_assembly(assembly: IssueAssembly, path: Path) -> None:
    """Строка лога по собранному выпуску.

    Args:
        assembly: Итог сборки.
        path: Куда записан файл.
    """
    logger.info(
        "выпуск %s собран: %d полос, переносов через границу %d (составных %d, дублей половины %d), "
        "абзацев сшито %d, пропущено %d → %s",
        assembly.issue,
        len(assembly.pages),
        assembly.count(JoinKind.HYPHEN),
        assembly.count(JoinKind.COMPOUND),
        assembly.count(JoinKind.DUPLICATE),
        assembly.count(JoinKind.PARAGRAPH),
        len(assembly.missing),
        path,
    )
