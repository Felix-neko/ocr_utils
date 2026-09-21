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
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from ocr_utils.external_ocr_services import PROMPT_VERSION
from ocr_utils.external_ocr_services.boundary import (
    BoundaryChecker,
    BoundaryVerdict,
    CheckerStats,
    DoubtReason,
    JoinKind,
    Seam,
    apply_verdict,
    decide_boundary,
    doubt_reason,
)
from ocr_utils.external_ocr_services.hyphen_join import JoinRule, Morph, default_morph
from ocr_utils.external_ocr_services.numbering import NumberSource, suggest_page_numbers
from ocr_utils.external_ocr_services.ocr import output_paths, read_meta
from ocr_utils.external_ocr_services.heading_check import (
    LINES_PER_ARTICLE,
    CandidateLine,
    HeadingChecker,
    HeadingQuery,
    HintStatus,
)
from ocr_utils.external_ocr_services.reconcile import (
    BlockEdits,
    HeadingHint,
    PageRefs,
    ReconcileReport,
    candidate_lines,
    reconcile_headings,
)
from ocr_utils.external_ocr_services.render import _yaml_value, space_author_tags
from ocr_utils.external_ocr_services.schema import (
    BlockTag,
    DamageTag,
    ParseError,
    Stage,
    StructureTag,
    TocKind,
    parse_json_text,
)
from ocr_utils.external_ocr_services.structure import _MARKER_TEXT, _RUBRIC, join_paragraphs, paragraphs_of
from ocr_utils.external_ocr_services.toc import IssueToc, from_dict as toc_from_dict, normalize_title, title_matches

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


@dataclass
class PageEntry:
    """Полоса в sidecar: откуда взята и где начинается в тексте выпуска."""

    file: str  # путь полосы относительно out-dir без суффикса
    page_number: str | None  # напечатанный номер по ответу модели
    stage: str  # этап, ответ которого вошёл в текст (page / toc)
    offset: int  # смещение первого символа полосы в тексте выпуска (внутри сшитого абзаца — внутри него)
    line: int  # строка (с 1) того же места
    suggested_page_number: int | None = (
        None  # номер по соседям (numbering), когда напечатанного нет или он подозрителен
    )
    page_number_source: str = NumberSource.NONE.value  # каким номером пользуется сверка: printed / suggested / none


@dataclass
class JoinRecord:
    """Склейка на границе двух полос — для sidecar и проверки глазами."""

    after: str  # полоса перед границей (file)
    before: str  # полоса после границы
    kind: JoinKind
    word: str | None = None  # склеенное слово «снаб-жения» у hyphen/compound, что заменено у duplicate/model
    reason: DoubtReason | None = None  # почему стык показан модели; None — не показывался
    model: dict | None = (
        None  # ответ модели (boundary.BoundaryVerdict.as_dict) и «confirmed», если эвристика подтверждена
    )

    def as_dict(self) -> dict:
        """Для sidecar: перечисления строками, пустые поля модели опущены."""
        data = {"after": self.after, "before": self.before, "kind": self.kind.value, "word": self.word}
        if self.reason is not None:
            data["reason"] = self.reason.value
        if self.model is not None:
            data["model"] = self.model
        return data


@dataclass
class IssueAssembly:
    """Итог сборки выпуска: текст, привязка полос, склейки, пропуски."""

    issue: str  # «год/выпуск»
    text: str  # содержимое файла выпуска (шапка + тело)
    pages: list[PageEntry] = field(default_factory=list)
    joins: list[JoinRecord] = field(default_factory=list)
    missing: list[dict] = field(default_factory=list)  # {"file", "reason"}
    checked: CheckerStats = field(default_factory=CheckerStats)  # запросы к модели по стыкам этого выпуска
    repeated_rubrics: list[dict] = field(default_factory=list)  # убранные повторы <rubric>: {"rubric", "page"}
    toc_merged: list[dict] = field(default_factory=list)  # слитые блоки <toc> полос-продолжений: {"after", "before"}
    reconcile: ReconcileReport = field(default_factory=ReconcileReport)  # сверка `#`/<rubric> с оглавлением
    heading_checked: CheckerStats = field(default_factory=CheckerStats)  # вспомогательные запросы по заголовкам

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
            "joins": [join.as_dict() for join in self.joins],
            "missing": self.missing,
            "counts": {kind.value: self.count(kind) for kind in JoinKind},
            "checked": vars(self.checked),
            "heading_checked": vars(self.heading_checked),
            "repeated_rubrics": self.repeated_rubrics,
            "toc_merged": self.toc_merged,
            **self.reconcile.as_dict(),
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
    rel: Path = Path()  # путь полосы относительно корня, как пришёл (для полосок строк проверки)
    page_number: str | None = None
    stage: str = Stage.PAGE.value
    blocks: list[_Block] = field(default_factory=list)
    reason: str | None = None  # не None — полосы нет, в текст идёт комментарий
    toc_continues: bool = False  # полоса-продолжение оглавления/указателя (``toc.continues_previous`` этапа toc)
    refs: PageRefs | None = None  # id заголовков, рубрик и колонтитулов по ответу модели — для сверки с оглавлением


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
        return _PageBody(file, rel, reason=meta.get("error") or meta.get("parse_error") or "нет .json")
    if meta.get("error") or meta.get("parse_error"):
        return _PageBody(file, rel, reason=str(meta.get("error") or meta.get("parse_error")))
    stage = Stage(meta.get("stage") or Stage.PAGE)
    try:
        result = parse_json_text(paths.json.read_text(encoding="utf-8"), stage)
    except (ParseError, OSError) as error:
        return _PageBody(file, rel, reason=f".json не читается: {error}")
    continues = stage is Stage.TOC and result.toc is not None and bool(result.toc.continues_previous)
    refs = PageRefs(
        file,
        stage.value,
        None,
        list(result.headings),
        list(result.rubrics),
        result.running_header,
        result.running_footer,
        result.running_header_article_id,
        result.running_header_rubric_id,
        result.running_footer_article_id,
        result.running_footer_rubric_id,
        toc_kind=result.toc_kind.value,
    )
    return _PageBody(
        file, rel, result.page_number, stage.value, _blocks_of(result.content_markdown), None, continues, refs
    )


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


@dataclass
class _Pass:
    """Итог одного прохода сборки: блоки, начала полос, записи и собранные сомнительные стыки."""

    blocks: list[_Block] = field(default_factory=list)
    starts: list[tuple[int, int]] = field(default_factory=list)  # (индекс блока, смещение внутри) для каждой полосы
    pages: list[PageEntry] = field(default_factory=list)
    joins: list[JoinRecord] = field(default_factory=list)
    missing: list[dict] = field(default_factory=list)
    seams: list[Seam] = field(default_factory=list)  # сомнительные стыки — на проверку моделью
    toc_merged: list[dict] = field(default_factory=list)  # слитые блоки <toc> соседних полос: {"after", "before"}


def _assemble_pass(
    bodies: list[_PageBody],
    morph: Morph | None,
    rule: JoinRule,
    join_paragraphs_across: bool,
    collect_doubts: bool,
    verdicts: dict[int, BoundaryVerdict] | None,
) -> _Pass:
    """Один проход сборки по прочитанным полосам.

    Первый проход (``collect_doubts=True``, ``verdicts=None``) решает стыки эвристиками и собирает
    сомнительные; второй (``verdicts`` от модели) кладёт вердикты поверх эвристик. Стыки нумеруются
    одинаково в обоих проходах — по порядку границ, где хвост и голова простой текст.

    Args:
        bodies: Полосы выпуска по порядку (уже прочитанные).
        morph: Анализатор; ``None`` — без словаря.
        rule: Правило склейки переносов.
        join_paragraphs_across: Сшивать оборванные предложения.
        collect_doubts: Собирать ли сомнительные стыки в ``seams``.
        verdicts: Вердикты модели по номерам стыков (второй проход) или ``None``.
    """
    result = _Pass()
    blocks = result.blocks
    previous: _PageBody | None = None
    previous_first = 0  # индекс первого блока предыдущей полосы
    seam_index = 0
    for page in bodies:
        page_blocks = list(page.blocks)  # свой список — прочитанные полосы не меняем между проходами
        if page.reason is not None:
            result.missing.append({"file": page.file, "reason": page.reason})
            page_blocks = [_Block(f"<!-- полоса {page.file} не распознана: {page.reason} -->", False)]
        # Полоса-продолжение оглавления/указателя: её `<toc>` продолжает блок предыдущей полосы —
        # закрывающий `</toc>` предыдущей и открывающий `<toc>` этой убираются, список идёт одним блоком.
        if (
            previous is not None
            and page.toc_continues
            and previous.stage == Stage.TOC.value
            and blocks
            and blocks[-1].text.strip() == f"</{BlockTag.TOC}>"
            and page_blocks
            and page_blocks[0].text.strip() == f"<{BlockTag.TOC}>"
        ):
            blocks.pop()
            page_blocks = page_blocks[1:]
            result.toc_merged.append({"after": previous.file, "before": page.file})
        first = len(blocks)
        start = (first, 0)
        tail_index = _tail_index(blocks, previous_first) if previous is not None and blocks else None
        head_index = _head_index(page_blocks)
        if tail_index is not None and head_index is not None and page.reason is None:
            seam_index += 1
            tail, head = blocks[tail_index].text, page_blocks[head_index].text
            decision = decide_boundary(tail, head, morph, rule, join_paragraphs_across)
            reason = doubt_reason(tail, head, decision, morph) if collect_doubts or verdicts is not None else None
            model: dict | None = None
            if reason is not None and collect_doubts:
                # Плавающие блоки между хвостом и краем полосы (или краем и головой): полоска строк
                # для модели берётся шире, и она предупреждается, что нужная строка не у края.
                tail_floating = tail_index < len(blocks) - 1
                head_floating = head_index > 0
                result.seams.append(
                    Seam(
                        seam_index, Path(previous.rel), Path(page.rel), tail, head, reason, tail_floating, head_floating
                    )
                )
            if reason is not None and verdicts is not None and seam_index in verdicts:
                # Вердикт модели ложится поверх эвристики (или подтверждает её).
                verdict = verdicts[seam_index]
                decision, status = apply_verdict(tail, head, verdict, decision)
                model = {**verdict.as_dict(), "status": status.value}
            if decision is not None:
                blocks[tail_index] = _Block(decision.text, True)
                start = (tail_index, decision.head_start)
                result.joins.append(JoinRecord(previous.file, page.file, decision.kind, decision.word, reason, model))
                # Голова ушла в хвост; плавающие блоки перед ней (таблица, сноска) и всё остальное
                # идут следом за сшитым абзацем в прежнем порядке.
                page_blocks = page_blocks[:head_index] + page_blocks[head_index + 1 :]
            elif reason is not None:
                # Стык не сшит, но сомнителен — запись без склейки, чтобы вердикт был виден в sidecar.
                result.joins.append(JoinRecord(previous.file, page.file, JoinKind.NONE, None, reason, model))
        blocks.extend(page_blocks)
        result.starts.append(start)
        result.pages.append(PageEntry(page.file, page.page_number, page.stage, 0, 0))
        # Хвост следующей границы ищется от начала этой полосы; у сшитой полосы её начало — хвост
        # предыдущей, так что однобзацная полоса, целиком ушедшая в склейку, тоже даёт хвост.
        previous, previous_first = page, start[0]
    return result


def assemble_issue(
    out_dir: Path,
    issue_key: str,
    page_rels: list[Path],
    *,
    join_hyphens: bool = True,
    join_paragraphs_across: bool = True,
    rule: JoinRule = JoinRule.E,
    checker: BoundaryChecker | None = None,
    toc: IssueToc | None = None,
    heading_checker: HeadingChecker | None = None,
) -> IssueAssembly:
    """Собрать текст выпуска из готовых полос (в память; запись — :func:`write_issue`).

    С ``checker`` сборка идёт в два прохода: первый решает стыки эвристиками и собирает сомнительные,
    они уходят модели **одним запросом на выпуск** (``checker.check_issue``), второй проход кладёт
    вердикты поверх эвристик. Без сомнительных стыков запроса нет.

    Args:
        out_dir: Корень выхода с ``{год}/{выпуск}/{полоса}.json``.
        issue_key: «год/выпуск».
        page_rels: Полосы выпуска по порядку (пути относительно корня, суффикс любой).
        join_hyphens: Сшивать слова с дефисом на границе (иначе граница с дефисом — как без него).
        join_paragraphs_across: Сшивать оборванные предложения через пробел.
        rule: Правило склейки переносов.
        checker: Проверка сомнительных стыков моделью (``boundary.BoundaryChecker``); ``None`` — только эвристики.
        toc: «Содержание» выпуска для сверки заголовков; ``None`` — читается из ``toc.json`` рядом с полосами.
        heading_checker: Вспомогательный текстовый запрос по статьям, оставшимся без заголовка после
            сверки (``heading_check.HeadingChecker``); ``None`` — без него.

    Returns:
        :class:`IssueAssembly`: текст файла, смещения полос, склейки, пропуски, счётчики проверки,
        итог сверки с оглавлением.
    """
    morph = default_morph() if join_hyphens else None  # словарь грузится один раз на процесс
    checker_start = replace(checker.stats) if checker is not None else CheckerStats()  # снимок счётчиков до выпуска
    bodies = [_read_page(out_dir, rel) for rel in page_rels]
    result = _assemble_pass(bodies, morph, rule, join_paragraphs_across, checker is not None, None)
    if checker is not None and result.seams:
        verdicts = checker.check_issue(issue_key, result.seams)
        result = _assemble_pass(bodies, morph, rule, join_paragraphs_across, False, verdicts)
    # Сверка `#` и `<rubric>` с оглавлением выпуска: один заголовок на статью, одна рубрика на раздел;
    # номера полос без напечатанного номера выводятся по соседям.
    toc = toc_contents(out_dir, issue_key) if toc is None else toc
    heading_start = replace(heading_checker.stats) if heading_checker is not None else CheckerStats()
    hints = None
    if heading_checker is not None and toc is not None:
        # Первый проход сверки — на копии: узнать, кому не хватило заголовка; подсказки модели идут
        # во второй проход по исходным блокам (индексы подсказок — по ним).
        trial = deepcopy(result)
        _, trial_report = _reconcile(trial, bodies, toc)
        hints = _heading_hints(result, bodies, toc, trial_report, heading_checker, issue_key)
    anchors, reconciled = _reconcile(result, bodies, toc, hints)
    repeated = drop_repeated_rubrics(result)
    header = _header(issue_key, len(result.pages), len(result.missing))
    # Переводы строк у тегов автора удваиваются в самом конце — по ним же режется на абзацы выше.
    texts = [space_author_tags(block.text) for block in result.blocks]
    body = join_paragraphs(texts) if texts else ""
    text = header + body
    # Смещения: накопленная длина абзацев до нужного плюс смещение внутри него (внутреннее —
    # по тому же преобразованию префикса блока, чтобы удвоения перед головой учлись).
    offsets = [len(header)]
    for block_text in texts:
        offsets.append(offsets[-1] + len(block_text) + 2)  # «\n\n» между абзацами
    for entry, (index, inner) in zip(result.pages, result.starts):
        if index < len(result.blocks):
            entry.offset = offsets[index] + len(space_author_tags(result.blocks[index].text[:inner]))
        else:
            entry.offset = len(text)
        entry.line = text.count("\n", 0, entry.offset) + 1
    # Настоящие `#` и `<rubric>` — смещения в тексте для нарезки по статьям.
    for kind, entries in (("article", reconciled.articles), ("rubric", reconciled.rubrics)):
        for entry_dict in entries:
            block = anchors.get((kind, entry_dict["id"]))
            if block is not None and block < len(result.blocks):
                entry_dict["offset"] = offsets[block]
                entry_dict["line"] = text.count("\n", 0, offsets[block]) + 1
    checked = checker.stats - checker_start if checker is not None else CheckerStats()
    heading_checked = heading_checker.stats - heading_start if heading_checker is not None else CheckerStats()
    return IssueAssembly(
        issue_key,
        text,
        result.pages,
        result.joins,
        result.missing,
        checked,
        repeated,
        result.toc_merged,
        reconciled,
        heading_checked,
    )


def _heading_hints(
    result: _Pass,
    bodies: list[_PageBody],
    toc: IssueToc,
    trial: ReconcileReport,
    checker: HeadingChecker,
    issue_key: str,
) -> dict[str, HeadingHint]:
    """Спросить модель о статьях, оставшихся без заголовка после пробной сверки.

    Args:
        result: Исходный итог сборки (до правок сверки) — индексы блоков в подсказках относятся к нему.
        bodies: Прочитанные полосы.
        toc: «Содержание» выпуска.
        trial: Отчёт пробной сверки — статьи со ``status: missing`` уходят в запрос.
        checker: Вспомогательный запрос.
        issue_key: «год/выпуск».

    Returns:
        ``{id статьи: подсказка}`` — по всем статьям запроса, в том числе отклонённым (для sidecar).
    """
    missing = [entry for entry in trial.articles if entry["status"] == "missing"]
    if not missing:
        return {}
    refs, page_of = _page_refs(result, bodies)
    blocks = [block.text for block in result.blocks]
    plain = [block.plain for block in result.blocks]
    floating = [block.floating for block in result.blocks]
    queries: list[HeadingQuery] = []
    for entry in missing:
        article = toc.article_by_id(entry["id"])
        if article is None:
            continue
        lines = [
            CandidateLine(number, block, refs[page_of[block]].file, refs[page_of[block]].number, text, in_window)
            for number, (block, in_window, text) in enumerate(
                candidate_lines(blocks, floating, plain, page_of, refs, entry["toc_page"], LINES_PER_ARTICLE), 1
            )
        ]
        authors = [author["name"] for author in article.authors if author.get("name")]
        queries.append(HeadingQuery(article.id, article.title, authors, article.page, lines))
    if not queries:
        return {}
    toc_entries = [
        {"id": a.id, "title": a.title, "authors": [x["name"] for x in a.authors if x.get("name")], "page": a.page}
        for a in toc.articles
    ]
    verdicts = checker.check_issue(issue_key, toc_entries, queries)
    rubric_titles = [entry["title"] for entry in toc.rubric_entries]
    hints: dict[str, HeadingHint] = {}
    for article_id, verdict in verdicts.items():
        block = verdict.block if verdict.status is HintStatus.ACCEPTED else None
        # Маркер с названием рубрики оглавления — это шапка раздела, а не название статьи (1976/10:
        # «ОФИЦИАЛЬНЫЙ ОТДЕЛ» за «Типовые положения…»); годится, только если статья так и называется.
        if block is not None and (marker := _MARKER_TEXT.match(blocks[block])) is not None:
            article = toc.article_by_id(article_id)
            if title_matches(marker.group(1), rubric_titles) and not title_matches(marker.group(1), [article.title]):
                verdict.status, verdict.notes = HintStatus.REJECTED, f"маркер рубрики, не название; {verdict.notes}"
                block = None
        hints[article_id] = HeadingHint(block, verdict.as_dict())
    return hints


def toc_contents(out_dir: Path, issue_key: str) -> IssueToc | None:
    """«Содержание» выпуска из ``toc.json`` рядом с полосами (с id статей и рубрик); нет файла — ``None``.

    Args:
        out_dir: Корень выхода.
        issue_key: «год/выпуск».
    """
    path = out_dir / issue_key / "toc.json"
    if not path.is_file():
        return None
    try:
        tocs = toc_from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError) as error:
        logger.warning("%s: toc.json не читается (%s) — сверка заголовков с оглавлением пропущена", issue_key, error)
        return None
    return tocs.get(TocKind.CONTENTS)


def _reconcile(
    result: _Pass, bodies: list[_PageBody], toc: IssueToc | None, hints: dict[str, HeadingHint] | None = None
) -> tuple[dict[tuple[str, str], int], ReconcileReport]:
    """Сверить заголовки и рубрики выпуска с оглавлением и применить правки к блокам на месте.

    Номера полос: напечатанные проверяются по соседям, отсутствующие выводятся (``numbering``) и
    пишутся в записи sidecar; сверке уходит номер по ``source``.

    Args:
        result: Итог прохода сборки; ``blocks``, ``starts`` правятся на месте, ``pages`` дополняются номерами.
        bodies: Прочитанные полосы (в том же порядке, что ``result.pages``).
        toc: «Содержание» выпуска; ``None`` — правятся только полосы оглавления.
        hints: Подсказки вспомогательного запроса по заголовкам (``_heading_hints``); ``None`` — без них.

    Returns:
        ``(якоря, отчёт)``: якоря — индексы блоков настоящих `#`/`<rubric>` после правок по метке.
    """
    refs, page_of = _page_refs(result, bodies)
    edits, report = reconcile_headings(
        [block.text for block in result.blocks],
        [block.plain for block in result.blocks],
        [block.floating for block in result.blocks],
        page_of,
        refs,
        toc,
        hints,
    )
    anchors = _apply_edits(result, edits)
    return anchors, report


def _page_refs(result: _Pass, bodies: list[_PageBody]) -> tuple[list[PageRefs], list[int]]:
    """Данные полос для сверки (с номерами по соседям) и полоса каждого блока.

    Номера: напечатанные проверяются по соседям, отсутствующие выводятся (``numbering``) и пишутся в
    записи sidecar (``result.pages``).

    Args:
        result: Итог прохода сборки.
        bodies: Прочитанные полосы (в том же порядке, что ``result.pages``).

    Returns:
        ``(данные полос, индекс полосы для каждого блока)``.
    """
    guesses = suggest_page_numbers([body.page_number for body in bodies])
    merged_into_previous = {item["before"] for item in result.toc_merged}
    refs: list[PageRefs] = []
    for body, entry, guess in zip(bodies, result.pages, guesses):
        entry.suggested_page_number = guess.suggested
        entry.page_number_source = guess.source.value
        page_refs = body.refs if body.refs is not None else PageRefs(body.file, body.stage)
        page_refs.number = guess.number
        page_refs.toc_merged = body.file in merged_into_previous
        refs.append(page_refs)
    # Полоса каждого блока: последняя полоса, начавшаяся не позже него.
    page_of: list[int] = []
    starts = [start for start, _ in result.starts]
    current = 0
    for index in range(len(result.blocks)):
        while current + 1 < len(starts) and starts[current + 1] <= index:
            current += 1
        page_of.append(current)
    return refs, page_of


def _apply_edits(result: _Pass, edits: BlockEdits) -> dict[tuple[str, str], int]:
    """Применить правки сверки к блокам: удаления, замены, вставки; начала полос пересчитать.

    Начало полосы, попавшее на убранный блок, переезжает на следующий уцелевший; вставки перед блоком
    относятся к полосе этого блока (комментарий с id и восстановленный заголовок идут в её начало).

    Args:
        result: Итог прохода; ``blocks`` и ``starts`` правятся на месте.
        edits: Правки по индексам исходного списка блоков.

    Returns:
        Индексы блоков настоящих `#`/`<rubric>` после правок по метке (первое вхождение).
    """
    if not edits.delete and not edits.replace and not edits.insert_before:
        return {label: index for label, index in reversed(edits.anchors)}
    old_anchor = {index: label for label, index in reversed(edits.anchors)}
    new_blocks: list[_Block] = []
    new_index: list[int] = []  # индекс, куда переезжает начало полосы, стоявшее на старом блоке
    anchors: dict[tuple[str, str], int] = {}
    for index, block in enumerate(result.blocks):
        first_here = len(new_blocks)
        for text, label in edits.insert_before.get(index, []):
            if label is not None:
                anchors.setdefault(label, len(new_blocks))
            new_blocks.append(_Block(text, _is_plain(text), bool(_FLOATING_START.match(text))))
        new_index.append(first_here)
        if index in edits.delete:
            continue
        if index in old_anchor:
            anchors.setdefault(old_anchor[index], len(new_blocks))
        if index in edits.replace:
            block = _Block(edits.replace[index], _is_plain(edits.replace[index]), block.floating)
        new_blocks.append(block)
    result.starts = [
        (new_index[index] if index < len(new_index) else len(new_blocks), 0 if index in edits.delete else inner)
        for index, inner in result.starts
    ]
    result.blocks = new_blocks
    return anchors


def drop_repeated_rubrics(result: _Pass) -> list[dict]:
    """Убрать повтор `<rubric>` с тем же текстом, что у предыдущей встреченной, — в тексте выпуска подряд
    (хотя бы и через другой контент) две одинаковые рубрики идти не могут; чередование A → B → A остаётся.

    Модель пишет рубрику из списка статей над каждой статьёй раздела (0280_1L 1976/12: напечатана
    шапка «Резервы — на службу…», а в теге — рубрика раздела из списка), и в целиковом тексте
    рубрика раздела повторялась бы перед каждой статьёй. Начала полос, попавшие на убранный блок,
    переезжают на следующий блок.

    Args:
        result: Итог прохода сборки; ``blocks`` и ``starts`` правятся на месте.

    Returns:
        Убранные рубрики: ``[{"rubric", "page"}]`` — текст и полоса, где стоял повтор.
    """
    dropped: list[dict] = []
    keep: list[bool] = []
    last: str | None = None
    for block in result.blocks:
        match = _RUBRIC.match(block.text)
        if match is None:
            keep.append(True)
            continue
        name = normalize_title(match.group(1))
        keep.append(name != last)
        last = name
    if all(keep):
        return dropped
    # Новые индексы блоков: убранный блок указывает на следующий уцелевший (начало полосы — туда же).
    new_index: list[int] = []
    survivors = 0
    for kept in keep:
        new_index.append(survivors)
        survivors += kept
    for index, (block, kept) in enumerate(zip(result.blocks, keep)):
        if not kept:
            dropped.append({"rubric": _RUBRIC.match(block.text).group(1), "page": _page_of_block(result, index)})
    result.starts = [
        (new_index[index], inner if index < len(keep) and keep[index] else 0) for index, inner in result.starts
    ]
    result.blocks = [block for block, kept in zip(result.blocks, keep) if kept]
    return dropped


def _page_of_block(result: _Pass, index: int) -> str:
    """Полоса, которой принадлежит блок с данным индексом (последняя полоса, начавшаяся не позже него).

    Args:
        result: Итог прохода сборки.
        index: Индекс блока в ``result.blocks``.
    """
    file = result.pages[0].file if result.pages else ""
    for entry, (start, _) in zip(result.pages, result.starts):
        if start <= index:
            file = entry.file
    return file


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
        "абзацев сшито %d, стыков проверено моделью %d (из кэша %d, переписано %d, сбоев %d, $%.4f), "
        "статей с `#` %d из %d (восстановлено %d, запросов по заголовкам %d, из кэша %d, $%.4f), фантомов `#` убрано %d, лишних <rubric> %d, "
        "повторов рубрик убрано %d, блоков toc слито %d, пропущено %d → %s",
        assembly.issue,
        len(assembly.pages),
        assembly.count(JoinKind.HYPHEN),
        assembly.count(JoinKind.COMPOUND),
        assembly.count(JoinKind.DUPLICATE),
        assembly.count(JoinKind.PARAGRAPH),
        assembly.checked.requests,
        assembly.checked.cache_hits,
        assembly.count(JoinKind.MODEL),
        assembly.checked.errors,
        assembly.checked.cost_usd,
        sum(1 for entry in assembly.reconcile.articles if entry["status"] in ("found", "restored")),
        len(assembly.reconcile.articles),
        sum(1 for entry in assembly.reconcile.articles if entry["status"] == "restored"),
        assembly.heading_checked.requests,
        assembly.heading_checked.cache_hits,
        assembly.heading_checked.cost_usd,
        len(assembly.reconcile.phantom_headings),
        len(assembly.reconcile.phantom_rubrics),
        len(assembly.repeated_rubrics),
        len(assembly.toc_merged),
        len(assembly.missing),
        path,
    )
