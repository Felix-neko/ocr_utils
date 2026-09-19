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
    return _PageBody(file, rel, result.page_number, stage.value, _blocks_of(result.content_markdown))


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

    Returns:
        :class:`IssueAssembly`: текст файла, смещения полос, склейки, пропуски, счётчики проверки.
    """
    morph = default_morph() if join_hyphens else None  # словарь грузится один раз на процесс
    checker_start = replace(checker.stats) if checker is not None else CheckerStats()  # снимок счётчиков до выпуска
    bodies = [_read_page(out_dir, rel) for rel in page_rels]
    result = _assemble_pass(bodies, morph, rule, join_paragraphs_across, checker is not None, None)
    if checker is not None and result.seams:
        verdicts = checker.check_issue(issue_key, result.seams)
        result = _assemble_pass(bodies, morph, rule, join_paragraphs_across, False, verdicts)
    header = _header(issue_key, len(result.pages), len(result.missing))
    body = join_paragraphs([block.text for block in result.blocks]) if result.blocks else ""
    text = header + body
    # Смещения: накопленная длина абзацев до нужного плюс смещение внутри него.
    offsets = [len(header)]
    for block in result.blocks:
        offsets.append(offsets[-1] + len(block.text) + 2)  # «\n\n» между абзацами
    for entry, (index, inner) in zip(result.pages, result.starts):
        entry.offset = offsets[index] + inner if index < len(result.blocks) else len(text)
        entry.line = text.count("\n", 0, entry.offset) + 1
    checked = checker.stats - checker_start if checker is not None else CheckerStats()
    return IssueAssembly(issue_key, text, result.pages, result.joins, result.missing, checked)


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
        "абзацев сшито %d, стыков проверено моделью %d (из кэша %d, переписано %d, сбоев %d, $%.4f), пропущено %d → %s",
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
        len(assembly.missing),
        path,
    )
