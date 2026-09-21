"""Обход пака по выпускам: полосы оглавления -> список статей -> остальные полосы -> fallback.

На выпуск: (1) полосы, помеченные в базе как «Содержание» или указатель, распознаются этапом
``toc`` («предварительно оглавление» — модель решает сама) и сливаются в оглавление выпуска
(``toc.json`` / ``toc.md``); полоса, которую модель не признала оглавлением, понижается: её
вклад в оглавление остаётся (``имя.toc.json``), а сама она идёт ещё и этапом ``page``;
(2) остальные полосы идут этапом ``page`` с рубриками и статьями «Содержания» в промпте;
(3) если модель на обычной полосе
увидела оглавление, которого в базе нет (и нет вето ``force_is_not_toc``), — предупреждение и,
по ``--on-missed-toc``, повтор выпуска: найденные полосы распознаются как оглавление, список
пересобирается, обычные полосы идут заново (у них меняется ``toc_hash``, готовые с прежним
списком не считаются сделанными). Один круг повтора на выпуск. По ``--redo-scope structured``
(умолчание) на этом круге заново запрашиваются не все обычные полосы, а только те, на которых
списки выпуска вообще могут что-то изменить (см. :func:`redo_reason`); остальные берутся с диска.

Запросы к сети — в пуле потоков ``--jobs`` внутри этапа; выпуски идут последовательно, потому
что этап 2 зависит от этапа 1. В базу ничего не пишется: пропущенные оглавления складываются в
``missed_toc.txt``, а теги ставит человек в CVAT.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import partial
from pathlib import Path

import click

from ocr_utils.external_ocr_services import toc as toc_module
from ocr_utils.external_ocr_services.assemble import assemble_issue, log_assembly, write_issue
from ocr_utils.external_ocr_services.boundary import BoundaryChecker, JoinKind
from ocr_utils.external_ocr_services.heading_check import HeadingChecker
from ocr_utils.external_ocr_services.client import OpenRouterClient
from ocr_utils.external_ocr_services.models import ModelSpec
from ocr_utils.external_ocr_services.ocr import (
    PageJob,
    PassChoice,
    RunOptions,
    is_done,
    load_result,
    output_paths,
    read_meta,
    recognize_page,
    recognize_with_second_pass,
    save_demoted_toc,
    write_meta,
)
from ocr_utils.external_ocr_services.pages import PageFlags, flags_for, group_by_issue, list_pages
from ocr_utils.external_ocr_services.schema import PageResult, Stage, StructureTag, TocKind
from ocr_utils.external_ocr_services.toc import normalize_title

logger = logging.getLogger(__name__)

# Файл в корне out-dir: полосы, где модель увидела оглавление, а база молчит, — дописывается за
# каждый выпуск без повтора; человеку на разметку в CVAT.
MISSED_LIST = "missed_toc.txt"
# Обратный случай: полосы, помеченные оглавлением в базе, которые модель сочла обычными и которые
# после этапа toc пошли ещё и этапом page; человеку — снять тег или поставить вето в CVAT.
DEMOTED_LIST = "demoted_toc.txt"
# Оглавление выпуска в папке «{год}/{выпуск}» под out-dir: машинный JSON и тот же список глазами.
TOC_JSON = "toc.json"
TOC_MD = "toc.md"


class OnMissedToc(StrEnum):
    """Что делать, когда модель увидела оглавление на полосе, которую база считает обычной (``--on-missed-toc``).

    ``REDO`` — перераспознать выпуск с этими полосами как оглавлением (умолчание); ``ASK`` —
    спросить в терминале (без терминала = ``SKIP``); ``SKIP`` — только записать в ``missed_toc.txt``.
    """

    ASK = "ask"
    REDO = "redo"
    SKIP = "skip"


class RedoScope(StrEnum):
    """Какие обычные полосы запрашивать заново на круге повтора выпуска (``--redo-scope``).

    ``STRUCTURED`` (умолчание) — только полосы, на которых списки выпуска могут что-то изменить
    (см. :func:`redo_reason`), остальные берутся с диска с новым ``toc_hash``; ``ALL`` — все
    обычные полосы, как было до появления опции.
    """

    ALL = "all"
    STRUCTURED = "structured"


class RedoReason(StrEnum):
    """Почему полоса идёт в сеть заново на круге повтора; пишется в лог по полосе.

    ``DEMOTED`` — полоса понижена (в базе оглавление, модель: none) и идёт этапом page с новым
    списком; ``HEADING`` — в теле есть заголовок ``#``/``##``; ``STRUCTURE_TAG`` — есть ``<rubric>``,
    ``<marker>`` (так помечается рубрика-надпись не из списка), ``<author>`` или ``<position>``;
    ``TITLE_OR_AUTHORS`` — модель заполнила ``title`` / ``rubric`` / ``authors``; всё это — не считая
    текста колонтитула (``running_header`` / ``running_footer``); ``STRUCTURE_EDITS`` —
    пост-обработка что-то правила (понижала ``#``, снимала утёкшие ``##``, переносила рубрики);
    ``ARTICLE_START`` — номер страницы совпал с началом статьи по новому «Содержанию»;
    ``NO_FIRST_PASS`` — результата первого круга на диске нет или он битый.
    """

    DEMOTED = "demoted"
    HEADING = "heading"
    STRUCTURE_TAG = "structure_tag"
    TITLE_OR_AUTHORS = "title_or_authors"
    STRUCTURE_EDITS = "structure_edits"
    ARTICLE_START = "article_start"
    NO_FIRST_PASS = "no_first_pass"


# Строка-заголовок markdown любого уровня — признак полосы, чувствительной к спискам выпуска.
_HEADING = re.compile(r"^#{1,6} (.+?)\s*$", re.M)
# Fenced-блок иллюстрации целиком — вырезается перед поиском заголовков.
_FENCED = re.compile(r"^[ \t]*```[^\n]*\n.*?^[ \t]*```[ \t]*$", re.M | re.S)
# Рубрика или маркер с текстом: модель дублирует в них рубрику из колонтитула почти на каждой полосе
# (МТС 1991/02: «ПРОБЛЕМЫ И СУЖДЕНИЯ» на с. 4–9), а колонтитул от списков выпуска не зависит.
_RUBRIC_OR_MARKER = re.compile(rf"<({StructureTag.RUBRIC}|{StructureTag.MARKER})>\*{{0,2}}(.+?)\*{{0,2}}</\1>", re.S)
# Теги, которые считаются признаком всегда: автор и должность привязываются к статье из списка.
_AUTHOR_TAGS = (StructureTag.AUTHOR, StructureTag.POSITION)


# Колонки сводки прогона; пересобирается по всем .meta.json под out-dir. Ключи совпадают с
# ключами meta полосы (см. ocr.recognize_page), лишние ключи meta в сводку не попадают.
SUMMARY_FIELDS = (
    "page",
    "stage",
    "toc_kind_expected",
    "toc_hash",
    "toc_demoted",
    "redo_kept",
    "articles_in_prompt",
    "second_pass_reason",
    "second_pass_chosen",
    "model",
    "provider",
    "json_mode_used",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "cost_usd",
    "latency_s",
    "attempts",
    "cache_hit",
    "cache_entry",
    "page_number",
    "toc_kind",
    "title",
    "title_in_list",
    "headings",
    "heading_ids",
    "rubric_ids",
    "content_chars",
    "toc_articles",
    "tags",
    "is_damaged",
    "hyphens_joined",
    "masked",
    "repaired_escapes",
    "retry_note",
    "messages",
    "parse_error",
    "error",
)


@dataclass
class PipelineParams:
    """Параметры прогона из CLI: пути, теги полос, параллелизм, отбор входа, поведение fallback."""

    in_dir: Path  # корень входа с раскладкой {год}/{выпуск}/{полоса}
    out_dir: Path  # корень выхода той же раскладки: .md, .json, .meta.json, toc.*, summary.csv
    options: RunOptions  # всё, что касается одного запроса: тайлы, модель, второй проход, debug-dir
    flags: dict[str, PageFlags] | None = None  # из базы или списков; None — база не задана
    jobs: int = 4  # потоков на сетевые запросы внутри этапа; ограничение — лимиты провайдера, не CPU
    skip_done: bool = False  # не запрашивать полосы, у которых выход на месте и совпадает toc_hash
    on_missed_toc: OnMissedToc = OnMissedToc.REDO  # что делать с оглавлением вне базы
    redo_scope: RedoScope = RedoScope.STRUCTURED  # какие обычные полосы запрашивать заново на круге повтора
    pages_file: Path | None = None  # --pages: явный список относительных путей вместо обхода
    only_year: str | None = None  # --only-year / --only-issue: отбор по первым папкам пути
    only_issue: str | None = None
    limit: int | None = None  # --limit: первые N полос после отбора, для проб
    # После каждого выпуска собирать его в один markdown (`assemble.py`): `{год}/{выпуск}.md` и
    # sidecar `.pages.json` рядом с папкой полос. Выключается ``--no-assemble``.
    assemble: bool = True
    # При сборке показывать сомнительные стыки полос модели — один запрос на выпуск, полоски строк
    # обеих полос (`boundary.py`). Выключается ``--no-check-boundaries``.
    check_boundaries: bool = True
    # Спрашивать модель текстом о статьях без `#` после сверки с оглавлением (`heading_check.py`).
    check_headings: bool = True

    def __post_init__(self) -> None:
        # Строка из старых вызовов («redo») → перечисление; чужое значение падает сразу.
        self.on_missed_toc = OnMissedToc(self.on_missed_toc)
        self.redo_scope = RedoScope(self.redo_scope)


@dataclass
class PipelineStats:
    """Счётчики прогона для итоговой строки лога и тестов; копятся по всем выпускам."""

    issues: int = 0  # выпусков обработано (повтор выпуска не считается вторым)
    pages: int = 0  # полос на входе после отбора
    requests: int = 0  # полос, ушедших в сеть (второй проход — та же полоса, не второй запрос)
    reused: int = 0  # полос, взятых готовыми с диска (--skip-done или круг повтора)
    failed: int = 0  # полос без результата: сбой сети, не разобранный JSON, исчерпаны попытки
    cost_usd: float = 0.0  # сумма cost_usd по meta запрошенных полос (оба прохода, без эха формата и без кэша)
    cache_hits: int = 0  # полос, чей финальный ответ взят из кэша запросов (--cache-dir), сеть не тратилась
    toc_pages: int = 0  # полос, которые база (или списки) считает оглавлением/указателем
    unknown_pages: int = 0  # полос без записи в базе
    second_passes: int = 0  # полос, ушедших во второй проход
    second_pass_kept_first: int = 0  # из них оставлен первый проход (страховка)
    missed: list[str] = field(default_factory=list)  # «выпуск: полосы», где оглавление не в базе
    redone_issues: list[str] = field(default_factory=list)  # выпуски, прошедшие круг повтора
    demoted_toc: list[str] = field(default_factory=list)  # «выпуск: полосы», которые модель не признала оглавлением
    redo_kept: int = 0  # полос, оставленных без повтора на круге повтора (--redo-scope structured)
    page_messages: int = 0  # полос с замечаниями пост-обработки (meta.messages)
    page_masked: int = 0  # полос, где при разборе замаскирован мусор модели (meta.masked)
    heading_ids_model: int = 0  # `#`, чей id статьи дала модель (согласован с текстом)
    heading_ids_other: int = 0  # `#`, чей id код нашёл по названию (модель id не дала или дала чужой)
    boundary_requests: int = 0  # запросов по стыкам полос при сборке (по одному на выпуск с сомнительными стыками)
    boundary_rewritten: int = 0  # стыков, переписанных по вердикту модели
    boundary_cost_usd: float = 0.0  # их стоимость (входит и в cost_usd)
    heading_requests: int = 0  # вспомогательных запросов по статьям без `#` при сборке
    heading_restored: int = 0  # статей, получивших `#` по ответу модели
    heading_cost_usd: float = 0.0  # их стоимость (входит и в cost_usd)

    def __add__(self, other: PipelineStats) -> PipelineStats:
        """Сумма счётчиков двух прогонов (этапов, выпусков): числа складываются, списки склеиваются.

        Ни один операнд не меняется — так этапы и выпуски отдают свои счётчики возвращаемым
        значением, а вызывающий складывает их сам.

        Args:
            other: Счётчики, которые прибавляются к этим.

        Returns:
            Новый ``PipelineStats`` с суммой всех полей.
        """
        return PipelineStats(
            issues=self.issues + other.issues,
            pages=self.pages + other.pages,
            requests=self.requests + other.requests,
            reused=self.reused + other.reused,
            failed=self.failed + other.failed,
            cost_usd=self.cost_usd + other.cost_usd,
            cache_hits=self.cache_hits + other.cache_hits,
            toc_pages=self.toc_pages + other.toc_pages,
            unknown_pages=self.unknown_pages + other.unknown_pages,
            second_passes=self.second_passes + other.second_passes,
            second_pass_kept_first=self.second_pass_kept_first + other.second_pass_kept_first,
            missed=self.missed + other.missed,
            redone_issues=self.redone_issues + other.redone_issues,
            demoted_toc=self.demoted_toc + other.demoted_toc,
            redo_kept=self.redo_kept + other.redo_kept,
            page_messages=self.page_messages + other.page_messages,
            page_masked=self.page_masked + other.page_masked,
            heading_ids_model=self.heading_ids_model + other.heading_ids_model,
            heading_ids_other=self.heading_ids_other + other.heading_ids_other,
            boundary_requests=self.boundary_requests + other.boundary_requests,
            boundary_rewritten=self.boundary_rewritten + other.boundary_rewritten,
            boundary_cost_usd=self.boundary_cost_usd + other.boundary_cost_usd,
            heading_requests=self.heading_requests + other.heading_requests,
            heading_restored=self.heading_restored + other.heading_restored,
            heading_cost_usd=self.heading_cost_usd + other.heading_cost_usd,
        )


def _recognize_one(
    client: OpenRouterClient, spec: ModelSpec, params: PipelineParams, job: PageJob
) -> tuple[PageJob, dict, PageResult | None]:
    """Одна полоса в потоке пула: первый проход и, если полоса повреждена, второй.

    Args:
        client: Клиент OpenRouter, общий на прогон.
        spec: Модель из реестра.
        params: Параметры прогона: корни входа/выхода и настройки запроса (``options``).
        job: Задание на полосу.

    Returns:
        ``(job, meta, result)``: то же задание (чтобы сопоставить ответ пула с полосой), meta
        полосы как записана в ``.meta.json`` и разобранный результат — ``None`` при сбое сети или
        разбора (причина — в ``meta["error"]`` / ``meta["parse_error"]``).
    """
    # Второй проход (по умолчанию выключен, ``--second-pass``) — обёртка над recognize_page: та же полоса ещё раз
    # с подсказками первого ответа, если модель сочла её повреждённой; выбор финала — внутри.
    recognize = recognize_with_second_pass if params.options.second_pass else recognize_page
    meta, result = recognize(client, spec, params.in_dir / job.rel, job, params.out_dir, params.options)
    return job, meta, result


def _recognize_many(
    client: OpenRouterClient, spec: ModelSpec, params: PipelineParams, jobs: list[PageJob], reuse: bool
) -> tuple[dict[Path, PageResult | None], PipelineStats]:
    """Полосы пачкой в пуле потоков; готовые (по ``is_done``) не запрашиваются, если ``reuse``.

    Файлы выхода пишет сам ``recognize_*`` — здесь только счётчики и лог.

    Args:
        client: Клиент OpenRouter, общий на прогон.
        spec: Модель из реестра.
        params: Параметры прогона: пути, ``jobs`` (размер пула), настройки запроса.
        jobs: Задания на полосы одного этапа (у всех одинаковые списки выпуска).
        reuse: Брать ли готовые результаты с диска вместо запроса (``--skip-done`` или круг повтора).

    Returns:
        ``(результаты, счётчики)``: результат каждой полосы из ``jobs`` по её относительному пути —
        ``None`` у сбойных, чтобы вызывающий отличал «сбой» от «полосы не было» (такой ключ
        отсутствует); счётчики этого этапа — запросы, взятые с диска, сбои, стоимость, вторые проходы.
    """
    results: dict[Path, PageResult | None] = {}
    stats = PipelineStats()
    todo: list[PageJob] = []
    for job in jobs:
        # Готовая полоса: выход на месте, без ошибки, тот же этап, тот же вид/отпечаток списков.
        # Её .json перечитывается тем же разбором, что и ответ модели; битый файл — в очередь.
        if reuse and is_done(params.out_dir, job):
            results[job.rel] = load_result(params.out_dir, job)
            if results[job.rel] is not None:
                stats.reused += 1
                continue
        todo.append(job)
    if not todo:
        return results, stats

    # Пул потоков, не процессов: работа — ожидание сети, GIL не мешает; тайлы режутся в потоке
    # перед запросом. pool.map отдаёт результаты в порядке очереди, счётчики правятся в одном потоке.
    work = partial(_recognize_one, client, spec, params)
    with ThreadPoolExecutor(max_workers=max(1, params.jobs)) as pool:
        for job, meta, result in pool.map(work, todo):
            stats.requests += 1
            # cost_usd в meta — сумма обоих проходов по удачным ответам; выброшенные ответы (эхо
            # response_format) лежат отдельно в cost_usd_wasted и в итог не входят. Ответ из кэша
            # запросов денег не стоил: в meta цена историческая, в итог прогона не идёт.
            if meta.get("cache_hit"):
                stats.cache_hits += 1
            else:
                stats.cost_usd += float(meta.get("cost_usd") or 0.0)
            # Второй проход был — есть причина; какой проход ушёл в финал, пишет second_pass_chosen.
            if meta.get("second_pass_reason"):
                stats.second_passes += 1
                stats.second_pass_kept_first += meta.get("second_pass_chosen") == PassChoice.PASS1
            if meta.get("messages"):
                stats.page_messages += 1
            if meta.get("masked"):
                stats.page_masked += 1
            # Источник id статей у `#` (meta.structure.heading_ids): доля не от модели — метрика промпта.
            ids = (meta.get("structure") or {}).get("heading_ids") or {}
            stats.heading_ids_model += int(ids.get("model") or 0)
            stats.heading_ids_other += int(ids.get("title") or 0)  # «wrong» — подмножество «title»
            # В строке лога — сетевая ошибка, иначе ошибка разбора, иначе «ok».
            status = meta.get("error") or meta.get("parse_error") or "ok"
            if result is None:
                stats.failed += 1
            logger.info(
                "%s [%s] %s%s: %s tok → %s tok, $%s, %.0f с",
                job.rel,
                job.stage,
                status,
                " (из кэша)" if meta.get("cache_hit") else "",
                meta.get("prompt_tokens", "-"),
                meta.get("completion_tokens", "-"),
                meta.get("cost_usd", "-"),
                meta.get("latency_s") or 0.0,
            )
            results[job.rel] = result
    return results, stats


def build_issue_toc(
    pages: list[Path], toc_pages: dict[Path, TocKind], results: dict[Path, PageResult | None]
) -> dict[TocKind, toc_module.IssueToc]:
    """Слить результаты этапа toc в оглавления выпуска по видам, в порядке полос.

    «Содержание» и «Годовой указатель» сливаются порознь: у каждого вида свой список полос, и
    указатель в промпт обычных полос не идёт. Вид без единой удачной полосы в ответе отсутствует.

    Args:
        pages: Все полосы выпуска по порядку имён — задаёт порядок слияния.
        toc_pages: Полосы этапа toc → их вид (``CONTENTS`` / ``INDEX``).
        results: Результаты этапа toc по полосам; ``None`` — сбой.

    Returns:
        Слитые оглавления по видам; вид без единой удачной полосы в словаре отсутствует.
    """
    tocs: dict[TocKind, toc_module.IssueToc] = {}
    for kind in toc_module.KINDS:
        # Порядок — по ``pages`` (имена полос в выпуске отсортированы), а не по порядку ответов пула:
        # слияние опирается на continues_previous, и «предыдущая полоса» должна быть предыдущей по
        # номеру. Сбойные полосы (None) и ответы без блока toc пропускаются.
        ordered = [
            (rel.as_posix(), results[rel].toc)
            for rel in pages
            if toc_pages.get(rel) == kind and results.get(rel) is not None and results[rel].toc is not None
        ]
        if ordered:
            tocs[kind] = toc_module.merge_pages(kind, ordered)
    return tocs


def _write_issue_toc(out_dir: Path, issue_key: str, tocs: dict[TocKind, toc_module.IssueToc]) -> None:
    """``toc.json`` и ``toc.md`` в папку выпуска; перезаписываются на каждом проходе выпуска.

    Args:
        out_dir: Корень выхода.
        issue_key: Ключ выпуска (``«1966/03»``) — папка под out_dir.
        tocs: Оглавления выпуска по видам из ``build_issue_toc``.
    """
    issue_dir = out_dir / issue_key
    issue_dir.mkdir(parents=True, exist_ok=True)
    # JSON — полная структура (рубрики → статьи с авторами, список полос) для повторов и сборки
    # выпуска; indent=1 — чтобы diff между прогонами читался построчно.
    (issue_dir / TOC_JSON).write_text(
        json.dumps(toc_module.to_dict(tocs), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    # Markdown — то же самое глазами: проверить, что модель прочла оглавление правильно.
    (issue_dir / TOC_MD).write_text(toc_module.to_markdown(tocs), encoding="utf-8")


def decide_redo(mode: OnMissedToc, issue_key: str, missed: list[tuple[Path, TocKind]]) -> bool:
    """Перераспознавать ли выпуск после находки оглавления вне базы.

    ``ASK`` спрашивает в терминале; без терминала (фоновый прогон с ``< /dev/null``) ведёт себя как
    ``SKIP`` — молча пересчитывать чужой счёт нельзя. Предупреждение в лог пишется при любом
    режиме: теги в базе всё равно должен поставить человек, повтор их не заменяет.

    Args:
        mode: Режим из ``--on-missed-toc``.
        issue_key: Ключ выпуска — для лога и вопроса.
        missed: Полосы, где модель увидела оглавление, с видом по её мнению.

    Returns:
        ``True`` — выпуск надо перераспознать с этими полосами как оглавлением; ``False`` — только
        записать их в ``missed_toc.txt``.
    """
    # Строки вида «  1966/03/IMG_0012.jpg  # contents» — тот же формат, что у toc_pages.txt
    # (там имя без папки выпуска), чтобы переносить в списки --toc-lists без правки.
    lines = "\n".join(f"  {rel.as_posix()}  # {kind}" for rel, kind in missed)
    logger.warning(
        "ВНИМАНИЕ: в выпуске %s модель считает оглавлением полосы, не помеченные в базе:\n%s\n"
        "Проставьте теги в CVAT («Оглавление»/«Годовой указатель» или вето «Не оглавление») и заберите базу from-cvat.",
        issue_key,
        lines,
    )
    mode = OnMissedToc(mode)
    if mode is OnMissedToc.REDO:
        return True
    if mode is OnMissedToc.SKIP:
        return False
    # mode is ASK: вопрос имеет смысл только с живым терминалом; фоновый прогон через
    # setsid … < /dev/null получил бы EOF и упал, поэтому без tty — как skip, с пометкой в логе.
    if not sys.stdin.isatty():
        logger.warning("%s: терминала нет, перераспознание пропущено (см. %s)", issue_key, MISSED_LIST)
        return False
    # По умолчанию «нет»: Enter по инерции не должен запускать платный повтор выпуска.
    return click.confirm(f"Перераспознать выпуск {issue_key} с этими полосами как оглавлением?", default=False)


def _append_demoted(out_dir: Path, issue_key: str, demoted: list[Path]) -> None:
    """Дописать понижённые полосы в ``demoted_toc.txt`` (файл копится между прогонами, как ``missed_toc.txt``).

    Args:
        out_dir: Корень выхода — файл лежит в нём.
        issue_key: Ключ выпуска — в комментарий строки.
        demoted: Полосы, которые модель не признала оглавлением.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / DEMOTED_LIST).open("a", encoding="utf-8") as handle:
        for rel in demoted:
            handle.write(f"{rel.as_posix()}  # в базе оглавление, модель: none, выпуск {issue_key}\n")


def _append_missed(out_dir: Path, issue_key: str, missed: list[tuple[Path, TocKind]]) -> None:
    """Дописать полосы с оглавлением вне базы в ``missed_toc.txt`` (файл копится между прогонами).

    Args:
        out_dir: Корень выхода — файл лежит в нём.
        issue_key: Ключ выпуска — в комментарий строки.
        missed: Полосы и вид оглавления по мнению модели.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Режим «a», не «w»: прогон с --skip-done видит только часть выпусков, а список должен
    # накапливать все находки; при повторной находке той же полосы строка задвоится — это терпимо.
    with (out_dir / MISSED_LIST).open("a", encoding="utf-8") as handle:
        for rel, kind in missed:
            handle.write(f"{rel.as_posix()}  # {kind}, выпуск {issue_key}\n")


def article_start_pages(toc: toc_module.IssueToc | None) -> set[str]:
    """Номера страниц, с которых по «Содержанию» начинаются статьи, — как напечатаны, без пробелов.

    Args:
        toc: Слитое «Содержание» выпуска; ``None`` — оглавления нет.

    Returns:
        Множество строк-номеров (``{"3", "10", …}``); пустое, если оглавления нет или номера не указаны.
    """
    if toc is None:
        return set()
    return {article.page.strip() for article in toc.articles if article.page and article.page.strip()}


def _outside_fences(body: str) -> str:
    """Тело без fenced-блоков иллюстраций: их строки (надписи на схеме) — не заголовки и не теги.

    Args:
        body: Тело полосы в markdown.

    Returns:
        Тот же текст с вырезанными блоками между парами строк ```.
    """
    return _FENCED.sub("", body)


def redo_reason(
    result: PageResult | None, meta: dict | None, start_pages: set[str], demoted: bool
) -> RedoReason | None:
    """Надо ли запрашивать полосу заново на круге повтора, и почему.

    Списки выпуска (рубрики и статьи «Содержания») влияют только на структуру ответа: ``#`` у
    названий из списка, ``<rubric>`` / ``<marker>``, привязку ``<author>``, правки ``structure.apply``.
    Полоса без единого такого признака при повторе даст тот же текст — её можно оставить.
    Колонтитул не в счёт: заголовок, рубрика, маркер или поле ``rubric`` с текстом
    ``running_header`` / ``running_footer`` — это переписанный колонтитул, списки его не меняют.

    Args:
        result: Результат первого круга (этап ``page``) с диска; ``None`` — нет или битый.
        meta: Meta первого круга; ``None`` — нет или битая.
        start_pages: Номера страниц начала статей по новому «Содержанию» (:func:`article_start_pages`).
        demoted: Полоса понижена (шла этапом toc, модель сказала «не оглавление»).

    Returns:
        Причина повтора или ``None`` — полосу можно взять с диска как есть.
    """
    if demoted:
        return RedoReason.DEMOTED
    if result is None or meta is None:
        return RedoReason.NO_FIRST_PASS
    body = result.content_markdown
    # Нормализованные колонтитулы — с ними сверяются заголовки, рубрики и маркеры на полосе.
    headers = {normalize_title(text) for text in (result.running_header, result.running_footer) if text}
    headers.discard("")
    outside = _outside_fences(body)
    if any(normalize_title(text) not in headers for text in _HEADING.findall(outside)):
        return RedoReason.HEADING
    if any(f"<{tag}>" in body for tag in _AUTHOR_TAGS):
        return RedoReason.STRUCTURE_TAG
    if any(normalize_title(text) not in headers for _, text in _RUBRIC_OR_MARKER.findall(body)):
        return RedoReason.STRUCTURE_TAG
    if result.title or result.authors or (result.rubric and normalize_title(result.rubric) not in headers):
        return RedoReason.TITLE_OR_AUTHORS
    # Пост-обработка что-то правила: следы лежат в meta.structure списками (пустые — не правила).
    # Колонтитул, переведённый из <rubric> в <marker> (markers_from_rubrics), — не правка по списку.
    structure = meta.get("structure") or {}
    for key, value in structure.items():
        if not isinstance(value, list):
            continue
        if key == "markers_from_rubrics":
            value = [text for text in value if normalize_title(str(text)) not in headers]
        if value:
            return RedoReason.STRUCTURE_EDITS
    # Страховка: по новому оглавлению здесь начинается статья, а заголовок первым проходом не выделен.
    if result.page_number and result.page_number.strip() in start_pages:
        return RedoReason.ARTICLE_START
    return None


def keep_unchanged_pages(
    out_dir: Path, regular: list[Path], demoted: set[Path], digest: str, start_pages: set[str]
) -> tuple[list[Path], dict[Path, RedoReason]]:
    """Круг повтора по ``--redo-scope structured``: разделить обычные полосы на «оставить» и «заново».

    У оставленных полос meta переписывается с новым ``toc_hash`` и пометкой ``redo_kept`` —
    дальше штатный ``is_done`` считает их готовыми, и ``_recognize_many`` берёт их с диска.
    ``articles_in_prompt`` не трогается: список в их промпте был старый, и это должно быть видно.

    Args:
        out_dir: Корень выхода — там лежат .json/.meta.json первого круга.
        regular: Обычные полосы выпуска (включая понижённые).
        demoted: Понижённые полосы — идут заново всегда.
        digest: Отпечаток новых списков (``toc.toc_hash``).
        start_pages: Номера страниц начала статей по новому «Содержанию».

    Returns:
        ``(оставленные полосы, {полоса: причина повтора} для остальных)``.
    """
    kept: list[Path] = []
    redo: dict[Path, RedoReason] = {}
    for rel in regular:
        job = PageJob(rel, Stage.PAGE)
        meta = read_meta(out_dir, rel)
        # Битая или сбойная полоса первого круга — как «нет результата»: пойдёт заново.
        result = load_result(out_dir, job) if meta and not (meta.get("error") or meta.get("parse_error")) else None
        reason = redo_reason(result, meta, start_pages, rel in demoted)
        if reason is not None:
            redo[rel] = reason
            continue
        meta = dict(meta)
        meta.update(toc_hash=digest, redo_kept=True)
        write_meta(output_paths(out_dir, rel).meta, meta)
        kept.append(rel)
    return kept, redo


def run_issue(
    client: OpenRouterClient,
    spec: ModelSpec,
    params: PipelineParams,
    issue_key: str,
    pages: list[Path],
    extra_toc: dict[Path, TocKind] | None = None,
) -> PipelineStats:
    """Один выпуск целиком: этап toc, слияние, этап page, fallback.

    Последовательность: (1) полосы, которые база (или ``extra_toc``) считает оглавлением/указателем,
    распознаются этапом ``toc`` и сливаются в ``toc.json`` / ``toc.md`` выпуска; те из них, что модель
    не признала оглавлением, понижаются — их вклад в оглавление сохраняется, а сами они идут ещё и
    этапом ``page`` (``demoted_toc.txt``); (2) из «Содержания»
    берутся рубрики и статьи для промпта, остальные полосы идут этапом ``page``; (3) если модель на
    обычной полосе увидела оглавление, которого в базе нет, — предупреждение и, по
    ``params.on_missed_toc``, один повторный вызов самой себя с этими полосами как оглавлением; по
    ``params.redo_scope`` на этом круге заново идут либо все обычные полосы, либо только те, где
    списки могут что-то изменить (:func:`redo_reason`). Файлы — под ``params.out_dir``.

    Args:
        client: Клиент OpenRouter; общий на весь прогон, потокобезопасный (ходит из пула потоков).
        spec: Модель из реестра: id для OpenRouter, режим JSON, рассуждения, порядок провайдеров.
        params: Параметры прогона: пути входа/выхода, флаги полос из базы, ``jobs``, ``skip_done``,
            ``on_missed_toc`` и настройки запроса (``params.options``).
        issue_key: Ключ выпуска — путь папки относительно ``in_dir`` (``«1966/03»``); из него берётся
            год для промпта и под ним пишутся ``toc.json`` / ``toc.md``. Полосы в корне входа
            группируются под ключом ``«.»`` — года у них нет.
        pages: Все полосы выпуска (относительные пути), отсортированные по имени. Порядок важен:
            по нему сливаются многостраничные оглавления (продолжение — к предыдущей полосе).
        extra_toc: Полосы, которые надо считать оглавлением помимо базы: ``{путь: «contents»
            | «index»}``. Заполняется только при повторном вызове из fallback (то, что модель нашла
            на обычных полосах); тег базы при этом главнее. ``None`` — обычный, первый вызов.
            Повторный вызов идёт с копией ``params``, где ``on_missed_toc=SKIP``: один круг на
            выпуск, иначе модель, «увидев» оглавление на очередной полосе, гоняла бы выпуск по кругу
            за деньги.

    Returns:
        Счётчики этого выпуска (обоих этапов и круга повтора, если он был): запросы, сбои,
        стоимость, полосы без записи в базе, ``missed`` / ``redone_issues`` / ``demoted_toc`` /
        ``redo_kept`` — вызывающий складывает их со своими через ``+``.
    """
    stats = PipelineStats()
    # Ключ выпуска — «{год}/{выпуск}» относительно in-dir; год уходит в пользовательский промпт
    # («выпуск такого-то года»). Плоская папка без подпапок группируется под ключом «.», года нет.
    year = issue_key.split("/")[0] if issue_key != "." else ""
    # Полосы этапа toc: относительный путь -> вид («Содержание» или «Годовой указатель»).
    toc_pages: dict[Path, TocKind] = {}
    for rel_path in pages:
        # Теги полосы из базы разметки (или из --toc-lists); без источника — все флаги пустые.
        flags = flags_for(rel_path, params.flags)
        # Источник тегов есть, а полосы в нём нет (не в базе, не в списках): считаем и предупредим
        # в итоге — такая полоса пойдёт как обычная, и оглавление на ней найдёт только fallback.
        if not flags.known and params.flags is not None:
            stats.unknown_pages += 1
        toc_kind = flags.toc_kind
        # Круг повтора: полосы, где модель увидела оглавление, а база молчит, — приходят через
        # extra_toc и тоже становятся полосами этапа toc; тег из базы при этом главнее.
        if toc_kind is None and extra_toc and rel_path in extra_toc:
            toc_kind = extra_toc[rel_path]
        if toc_kind is not None:
            toc_pages[rel_path] = toc_kind
    # Брать ли готовые результаты с диска: при --skip-done — по просьбе, на круге повтора — всегда,
    # чтобы не платить второй раз за полосы оглавления из базы (их вид не изменился, is_done верен);
    # обычные полосы повтор всё равно пересчитает — у них сменится toc_hash.
    reuse = params.skip_done or extra_toc is not None
    logger.info("Выпуск %s: полос %d, из них оглавление/указатель %d", issue_key, len(pages), len(toc_pages))

    # Этап 1: полосы оглавления/указателя — без списка статей (его ещё нет), с видом полосы в промпте.
    toc_jobs = [PageJob(rel, Stage.TOC, kind, year) for rel, kind in toc_pages.items()]
    toc_results, toc_stats = _recognize_many(client, spec, params, toc_jobs, reuse)
    stats = stats + toc_stats
    # Слияние ответов по видам в порядке полос выпуска: продолжения списка дописываются к
    # рубрикам предыдущей полосы того же вида. Сбои этапа toc в результатах отсутствуют.
    tocs = build_issue_toc(pages, toc_pages, toc_results)
    # Полоса пришла как «предварительно оглавление», а модель сказала «не оглавление»: её статьи (если
    # она их всё же извлекла) уже вошли в слияние выше, а сама полоса дальше идёт этапом page как
    # обычная. Ответ этапа toc сохраняется в .toc.json — повтор с --skip-done прочтёт его без запроса.
    demoted = [rel for rel in pages if (result := toc_results.get(rel)) is not None and result.toc_kind is TocKind.NONE]
    for rel in demoted:
        save_demoted_toc(params.out_dir, rel, toc_results[rel])
    # На круге повтора понижённые полосы те же (их ответ этапа toc взят с диска) — отчитаны первым кругом.
    if demoted and extra_toc is None:
        stats.demoted_toc.append(f"{issue_key}: " + ", ".join(rel.name for rel in demoted))
        _append_demoted(params.out_dir, issue_key, demoted)
        logger.warning(
            "%s: модель не считает оглавлением полосы %s — они пойдут как обычные; проверьте теги в CVAT (см. %s)",
            issue_key,
            ", ".join(rel.name for rel in demoted),
            DEMOTED_LIST,
        )
    if not toc_pages:
        # Полос оглавления в этом прогоне нет (--pages со списком обычных полос, повтор сбойных): списки
        # берутся из toc.json прошлого прогона, иначе полосы пошли бы без структуры и без понижения `#`.
        saved = params.out_dir / issue_key / "toc.json"
        if saved.is_file():
            tocs = toc_module.from_dict(json.loads(saved.read_text(encoding="utf-8")))
            logger.info("%s: оглавление взято из %s (полос оглавления в прогоне нет)", issue_key, saved)
    if toc_pages:
        # toc.json (машинный, для повторов и сборки) и toc.md (глазами) в папку выпуска под out-dir.
        _write_issue_toc(params.out_dir, issue_key, tocs)
        for toc_kind, issue_toc in tocs.items():
            logger.info(
                "%s: %s — рубрик %d, статей %d, полос %d (продолжений %d)",
                issue_key,
                toc_kind,
                len(issue_toc.rubrics),
                len(issue_toc.articles),
                len(issue_toc.pages),
                issue_toc.continuations,
            )
    # Списки в промпт этапа page берутся только из «Содержания»: годовой указатель описывает весь
    # год, а не этот выпуск. Нет «Содержания» — списки пустые, полосы идут без структуры.
    rubrics, articles = toc_module.prompt_lists(tocs.get(TocKind.CONTENTS))
    # Отпечаток списков: пишется в meta каждой полосы, и is_done считает полосу готовой только при
    # совпадении — так после изменившегося оглавления обычные полосы пересчитываются сами.
    digest = toc_module.toc_hash(rubrics, articles)

    # Этап 2: все остальные полосы выпуска плюс понижённые, с рубриками и статьями «Содержания» в промпте.
    regular = [rel for rel in pages if rel not in toc_pages or rel in demoted]
    # Круг повтора по --redo-scope structured: полосы, на которых новые списки ничего не изменят
    # (нет заголовков, рубрик/маркеров/авторов, правок пост-обработки, и это не начало статьи по
    # новому оглавлению), получают новый toc_hash в meta и ниже берутся с диска; остальные — заново.
    if extra_toc is not None and params.redo_scope is RedoScope.STRUCTURED:
        start_pages = article_start_pages(tocs.get(TocKind.CONTENTS))
        kept, redo = keep_unchanged_pages(params.out_dir, regular, set(demoted), digest, start_pages)
        stats.redo_kept += len(kept)
        logger.info(
            "%s: круг повтора — заново %d полос, оставлено без повтора %d:\n%s",
            issue_key,
            len(redo),
            len(kept),
            "\n".join(f"  {rel.name}: {reason}" for rel, reason in redo.items()),
        )
    page_jobs = [
        PageJob(rel, Stage.PAGE, TocKind.NONE, year, tuple(rubrics), tuple(articles), digest, None, rel in demoted)
        for rel in regular
    ]
    page_results, page_stats = _recognize_many(client, spec, params, page_jobs, reuse)
    stats = stats + page_stats

    # Fallback: обычные полосы, на которых модель увидела оглавление или указатель. Полосы, где
    # человек поставил вето «Не оглавление», не считаются — модель на них ошибается регулярно
    # (списки литературы, программы, таблицы). Сбои (нет результата) и понижённые полосы (они
    # уже прошли этап toc) тоже не считаются.
    missed = [
        (rel, result.toc_kind)
        for rel in regular
        if (result := page_results.get(rel)) is not None
        and result.toc_kind is not TocKind.NONE
        and rel not in demoted
        and not flags_for(rel, params.flags).force_is_not_toc
    ]
    if not missed:
        return stats
    # В итоговую строку прогона — всегда, независимо от того, будет ли повтор.
    stats.missed.append(f"{issue_key}: " + ", ".join(rel.name for rel, _ in missed))
    # Решение — по --on-missed-toc: redo/skip без вопросов, ask — вопрос в терминал (без терминала
    # = skip). Повтор — только один круг: вложенный вызов получает копию params с on_missed_toc=SKIP,
    # иначе модель, увидев оглавление на очередной полосе, гоняла бы выпуск по кругу.
    if decide_redo(params.on_missed_toc, issue_key, missed):
        stats.redone_issues.append(issue_key)
        # Тот же выпуск заново: найденные полосы — как оглавление, список пересобирается, обычные
        # полосы с новым toc_hash не считаются готовыми и распознаются ещё раз (все или только
        # чувствительные к спискам — по redo_scope).
        redo_stats = run_issue(
            client, spec, replace(params, on_missed_toc=OnMissedToc.SKIP), issue_key, pages, extra_toc=dict(missed)
        )
        stats = stats + redo_stats
    else:
        # Без повтора: полосы в missed_toc.txt — человеку на разметку в CVAT; в базу не пишем.
        _append_missed(params.out_dir, issue_key, missed)
    return stats


def collect_meta(out_dir: Path) -> list[dict]:
    """Все .meta.json под out_dir: сводка строится по ним, а не по одному прогону.

    Args:
        out_dir: Корень выхода; обходится рекурсивно.

    Returns:
        Список meta-словарей (по одному на полосу) в порядке путей; битые файлы пропущены.
    """
    rows: list[dict] = []
    # Битый или недописанный meta (прогон прервали на записи) — предупреждение, не остановка:
    # сводка по остальным полосам всё равно нужна.
    for path in sorted(out_dir.rglob("*.meta.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            logger.warning("битый %s", path)
    return rows


def write_summary(out_dir: Path, rows: list[dict]) -> Path:
    """``summary.csv`` в корне out_dir: по строке на полосу, колонки — :data:`SUMMARY_FIELDS`.

    Args:
        out_dir: Корень выхода — файл лежит в нём.
        rows: Meta всех полос из ``collect_meta``.

    Returns:
        Путь к записанному ``summary.csv``.
    """
    path = out_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        # extrasaction="ignore": в meta есть вложенные словари (tiling, structure, second_pass),
        # в плоскую таблицу они не идут; отсутствующая колонка пишется пустой.
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        # Порядок — по относительному пути полосы, то есть год → выпуск → имя файла.
        for row in sorted(rows, key=lambda item: item.get("page", "")):
            writer.writerow(row)
    return path


def run_pipeline(client: OpenRouterClient, spec: ModelSpec, params: PipelineParams) -> PipelineStats:
    """Весь вход по выпускам; возвращает сводку. summary.csv пересобирается по всем meta под out-dir.

    Args:
        client: Клиент OpenRouter, общий на прогон.
        spec: Модель из реестра.
        params: Параметры прогона из CLI: пути, флаги полос, отбор входа, параллелизм, fallback.

    Returns:
        Счётчики прогона (``PipelineStats``): выпуски, полосы, запросы, сбои, стоимость, список
        оглавлений вне базы — то, что CLI печатает в итоговых строках.
    """
    # Отбор входа: обход in-dir или файл --pages, затем --only-year/--only-issue и --limit.
    rels = list_pages(params.in_dir, params.pages_file, params.only_year, params.only_issue, params.limit)
    # {«год/выпуск»: [полосы по имени]} — единица работы дальше выпуск, не полоса.
    groups = group_by_issue(rels)
    # Свои счётчики прогона: сколько полос на входе, сколько из них база считает оглавлением;
    # счётчики выпусков прибавляются ниже из возвращаемых значений run_issue.
    stats = PipelineStats(
        pages=len(rels),
        issues=len(groups),
        toc_pages=sum(1 for rel in rels if flags_for(rel, params.flags).toc_kind is not None),
    )
    started = time.monotonic()
    logger.info(
        "%s (%s): выпусков %d, полос %d, потоков %d, тайл %d px → %d px",
        spec.name,
        spec.openrouter_id,
        len(groups),
        len(rels),
        params.jobs,
        params.options.max_src_tile,
        params.options.max_model_tile,
    )
    # Проверка стыков полос при сборке — тем же клиентом и моделью, один запрос на выпуск.
    checker = None
    heading_checker = None
    if params.assemble and params.check_boundaries:
        checker = BoundaryChecker(client, spec, params.in_dir, params.options.cache_dir, params.options.quality)
    if params.assemble and params.check_headings:
        heading_checker = HeadingChecker(client, spec, params.options.cache_dir)
    # Выпуски идут строго по одному: этап page зависит от этапа toc того же выпуска, а полосы
    # внутри этапа распараллеливает run_issue своим пулом потоков.
    for issue_key, pages in groups.items():
        # Весь цикл выпуска: полосы оглавления → toc.json → остальные полосы со списком → fallback.
        stats = stats + run_issue(client, spec, params, issue_key, pages)
        # Выпуск целиком в один markdown — по готовым .json с диска, после круга повтора, если он был.
        if params.assemble:
            assembly = assemble_issue(
                params.out_dir,
                issue_key,
                pages,
                join_hyphens=params.options.join_hyphens,
                checker=checker,
                heading_checker=heading_checker,
            )
            log_assembly(assembly, write_issue(params.out_dir, assembly))
            stats = stats + PipelineStats(
                cost_usd=assembly.checked.cost_usd + assembly.heading_checked.cost_usd,
                boundary_requests=assembly.checked.requests,
                boundary_rewritten=assembly.count(JoinKind.MODEL),
                boundary_cost_usd=assembly.checked.cost_usd,
                heading_requests=assembly.heading_checked.requests,
                heading_restored=sum(1 for a in assembly.reconcile.articles if a.get("source") == "model"),
                heading_cost_usd=assembly.heading_checked.cost_usd,
            )
    # Сводка строится по ВСЕМ .meta.json под out-dir, а не по этому прогону: с --skip-done прогон
    # видел только недоделанные полосы, а summary.csv должен описывать папку целиком.
    rows = collect_meta(params.out_dir)
    summary = write_summary(params.out_dir, rows)
    # Доля токенов входа, взятых провайдером из кэша префикса: по ней видно, что кэш работает
    # (одинаковое начало системного промпта внутри выпуска) и сколько он экономит.
    prompt_total = sum(int(r.get("prompt_tokens") or 0) for r in rows)
    cached_total = sum(int(r.get("cached_tokens") or 0) for r in rows)
    if prompt_total:  # пустая папка или одни сбои — делить не на что
        logger.info(
            "кэш префикса: %d из %d токенов входа (%.0f %%)",
            cached_total,
            prompt_total,
            100 * cached_total / prompt_total,
        )
    # Итог именно этого прогона (в отличие от summary.csv): что запросили, что пропустили, сколько
    # это стоило. Полосы с запросом без результата — «сбои», их можно догнать повтором с --skip-done.
    logger.info(
        "готово за %.0f с: выпусков %d, полос %d, запросов %d (готовых пропущено %d, из кэша запросов %d), сбоев %d, "
        "стоимость $%.4f, сводка %s",
        time.monotonic() - started,
        stats.issues,
        stats.pages,
        stats.requests,
        stats.reused,
        stats.cache_hits,
        stats.failed,
        stats.cost_usd,
        summary,
    )
    # Полосы, которых нет в базе при заданной базе: обычно расхождение имён между входом и разметкой
    # (не тот пак, полоса переименована) — стоит проверить, иначе оглавление на них найдёт только fallback.
    if stats.unknown_pages:
        logger.warning("полос без записи в базе: %d — считались обычными", stats.unknown_pages)
    # Доля повреждённых полос по мнению модели и сколько раз страховка вернула первый проход.
    if params.options.second_pass:
        logger.info(
            "второй проход: %d полос из %d, оставлен первый у %d",
            stats.second_passes,
            stats.pages,
            stats.second_pass_kept_first,
        )
    # Напоминание в самом конце лога, чтобы не потерялось среди строк по полосам: эти теги надо
    # проставить в CVAT независимо от того, был ли повтор выпуска.
    # Полосы с замечаниями пост-обработки (сверка оглавления и т. п.) — искать по колонке messages в summary.csv.
    if stats.page_messages:
        logger.warning("полос с замечаниями пост-обработки: %d (колонка messages в summary.csv)", stats.page_messages)
    if stats.page_masked:
        logger.info("полос с замаскированным мусором модели: %d (колонка masked в summary.csv)", stats.page_masked)
    # Сколько `#` получили id статьи не от модели: свыше 30 % — промпт с id модели не по силам.
    headings_total = stats.heading_ids_model + stats.heading_ids_other
    if headings_total:
        logger.info(
            "id статей у `#`: от модели %d, по названию или исправлено %d из %d (%.0f %% не от модели; колонка heading_ids)",
            stats.heading_ids_model,
            stats.heading_ids_other,
            headings_total,
            100.0 * stats.heading_ids_other / headings_total,
        )
    if stats.demoted_toc:
        logger.warning(
            "в базе оглавление, модель — нет (%d выпусков): %s", len(stats.demoted_toc), "; ".join(stats.demoted_toc)
        )
    if stats.redo_kept:
        logger.info(
            "круг повтора: без повтора оставлено %d полос (--redo-scope %s)", stats.redo_kept, params.redo_scope
        )
    if stats.boundary_requests:
        logger.info(
            "стыки полос: запросов %d, стыков переписано по вердикту модели %d, $%.4f",
            stats.boundary_requests,
            stats.boundary_rewritten,
            stats.boundary_cost_usd,
        )
    if stats.heading_requests:
        logger.info(
            "заголовки без `#`: запросов %d, восстановлено по ответу модели %d, $%.4f",
            stats.heading_requests,
            stats.heading_restored,
            stats.heading_cost_usd,
        )
    if stats.missed:
        logger.warning("оглавления вне базы (%d выпусков): %s", len(stats.missed), "; ".join(stats.missed))
    return stats
