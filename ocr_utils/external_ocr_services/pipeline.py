"""Обход пака по выпускам: полосы оглавления -> список статей -> остальные полосы -> fallback.

На выпуск: (1) полосы, помеченные в базе как «Содержание» или указатель, распознаются этапом
``toc`` и сливаются в оглавление выпуска (``toc.json`` / ``toc.md``); (2) остальные полосы идут
этапом ``page`` с рубриками и статьями «Содержания» в промпте; (3) если модель на обычной полосе
увидела оглавление, которого в базе нет (и нет вето ``force_is_not_toc``), — предупреждение и,
по ``--on-missed-toc``, повтор выпуска: найденные полосы распознаются как оглавление, список
пересобирается, обычные полосы идут заново (у них меняется ``toc_hash``, готовые с прежним
списком не считаются сделанными). Один круг повтора на выпуск.

Запросы к сети — в пуле потоков ``--jobs`` внутри этапа; выпуски идут последовательно, потому
что этап 2 зависит от этапа 1. В базу ничего не пишется: пропущенные оглавления складываются в
``missed_toc.txt``, а теги ставит человек в CVAT.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import click

from ocr_utils.external_ocr_services import toc as toc_module
from ocr_utils.external_ocr_services.client import OpenRouterClient
from ocr_utils.external_ocr_services.models import ModelSpec
from ocr_utils.external_ocr_services.ocr import (
    PageJob,
    RunOptions,
    is_done,
    load_result,
    recognise_page,
    recognise_with_second_pass,
)
from ocr_utils.external_ocr_services.pages import PageFlags, flags_for, group_by_issue, list_pages
from ocr_utils.external_ocr_services.schema import PageResult

logger = logging.getLogger(__name__)

# Файл в корне out-dir: полосы, где модель увидела оглавление, а база молчит, — дописывается за
# каждый выпуск без повтора; человеку на разметку в CVAT.
MISSED_LIST = "missed_toc.txt"
# Оглавление выпуска в папке «{год}/{выпуск}» под out-dir: машинный JSON и тот же список глазами.
TOC_JSON = "toc.json"
TOC_MD = "toc.md"
# Значения --on-missed-toc; порядок только для click.Choice.
ON_MISSED_CHOICES = ("ask", "redo", "skip")

# Колонки сводки прогона; пересобирается по всем .meta.json под out-dir. Ключи совпадают с
# ключами meta полосы (см. ocr.recognise_page), лишние ключи meta в сводку не попадают.
SUMMARY_FIELDS = (
    "page",
    "stage",
    "toc_kind_expected",
    "toc_hash",
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
    "page_number",
    "toc_kind",
    "title",
    "title_in_list",
    "content_chars",
    "toc_articles",
    "tags",
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
    on_missed_toc: str = "redo"  # что делать с оглавлением вне базы: redo / ask / skip
    pages_file: Path | None = None  # --pages: явный список относительных путей вместо обхода
    only_year: str | None = None  # --only-year / --only-issue: отбор по первым папкам пути
    only_issue: str | None = None
    limit: int | None = None  # --limit: первые N полос после отбора, для проб


@dataclass
class PipelineStats:
    """Счётчики прогона для итоговой строки лога и тестов; копятся по всем выпускам."""

    issues: int = 0  # выпусков обработано (повтор выпуска не считается вторым)
    pages: int = 0  # полос на входе после отбора
    requests: int = 0  # полос, ушедших в сеть (второй проход — та же полоса, не второй запрос)
    reused: int = 0  # полос, взятых готовыми с диска (--skip-done или круг повтора)
    failed: int = 0  # полос без результата: сбой сети, не разобранный JSON, исчерпаны попытки
    cost_usd: float = 0.0  # сумма cost_usd по meta запрошенных полос (оба прохода, без эха формата)
    toc_pages: int = 0  # полос, которые база (или списки) считает оглавлением/указателем
    unknown_pages: int = 0  # полос без записи в базе
    second_passes: int = 0  # полос, ушедших во второй проход
    second_pass_kept_first: int = 0  # из них оставлен первый проход (страховка)
    missed: list[str] = field(default_factory=list)  # «выпуск: полосы», где оглавление не в базе
    redone_issues: list[str] = field(default_factory=list)  # выпуски, прошедшие круг повтора


def _recognise_many(
    client: OpenRouterClient,
    spec: ModelSpec,
    params: PipelineParams,
    jobs: list[PageJob],
    reuse: bool,
    stats: PipelineStats,
) -> dict[Path, PageResult | None]:
    """Полосы пачкой в пуле потоков; готовые (по ``is_done``) не запрашиваются, если ``reuse``.

    Возвращает результат каждой полосы из ``jobs``: ``None`` у сбойных, чтобы вызывающий отличал
    «сбой» от «полосы не было». Файлы выхода пишет сам ``recognise_*`` — здесь только счётчики и лог.
    """
    results: dict[Path, PageResult | None] = {}
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
        return results

    def work(job: PageJob) -> tuple[PageJob, dict, PageResult | None]:
        # Второй проход (по умолчанию включён) — обёртка над recognise_page: та же полоса ещё раз
        # с подсказками первого ответа, если модель сочла её повреждённой; выбор финала — внутри.
        recognise = recognise_with_second_pass if params.options.second_pass else recognise_page
        meta, result = recognise(client, spec, params.in_dir / job.rel, job, params.out_dir, params.options)
        return job, meta, result

    # Пул потоков, не процессов: работа — ожидание сети, GIL не мешает; тайлы режутся в потоке
    # перед запросом. pool.map отдаёт результаты в порядке очереди, счётчики правятся в одном потоке.
    with ThreadPoolExecutor(max_workers=max(1, params.jobs)) as pool:
        for job, meta, result in pool.map(work, todo):
            stats.requests += 1
            # cost_usd в meta — сумма обоих проходов по удачным ответам; выброшенные ответы (эхо
            # response_format) лежат отдельно в cost_usd_wasted и в итог не входят.
            stats.cost_usd += float(meta.get("cost_usd") or 0.0)
            # Второй проход был — есть причина; какой проход ушёл в финал, пишет second_pass_chosen.
            if meta.get("second_pass_reason"):
                stats.second_passes += 1
                stats.second_pass_kept_first += meta.get("second_pass_chosen") == "pass1"
            # В строке лога — сетевая ошибка, иначе ошибка разбора, иначе «ok».
            status = meta.get("error") or meta.get("parse_error") or "ok"
            if result is None:
                stats.failed += 1
            logger.info(
                "%s [%s] %s: %s tok → %s tok, $%s, %.0f с",
                job.rel,
                job.stage,
                status,
                meta.get("prompt_tokens", "-"),
                meta.get("completion_tokens", "-"),
                meta.get("cost_usd", "-"),
                meta.get("latency_s") or 0.0,
            )
            results[job.rel] = result
    return results


def build_issue_toc(
    pages: list[Path], toc_pages: dict[Path, str], results: dict[Path, PageResult | None]
) -> dict[str, toc_module.IssueToc]:
    """Слить результаты этапа toc в оглавления выпуска по видам, в порядке полос.

    «Содержание» и «Годовой указатель» сливаются порознь: у каждого вида свой список полос, и
    указатель в промпт обычных полос не идёт. Вид без единой удачной полосы в ответе отсутствует.
    """
    tocs: dict[str, toc_module.IssueToc] = {}
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


def _write_issue_toc(out_dir: Path, issue_key: str, tocs: dict[str, toc_module.IssueToc]) -> None:
    """``toc.json`` и ``toc.md`` в папку выпуска; перезаписываются на каждом проходе выпуска."""
    issue_dir = out_dir / issue_key
    issue_dir.mkdir(parents=True, exist_ok=True)
    # JSON — полная структура (рубрики → статьи с авторами, список полос) для повторов и сборки
    # выпуска; indent=1 — чтобы diff между прогонами читался построчно.
    (issue_dir / TOC_JSON).write_text(
        json.dumps(toc_module.to_dict(tocs), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    # Markdown — то же самое глазами: проверить, что модель прочла оглавление правильно.
    (issue_dir / TOC_MD).write_text(toc_module.to_markdown(tocs), encoding="utf-8")


def decide_redo(mode: str, issue_key: str, missed: list[tuple[Path, str]]) -> bool:
    """Перераспознавать ли выпуск после находки оглавления вне базы.

    ``ask`` спрашивает в терминале; без терминала (фоновый прогон с ``< /dev/null``) ведёт себя как
    ``skip`` — молча пересчитывать чужой счёт нельзя. Предупреждение в лог пишется при любом
    режиме: теги в базе всё равно должен поставить человек, повтор их не заменяет.
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
    if mode == "redo":
        return True
    if mode == "skip":
        return False
    # mode == "ask": вопрос имеет смысл только с живым терминалом; фоновый прогон через
    # setsid … < /dev/null получил бы EOF и упал, поэтому без tty — как skip, с пометкой в логе.
    if not sys.stdin.isatty():
        logger.warning("%s: терминала нет, перераспознание пропущено (см. %s)", issue_key, MISSED_LIST)
        return False
    # По умолчанию «нет»: Enter по инерции не должен запускать платный повтор выпуска.
    return click.confirm(f"Перераспознать выпуск {issue_key} с этими полосами как оглавлением?", default=False)


def _append_missed(out_dir: Path, issue_key: str, missed: list[tuple[Path, str]]) -> None:
    """Дописать полосы с оглавлением вне базы в ``missed_toc.txt`` (файл копится между прогонами)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Режим «a», не «w»: прогон с --skip-done видит только часть выпусков, а список должен
    # накапливать все находки; при повторной находке той же полосы строка задвоится — это терпимо.
    with (out_dir / MISSED_LIST).open("a", encoding="utf-8") as handle:
        for rel, kind in missed:
            handle.write(f"{rel.as_posix()}  # {kind}, выпуск {issue_key}\n")


def run_issue(
    client: OpenRouterClient,
    spec: ModelSpec,
    params: PipelineParams,
    issue_key: str,
    pages: list[Path],
    stats: PipelineStats,
    extra_toc: dict[Path, str] | None = None,
) -> None:
    """Один выпуск целиком: этап toc, слияние, этап page, fallback.

    Последовательность: (1) полосы, которые база (или ``extra_toc``) считает оглавлением/указателем,
    распознаются этапом ``toc`` и сливаются в ``toc.json`` / ``toc.md`` выпуска; (2) из «Содержания»
    берутся рубрики и статьи для промпта, остальные полосы идут этапом ``page``; (3) если модель на
    обычной полосе увидела оглавление, которого в базе нет, — предупреждение и, по
    ``params.on_missed_toc``, один повторный вызов самой себя с этими полосами как оглавлением.
    Ничего не возвращает: результат — файлы под ``params.out_dir`` и счётчики в ``stats``.

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
        stats: Счётчики прогона; сюда прибавляются запросы, сбои, стоимость, полосы без записи в
            базе, а также ``missed`` / ``redone_issues`` для итоговой строки лога.
        extra_toc: Полосы, которые надо считать оглавлением помимо базы: ``{путь: «contents»
            | «index»}``. Заполняется только при повторном вызове из fallback (то, что модель нашла
            на обычных полосах); тег базы при этом главнее. ``None`` — обычный, первый вызов.
            Повторный вызов идёт с копией ``params``, где ``on_missed_toc="skip"``: один круг на
            выпуск, иначе модель, «увидев» оглавление на очередной полосе, гоняла бы выпуск по кругу
            за деньги.
    """
    # Ключ выпуска — «{год}/{выпуск}» относительно in-dir; год уходит в пользовательский промпт
    # («выпуск такого-то года»). Плоская папка без подпапок группируется под ключом «.», года нет.
    year = issue_key.split("/")[0] if issue_key != "." else ""
    # Полосы этапа toc: относительный путь -> вид («Содержание» или «Годовой указатель»).
    toc_pages: dict[Path, str] = {}
    for rel in pages:
        # Теги полосы из базы разметки (или из --toc-lists); без источника — все флаги пустые.
        flags = flags_for(rel, params.flags)
        # Источник тегов есть, а полосы в нём нет (не в базе, не в списках): считаем и предупредим
        # в итоге — такая полоса пойдёт как обычная, и оглавление на ней найдёт только fallback.
        if not flags.known and params.flags is not None:
            stats.unknown_pages += 1
        kind = flags.toc_kind
        # Круг повтора: полосы, где модель увидела оглавление, а база молчит, — приходят через
        # extra_toc и тоже становятся полосами этапа toc; тег из базы при этом главнее.
        if kind is None and extra_toc and rel in extra_toc:
            kind = extra_toc[rel]
        if kind is not None:
            toc_pages[rel] = kind
    # Брать ли готовые результаты с диска: при --skip-done — по просьбе, на круге повтора — всегда,
    # чтобы не платить второй раз за полосы оглавления из базы (их вид не изменился, is_done верен);
    # обычные полосы повтор всё равно пересчитает — у них сменится toc_hash.
    reuse = params.skip_done or extra_toc is not None
    logger.info("Выпуск %s: полос %d, из них оглавление/указатель %d", issue_key, len(pages), len(toc_pages))

    # Этап 1: полосы оглавления/указателя — без списка статей (его ещё нет), с видом полосы в промпте.
    toc_jobs = [PageJob(rel, "toc", kind, year) for rel, kind in toc_pages.items()]
    toc_results = _recognise_many(client, spec, params, toc_jobs, reuse, stats)
    # Слияние ответов по видам в порядке полос выпуска: продолжения списка дописываются к
    # рубрикам предыдущей полосы того же вида. Сбои этапа toc в результатах отсутствуют.
    tocs = build_issue_toc(pages, toc_pages, toc_results)
    if toc_pages:
        # toc.json (машинный, для повторов и сборки) и toc.md (глазами) в папку выпуска под out-dir.
        _write_issue_toc(params.out_dir, issue_key, tocs)
        for kind, issue_toc in tocs.items():
            logger.info(
                "%s: %s — рубрик %d, статей %d, полос %d (продолжений %d)",
                issue_key,
                kind,
                len(issue_toc.rubrics),
                len(issue_toc.articles),
                len(issue_toc.pages),
                issue_toc.continuations,
            )
    # Списки в промпт этапа page берутся только из «Содержания»: годовой указатель описывает весь
    # год, а не этот выпуск. Нет «Содержания» — списки пустые, полосы идут без структуры.
    rubrics, articles = toc_module.prompt_lists(tocs.get(toc_module.KIND_CONTENTS))
    # Отпечаток списков: пишется в meta каждой полосы, и is_done считает полосу готовой только при
    # совпадении — так после изменившегося оглавления обычные полосы пересчитываются сами.
    digest = toc_module.toc_hash(rubrics, articles)

    # Этап 2: все остальные полосы выпуска, с рубриками и статьями «Содержания» в промпте.
    regular = [rel for rel in pages if rel not in toc_pages]
    page_jobs = [PageJob(rel, "page", "none", year, tuple(rubrics), tuple(articles), digest) for rel in regular]
    page_results = _recognise_many(client, spec, params, page_jobs, reuse, stats)

    # Fallback: обычные полосы, на которых модель увидела оглавление или указатель. Полосы, где
    # человек поставил вето «Не оглавление», не считаются — модель на них ошибается регулярно
    # (списки литературы, программы, таблицы). Сбои (нет результата) тоже не считаются.
    missed = [
        (rel, result.toc_kind)
        for rel in regular
        if (result := page_results.get(rel)) is not None
        and result.toc_kind != "none"
        and not flags_for(rel, params.flags).force_is_not_toc
    ]
    if not missed:
        return
    # В итоговую строку прогона — всегда, независимо от того, будет ли повтор.
    stats.missed.append(f"{issue_key}: " + ", ".join(rel.name for rel, _ in missed))
    # Решение — по --on-missed-toc: redo/skip без вопросов, ask — вопрос в терминал (без терминала
    # = skip). Повтор — только один круг: вложенный вызов получает копию params с on_missed_toc="skip",
    # иначе модель, увидев оглавление на очередной полосе, гоняла бы выпуск по кругу.
    if decide_redo(params.on_missed_toc, issue_key, missed):
        stats.redone_issues.append(issue_key)
        # Тот же выпуск заново: найденные полосы — как оглавление, список пересобирается, обычные
        # полосы с новым toc_hash не считаются готовыми и распознаются ещё раз.
        run_issue(client, spec, replace(params, on_missed_toc="skip"), issue_key, pages, stats, extra_toc=dict(missed))
    else:
        # Без повтора: полосы в missed_toc.txt — человеку на разметку в CVAT; в базу не пишем.
        _append_missed(params.out_dir, issue_key, missed)


def collect_meta(out_dir: Path) -> list[dict]:
    """Все .meta.json под out_dir: сводка строится по ним, а не по одному прогону."""
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
    """``summary.csv`` в корне out_dir: по строке на полосу, колонки — :data:`SUMMARY_FIELDS`."""
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
    """Весь вход по выпускам; возвращает сводку. summary.csv пересобирается по всем meta под out-dir."""
    stats = PipelineStats()
    # Отбор входа: обход in-dir или файл --pages, затем --only-year/--only-issue и --limit.
    rels = list_pages(params.in_dir, params.pages_file, params.only_year, params.only_issue, params.limit)
    # {«год/выпуск»: [полосы по имени]} — единица работы дальше выпуск, не полоса.
    groups = group_by_issue(rels)
    stats.pages = len(rels)
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
    # Выпуски идут строго по одному: этап page зависит от этапа toc того же выпуска, а полосы
    # внутри этапа распараллеливает run_issue своим пулом потоков.
    for issue_key, pages in groups.items():
        stats.issues += 1
        # Весь цикл выпуска: полосы оглавления → toc.json → остальные полосы со списком → fallback.
        run_issue(client, spec, params, issue_key, pages, stats)
    # Сколько полос входа база (или списки) считает оглавлением/указателем — для итоговой строки.
    stats.toc_pages = sum(1 for rel in rels if flags_for(rel, params.flags).toc_kind is not None)
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
        "готово за %.0f с: выпусков %d, полос %d, запросов %d (готовых пропущено %d), сбоев %d, "
        "стоимость $%.4f, сводка %s",
        time.monotonic() - started,
        stats.issues,
        stats.pages,
        stats.requests,
        stats.reused,
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
    if stats.missed:
        logger.warning("оглавления вне базы (%d выпусков): %s", len(stats.missed), "; ".join(stats.missed))
    return stats
