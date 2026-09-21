"""Команды пакета: ``uv run python -m ocr_utils.external_ocr_services <команда>``.

Докстринги команд — это их ``--help``, поэтому они короткие; смысл каждого аргумента описан в
``help=`` соответствующей опции, а сборка параметров прогона прокомментирована в теле ``run``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from ocr_utils.external_ocr_services import models as registry
from ocr_utils.external_ocr_services.assemble import assemble_issue, issue_pages, list_issues, log_assembly, write_issue
from ocr_utils.external_ocr_services.boundary import BoundaryChecker, CheckerStats, JoinKind
from ocr_utils.external_ocr_services.heading_check import HeadingChecker
from ocr_utils.external_ocr_services.client import DEFAULT_ATTEMPTS, DEFAULT_TIMEOUT, OpenRouterClient, api_key_from
from ocr_utils.external_ocr_services.models import Reasoning
from ocr_utils.external_ocr_services.ocr import DEFAULT_MAX_TOKENS, RunOptions, output_paths, read_meta
from ocr_utils.external_ocr_services.render import to_markdown
from ocr_utils.external_ocr_services.schema import ParseError, Stage, parse_json_text
from ocr_utils.external_ocr_services.pages import flags_from_db, flags_from_lists
from ocr_utils.external_ocr_services.pipeline import OnMissedToc, PipelineParams, RedoScope, run_pipeline
from ocr_utils.external_ocr_services.tiling import DEFAULT_MAX_MODEL_TILE, DEFAULT_MAX_SRC_TILE, DEFAULT_QUALITY

logger = logging.getLogger("ocr_utils.external_ocr_services")


def _setup_logging(pages_dir: Path | None, level: str) -> None:
    """Лог в терминал и, если задан ``pages_dir``, ещё и в ``run.log`` рядом с выходом.

    Args:
        pages_dir: Корень выхода прогона; ``None`` — только терминал (команды без выхода на диск).
        level: Имя уровня logging (``INFO``, ``DEBUG`` …); неизвестное — INFO.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if pages_dir is not None:
        pages_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(pages_dir / "run.log", encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


@click.group()
@click.option("--log-level", default="INFO", show_default=True)
def main(log_level: str) -> None:
    """Внешний OCR выпусков через VLM: оглавление → список статей → остальные полосы."""
    _setup_logging(None, log_level)


@main.command()
@click.option(
    "--in-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Полосы: {год}/{выпуск}/{полоса}.",
)
@click.option(
    "--pages-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Папка полос той же раскладки: .md, .json, .meta.json; на выпуск toc.json/toc.md и sidecar; summary.csv, run.log.",
)
@click.option(
    "--issues-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Папка md выпусков: только {год}/{год}_{выпуск}.md. Обязательна, если сборка не выключена (--no-assemble).",
)
@click.option(
    "--debug-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Сырые ответы, промпты и отправленные тайлы.",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Кэш запросов: каждый запрос к модели — своя папка с промптами и ответом; тот же запрос второй раз "
    "берётся с диска, а не из сети (README, «Кэш запросов»).",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="База разметки (обычно уточнённая, после from-cvat): флаги оглавления и вето. Только чтение.",
)
@click.option("--pack-name", default=None, help="Имя пака в базе; по умолчанию — имя папки --in-dir.")
@click.option(
    "--toc-lists",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Запасной вход без базы: корень списков <год>/<выпуск>/toc_pages.txt (scan_markup toc-pages).",
)
@click.option(
    "--model",
    "model_name",
    default=registry.DEFAULT_MODEL,
    show_default=True,
    help="Имя модели из реестра (см. models).",
)
@click.option(
    "--max-src-tile-size",
    type=int,
    default=DEFAULT_MAX_SRC_TILE,
    show_default=True,
    help="Шаг сетки тайлов в пикселях исходника (пак-1, 600 dpi: обычная полоса 1×2, разворот 2×2).",
)
@click.option(
    "--max-model-tile-size",
    type=int,
    default=DEFAULT_MAX_MODEL_TILE,
    show_default=True,
    help="Длинная сторона тайла после уменьшения, px.",
)
@click.option("--quality", type=int, default=DEFAULT_QUALITY, show_default=True, help="Качество JPEG тайла.")
@click.option("--source", default="", help="Описание издания для промпта; «{year}» заменяется годом выпуска.")
@click.option(
    "--second-pass/--no-second-pass",
    default=False,
    show_default=True,
    help="Второй проход по полосам с is_damaged: та же полоса ещё раз с подсказкой из первого ответа. "
    "Выключен: на повреждённых полосах он то чинил, то портил текст (reports/external_ocr_hyphenation_second_pass.md).",
)
@click.option(
    "--join-hyphens/--no-join-hyphens",
    default=True,
    show_default=True,
    help="Склеивать разорванные переносы («кре-диты» → «кредиты») по словарю pymorphy3 после разбора; склейки — в meta.",
)
@click.option(
    "--second-pass-transcript",
    is_flag=True,
    help="Во второй проход (если включён) передавать и полный текст первого (иначе только описание, строки и счётчики).",
)
@click.option(
    "--reasoning",
    # NONE («модель параметра не знает») — свойство модели из реестра, из CLI его не задать.
    type=click.Choice([level.value for level in Reasoning if level is not Reasoning.NONE]),
    default=None,
    help="Переопределить уровень рассуждений из реестра.",
)
@click.option("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, show_default=True)
@click.option(
    "--jobs",
    type=int,
    default=4,
    show_default=True,
    help="Параллельных запросов; это сеть и лимиты провайдера, а не CPU.",
)
@click.option(
    "--attempts", type=int, default=DEFAULT_ATTEMPTS, show_default=True, help="Попыток на запрос при 429/5xx."
)
@click.option("--timeout", type=float, default=DEFAULT_TIMEOUT, show_default=True, help="Таймаут одного запроса, с.")
@click.option("--only-year", default=None, help="Только этот годовой комплект.")
@click.option("--only-issue", default=None, help="Только этот выпуск (имя папки).")
@click.option(
    "--pages",
    "pages_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Файл со списком относительных путей полос.",
)
@click.option("--limit", type=int, default=None, help="Взять только первые N полос.")
@click.option("--skip-done", is_flag=True, help="Не запрашивать полосы с готовым .json и тем же списком статей.")
@click.option(
    "--assemble/--no-assemble",
    default=True,
    show_default=True,
    help="После каждого выпуска собирать его в один markdown ({год}/{выпуск}.md и .pages.json рядом с папкой полос).",
)
@click.option(
    "--check-boundaries/--no-check-boundaries",
    default=True,
    show_default=True,
    help="При сборке показывать сомнительные стыки полос модели (полоски строк обеих полос, один запрос на выпуск).",
)
@click.option(
    "--check-headings/--no-check-headings",
    default=True,
    show_default=True,
    help="При сборке спрашивать модель (текстом, без картинок) о статьях, оставшихся без `#` после сверки с оглавлением.",
)
@click.option(
    "--on-missed-toc",
    type=click.Choice([mode.value for mode in OnMissedToc]),
    default=OnMissedToc.REDO.value,
    show_default=True,
    help="Модель нашла оглавление вне базы: redo — перераспознать выпуск с ним (умолчание), ask — спросить в терминале (без терминала = skip), skip — только записать в missed_toc.txt.",
)
@click.option(
    "--redo-scope",
    type=click.Choice([scope.value for scope in RedoScope]),
    default=RedoScope.STRUCTURED.value,
    show_default=True,
    help="Какие обычные полосы запрашивать заново на круге повтора: structured — только с заголовками, рубриками/маркерами, авторами, правками пост-обработки и начала статей по новому оглавлению (остальные берутся с диска); all — все.",
)
@click.option("--api-key", default=None, help="Ключ OpenRouter; по умолчанию $OPENROUTER_API_KEY.")
@click.option("--log-level", default="INFO", show_default=True)
def run(
    in_dir: Path,
    pages_dir: Path,
    issues_dir: Path | None,
    debug_dir: Path | None,
    cache_dir: Path | None,
    db_path: Path | None,
    pack_name: str | None,
    toc_lists: Path | None,
    model_name: str,
    max_src_tile_size: int,
    max_model_tile_size: int,
    quality: int,
    source: str,
    second_pass: bool,
    join_hyphens: bool,
    second_pass_transcript: bool,
    reasoning: str | None,
    max_tokens: int,
    jobs: int,
    attempts: int,
    timeout: float,
    only_year: str | None,
    only_issue: str | None,
    pages_file: Path | None,
    limit: int | None,
    skip_done: bool,
    assemble: bool,
    check_boundaries: bool,
    check_headings: bool,
    on_missed_toc: str,
    redo_scope: str,
    api_key: str | None,
    log_level: str,
) -> None:
    """Прогнать папку полос по выпускам: оглавление, потом остальные полосы со списком статей."""
    if assemble and issues_dir is None:
        raise click.UsageError("нужна --issues-dir (папка md выпусков) или --no-assemble")
    _setup_logging(pages_dir, log_level)
    # Модель — по короткому имени из реестра; неизвестное имя — понятная ошибка со списком.
    try:
        spec = registry.resolve(model_name)
    except KeyError as error:
        raise click.ClickException(str(error)) from None
    # Источник тегов оглавления: база разметки (только чтение) → списки toc_pages.txt → ничего
    # (тогда оглавление ловит только fallback по ответу модели).
    flags = None
    if db_path is not None:
        try:
            flags = flags_from_db(db_path, pack_name or in_dir.name)
        except LookupError as error:
            raise click.ClickException(str(error)) from None
        logger.info("флаги оглавления: %d полос из %s", len(flags), db_path)
    elif toc_lists is not None:
        flags = flags_from_lists(toc_lists)
        logger.info("флаги оглавления: %d полос из списков %s", len(flags), toc_lists)
    else:
        logger.warning("ни --db, ни --toc-lists: оглавление ищется только по ответу модели (fallback)")

    # Всё, что относится к одному запросу: тайлы, модель, второй проход, отладочный выход.
    options = RunOptions(
        max_src_tile=max_src_tile_size,
        max_model_tile=max_model_tile_size,
        quality=quality,
        reasoning=Reasoning(reasoning) if reasoning is not None else None,
        max_tokens=max_tokens,
        source=source,
        debug_dir=debug_dir,
        cache_dir=cache_dir,
        second_pass=second_pass,
        second_pass_transcript=second_pass_transcript,
        join_hyphens=join_hyphens,
    )
    # Всё, что относится к прогону целиком: пути, флаги, отбор входа, параллелизм, fallback.
    params = PipelineParams(
        in_dir=in_dir,
        pages_dir=pages_dir,
        issues_dir=issues_dir,
        options=options,
        flags=flags,
        jobs=max(1, jobs),
        skip_done=skip_done,
        assemble=assemble,
        check_boundaries=check_boundaries,
        check_headings=check_headings,
        on_missed_toc=OnMissedToc(on_missed_toc),
        redo_scope=RedoScope(redo_scope),
        pages_file=pages_file,
        only_year=only_year,
        only_issue=only_issue,
        limit=limit,
    )
    # Ключ: --api-key или $OPENROUTER_API_KEY; нет ни того, ни другого — ошибка без запуска.
    try:
        client = OpenRouterClient(api_key_from(api_key), timeout=timeout, attempts=attempts)
    except Exception as error:
        raise click.ClickException(str(error)) from None
    # FileNotFoundError — полоса из --pages отсутствует на диске: опечатка в списке, а не сбой.
    try:
        stats = run_pipeline(client, spec, params)
    except FileNotFoundError as error:
        raise click.ClickException(str(error)) from None
    # Итог прогона в терминал (то же есть в run.log): что запросили, что пропустили, почём.
    click.echo(
        f"Выпусков: {stats.issues}, полос: {stats.pages} (оглавление/указатель по базе: {stats.toc_pages}), "
        f"запросов: {stats.requests}, готовых пропущено: {stats.reused}, из кэша запросов: {stats.cache_hits}, "
        f"сбоев: {stats.failed}, стоимость ${stats.cost_usd:.4f}."
    )
    if second_pass:
        click.echo(f"Второй проход: {stats.second_passes} полос, оставлен первый у {stats.second_pass_kept_first}.")
    if stats.boundary_requests:
        click.echo(
            f"Стыки полос: запросов {stats.boundary_requests}, переписано {stats.boundary_rewritten}, "
            f"${stats.boundary_cost_usd:.4f}."
        )
    if stats.missed:
        click.echo("ОГЛАВЛЕНИЯ ВНЕ БАЗЫ: " + "; ".join(stats.missed))
    if stats.demoted_toc:
        click.echo("В БАЗЕ ОГЛАВЛЕНИЕ, МОДЕЛЬ — НЕТ (см. demoted_toc.txt): " + "; ".join(stats.demoted_toc))
    if stats.redone_issues:
        click.echo("Перераспознаны: " + ", ".join(stats.redone_issues))


@main.command("assemble")
@click.option(
    "--pages-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Папка полос прогона run: {год}/{выпуск}/{полоса}.json; сюда же ложится sidecar {год}_{выпуск}.pages.json.",
)
@click.option(
    "--issues-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Папка md выпусков: {год}/{год}_{выпуск}.md — и больше ничего.",
)
@click.option(
    "--in-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Полосы {год}/{выпуск}/{полоса} — для проверки стыков моделью (полоски строк); без него проверки нет.",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Кэш запросов (тот же, что у run): повтор проверки стыков берётся с диска.",
)
@click.option(
    "--check-boundaries/--no-check-boundaries",
    default=True,
    show_default=True,
    help="Показывать сомнительные стыки модели (один запрос на выпуск); нужен --in-dir и ключ OpenRouter.",
)
@click.option(
    "--check-headings/--no-check-headings",
    default=True,
    show_default=True,
    help="Спрашивать модель текстом о статьях без `#` после сверки с оглавлением (один запрос на выпуск); нужен ключ OpenRouter.",
)
@click.option(
    "--model", "model_name", default=registry.DEFAULT_MODEL, show_default=True, help="Модель для проверки стыков."
)
@click.option("--api-key", default=None, help="Ключ OpenRouter; по умолчанию $OPENROUTER_API_KEY.")
@click.option("--only-year", default=None, help="Только этот годовой комплект.")
@click.option("--only-issue", default=None, help="Только этот выпуск (имя папки).")
@click.option(
    "--join-hyphens/--no-join-hyphens",
    default=True,
    show_default=True,
    help="Сшивать слова с переносом на границе полос по словарю pymorphy3.",
)
@click.option(
    "--join-paragraphs/--no-join-paragraphs",
    default=True,
    show_default=True,
    help="Сшивать оборванное на границе полос предложение в один абзац.",
)
@click.option("--log-level", default="INFO", show_default=True)
def assemble_command(
    pages_dir: Path,
    issues_dir: Path,
    in_dir: Path | None,
    cache_dir: Path | None,
    check_boundaries: bool,
    check_headings: bool,
    model_name: str,
    api_key: str | None,
    only_year: str | None,
    only_issue: str | None,
    join_hyphens: bool,
    join_paragraphs: bool,
    log_level: str,
) -> None:
    """Собрать выпуски в один markdown каждый из готовых .json полос; к модели — только сомнительные стыки."""
    _setup_logging(pages_dir, log_level)
    keys = list_issues(pages_dir, only_year, only_issue)
    if not keys:
        raise click.ClickException(f"в {pages_dir} нет выпусков с готовыми .json полос")
    # Проверка стыков: нужны полосы на входе (полоски строк) и ключ; без --in-dir — только эвристики.
    checker = None
    heading_checker = None
    if check_boundaries and in_dir is None:
        logger.warning("без --in-dir стыки полос модели не показываются — только словарные эвристики")
    if (check_boundaries and in_dir is not None) or check_headings:
        try:
            spec = registry.resolve(model_name)
            client = OpenRouterClient(api_key_from(api_key))
        except Exception as error:
            raise click.ClickException(str(error)) from None
        if check_boundaries and in_dir is not None:
            checker = BoundaryChecker(client, spec, in_dir, cache_dir)
        if check_headings:
            heading_checker = HeadingChecker(client, spec, cache_dir)
    # Каждый выпуск — по своим .json в порядке имён файлов; итоги в лог и в терминал одной строкой.
    totals = {kind: 0 for kind in JoinKind}
    missing = 0
    checked = CheckerStats()
    heading_checked = CheckerStats()
    restored_by_model = 0
    for key in keys:
        assembly = assemble_issue(
            pages_dir,
            key,
            issue_pages(pages_dir, key),
            join_hyphens=join_hyphens,
            join_paragraphs_across=join_paragraphs,
            checker=checker,
            heading_checker=heading_checker,
        )
        log_assembly(assembly, write_issue(pages_dir, issues_dir, assembly))
        for kind in JoinKind:
            totals[kind] += assembly.count(kind)
        missing += len(assembly.missing)
        checked = checked + assembly.checked
        heading_checked = heading_checked + assembly.heading_checked
        restored_by_model += sum(1 for a in assembly.reconcile.articles if a.get("source") == "model")
    click.echo(
        f"Выпусков собрано: {len(keys)}; переносов через границу: {totals[JoinKind.HYPHEN]} "
        f"(составных {totals[JoinKind.COMPOUND]}, дублей половины {totals[JoinKind.DUPLICATE]}), "
        f"абзацев сшито: {totals[JoinKind.PARAGRAPH]}, полос без результата: {missing}."
    )
    if checker is not None:
        click.echo(
            f"Стыки полос: запросов {checked.requests} (из кэша {checked.cache_hits}), переписано {totals[JoinKind.MODEL]}, "
            f"сбоев {checked.errors}, ${checked.cost_usd:.4f}."
        )
    if heading_checker is not None:
        click.echo(
            f"Заголовки без `#`: запросов {heading_checked.requests} (из кэша {heading_checked.cache_hits}), "
            f"восстановлено по ответу модели {restored_by_model}, сбоев {heading_checked.errors}, ${heading_checked.cost_usd:.4f}."
        )


@main.command("normalize")
@click.option("--pages-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--only-year", default=None)
@click.option("--only-issue", default=None)
@click.option("--log-level", default="INFO", show_default=True)
def normalize_command(pages_dir: Path, only_year: str | None, only_issue: str | None, log_level: str) -> None:
    """Переразобрать готовые .json полос текущим разбором и переписать .json/.md — без запросов к модели.

    Разбор приводит старые форматы к текущему (сноски `[^1]` → надстрочные цифры, старые теги,
    escape); md выпуска и так собирается через разбор, а полосные файлы иначе остались бы в старом
    виде. Meta не трогается; полоса без .json или с ошибкой пропускается.
    """
    _setup_logging(pages_dir, log_level)
    keys = list_issues(pages_dir, only_year, only_issue)
    if not keys:
        raise click.ClickException(f"в {pages_dir} нет выпусков с готовыми .json полос")
    pages = changed = 0
    for key in keys:
        for rel in issue_pages(pages_dir, key):
            paths = output_paths(pages_dir, rel)
            meta = read_meta(pages_dir, rel) or {}
            if meta.get("error") or meta.get("parse_error"):
                continue
            try:
                result = parse_json_text(paths.json.read_text(encoding="utf-8"), Stage(meta.get("stage") or Stage.PAGE))
            except ParseError as error:
                logger.warning("%s: .json не читается (%s), пропущена", rel, error)
                continue
            pages += 1
            new_json = result.to_json()
            if new_json != paths.json.read_text(encoding="utf-8"):
                changed += 1
            paths.json.write_text(new_json, encoding="utf-8")
            paths.md.write_text(to_markdown(result), encoding="utf-8")
        logger.info("%s: полосы переразобраны", key)
    click.echo(f"Полос переразобрано: {pages}, из них изменилось: {changed} (выпусков {len(keys)}).")


@main.command("models")
def models_command() -> None:
    """Реестр моделей со справочными ценами."""
    # Таблица фиксированной ширины; звёздочка — модель по умолчанию.
    click.echo(f"{'имя':24} {'id':40} {'$/M in':>7} {'$/M out':>8} {'json':12} {'reasoning':9} заметки")
    for spec in registry.MODELS:
        mark = "*" if spec.name == registry.DEFAULT_MODEL else " "
        click.echo(
            f"{mark}{spec.name:23} {spec.openrouter_id:40} {spec.price_in_per_m:7.2f} {spec.price_out_per_m:8.2f} "
            f"{spec.json_mode:12} {spec.reasoning:9} {spec.notes}"
        )
    click.echo("\n* — умолчание")


@main.command()
@click.option("--api-key", default=None)
def balance(api_key: str | None) -> None:
    """Баланс ключа OpenRouter — сверить расход до и после прогона."""
    from ocr_utils.external_ocr_services.client import credits

    # OpenRouter отдаёт «куплено» и «потрачено» за всё время; остаток — их разность.
    data = credits(api_key_from(api_key))
    total, used = float(data.get("total_credits") or 0), float(data.get("total_usage") or 0)
    click.echo(f"куплено ${total:.4f}, потрачено ${used:.4f}, остаток ${total - used:.4f}")
