"""Команды пакета: ``uv run python -m ocr_utils.external_ocr_services <команда>``.

Докстринги команд — это их ``--help``, поэтому они короткие; смысл каждого аргумента описан в
``help=`` соответствующей опции, а сборка параметров прогона прокомментирована в теле ``run``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from ocr_utils.external_ocr_services import models as registry
from ocr_utils.external_ocr_services.client import DEFAULT_ATTEMPTS, DEFAULT_TIMEOUT, OpenRouterClient, api_key_from
from ocr_utils.external_ocr_services.models import Reasoning
from ocr_utils.external_ocr_services.ocr import DEFAULT_MAX_TOKENS, RunOptions
from ocr_utils.external_ocr_services.pages import flags_from_db, flags_from_lists
from ocr_utils.external_ocr_services.pipeline import OnMissedToc, PipelineParams, run_pipeline
from ocr_utils.external_ocr_services.tiling import DEFAULT_MAX_MODEL_TILE, DEFAULT_MAX_SRC_TILE, DEFAULT_QUALITY

logger = logging.getLogger("ocr_utils.external_ocr_services")


def _setup_logging(out_dir: Path | None, level: str) -> None:
    """Лог в терминал и, если задан ``out_dir``, ещё и в ``run.log`` рядом с выходом.

    Args:
        out_dir: Корень выхода прогона; ``None`` — только терминал (команды без выхода на диск).
        level: Имя уровня logging (``INFO``, ``DEBUG`` …); неизвестное — INFO.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(out_dir / "run.log", encoding="utf-8"))
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
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Выход той же раскладки: .md, .json, .meta.json; на выпуск toc.json/toc.md.",
)
@click.option(
    "--debug-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Сырые ответы, промпты и отправленные тайлы.",
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
    "--second-pass-transcript",
    is_flag=True,
    help="Во второй проход передавать и полный текст первого (иначе только описание, строки и счётчики).",
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
    "--on-missed-toc",
    type=click.Choice([mode.value for mode in OnMissedToc]),
    default=OnMissedToc.REDO.value,
    show_default=True,
    help="Модель нашла оглавление вне базы: redo — перераспознать выпуск с ним (умолчание), ask — спросить в терминале (без терминала = skip), skip — только записать в missed_toc.txt.",
)
@click.option("--api-key", default=None, help="Ключ OpenRouter; по умолчанию $OPENROUTER_API_KEY.")
@click.option("--log-level", default="INFO", show_default=True)
def run(
    in_dir: Path,
    out_dir: Path,
    debug_dir: Path | None,
    db_path: Path | None,
    pack_name: str | None,
    toc_lists: Path | None,
    model_name: str,
    max_src_tile_size: int,
    max_model_tile_size: int,
    quality: int,
    source: str,
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
    on_missed_toc: str,
    api_key: str | None,
    log_level: str,
) -> None:
    """Прогнать папку полос по выпускам: оглавление, потом остальные полосы со списком статей."""
    _setup_logging(out_dir, log_level)
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
        second_pass_transcript=second_pass_transcript,
    )
    # Всё, что относится к прогону целиком: пути, флаги, отбор входа, параллелизм, fallback.
    params = PipelineParams(
        in_dir=in_dir,
        out_dir=out_dir,
        options=options,
        flags=flags,
        jobs=max(1, jobs),
        skip_done=skip_done,
        on_missed_toc=OnMissedToc(on_missed_toc),
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
        f"запросов: {stats.requests}, готовых пропущено: {stats.reused}, сбоев: {stats.failed}, "
        f"стоимость ${stats.cost_usd:.4f}."
    )
    click.echo(f"Второй проход: {stats.second_passes} полос, оставлен первый у {stats.second_pass_kept_first}.")
    if stats.missed:
        click.echo("ОГЛАВЛЕНИЯ ВНЕ БАЗЫ: " + "; ".join(stats.missed))
    if stats.demoted_toc:
        click.echo("В БАЗЕ ОГЛАВЛЕНИЕ, МОДЕЛЬ — НЕТ (см. demoted_toc.txt): " + "; ".join(stats.demoted_toc))
    if stats.redone_issues:
        click.echo("Перераспознаны: " + ", ".join(stats.redone_issues))


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
