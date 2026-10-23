"""Команды стенда: ``estimate``, ``run`` (синхронно), ``batch-submit`` / ``batch-collect`` (Batch API), ``report``, ``check``."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path

import click

from ocr_utils.external_ocr_services.cache import cache_for
from ocr_utils.external_ocr_services.client import OpenRouterClient, api_key_from, credits
from research.article_problems import PROMPT_VERSION
from research.article_problems.articles import Article, iter_articles, parse_article_key
from research.article_problems.batch import collect, submit
from research.article_problems.compare import load_rows, write_comparison
from research.article_problems.export import CLASS_DIRS, ModelRun, export_by_class
from research.article_problems.payload import DEFAULT_MODEL, ReasoningLevel, Transport, Variant, estimate_cost_usd
from research.article_problems.report import load_results, write_report
from research.article_problems.run import run_sync
from research.article_problems.schema import CLASS_TITLES

logger = logging.getLogger(__name__)

DEFAULT_PACK_DIR = Path("/mnt/dump3/yandex_disk_linux_baby_zergling/Общее/Фотки/МТС/md-файлы/пак-1")
# Выход — на SSD рядом с разметкой; на Я.Диск (там же, где вход) ничего не пишется.
DEFAULT_WORK_DIR = Path("/mnt/hotstore/scan_processing/mts_markup/article_problems")


def select_articles(pack_dir: Path, issues: str | None, articles: str | None, whole_pack: bool) -> list[Article]:
    """Статьи прогона по опциям CLI.

    Args:
        pack_dir: Папка пака.
        issues: Выпуски через запятую (``1968_09,1973_05``).
        articles: Статьи через запятую (``1968_09:A15,1973_05:A6``).
        whole_pack: Весь пак.

    Returns:
        Статьи в порядке выпусков; ни одной опции — ``UsageError``.
    """
    if whole_pack:
        return list(iter_articles(pack_dir))
    selected: list[Article] = []
    if issues:
        selected += list(iter_articles(pack_dir, [name.strip() for name in issues.split(",") if name.strip()]))
    if articles:
        wanted = [parse_article_key(key.strip()) for key in articles.split(",") if key.strip()]
        by_issue: dict[str, set[str]] = {}
        for issue, article_id in wanted:
            by_issue.setdefault(issue, set()).add(article_id)
        for issue, ids in by_issue.items():
            found = [article for article in iter_articles(pack_dir, [issue]) if article.article_id in ids]
            missing = ids - {article.article_id for article in found}
            if missing:
                raise click.UsageError(f"в выпуске {issue} нет статей {sorted(missing)}")
            selected += found
    if not selected:
        raise click.UsageError("укажите --issues, --articles или --all")
    # Дубли (статья и в --issues, и в --articles) убираются с сохранением порядка.
    unique = {article.key: article for article in selected}
    return list(unique.values())


def selection_options(func):
    """Общие опции выбора статей и варианта прогона."""
    options = [
        click.option("--pack-dir", type=click.Path(path_type=Path, file_okay=False), default=DEFAULT_PACK_DIR),
        click.option("--work-dir", type=click.Path(path_type=Path, file_okay=False), default=DEFAULT_WORK_DIR),
        click.option("--issues", help="Выпуски через запятую: 1968_09,1973_05."),
        click.option("--articles", help="Статьи через запятую: 1968_09:A15."),
        click.option("--all", "whole_pack", is_flag=True, help="Весь пак."),
        click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Базовый слаг модели OpenRouter."),
        click.option(
            "--transport",
            type=click.Choice([item.value for item in Transport]),
            default=Transport.SYNC.value,
            show_default=True,
            help="sync — chat/completions сразу; batch — Batch API (:batch, до 24 ч).",
        ),
        click.option(
            "--reasoning",
            type=click.Choice([item.value for item in ReasoningLevel]),
            default=ReasoningLevel.OFF.value,
            show_default=True,
        ),
        click.option("--prompt-version", type=int, default=PROMPT_VERSION, show_default=True),
        click.option("--tag", default="", help="Суффикс папки сводки: runs/{вариант}_{tag} (пилот, кандидаты…)."),
    ]
    for option in reversed(options):
        func = option(func)
    return func


def make_variant(model: str, transport: str, reasoning: str, prompt_version: int) -> Variant:
    """Вариант прогона из опций CLI.

    Args:
        model: Базовый слаг модели.
        transport: ``sync`` / ``batch``.
        reasoning: ``off`` / ``low`` / ``high``.
        prompt_version: Версия промпта.
    """
    return Variant(model, Transport(transport), ReasoningLevel(reasoning), prompt_version)


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Подробный лог.")
def cli(verbose: bool) -> None:
    """Отбор статей пака о проблемах предприятий через OpenRouter."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # SDK и httpx пишут каждую попытку на INFO — это шум рядом с логом статей.
    logging.getLogger("httpx").setLevel(logging.WARNING)


@cli.command()
@selection_options
def estimate(
    pack_dir, work_dir, issues, articles, whole_pack, model, transport, reasoning, prompt_version, tag
) -> None:
    """Прикидка цены прогона по справочному прайсу (без запросов)."""
    selected = select_articles(pack_dir, issues, articles, whole_pack)
    for level in ReasoningLevel:
        for mode in Transport:
            variant = make_variant(model, mode.value, level.value, prompt_version)
            click.echo(f"{variant.label:24} {len(selected)} статей ≈ ${estimate_cost_usd(selected, variant):.3f}")


@cli.command()
@selection_options
@click.option("--jobs", type=int, default=16, show_default=True, help="Одновременных запросов.")
@click.option("--budget-usd", type=float, default=10.0, show_default=True, help="Потолок трат прогона.")
@click.option("--api-key", default=None, help="Ключ OpenRouter; по умолчанию $OPENROUTER_API_KEY.")
def run(
    pack_dir,
    work_dir,
    issues,
    articles,
    whole_pack,
    model,
    transport,
    reasoning,
    prompt_version,
    tag,
    jobs,
    budget_usd,
    api_key,
) -> None:
    """Синхронный прогон статей и сводка."""
    variant = make_variant(model, transport, reasoning, prompt_version)
    if variant.transport != Transport.SYNC:
        raise click.UsageError("run — синхронный путь; для Batch API — batch-submit и batch-collect")
    selected = select_articles(pack_dir, issues, articles, whole_pack)
    key = api_key_from(api_key)
    before = credits(key)
    cache_root = work_dir / "cache"
    client = OpenRouterClient(key, timeout=600.0)
    click.echo(f"{len(selected)} статей, прикидка ≈ ${estimate_cost_usd(selected, variant):.3f}, баланс {before}")
    stats = run_sync(selected, variant, client, cache_for(cache_root), cache_root, jobs, budget_usd)
    click.echo(f"итог: {stats}")
    _report(selected, variant, work_dir, tag)
    after = credits(key)
    click.echo(f"баланс после: {after}; списано по балансу ${after['total_usage'] - before['total_usage']:.4f}")


@cli.command("batch-submit")
@selection_options
@click.option("--budget-usd", type=float, default=10.0, show_default=True, help="Не отправлять, если прикидка выше.")
@click.option("--api-key", default=None)
def batch_submit(
    pack_dir,
    work_dir,
    issues,
    articles,
    whole_pack,
    model,
    transport,
    reasoning,
    prompt_version,
    tag,
    budget_usd,
    api_key,
) -> None:
    """Отправить статьи в Batch API (только те, что не в кэше и не в ждущих пачках)."""
    variant = make_variant(model, Transport.BATCH.value, reasoning, prompt_version)
    selected = select_articles(pack_dir, issues, articles, whole_pack)
    estimate_usd = estimate_cost_usd(selected, variant)
    click.echo(f"{len(selected)} статей, прикидка ≈ ${estimate_usd:.3f}")
    if estimate_usd > budget_usd:
        raise click.ClickException(f"прикидка ${estimate_usd:.2f} выше бюджета ${budget_usd:.2f}")
    cache_root = work_dir / "cache"
    ids = submit(selected, variant, api_key_from(api_key), cache_for(cache_root), cache_root, work_dir / "batches")
    click.echo(f"отправлено пачек: {len(ids)}: {ids}")


@cli.command("batch-collect")
@click.option("--work-dir", type=click.Path(path_type=Path, file_okay=False), default=DEFAULT_WORK_DIR)
@click.option("--wait", is_flag=True, help="Ждать, пока все пачки не завершатся.")
@click.option("--poll-seconds", type=int, default=300, show_default=True)
@click.option("--api-key", default=None)
def batch_collect(work_dir, wait, poll_seconds, api_key) -> None:
    """Забрать готовые пачки в кэш; с --wait — опрашивать до конца."""
    key = api_key_from(api_key)
    cache_root = work_dir / "cache"
    while True:
        stats = collect(work_dir / "batches", key, cache_for(cache_root))
        click.echo(f"{time.strftime('%H:%M:%S')} {stats}")
        if not wait or stats.batches_pending == 0:
            break
        time.sleep(poll_seconds)


def _report(selected: list[Article], variant: Variant, work_dir: Path, tag: str) -> Path:
    """Собрать сводку прогона из кэша в ``runs/{метка}``.

    Args:
        selected: Статьи прогона.
        variant: Вариант.
        work_dir: Рабочая папка стенда.
        tag: Суффикс папки сводки; пусто — только метка варианта.

    Returns:
        Папка сводки.
    """
    cache_root = work_dir / "cache"
    # Ответ из пачки, обрезанный или не разобранный, подменяется синхронным повтором той же модели.
    fallbacks = (replace(variant, transport=Transport.SYNC),) if variant.transport == Transport.BATCH else ()
    results = load_results(selected, variant, cache_for(cache_root), cache_root, fallbacks)
    out_dir = work_dir / "runs" / (f"{variant.label}_{tag}" if tag else variant.label)
    numbers = write_report(results, variant, out_dir)
    click.echo(f"сводка: {out_dir} ({numbers})")
    return out_dir


@cli.command()
@selection_options
def report(pack_dir, work_dir, issues, articles, whole_pack, model, transport, reasoning, prompt_version, tag) -> None:
    """Сводка прогона из кэша без запросов."""
    variant = make_variant(model, transport, reasoning, prompt_version)
    _report(select_articles(pack_dir, issues, articles, whole_pack), variant, work_dir, tag)


@cli.command()
@selection_options
def check(pack_dir, work_dir, issues, articles, whole_pack, model, transport, reasoning, prompt_version, tag) -> None:
    """Показать разобранные ответы по статьям из кэша (классы, случаи, цитаты)."""
    variant = make_variant(model, transport, reasoning, prompt_version)
    cache_root = work_dir / "cache"
    results = load_results(
        select_articles(pack_dir, issues, articles, whole_pack), variant, cache_for(cache_root), cache_root
    )
    for result in results:
        article, answer = result.article, result.answer
        click.echo(f"\n=== {article.key} «{article.title}» ({article.rubric}), {article.n_chars} симв.")
        if answer is None:
            click.echo("   ответа в кэше нет")
            continue
        click.echo(
            f"   {answer.verdict.value}; классы {sorted(c.value for c in answer.classes)}; "
            f"{result.prompt_tokens}+{result.completion_tokens} ток. (рассужд. {result.reasoning_tokens}), ${result.cost_usd:.5f}"
        )
        for cls, text in sorted(answer.class_summaries.items(), key=lambda item: float(item[0].value)):
            click.echo(f"   [{cls.value}] {CLASS_TITLES[cls]}: {text}")
        for number, case in enumerate(answer.cases, start=1):
            flag = "" if case.quote_found else f"  ⚠ цитата не найдена ({case.quote_score:.0f})"
            click.echo(
                f"   {number}. {[c.value for c in case.classes]} {case.victim} ← {case.culprit} "
                f"[{case.culprit_kind.value}] ({case.product}); concrete={case.concrete}{flag}"
            )
            click.echo(f"      {case.explanation}")
            click.echo(f"      > {case.quote}  (строка {case.quote_line})")


@cli.command()
@click.argument("first", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.argument("second", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--names", default="A,B", show_default=True, help="Короткие имена прогонов через запятую.")
@click.option("--out-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
def compare(first, second, names, out_dir) -> None:
    """Сравнить два прогона по их results.jsonl: совпадение по классам и листы «только у одного»."""
    first_name, second_name = [name.strip() for name in names.split(",")]
    for item in write_comparison(first, second, first_name, second_name, out_dir):
        click.echo(
            f"{item.cls.value:>4}: {item.first:5} {item.second:5} обе {item.both:5} "
            f"только {item.only_first:4} / {item.only_second:4}  Жаккар {item.jaccard:.2f}"
        )
    click.echo(f"сравнение: {out_dir}")


@cli.command("export")
@click.option("--pack-dir", type=click.Path(path_type=Path, file_okay=False), default=DEFAULT_PACK_DIR)
@click.option(
    "--run",
    "runs",
    multiple=True,
    required=True,
    help="Прогон «Имя модели=путь/к/results.jsonl»; можно несколько — классы объединяются.",
)
@click.option("--out-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
def export_cmd(pack_dir, runs, out_dir) -> None:
    """Экспорт найденных статей: папка на класс, файл на статью с шапкой «кто нашёл и почему»."""
    model_runs = []
    for item in runs:
        name, _, path = item.partition("=")
        model_runs.append(ModelRun(name.strip(), load_rows(Path(path.strip()))))
    counts = export_by_class(list(iter_articles(pack_dir)), model_runs, out_dir)
    for cls, count in counts.items():
        click.echo(f"{CLASS_DIRS[cls]:40} {count}")
    click.echo(f"экспорт: {out_dir}")


def main() -> int:
    """Запуск CLI; код возврата для ``sys.exit``."""
    cli()
    return 0
