"""Команды пакета: ``uv run python -m research.external_ocr_models <команда>``.

Разделение обычное для исследовательских пакетов: ``run`` ходит в сеть (или на GPU) и
пишет выходы по полосам, ``report`` только читает готовые выходы и собирает сводку — его
можно перезапускать сколько угодно, не тратя ни цента.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click

from research.external_ocr_models import models as registry
from research.external_ocr_models.client import DEFAULT_ATTEMPTS, DEFAULT_TIMEOUT, OpenRouterClient, api_key_from
from research.external_ocr_models.imaging import DEFAULT_MAX_SIDE, DEFAULT_QUALITY
from research.external_ocr_models.ocr import DEFAULT_MAX_TOKENS, RunOptions, is_done, recognise_page

logger = logging.getLogger("research.external_ocr_models")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")

# Колонки сводки прогона. Пишется по завершении run и пересобирается report.
SUMMARY_FIELDS = (
    "page",
    "model",
    "provider",
    "json_mode_used",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cost_usd",
    "latency_s",
    "attempts",
    "page_number",
    "is_toc",
    "content_chars",
    "parse_error",
    "error",
)


def _setup_logging(out_dir: Path | None, level: str) -> None:
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


def list_pages(in_dir: Path, pages_file: Path | None, limit: int | None) -> list[Path]:
    """Относительные пути полос: все картинки под in_dir или только из файла-списка."""
    if pages_file is not None:
        # Строка: путь, дальше можно комментарий после «#».
        wanted = [line.split("#", 1)[0].strip() for line in pages_file.read_text(encoding="utf-8").splitlines()]
        rels = [Path(item) for item in wanted if item]
        missing = [rel for rel in rels if not (in_dir / rel).is_file()]
        if missing:
            raise click.ClickException(f"в {in_dir} нет полос из списка: {', '.join(map(str, missing))}")
    else:
        rels = sorted(
            path.relative_to(in_dir)
            for path in in_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    if limit is not None:
        rels = rels[:limit]
    return rels


def collect_meta(out_dir: Path) -> list[dict]:
    """Все .meta.json под out_dir — сводка строится по ним, а не по одному прогону: с
    --skip-done прогон видит только недоделанные полосы."""
    rows: list[dict] = []
    for path in sorted(out_dir.rglob("*.meta.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            logger.warning("битый %s", path)
    return rows


def write_summary(out_dir: Path, rows: list[dict]) -> Path:
    path = out_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item.get("page", "")):
            writer.writerow(row)
    return path


@click.group()
@click.option("--log-level", default="INFO", show_default=True)
def main(log_level: str) -> None:
    """Внешние OCR-модели: полоса журнала → размеченный markdown."""
    _setup_logging(None, log_level)


@main.command()
@click.option(
    "--in-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Папка с картинками; обходится рекурсивно.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Куда класть выходы; структура папок повторяет входную.",
)
@click.option("--model", "model_name", required=True, help="Имя модели из реестра (см. команду models).")
@click.option("--format", "out_format", type=click.Choice(["json", "md", "both"]), default="both", show_default=True)
@click.option(
    "--output-mode",
    type=click.Choice(["json", "markdown"]),
    default="json",
    show_default=True,
    help="Что просить у модели: JSON по схеме или markdown с YAML-шапкой.",
)
@click.option(
    "--strips",
    type=click.IntRange(1, 4),
    default=1,
    show_default=True,
    help="Резать полосу на N горизонтальных кусков и слать их одним запросом.",
)
@click.option(
    "--max-side",
    type=int,
    default=DEFAULT_MAX_SIDE,
    show_default=True,
    help="Длинная сторона картинки (или куска) в пикселях.",
)
@click.option("--quality", type=int, default=DEFAULT_QUALITY, show_default=True, help="Качество JPEG.")
@click.option(
    "--reasoning",
    type=click.Choice(["off", "low", "medium"]),
    default=None,
    help="Переопределить уровень рассуждений из реестра.",
)
@click.option("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, show_default=True)
@click.option(
    "--restore", is_flag=True, help="Достраивать повреждённые буквы по контексту и помечать их <restored>…</restored>."
)
@click.option("--hint", default="", help="Подсказка модели про полосы, например «левый край срезан корешком».")
@click.option("--jobs", type=int, default=4, show_default=True, help="Параллельных запросов; это сеть, а не CPU.")
@click.option(
    "--attempts", type=int, default=DEFAULT_ATTEMPTS, show_default=True, help="Попыток на запрос при 429/5xx."
)
@click.option("--timeout", type=float, default=DEFAULT_TIMEOUT, show_default=True, help="Таймаут одного запроса, с.")
@click.option(
    "--pages",
    "pages_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Файл со списком относительных путей полос (по одному на строку).",
)
@click.option("--limit", type=int, default=None, help="Взять только первые N полос.")
@click.option("--skip-done", is_flag=True, help="Пропускать полосы с готовым .meta.json без ошибки.")
@click.option("--api-key", default=None, help="Ключ OpenRouter; по умолчанию $OPENROUTER_API_KEY.")
@click.option("--log-level", default="INFO", show_default=True)
def run(
    in_dir: Path,
    out_dir: Path,
    model_name: str,
    out_format: str,
    output_mode: str,
    strips: int,
    max_side: int,
    quality: int,
    reasoning: str | None,
    max_tokens: int,
    restore: bool,
    hint: str,
    jobs: int,
    attempts: int,
    timeout: float,
    pages_file: Path | None,
    limit: int | None,
    skip_done: bool,
    api_key: str | None,
    log_level: str,
) -> None:
    """Прогнать папку картинок через одну модель."""
    _setup_logging(out_dir, log_level)
    try:
        spec = registry.resolve(model_name)
    except KeyError as error:
        raise click.ClickException(str(error)) from None
    options = RunOptions(
        output_mode=output_mode,
        strips=strips,
        max_side=max_side,
        quality=quality,
        reasoning=reasoning,
        max_tokens=max_tokens,
        restore=restore,
        hint=hint,
        write_json=out_format in ("json", "both"),
        write_md=out_format in ("md", "both"),
    )
    rels = list_pages(in_dir, pages_file, limit)
    if skip_done:
        before = len(rels)
        rels = [rel for rel in rels if not is_done(out_dir, rel)]
        logger.info("пропущено готовых: %d", before - len(rels))
    if not rels:
        logger.info("делать нечего")
        return

    if spec.local:
        from research.external_ocr_models.local import run_local

        rows = run_local(spec, in_dir, rels, out_dir, options)
    else:
        client = OpenRouterClient(api_key_from(api_key), timeout=timeout, attempts=attempts)
        rows = _run_remote(client, spec, in_dir, rels, out_dir, options, max(1, jobs))

    summary = write_summary(out_dir, collect_meta(out_dir))
    failed = [row for row in rows if row.get("error") or row.get("parse_error")]
    cost = sum(float(row.get("cost_usd") or 0.0) for row in rows)
    logger.info(
        "готово: %d полос, сбоев %d, стоимость $%.4f (%.3f ¢/полоса), сводка %s",
        len(rows),
        len(failed),
        cost,
        100 * cost / max(1, len(rows)),
        summary,
    )


def _run_remote(
    client: OpenRouterClient,
    spec: registry.ModelSpec,
    in_dir: Path,
    rels: list[Path],
    out_dir: Path,
    options: RunOptions,
    jobs: int,
) -> list[dict]:
    rows: list[dict] = []
    started = time.monotonic()
    logger.info(
        "%s (%s): %d полос, %d потоков, strips=%d, max_side=%d",
        spec.name,
        spec.openrouter_id,
        len(rels),
        jobs,
        options.strips,
        options.max_side,
    )
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(recognise_page, client, spec, in_dir / rel, rel, out_dir, options): rel for rel in rels}
        for future in as_completed(futures):
            rel = futures[future]
            try:
                meta = future.result()
            except Exception:  # неожиданное исключение — логируем и идём дальше
                logger.exception("%s: полоса упала", rel)
                meta = {"page": rel.as_posix(), "model": spec.name, "error": "исключение, см. run.log"}
            rows.append(meta)
            status = meta.get("error") or meta.get("parse_error") or "ok"
            logger.info(
                "%d/%d %s: %s, %s tok → %s tok, $%s, %.0f с",
                len(rows),
                len(rels),
                rel,
                status,
                meta.get("prompt_tokens", "-"),
                meta.get("completion_tokens", "-"),
                meta.get("cost_usd", "-"),
                meta.get("latency_s") or 0.0,
            )
    logger.info("прогон занял %.0f с", time.monotonic() - started)
    return rows


@main.command()
@click.option(
    "--out-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Корень с подпапками по моделям.",
)
@click.option("--models", "model_names", default=None, help="Через запятую: какие подпапки брать; по умолчанию все.")
@click.option(
    "--reference-pdf",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="PDF FineReader с текстовым слоем; страница i ↔ i-я полоса выпуска.",
)
@click.option(
    "--in-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Папка полос выпуска — нужна, чтобы сопоставить страницы PDF с именами.",
)
@click.option("--issue", default=None, help="Относительный путь выпуска внутри --in-dir, например 1966/03.")
@click.option(
    "--rotated-info-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Папка info/ из pack1_rotated_tables.",
)
@click.option(
    "--report",
    "report_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Куда положить markdown-отчёт.",
)
@click.option("--title", default="Внешние OCR-модели: сводка", show_default=True)
def report(
    out_root: Path,
    model_names: str | None,
    reference_pdf: Path | None,
    in_dir: Path | None,
    issue: str | None,
    rotated_info_dir: Path | None,
    report_path: Path | None,
    title: str,
) -> None:
    """Собрать сводку по готовым выходам (ничего не запрашивает)."""
    from research.external_ocr_models import evaluate, report as report_module

    wanted = set(model_names.split(",")) if model_names else None
    by_model: dict[str, list[evaluate.PageOutput]] = {}
    for model_dir in sorted(path for path in out_root.iterdir() if path.is_dir()):
        if wanted and model_dir.name not in wanted:
            continue
        outputs = evaluate.load_outputs(model_dir)
        if outputs:
            by_model[model_dir.name] = outputs
    if not by_model:
        raise click.ClickException(f"в {out_root} нет выходов")

    reference: dict[str, str] = {}
    expected_numbers: dict[str, str] = {}
    if in_dir is not None and issue is not None:
        names = list_pages(in_dir / issue, None, None)
        # Печатный номер полосы в этом журнале равен её порядковому номеру в выпуске
        # (обложка — 0 и без номера); это проверено по пробнику и даёт бесплатную метрику.
        expected_numbers = {name.stem: str(index) for index, name in enumerate(names) if index > 0}
        if reference_pdf is not None:
            texts = evaluate.finereader_pages(reference_pdf)
            if len(names) != len(texts):
                logger.warning(
                    "в PDF %d страниц, полос в %s — %d; сопоставляю по порядку", len(texts), issue, len(names)
                )
            for name, text in zip(names, texts):
                reference[name.stem] = evaluate.normalize(text)
    elif reference_pdf is not None:
        raise click.ClickException("--reference-pdf требует --in-dir и --issue")
    rotated: dict[str, list[str]] = {}
    if rotated_info_dir is not None and issue is not None:
        rotated = evaluate.rotated_phrases(rotated_info_dir, issue.replace("/", "_"))

    scores = evaluate.score_outputs(by_model, reference, rotated, expected_numbers)
    text = report_module.build_report(scores, title)
    click.echo(text)
    report_module.write_scores_csv(scores, out_root / "scores.csv")
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text, encoding="utf-8")
        logger.info("отчёт: %s, оценки: %s", report_path, out_root / "scores.csv")


@main.command("models")
def models_command() -> None:
    """Реестр моделей со справочными ценами."""
    click.echo(f"{'имя':24} {'id':44} {'$/M in':>7} {'$/M out':>8} {'json':12} {'reasoning':9} заметки")
    for spec in registry.OPENROUTER_MODELS + registry.LOCAL_ENGINES:
        mark = "*" if spec.name in registry.PROBE_MODELS else " "
        click.echo(
            f"{mark}{spec.name:23} {spec.openrouter_id:44} {spec.price_in_per_m:7.2f} {spec.price_out_per_m:8.2f} {spec.json_mode:12} {spec.reasoning:9} {spec.notes}"
        )
    click.echo("\n* — входит в пробник по умолчанию")


@main.command()
@click.option("--api-key", default=None)
def balance(api_key: str | None) -> None:
    """Баланс ключа OpenRouter — сверить расход до и после прогона."""
    from research.external_ocr_models.client import credits

    data = credits(api_key_from(api_key))
    total, used = float(data.get("total_credits") or 0), float(data.get("total_usage") or 0)
    click.echo(f"куплено ${total:.4f}, потрачено ${used:.4f}, остаток ${total - used:.4f}")
