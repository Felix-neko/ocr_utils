"""Команды пакета: ``uv run python -m research.geometry_regression <команда>``.

``run`` — тяжёлая часть: рендерит пары страниц, меряет и пишет JSON на страницу плюс общий
CSV. ``report`` — только читает: перефлаговывает CSV другими порогами, собирает сводку и
картинки «было | стало» по годам; его можно гонять сколько угодно.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import click

from research.geometry_regression import VERSION
from research.geometry_regression.metrics import Params, measure_pair
from research.geometry_regression.render import PdfPair, pair_pdfs, render_gray, to_work
from research.geometry_regression.report import (
    PageRow,
    load_labels,
    markdown_report,
    pair_name,
    read_csv,
    reflag,
    write_csv,
)
from research.geometry_regression.scoring import Thresholds

logger = logging.getLogger("research.geometry_regression")

# Страниц одного PDF на задачу пула: PDF открывается в воркере один раз на пачку.
CHUNK_PAGES = 8


def _init_worker() -> None:
    """Инициализатор воркера: без hugepage и потоков BLAS (``.claude/rules/gpu_and_pools.md``)."""
    import cv2
    import numpy as np
    import threadpoolctl

    np._core.multiarray._set_madvise_hugepage(False)
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)


def cache_path(out_dir: Path, pdf: str, page: int) -> Path:
    return out_dir / "cache" / pdf / f"p{page:03d}.json"


def _measure_chunk(args: tuple) -> list[dict]:
    """Одна пачка страниц одного выпуска; возвращает записи для CSV."""
    import fitz

    pair_dict, pages, out_dir, params_dict, skip_done = args
    pair = PdfPair(**pair_dict)
    params = Params(**params_dict)
    out_dir = Path(out_dir)
    results = []
    with fitz.open(pair.geo) as geo, fitz.open(pair.nogeo) as nogeo:
        for page in pages:
            path = cache_path(out_dir, pair.name, page)
            record = {"pdf": pair.name, "page": page, "metrics": {}, "error": ""}
            try:
                if skip_done and path.is_file():
                    cached = json.loads(path.read_text(encoding="utf-8"))
                    if cached.get("version") == VERSION and "metrics" in cached:
                        record["metrics"] = cached["metrics"]
                        results.append(record)
                        continue
                started = time.time()
                measure = measure_pair(render_gray(nogeo, page - 1), render_gray(geo, page - 1), params)
                measure.metrics["seconds"] = round(time.time() - started, 2)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "version": VERSION,
                            "metrics": measure.metrics,
                            "culprits": measure.culprits,
                            "raw": measure.raw,
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                record["metrics"] = measure.metrics
            except Exception as error:  # страница не должна валить прогон
                record["error"] = f"{type(error).__name__}: {error}"
            results.append(record)
    return results


def _parse_pages(text: str | None, total: int) -> list[int]:
    if not text:
        return list(range(1, total + 1))
    pages: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            low, high = part.split("-", 1)
            pages.extend(range(int(low), int(high) + 1))
        elif part:
            pages.append(int(part))
    return [p for p in pages if 1 <= p <= total]


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO), format="%(levelname)s %(name)s: %(message)s"
    )


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--log-level", default="INFO", show_default=True)
def main(log_level: str) -> None:
    """Страницы, где коррекция геометрии FineReader сделала хуже."""
    _setup_logging(log_level)


@main.command()
@click.option(
    "--geo-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="PDF с коррекцией геометрии",
)
@click.option(
    "--nogeo-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="PDF без коррекции",
)
@click.option("--out-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--jobs", default=16, show_default=True, type=int, help="воркеров; задача упирается в CPU")
@click.option("--only", multiple=True, help="подстрока имени PDF (можно несколько)")
@click.option("--pages", help="страницы для пробы: 1,5-9 (с единицы)")
@click.option("--limit", type=int, help="не больше стольких PDF")
@click.option("--skip-done/--redo", default=True, show_default=True, help="брать готовые JSON из кэша")
@click.option(
    "--stroke-min-mm",
    default=Params.stroke_min_mm,
    show_default=True,
    type=float,
    help="минимальная длина прямого штриха",
)
@click.option(
    "--line-min-mm",
    default=Params.line_min_mm,
    show_default=True,
    type=float,
    help="минимальная длина строки в попарных метриках",
)
def run(geo_dir, nogeo_dir, out_dir, jobs, only, pages, limit, skip_done, stroke_min_mm, line_min_mm) -> None:
    """Померить все пары страниц → JSON на страницу и metrics.csv."""
    from tqdm import tqdm

    pairs, notes = pair_pdfs(geo_dir, nogeo_dir)
    if only:
        pairs = [p for p in pairs if any(sub in p.name for sub in only)]
    if limit:
        pairs = pairs[:limit]
    for note in notes:
        logger.warning(note)
    params = Params(stroke_min_mm=stroke_min_mm, line_min_mm=line_min_mm)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(
        json.dumps(
            {
                "geo_dir": str(geo_dir),
                "nogeo_dir": str(nogeo_dir),
                "params": asdict(params),
                "version": VERSION,
                "notes": notes,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    tasks = []
    for pair in pairs:
        wanted = _parse_pages(pages, pair.pages)
        for start in range(0, len(wanted), CHUNK_PAGES):
            tasks.append((asdict(pair), wanted[start : start + CHUNK_PAGES], str(out_dir), asdict(params), skip_done))
    total_pages = sum(len(t[1]) for t in tasks)
    click.echo(f"PDF-пар: {len(pairs)}, страниц: {total_pages}, задач: {len(tasks)}, воркеров: {jobs}")
    thresholds = Thresholds()
    rows: list[PageRow] = []
    if jobs <= 1:
        outcomes = map(_measure_chunk, tasks)
    else:
        pool = ProcessPoolExecutor(
            max_workers=jobs, mp_context=multiprocessing.get_context("forkserver"), initializer=_init_worker
        )
        outcomes = pool.map(_measure_chunk, tasks)
    with tqdm(total=total_pages, desc="страницы", unit="стр") as bar:
        for records in outcomes:
            for record in records:
                row = PageRow(record["pdf"], record["page"], record["metrics"], error=record["error"])
                if not row.error:
                    row.apply(thresholds.apply(row.metrics))
                rows.append(row)
            bar.update(len(records))
    write_csv(out_dir / "metrics.csv", rows)
    errors = [r for r in rows if r.error]
    bad = sum(1 for r in rows if r.verdict == "bad")
    mixed = sum(1 for r in rows if r.verdict == "mixed")
    click.echo(
        f"Готово: {len(rows)} страниц, по порогам в коде bad: {bad}, mixed: {mixed}, ошибок: {len(errors)}. "
        f"CSV: {out_dir / 'metrics.csv'}"
    )
    for row in errors[:10]:
        click.echo(f"  {row.pdf} с.{row.page}: {row.error}")


def _write_pair(args: tuple) -> str:
    """Одна картинка «было | стало» (в пуле)."""
    import fitz

    geo, nogeo, pdf, page, culprit, field_raw, out_path, caption = args
    from research.geometry_regression.overlay import pair_image

    try:
        with fitz.open(nogeo) as document:
            before = to_work(render_gray(document, page - 1))
        with fitz.open(geo) as document:
            after = to_work(render_gray(document, page - 1))
        image = pair_image(
            before, after, culprit, f"было: {pdf} стр. {page} (без коррекции)", "стало: с коррекцией", field_raw
        )
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path, quality=85)
    except Exception as error:
        return f"{pdf} с.{page}: {error}"
    return ""


@main.command()
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="каталог прогона run (с metrics.csv и cache/)",
)
@click.option("--thr", multiple=True, help="перекрытие порога порчи или выигрыша: имя=число")
@click.option("--hard", default=2.0, show_default=True, type=float, help="порча не ниже — bad независимо от выигрыша")
@click.option(
    "--min-gain", default=1.0, show_default=True, type=float, help="выигрыш не ниже переводит мелкую порчу в mixed"
)
@click.option("--min-score", default=1.0, show_default=True, type=float, help="картинки для страниц с порчей не ниже")
@click.option("--max-score", type=float, help="и не выше (для выборки поясов глазами)")
@click.option(
    "--pairs-dir", type=click.Path(file_okay=False, path_type=Path), help="куда класть картинки (подпапка на год)"
)
@click.option("--md-report", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--labels", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="эталон pdf,page,label,note"
)
@click.option("--arrows/--no-arrows", default=False, show_default=True, help="третья панель со стрелками поля")
@click.option("--limit-pairs", type=int, help="не больше стольких картинок (по убыванию score)")
@click.option("--jobs", default=8, show_default=True, type=int)
@click.option("--list-thresholds", is_flag=True)
def report(
    out_dir,
    thr,
    hard,
    min_gain,
    min_score,
    max_score,
    pairs_dir,
    md_report,
    labels,
    arrows,
    limit_pairs,
    jobs,
    list_thresholds,
) -> None:
    """Перефлаговать metrics.csv, собрать сводку и картинки «было | стало» по вердиктам."""
    from tqdm import tqdm

    thresholds = Thresholds.parse(tuple(thr), hard, min_gain, ratio)
    if list_thresholds:
        click.echo(thresholds.describe())
        return
    rows = read_csv(out_dir / "metrics.csv")
    reflag(rows, thresholds)
    write_csv(out_dir / "metrics_flagged.csv", rows)
    label_map = load_labels(labels) if labels else None
    text = markdown_report(rows, thresholds, label_map, out_dir)
    if md_report:
        md_report.parent.mkdir(parents=True, exist_ok=True)
        md_report.write_text(text, encoding="utf-8")
        click.echo(f"Отчёт: {md_report}")
    click.echo("\n".join(text.splitlines()[:4]))
    if pairs_dir is None:
        return
    run_info = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    chosen = [r for r in rows if not r.error and r.score >= min_score and (max_score is None or r.score < max_score)]
    chosen.sort(key=lambda r: -r.score)
    if limit_pairs:
        chosen = chosen[:limit_pairs]
    tasks = []
    for row in chosen:
        cache = json.loads(cache_path(out_dir, row.pdf, row.page).read_text(encoding="utf-8"))
        culprits = cache.get("culprits", {})
        culprit = None
        if row.flags:
            best = max(row.flags, key=row.flags.get)
            culprit = culprits.get(best)
        field_raw = cache.get("raw", {}).get("field") if arrows else None
        # Раскладка pairs/<вердикт>/<год>/: сначала по вердикту, чтобы смотреть все bad подряд.
        out_path = pairs_dir / row.verdict / row.year / pair_name(row)
        geo = str(Path(run_info["geo_dir"]) / f"{row.pdf}.pdf")
        nogeo = str(Path(run_info["nogeo_dir"]) / f"{row.pdf}.pdf")
        caption = (
            f"стало: {row.verdict}, порча {row.score:.2f} ({row.reason}), выигрыш {row.gain:.2f} ({row.gain_reason})"
        )
        tasks.append((geo, nogeo, row.pdf, row.page, culprit, field_raw, str(out_path), caption))
    if jobs <= 1 or len(tasks) <= 1:
        outcomes = map(_write_pair, tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("forkserver"))
        outcomes = pool.map(_write_pair, tasks)
    errors = [e for e in tqdm(outcomes, total=len(tasks), desc="пары", unit="стр") if e]
    click.echo(f"Картинок: {len(tasks) - len(errors)} в {pairs_dir}; ошибок: {len(errors)}")
    for error in errors[:10]:
        click.echo(f"  {error}")


if __name__ == "__main__":
    main()


def _load_probe_pages(out_dir: Path, thr: tuple[str, ...], labels: Path | None):
    thresholds = Thresholds.parse(tuple(thr))
    rows = read_csv(out_dir / "metrics.csv")
    reflag(rows, thresholds)
    label_map = load_labels(labels) if labels else {}
    return rows, label_map


@main.command("vlm-probe")
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="каталог прогона run",
)
@click.option("--labels", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--thr", multiple=True, help="пороги классики для поясов: имя=число")
@click.option("--variants", default=",".join(("overlay", "side", "two", "crop")), show_default=True)
@click.option("--repeats", default=2, show_default=True, type=int)
@click.option("--per-belt", default=30, show_default=True, type=int, help="случайных страниц из каждого пояса score")
@click.option("--only-labelled", is_flag=True, help="только эталонные страницы")
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--model", default="deepseek-v41-flash", show_default=True)
@click.option("--api-key", help="ключ OpenRouter; иначе $OPENROUTER_API_KEY")
@click.option("--jobs", default=4, show_default=True, type=int, help="параллельных запросов (лимиты провайдера)")
@click.option("--budget-usd", default=2.0, show_default=True, type=float, help="стоп по сумме usage.cost")
@click.option("--skip-done/--redo", default=True, show_default=True)
@click.option("--dry-run", is_flag=True, help="только сохранить картинки в vlm/images/, без запросов")
def vlm_probe(
    out_dir,
    labels,
    thr,
    variants,
    repeats,
    per_belt,
    only_labelled,
    seed,
    model,
    api_key,
    jobs,
    budget_usd,
    skip_done,
    dry_run,
) -> None:
    """Спросить VLM «стало ли хуже?» по выборке страниц в нескольких вариантах картинки."""
    import csv
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import fitz
    from tqdm import tqdm

    from ocr_utils.external_ocr_services.client import api_key_from
    from ocr_utils.external_ocr_services.models import resolve
    from research.geometry_regression import vlm

    variant_list = [v.strip() for v in variants.split(",") if v.strip()]
    unknown = set(variant_list) - set(vlm.VARIANTS)
    if unknown:
        raise click.ClickException(
            f"неизвестные варианты: {', '.join(sorted(unknown))}; есть {', '.join(vlm.VARIANTS)}"
        )
    rows, label_map = _load_probe_pages(out_dir, thr, labels)
    probes = vlm.sample_pages(rows, label_map, per_belt, seed, only_labelled)
    run_info = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    dpi = float(run_info["params"]["dpi"])
    (out_dir / "vlm").mkdir(exist_ok=True)
    with (out_dir / "vlm" / "sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pdf", "page", "belt", "score"])
        writer.writerows((p.pdf, p.page, p.belt, f"{p.score:.3f}") for p in probes)
    click.echo(f"Страниц в выборке: {len(probes)}; варианты: {variant_list}; повторов: {repeats}")
    by_key = {r.key: r for r in rows}

    spec = resolve(model)
    client = None if dry_run else vlm.OpenRouterClient(api_key_from(api_key))
    spent = 0.0
    done = skipped = failed = 0
    pool = ThreadPoolExecutor(max_workers=jobs)
    futures = {}

    def submit(probe, variant, repeat, images):
        path = vlm.answer_path(out_dir, variant, probe.pdf, probe.page, repeat)
        futures[pool.submit(vlm.ask, client, spec, variant, images)] = (probe, variant, repeat, path)

    for probe in tqdm(probes, desc="страницы", unit="стр"):
        wanted = [
            (variant, repeat)
            for variant in variant_list
            for repeat in range(1, repeats + 1)
            if dry_run
            or not (skip_done and vlm.is_done(vlm.answer_path(out_dir, variant, probe.pdf, probe.page, repeat)))
        ]
        skipped += len(variant_list) * repeats - len(wanted)
        if not wanted:
            continue
        cache = json.loads(cache_path(out_dir, probe.pdf, probe.page).read_text(encoding="utf-8"))
        cache["flags"] = by_key[(probe.pdf, probe.page)].flags if (probe.pdf, probe.page) in by_key else {}
        with fitz.open(Path(run_info["nogeo_dir"]) / f"{probe.pdf}.pdf") as document:
            gray300_b = render_gray(document, probe.page - 1)
        with fitz.open(Path(run_info["geo_dir"]) / f"{probe.pdf}.pdf") as document:
            gray300_a = render_gray(document, probe.page - 1)
        before, after = to_work(gray300_b), to_work(gray300_a)
        for variant in {v for v, _ in wanted}:
            images = vlm.variant_images(variant, before, after, gray300_b, gray300_a, cache, dpi)
            if dry_run:
                for index, image in enumerate(images):
                    target = out_dir / "vlm" / "images" / variant / f"{probe.pdf}_p{probe.page:03d}_{index}.jpg"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    image.save(target, quality=85)
                continue
            for repeat in [r for v, r in wanted if v == variant]:
                if spent >= budget_usd:
                    break
                submit(probe, variant, repeat, images)
        # Забрать готовое, чтобы считать бюджет по ходу, а не в конце.
        for future in [f for f in list(futures) if f.done()]:
            probe_done, variant, repeat, path = futures.pop(future)
            spent, done, failed = _store_answer(future, probe_done, variant, repeat, path, spent, done, failed)
        if spent >= budget_usd:
            click.echo(f"Бюджет ${budget_usd:.2f} исчерпан, дальше не спрашиваю.")
            break
    for future in as_completed(list(futures)):
        probe_done, variant, repeat, path = futures.pop(future)
        spent, done, failed = _store_answer(future, probe_done, variant, repeat, path, spent, done, failed)
    pool.shutdown()
    if dry_run:
        click.echo(f"Картинки: {out_dir / 'vlm' / 'images'}")
    else:
        click.echo(f"Ответов: {done}, пропущено готовых: {skipped}, без ответа: {failed}, потрачено ${spent:.3f}")


def _store_answer(future, probe, variant, repeat, path: Path, spent: float, done: int, failed: int):
    try:
        answer, meta = future.result()
    except Exception as error:
        answer, meta = None, {"error": f"{type(error).__name__}: {error}", "variant": variant, "cost_usd": 0.0}
        logger.warning("%s с.%d %s r%d: %s", probe.pdf, probe.page, variant, repeat, meta["error"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "pdf": probe.pdf,
                "page": probe.page,
                "belt": probe.belt,
                "score": probe.score,
                "repeat": repeat,
                "answer": answer,
                "meta": meta,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return spent + float(meta.get("cost_usd", 0.0) or 0.0), done + (answer is not None), failed + (answer is None)


@main.command("vlm-report")
@click.option("--out-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--labels", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--md-report", type=click.Path(dir_okay=False, path_type=Path))
def vlm_report(out_dir, labels, md_report) -> None:
    """Сводка ответов VLM: согласие с эталоном и между повторами, цена."""
    from research.geometry_regression import vlm

    records = vlm.load_answers(out_dir)
    text = vlm.vlm_markdown(records, load_labels(labels) if labels else {})
    if md_report:
        md_report.parent.mkdir(parents=True, exist_ok=True)
        md_report.write_text(text, encoding="utf-8")
        click.echo(f"Отчёт: {md_report}")
    click.echo("\n".join(text.splitlines()[:12]))
