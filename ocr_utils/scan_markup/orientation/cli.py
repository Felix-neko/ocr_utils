"""Командная строка подпакета: ``run`` — прогон по дереву, ``validate`` — проверка."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click

from ocr_utils.scan_cropping.image_io import IMAGE_EXTS
from ocr_utils.scan_markup.orientation import ORIENTATION_VERSION
from ocr_utils.scan_markup.orientation import analysis, report, validation
from ocr_utils.scan_markup.orientation.detectors import (
    CHOICES,
    DEFAULT_SET,
    DETECTORS,
    ROTATIONS,
    registry_text,
    resolve,
)
from ocr_utils.scan_markup.orientation.pdf_pages import PdfPage, load_pdf_pages
from ocr_utils.scan_markup.scan_tree import is_ignored_dir

logger = logging.getLogger(__name__)

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]

# Умолчание по числу процессов. В машине 16 физических ядер; два оставлены под родителя,
# который тем временем гоняет GPU-детекторы и пишет отчёты (CLAUDE.md).
DEFAULT_JOBS = 14


def _set_log_level(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(levelname)s %(name)s: %(message)s")


def collect_images(root: Path) -> list[Path]:
    """Все картинки дерева, кроме служебных каталогов ScanTailor и мусора Windows."""
    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        if any(is_ignored_dir(parent) for parent in path.parents):
            continue
        found.append(path)
    return found


def _resolve_detectors(names: str):
    if names.strip() == "list":
        click.echo("Детекторы:\n" + registry_text())
        raise SystemExit(0)
    wanted = tuple(part.strip() for part in names.split(",") if part.strip())
    try:
        detectors = resolve(wanted)
    except KeyError as error:
        raise click.UsageError(str(error)) from error
    missing = [detector.name for detector in detectors if not detector.available()]
    if missing:
        raise click.UsageError(f"недоступны: {', '.join(missing)}. Список с пометками: --detectors list")
    return detectors


def _pdf_pages(db: Path | None, pack_name: str) -> dict[str, PdfPage]:
    if db is None:
        return {}
    try:
        return load_pdf_pages(db, pack_name)
    except Exception as error:
        # Номера страниц PDF — удобство, а не условие работы: без базы отчёт по-прежнему
        # называет полосу путём файла, и прогон не должен из-за этого падать.
        logger.warning("Номера страниц PDF недоступны (%s), обойдёмся без них", error)
        return {}


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main() -> None:
    """Поиск полос, напечатанных боком или вверх ногами."""


def _common(func):
    func = click.option(
        "--detectors",
        "detector_names",
        default=",".join(DEFAULT_SET),
        show_default=True,
        help="через запятую; 'list' — показать реестр",
    )(func)
    func = click.option("--jobs", default=DEFAULT_JOBS, show_default=True, type=int, help="процессов на CPU-этап")(func)
    func = click.option(
        "--source-dpi", default=600, show_default=True, type=int, help="разрешение исходника, если в файле нет тега"
    )(func)
    func = click.option(
        "--gpu-side", default=1536, show_default=True, type=int, help="длинная сторона копии, уезжающей на GPU"
    )(func)
    func = click.option("--gpu-batch", default=32, show_default=True, type=int, help="полос в пачке на GPU")(func)
    func = click.option(
        "--angles",
        default="0,90,180,270",
        show_default=True,
        help="какие повороты вообще рассматривать, через запятую; сужать полезно — "
        "невозможный ответ не будет дан, и арбитру меньше работы",
    )(func)
    func = click.option("--log-level", default="INFO", show_default=True, type=click.Choice(LOG_LEVELS))(func)
    return func


def _parse_angles(raw: str) -> tuple[int, ...]:
    try:
        angles = tuple(sorted({int(part.strip()) % 360 for part in raw.split(",") if part.strip()}))
    except ValueError as error:
        raise click.UsageError(f"--angles: не разобрать {raw!r}") from error
    bad = [a for a in angles if a not in ROTATIONS]
    if bad or not angles:
        raise click.UsageError(f"--angles: бывают только {ROTATIONS}, получено {raw!r}")
    if 0 not in angles:
        raise click.UsageError("--angles: 0 обязан быть в наборе, иначе полосу без поворота некуда отнести")
    return angles


@main.command("run")
@click.option(
    "--root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="дерево полос: год/выпуск/полоса",
)
@click.option(
    "--link-dir", type=click.Path(file_okay=False, path_type=Path), help="каталог симлинков на найденные полосы"
)
@click.option(
    "--link-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="на какое дерево целить симлинки, если не на --root",
)
@click.option("--csv", "csv_path", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--md-report", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--sheet",
    type=click.Path(dir_okay=False, path_type=Path),
    help="контактный лист миниатюр находок, уже повёрнутых на предложенный угол",
)
@click.option(
    "--db",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="база разметки — ради номеров страниц промежуточного PDF",
)
@click.option("--pack-name", default="пак-1", show_default=True)
@click.option("--limit", type=int, help="взять только первые N полос — для пробы")
@click.option("--only", multiple=True, help="конкретная полоса (путь от --root); можно повторять")
@click.option("--arbiter/--no-arbiter", default=True, show_default=True, help="второй проход арбитром по кандидатам")
@_common
def run_command(
    root,
    link_dir,
    link_root,
    csv_path,
    md_report,
    sheet,
    db,
    pack_name,
    limit,
    only,
    arbiter,
    detector_names,
    jobs,
    source_dpi,
    gpu_side,
    gpu_batch,
    angles,
    log_level,
) -> None:
    """Прогнать детекторы по дереву полос и разложить находки симлинками."""
    _set_log_level(log_level)
    allowed = _parse_angles(angles)
    detectors = _resolve_detectors(detector_names)
    fast = [detector for detector in detectors if not detector.arbiter]
    arbiters = [detector for detector in detectors if detector.arbiter]
    if not fast:
        raise click.UsageError("нужен хотя бы один не-арбитр: арбитру нечего будет проверять")

    if only:
        tasks = [(root / rel, rel) for rel in only]
        for path, rel in tasks:
            if not path.is_file():
                raise click.UsageError(f"нет файла {path}")
    else:
        paths = collect_images(root)
        if limit:
            paths = paths[:limit]
        tasks = [(path, path.relative_to(root).as_posix()) for path in paths]
    if not tasks:
        raise click.UsageError(f"в {root} не нашлось картинок")

    click.echo(
        f"Полос: {len(tasks)}. Детекторы: {', '.join(d.name for d in detectors)} "
        f"(версия {ORIENTATION_VERSION}). Допустимые повороты: {', '.join(map(str, allowed))}"
    )
    results = analysis.analyse(
        tasks, fast, workers=jobs, default_dpi=source_dpi, gpu_side=gpu_side, gpu_batch=gpu_batch, desc="быстрые"
    )
    analysis.apply_combo(results, allowed)

    # Кандидатов отмечаем всегда, а не только когда есть кому их проверять: по этой отметке
    # раскладывается каталог «кандидаты», и без арбитра он тем более нужен — там будут все,
    # кого отметили быстрые детекторы, и отвергать их будет некому.
    chosen = analysis.candidates(results)
    click.echo(f"Кандидатов: {len(chosen)}")

    if arbiter and arbiters:
        if chosen:
            checked = analysis.analyse(
                [(r.path, r.rel_path) for r in chosen],
                arbiters,
                workers=jobs,
                default_dpi=source_dpi,
                gpu_side=gpu_side,
                gpu_batch=gpu_batch,
                desc="арбитр",
                allowed=allowed,
            )
            by_path = {result.rel_path: result for result in checked}
            for result in chosen:
                verdicts = by_path[result.rel_path].verdicts
                result.verdicts.update(verdicts)
                result.seconds.update(by_path[result.rel_path].seconds)
            analysis.apply_combo(results, allowed)

    names = [detector.name for detector in detectors]
    pdf_pages = _pdf_pages(db, pack_name)
    link_counts: dict[str, int] = {}
    if link_dir is not None:
        link_counts = report.write_link_dir(link_dir, results, names, pdf_pages, link_root)
        click.echo(f"Симлинки: {link_dir}")
    if csv_path is not None:
        report.write_csv(csv_path, results, names, pdf_pages)
    if sheet is not None:
        for written in report.contact_sheet(sheet, results, pdf_pages):
            click.echo(f"Контактный лист: {written}")
    text = report.markdown_report(
        results, names, pdf_pages, root, link_dir, link_counts, arbiters=[d.name for d in arbiters]
    )
    if md_report is not None:
        md_report.write_text(text, encoding="utf-8")
        click.echo(f"Отчёт: {md_report}")
    click.echo(report.counts_table(results, names))


@main.command("validate")
@click.option("--root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--sample", default=200, show_default=True, type=int, help="сколько прямых полос взять в эталон")
@click.option("--seed", default=0, show_default=True, type=int)
@click.option("--md-report", type=click.Path(dir_okay=False, path_type=Path))
@_common
def validate_command(
    root, sample, seed, md_report, detector_names, jobs, source_dpi, gpu_side, gpu_batch, angles, log_level
) -> None:
    """Проверить детекторов на синтетических поворотах заведомо прямых полос."""
    _set_log_level(log_level)
    allowed = _parse_angles(angles)
    detectors = _resolve_detectors(detector_names)

    paths = collect_images(root)
    if not paths:
        raise click.UsageError(f"в {root} не нашлось картинок")
    # Кандидатов берём с небольшим запасом: полоса пака прямая почти всегда, отсеивается
    # единицы процентов, а отбор идёт арбитром — самой дорогой частью прогона. Брать втрое
    # больше, чем нужно, значило бы втрое переплатить за этап, который ничего не сравнивает.
    tasks = validation.sample_paths(paths, root, max(sample + 50, int(sample * 1.25)), seed)
    click.echo(f"Отбор эталона из {len(tasks)} случайных полос")
    scouted = analysis.analyse(
        tasks, detectors, workers=jobs, default_dpi=source_dpi, gpu_side=gpu_side, gpu_batch=gpu_batch, desc="отбор"
    )
    arbiters = [detector.name for detector in detectors if detector.arbiter]
    reference = validation.pick_reference(scouted, sample, arbiters[0] if arbiters else None)
    if not reference:
        raise click.UsageError("не нашлось полос, про которые все согласны, что они прямые")
    click.echo(f"Эталон: {len(reference)} полос")

    trials = validation.run_trials(
        reference, detectors, workers=jobs, default_dpi=source_dpi, gpu_side=gpu_side, gpu_batch=gpu_batch
    )
    names = [detector.name for detector in detectors]
    text = report.validation_report(trials, names, root, len(reference), arbiters[0] if arbiters else None)
    if md_report is not None:
        md_report.write_text(text, encoding="utf-8")
        click.echo(f"Отчёт: {md_report}")
    click.echo(text)


if __name__ == "__main__":
    main()
