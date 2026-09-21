"""Командная строка: ``run`` — прогон по дереву полос, ``report`` — пересборка отчётов из CSV."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import click

from ocr_utils.scan_cropping.image_io import IMAGE_EXTS
from ocr_utils.scan_markup.curved_lines import CURVED_LINES_VERSION, analysis, overlay, report
from ocr_utils.scan_markup.curved_lines import labels as labelling
from ocr_utils.scan_markup.curved_lines.detectors import DEFAULT_SET, DETECTORS, registry_text, resolve
from ocr_utils.scan_markup.curved_lines.flags import ThresholdError, Thresholds
from ocr_utils.page_layout.orientation.cli import collect_images
from ocr_utils.page_layout.orientation.pdf_pages import PdfPage, load_pdf_pages, lookup

logger = logging.getLogger(__name__)

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]
OVERLAY_MODES = ["flagged", "all", "labelled", "none"]

# Процессов на CPU-этап. 16 физических ядер; несколько остаётся родителю, который в это
# время гоняет surya и её постобработку на CPU.
DEFAULT_JOBS = 12
# «Сильный» одиночный голос. 1.5 на прогоне по паку давало 761 полосу с одним голосом, и
# выборка глазами показала, что почти все они — выброс одного тайла или ленты; 2.5
# оставляет только заведомо кривые полосы, которые видит один детектор.
DEFAULT_COMBO_STRONG = 2.5


def _set_log_level(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(levelname)s %(name)s: %(message)s")


def _resolve_detectors(names: str):
    if names.strip() == "list":
        click.echo("Детекторы:\n" + registry_text())
        raise SystemExit(0)
    wanted = tuple(part.strip() for part in names.split(",") if part.strip())
    try:
        detectors = resolve(wanted)
    except KeyError as error:
        raise click.UsageError(str(error)) from error
    missing = [d.name for d in detectors if not d.available()]
    if missing:
        raise click.UsageError(f"недоступны: {', '.join(missing)}. Список с пометками: --detectors list")
    return detectors


def _thresholds(detectors, overrides: Sequence[str]) -> Thresholds:
    try:
        return Thresholds.from_detectors(detectors, overrides)
    except ThresholdError as error:
        raise click.UsageError(str(error)) from error


def _pdf_pages(db: Path | None, pack_name: str) -> dict[str, PdfPage]:
    if db is None:
        return {}
    try:
        return load_pdf_pages(db, pack_name)
    except Exception as error:
        logger.warning("Номера страниц PDF недоступны (%s), обойдёмся без них", error)
        return {}


def _attach_labels(results, labels_path: Path | None) -> None:
    if labels_path is None:
        return
    labels = labelling.load_labels(labels_path)
    for result in results:
        found = labelling.lookup(labels, result.rel_path)
        if found is not None:
            result.label = found.label


def _labelled_tasks(root: Path, labels_path: Path) -> list[tuple[Path, str]]:
    """Размеченные полосы под ``--root``: по основе имени, расширение любое."""
    tasks = []
    for key in labelling.load_labels(labels_path):
        candidates = sorted(
            p for p in (root / key).parent.glob(f"{Path(key).name}.*") if p.suffix.lower() in IMAGE_EXTS
        )
        if not candidates:
            raise click.UsageError(f"размеченной полосы {key} нет под {root}")
        tasks.append((candidates[0], candidates[0].relative_to(root).as_posix()))
    return tasks


def _common(func):
    func = click.option(
        "--detectors",
        "detector_names",
        default=",".join(DEFAULT_SET),
        show_default=True,
        help="через запятую; 'list' — реестр",
    )(func)
    func = click.option("--jobs", default=DEFAULT_JOBS, show_default=True, type=int, help="процессов на CPU-этап")(func)
    func = click.option("--source-dpi", default=600, show_default=True, type=int, help="разрешение исходника без тега")(
        func
    )
    func = click.option("--gpu-side", default=1536, show_default=True, type=int, help="длинная сторона копии на GPU")(
        func
    )
    func = click.option("--gpu-batch", default=16, show_default=True, type=int, help="полос в пачке на GPU")(func)
    func = click.option("--surya-tile", default=1100, show_default=True, type=int, help="сторона тайла surya")(func)
    func = click.option("--surya-overlap", default=300, show_default=True, type=int, help="перекрытие тайлов surya")(
        func
    )
    func = click.option("--log-level", default="INFO", show_default=True, type=click.Choice(LOG_LEVELS))(func)
    return func


def _outputs(func):
    func = click.option(
        "--link-dir", type=click.Path(file_okay=False, path_type=Path), help="каталог симлинков на находки"
    )(func)
    func = click.option(
        "--link-root",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        help="целить симлинки на это дерево (оригиналы)",
    )(func)
    func = click.option("--md-report", type=click.Path(dir_okay=False, path_type=Path))(func)
    func = click.option("--sheet", type=click.Path(dir_okay=False, path_type=Path), help="контактный лист находок")(
        func
    )
    func = click.option(
        "--db",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="база разметки — ради номеров страниц PDF",
    )(func)
    func = click.option("--pack-name", default="пак-1", show_default=True)(func)
    func = click.option("--overlay-dir", type=click.Path(file_okay=False, path_type=Path), help="отладочные оверлеи")(
        func
    )
    func = click.option(
        "--overlay", "overlay_mode", default="flagged", show_default=True, type=click.Choice(OVERLAY_MODES)
    )(func)
    func = click.option("--thr", "overrides", multiple=True, help="порог: ДЕТЕКТОР.МЕТРИКА=ЧИСЛО; можно повторять")(
        func
    )
    func = click.option("--list-thresholds", is_flag=True, help="показать пороги и выйти")(func)
    func = click.option("--combo-votes", type=int, help="сколько флагов нужно своду [по умолчанию min(2, детекторов)]")(
        func
    )
    func = click.option(
        "--combo-strong",
        default=DEFAULT_COMBO_STRONG,
        show_default=True,
        type=float,
        help="score, при котором своду хватает одного голоса",
    )(func)
    func = click.option(
        "--labels",
        "labels_path",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="CSV меток curved/straight",
    )(func)
    func = click.option("--suggest-thresholds", is_flag=True, help="напечатать таблицу разделения по меткам")(func)
    func = click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path), help="кэш измерений (SSD)")(
        func
    )
    return func


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main() -> None:
    """Поиск полос с кривыми, волнистыми или неравномерно наклонёнными строками."""


def _emit(
    results,
    detector_names,
    thresholds: Thresholds,
    votes: int,
    strong: float,
    root: Path,
    *,
    link_dir,
    link_root,
    csv_path,
    md_report,
    sheet,
    overlay_dir,
    overlay_mode,
    cache_dir,
    jobs,
    source_dpi,
    pdf_pages,
    suggest,
) -> None:
    analysis.apply_flags(results, thresholds, votes, strong)
    link_counts: dict[str, int] = {}
    if link_dir is not None:
        link_counts = report.write_link_dir(link_dir, results, detector_names, pdf_pages, link_root)
        click.echo(f"Симлинки: {link_dir}")
    if csv_path is not None:
        report.write_csv(csv_path, results, detector_names, pdf_pages)
        click.echo(f"CSV: {csv_path}")
    if sheet is not None:
        for written in report.contact_sheet(sheet, results, pdf_pages):
            click.echo(f"Контактный лист: {written}")
    if overlay_dir is not None and overlay_mode != "none":
        chosen = [
            r
            for r in results
            if overlay_mode == "all"
            or (overlay_mode == "flagged" and (r.flagged or any(m.flag for m in r.measures.values())))
            or (overlay_mode == "labelled" and r.label)
        ]
        jobs_list = []
        for ordinal, result in enumerate(chosen, start=1):
            pdf = lookup(pdf_pages, result.rel_path)
            score = result.combo.score if result.combo else 0.0
            infos = {}
            for name in detector_names:
                measure = result.measures.get(name)
                if measure is None:
                    continue
                infos[name] = {
                    "flag": measure.flag,
                    "score": measure.score,
                    "silent": measure.silent,
                    "note": measure.note,
                    "metrics": measure.metrics,
                    "keys": list(DETECTORS[name].thresholds),
                }
            jobs_list.append(
                overlay.OverlayJob(
                    path=result.path,
                    rel_path=result.rel_path,
                    out_path=overlay_dir / report.link_name(result, pdf, score, ordinal, ".jpg"),
                    default_dpi=source_dpi,
                    names=tuple(detector_names),
                    cache_root=cache_dir,
                    raws={
                        name: (result.measures[name].raw if name in result.measures else None)
                        for name in detector_names
                    },
                    infos=infos,
                )
            )
        # Старые оверлеи чистятся: имя несёт score, и после смены порогов та же полоса
        # получила бы второй файл рядом с устаревшим.
        if overlay_dir.is_dir():
            for stale in overlay_dir.glob("*_p*_s*.jpg"):
                stale.unlink()
        errors = overlay.write_many(jobs_list, jobs)
        click.echo(f"Оверлеи: {overlay_dir} ({len(jobs_list) - len(errors)} шт.)")
        for error in errors[:20]:
            click.echo(f"  {error}")
    text = report.markdown_report(
        results, detector_names, pdf_pages, root, link_dir, link_counts, thresholds, votes, strong
    )
    if md_report is not None:
        md_report.write_text(text, encoding="utf-8")
        click.echo(f"Отчёт: {md_report}")
    click.echo(report.counts_table(results, detector_names))
    if suggest:
        pairs = [(r.label, r.metrics_by_detector()) for r in results if r.label]
        if pairs:
            click.echo(labelling.separation_table(labelling.separation(pairs, thresholds.values, detector_names)))
        else:
            click.echo("Нет размеченных полос — таблицу разделения строить не по чему")


@main.command("run")
@click.option(
    "--root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="дерево полос: год/выпуск/полоса",
)
@click.option("--csv", "csv_path", type=click.Path(dir_okay=False, path_type=Path), help="все полосы и все метрики")
@click.option("--limit", type=int, help="взять только первые N полос — для пробы")
@click.option("--only", multiple=True, help="конкретная полоса (путь от --root); можно повторять")
@click.option("--labelled-only", is_flag=True, help="только размеченные полосы из --labels")
@_outputs
@_common
def run_command(
    root,
    csv_path,
    limit,
    only,
    labelled_only,
    link_dir,
    link_root,
    md_report,
    sheet,
    db,
    pack_name,
    overlay_dir,
    overlay_mode,
    overrides,
    list_thresholds,
    combo_votes,
    combo_strong,
    labels_path,
    suggest_thresholds,
    cache_dir,
    detector_names,
    jobs,
    source_dpi,
    gpu_side,
    gpu_batch,
    surya_tile,
    surya_overlap,
    log_level,
) -> None:
    """Прогнать детекторы по дереву полос и разложить находки симлинками."""
    _set_log_level(log_level)
    detectors = _resolve_detectors(detector_names)
    thresholds = _thresholds(detectors, overrides)
    if list_thresholds:
        click.echo(thresholds.table())
        return

    if only:
        tasks = [(root / rel, rel) for rel in only]
        for path, _ in tasks:
            if not path.is_file():
                raise click.UsageError(f"нет файла {path}")
    elif labelled_only:
        if labels_path is None:
            raise click.UsageError("--labelled-only требует --labels")
        tasks = _labelled_tasks(root, labels_path)
    else:
        paths = collect_images(root)
        if limit:
            paths = paths[:limit]
        tasks = [(path, path.relative_to(root).as_posix()) for path in paths]
    if not tasks:
        raise click.UsageError(f"в {root} не нашлось картинок")

    names = [d.name for d in detectors]
    votes = combo_votes if combo_votes is not None else min(2, len(detectors))
    click.echo(f"Полос: {len(tasks)}. Детекторы: {', '.join(names)} (версия {CURVED_LINES_VERSION})")
    results = analysis.analyse(
        tasks,
        detectors,
        workers=jobs,
        default_dpi=source_dpi,
        gpu_side=gpu_side,
        gpu_batch=gpu_batch,
        cache_root=cache_dir,
        keep_raw=overlay_dir is not None and cache_dir is None,
        gpu_options={"tile_side": surya_tile, "tile_overlap": surya_overlap},
    )
    _attach_labels(results, labels_path)
    _emit(
        results,
        names,
        thresholds,
        votes,
        combo_strong,
        root,
        link_dir=link_dir,
        link_root=link_root,
        csv_path=csv_path,
        md_report=md_report,
        sheet=sheet,
        overlay_dir=overlay_dir,
        overlay_mode=overlay_mode,
        cache_dir=cache_dir,
        jobs=jobs,
        source_dpi=source_dpi,
        pdf_pages=_pdf_pages(db, pack_name),
        suggest=suggest_thresholds,
    )


@main.command("report")
@click.option(
    "--csv",
    "csv_in",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="CSV прогона run",
)
@click.option(
    "--root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="дерево полос, по которому шёл run",
)
@click.option("--out-csv", type=click.Path(dir_okay=False, path_type=Path), help="переписать CSV с новыми флагами")
@_outputs
@_common
def report_command(
    csv_in,
    root,
    out_csv,
    link_dir,
    link_root,
    md_report,
    sheet,
    db,
    pack_name,
    overlay_dir,
    overlay_mode,
    overrides,
    list_thresholds,
    combo_votes,
    combo_strong,
    labels_path,
    suggest_thresholds,
    cache_dir,
    detector_names,
    jobs,
    source_dpi,
    gpu_side,
    gpu_batch,
    surya_tile,
    surya_overlap,
    log_level,
) -> None:
    """Пересобрать флаги, симлинки и отчёты из CSV прогона — с другими порогами, без пересчёта."""
    _set_log_level(log_level)
    wanted = None if detector_names == ",".join(DEFAULT_SET) else [d.name for d in _resolve_detectors(detector_names)]
    results, names = report.read_csv(csv_in, root, wanted)
    if not names:
        raise click.UsageError("в CSV нет колонок детекторов (*_flag)")
    detectors = [DETECTORS[name] for name in names]
    thresholds = _thresholds(detectors, overrides)
    if list_thresholds:
        click.echo(thresholds.table())
        return
    _attach_labels(results, labels_path)
    votes = combo_votes if combo_votes is not None else min(2, len(detectors))
    click.echo(f"Полос из CSV: {len(results)}. Детекторы: {', '.join(names)}")
    _emit(
        results,
        names,
        thresholds,
        votes,
        combo_strong,
        root,
        link_dir=link_dir,
        link_root=link_root,
        csv_path=out_csv,
        md_report=md_report,
        sheet=sheet,
        overlay_dir=overlay_dir,
        overlay_mode=overlay_mode,
        cache_dir=cache_dir,
        jobs=jobs,
        source_dpi=source_dpi,
        pdf_pages=_pdf_pages(db, pack_name),
        suggest=suggest_thresholds,
    )


if __name__ == "__main__":
    main()
