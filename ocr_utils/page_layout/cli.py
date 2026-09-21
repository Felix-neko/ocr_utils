"""CLI разбора структуры страницы: кэш surya (импорт старого, проверка) и разбор одной страницы."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from ocr_utils.page_layout.image import Variant
from ocr_utils.page_layout.surya.cache import SuryaCache, import_legacy

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
VARIANTS = tuple(v.value for v in Variant)


def _set_log_level(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@click.group()
def main() -> None:
    """Разбор структуры страницы: растр, таблицы, line art, повёрнутый текст, ориентация, surya."""


@main.command("import-legacy-cache")
@click.option(
    "--src", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path), help="Старый кэш pickle."
)
@click.option("--dst", required=True, type=click.Path(file_okay=False, path_type=Path), help="Корень нового кэша.")
@click.option("--variant", required=True, type=click.Choice(VARIANTS), help="Под каким вариантом класть.")
@click.option(
    "--overwrite/--no-overwrite", default=False, show_default=True, help="Перезаписывать существующие записи."
)
@click.option("--log-level", default="INFO", show_default=True, type=click.Choice(LOG_LEVELS, case_sensitive=False))
def import_legacy_command(src: Path, dst: Path, variant: str, overwrite: bool, log_level: str) -> None:
    """Перенести старый кэш surya (pickle по rel_path) в новый формат записями legacy."""
    _set_log_level(log_level)
    done, skipped, broken = import_legacy(src, Variant(variant), SuryaCache(dst), overwrite=overwrite)
    click.echo(f"Перенесено {done}, уже было {skipped}, битых {broken} -> {dst / variant}")


@main.command("verify-cache")
@click.option("--cache", "cache_root", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--variant", required=True, type=click.Choice(VARIANTS))
@click.option("--limit", default=None, type=int, help="Проверить только первые N записей.")
@click.option("--log-level", default="INFO", show_default=True, type=click.Choice(LOG_LEVELS, case_sensitive=False))
def verify_cache_command(cache_root: Path, variant: str, limit: int | None, log_level: str) -> None:
    """Прочитать записи кэша и посчитать битые, legacy и без источника — без обращения к картинкам."""
    _set_log_level(log_level)
    cache = SuryaCache(cache_root, readonly=True)
    total = broken = legacy = no_source = 0
    for name in cache.iter_names(Variant(variant)):
        if limit is not None and total >= limit:
            break
        total += 1
        entry = cache.read(Variant(variant), name)
        if entry is None:
            broken += 1
            continue
        legacy += entry.legacy
        no_source += entry.source is None and not entry.legacy
    click.echo(f"Записей {total}: битых {broken}, legacy {legacy}, без отпечатка источника {no_source}.")


# Цвета рамок на оверлее (BGR) — те же, что у меток CVAT, чтобы читались одинаково.
OVERLAY_COLORS = {
    "color": (0, 230, 118),
    "grayscale": (255, 176, 0),
    "stamp_suspect": (0, 109, 255),
    "color_text": (98, 17, 197),
    "table": (254, 79, 48),
    "line_art_schema": (65, 76, 109),
    "rotated_text": (92, 105, 0),
}
OVERLAY_SIDE = 1400


def _source_and_model(cache_root: Path | None, use_surya: bool, readonly: bool):
    """Источник surya для однопроцессного разбора: кэш + модель (в родителе) либо ничего."""
    from ocr_utils.page_layout.surya.model import SuryaLayoutModel
    from ocr_utils.page_layout.surya.source import SuryaSource

    if not use_surya:
        return None
    cache = SuryaCache(cache_root, readonly=readonly) if cache_root is not None else None
    return SuryaSource(cache, None if readonly else SuryaLayoutModel())


def _draw_overlay(image, layout, out: Path) -> None:
    """Картинка с рамками всех областей и блоков surya (тонкой серой линией)."""
    import cv2

    from ocr_utils.page_layout import WORK_DPI

    work = image.bgr_at(WORK_DPI).copy()
    k = WORK_DPI / image.dpi
    if layout.raw_surya_content is not None:
        for block in layout.raw_surya_content.blocks:
            b = block.box.scaled(k)
            cv2.rectangle(work, (b.x0, b.y0), (b.x1, b.y1), (160, 160, 160), 1)
            cv2.putText(work, block.label, (b.x0 + 2, b.y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    for region in layout.regions:
        b = region.box.scaled(k)
        color = OVERLAY_COLORS.get(region.kind.value, (255, 255, 255))
        cv2.rectangle(work, (b.x0, b.y0), (b.x1, b.y1), color, 3)
        label = region.kind.value + (f" {region.confidence:.2f}" if region.confidence is not None else "")
        cv2.putText(work, label, (b.x0 + 4, max(14, b.y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    scale = min(1.0, OVERLAY_SIDE / max(work.shape[:2]))
    if scale < 1.0:
        work = cv2.resize(work, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), work, [cv2.IMWRITE_JPEG_QUALITY, 85])


@main.command("analyze")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--page", default=0, show_default=True, type=int, help="Номер страницы PDF с нуля.")
@click.option("--variant", default="scan", show_default=True, type=click.Choice(VARIANTS))
@click.option("--dpi", default=None, type=int, help="Разрешение, если у файла нет тега (или родное для PDF).")
@click.option("--cache", "cache_root", default=None, type=click.Path(file_okay=False, path_type=Path))
@click.option("--cache-name", default=None, help="Имя в кэше; по умолчанию — имя файла (для PDF — <stem>/pNNNN).")
@click.option("--surya/--no-surya", default=True, show_default=True)
@click.option("--cache-only", is_flag=True, help="Не грузить модель: промах кэша — ошибка.")
@click.option(
    "--find", "finds", default="", help="Что искать через запятую: raster,tables,line_art,rotated_text,orientation."
)
@click.option("--orientation-detectors", default="ink_axis", show_default=True)
@click.option("--overlay", default=None, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--json", "json_out", default=None, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--log-level", default="INFO", show_default=True, type=click.Choice(LOG_LEVELS, case_sensitive=False))
def analyze_command(
    input_path: Path,
    page: int,
    variant: str,
    dpi: int | None,
    cache_root: Path | None,
    cache_name: str | None,
    surya: bool,
    cache_only: bool,
    finds: str,
    orientation_detectors: str,
    overlay: Path | None,
    json_out: Path | None,
    log_level: str,
) -> None:
    """Разобрать одну страницу (картинку или страницу PDF) и показать области."""
    import json

    from ocr_utils.page_layout.analysis import Find, LayoutOptions, PageLayout
    from ocr_utils.page_layout.image import PageImage

    _set_log_level(log_level)
    kind = Variant(variant)
    if input_path.suffix.lower() == ".pdf":
        import fitz

        document = fitz.open(str(input_path))
        image = PageImage.from_pdf_page(document, page, kind, cache_name, native_dpi=dpi)
    else:
        image = PageImage.from_file(input_path, kind, cache_name or input_path.stem, default_dpi=dpi)
    wanted = {Find(part.strip()) for part in finds.split(",") if part.strip()} or None
    detectors = tuple(part.strip() for part in orientation_detectors.split(",") if part.strip())
    options = LayoutOptions(use_surya=surya, orientation_detectors=detectors)
    layout = PageLayout(image, wanted, options).process(_source_and_model(cache_root, surya, cache_only))

    click.echo(
        f"{input_path.name}: {image.width}x{image.height} @ {image.dpi} dpi, surya: {'да' if layout.surya_used else 'нет'}"
    )
    for region in layout.regions:
        conf = f" {region.confidence:.2f}" if region.confidence is not None else ""
        click.echo(f"  {region.kind.value:16s}{conf:6s} {region.box.as_tuple()}  {region.info.get('kind', '')}")
    if layout.best_page_orientation is not None:
        verdict = layout.best_page_orientation
        click.echo(f"  ориентация: повернуть на {verdict.rotate_cw} (уверенность {verdict.confidence:.2f})")
    if overlay is not None:
        _draw_overlay(image, layout, overlay)
        click.echo(f"Оверлей: {overlay}")
    if json_out is not None:
        payload = {
            "width": image.width,
            "height": image.height,
            "dpi": image.dpi,
            "surya_used": layout.surya_used,
            "regions": [
                {"kind": r.kind.value, "box": list(r.box.as_tuple()), "confidence": r.confidence, "info": r.info}
                for r in layout.regions
            ],
            "orientation": (
                None
                if layout.best_page_orientation is None
                else {
                    "rotate_cw": layout.best_page_orientation.rotate_cw,
                    "confidence": layout.best_page_orientation.confidence,
                }
            ),
        }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


@main.command("prefill-surya")
@click.option("--cache", "cache_root", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--variant", required=True, type=click.Choice(VARIANTS))
@click.option("--pdf-dir", "pdf_dirs", multiple=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--scan-dir", default=None, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--default-dpi", default=None, type=int, help="Разрешение картинок без тега.")
@click.option("--jobs", default=16, show_default=True, type=int, help="Воркеров на декодирование/рендер.")
@click.option("--chunk", default=64, show_default=True, type=int, help="Кадров впереди модели.")
@click.option("--log-level", default="INFO", show_default=True, type=click.Choice(LOG_LEVELS, case_sensitive=False))
def prefill_command(
    cache_root: Path,
    variant: str,
    pdf_dirs: tuple[Path, ...],
    scan_dir: Path | None,
    default_dpi: int | None,
    jobs: int,
    chunk: int,
    log_level: str,
) -> None:
    """Набить кэш surya по всем страницам PDF из папок или по всем картинкам под корнем (GPU в родителе)."""
    from ocr_utils.page_layout.prefill import pdf_requests, prefill, scan_requests
    from ocr_utils.page_layout.surya.model import SuryaLayoutModel

    _set_log_level(log_level)
    kind = Variant(variant)
    requests = []
    for pdf_dir in pdf_dirs:
        requests.extend(pdf_requests(sorted(Path(pdf_dir).glob("*.pdf")), kind))
    if scan_dir is not None:
        requests.extend(scan_requests(scan_dir, kind, default_dpi))
    if not requests:
        raise click.ClickException("нечего набивать: укажите --pdf-dir и/или --scan-dir")
    stats = prefill(requests, SuryaCache(cache_root), SuryaLayoutModel(), jobs=jobs, chunk=chunk)
    click.echo(
        f"Страниц {stats.requested}: уже в кэше {stats.cached}, размечено {stats.done}, не удалось {stats.failed} "
        f"-> {cache_root / variant}"
    )
