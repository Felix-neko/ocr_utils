"""CLI подпакета: разбор отобранных страниц пары PDF с оверлеями и сводкой."""

from __future__ import annotations

from pathlib import Path

import click

from ocr_utils.curved_layout import WORK_DPI
from ocr_utils.curved_layout.blocks import COARSE_FACTOR, SMOOTH_PITCHES
from ocr_utils.curved_layout.lines import SMOOTH_HEIGHTS
from ocr_utils.curved_layout.page import Variant, analyse_gray, render_page
from ocr_utils.curved_layout.report import markdown, write_csv, write_json

# Движок по умолчанию — свой, по краске; остальные подключаются адаптерами.
ENGINE_CHOICES = ("ink", "surya", "kraken", "pero", "eynollah")


def parse_pages(text: str) -> list[tuple[str, int]]:
    """Разбор списка страниц ``full_1973_06:65,full_1971_10:87`` в пары ``(pdf, страница)``."""
    out: list[tuple[str, int]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, page = item.partition(":")
        out.append((name, int(page)))
    return out


def make_engine(name: str, options: dict):
    """Создать поставщика строк по имени движка."""
    if name == "ink":
        from ocr_utils.curved_layout.engines.ink import InkEngine

        return InkEngine()
    if name == "kraken":
        from ocr_utils.curved_layout.engines.kraken import KrakenEngine

        return KrakenEngine(python=options["kraken_python"])
    if name == "pero":
        from ocr_utils.curved_layout.engines.pero import PeroEngine

        return PeroEngine(python=options["pero_python"], config=options["pero_config"])
    if name == "eynollah":
        from ocr_utils.curved_layout.engines.eynollah import EynollahEngine

        return EynollahEngine(python=options["eynollah_python"], models=options["eynollah_models"])
    if name == "surya":
        from ocr_utils.curved_layout.engines.surya_cache import SuryaEngine

        return SuryaEngine(cache_dir=options["surya_cache"])
    raise click.BadParameter(f"неизвестный движок: {name}")


@click.group()
def main() -> None:
    """Разбор текста страницы с кривыми строками: оси строк и огибающие блоков."""


@main.command()
@click.option("--geo-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--nogeo-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--pages", required=True, help="список ``pdf:страница`` через запятую")
@click.option("--variant", type=click.Choice(["geo", "nogeo", "both"]), default="both", show_default=True)
@click.option("--engine", "engines", multiple=True, default=("ink",), type=click.Choice(ENGINE_CHOICES))
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--smooth-line", default=SMOOTH_HEIGHTS, show_default=True, type=float, help="окно оси строки, высот")
@click.option(
    "--smooth-block", default=SMOOTH_PITCHES, show_default=True, type=float, help="окно огибающей, шагов строк"
)
@click.option("--coarse-factor", default=COARSE_FACTOR, show_default=True, type=float)
@click.option("--dpi", default=WORK_DPI, show_default=True, type=float, help="разрешение рабочей копии")
@click.option("--overlay-width", default=1400, show_default=True, type=int)
@click.option("--kraken-python", type=click.Path(path_type=Path), default=None, help="python окружения kraken")
@click.option("--pero-python", type=click.Path(path_type=Path), default=None)
@click.option("--pero-config", type=click.Path(path_type=Path), default=None, help="config.ini модели pero")
@click.option("--eynollah-python", type=click.Path(path_type=Path), default=None)
@click.option("--eynollah-models", type=click.Path(path_type=Path), default=None)
@click.option("--surya-cache", type=click.Path(path_type=Path), default=None)
def analyze(
    geo_dir: Path,
    nogeo_dir: Path,
    pages: str,
    variant: str,
    engines: tuple[str, ...],
    out_dir: Path,
    smooth_line: float,
    smooth_block: float,
    coarse_factor: float,
    dpi: float,
    overlay_width: int,
    **options,
) -> None:
    """Разобрать отобранные страницы и выложить JSON, CSV, оверлеи и сводку."""
    from ocr_utils.curved_layout import overlay

    variants = [Variant.GEO.value, Variant.NOGEO.value] if variant == "both" else [variant]
    dirs = {Variant.GEO.value: geo_dir, Variant.NOGEO.value: nogeo_dir}
    analyses = []
    for engine_name in engines:
        engine = make_engine(engine_name, options)
        for name, page in parse_pages(pages):
            for current in variants:
                pdf = dirs[current] / f"{name}.pdf"
                if not pdf.is_file():
                    click.echo(f"нет файла: {pdf}")
                    continue
                gray300 = render_page(pdf, page)
                analysis = analyse_gray(
                    gray300,
                    engine,
                    dpi=dpi,
                    smooth_line=smooth_line,
                    smooth_block=smooth_block,
                    coarse_factor=coarse_factor,
                    name=name,
                    page=page,
                    variant=current,
                )
                write_json(analysis, out_dir / "pages")
                overlay.write(analysis, gray300, out_dir / "overlays" / f"{analysis.key}.jpg", overlay_width)
                analyses.append(analysis)
                click.echo(
                    f"{analysis.key}: блоков {len(analysis.blocks)}, строк {len(analysis.axes)}, "
                    f"{analysis.seconds:.1f} с"
                )
    if analyses:
        write_csv(analyses, out_dir / "blocks.csv")
        (out_dir / "report.md").write_text(markdown(analyses), encoding="utf-8")
        click.echo(f"готово: {out_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
