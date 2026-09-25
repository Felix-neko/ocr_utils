"""CLI подпакета: разбор отобранных страниц пары PDF с оверлеями и сводкой."""

from __future__ import annotations

from pathlib import Path

import click

from ocr_utils.page_layout.text_blocks import LINKING_CHOICES, LINKING_DEFAULT, RENDER_DPI, WORK_DPI
from ocr_utils.page_layout.text_blocks.blocks import COARSE_FACTOR, DILATE_GLYPHS, SMOOTH_PITCHES
from ocr_utils.page_layout.text_blocks.lines import SMOOTH_HEIGHTS
from ocr_utils.page_layout.text_blocks.page import Variant, analyse_gray, render_page
from ocr_utils.page_layout.text_blocks.report import markdown, write_csv, write_json
from ocr_utils.page_layout.text_blocks.sides import DEFAULT_SIDES_METHOD, SidesMethod

from ocr_utils.page_layout.text_blocks.engines.catalog import WorkerEngineName

# Движок по умолчанию — свой, по краске; остальные подключаются адаптерами: у kraken, pero и eynollah
# свои, у движков каталога (surya, orli, paddle …) — общий адаптер с единым JSON воркера.
ENGINE_CHOICES = ("ink", "kraken", "pero", "eynollah", *(item.value for item in WorkerEngineName))


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
        from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine

        return InkEngine(linking=options.get("linking", LINKING_DEFAULT), hints=options.get("hints"))
    if name == "kraken":
        from ocr_utils.page_layout.text_blocks.engines.kraken import KrakenEngine

        return KrakenEngine(python=options.get("kraken_python"))
    if name == "pero":
        from ocr_utils.page_layout.text_blocks.engines.pero import PeroEngine

        return PeroEngine(python=options.get("pero_python"), config=options.get("pero_config"))
    if name == "eynollah":
        from ocr_utils.page_layout.text_blocks.engines.eynollah import EynollahEngine

        return EynollahEngine(python=options.get("eynollah_python"), models=options.get("eynollah_models"))
    if name in {item.value for item in WorkerEngineName}:
        from ocr_utils.page_layout.text_blocks.engines.catalog import spec_of
        from ocr_utils.page_layout.text_blocks.engines.generic import WorkerEngine

        # Свой интерпретатор движка каталога — из ``--engine-python имя=путь``.
        pythons = dict(item.split("=", 1) for item in options.get("engine_python") or ())
        return WorkerEngine(spec_of(WorkerEngineName(name), pythons.get(name)))
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
@click.option(
    "--linking",
    type=click.Choice(LINKING_CHOICES),
    default=LINKING_DEFAULT,
    show_default=True,
    help="способ сцепки кусков в строки: zones — по зонам поиска, greedy — прежняя жадная цепочка",
)
@click.option(
    "--dilate-glyphs",
    default=DILATE_GLYPHS,
    show_default=True,
    type=float,
    help="на какую долю размера символа раздувать границу блока",
)
@click.option(
    "--dilate-compare",
    default="",
    help="доли размера символа через запятую: их границы рисуются в оверлее для сравнения",
)
@click.option(
    "--layout-cache",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="кэш surya page_layout: включает подсказки — растр, таблицы, схемы, боковой текст",
)
@click.option(
    "--forbid-figures",
    is_flag=True,
    default=False,
    help="запретить текст ВНУТРИ таблиц и line art, а не только резать по их рёбрам",
)
@click.option(
    "--text-layer-cache",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="кэш text_layer_fix (<прогон>/cache): ячейки таблиц и поворот текста в них",
)
@click.option("--kraken-python", type=click.Path(path_type=Path), default=None, help="python окружения kraken")
@click.option("--pero-python", type=click.Path(path_type=Path), default=None)
@click.option("--pero-config", type=click.Path(path_type=Path), default=None, help="config.ini модели pero")
@click.option("--eynollah-python", type=click.Path(path_type=Path), default=None)
@click.option("--eynollah-models", type=click.Path(path_type=Path), default=None)
@click.option(
    "--engine-python",
    multiple=True,
    help="свой python движка каталога: ``имя=путь`` (по умолчанию — окружение в LINE_ENGINES_ROOT)",
)
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
    dilate_glyphs: float,
    dilate_compare: str,
    layout_cache: Path | None,
    text_layer_cache: Path | None,
    forbid_figures: bool,
    **options,
) -> None:
    """Разобрать отобранные страницы и выложить JSON, CSV, оверлеи и сводку."""
    from ocr_utils.page_layout.text_blocks import overlay

    extra = tuple(float(item) for item in dilate_compare.split(",") if item.strip())
    variants = [Variant.GEO.value, Variant.NOGEO.value] if variant == "both" else [variant]
    dirs = {Variant.GEO.value: geo_dir, Variant.NOGEO.value: nogeo_dir}
    analyses = []
    surya = None
    if layout_cache is not None:
        from ocr_utils.page_layout.surya import SuryaSourceConfig

        surya = SuryaSourceConfig(layout_cache).open()
    for engine_name in engines:
        for name, page in parse_pages(pages):
            for current in variants:
                pdf = dirs[current] / f"{name}.pdf"
                if not pdf.is_file():
                    click.echo(f"нет файла: {pdf}")
                    continue
                gray300 = render_page(pdf, page)
                hints = _page_hints(pdf, page, gray300, dpi, surya, text_layer_cache, current, forbid_figures)
                engine = make_engine(engine_name, {**options, "hints": hints})
                analysis = analyse_gray(
                    gray300,
                    engine,
                    dpi=dpi,
                    smooth_line=smooth_line,
                    smooth_block=smooth_block,
                    coarse_factor=coarse_factor,
                    dilate=dilate_glyphs,
                    name=name,
                    page=page,
                    variant=current,
                    hints=hints,
                )
                write_json(analysis, out_dir / "pages")
                overlay.write(
                    analysis, gray300, out_dir / "overlays" / f"{analysis.key}.jpg", overlay_width, extra, hints
                )
                analyses.append(analysis)
                click.echo(
                    f"{analysis.key}: блоков {len(analysis.blocks)}, строк {len(analysis.axes)}, "
                    f"{analysis.seconds:.1f} с"
                )
    if analyses:
        write_csv(analyses, out_dir / "blocks.csv")
        (out_dir / "report.md").write_text(markdown(analyses), encoding="utf-8")
        click.echo(f"готово: {out_dir}")


def _page_hints(
    pdf: Path,
    page: int,
    gray300,
    dpi: float,
    surya,
    text_layer_cache: Path | None = None,
    variant=None,
    forbid_figures: bool = False,
):
    """Подсказки страницы: находки ``page_layout`` плюс ячейки таблиц из кэша ``text_layer_fix``.

    Координаты нужны в рабочей копии разбора, поэтому её размер передаётся явно: кадр
    ``page_layout`` считается в своём разрешении и округляется иначе.

    Args:
        pdf: Файл PDF разбираемого прогона FineReader.
        page: Номер страницы С ЕДИНИЦЫ.
        gray300: Серый рендер страницы.
        dpi: Разрешение рабочей копии.
        surya: ``SuryaSource`` или ``None``.
        text_layer_cache: Корень кэша ``text_layer_fix`` или ``None``.
        variant: Какой вариант разбирается, ``geo`` или ``nogeo``: от него зависит и ключ кэша
            surya, и параметры рендера ``page_layout``.
        forbid_figures: Запрещать ли текст внутри таблиц и line art.

    Returns:
        :class:`hints.LayoutHints` или ``None``, если не задан ни один кэш.
    """
    if surya is None and text_layer_cache is None:
        return None
    import cv2
    import fitz

    from ocr_utils.page_layout.text_blocks.from_layout import hints_of as layout_hints
    from ocr_utils.page_layout.text_blocks.from_text_layer import hints_of as cell_hints
    from ocr_utils.page_layout.image import Variant as PageVariant

    width = int(round(gray300.shape[1] * dpi / RENDER_DPI))
    height = int(round(gray300.shape[0] * dpi / RENDER_DPI))
    kind = PageVariant.FR_GEO if variant == Variant.GEO.value else PageVariant.FR_NOGEO
    hints = None
    if surya is not None:
        with fitz.open(pdf) as document:
            hints = layout_hints(
                document,
                page - 1,
                dpi=dpi,
                surya=surya,
                width=width,
                height=height,
                variant=kind,
                forbid_figures=forbid_figures,
            )
    if text_layer_cache is not None:
        work = cv2.resize(gray300, (width, height), interpolation=cv2.INTER_AREA)
        hints = cell_hints(pdf.name, page - 1, text_layer_cache, work, dpi=dpi, base=hints)
    return hints


@main.command()
@click.option("--pdf", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--page", type=int, required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--variant", default="nogeo", show_default=True, help="подпись варианта рендера")
@click.option("--dpi", default=WORK_DPI, show_default=True, type=float, help="разрешение рабочей копии")
@click.option(
    "--crop",
    default=None,
    help="вырезка ``x0,y0,x1,y1`` в пикселях рабочей копии; по умолчанию — вокруг самой непрямой оси",
)
@click.option("--linking", type=click.Choice(LINKING_CHOICES), default=LINKING_DEFAULT, show_default=True)
def stages(pdf: Path, page: int, out_dir: Path, variant: str, dpi: float, crop: str | None, linking: str) -> None:
    """Отладочные оверлеи по ЭТАПАМ построения осевой линии для одной полосы."""
    from ocr_utils.page_layout.text_blocks import stages as stages_module

    gray300 = render_page(pdf, page)
    box = tuple(int(value) for value in crop.split(",")) if crop else None
    pictures = stages_module.render(gray300, out_dir, pdf.stem, page, variant, dpi, box, linking)
    for picture in pictures:
        click.echo(f"{picture.path.name}: {picture.title}" + (f" — {picture.note}" if picture.note else ""))
    click.echo(f"готово: {out_dir}")


@main.command()
@click.option("--out-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option(
    "--against",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="выкладка другого прогона: тогда печатаются только итоги двух рядом",
)
@click.option("--full", is_flag=True, help="печатать таблицу по страницам, а не только итог")
def metrics(out_dir: Path, against: Path | None, full: bool) -> None:
    """Валидационные метрики по выкладке прогона: скрещивания осей, перескоки, блоки, краска."""
    from ocr_utils.page_layout.text_blocks.metrics import read_pages, table

    pages = read_pages(out_dir)
    if not pages:
        raise click.ClickException(f"в {out_dir} нет разборов (pages/*.json)")
    if against is not None:
        click.echo(table(pages, read_pages(against)))
    elif full:
        click.echo(table(pages))
    else:
        click.echo(table(pages, brief=True))


@main.command()
@click.option("--geo-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--nogeo-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--pages", required=True, help="список ``pdf:страница`` через запятую")
@click.option("--variant", type=click.Choice(["geo", "nogeo", "both"]), default="nogeo", show_default=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--dpi", default=WORK_DPI, show_default=True, type=float, help="разрешение рабочей копии")
@click.option("--overlay-width", default=1000, show_default=True, type=int, help="ширина одной картинки")
@click.option(
    "--sides-method",
    "--align-sides",
    "sides_method",
    type=click.Choice([item.value for item in SidesMethod]),
    default=DEFAULT_SIDES_METHOD.value,
    show_default=True,
    help="разметка сторон по умолчанию: по ней рисуется выравнивание (оверлеи сторон — всеми тремя)",
)
def sides(
    geo_dir: Path,
    nogeo_dir: Path,
    pages: str,
    variant: str,
    out_dir: Path,
    dpi: float,
    overlay_width: int,
    sides_method: str,
) -> None:
    """Стороны границы блока и выравнивание по ним: все методы, оверлеи рядом и сводка сравнения."""
    import json

    from ocr_utils.page_layout.text_blocks import sides_overlay
    from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
    from ocr_utils.page_layout.text_blocks.report import sides_json, sides_markdown

    variants = [Variant.GEO.value, Variant.NOGEO.value] if variant == "both" else [variant]
    dirs = {Variant.GEO.value: geo_dir, Variant.NOGEO.value: nogeo_dir}
    analyses = []
    for name, page in parse_pages(pages):
        for current in variants:
            pdf = dirs[current] / f"{name}.pdf"
            if not pdf.is_file():
                click.echo(f"нет файла: {pdf}")
                continue
            gray300 = render_page(pdf, page)
            # Разбор тот же, что у ``analyze`` без подсказок; стороны и выравнивание — поверх него.
            analysis = analyse_gray(gray300, InkEngine(), dpi=dpi, name=name, page=page, variant=current)
            sides_overlay.write_all(analysis, gray300, out_dir, overlay_width, SidesMethod(sides_method))
            path = out_dir / "pages" / f"{analysis.key}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(sides_json(analysis), ensure_ascii=False), encoding="utf-8")
            analyses.append(analysis)
            click.echo(f"{analysis.key}: блоков {len(analysis.blocks)}")
    if analyses:
        (out_dir / "report.md").write_text(sides_markdown(analyses), encoding="utf-8")
        click.echo(f"готово: {out_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
