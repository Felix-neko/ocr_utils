"""Простой пример: найти текстовые блоки на картинке страницы и напечатать, что нашлось.

Запуск (картинка рядом, ничего скачивать не надо)::

    uv run python -m ocr_utils.page_layout.text_blocks.example
    uv run python -m ocr_utils.page_layout.text_blocks.example --overlay /tmp/blocks.jpg

Пример намеренно минимальный: одна картинка, один движок, никаких подсказок от других
детекторов. Как подать подсказки (растр, таблицы, боковой текст, ячейки) и как разобрать
результат подробнее — в README подпакета.
"""

from __future__ import annotations

from pathlib import Path

import click
import cv2

from ocr_utils.page_layout import px_to_mm
from ocr_utils.page_layout.text_blocks import RENDER_DPI, WORK_DPI
from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
from ocr_utils.page_layout.text_blocks.page import analyse_gray

# Страница журнала «Материально-техническое снабжение» (1970/02, с. 90) — бинаризованный рендер
# FineReader при ``RENDER_DPI``: рубрика, заголовок, вводка полужирным и два столбца корпуса.
EXAMPLE_PAGE = Path(__file__).with_name("example_page.png")


@click.command()
@click.option(
    "--image",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=EXAMPLE_PAGE,
    show_default="example_page.png рядом с примером",
    help=f"серая картинка страницы при {RENDER_DPI} dpi бумаги",
)
@click.option(
    "--render-dpi",
    default=float(RENDER_DPI),
    show_default=True,
    type=float,
    help="разрешение КАРТИНКИ; размеры детектора привязаны к бумаге, и врать тут нельзя",
)
@click.option(
    "--overlay",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="куда положить оверлей разбора (оси строк и границы блоков поверх страницы)",
)
def main(image: Path, render_dpi: float, overlay: Path | None) -> None:
    """Найти текстовые блоки на картинке страницы и напечатать их."""
    # ВХОД: одна серая картинка. Цветную или бинарную приводим к серой 8-битной — детектор
    # бинаризует сам (Оцу), и заранее бинаризовать не нужно.
    gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise click.ClickException(f"не прочиталась картинка: {image}")
    if abs(render_dpi - RENDER_DPI) > 1e-6:
        # Все пороги детектора — в миллиметрах бумаги, поэтому картинку с другим разрешением
        # проще пересчитать до ожидаемого, чем уговаривать детектор.
        scale = RENDER_DPI / render_dpi
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    analysis = analyse_gray(gray, InkEngine(), name=image.stem)

    # ВЫХОД: оси строк (``analysis.axes``) и блоки (``analysis.blocks``) в пикселях РАБОЧЕЙ копии
    # (``WORK_DPI``), то есть вдвое мельче картинки. В миллиметры — через ``px_to_mm``.
    click.echo(
        f"{image.name}: {analysis.width}×{analysis.height} px рабочей копии при {analysis.dpi:.0f} dpi, "
        f"строк {len(analysis.axes)}, блоков {len(analysis.blocks)}, {analysis.seconds:.1f} с"
    )
    for block, alignment in zip(analysis.blocks, analysis.alignments):
        polygon = block.envelope.polygon
        x0, y0 = polygon[:, 0].min(), polygon[:, 1].min()
        x1, y1 = polygon[:, 0].max(), polygon[:, 1].max()
        click.echo(
            f"  блок кол.{block.column}.{block.index}: строк {block.lines}, "
            f"шаг {block.pitch_mm:.1f} мм, выключка {alignment.kind.value}, "
            f"рамка {px_to_mm(x0, analysis.dpi):.0f}×{px_to_mm(y0, analysis.dpi):.0f} — "
            f"{px_to_mm(x1, analysis.dpi):.0f}×{px_to_mm(y1, analysis.dpi):.0f} мм"
        )

    if overlay is not None:
        from ocr_utils.page_layout.text_blocks import overlay as draw

        draw.write(analysis, gray, overlay)
        click.echo(f"оверлей: {overlay}")


if __name__ == "__main__":
    main()
