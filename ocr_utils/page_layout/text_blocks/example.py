"""Простой пример: найти текстовые блоки на картинке страницы, напечатать и ПОКАЗАТЬ их.

Запуск (картинка рядом, ничего скачивать не надо)::

    uv run python -m ocr_utils.page_layout.text_blocks.example
    uv run python -m ocr_utils.page_layout.text_blocks.example --picture /tmp/blocks.png
    uv run python -m ocr_utils.page_layout.text_blocks.example --no-show   # только консоль

По умолчанию открывается окно matplotlib: страница серым, поверх неё зелёные осевые кривые строк и
синие границы текстовых блоков. Окно интерактивное — колесом и лупой можно приблизиться к любой
строке и посмотреть, как ось идёт по буквам.

Пример намеренно минимальный: одна картинка, один движок, никаких подсказок от других
детекторов. Как подать подсказки (растр, таблицы, боковой текст, ячейки) и как разобрать
результат подробнее — в README подпакета.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.page_layout import px_to_mm
from ocr_utils.page_layout.show import is_interactive_backend, open_in_viewer
from ocr_utils.page_layout.text_blocks import RENDER_DPI
from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
from ocr_utils.page_layout.text_blocks.page import PageAnalysis, analyse_gray

# Страница журнала «Материально-техническое снабжение» (1970/02, с. 90) — бинаризованный рендер
# FineReader при ``RENDER_DPI``: рубрика, заголовок, вводка полужирным и два столбца корпуса.
EXAMPLE_PAGE = Path(__file__).with_name("example_page.png")

# Цвета рисунка — палитра оверлеев проекта (навык draw-overlay), переведённая из BGR в RGB 0..1.
COLOR_AXIS = (40 / 255, 170 / 255, 40 / 255)  # ось строки — зелёный
COLOR_MARK = (1.0, 165 / 255, 0.0)  # участок оси над точкой или запятой — оранжевый
COLOR_ENVELOPE = (20 / 255, 90 / 255, 220 / 255)  # граница блока (полоса вокруг оси) — синий
COLOR_ENVELOPE_INK = (120 / 255, 170 / 255, 150 / 255)  # справочная граница по краске — приглушённый
ENVELOPE_ALPHA = 0.55  # границы блоков полупрозрачны: под ними должны читаться буквы
FILL_ALPHA = 0.08  # заливка блока — едва заметная, только чтобы блок читался как единое целое


def draw(analysis: PageAnalysis, gray: np.ndarray, title: str, out: Path | None, show: bool = True) -> None:
    """Показать разбор поверх страницы: оси строк и границы блоков.

    Args:
        analysis: Результат разбора; его координаты — в пикселях рабочей копии.
        gray: Страница в родных пикселях (вдвое крупнее рабочей копии).
        title: Заголовок рисунка.
        out: Куда сохранить PNG; ``None`` — только показать окно (без окна — во временную папку).
        show: Показывать ли рисунок. ``False`` — только сохранить в ``out``.
    """
    # Импорт здесь: matplotlib нужен только рисунку, а модуль импортируют и ради функций разбора.
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    height, width = gray.shape[:2]
    # Разбор отдаёт координаты в рабочей копии (WORK_DPI), рисуем в пикселях поданной картинки.
    scale = width / analysis.width
    figure, axes = plt.subplots(figsize=(9, 9 * height / width))
    axes.imshow(gray, cmap="gray", vmin=0, vmax=255)

    for block in analysis.blocks:
        # Справочная граница ПО КРАСКЕ идёт первой и тоньше: главная не должна за ней теряться.
        if block.envelope_ink is not None:
            ink = block.envelope_ink.polygon * scale
            axes.add_patch(Polygon(ink, closed=True, fill=False, ec=COLOR_ENVELOPE_INK, lw=1, alpha=ENVELOPE_ALPHA))
        polygon = block.envelope.polygon * scale
        axes.add_patch(Polygon(polygon, closed=True, fc=COLOR_ENVELOPE, ec="none", alpha=FILL_ALPHA))
        axes.add_patch(Polygon(polygon, closed=True, fill=False, ec=COLOR_ENVELOPE, lw=1.6, alpha=ENVELOPE_ALPHA))
        # Подпись блока — у его левого верхнего угла, подложкой, чтобы читалась поверх текста.
        axes.text(
            polygon[:, 0].min(),
            polygon[:, 1].min() - 4,
            f"кол.{block.column}.{block.index}: строк {block.lines}, шаг {block.pitch_mm:.1f} мм",
            fontsize=6,
            va="bottom",
            color="black",
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5),
        )

    for axis in analysis.axes:
        points = np.asarray(axis.points, dtype=np.float64) * scale
        axes.plot(points[:, 0], points[:, 1], color=COLOR_AXIS, lw=1.0)
        # Над точкой и запятой ось провисает к базовой линии: эти участки метятся отдельно, чтобы
        # провисание не принимали за дефект (в меры формы строки они не входят).
        for x0, x1 in axis.mark_spans:
            middle = (x0 + x1) / 2.0 * scale
            axes.plot([middle], [np.interp(middle, points[:, 0], points[:, 1])], marker=".", color=COLOR_MARK, ms=3)

    legend = [
        Line2D([], [], color=COLOR_AXIS, lw=1.2, label="осевая кривая строки"),
        Line2D([], [], color=COLOR_MARK, lw=0, marker=".", ms=6, label="ось над точкой, запятой"),
        Line2D([], [], color=COLOR_ENVELOPE, lw=1.6, alpha=ENVELOPE_ALPHA, label="граница блока: полоса вокруг оси"),
        Line2D([], [], color=COLOR_ENVELOPE_INK, lw=1, alpha=ENVELOPE_ALPHA, label="граница по краске, справочно"),
    ]
    # Легенда — под картинкой, чтобы не закрывать текст полосы.
    figure.legend(handles=legend, loc="lower center", ncol=2, fontsize=8, frameon=False)
    axes.set_title(title)
    axes.set_axis_off()
    figure.tight_layout(rect=(0, 0.06, 1, 1))

    interactive = show and is_interactive_backend()
    # Показать нечем и не сохранено — сохраняем во временную папку, иначе рисунок пропадёт зря.
    if out is None and show and not interactive:
        out = Path(tempfile.gettempdir()) / "text_blocks_example.png"
    if out is not None:
        figure.savefig(out, dpi=150)
        click.echo(f"рисунок сохранён: {out}")
    if interactive:
        plt.show()
    elif show:
        open_in_viewer(out)
    plt.close(figure)


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
    "--picture",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="куда сохранить рисунок PNG; если окно matplotlib недоступно, рисунок сохраняется во "
    "временную папку и открывается просмотрщиком (xdg-open)",
)
@click.option("--show/--no-show", default=True, show_default=True, help="показывать ли рисунок")
def main(image: Path, render_dpi: float, picture: Path | None, show: bool) -> None:
    """Найти текстовые блоки на картинке страницы, напечатать их и показать поверх страницы."""
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
    header = (
        f"{image.name}: {analysis.width}×{analysis.height} px рабочей копии при {analysis.dpi:.0f} dpi, "
        f"строк {len(analysis.axes)}, блоков {len(analysis.blocks)}, {analysis.seconds:.1f} с"
    )
    click.echo(header)
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

    if show or picture is not None:
        draw(analysis, gray, header, picture, show=show)


if __name__ == "__main__":
    main()
