"""Пример детектора таблиц: surya + детектор на полосе, трассы линеек сплайнами и сетка ячеек по кривым, печать и рисунок.

Запуск без параметров::

    uv run python -m ocr_utils.page_layout.tables.example

Рядом с модулем лежат две картинки пака-1 в 300 dpi (выбор — ``--sample``). Картинки примера хранятся
только в JPEG, без цвета — в градациях серого (один канал), качество 85–92: так они меньше весят в git.

* ``clean`` (по умолчанию) — ``example_table_clean.jpg``, правая колонка полосы 1968/05 с.70, таблицы 4 и
  5 на изогнутой полосе: таблица наклонена на −0,7° и +0,4°, наклон отдельных линеек плавает от
  −1,2° до +0,8°, нижняя линейка таблицы 5 волнистая;
* ``curved`` — ``example_table_curved.jpg``, полоса 1968/03 с.4, таблица 1: наклон −2,5°, линейки от
  −5° до +1,5° по длине, у верхней — излом. Прежняя сетка по осям (``--axis-grid``) здесь ломается;
  сетка по кривым даёт верные 10 ячеек.

Пример каждый раз зовёт surya layout (GPU, без кэша), ищет таблицы детектором, по каждой таблице
строит трассы линеек (сглаживающий сплайн вдоль каждой, ``traces``) и сетку ячеек по этим кривым
(``curved_grid``), печатает рёбра и ячейки в консоль и показывает всё поверх картинки в окне matplotlib.

Три шага и три разрешения — так же, как в рабочем конвейере (``text_layer_fix.zones``):

1. surya и детектор таблиц — на копии 150 dpi (``ruling.WORK_DPI``), под которую калиброван детектор;
2. трассы и сетка — на вырезке таблицы в 300 dpi (``grid.WORK_DPI``), под которую калиброваны пороги;
3. всё, что печатается и рисуется, пересчитано в пиксели поданной картинки.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import click
import cv2
import numpy as np

from ocr_utils.page_layout.geometry import KIND_TABLE, Grid, TableBox
from ocr_utils.page_layout.image import PageImage, Variant
from ocr_utils.page_layout.surya.blocks import LayoutBlocks
from ocr_utils.page_layout.surya.model import SuryaLayoutModel
from ocr_utils.page_layout.tables import grid as grid_module
from ocr_utils.page_layout.tables import ruling
from ocr_utils.page_layout.tables.curved_grid import CurvedGrid, curved_table
from ocr_utils.page_layout.tables.detector import detect
from ocr_utils.page_layout.tables.ruling import find_lines, mm_to_px
from ocr_utils.page_layout.tables.traces import Axis, RuleTrace

logger = logging.getLogger(__name__)

# Разрешение тестовых картинок; оно записано и в их JPEG-теге, но задаётся явно.
EXAMPLE_DPI = 300

# Поле вокруг рамки таблицы при вырезке под сетку: 3 мм, как ``rotated_text.tables.source.CROP_PAD_MM``.
# Без поля крайняя линейка таблицы легла бы на край вырезки, и сетка не отличила бы её от края.
CROP_PAD_MM = 3.0

# Цвета рисунка — палитра оверлеев проекта (навык draw-overlay), переведённая из BGR в RGB 0..1.
COLOR_TABLE = (20 / 255, 90 / 255, 220 / 255)  # рамка таблицы — синий, главная сущность
COLOR_CELL = (1.0, 165 / 255, 0.0)  # ячейка — оранжевый, полупрозрачно
COLOR_HEADER = (200 / 255, 60 / 255, 200 / 255)  # ячейка шапки — фиолетовый, полупрозрачно
COLOR_HORIZONTAL = (40 / 255, 170 / 255, 40 / 255)  # горизонтальная линейка — зелёный
COLOR_VERTICAL = (220 / 255, 0.0, 0.0)  # вертикальная линейка — красный
COLOR_CHORD = (20 / 255, 20 / 255, 20 / 255)  # отрезок «начало–конец» — тонкий чёрный пунктир
COLOR_AXIS_GRID = (170 / 255, 170 / 255, 170 / 255)  # прежняя сетка по осям — серый, справочно
CELL_ALPHA = 0.3  # прозрачность заливки ячеек: буквы под ней должны читаться


class Sample(StrEnum):
    """Тестовая картинка рядом с модулем: ровная сетка / изогнутая таблица, на которой ломалась сетка по осям."""

    CLEAN = "clean"  # 1968/05 с.70, таблицы 4 и 5: наклон линеек плавает в пределах ±1,2°
    CURVED = "curved"  # 1968/03 с.4, таблица 1: наклон линеек до −5°, излом верхней линейки

    @property
    def path(self) -> Path:
        """Файл картинки рядом с модулем."""
        return Path(__file__).with_name(f"example_table_{self.value}.jpg")


@dataclass(frozen=True, eq=False)
class TableStructure:
    """Одна находка детектора с трассами линеек и сетками; всё — в пикселях поданной картинки."""

    table: TableBox  # находка детектора, рамки пересчитаны в пиксели поданной картинки
    traces: list[RuleTrace]  # линейки таблицы: сглаживающий сплайн вдоль каждой
    curved: CurvedGrid  # сетка по кривым: ячейки-многоугольники, объединения, шапка
    axis_grid: "Grid | None"  # прежняя сетка по осям (для сравнения, только с --axis-grid)


def surya_blocks(image: PageImage, work_size: tuple[int, int]) -> LayoutBlocks:
    """Разметка surya layout страницы, пересчитанная в пиксели рабочей копии детектора.

    Модель грузится при каждом вызове и отпускается сразу после ответа: пример запускается
    ради одной картинки, кэш surya (``SuryaCache``) здесь намеренно не используется.

    Args:
        image: Страница; surya получает её кадр ``image.surya_frame`` (RGB, 150 dpi).
        work_size: ``(ширина, высота)`` копии, на которой работает детектор таблиц.

    Returns:
        Блоки surya (Table, Figure, Text…) в пикселях копии ``work_size``.
    """
    model = SuryaLayoutModel()
    try:
        blocks = model.predict_one(image.surya_frame)
    finally:
        model.close()  # вернуть видеопамять: она одна на всех
    return blocks.scaled_to(*work_size)


def scale_table(table: TableBox, factor: float, width: int, height: int) -> TableBox:
    """Находка детектора, пересчитанная из пикселей рабочей копии в пиксели крупнее в ``factor`` раз.

    Args:
        table: Находка в пикселях копии 150 dpi.
        factor: Во сколько раз целевая картинка крупнее копии.
        width: Ширина целевой картинки — рамки зажимаются в кадр.
        height: Высота целевой картинки.

    Returns:
        Та же находка с пересчитанными ``box`` и ``rule_box``; вид, наклон, балл и метрики не меняются.
    """
    return TableBox(
        box=table.box.scaled(factor).clipped(width, height),
        score=table.score,
        source=table.source,
        origin=table.origin,
        skew_deg=table.skew_deg,
        metrics=table.metrics,
        rule_box=table.rule_box.scaled(factor).clipped(width, height) if table.rule_box is not None else None,
        kind=table.kind,
    )


def find_tables(image: PageImage, use_surya: bool = True) -> list[TableBox]:
    """Детектор таблиц на странице: копия 150 dpi, surya по желанию, ответ в пикселях страницы.

    Args:
        image: Страница в любом разрешении.
        use_surya: Звать ли surya layout; без неё вид объекта решают только линейки (CPU).

    Returns:
        Находки детектора — таблицы, схемы и рисунки — в пикселях родного кадра ``image``.
    """
    work_dpi = ruling.WORK_DPI
    gray_work = image.gray_at(work_dpi)
    height_work, width_work = gray_work.shape[:2]
    # Surya — необязательный второй вход: источник ВИДА объекта и подсказка протяжённости.
    layout = surya_blocks(image, (width_work, height_work)) if use_surya else None
    found = detect(gray_work, work_dpi, layout=layout)
    # Рамки детектора — в пикселях копии 150 dpi; переводим в пиксели родного кадра.
    factor = image.scale_to_native(work_dpi)
    return [scale_table(table, factor, image.width, image.height) for table in found]


def table_structure(image: PageImage, table: TableBox, with_axis_grid: bool = False) -> TableStructure:
    """Трассы линеек и сетка по кривым одной таблицы: вырезка в 300 dpi → ``curved_table`` → пиксели страницы.

    Args:
        image: Страница, на которой нашли таблицу.
        table: Находка детектора в пикселях родного кадра ``image``.
        with_axis_grid: Построить ещё и прежнюю сетку по осям (``find_lines`` → ``grid_from_lines``).

    Returns:
        Трассы и сетки таблицы в пикселях родного кадра.
    """
    work_dpi = grid_module.WORK_DPI
    # Вырезка с полем в 3 мм, чтобы крайние линейки не легли на край кадра.
    crop_box = table.box.padded(mm_to_px(CROP_PAD_MM, image.dpi)).clipped(image.width, image.height)
    crop = image.gray[crop_box.slice]
    # Пороги трасс и сетки калиброваны на 300 dpi: вырезку приводим к нему.
    to_work = work_dpi / image.dpi
    if to_work != 1.0:
        interpolation = cv2.INTER_AREA if to_work < 1 else cv2.INTER_CUBIC
        crop = cv2.resize(crop, None, fx=to_work, fy=to_work, interpolation=interpolation)
    traces, curved = curved_table(crop, work_dpi)
    # Из пикселей вырезки 300 dpi — в пиксели страницы: масштаб, потом сдвиг на угол вырезки.
    factor = image.dpi / work_dpi
    page_traces = [trace.mapped(factor, crop_box.x0, crop_box.y0) for trace in traces.all]
    page_curved = curved.mapped(factor, crop_box.x0, crop_box.y0)
    axis_grid = None
    if with_axis_grid:
        lines = find_lines(crop, work_dpi)
        grid = grid_module.grid_from_lines(lines, crop.shape[:2], work_dpi, grid_module.text_ink(crop, lines))
        axis_grid = scale_grid(grid, factor).shifted(crop_box.x0, crop_box.y0)
    return TableStructure(table=table, traces=page_traces, curved=page_curved, axis_grid=axis_grid)


def scale_grid(grid: Grid, factor: float) -> Grid:
    """Сетка по осям, пересчитанная в разрешение крупнее в ``factor`` раз (без сдвига).

    Args:
        grid: Сетка в пикселях вырезки.
        factor: Множитель координат.

    Returns:
        Новая сетка с теми же строками, колонками и объединениями.
    """
    if factor == 1.0:
        return grid
    payload = grid.to_json()
    payload["xs"] = [round(x * factor) for x in payload["xs"]]
    payload["ys"] = [round(y * factor) for y in payload["ys"]]
    payload["double_rule_ys"] = [round(y * factor) for y in payload["double_rule_ys"]]
    for cell in payload["cells"]:
        cell["box"] = [round(v * factor) for v in cell["box"]]
        if cell["inner"] is not None:
            cell["inner"] = [round(v * factor) for v in cell["inner"]]
    scaled = Grid.from_json(payload)
    # ``to_json`` не пишет места разреза — переносим их отдельно.
    scaled.column_cuts = [round(x * factor) for x in grid.column_cuts]
    return scaled


def print_structure(index: int, structure: TableStructure) -> None:
    """Печать находки: рамка, рёбра (концы, наклон хорды и по длине, отклонение от хорды), ячейки.

    Args:
        index: Номер находки на странице.
        structure: Находка с трассами и сетками.
    """
    table, curved = structure.table, structure.curved
    print(f"\n=== Находка {index}: {table.kind}, балл {table.score:.2f}, наклон линеек {table.skew_deg:+.2f}°")
    print(f"рамка (x0, y0, x1, y1): {table.box.as_tuple()}; рамка линеек: {table.rules.as_tuple()}")

    print(f"\nрёбра — {len(structure.traces)} линеек (координаты — пиксели картинки):")
    print(
        f"{'№':>3} {'ось':<11} {'начало (x, y)':>16} {'конец (x, y)':>16} {'длина':>6} {'хорда':>7} "
        f"{'наклон по длине':>16} {'откл. от хорды':>14} {'полоса':>6}"
    )
    for number, trace in enumerate(structure.traces):
        (x0, y0), (x1, y1) = trace.chord
        angles = trace.angle_deg_at(trace.along_grid(1.0))
        deviation, _ = trace.max_chord_deviation_px()
        deviation_mm = deviation * 25.4 / trace.output_dpi
        print(
            f"{number:>3} {trace.axis:<11} {f'({x0:.0f}, {y0:.0f})':>16} {f'({x1:.0f}, {y1:.0f})':>16} "
            f"{trace.length_mm:>4.0f}мм {trace.chord_angle_deg:>+6.2f}° "
            f"{f'{angles.min():+.2f}…{angles.max():+.2f}°':>16} {deviation_mm:>11.2f} мм {trace.bandwidth_mm:>4.1f}мм"
        )

    print(f"\nсетка по кривым: {curved.n_rows} строк × {curved.n_cols} колонок, строк шапки {curved.header_rows}")
    print(f"ячейки — {len(curved.cells)}:")
    print(f"{'стр':>3} {'кол':>3} {'объед.':>6} {'шапка':>5}  рамка многоугольника       точек  область ячейки (px)")
    for cell in curved.cells:
        span = f"{cell.row_span}×{cell.col_span}"
        header = "да" if cell.is_header else ""
        print(
            f"{cell.row:>3} {cell.col:>3} {span:>6} {header:>5}  {str(cell.box.as_tuple()):<26} "
            f"{len(cell.polygon):>5}  {int(cell.region_area):>6}"
        )
    if structure.axis_grid is not None:
        axis = structure.axis_grid
        print(f"\nдля сравнения, сетка по осям: {axis.n_rows} строк × {axis.n_cols} колонок, ячеек {len(axis.cells)}")


def draw(
    gray: np.ndarray, structures: list[TableStructure], others: list[TableBox], title: str, out: Path | None
) -> None:
    """Рисунок: полоса серым, ячейки-многоугольники полупрозрачно, линейки кривыми, хорды пунктиром.

    Args:
        gray: Страница в родных пикселях.
        structures: Таблицы с трассами и сетками.
        others: Находки другого вида (схемы, рисунки) — только рамкой.
        title: Заголовок рисунка.
        out: Куда сохранить PNG; ``None`` — только показать окно (без окна — во временную папку).
    """
    # Импорт здесь: matplotlib нужен только рисунку, а модуль импортируют и ради функций разбора.
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Polygon, Rectangle

    height, width = gray.shape[:2]
    figure, axes = plt.subplots(figsize=(11, 11 * height / width))
    axes.imshow(gray, cmap="gray", vmin=0, vmax=255)
    for structure in structures:
        box = structure.table.box
        axes.add_patch(Rectangle((box.x0, box.y0), box.width, box.height, fill=False, ec=COLOR_TABLE, lw=1.5))
        if structure.axis_grid is not None:
            for cell in structure.axis_grid.cells:
                b = cell.box
                axes.add_patch(Rectangle((b.x0, b.y0), b.width, b.height, fill=False, ec=COLOR_AXIS_GRID, lw=0.8))
        # Ячейки — многоугольниками по кривым, с подписью «строка,колонка» и объединением.
        for cell in structure.curved.cells:
            color = COLOR_HEADER if cell.is_header else COLOR_CELL
            axes.add_patch(Polygon(cell.polygon, closed=True, fc=color, ec="none", alpha=CELL_ALPHA))
            spans = f" ({cell.row_span}×{cell.col_span})" if cell.row_span * cell.col_span > 1 else ""
            x, y = cell.polygon[0]
            axes.text(
                x + 4,
                y + 4,
                f"{cell.row},{cell.col}{spans}",
                fontsize=6,
                va="top",
                color="black",
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5),
            )
        # Линейки — сплайном цветом оси, хорда — тонким пунктиром, место наибольшего расхождения — точкой.
        for trace in structure.traces:
            color = COLOR_HORIZONTAL if trace.axis == Axis.HORIZONTAL else COLOR_VERTICAL
            curve = trace.sample(0.5)
            axes.plot(curve[:, 0], curve[:, 1], color=color, lw=1.3)
            (x0, y0), (x1, y1) = trace.chord
            axes.plot([x0, x1], [y0, y1], color=COLOR_CHORD, lw=0.6, ls="--")
            _deviation, where = trace.max_chord_deviation_px()
            point = trace.point_at(where)[0]
            axes.plot(point[0], point[1], "o", color=color, ms=3)
    for other in others:
        b = other.box
        axes.add_patch(Rectangle((b.x0, b.y0), b.width, b.height, fill=False, ec=COLOR_HEADER, lw=2, ls=":"))

    # Легенда обязательна; образцы заливки — с той же прозрачностью, что на странице.
    legend = [
        Line2D([], [], color=COLOR_TABLE, lw=1.5, label="рамка таблицы (box)"),
        Patch(fc=COLOR_CELL, alpha=CELL_ALPHA, label="ячейка тела"),
        Patch(fc=COLOR_HEADER, alpha=CELL_ALPHA, label="ячейка шапки"),
        Line2D([], [], color=COLOR_HORIZONTAL, lw=1.3, label="горизонтальная линейка (сплайн)"),
        Line2D([], [], color=COLOR_VERTICAL, lw=1.3, label="вертикальная линейка (сплайн)"),
        Line2D([], [], color=COLOR_CHORD, lw=0.6, ls="--", label="отрезок «начало–конец»"),
        Line2D([], [], color=COLOR_CHORD, marker="o", ls="none", ms=3, label="наибольшее отклонение от отрезка"),
    ]
    if any(structure.axis_grid is not None for structure in structures):
        legend.append(Line2D([], [], color=COLOR_AXIS_GRID, lw=0.8, label="прежняя сетка по осям"))
    if others:
        legend.append(Line2D([], [], color=COLOR_HEADER, lw=2, ls=":", label="схема / рисунок"))
    # Легенда — под картинкой, чтобы не закрывать текст полосы.
    figure.legend(handles=legend, loc="lower center", ncol=3, fontsize=8, frameon=False)
    axes.set_title(title)
    axes.set_axis_off()
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    interactive = is_interactive_backend()
    # Без окна рисунок обязательно сохраняется: иначе показать его нечем.
    if out is None and not interactive:
        out = Path(tempfile.gettempdir()) / "table_example.png"
    if out is not None:
        figure.savefig(out, dpi=150)
        print(f"\nрисунок сохранён: {out}")
    if interactive:
        plt.show()
    else:
        open_in_viewer(out)


def is_interactive_backend() -> bool:
    """Умеет ли выбранный backend matplotlib открывать окно.

    Returns:
        ``False`` для Agg и прочих файловых backend'ов (pdf, svg…), ``True`` для TkAgg, QtAgg и т. п.
    """
    import matplotlib
    from matplotlib import backends

    non_interactive = backends.backend_registry.list_builtin(backends.BackendFilter.NON_INTERACTIVE)
    return matplotlib.get_backend().lower() not in {name.lower() for name in non_interactive}


def open_in_viewer(path: Path) -> None:
    """Открыть картинку системным просмотрщиком (``xdg-open``), не дожидаясь его закрытия.

    Нужна, когда matplotlib не может открыть окно сам: на машине без экрана или когда у
    интерпретатора нет рабочего Tk/Qt и backend откатывается на Agg.

    Args:
        path: Сохранённый PNG.
    """
    viewer = shutil.which("xdg-open")
    if viewer is None:
        print("окно matplotlib недоступно и xdg-open не найден — откройте файл сами")
        return
    # Отдельная сессия и заглушенный вывод: просмотрщик живёт своей жизнью и не пишет в консоль примера.
    subprocess.Popen([viewer, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


@click.command()
@click.option(
    "--sample",
    type=click.Choice([sample.value for sample in Sample]),
    default=Sample.CLEAN.value,
    show_default=True,
    help="Тестовая картинка рядом с модулем: clean — умеренный наклон, curved — сильный наклон и излом линеек.",
)
@click.option(
    "--image",
    "image_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Своя картинка страницы (серая или цветная) вместо тестовой.",
)
@click.option("--dpi", type=int, default=EXAMPLE_DPI, show_default=True, help="Разрешение картинки.")
@click.option("--no-surya", is_flag=True, help="Не звать surya: вид объекта решают только линейки (CPU).")
@click.option(
    "--axis-grid", is_flag=True, help="Построить и нарисовать ещё УСТАРЕВШУЮ сетку по осям (grid.py) — для сравнения."
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Куда сохранить рисунок PNG. Если matplotlib не может открыть окно, рисунок сохраняется во "
    "временную папку и открывается просмотрщиком (xdg-open).",
)
def main(sample: str, image_path: Path | None, dpi: int, no_surya: bool, axis_grid: bool, out: Path | None) -> None:
    """Найти таблицы на картинке, разобрать их на линейки-кривые и ячейки, напечатать и показать."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # Своя картинка важнее тестовой: --sample действует, только если --image не задан.
    if image_path is None:
        image_path = Sample(sample).path
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise click.ClickException(f"не читается картинка: {image_path}")
    image = PageImage.from_array(gray, dpi=dpi, variant=Variant.SCAN)

    logger.info(
        "Ищу таблицы: %s, %d×%d px, %d dpi, surya %s",
        image_path.name,
        image.width,
        image.height,
        dpi,
        "нет" if no_surya else "да",
    )
    found = find_tables(image, use_surya=not no_surya)
    # Трассы и ячейки строятся только у таблиц: у схемы и рисунка сетки нет.
    tables = [table for table in found if table.kind == KIND_TABLE]
    others = [table for table in found if table.kind != KIND_TABLE]
    print(f"находок: {len(found)} (таблиц {len(tables)}, схем и рисунков {len(others)})")

    structures = [table_structure(image, table, axis_grid) for table in tables]
    for index, structure in enumerate(structures):
        print_structure(index, structure)
    draw(gray, structures, others, f"Детектор таблиц: {image_path.name}", out)


if __name__ == "__main__":
    main()
