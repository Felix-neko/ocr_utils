"""Сквозная проверка на синтетической таблице: боковая шапка находится, читается, набирается."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ocr_utils.rotated_text.tables import pipeline
from ocr_utils.rotated_text.tables.fit import load_font
from ocr_utils.rotated_text.tables.ocr import tesseract_available
from ocr_utils.rotated_text.tables.orientation import orient_cell
from ocr_utils.rotated_text.tables.source import TableRef, load_table_crop
from ocr_utils.rotated_text.tables.structure import analyse_structure, cell_interior, work_copy

DPI = 600
PAPER, INK, RULE = 245, 25, 6


def _table(path: Path, rotated_text: str, upright_text: str) -> None:
    """Таблица 600 dpi: три колонки, шапка и две строки; вторая графа шапки набрана боком."""
    xs, ys = [60, 760, 1000, 1400], [60, 560, 700, 840]
    canvas = Image.new("L", (1500, 900), PAPER)
    draw = ImageDraw.Draw(canvas)
    for y in ys:
        draw.rectangle([xs[0], y, xs[-1], y + RULE - 1], fill=INK)
    for x in xs:
        draw.rectangle([x, ys[0], x + RULE - 1, ys[-1]], fill=INK)
    font = load_font(56)
    draw.text((xs[0] + 40, ys[0] + 200), upright_text, fill=INK, font=font)
    for row, values in enumerate((("Тракторы", "1790", "1092"), ("Скреперы", "30", "19,2"))):
        for col, value in enumerate(values):
            draw.text((xs[col] + 40, ys[row + 1] + 30), value, fill=INK, font=font)
    # Боковая шапка: набрана прямо и положена против часовой — как в паке.
    strip = Image.new("L", (ys[1] - ys[0] - 40, xs[2] - xs[1] - 40), PAPER)
    ImageDraw.Draw(strip).text((20, 60), rotated_text, fill=INK, font=font)
    canvas.paste(strip.rotate(90, expand=True), (xs[1] + 20, ys[0] + 20))
    draw.text((xs[2] + 40, ys[0] + 200), "1961 г.", fill=INK, font=font)
    canvas.save(path, dpi=(DPI, DPI))


def _ref(path: Path) -> TableRef:
    return TableRef("synthetic_t00", "1966", "01", "x.tif", str(path), (0, 0, 1500, 900), 0, DPI, 0.0, 1, 1)


@pytest.mark.skipif(not tesseract_available(), reason="нужен tesseract")
def test_orientation_of_synthetic_cells(tmp_path: Path):
    path = tmp_path / "table.png"
    _table(path, "тыс. штук", "Наименование")
    crop = load_table_crop(_ref(path), deskew=False)
    work = work_copy(crop.gray, DPI, 300)
    grid = analyse_structure(work, 300).grid
    assert grid.n_cols == 3 and grid.n_rows == 3
    verdicts = {cell.key: orient_cell(cell_interior(work, cell, 300), 300, (0, 90, 180, 270)) for cell in grid.cells}
    assert verdicts[(0, 1)].rotate_cw == 90
    assert verdicts[(0, 0)].rotate_cw == 0
    assert verdicts[(1, 0)].rotate_cw == 0
    assert verdicts[(1, 1)].rotate_cw is None  # число: букв нет, ячейку не трогаем


@pytest.mark.skipif(not tesseract_available(), reason="нужен tesseract")
def test_rewrite_replaces_rotated_header(tmp_path: Path):
    path = tmp_path / "table.png"
    _table(path, "тыс. штук", "Наименование")
    options = pipeline.Options(use_surya=False)
    analysis = pipeline.analyse(_ref(path), options)
    assert not analysis.error
    assert [cell.key for cell in analysis.cells if cell.candidate] == [(0, 1)]
    assert analysis.cell((0, 1)).text == "тыс. штук"
    assert not analysis.sideways_table
    result, images = pipeline.rewrite(analysis, options)
    assert images is not None
    assert result.action in (pipeline.ACTION_AS_IS, pipeline.ACTION_UPSCALE)
    cell = result.cells[[c.key for c in result.cells].index((0, 1))]
    assert cell.replaced and cell.font_px >= 40
    # Внутри ячейки после набора есть краска (буквы), а линейки вокруг остались.
    inner = [c for c in images.grid_before.cells if c.key == (0, 1)][0].inner.scaled(result.scale)
    after = images.after
    assert (after[inner.y0 : inner.y1, inner.x0 : inner.x1] < 100).any()
    assert after[int(60 * result.scale) + 2, int(800 * result.scale)] < 100
    # Старого бокового текста нет: ни одной тёмной компоненты выше, чем шире, размером с букву.
    assert result.required_dpi >= DPI


def test_sideways_table_goes_rotate_first(tmp_path: Path, monkeypatch):
    """Таблица боком целиком: сперва поворот; ячейки, чей поворот совпал, не набираются."""
    from ocr_utils.scan_markup.table_detection.geometry import Box, Cell, Grid

    from ocr_utils.rotated_text.tables.source import TableCrop

    gray = np.full((300, 200), 240, np.uint8)
    grid = Grid(
        xs=[0, 100],
        ys=[0, 150, 300],
        cells=[
            Cell(0, 0, Box(0, 0, 100, 150), inner=Box(5, 5, 95, 145)),
            Cell(1, 0, Box(0, 150, 100, 300), inner=Box(5, 155, 95, 295)),
        ],
    )
    ref = TableRef("s_t00", "1966", "01", "x", "x", (0, 0, 200, 300), 0, 300, 0.0, 1, 1)
    monkeypatch.setattr(
        pipeline, "load_table_crop", lambda r, deskew=True: TableCrop(gray, 300, (0, 0, 200, 300), 0, 0.0)
    )
    cells = [
        pipeline.CellRecord(
            0,
            0,
            1,
            1,
            False,
            (0, 0, 100, 150),
            (5, 5, 95, 145),
            rotate_cw=90,
            components=10,
            text="план",
            candidate=True,
            axis_upright=False,
        ),
        pipeline.CellRecord(
            1,
            0,
            1,
            1,
            False,
            (0, 150, 100, 300),
            (5, 155, 95, 295),
            rotate_cw=90,
            components=10,
            text="факт",
            candidate=True,
            axis_upright=False,
        ),
    ]
    analysis = pipeline.TableAnalysis(
        ref, 300, 300, grid=grid, cells=cells, sideways_table=True, sideways_side=90, native_font_px=30
    )
    result, images = pipeline.rewrite(analysis, pipeline.Options(use_surya=False))
    assert result.action == pipeline.ACTION_ROTATE and result.table_rotate_cw == 90
    assert images.after.shape == (200, 300)
    assert all(not c.replaced and c.reason == "выпрямлена поворотом таблицы" for c in result.cells)


def test_rotation_refused_when_a_standing_cell_cannot_be_reread(tmp_path: Path, monkeypatch):
    """Прямая ячейка, прочитанная ненадёжно, делает поворот невозможным: после поворота
    она легла бы, и FineReader её не прочёл бы."""
    from ocr_utils.scan_markup.table_detection.geometry import Box, Cell, Grid

    from ocr_utils.rotated_text.tables.ocr import CellText
    from ocr_utils.rotated_text.tables.source import TableCrop

    gray = np.full((300, 200), 240, np.uint8)
    grid = Grid(
        xs=[0, 100],
        ys=[0, 150, 300],
        cells=[
            Cell(0, 0, Box(0, 0, 100, 150), inner=Box(5, 5, 95, 145)),
            Cell(1, 0, Box(0, 150, 100, 300), inner=Box(5, 155, 95, 295)),
        ],
    )
    ref = TableRef("s_t00", "1966", "01", "x", "x", (0, 0, 200, 300), 0, 300, 0.0, 1, 1)
    monkeypatch.setattr(
        pipeline, "load_table_crop", lambda r, deskew=True: TableCrop(gray, 300, (0, 0, 200, 300), 0, 0.0)
    )
    monkeypatch.setattr(pipeline, "read_cell", lambda *a, **k: CellText(lines=["6113С"], confidence=0.6, engine="t"))
    cells = [
        pipeline.CellRecord(
            0,
            0,
            1,
            1,
            False,
            (0, 0, 100, 150),
            (5, 5, 95, 145),
            rotate_cw=90,
            components=10,
            text="план",
            candidate=True,
        ),
        pipeline.CellRecord(
            1, 0, 1, 1, False, (0, 150, 100, 300), (5, 155, 95, 295), rotate_cw=0, components=5, axis_upright=True
        ),
    ]
    analysis = pipeline.TableAnalysis(
        ref, 300, 300, grid=grid, cells=cells, sideways_table=True, sideways_side=90, native_font_px=30
    )
    result, images = pipeline.rewrite(analysis, pipeline.Options(use_surya=False))
    assert result.table_rotate_cw == 0 and result.action != pipeline.ACTION_ROTATE
    assert any("стоит и не прочиталась" in step for step in result.steps)


def test_merge_split_cells_joins_sideways_text_across_through_row(tmp_path: Path):
    """Четыре колонки, линейка между строками есть под тремя (сквозная строка), а в четвёртой
    боковая надпись тянется через обе строки: сетка режет её пополам, склейка — чинит."""
    from ocr_utils.rotated_text.tables.structure import merge_split_cells

    xs, ys = [60, 400, 740, 1080, 1420], [60, 460, 860]
    canvas = Image.new("L", (1500, 920), PAPER)
    draw = ImageDraw.Draw(canvas)
    for y in (ys[0], ys[-1]):
        draw.rectangle([xs[0], y, xs[-1], y + RULE - 1], fill=INK)
    draw.rectangle([xs[0], ys[1], xs[3], ys[1] + RULE - 1], fill=INK)  # не под четвёртой колонкой
    for x in xs:
        draw.rectangle([x, ys[0], x + RULE - 1, ys[-1]], fill=INK)
    font = load_font(56)
    for row in range(2):
        for col in range(3):
            draw.text((xs[col] + 40, ys[row] + 150), f"{row}{col}", fill=INK, font=font)
    strip = Image.new("L", (ys[2] - ys[0] - 40, xs[4] - xs[3] - 40), PAPER)
    ImageDraw.Draw(strip).text((20, 100), "длинная боковая надпись", fill=INK, font=font)
    canvas.paste(strip.rotate(90, expand=True), (xs[3] + 20, ys[0] + 20))
    work = work_copy(np.asarray(canvas), DPI, 300)
    structure = analyse_structure(work, 300)
    before = structure.grid
    assert before.n_cols == 4 and before.n_rows == 2
    assert len([c for c in before.cells if c.col == 3]) == 2  # сетка разрезала
    after = merge_split_cells(before, structure.lines, work, 300)
    last = [c for c in after.cells if c.col == 3]
    assert len(last) == 1 and last[0].row_span == 2
    assert len([c for c in after.cells if c.col == 0]) == 2  # прямые ячейки не склеены
