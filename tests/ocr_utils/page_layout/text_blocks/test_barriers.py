"""Линейки-барьеры: через них не сращиваются ни строки, ни ряды, ни блоки (прямые и изогнутые)."""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.text_blocks import RENDER_DPI, WORK_DPI
from ocr_utils.page_layout.text_blocks.barriers import BarrierLines
from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
from ocr_utils.page_layout.text_blocks.hints import LayoutHints
from ocr_utils.page_layout.text_blocks.page import _forward_points, analyse_gray
from tests.ocr_utils.page_layout.text_blocks.synthetic import column_page

SCALE = WORK_DPI / RENDER_DPI


def _work_size(page: np.ndarray) -> tuple[int, int]:
    """Размер рабочей копии страницы ``(ширина, высота)``."""
    return int(round(page.shape[1] * SCALE)), int(round(page.shape[0] * SCALE))


def _analyse(page: np.ndarray, rules=()):
    """Разбор страницы с линейками-барьерами ``rules`` (ломаные в пикселях рабочей копии)."""
    hints = LayoutHints(rules=tuple(tuple(map(tuple, line)) for line in rules))
    return analyse_gray(page, InkEngine(hints=hints), hints=hints)


def _vertical(x: float, height: int, bow: float = 0.0) -> list[tuple[float, float]]:
    """Вертикальная линейка на всю высоту, с прогибом ``bow`` px в середине (0 — прямая)."""
    ys = np.linspace(0, height - 1, 40)
    return [(x + bow * np.sin(np.pi * y / (height - 1)), y) for y in ys]


def _horizontal(y: float, width: int, sag: float = 0.0) -> list[tuple[float, float]]:
    """Горизонтальная линейка на всю ширину, с прогибом ``sag`` px в середине (0 — прямая)."""
    xs = np.linspace(0, width - 1, 40)
    return [(x, y + sag * np.sin(np.pi * x / (width - 1))) for x in xs]


def _crossing_axes(analysis, line: list[tuple[float, float]]) -> int:
    """Сколько осей строк пересекает ломаную (ось — ломаная своих точек)."""
    barrier = BarrierLines.of([line])
    count = 0
    for axis in analysis.axes:
        points = np.asarray(axis.points)
        if any(barrier.crosses(tuple(a), tuple(b)) for a, b in zip(points[:-1], points[1:])):
            count += 1
    return count


def test_vertical_barrier_splits_lines() -> None:
    """Вертикальная линейка посреди колонки: без неё строки идут насквозь, с ней — ни одна ось не пересекает."""
    page = column_page(columns=1)
    width, height = _work_size(page)
    line = _vertical(width / 2, height)
    plain = _analyse(page)
    assert _crossing_axes(plain, line) > 10
    split = _analyse(page, [line])
    assert _crossing_axes(split, line) == 0
    assert len(split.axes) > len(plain.axes)


def test_curved_vertical_barrier_splits_lines() -> None:
    """То же с изогнутой вертикалью (прогиб 12 px рабочей копии, ≈ 2 мм)."""
    page = column_page(columns=1)
    width, height = _work_size(page)
    line = _vertical(width / 2, height, bow=12.0)
    assert _crossing_axes(_analyse(page), line) > 10
    assert _crossing_axes(_analyse(page, [line]), line) == 0


def _blocks_across(analysis, line: list[tuple[float, float]]) -> int:
    """Сколько блоков имеют ряды и над линейкой, и под ней (в середине ширины блока)."""
    barrier = BarrierLines.of([line])
    count = 0
    for block in analysis.blocks:
        ys = [row.y for row in block.rows]
        middle = (block.span[0] + block.span[1]) / 2
        # Уровень линейки на середине блока — по пересечению с вертикалью через середину.
        above = any(
            not barrier.crosses((middle, y), (middle, 0.0)) and barrier.crosses((middle, y), (middle, 1e6)) for y in ys
        )
        below = any(barrier.crosses((middle, y), (middle, 0.0)) for y in ys)
        count += above and below
    return count


def test_horizontal_barrier_splits_block() -> None:
    """Горизонтальная линейка между двумя рядами колонки режет блок, хотя разрыва нет."""
    page = column_page(columns=1)
    width, height = _work_size(page)
    plain = _analyse(page)
    rows = sorted(row.y for block in plain.blocks for row in block.rows)
    middle = len(rows) // 2
    line = _horizontal((rows[middle - 1] + rows[middle]) / 2, width)
    assert _blocks_across(plain, line) >= 1
    assert _blocks_across(_analyse(page, [line]), line) == 0


def test_curved_horizontal_barrier_splits_block() -> None:
    """То же с дугой: прогиб меньше межстрочного интервала, линейка всё время между рядами."""
    page = column_page(columns=1)
    width, height = _work_size(page)
    plain = _analyse(page)
    rows = sorted(row.y for block in plain.blocks for row in block.rows)
    middle = len(rows) // 2
    gap = rows[middle] - rows[middle - 1]
    line = _horizontal(rows[middle - 1] + 0.3 * gap, width, sag=0.4 * gap)
    assert _blocks_across(plain, line) >= 1
    assert _blocks_across(_analyse(page, [line]), line) == 0


def test_geometry_crosses_and_between_rows() -> None:
    """Пересечение отрезка с ломаной и «линейка между рядами» на изогнутой линейке."""
    barrier = BarrierLines.of([[(10, 0), (14, 50), (10, 100)], [(0, 200), (100, 212), (200, 200)]])
    assert barrier.crosses((0, 50), (30, 50))
    assert not barrier.crosses((0, 50), (8, 50))
    assert barrier.between_rows(0, 200, 190, 230)
    assert not barrier.between_rows(0, 200, 213, 230)
    assert len(barrier.separators()) == 2


def test_in_area_turns_with_the_zone() -> None:
    """Перенос в выпрямленную вырезку: сдвиг на угол области и поворот как у ``_forward_points``."""
    barrier = BarrierLines.of([[(110, 205), (150, 205)]])
    area = barrier.in_area((100, 200, 300, 400), 90, _forward_points)
    expected = _forward_points(np.array([[10.0, 5.0], [50.0, 5.0]]), (200, 200), 90)
    assert np.allclose(area.lines[0], expected)
    assert barrier.in_area((400, 400, 500, 500), 0, _forward_points).empty


def _zigzag_page() -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Колонка строк из двух слов с пробелом в разных местах и ломаная линейка через эти пробелы.

    Пробел шире смыкания RLSA (30 px рендера = 15 px рабочей копии), но обычный межсловный: без
    барьера слова сцепляются в строку. Пробелы гуляют по x, поэтому межколонника не возникает, а
    линейка идёт зигзагом — так проверяется сцепка кусков и сращение осей в ряд, а не разрез маски.

    Returns:
        Страница 300 dpi и ломаная в пикселях рабочей копии.
    """
    import random

    from tests.ocr_utils.page_layout.orientation.synthetic import GLYPH_H, LINE_STEP, MARGIN, paper
    from tests.ocr_utils.page_layout.text_blocks.synthetic import _draw_line

    image = paper((1400, 1200))
    generator = random.Random(0)
    points: list[tuple[float, float]] = []
    offsets = [0, 70, -60, 40, -30, 80, -70, 20, -40, 60, -20, 50, -50, 30, -10]
    for row, y in enumerate(range(MARGIN, 1400 - MARGIN - GLYPH_H, LINE_STEP)):
        gap_x = 600 + offsets[row % len(offsets)]
        # Две половины строки набором из «букв»; каждая прижата к пробелу (выключка по формату).
        _draw_line(image, generator, MARGIN, gap_x - 15, y, True)
        _draw_line(image, generator, gap_x + 15, 1200 - MARGIN, y, True)
        # Узел ломаной в середине пробела на уровне строки и над ней (между строками — переход).
        points.append((gap_x * SCALE, (y - LINE_STEP / 2 + GLYPH_H) * SCALE))
        points.append((gap_x * SCALE, (y + GLYPH_H) * SCALE))
    return image, points


def test_barrier_through_word_gaps_stops_linking() -> None:
    """Линейка зигзагом через межсловные пробелы: без неё строки целые, с ней ни одна не пересекает."""
    page, line = _zigzag_page()
    plain = _analyse(page)
    assert _crossing_axes(plain, line) > 5
    split = _analyse(page, [line])
    assert _crossing_axes(split, line) == 0
    # И ряды блока не срастаются через линейку: ни один ряд не накрывает обе половины.
    for block in split.blocks:
        for row in block.rows:
            assert not (row.x0 < 400 * SCALE and row.x1 > 800 * SCALE)
