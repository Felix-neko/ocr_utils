"""Тесты порчи черт по всем буквам: сборка наклонных строк и порча, видимая только относительно всей полосы."""

import numpy as np
import pandas as pd

from research.spread_defects.lines import Glyphs
from research.spread_defects.page_glyphs import cell_maps, link_rows, side_summary


def _glyphs(rows: int, per_row: int, slope: float) -> Glyphs:
    """Сетка «букв» 20×24 px: ``rows`` строк по ``per_row`` букв, строки с наклоном ``slope`` (dy/dx), шаг строк 50 px."""
    x0, y0 = [], []
    for r in range(rows):
        for k in range(per_row):
            x = 100 + 30 * k
            x0.append(x)
            y0.append(int(100 + 50 * r + slope * x))
    x0, y0 = np.array(x0), np.array(y0)
    w, h = np.full(len(x0), 20), np.full(len(x0), 24)
    return Glyphs(x0, y0, w, h, x0 + 10.0, y0 + 12.0, w * h, 24.0)


def test_link_rows_tilted_lines() -> None:
    """Наклонные строки (за длину строки уходят на 2 шага) собираются по одной, а не слипаются по высоте."""
    glyphs = _glyphs(rows=6, per_row=40, slope=0.08)
    rows = link_rows(glyphs, np.arange(len(glyphs.x0)))
    assert len(np.unique(rows)) == 6
    assert all(len(np.unique(rows[r * 40 : (r + 1) * 40])) == 1 for r in range(6))


def test_damaged_corner_in_cells_and_runs() -> None:
    """Угол полосы с вдвое меньшей долей ядра попадает в клетки ниже 0.6 и в худший отрезок строк."""
    rng = np.random.default_rng(0)
    rows = []
    for row in range(20):
        for k in range(40):
            damaged = row >= 16 and k >= 30
            core = 0.3 if damaged else 0.6
            rows.append(
                {"side": "R", "row": row, "cx": 100.0 + 30 * k, "cy": 100.0 + 50 * row, "h": 24,
                 "core_share": core + rng.normal(0, 0.01), "pieces": 3 if damaged else 1}
            )  # fmt: skip
    table = pd.DataFrame(rows)
    table["core_page"] = table.core_share / table.core_share.median()
    core, broken, cell = cell_maps(table, (1200, 1400), 24.0)
    assert np.nanmin(core) < 0.6
    assert np.nanmax(broken) > 0.5
    summary = side_summary(table)
    assert summary.worst_broken.iloc[0] > 0.2
