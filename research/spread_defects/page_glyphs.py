"""Порча черт по всем буквам полосы: мера ``glyph_quality.measure_glyph`` не только у сгиба, а на каждой букве.

Нынешний детектор (``pass3``) сравнивает ~3 буквы у сгиба с опорными буквами той же строки и потому не видит порчу
посреди полосы: выгоревший угол ксерокопии, скотч, пятно. Здесь каждая буква полосы меряется той же функцией, а
сравнивается двумя способами:

* ``core_line`` — доля ядра штриха к медиане своей строки (как у ``pass3``: печать, кегль и свет у строки общие);
  ловит порчу части строки, но слеп к порче, накрывшей строку целиком;
* ``core_page`` — к медиане всей полосы; ловит выгоревший низ или угол страницы, где портятся строки целиком.

Плюс ``pieces`` — число кусков буквы на уровне половины глубины краски (broken character factor): разрывы
тонких штрихов, типичные для бледной ксерокопии.
"""

from __future__ import annotations

from dataclasses import fields

import cv2
import numpy as np
import pandas as pd

from research.spread_defects.glyph_quality import GlyphMeasure, measure_glyph
from research.spread_defects.lines import LEFT, RIGHT, Glyphs

# Соседка по строке: отклонение центра по высоте не больше этой доли высоты буквы, зазор не больше LINK_GAP высот.
ROW_DY = 0.45
LINK_GAP = 2.6
# Буквы не ближе этого к сгибу, px полного кадра (тень и изгиб листа меряет ``pass3``).
FOLD_SKIP = 120
# Клетка тепловой карты — столько высот буквы по каждой оси.
CELL_GLYPHS = 6
# Клетка карты в счёт, если в ней не меньше стольких букв.
CELL_MIN = 12
# Строка в счёт для худшего отрезка, если в ней не меньше стольких букв.
ROW_MIN = 10
# Отрезок строк подряд для худшего места (как ``worst_run.WINDOW``).
WINDOW = 4


def _root(parent: np.ndarray, k: int) -> int:
    """Корень множества в union-find со сжатием пути.

    Args:
        parent: Массив родителей (меняется на месте — сжатие пути).
        k: Элемент.

    Returns:
        Корень множества элемента.
    """
    while parent[k] != k:
        parent[k] = parent[parent[k]]
        k = parent[k]
    return k


def link_rows(glyphs: Glyphs, idx: np.ndarray) -> np.ndarray:
    """Строки полосы: цепочки букв, где каждая связана с ближайшей соседкой справа.

    Сортировка по высоте центра не годится: строки на снимке наклонены и изогнуты, и по высоте соседние строки
    перекрываются. Соседка — буква правее, с зазором не больше ``LINK_GAP`` высот и отклонением центра по высоте
    не больше ``ROW_DY`` высоты; связи объединяются (union-find).

    Args:
        glyphs: Буквы кадра.
        idx: Индексы букв одной полосы.

    Returns:
        int[len(idx)] — номер строки каждой буквы (номера по возрастанию медианной высоты строки).
    """
    from scipy.spatial import cKDTree

    hgt = glyphs.glyph_h
    centers = np.column_stack([glyphs.cx[idx], glyphs.cy[idx]])
    right_edge = (glyphs.x0 + glyphs.w)[idx].astype(float)
    left_edge = glyphs.x0[idx].astype(float)
    parent = np.arange(len(idx))
    tree = cKDTree(centers)
    for k, near in enumerate(tree.query_ball_point(centers, r=(LINK_GAP + 2.0) * hgt)):
        near = np.array([j for j in near if j != k])
        if len(near) == 0:
            continue
        gap = left_edge[near] - right_edge[k]
        dy = np.abs(centers[near, 1] - centers[k, 1])
        ok = (centers[near, 0] > centers[k, 0]) & (gap < LINK_GAP * hgt) & (dy < ROW_DY * hgt)
        if not ok.any():
            continue
        j = near[ok][np.argmin(centers[near[ok], 0])]
        parent[_root(parent, k)] = _root(parent, j)
    roots = np.array([_root(parent, k) for k in range(len(idx))])
    # Перенумерация строк по медианной высоте.
    order = pd.Series(centers[:, 1]).groupby(roots).median().sort_values()
    rank = {root: n for n, root in enumerate(order.index)}
    return np.array([rank[r] for r in roots])


def glyph_table(gray: np.ndarray, glyphs: Glyphs, fold: np.ndarray, edge_threshold: float) -> pd.DataFrame:
    """Признаки всех букв обеих полос кадра.

    Два прохода: сначала глубина краски каждой буквы по ней самой, затем пороги уровней краски — от медианной
    глубины своей строки (как у ``pass3``, где опора — буквы той же строки).

    Args:
        gray: Полутоновый кадр float32 полного разрешения.
        glyphs: Буквы кадра (``lines.find_glyphs``).
        fold: Прямая сгиба ``x = k*y + b``.
        edge_threshold: Порог края для DOM (``dom.gradient_threshold`` по кадру).

    Returns:
        Таблица по буквам: side, row (номер строки на стороне), cx, cy, h, признаки ``GlyphMeasure``,
        ``core_line``, ``core_page``.
    """
    fold_x = np.polyval(fold, glyphs.cy)
    side = np.where(glyphs.cx < fold_x, LEFT, RIGHT)
    keep = np.abs(glyphs.cx - fold_x) > FOLD_SKIP
    rows = []
    for s in (LEFT, RIGHT):
        idx = np.nonzero(keep & (side == s))[0]
        if len(idx) == 0:
            continue
        # Строки: буква связывается с ближайшей соседкой справа на той же строке, связные цепочки — строки.
        row_of = link_rows(glyphs, idx)
        # Проход 1: глубина краски по самой букве.
        depth = np.array([measure_glyph(gray, glyphs, i, np.nan, edge_threshold).depth for i in idx])
        for row in np.unique(row_of):
            members = idx[row_of == row]
            ref_depth = float(np.median(depth[row_of == row]))
            for i in members:
                # Проход 2: пороги краски — от глубины строки.
                measure = measure_glyph(gray, glyphs, i, ref_depth, edge_threshold)
                rows.append(
                    {
                        "side": s,
                        "row": int(row),
                        "cx": float(glyphs.cx[i]),
                        "cy": float(glyphs.cy[i]),
                        "h": int(glyphs.h[i]),
                        **{f.name: getattr(measure, f.name) for f in fields(GlyphMeasure)},
                    }
                )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["core_line"] = table.core_share / table.groupby(["side", "row"]).core_share.transform("median")
    table["core_page"] = table.core_share / table.groupby("side").core_share.transform("median")
    return table


def side_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Сводка по полосам: медианы и худшие отрезки из ``WINDOW`` строк подряд.

    Args:
        table: Таблица ``glyph_table``.

    Returns:
        По строке на сторону: glyphs, rows (строк от ``ROW_MIN`` букв), core_share (медиана), broken_share (доля
        букв из 2+ кусков), worst_core (медиана ``core_page`` худшего отрезка строк) и его высота worst_core_y, px,
        worst_broken (доля рассыпавшихся букв в худшем отрезке) и worst_broken_y.
    """
    out = []
    for side, part in table.groupby("side"):
        per_row = part.groupby("row").agg(
            n=("cy", "size"),
            cy=("cy", "median"),
            core=("core_page", "median"),
            broken=("pieces", lambda v: float((v >= 2).mean())),
        )
        per_row = per_row[per_row.n >= ROW_MIN].sort_values("cy")
        windows = [per_row.iloc[k : k + WINDOW] for k in range(0, len(per_row) - WINDOW + 1)]
        core_runs = [(float(w.core.median()), float(w.cy.mean())) for w in windows] or [(np.nan, np.nan)]
        broken_runs = [(float(w.broken.median()), float(w.cy.mean())) for w in windows] or [(np.nan, np.nan)]
        worst_core = min(core_runs, key=lambda r: r[0] if np.isfinite(r[0]) else np.inf)
        worst_broken = max(broken_runs, key=lambda r: r[0] if np.isfinite(r[0]) else -np.inf)
        out.append(
            {
                "side": side,
                "glyphs": len(part),
                "rows": len(per_row),
                "core_share": round(float(part.core_share.median()), 3),
                "broken_share": round(float((part.pieces >= 2).mean()), 3),
                "worst_core": round(worst_core[0], 3),
                "worst_core_y": round(worst_core[1]),
                "worst_broken": round(worst_broken[0], 3),
                "worst_broken_y": round(worst_broken[1]),
            }
        )
    return pd.DataFrame(out)


def cell_maps(table: pd.DataFrame, shape: tuple[int, int], glyph_h: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Тепловые карты по клеткам ``CELL_GLYPHS`` высот буквы: медиана ``core_page`` и доля букв из 2+ кусков.

    Args:
        table: Таблица ``glyph_table``.
        shape: (высота, ширина) кадра, px.
        glyph_h: Медианная высота буквы кадра, px.

    Returns:
        (core[gy, gx], broken[gy, gx] — NaN в клетках, где букв строк от ``ROW_MIN`` букв меньше ``CELL_MIN``;
        сторона клетки, px).
    """
    cell = max(int(CELL_GLYPHS * glyph_h), 16)
    gy, gx = shape[0] // cell + 1, shape[1] // cell + 1
    core = np.full((gy, gx), np.nan, np.float32)
    broken = np.full((gy, gx), np.nan, np.float32)
    # Только буквы настоящих строк: обрез книжного блока, линейки и сор дают одиночные «буквы» с тонким штрихом.
    table = table[table.groupby(["side", "row"]).cy.transform("size") >= ROW_MIN]
    keys = (table.cy // cell).astype(int) * gx + (table.cx // cell).astype(int)
    for key, part in table.groupby(keys):
        if len(part) < CELL_MIN:
            continue
        core[key // gx, key % gx] = float(part.core_page.median())
        broken[key // gx, key % gx] = float((part.pieces >= 2).mean())
    return core, broken, cell


def heat_overlay(rgb: np.ndarray, core: np.ndarray, cell: int, title: list[str]) -> np.ndarray:
    """Кадр с полупрозрачной заливкой клеток по поясам ``core_page`` и легендой в поле.

    Пояса — как у ``pass3`` (по худшим строкам у сгиба): сильная порча < 0.45, средняя 0.45–0.60,
    слабая 0.60–0.75.

    Args:
        rgb: Кадр RGB.
        core: Карта медианы ``core_page`` по клеткам.
        cell: Сторона клетки, px.
        title: Строки шапки.

    Returns:
        BGR-картинка, ужатая до ~2000 px по ширине, с шапкой и легендой в белых полях.
    """
    from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed

    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    layer = canvas.copy()
    bands = [(0.45, (0, 0, 220), "сильная порча: ядро < 0.45 медианы полосы"),
             (0.60, (0, 140, 255), "средняя: 0.45–0.60"),
             (0.75, (0, 210, 230), "слабая: 0.60–0.75")]  # fmt: skip
    for (row, col), value in np.ndenumerate(core):
        if not np.isfinite(value):
            continue
        for limit, colour, _ in bands:
            if value < limit:
                cv2.rectangle(layer, (col * cell, row * cell), ((col + 1) * cell, (row + 1) * cell), colour, -1)
                break
    alpha = 0.4
    cv2.addWeighted(layer, alpha, canvas, 1 - alpha, 0, canvas)
    scale = 2000 / canvas.shape[1]
    canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    legend = [LegendEntry(text, colour, alpha=alpha, style=SampleStyle.BOX) for _, colour, text in bands]
    return framed(canvas, title, legend)
