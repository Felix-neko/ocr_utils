"""Оценки детекторов символов у кандидатов: CRAFT — максимум карты «регион» по пикселям компоненты, pero — вероятность базовой линии в окне ряда (как ``edge_guard.component_scores``)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ocr_utils.page_layout.text_blocks.edge_guard import GlyphEngine, component_scores
from research.pseudo_glyphs.candidates import components_of
from ocr_utils.units import mm_to_px_float
from ocr_utils.page_layout.text_blocks import RENDER_DPI


def row_geometry(row) -> tuple[float, float] | None:
    """Середина ряда и x-высота кандидата ряда, px 300 dpi (восстановлены из признаков ``above_xh``, ``x_h_mm``).

    Args:
        row: Кандидат (строка таблицы).

    Returns:
        ``(середина, x-высота)`` или ``None`` у кандидата края (ряда нет).
    """
    if row.kind == "edge" or pd.isna(row.x_h_mm):
        return None
    x_h = mm_to_px_float(row.x_h_mm, RENDER_DPI)
    row_h = mm_to_px_float(row.row_h_mm, RENDER_DPI)
    # above_xh = (верх полосы ряда − y0) / x_h, верх полосы = середина − высота/2.
    return row.y0 + row.above_xh * x_h + 0.5 * row_h, x_h


def page_scores(gray_png: Path, maps_dir: Path, map_key: str, part: pd.DataFrame) -> pd.DataFrame:
    """Оценки CRAFT и pero всех кандидатов одной страницы.

    Args:
        gray_png: Серый 300 dpi страницы (тот, на котором искались кандидаты).
        maps_dir: Корень карт (``<детектор>/<ключ>.png``).
        map_key: Ключ карты страницы.
        part: Кандидаты страницы.

    Returns:
        ``cid, craft, pero`` (pero — ``nan``, если воздержался: кандидат вне полосы ряда или кандидат края).
    """
    gray = cv2.imread(str(gray_png), cv2.IMREAD_GRAYSCALE)
    comps = components_of(gray)
    maps = {}
    for engine in (GlyphEngine.CRAFT, GlyphEngine.PERO):
        path = maps_dir / engine.value / f"{map_key}.png"
        if path.exists():
            heat = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            # Карта могла выйти другого размера (кратность сети) — приводим к серому.
            if heat.shape != gray.shape:
                heat = cv2.resize(heat, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
            maps[engine] = heat
    out = []
    for row in part.itertuples():
        geometry = row_geometry(row)
        # Кандидату края ряд неизвестен: pero не голосует (середина ряда — заведомо вне полосы).
        scores = component_scores(comps.labels, comps.stats, int(row.comp), maps,
                                  geometry if geometry is not None else (-1e9, 1.0))  # fmt: skip
        out.append({"cid": row.cid, "craft": scores.get(GlyphEngine.CRAFT, np.nan),
                    "pero": scores.get(GlyphEngine.PERO, np.nan)})  # fmt: skip
    return pd.DataFrame(out)


__all__ = ["page_scores", "row_geometry"]
