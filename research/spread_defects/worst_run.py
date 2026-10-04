"""Порча черт по худшему отрезку строк: медиана признака по ``WINDOW`` подряд идущим строкам у сгиба, худший отрезок кадра.

Медиана по четверти полосы (~10–15 строк) сглаживает порчу, сосредоточенную в нескольких последних строках
(1974/04 IMG_0035: «фондоемкость», «В легкой», «ткацкий» — 0.77 по четверти). Отрезок из 4 строк подряд ловит
такую локальную порчу, а медиана внутри отрезка не даёт сработать одной случайной строке.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.spread_defects.pass3_rank import EDGE_MAX

# Строк в отрезке.
WINDOW = 4


def worst_run(table: pd.DataFrame, feature: str = "core_share") -> tuple[float, str, float]:
    """Худший отрезок строк кадра.

    Args:
        table: Строки кадра (npz третьего прохода): side, cy, edge, at_edge, glyph_h и признак.
        feature: Признак (порча — падение).

    Returns:
        (медиана признака в худшем отрезке, сторона, высота центра отрезка, px); NaN, если строк мало.
    """
    best = (np.nan, "", np.nan)
    use = table[table.at_edge & (table.edge <= EDGE_MAX * table.glyph_h)].dropna(subset=[feature])
    for side, part in use.groupby("side"):
        part = part.sort_values("cy")
        values, ys = part[feature].to_numpy(), part.cy.to_numpy()
        for start in range(0, len(values) - WINDOW + 1):
            score = float(np.median(values[start : start + WINDOW]))
            if not np.isfinite(best[0]) or score < best[0]:
                best = (score, side, float(np.mean(ys[start : start + WINDOW])))
    return best


def rank(out_root: Path) -> pd.DataFrame:
    """Худший отрезок по всем кадрам третьего прохода.

    Args:
        out_root: Корень выхода стенда (``pass3_scores.csv``, ``lines/``).

    Returns:
        Таблица: key, path, run_core (медиана доли ядра в худшем отрезке), run_side, run_y, run_y_share (доля высоты).
    """
    scores = pd.read_csv(out_root / "pass3_scores.csv", dtype={"issue": str})
    rows = []
    for r in scores.itertuples():
        path = out_root / "lines" / r.issue / f"{Path(r.path).stem}.npz"
        with np.load(path) as data:
            if "side" not in data:
                continue
            table = pd.DataFrame({c: data[c] for c in ("side", "cy", "edge", "at_edge", "glyph_h", "core_share")})
            height = float(data["height"])
        value, side, y = worst_run(table)
        rows.append({"key": r.key, "path": r.path, "run_core": value, "run_side": side, "run_y": y,
                     "run_y_share": y / height if np.isfinite(y) else np.nan})  # fmt: skip
    return pd.DataFrame(rows)
