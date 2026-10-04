"""Сводка третьего прохода по кадрам: сжатие и порча черт по худшей четверти, расфокус по глифам, пересвет, буквы под переплётом.

Вход — таблицы строк кадра (``lines/<выпуск>/<кадр>.npz``, ``glyph_quality.lines_table``): по строке у сгиба
отношения признаков концевых букв к опорным буквам той же строки, положение концевой буквы и уровни опорных букв.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.spread_defects.glyph_quality import FEATURES

# Тело полосы по высоте для деления на четверти, доли высоты кадра.
BODY_TOP, BODY_BOTTOM = 0.10, 0.92
QUARTERS = 4
# Четверть полосы учитывается, если в ней строк у края колонки не меньше.
MIN_QUARTER_LINES = 4
# «Текст у самого сгиба»: концевая буква ближе стольких высот буквы к линии сгиба.
TOUCH = 0.5

# Строка учитывается, если её концевая буква не дальше стольких высот буквы от сгиба.
EDGE_MAX = 6.0
# Признаки, у которых порча — рост (остальные — падение).
GROWING = ("pieces", "midtone")


def load_lines(path: Path) -> tuple[pd.DataFrame, dict]:
    """Таблица строк кадра и служебные величины.

    Args:
        path: npz третьего прохода.

    Returns:
        (строки кадра; словарь height, width, fold_ink, glyph_h). Таблица пуста, если строк нет.
    """
    with np.load(path) as data:
        meta = {k: float(data[k]) for k in ("height", "width", "fold_ink", "frame_glyph_h")}
        if "side" not in data:
            return pd.DataFrame(), meta
        columns = ["side", "cy", "edge", "at_edge", "glyph_h", "ref_depth", "ref_paper", *FEATURES]
        table = pd.DataFrame({c: data[c] for c in columns})
    return table, meta


def frame_scores(table: pd.DataFrame, height: float) -> dict:
    """Признаки кадра из таблицы строк.

    Args:
        table: Строки кадра.
        height: Высота кадра, px.

    Returns:
        Словарь: для каждого признака ``FEATURES`` — худшая четверть (``q_<имя>``), её сторона и номер
        (``q_<имя>_at``), и медиана худшей стороны (``s_<имя>``); пересвет — медианная глубина опорных букв
        по сторонам (``L_depth``, ``R_depth``, ``depth``); буквы под переплётом — по сторонам 25-й процентиль
        расстояния концов строк до сгиба в высотах буквы (``L_e25``, ``R_e25``) и доля строк у самого сгиба
        (``L_touch``, ``R_touch``).
    """
    out: dict = {}
    if table.empty:
        return out
    table = table.copy()
    table["q"] = np.clip(((table.cy / height - BODY_TOP) / (BODY_BOTTOM - BODY_TOP) * QUARTERS).astype(int), 0, 3)
    # Порча у корешка — только у строк, подходящих к сгибу (не дальше ``EDGE_MAX`` высот буквы): у оглавлений
    # текст кончается за колонкой номеров, далеко от сгиба, и мерить там «порчу у корешка» нечего.
    edge = table[table.at_edge & (table.edge <= EDGE_MAX * table.glyph_h)]
    groups = edge.groupby(["side", "q"])
    counts = groups.size()
    good = counts[counts >= MIN_QUARTER_LINES].index
    quarters = groups[list(FEATURES)].median().loc[good] if len(good) else None
    sides = edge.groupby("side")[list(FEATURES)].median()
    for name in FEATURES:
        worst = (lambda s: s.idxmax()) if name in GROWING else (lambda s: s.idxmin())
        if quarters is not None and quarters[name].notna().any():
            at = worst(quarters[name].dropna())
            out[f"q_{name}"] = float(quarters.loc[at, name])
            out[f"q_{name}_at"] = f"{at[0]}{at[1] + 1}"
        if sides[name].notna().any():
            out[f"s_{name}"] = float(sides[name].dropna().max() if name in GROWING else sides[name].dropna().min())
    for side, part in table.groupby("side"):
        e = part.edge / part.glyph_h
        out[f"{side}_e25"] = float(np.percentile(e, 25))
        out[f"{side}_touch"] = float((e < TOUCH).mean())
        out[f"{side}_lines"] = int(len(part))
        out[f"{side}_depth"] = float(part.ref_depth.median())
        out[f"{side}_paper"] = float(part.ref_paper.median())
    depths = [out[k] for k in ("L_depth", "R_depth") if k in out]
    out["depth"] = float(min(depths)) if depths else np.nan
    return out


def rank(out_root: Path) -> pd.DataFrame:
    """Признаки всех кадров третьего прохода.

    Args:
        out_root: Корень выхода стенда (``pass3.csv``, ``lines/``).

    Returns:
        Таблица кадров: ключ, путь, выпуск и признаки ``frame_scores``; пересвет дополнительно в долях медианы
        выпуска (``depth_rel``).
    """
    frames = pd.read_csv(out_root / "pass3.csv", dtype={"issue": str})
    rows = []
    for r in frames.itertuples():
        path = out_root / "lines" / r.issue / f"{Path(r.path).stem}.npz"
        if not path.exists():
            continue
        table, meta = load_lines(path)
        rows.append(
            {
                "key": r.key,
                "issue": r.issue,
                "index": r.index,
                "path": r.path,
                **meta,
                **frame_scores(table, meta["height"]),
            }
        )
    scores = pd.DataFrame(rows)
    scores["depth_rel"] = scores.depth / scores.groupby("issue").depth.transform("median")
    return scores
