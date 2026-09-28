"""Точность правила «знак или мусор» по размеченным глазами крайним компонентам (без перепрогона страниц)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research.text_block_specks.filters import End, EndComponent
from research.text_block_specks.rules import SIGNS, Kind, Thresholds, kind_of

# Признаки компонента в JSON выборки (``components.collect``) — поля :class:`EndComponent` без бокса.
FEATURES = ["x_h", "w", "h", "area", "fill", "gap", "bottom", "top", "stroke"]


def verdicts(components: pd.DataFrame, thresholds: Thresholds = Thresholds()) -> pd.Series:
    """Вердикт правила по каждой строке таблицы компонентов (значения :class:`Kind`)."""
    out = []
    for row in components.itertuples():
        component = EndComponent(box=tuple(row.box), **{name: getattr(row, name) for name in FEATURES})
        out.append(kind_of(component, End(row.end), thresholds).value)
    return pd.Series(out, index=components.index)


def score(components: pd.DataFrame, kinds: pd.Series) -> dict:
    """Меры по размеченным компонентам: мусор (S) и знаки (P), «спорно» считается отдельно.

    Returns:
        ``S_убрано``, ``S_спорно``, ``S_оставлено``, ``P_убрано`` (вред), ``P_спорно``, ``P_оставлено``.
    """
    signs = {kind.value for kind in SIGNS}
    removed = kinds.isin([Kind.SPECK.value, Kind.MARK.value])
    unsure = kinds == Kind.UNSURE.value
    kept = kinds.isin(signs)
    out = {}
    for label in ("S", "P"):
        mask = components.label == label
        out[f"{label}_убрано"] = int((mask & removed).sum())
        out[f"{label}_спорно"] = int((mask & unsure).sum())
        out[f"{label}_оставлено"] = int((mask & kept).sum())
    return out


def load(directory: Path) -> pd.DataFrame:
    """Выборка компонентов с метками глазами (``components.json`` + ``comp_labels.csv``), без меток U."""
    records = pd.DataFrame(json.loads((directory / "components.json").read_text()))
    labels = pd.read_csv(directory / "comp_labels.csv")
    table = records.merge(labels, on="id")
    return table[table.label.isin(["S", "P"])].reset_index(drop=True)


__all__ = ["load", "score", "verdicts"]
