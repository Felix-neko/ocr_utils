"""Оценка правил по разметке: полнота по классам псевдоглифов, вред по классам настоящих знаков (на всей разметке и на случайной выборке концов рядов), подбор порога CRAFT поверх правил при вреде не больше заданного."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from research.pseudo_glyphs.rules import RuleParams, reasons, relative_height
from research.pseudo_glyphs.select import PSEUDO, TRUE, Label


def labelled(candidates: pd.DataFrame, labels: pd.DataFrame, scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """Размеченные кандидаты с признаками (и оценками детекторов).

    Args:
        candidates: ``candidates.csv``.
        labels: ``sets/labels.csv`` (``key, comp, label, band``).
        scores: ``scores.csv`` (``cid, craft, pero``) или ``None``.

    Returns:
        Кандидаты (одна строка на компоненту) с колонками ``label``, ``band``, ``group`` (``pseudo``/``true``/``other``).
    """
    # Высота в буквах страницы — по всем кандидатам страницы, до отбора размеченных.
    candidates = candidates.assign(h_rel=relative_height(candidates))
    first = candidates.drop_duplicates(["key", "comp"])
    if scores is not None:
        first = first.merge(scores, on="cid", how="left")
    frame = first.merge(labels[["key", "comp", "label", "band", "note"]], on=["key", "comp"], how="inner")
    pseudo = {item.value for item in PSEUDO}
    true = {item.value for item in TRUE}
    frame["group"] = np.where(frame.label.isin(pseudo), "pseudo", np.where(frame.label.isin(true), "true", "other"))
    return frame


def summary(frame: pd.DataFrame, fired: pd.Series) -> dict:
    """Сводка одного варианта: полнота псевдоглифов, вред на знаках (вся разметка и случайный пояс).

    Args:
        frame: Размеченные кандидаты (:func:`labelled`).
        fired: Булева серия «правило сработало».

    Returns:
        Словарь мер.
    """
    pseudo, true = frame.group == "pseudo", frame.group == "true"
    random_true = true & (frame.band == "random")
    # Зона риска: знаки, до которых правила вообще дотягиваются, — у края (до 4 мм) или далеко от ряда.
    far = (frame.gap_rel.fillna(0.0) > 1.0) & (frame.gap_mm.fillna(0.0) >= 2.0)
    risk_true = true & ((frame.edge_mm <= 4.0) | far)
    out = {
        "pseudo": int(pseudo.sum()),
        "recall": float(fired[pseudo].mean()) if pseudo.any() else float("nan"),
        "true": int(true.sum()),
        "harm": int(fired[true].sum()),
        "harm_share": float(fired[true].mean()) if true.any() else float("nan"),
        "risk_true": int(risk_true.sum()),
        "risk_harm_share": float(fired[risk_true].mean()) if risk_true.any() else float("nan"),
        "random_true": int(random_true.sum()),
        "random_harm": int(fired[random_true].sum()),
    }
    for label in Label:
        mask = frame.label == label.value
        if mask.any():
            out[f"{label.value}_n"] = int(mask.sum())
            out[f"{label.value}_fired"] = int(fired[mask].sum())
    return out


def by_class(frame: pd.DataFrame, fired: pd.Series, reason: pd.Series) -> pd.DataFrame:
    """Таблица по классам: число, сработало, доля, причины."""
    rows = []
    for label, part in frame.groupby("label"):
        hit = fired[part.index]
        reasons_of = reason[part.index][hit].value_counts().to_dict()
        rows.append({"label": label, "group": part.group.iloc[0], "n": len(part), "fired": int(hit.sum()),
                     "share": float(hit.mean()), "reasons": ", ".join(f"{k}:{v}" for k, v in reasons_of.items())})  # fmt: skip
    return pd.DataFrame(rows).sort_values(["group", "label"])


def craft_sweep(frame: pd.DataFrame, params: RuleParams, thresholds: np.ndarray) -> pd.DataFrame:
    """Правила + CRAFT для спорных кандидатов (у края или далеко от ряда) по сетке порогов CRAFT.

    Args:
        frame: Размеченные кандидаты с колонкой ``craft``.
        params: Пороги правил.
        thresholds: Пороги CRAFT.

    Returns:
        Сводки по порогам (:func:`summary`) с колонкой ``craft``.
    """
    rows = []
    for t in thresholds:
        fired = reasons(frame, replace(params, craft_doubt=float(t))) != ""
        rows.append({"craft": float(t), **summary(frame, fired)})
    return pd.DataFrame(rows)


def load(out_dir: Path, labels_csv: Path) -> pd.DataFrame:
    """Размеченные кандидаты прогона стенда (с оценками, если ``scores.csv`` есть)."""
    candidates = pd.read_csv(out_dir / "candidates.csv")
    scores_path = out_dir / "scores.csv"
    scores = pd.read_csv(scores_path) if scores_path.exists() else None
    return labelled(candidates, pd.read_csv(labels_csv), scores)


__all__ = ["by_class", "craft_sweep", "labelled", "load", "summary"]
