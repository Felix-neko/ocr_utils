"""Сводка движков по выборке: провалы, дорисовка, краска в дорисовке, согласие масок листа между движками (IoU)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def summarize_runs(runs_dir: Path, names: list[str]) -> pd.DataFrame:
    """Таблица мер по прогонам.

    Args:
        runs_dir: Папка ``runs``.
        names: Имена прогонов.

    Returns:
        DataFrame: строка на прогон — число провалов (лист < 0.6 кадра), медианы и p90 дорисовки, число кадров
        с краской в дорисовке > 0.002, p90 трапеции, медиана времени.
    """
    rows = []
    for name in names:
        d = pd.read_csv(runs_dir / name / "summary.csv")
        rows.append(
            dict(
                run=name,
                frames=len(d),
                fails=int((d.paper_frac < 0.6).sum()),
                fill_med=d.fill_frac.median(),
                fill_p90=d.fill_frac.quantile(0.9),
                ink_fill_frames=int((d.ink_in_fill > 0.002).sum()),
                ink_fill_max=d.ink_in_fill.max(),
                keystone_p90=d.keystone_deg.quantile(0.9),
                sec_med=d.seconds.median(),
            )
        )
    return pd.DataFrame(rows)


def paper_iou(runs_dir: Path, a: str, b: str) -> pd.Series:
    """IoU масок листа двух прогонов по кадрам.

    Args:
        runs_dir: Папка ``runs``.
        a: Первый прогон.
        b: Второй прогон.

    Returns:
        Series ключ кадра → IoU.
    """
    out = {}
    for path in sorted((runs_dir / a / "zones").rglob("*_paper.png")):
        rel = path.relative_to(runs_dir / a / "zones")
        other = runs_dir / b / "zones" / rel
        if not other.exists():
            continue
        ma = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
        mb = cv2.imread(str(other), cv2.IMREAD_GRAYSCALE) > 0
        out[str(rel).removesuffix("_paper.png")] = float((ma & mb).sum() / max((ma | mb).sum(), 1))
    return pd.Series(out)
