"""Сводка прогона «было | стало»: суммы и квантили мер сторон по режимам, страницы со сменами, закрашенное по причинам."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Режимы в порядке показа: без фильтра без правила 0.3 мм, без фильтра с правилом, фильтр по правилам, с CRAFT.
MODES = ("off", "on", "rules", "rules_craft")
SUM_FIELDS = ("blocks", "rows", "kind_both", "kind_left", "kind_right", "kind_center", "kind_ragged", "aligned_sides",
              "v2_sides", "v2_sides_spine", "v2_side_len_mm")  # fmt: skip
P95_FIELDS = ("side_p95_mm", "resid_p95_mm")


def load_pages(out_dir: Path) -> list[dict]:
    """Все JSON страниц прогона."""
    return [json.loads(p.read_text()) for p in sorted((out_dir / "pages").glob("*/*.json"))]


def summarize(out_dir: Path) -> pd.DataFrame:
    """Сводка по режимам → ``summary.csv`` и ``pages.csv`` (меры по страницам в широком виде).

    Args:
        out_dir: Папка прогона сравнения.

    Returns:
        Таблица «мера × режим»: суммы по страницам для счётчиков, медиана и p90 по страницам для мер p95, число
        закрашенных компонент по причинам.
    """
    pages = load_pages(out_dir)
    rows = []
    for page in pages:
        row = {"key": page["key"], "part": page["part"]}
        for mode in MODES:
            if mode not in page:
                continue
            for field, value in page[mode].items():
                row[f"{mode}.{field}"] = value
            if f"{mode}_drops" in page:
                row[f"{mode}.drops"] = len(page[f"{mode}_drops"])
        rows.append(row)
    wide = pd.DataFrame(rows)
    wide.to_csv(out_dir / "pages.csv", index=False)
    table = {}
    for mode in MODES:
        if f"{mode}.blocks" not in wide:
            continue
        col = {field: float(wide[f"{mode}.{field}"].sum()) for field in SUM_FIELDS}
        for field in P95_FIELDS:
            values = wide[f"{mode}.{field}"].dropna()
            col[f"{field} медиана"] = float(values.median()) if len(values) else np.nan
            col[f"{field} p90"] = float(values.quantile(0.9)) if len(values) else np.nan
        if f"{mode}.drops" in wide:
            col["закрашено компонент"] = float(wide[f"{mode}.drops"].sum())
            col["страниц с закрасом"] = float((wide[f"{mode}.drops"] > 0).sum())
        if mode != "off":
            # Страницы, где сменилась выключка хотя бы одного блока или число выровненных сторон.
            changed = np.zeros(len(wide), bool)
            for field in ("aligned_sides", "v2_sides", "kind_both", "kind_left", "kind_right", "kind_ragged"):
                changed |= wide[f"{mode}.{field}"].to_numpy() != wide[f"off.{field}"].to_numpy()
            col["страниц со сменой сторон/выключки"] = float(changed.sum())
            col["сторон v2 больше / меньше (страниц)"] = (
                f"{int((wide[f'{mode}.v2_sides'] > wide['off.v2_sides']).sum())} / "
                f"{int((wide[f'{mode}.v2_sides'] < wide['off.v2_sides']).sum())}"
            )
        table[mode] = col
    reasons = Counter(d["reason"] for page in pages for d in page.get("rules_drops", []))
    frame = pd.DataFrame(table)
    frame.to_csv(out_dir / "summary.csv")
    (out_dir / "reasons.json").write_text(json.dumps(dict(reasons), ensure_ascii=False))
    return frame


__all__ = ["MODES", "load_pages", "summarize"]
