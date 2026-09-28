"""Чужие движки строк на размеченных концах: тянется ли конец оси движка к соринке или пометке.

Для каждого размеченного конца строки (``candidates.csv`` + ``labels_v2.csv``) на полосах набора берутся
оси движка (``<engines>/pages/<pdf>_pNNN_nogeo_<движок>.json``), проходящие через ординату конца, и крайняя
точка этих осей у нужной стороны. «Недолёт» — насколько ось движка кончается раньше края краски ``ink``
(с соринкой), в мм, за вычетом обычного недолёта этого движка на чистых концах (медиана по меткам L: оси
чужих движков кончаются не на краю краски, а у середины последней буквы или раньше).

Конец с меткой S/M считается «мимо соринки», если недолёт не меньше половины выступа (``|resid_mm|/2``);
конец с меткой P — «знак потерян», если недолёт больше половины его выступа.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ocr_utils.page_layout.text_blocks import WORK_DPI

MM_PER_PX = 25.4 / WORK_DPI
# Ось движка относится к концу строки, если проходит у конца по вертикали не дальше стольких пикселей (у pero и
# kraken ось по базовой линии поднята с ошибкой до 1.2 мм)…
MATCH_Y_PX = 14.0
# …и кончается не дальше стольких мм от края краски в любую сторону.
MATCH_X_MM = 12.0


def engine_end(axes: list[dict], x: float, y: float, side: str) -> float | None:
    """Крайняя точка осей движка у конца строки ``(x, y)``: максимум x у правого конца, минимум — у левого."""
    best = None
    for axis in axes:
        points = np.asarray(axis["points"], dtype=float)
        if points.shape[0] < 2:
            continue
        end = points[-1] if side == "right" else points[0]
        # Ордината оси у её конца, ближайшего к краю строки.
        if abs(end[1] - y) > MATCH_Y_PX or abs(end[0] - x) * MM_PER_PX > MATCH_X_MM:
            continue
        value = float(end[0])
        if best is None or (value > best if side == "right" else value < best):
            best = value
    return best


def shortfalls(ends: pd.DataFrame, engines_dir: Path, engine: str) -> pd.Series:
    """Недолёт оси движка до края краски по каждому концу, мм (NaN — оси у конца нет)."""
    cache: dict[str, list[dict]] = {}
    out = []
    for row in ends.itertuples():
        key = f"{row.pdf[:-4]}_p{int(row.page):03d}_nogeo_{engine}"
        if key not in cache:
            path = engines_dir / "pages" / f"{key}.json"
            cache[key] = json.loads(path.read_text())["axes"] if path.exists() else []
        x = engine_end(cache[key], row.x, row.y, row.side)
        if x is None:
            out.append(np.nan)
            continue
        inward = 1.0 if row.side == "right" else -1.0
        out.append((row.x - x) * inward * MM_PER_PX)
    return pd.Series(out, index=ends.index)


def score(ends: pd.DataFrame, engines_dir: Path, engine: str) -> dict:
    """Сводка движка: обычный недолёт, доля S/M «мимо соринки», доля P с потерянным знаком, концы без оси."""
    short = shortfalls(ends, engines_dir, engine)
    base = float(short[ends.label == "L"].median())
    net = short - base
    half = ends.resid_mm.abs() / 2.0
    bad = ends.label.isin(["S", "M"])
    signs = ends.label == "P"
    return {
        "движок": engine,
        "недолёт L, мм": round(base, 2),
        "S/M с осью": int((bad & short.notna()).sum()),
        "мимо соринки": int((bad & (net >= half)).sum()),
        "P с осью": int((signs & short.notna()).sum()),
        "знак потерян": int((signs & (net > half)).sum()),
        "без оси (S/M)": int((bad & short.isna()).sum()),
    }


def labelled_ends(label_dir: Path, pages_file: Path) -> pd.DataFrame:
    """Размеченные концы строк на полосах набора (метки после перепроверки, ``labels_v2.csv``)."""
    candidates = pd.read_csv(label_dir / "candidates.csv")
    labels = pd.read_csv(label_dir / "labels_v2.csv")
    spec = json.loads(pages_file.read_text())
    keys = {item["key"] for group in ("defect", "clean") for item in spec[group]}
    table = candidates.merge(labels, on="id")
    return table[table.key.isin(keys)].reset_index(drop=True)


__all__ = ["engine_end", "labelled_ends", "score", "shortfalls"]


# Цвета концов осей движков на вырезке (BGR) и порядок подписей.
ENGINE_COLOURS = {
    "ink": (0, 0, 230),
    "pero": (0, 150, 0),
    "paddle6": (200, 120, 0),
    "kraken": (180, 0, 180),
    "surya": (0, 140, 230),
    "eynollah": (120, 120, 120),
}


def engine_crops(ends: pd.DataFrame, engines_dir: Path, pdf_dir: Path, engines: list[str]) -> dict[str, "np.ndarray"]:
    """Вырезки концов строк с концами осей всех движков (цветные штрихи над и под строкой, легенда сверху).

    Args:
        ends: Размеченные концы (``labelled_ends``) с полем ``id``.
        engines_dir: Корень прогона движков.
        pdf_dir: Каталог PDF.
        engines: Какие движки рисовать.

    Returns:
        ``id → вырезка`` BGR.
    """
    import cv2
    import fitz

    from ocr_utils.geometry_regression.render import render_gray
    from research.text_block_specks.sheets import K, end_crop, labelled

    out = {}
    for (pdf, page), items in ends.groupby(["pdf", "page"]):
        with fitz.open(pdf_dir / pdf) as document:
            gray = render_gray(document, int(page) - 1)
        axes = {}
        for engine in engines:
            path = engines_dir / "pages" / f"{pdf[:-4]}_p{int(page):03d}_nogeo_{engine}.json"
            axes[engine] = json.loads(path.read_text())["axes"] if path.exists() else []
        for row in items.itertuples():
            crop = end_crop(gray, {"x": row.x, "y": row.y, "side": row.side, "resid_mm": row.resid_mm})
            # Начало вырезки на рендере (как в ``sheets.end_crop``) и её масштаб на листе.
            from research.text_block_specks.sheets import CROP_HALF_MM, CROP_IN_MM, CROP_OUT_MM, ZOOM

            px = 300 / 25.4
            left = row.x * K - (CROP_IN_MM if row.side == "right" else CROP_OUT_MM) * px
            top = row.y * K - CROP_HALF_MM * px
            for number, engine in enumerate(engines):
                x = engine_end(axes[engine], row.x, row.y, row.side)
                if x is None:
                    continue
                cx = int((x * K - left) * ZOOM)
                cy = int((row.y * K - top) * ZOOM)
                offset = 10 + 7 * number
                cv2.line(crop, (cx, cy - offset - 8), (cx, cy - offset), ENGINE_COLOURS[engine], 3)
            out[row.id] = labelled(crop, f"{row.id} {row.label} {row.key[:7]} {row.key[8:]}")
    return out
