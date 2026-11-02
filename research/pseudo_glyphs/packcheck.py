"""Регрессия правил на размеченных наборах пака-1 (бинаризованные PDF nogeo): кандидаты ``research/edge_marks`` (сор S/M, знаки L) и компоненты ``research/text_block_specks`` (сор S, знаки P) — настоящие знаки не должны попадать под правила."""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import numpy as np
import pandas as pd

from ocr_utils.geometry_regression.render import render_gray
from ocr_utils.page_layout.text_blocks import RENDER_DPI, WORK_DPI
from research.pseudo_glyphs.candidates import page_candidates
from research.pseudo_glyphs.crops import analyze
from research.pseudo_glyphs.rules import RuleParams, reasons

# Бинаризованные полосы пака-1 без коррекции геометрии и разметки прежних стендов.
NOGEO_DIR = Path(
    "/mnt/hotstore/scan_processing/mts/pack1_pdf/full_pdfs_binary_no_bg_brightening_no_geometry_correction"
)
EDGE_MARKS_JSON = Path("/mnt/hotstore/scan_processing/mts/pack1_edge_marks/cand/candidates.json")
SPECKS_DIR = Path("/mnt/hotstore/scan_processing/mts/text_block_specks/eval/components")
# Размеченная рамка считается задетой правилом, если выброшенная компонента накрывает такую долю её площади.
HIT_SHARE = 0.3


def labelled_items() -> pd.DataFrame:
    """Размеченные элементы обоих наборов в одной таблице: ``set, id, pdf, page, truth (junk/sign), label, x0..y1`` (px 300 dpi)."""
    rows = []
    for c in json.loads(EDGE_MARKS_JSON.read_text()):
        x0, y0, x1, y1 = c["box"]
        rows.append(dict(set="edge_marks", id=c["id"], pdf=c["pdf"], page=int(c["page"]), label=c["label"],
                         truth=c["truth"], x0=x0, y0=y0, x1=x1, y1=y1))  # fmt: skip
    records = pd.DataFrame(json.loads((SPECKS_DIR / "components.json").read_text()))
    labels = pd.read_csv(SPECKS_DIR / "comp_labels.csv").set_index("id").label
    k = RENDER_DPI / WORK_DPI
    for r in records.itertuples():
        label = labels.get(r.id, "U")
        if label == "U":
            continue
        x0, y0, x1, y1 = (v * k for v in r.box)
        rows.append(dict(set="specks", id=r.id, pdf=r.pdf, page=int(r.page), label=label,
                         truth="junk" if label == "S" else "sign", x0=x0, y0=y0, x1=x1, y1=y1))  # fmt: skip
    return pd.DataFrame(rows)


def page_frame(pdf: str, page: int) -> pd.DataFrame:
    """Кандидаты полосы пака-1 с признаками (разбор движком по краске без подсказок, бинарный рендер — он же тон).

    Args:
        pdf: Имя PDF.
        page: Страница с единицы.

    Returns:
        Кандидаты (``pdf``, ``page`` добавлены).
    """
    with fitz.open(NOGEO_DIR / pdf) as document:
        gray300 = render_gray(document, page - 1)
    analysis = analyze(gray300, edge_rule=True)
    frame = pd.DataFrame(page_candidates(analysis, gray300, gray300, None))
    frame["pdf"], frame["page"] = pdf, page
    return frame


def hits(items: pd.DataFrame, frame: pd.DataFrame, params: RuleParams) -> pd.DataFrame:
    """Какие размеченные элементы задевает правило.

    Args:
        items: Размеченные элементы (:func:`labelled_items`).
        frame: Кандидаты тех же полос (:func:`page_frame`).
        params: Пороги правил.

    Returns:
        ``items`` с колонками ``reason`` (причина первой задевшей компоненты или ``""``) и ``candidate`` (элемент
        накрыт хотя бы одним кандидатом).
    """
    frame = frame.assign(reason=reasons(frame, params))
    out = items.copy()
    out["reason"], out["candidate"] = "", False
    for (pdf, page), part in frame.groupby(["pdf", "page"]):
        mine = (out.pdf == pdf) & (out.page == page)
        for idx in out.index[mine]:
            x0, y0, x1, y1 = out.loc[idx, ["x0", "y0", "x1", "y1"]]
            area = max((x1 - x0) * (y1 - y0), 1.0)
            ix = np.clip(np.minimum(part.x1, x1) - np.maximum(part.x0, x0), 0, None)
            iy = np.clip(np.minimum(part.y1, y1) - np.maximum(part.y0, y0), 0, None)
            share = ix * iy / area
            covered = part[share >= HIT_SHARE]
            out.loc[idx, "candidate"] = not covered.empty
            fired = covered[covered.reason != ""]
            if not fired.empty:
                out.loc[idx, "reason"] = fired.reason.iloc[0]
    return out


__all__ = ["hits", "labelled_items", "page_frame"]
