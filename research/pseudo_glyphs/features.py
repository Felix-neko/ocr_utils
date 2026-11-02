"""Проход по вырезкам: разбор как во втором преобразовании, кандидаты с признаками (CSV на страницу), меры сторон с правилом 0.3 мм и без, серый 300 dpi для карт CRAFT и pero."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import cv2
import pandas as pd

from ocr_utils.page_layout.text_blocks.alignment import alignment_of
from ocr_utils.page_layout.text_blocks.leaders import leaders_of
from ocr_utils.page_layout.text_blocks.page import PageAnalysis, _work_copy, mark_cut_rows
from research.pseudo_glyphs.candidates import EdgeSide, page_candidates
from research.pseudo_glyphs.crops import analyze, edge_touch, prepare
from research.pseudo_glyphs.measures import page_measures


def spine_side(part: str) -> EdgeSide:
    """Сторона корешка на вырезке: у правой страницы (``R``) — слева, у левой — справа."""
    return EdgeSide.LEFT if part == "R" else EdgeSide.RIGHT


def with_edge_rule(analysis: PageAnalysis, gray300) -> PageAnalysis:
    """Разбор, к которому применено правило 0.3 мм у края картинки (как с ним в боевом коде), без повторного разбора.

    Правило трогает только пометку недостоверных участков у обрезанных рядов (``mark_cut_rows``) и выключку
    (``alignment_of`` не мерит такие ряды); строки и блоки те же.

    Args:
        analysis: Разбор без правила (:func:`crops.analyze` с ``edge_rule=False``).
        gray300: Тот же серый 300 dpi (для отточий номеров).

    Returns:
        Разбор с правилом.
    """
    work = _work_copy(gray300, analysis.dpi)
    number_leaders = leaders_of(work, analysis.dpi, sparse=True)[0]
    with edge_touch(True):
        blocks = tuple(mark_cut_rows(block, analysis.width) for block in analysis.blocks)
        alignments = tuple(alignment_of(block, analysis.width, number_leaders) for block in blocks)
    return replace(analysis, blocks=blocks, alignments=alignments)


def page_paths(out_dir: Path, key: str) -> tuple[Path, Path, Path]:
    """Пути выхода страницы: CSV кандидатов, JSON мер, PNG серого 300 dpi."""
    return (out_dir / "features" / f"{key}.csv", out_dir / "measures" / f"{key}.json",
            out_dir / "gray300" / f"{key}.png")  # fmt: skip


def process_page(path: Path, out_dir: Path) -> tuple[str, int, float]:
    """Обработать одну вырезку (воркер пула).

    Args:
        path: PNG вырезки.
        out_dir: Корень выхода стенда.

    Returns:
        ``(ключ, число кандидатов, секунды)``.
    """
    started = time.monotonic()
    page = prepare(path)
    csv_path, json_path, png_path = page_paths(out_dir, page.key)
    for target in (csv_path, json_path, png_path):
        target.parent.mkdir(parents=True, exist_ok=True)
    analysis = analyze(page.gray300, edge_rule=False)
    spine = spine_side(page.part)
    rows = page_candidates(analysis, page.gray300, page.raw300, spine)
    frame = pd.DataFrame(rows)
    frame.insert(0, "key", page.key)
    frame.to_csv(csv_path, index=False)
    measures = {
        "key": page.key,
        "dpi": page.dpi,
        "part": page.part,
        "off": page_measures(analysis, page.dpi, spine is EdgeSide.LEFT),
        "on": page_measures(with_edge_rule(analysis, page.gray300), page.dpi, spine is EdgeSide.LEFT),
    }
    json_path.write_text(json.dumps(measures, ensure_ascii=False))
    cv2.imwrite(str(png_path), page.gray300, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return page.key, len(rows), time.monotonic() - started


def collect(out_dir: Path) -> pd.DataFrame:
    """Собрать CSV кандидатов всех страниц в один (``candidates.csv``) с номером кандидата ``cid``.

    Args:
        out_dir: Корень выхода стенда.

    Returns:
        Таблица кандидатов.
    """
    parts = [pd.read_csv(p) for p in sorted((out_dir / "features").glob("*/*.csv")) if p.stat().st_size > 1]
    frame = pd.concat(parts, ignore_index=True)
    frame.insert(0, "cid", [f"c{i:06d}" for i in range(len(frame))])
    frame.to_csv(out_dir / "candidates.csv", index=False)
    return frame


__all__ = ["collect", "page_paths", "process_page", "spine_side", "with_edge_rule"]
