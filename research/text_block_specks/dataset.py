"""Набор стенда: по полосе — все концы строк у сторон блоков с боксом и признаками крайнего компонента, прогон «до» и «после».

Один прогон полосы (:func:`page_ends`) отдаёт по каждому концу строки у вертикальной стороны блока: ключ
``блок/сторона/ряд``, отклонение от устойчивой кривой стороны, x конца, строку-сегмент (ось с глифами), к
которой относится крайний компонент, его бокс и признаки (:class:`filters.EndComponent`). Тот же прогон с
подменой (:class:`filters.SpeckPatch`) даёт «после»; концы сопоставляются по ряду (ординате оси).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from ocr_utils.page_layout.text_blocks import WORK_DPI
from ocr_utils.page_layout.text_blocks.page import PageAnalysis
from ocr_utils.page_layout.text_blocks.sides import AlignMethod, SideKind, side_alignment
from research.text_block_specks.filters import End, Rule, SpeckPatch, end_component, work_binary
from research.text_block_specks.pages import PdfPage
from research.text_block_specks.scan import analyse, row_glyphs

MM_PER_PX = 25.4 / WORK_DPI


def end_rows(analysis: PageAnalysis, gray: np.ndarray) -> list[dict]:
    """Концы строк у вертикальных сторон блоков страницы с крайним компонентом.

    Args:
        analysis: Разбор страницы.
        gray: Серый рендер ``RENDER_DPI``.

    Returns:
        Словари: ``block``, ``side``, ``row``, ``rows``, ``span`` (края ряда по x), ``x``, ``y`` (пиксели рабочей
        копии), ``resid_mm``,
        ``status``, ``component`` (поля :class:`filters.EndComponent` или ``None``).
    """
    binary = work_binary(gray)
    out = []
    for block_index, block in enumerate(analysis.blocks):
        for side in (SideKind.LEFT, SideKind.RIGHT):
            alignment = side_alignment(block, side, AlignMethod.ROBUST)
            if alignment is None:
                continue
            end = End.RIGHT if side is SideKind.RIGHT else End.LEFT
            for item in alignment.ends:
                component = end_component(row_glyphs(block.rows[item.row]), binary, end)
                out.append(
                    {
                        "block": block_index,
                        "side": side.value,
                        "row": item.row,
                        "rows": len(block.rows),
                        "span": [round(float(block.rows[item.row].x0), 1), round(float(block.rows[item.row].x1), 1)],
                        "x": round(float(item.point[0]), 1),
                        "y": round(float(item.point[1]), 1),
                        "resid_mm": round(float(item.resid_mm), 3),
                        "status": item.status.value,
                        "component": (
                            None if component is None else {k: _round(v) for k, v in asdict(component).items()}
                        ),
                    }
                )
    return out


def _round(value):
    """Округление признаков для JSON (кортежи — поэлементно)."""
    if isinstance(value, tuple):
        return [round(float(v), 2) for v in value]
    return round(float(value), 4)


def page_ends(pdf_dir: Path, page: PdfPage, rule: Rule | None) -> dict:
    """Прогон полосы с правилом (``None`` — «до») и его концы строк.

    Args:
        pdf_dir: Каталог PDF.
        page: Полоса.
        rule: Правило мусора для подмены.

    Returns:
        ``{"key", "pdf", "page", "rule", "ends", "decisions", "blocks"}``: ``decisions`` — решения правила по
        проверенным компонентам (бокс в пикселях рабочей копии, конец, источник, мусор ли, признаки),
        ``blocks`` — число блоков.
    """
    with SpeckPatch(rule) as patch:
        analysis, gray = analyse(pdf_dir, page)
        ends = end_rows(analysis, gray)
    decisions = [
        {
            "box": [round(v, 1) for v in d.box],
            "end": d.end.value,
            "source": d.source,
            "noise": d.noise,
            "component": {k: _round(v) for k, v in asdict(d.component).items() if k != "box"},
        }
        for d in patch.decisions
    ]
    return {
        "key": page.key,
        "pdf": page.pdf,
        "page": page.page,
        "rule": None if rule is None else rule.name,
        "ends": ends,
        "decisions": decisions,
        "blocks": len(analysis.blocks),
    }


__all__ = ["end_rows", "page_ends"]
