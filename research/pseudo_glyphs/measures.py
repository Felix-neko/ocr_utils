"""Меры сторон блоков страницы для «было | стало»: выключка, выровненные стороны разбора и стороны-опоры второго преобразования v2 (``aligned_sides``), разброс опор сторон p95."""

from __future__ import annotations

from collections import Counter

import numpy as np

from ocr_utils.page_layout.text_blocks.alignment import INDENT_MIN_MM, Side, _residuals
from ocr_utils.page_layout.text_blocks.page import PageAnalysis
from ocr_utils.scan_cropping.v2.rectify.text_field import PageText, aligned_sides, side_samples
from ocr_utils.units import px_to_mm


def page_measures(analysis: PageAnalysis, dpi: float, spine_left: bool | None = None) -> dict:
    """Меры одной страницы.

    Args:
        analysis: Разбор страницы.
        dpi: Плотность холста (опоры v2 — в px холста).
        spine_left: Корешок слева (правая страница) — ``True``, справа — ``False``, неизвестно — ``None``; по нему
            стороны v2 делятся на корешковые и внешние.

    Returns:
        Словарь: ``blocks``, ``rows``, ``kind_<вид>`` — число блоков с такой выключкой, ``aligned_sides`` — выровненные
        стороны разбора, ``v2_sides`` / ``v2_sides_spine`` — стороны-опоры v2 (всего и со стороны корешка),
        ``v2_side_len_mm`` — их суммарная длина, ``side_p95_mm`` — p95 |отклонения| точек опор от общей вертикали
        отбивки колонки (``side_samples``), ``resid_p95_mm`` — p95 |края рядов − тренд| на выровненных сторонах без заходов внутрь дальше ``INDENT_MIN_MM``.
    """
    kinds = Counter(al.kind.value for al in analysis.alignments)
    out = {"blocks": len(analysis.blocks), "rows": sum(len(b.rows) for b in analysis.blocks)}
    for kind in ("both", "left", "right", "center", "ragged"):
        out[f"kind_{kind}"] = kinds.get(kind, 0)
    out["aligned_sides"] = sum(int(al.left.aligned) + int(al.right.aligned) for al in analysis.alignments)
    page = PageText(analysis, np.zeros(2), dpi / analysis.dpi)
    sides = aligned_sides(page, dpi)
    out["v2_sides"] = len(sides)
    spine = 0 if spine_left else 1
    out["v2_sides_spine"] = sum(1 for which, _ in sides if spine_left is not None and which == spine)
    out["v2_side_len_mm"] = float(sum(px_to_mm(float(np.ptp(pts[:, 1])), dpi) for _, pts in sides))
    samples = side_samples(page, dpi)
    values = np.abs(np.asarray(samples.value, np.float64))
    out["side_p95_mm"] = px_to_mm(float(np.percentile(values, 95)), dpi) if values.size else float("nan")
    resid = []
    for block, al in zip(analysis.blocks, analysis.alignments):
        for side, stats in ((Side.LEFT, al.left), (Side.RIGHT, al.right)):
            if stats.aligned and len(block.rows) >= 2:
                # Отклонения в мм, внутрь — плюс; заходы внутрь дальше INDENT_MIN_MM (абзацы) — исключения выключки.
                values = np.array(
                    [px_to_mm(v, analysis.dpi) for v in _residuals(list(block.rows), block.envelope, side)]
                )
                resid.extend(np.abs(values[values < INDENT_MIN_MM]).tolist())
    out["resid_p95_mm"] = float(np.percentile(resid, 95)) if resid else float("nan")
    return out


__all__ = ["page_measures"]
