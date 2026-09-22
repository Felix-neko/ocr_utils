"""Выход разбора: JSON по странице, CSV по блокам и короткая сводка markdown."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ocr_utils.curved_layout.page import PageAnalysis

CSV_FIELDS = [
    "name",
    "page",
    "variant",
    "engine",
    "column",
    "block",
    "lines",
    "pitch_mm",
    "align",
    "left_core",
    "left_mad_mm",
    "left_indent_rows",
    "left_dev_mm",
    "left_bend_mm",
    "right_core",
    "right_mad_mm",
    "right_indent_rows",
    "right_dev_mm",
    "right_bend_mm",
    "axis_bend_p90_mm",
    "axis_resid_parabola_p90_mm",
    "seconds",
]


def _curve(points: np.ndarray) -> list[list[float]]:
    """Кривая в JSON: пары с округлением до 0.1 px (как в ``curved_lines.raw``)."""
    return [[round(float(x), 1), round(float(y), 1)] for x, y in np.asarray(points)]


def page_json(analysis: PageAnalysis) -> dict:
    """Полный разбор страницы в JSON-совместимом виде."""
    blocks = []
    for block, alignment in zip(analysis.blocks, analysis.alignments):
        blocks.append(
            {
                "column": block.column,
                "index": block.index,
                "span": list(block.span),
                "lines": block.lines,
                "pitch_mm": round(block.pitch_mm, 2),
                "envelope": {
                    "smooth_pitches": block.envelope.smooth_pitches,
                    "left": _curve(block.envelope.left),
                    "right": _curve(block.envelope.right),
                    "top": _curve(block.envelope.top),
                    "bottom": _curve(block.envelope.bottom),
                    "polygon": _curve(block.envelope.polygon),
                },
                "envelope_coarse": {
                    "smooth_pitches": block.envelope_coarse.smooth_pitches,
                    "left": _curve(block.envelope_coarse.left),
                    "right": _curve(block.envelope_coarse.right),
                },
                "rows": [
                    {
                        "y": round(row.y, 1),
                        "x0": round(row.x0, 1),
                        "x1": round(row.x1, 1),
                        "height": round(row.height, 1),
                    }
                    for row in block.rows
                ],
                "alignment": {
                    "kind": alignment.kind.value,
                    "left": alignment.left.__dict__,
                    "right": alignment.right.__dict__,
                },
            }
        )
    return {
        "name": analysis.name,
        "page": analysis.page,
        "variant": analysis.variant,
        "engine": analysis.engine,
        "width": analysis.width,
        "height": analysis.height,
        "dpi": analysis.dpi,
        "gutters": [
            {"points": [[round(y, 1), round(x0, 1), round(x1, 1)] for y, x0, x1 in g.points]} for g in analysis.gutters
        ],
        "zones": [{"y0": z.y0, "y1": z.y1, "columns": [list(c) for c in z.columns]} for z in analysis.zones],
        "seconds": round(analysis.seconds, 2),
        "note": analysis.note,
        "axes": [
            {
                "column": axis.column,
                "height": round(axis.height, 1),
                "sagitta_mm": round(axis.sagitta_mm, 2),
                "slope_deg": round(axis.slope_deg, 2),
                "bend_mm": round(axis.bend_mm, 2),
                "resid_parabola_mm": round(axis.resid_parabola_mm, 2),
                "points": _curve(axis.points),
            }
            for axis in analysis.axes
        ],
        "blocks": blocks,
    }


def write_json(analysis: PageAnalysis, directory: Path) -> Path:
    """Записать разбор страницы в ``<directory>/<ключ>.json``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{analysis.key}.json"
    path.write_text(json.dumps(page_json(analysis), ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def rows_for_csv(analysis: PageAnalysis) -> list[dict]:
    """Строки CSV: по строке на блок страницы."""
    out: list[dict] = []
    for block, alignment in zip(analysis.blocks, analysis.alignments):
        own = [axis for axis in analysis.axes if axis.column == block.column]
        bends = [axis.bend_mm for axis in own] or [0.0]
        resid = [axis.resid_parabola_mm for axis in own] or [0.0]
        out.append(
            {
                "name": analysis.name,
                "page": analysis.page,
                "variant": analysis.variant,
                "engine": analysis.engine,
                "column": block.column,
                "block": block.index,
                "lines": block.lines,
                "pitch_mm": round(block.pitch_mm, 2),
                "align": alignment.kind.value,
                "left_core": round(alignment.left.core_share, 3),
                "left_mad_mm": round(alignment.left.resid_mad_mm, 3),
                "left_indent_rows": alignment.left.indent_rows,
                "left_dev_mm": round(alignment.left.envelope_dev_mm, 3),
                "left_bend_mm": round(alignment.left.bend_mm, 3),
                "right_core": round(alignment.right.core_share, 3),
                "right_mad_mm": round(alignment.right.resid_mad_mm, 3),
                "right_indent_rows": alignment.right.indent_rows,
                "right_dev_mm": round(alignment.right.envelope_dev_mm, 3),
                "right_bend_mm": round(alignment.right.bend_mm, 3),
                "axis_bend_p90_mm": round(float(np.percentile(bends, 90)), 3),
                "axis_resid_parabola_p90_mm": round(float(np.percentile(resid, 90)), 3),
                "seconds": round(analysis.seconds, 2),
            }
        )
    return out


def write_csv(analyses: list[PageAnalysis], path: Path) -> Path:
    """Записать CSV по всем разобранным страницам."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for analysis in analyses:
            writer.writerows(rows_for_csv(analysis))
    return path


def markdown(analyses: list[PageAnalysis]) -> str:
    """Короткая сводка: страница, движок, блоки, строки, выключка, девиация огибающих."""
    lines = [
        "# Разбор текста с кривыми строками",
        "",
        "| страница | движок | блоков | строк | выключка | dev L/R, мм | изгиб L/R, мм | с |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for analysis in analyses:
        kinds = ",".join(alignment.kind.value for alignment in analysis.alignments) or "—"
        dev = ",".join(
            f"{alignment.left.envelope_dev_mm:.1f}/{alignment.right.envelope_dev_mm:.1f}"
            for alignment in analysis.alignments
        )
        bend = ",".join(
            f"{alignment.left.bend_mm:.1f}/{alignment.right.bend_mm:.1f}" for alignment in analysis.alignments
        )
        lines.append(
            f"| {analysis.name} с.{analysis.page} [{analysis.variant}] | {analysis.engine} | "
            f"{len(analysis.blocks)} | {len(analysis.axes)} | {kinds} | {dev or '—'} | {bend or '—'} | "
            f"{analysis.seconds:.1f} |"
        )
    return "\n".join(lines) + "\n"


__all__ = ["CSV_FIELDS", "markdown", "page_json", "rows_for_csv", "write_csv", "write_json"]
