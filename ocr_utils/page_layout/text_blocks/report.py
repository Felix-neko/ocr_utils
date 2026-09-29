"""Выход разбора: JSON по странице, CSV по блокам и короткая сводка markdown."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ocr_utils.page_layout.text_blocks.page import PageAnalysis
from ocr_utils.page_layout import px_to_mm

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
    "glyph_w_mm",
    "glyph_h_mm",
    "axis_bend_p90_mm",
    "axis_resid_parabola_p90_mm",
    "seconds",
    "ink_share",
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
                "glyph_mm": [
                    round(px_to_mm(block.glyph_size[0], block.dpi), 2),
                    round(px_to_mm(block.glyph_size[1], block.dpi), 2),
                ],
                "envelope": {
                    "smooth_pitches": block.envelope.smooth_pitches,
                    "polygon_dilated": (
                        _curve(block.envelope.polygon_dilated) if block.envelope.polygon_dilated is not None else None
                    ),
                    "dilate_mm": [
                        round(px_to_mm(block.envelope.dilate_px[0], block.dpi), 2),
                        round(px_to_mm(block.envelope.dilate_px[1], block.dpi), 2),
                    ],
                    "left": _curve(block.envelope.left),
                    "right": _curve(block.envelope.right),
                    "top": _curve(block.envelope.top),
                    "bottom": _curve(block.envelope.bottom),
                    "polygon": _curve(block.envelope.polygon),
                    # Недостоверные участки сторон гладкой границы (``BlocksMode.SMOOTH``): отрезки по высоте,
                    # где сторона — ступенька по выносу за колонку (пометка на полях, строка шире колонки).
                    "unreliable_left": [
                        list(map(float, span)) for span in getattr(block.envelope, "unreliable_left", ())
                    ],
                    "unreliable_right": [
                        list(map(float, span)) for span in getattr(block.envelope, "unreliable_right", ())
                    ],
                },
                # Справочная граница по краске: она одна обещает охват всех букв рядов, тогда как
                # главная — полоса вокруг оси — нарочно идёт мимо выносных элементов.
                "envelope_ink": (
                    {
                        "top": _curve(block.envelope_ink.top),
                        "bottom": _curve(block.envelope_ink.bottom),
                        "polygon": _curve(block.envelope_ink.polygon),
                    }
                    if block.envelope_ink is not None
                    else None
                ),
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
                        # Хвост последней строки и линия отсечки (``blocks._tail_of``): виртуальные,
                        # в ``axes`` страницы их нет.
                        "tail": _curve(row.tail) if row.tail is not None else None,
                        "cut": _curve(row.cut) if row.cut is not None else None,
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
        "ink_share": round(analysis.ink_share, 4),
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
                "mark_spans": [[round(float(a), 1), round(float(b), 1)] for a, b in axis.mark_spans],
                # Участки перескока на соседнюю строку: меры формы оси выше — по участку без них.
                "jump_spans": [[round(float(a), 1), round(float(b), 1)] for a, b in axis.jump_spans],
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
                "glyph_w_mm": round(px_to_mm(block.glyph_size[0], block.dpi), 2),
                "glyph_h_mm": round(px_to_mm(block.glyph_size[1], block.dpi), 2),
                "axis_bend_p90_mm": round(float(np.percentile(bends, 90)), 3),
                "axis_resid_parabola_p90_mm": round(float(np.percentile(resid, 90)), 3),
                "seconds": round(analysis.seconds, 2),
                "ink_share": round(analysis.ink_share, 4),
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


def _axis_deviation(axes: list, reference: list) -> list[float]:
    """Расхождение осей движка с осями по краске: для каждой оси-эталона — медиана |dy| в мм.

    Ось движка подбирается ближайшая по центру; сравнение идёт на общем отрезке по x, поэтому
    короткая ось не наказывается за то, что не дошла до края.

    Args:
        axes: Оси движка.
        reference: Оси эталона (движок ``ink``).

    Returns:
        Список расхождений в миллиметрах, по одному на ось эталона, у которой нашлась пара.
    """
    out: list[float] = []
    for target in reference:
        best, best_distance = None, None
        for axis in axes:
            distance = abs(axis.cy - target.cy) + abs((axis.x0 + axis.x1) / 2.0 - (target.x0 + target.x1) / 2.0)
            if best_distance is None or distance < best_distance:
                best, best_distance = axis, distance
        if best is None:
            continue
        left, right = max(best.x0, target.x0), min(best.x1, target.x1)
        if right - left < 10:
            continue  # оси почти не пересекаются по x: это разные строки
        grid = np.linspace(left, right, num=25)
        own = np.interp(grid, best.points[:, 0], best.points[:, 1])
        mine = np.interp(grid, target.points[:, 0], target.points[:, 1])
        out.append(px_to_mm(float(np.median(np.abs(own - mine))), target.dpi))
    return out


def engines_summary(analyses: list[PageAnalysis]) -> list[str]:
    """Сравнение движков на одних и тех же страницах: полнота, расхождение с краской, время.

    Args:
        analyses: Разборы всех страниц всеми движками.

    Returns:
        Строки markdown с таблицей; пустой список, если движок был один.
    """
    engines = sorted({analysis.engine for analysis in analyses})
    if len(engines) < 2:
        return []
    # Эталон — только разбор движком ``ink``: в общий словарь иначе попадает последний по
    # порядку движок, и расхождение считать не с чем.
    reference = {
        (analysis.name, analysis.page, analysis.variant): analysis for analysis in analyses if analysis.engine == "ink"
    }
    lines = [
        "",
        "## Сравнение движков",
        "",
        "| движок | страниц | строк | блоков | краска под строками | строк через межколонник | "
        "медиана \\|dy\\| от ink, мм | с/страницу |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for engine in engines:
        own = [analysis for analysis in analyses if analysis.engine == engine]
        deviations: list[float] = []
        for analysis in own:
            base = reference.get((analysis.name, analysis.page, analysis.variant))
            if base is not None and engine != "ink":
                deviations.extend(_axis_deviation(list(analysis.axes), list(base.axes)))
        crossed = sum(1 for analysis in own for axis in analysis.axes if axis.cross)
        share = float(np.mean([analysis.ink_share for analysis in own])) if own else 0.0
        median = f"{float(np.median(deviations)):.2f}" if deviations else "—"
        lines.append(
            f"| {engine} | {len(own)} | {sum(len(analysis.axes) for analysis in own)} | "
            f"{sum(len(analysis.blocks) for analysis in own)} | {share:.0%} | {crossed} | {median} | "
            f"{float(np.mean([analysis.seconds for analysis in own])):.1f} |"
        )
    return lines


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
    tails = sum(row.tail is not None for analysis in analyses for block in analysis.blocks for row in block.rows)
    lines.append("")
    lines.append(f"Рядов с хвостом (последняя строка, достроенная до линии отсечки): {tails}.")
    lines.extend(engines_summary(analyses))
    return "\n".join(lines) + "\n"


def sides_json(analysis: PageAnalysis) -> dict:
    """Разметка сторон всеми методами и выравнивание всеми методами — JSON по странице.

    Args:
        analysis: Разбор страницы.

    Returns:
        Словарь: ключ страницы и по блоку — контур, метки и угловые флаги каждого метода, меры
        сторон без углов, а по вертикальным сторонам — концы строк с отклонениями у каждого метода
        выравнивания.
    """
    from ocr_utils.page_layout.text_blocks.sides import AlignMethod, SideKind, SidesMethod, side_alignment
    from ocr_utils.page_layout.text_blocks.sides import DEFAULT_SIDES_METHOD, edge_axis, filled_side, side_measures
    from ocr_utils.page_layout.text_blocks.sides import sides_of

    blocks = []
    for block in analysis.blocks:
        # Дополнительная линия стороны строится по разметке сторон по умолчанию — той же, что на оверлеях.
        default_sides = sides_of(block, DEFAULT_SIDES_METHOD)
        methods = {}
        for method in SidesMethod:
            sides = sides_of(block, method)
            methods[method.value] = {
                "polygon": _curve(sides.polygon),
                "labels": [label.value for label in sides.labels],
                "corner": [bool(flag) for flag in sides.corner],
                "uncertain": [bool(flag) for flag in sides.uncertain] if sides.uncertain is not None else None,
                "measures": [
                    {
                        "side": item.side.value,
                        "length_mm": round(item.length_mm, 1),
                        "corner_share": round(item.corner_share, 3),
                        "tilt_deg": round(item.tilt_deg, 3),
                        "bend_mm": round(item.bend_mm, 2),
                    }
                    for item in side_measures(sides, block.dpi)
                ],
            }
        align = {}
        for method in AlignMethod:
            own = {}
            for side in (SideKind.LEFT, SideKind.RIGHT):
                result = side_alignment(block, side, method)
                if result is None:
                    continue
                own[side.value] = {
                    "on_share": round(result.on_share, 3),
                    "aligned_share": round(result.aligned_share, 3),
                    "ends": [
                        {
                            "row": end.row,
                            "point": [round(float(end.point[0]), 1), round(float(end.point[1]), 1)],
                            "resid_mm": round(end.resid_mm, 2),
                            "status": end.status.value,
                            "aligned": end.aligned,
                        }
                        for end in result.ends
                    ],
                    "filled": filled_json(filled_side(default_sides, result, block)),
                }
            align[method.value] = own
        edges = {}
        for top, name in ((True, "top"), (False, "bottom")):
            edge = edge_axis(block, top)
            if edge is not None:
                edges[name] = {
                    "points": _curve(edge.points),
                    "real_x": [round(edge.real_x0, 1), round(edge.real_x1, 1)],
                    "reference": edge.reference,
                    "gap": round(edge.gap, 1),
                }
        blocks.append(
            {"column": block.column, "index": block.index, "edge_axes": edges, "sides": methods, "align": align}
        )
    return {
        "key": analysis.key,
        "name": analysis.name,
        "page": analysis.page,
        "variant": analysis.variant,
        "width": analysis.width,
        "height": analysis.height,
        "dpi": analysis.dpi,
        "default_sides_method": DEFAULT_SIDES_METHOD.value,
        "blocks": blocks,
    }


def filled_json(line) -> dict | None:
    """Дополнительная линия стороны (``sides.FilledSide``) для JSON: точки, флаги заплатки и меры.

    Args:
        line: Линия или ``None``, если она не построилась.

    Returns:
        Словарь или ``None``.
    """
    if line is None:
        return None
    return {
        "points": _curve(line.points),
        "filled": [bool(flag) for flag in line.filled],
        "length_mm": round(line.length_mm, 1),
        "tilt_deg": round(line.tilt_deg, 3),
        "bend_mm": round(line.bend_mm, 2),
        "raw_tilt_deg": round(line.raw_tilt_deg, 3),
        "raw_bend_mm": round(line.raw_bend_mm, 2),
    }


def sides_markdown(analyses: list[PageAnalysis]) -> str:
    """Сводка сравнения методов: согласие разметки сторон и выравнивание по сторонам.

    Args:
        analyses: Разборы страниц.

    Returns:
        Markdown: таблица согласия методов разметки по страницам (доля длины контура с одной и той
        же стороной, попарно, и доля угловой длины) и таблица выравнивания — доля рядов на кривой и
        в выровненных сериях по левой и правой стороне у каждого метода, по блокам из трёх и больше
        рядов; третья таблица — наклон и изгиб вертикальных сторон всей стороной и дополнительной линией
        (``sides.filled_side``) по каждому методу выравнивания.
    """
    from ocr_utils.page_layout.text_blocks.sides import AlignMethod, SideKind, SidesMethod, label_agreement
    from ocr_utils.page_layout.text_blocks.sides import DEFAULT_SIDES_METHOD, filled_side, side_alignment, sides_of

    pairs = [
        (SidesMethod.CONSTRUCT, SidesMethod.RAYS),
        (SidesMethod.CONSTRUCT, SidesMethod.FRAME),
        (SidesMethod.RAYS, SidesMethod.FRAME),
    ]
    lines = [
        "# Стороны границы блока и выравнивание",
        "",
        "## Согласие методов разметки (доля длины контура)",
        "",
        "| страница | блоков | "
        + " | ".join(f"{a.value}/{b.value}" for a, b in pairs)
        + " | углов: "
        + ", ".join(m.value for m in SidesMethod)
        + " |",
        "|---|---|" + "---|" * len(pairs) + "---|",
    ]
    for analysis in analyses:
        if not analysis.blocks:
            continue
        agree = np.zeros(len(pairs))
        corners = np.zeros(len(SidesMethod))
        for block in analysis.blocks:
            result = {method: sides_of(block, method) for method in SidesMethod}
            agree += [label_agreement(result[a], result[b]) for a, b in pairs]
            corners += [float(result[method].corner.mean()) for method in SidesMethod]
        count = len(analysis.blocks)
        lines.append(
            f"| {analysis.key} | {count} | "
            + " | ".join(f"{value / count:.3f}" for value in agree)
            + " | "
            + ", ".join(f"{value / count:.2f}" for value in corners)
            + " |"
        )
    lines += [
        "",
        "## Выравнивание по вертикальным сторонам (блоки от трёх рядов)",
        "",
        "Ячейка — «доля рядов на кривой / доля рядов в выровненных сериях» слева и справа.",
        "",
        "| страница | блок | рядов | " + " | ".join(method.value for method in AlignMethod) + " |",
        "|---|---|---|" + "---|" * len(AlignMethod),
    ]
    for analysis in analyses:
        for block in analysis.blocks:
            if len(block.rows) < 3:
                continue
            cells = []
            for method in AlignMethod:
                parts = []
                for side in (SideKind.LEFT, SideKind.RIGHT):
                    result = side_alignment(block, side, method)
                    parts.append("—" if result is None else f"{result.on_share:.2f}/{result.aligned_share:.2f}")
                cells.append(" · ".join(parts))
            lines.append(
                f"| {analysis.key} | {block.column}.{block.index} | {len(block.rows)} | " + " | ".join(cells) + " |"
            )
    lines += [
        "",
        "## Наклон и изгиб вертикальных сторон: вся сторона → дополнительная линия (блоки от трёх рядов)",
        "",
        "Дополнительная линия — сторона без невыровненных концов, невыровненная середина заменена "
        "заплаткой PCHIP (`sides.filled_side`). Ячейка — «наклон °, изгиб мм» всей стороны и линии, у линии "
        "в скобках её длина поперёк строк (на коротких наклон шумит); прочерк — линия не построилась "
        "(выровненного меньше двух точек). Вся сторона от метода не зависит и дана одна.",
        "",
        "| страница | блок | сторона | вся сторона | " + " | ".join(method.value for method in AlignMethod) + " |",
        "|---|---|---|---|" + "---|" * len(AlignMethod),
    ]
    for analysis in analyses:
        for block in analysis.blocks:
            if len(block.rows) < 3:
                continue
            sides = sides_of(block, DEFAULT_SIDES_METHOD)
            for side in (SideKind.LEFT, SideKind.RIGHT):
                raw = "—"
                cells = []
                for method in AlignMethod:
                    result = side_alignment(block, side, method)
                    line = None if result is None else filled_side(sides, result, block)
                    if line is None:
                        cells.append("—")
                        continue
                    raw = f"{line.raw_tilt_deg:+.2f}°, {line.raw_bend_mm:.1f}"
                    cells.append(f"{line.tilt_deg:+.2f}°, {line.bend_mm:.1f} ({line.length_mm:.0f} мм)")
                lines.append(
                    f"| {analysis.key} | {block.column}.{block.index} | {side.value} | {raw} | "
                    + " | ".join(cells)
                    + " |"
                )
    return "\n".join(lines) + "\n"


__all__ = [
    "CSV_FIELDS",
    "engines_summary",
    "filled_json",
    "markdown",
    "page_json",
    "rows_for_csv",
    "sides_json",
    "sides_markdown",
    "write_csv",
    "write_json",
]
