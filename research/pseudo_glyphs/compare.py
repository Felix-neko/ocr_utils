"""«Было | стало» по вырезкам: меры сторон блоков без фильтра (с правилом 0.3 мм и без) и с фильтром псевдоглифов (правила, правила + CRAFT), закрашенные компоненты с причинами, оверлеи."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed
from ocr_utils.page_layout.text_blocks import RENDER_DPI
from research.pseudo_glyphs.crops import PreparedPage, analyze, prepare
from research.pseudo_glyphs.features import spine_side, with_edge_rule
from research.pseudo_glyphs.measures import page_measures
from research.pseudo_glyphs.prototype import FilterMode, _page_drops, filter_page
from research.pseudo_glyphs.rules import RuleParams

# Цвета оверлея (палитра проекта): ось строки, сторона выровнена / нет, закрашенный псевдоглиф.
COLOUR_AXIS = (40, 170, 40)
COLOUR_ALIGNED = (0, 150, 0)
COLOUR_RAGGED = (0, 0, 220)
COLOUR_DROP = (200, 60, 200)
DROP_ALPHA = 0.45


def compare_page(path: Path, out_dir: Path, params: RuleParams, craft_dir: Path | None, overlay: bool) -> dict:
    """Одна вырезка: меры без фильтра (off, on) и с фильтром (rules, rules_craft) и закрашенное (воркер пула).

    Args:
        path: PNG вырезки.
        out_dir: Папка прогона сравнения (``pages/<ключ>.json``, ``overlays/``).
        params: Пороги правил (``craft_doubt`` — порог CRAFT для режима с CRAFT).
        craft_dir: Папка карт CRAFT (``<ключ без />.png``) или ``None`` — без режима с CRAFT.
        overlay: Рисовать ли оверлей «было | стало» для страниц, где что-то закрашено.

    Returns:
        Словарь: ключ, меры по режимам, закрашенное по режимам, секунды.
    """
    started = time.monotonic()
    page = prepare(path)
    spine = spine_side(page.part)
    first = analyze(page.gray300, edge_rule=False)
    spine_left = page.part == "R"
    result = {"key": page.key, "part": page.part,
              "off": page_measures(first, page.dpi, spine_left),
              "on": page_measures(with_edge_rule(first, page.gray300), page.dpi, spine_left)}  # fmt: skip
    modes = [(FilterMode.RULES, None)]
    if craft_dir is not None and params.craft_doubt is not None:
        craft_path = craft_dir / f"{page.key.replace('/', '__')}.png"
        if craft_path.exists():
            heat = cv2.imread(str(craft_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            modes.append((FilterMode.RULES_CRAFT, heat))
    first_pass: dict = {}
    for mode, heat in modes:
        # С CRAFT первый проход часто снимает то же, что и одни правила: тогда разбор не повторяется.
        if mode is FilterMode.RULES_CRAFT:
            _, found = _page_drops(first, page.gray300, page.raw300, spine, params, heat)
            if {d.comp for d in found} == first_pass.get(FilterMode.RULES):
                result[mode.value] = result[FilterMode.RULES.value]
                result[f"{mode.value}_drops"] = result[f"{FilterMode.RULES.value}_drops"]
                continue
        filtered = filter_page(page.gray300, page.raw300, spine, mode, params, heat, False, first)
        if mode is FilterMode.RULES:
            _, found = _page_drops(first, page.gray300, page.raw300, spine,
                                   RuleParams(**{**params.__dict__, "craft_doubt": None}), None)  # fmt: skip
            first_pass[mode] = {d.comp for d in found}
        result[mode.value] = page_measures(filtered.analysis, page.dpi, spine_left)
        result[f"{mode.value}_drops"] = [asdict(d) for d in filtered.drops]
        if overlay and filtered.drops and mode is FilterMode.RULES:
            target = out_dir / "overlays" / f"{page.key.replace('/', '__')}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), before_after(page, first, filtered), [cv2.IMWRITE_JPEG_QUALITY, 85])
    result["seconds"] = time.monotonic() - started
    target = out_dir / "pages" / f"{page.key}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False))
    return result


def _draw(page: PreparedPage, analysis, drops=()) -> np.ndarray:
    """Серый страницы с осями строк, сторонами блоков (выровненные — зелёные, нет — красные) и закрашенным."""
    canvas = cv2.cvtColor(page.raw300, cv2.COLOR_GRAY2BGR)
    k = RENDER_DPI / analysis.dpi
    if drops:
        layer = canvas.copy()
        for d in drops:
            x0, y0, x1, y1 = d.box
            cv2.rectangle(layer, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), COLOUR_DROP, -1)
        cv2.addWeighted(layer, DROP_ALPHA, canvas, 1 - DROP_ALPHA, 0, canvas)
    for axis in analysis.axes:
        cv2.polylines(canvas, [(axis.points * k).astype(np.int32)], False, COLOUR_AXIS, 1)
    for block, al in zip(analysis.blocks, analysis.alignments):
        env = block.envelope
        for curve, stats in ((env.core_left if env.core_left is not None else env.left, al.left),
                             (env.core_right if env.core_right is not None else env.right, al.right)):  # fmt: skip
            colour = COLOUR_ALIGNED if stats.aligned else COLOUR_RAGGED
            cv2.polylines(canvas, [(np.asarray(curve) * k).astype(np.int32)], False, colour, 2)
    return canvas


def before_after(page: PreparedPage, first, filtered) -> np.ndarray:
    """Склейка «было (без фильтра) | стало (фильтр по правилам)» с шапкой и легендой в полях."""
    left = _draw(page, first)
    right = _draw(page, filtered.analysis, filtered.drops)
    gap = np.full((left.shape[0], 12, 3), 255, np.uint8)
    body = np.hstack([left, gap, right])
    scale = min(1.0, 1800 / body.shape[0])
    body = cv2.resize(body, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    reasons = {}
    for d in filtered.drops:
        reasons[d.reason] = reasons.get(d.reason, 0) + 1
    return framed(
        body,
        [f"ХиП 1991 {page.key} — было (без фильтра, без правила 0.3 мм) | стало (фильтр псевдоглифов)",
         "закрашено: " + ", ".join(f"{k} {v}" for k, v in reasons.items())],  # fmt: skip
        [LegendEntry("ось строки", COLOUR_AXIS),
         LegendEntry("сторона блока выровнена", COLOUR_ALIGNED),
         LegendEntry("сторона не выровнена", COLOUR_RAGGED),
         LegendEntry("закрашенный псевдоглиф", COLOUR_DROP, alpha=DROP_ALPHA, style=SampleStyle.BOX)],  # fmt: skip
    )


__all__ = ["before_after", "compare_page"]
