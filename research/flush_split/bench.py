"""Разбор вырезки холста Кунса как в v2 (`rectify.text_field`) заданным способом `FlushSplit`; сводка блоков с мерами рядов."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from ocr_utils.page_layout.text_blocks import RENDER_DPI
from ocr_utils.page_layout.text_blocks import flush_split as F
from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
from ocr_utils.page_layout.text_blocks.hints import LayoutHints, masked_ink
from ocr_utils.page_layout.text_blocks import page as P
from ocr_utils.page_layout.text_blocks.page import analyse_gray
from ocr_utils.scan_cropping.v2.rectify.text_field import rule_ink, thicken_strokes


def analyze(png: Path, mode: F.FlushSplit, rule_barriers: P.RuleBarriers = P.RuleBarriers.OFF):
    """Разбор текстовых блоков вырезки способом ``mode`` (модульное умолчание подменяется на время разбора).

    Подготовка картинки — как во втором преобразовании v2: рендер ``RENDER_DPI``, утолщение штрихов, линейки вне
    краски. Без подсказок surya и без фильтра псевдоглифов (их нет в вырезках стенда).

    Args:
        png: Вырезка холста (рядом JSON с ``dpi``).
        mode: Способ разделения по смене выключки.
        rule_barriers: Вертикальные линейки краски — барьеры сборки рядов и блоков (``page.RuleBarriers``).

    Returns:
        Разбор страницы (``PageAnalysis``), координаты — рабочая копия ``WORK_DPI``.
    """
    meta = json.loads(png.with_suffix(".json").read_text())
    gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
    f = RENDER_DPI / meta["dpi"]
    g300 = thicken_strokes(cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA))
    g300 = masked_ink(g300, ~rule_ink(gray, meta["dpi"]))
    old, old_rules = F.DEFAULT_FLUSH_SPLIT, P.DEFAULT_RULE_BARRIERS
    F.DEFAULT_FLUSH_SPLIT, P.DEFAULT_RULE_BARRIERS = mode, rule_barriers
    try:
        return analyse_gray(g300, InkEngine(), hints=LayoutHints())
    finally:
        F.DEFAULT_FLUSH_SPLIT, P.DEFAULT_RULE_BARRIERS = old, old_rules


def summary(analysis) -> list[dict]:
    """Блоки разбора: ряды ``[y, x0, x1, высота, высота глифа, жирность]`` (px рабочей копии), выключка, шаг строк.

    Args:
        analysis: Разбор страницы.

    Returns:
        По словарю на блок.
    """
    out = []
    for block, al in zip(analysis.blocks, analysis.alignments):
        rows = [[round(r.y, 1), round(r.x0, 1), round(r.x1, 1), round(r.height, 1), round(r.glyph_h, 2),
                 round(r.weight, 3)] for r in block.rows]  # fmt: skip
        out.append({"rows": rows, "align": al.kind.value, "column": block.column, "pitch": round(block.pitch_px, 2)})
    return out


def run_page(args: tuple[Path, tuple[str, ...]]) -> dict:
    """Воркер: вырезка всеми способами; сводка блоков по способу или ошибка.

    Args:
        args: ``(PNG вырезки, имена способов FlushSplit)``; суффикс ``:ink`` — с вертикальными линейками краски как
            барьерами (``page.RuleBarriers.INK``), например ``align_right:ink``.

    Returns:
        ``{"page", "path", "width", <способ>: [блоки], "seconds"}`` или с ``"error"``.
    """
    png, modes = args
    t = time.monotonic()
    rec: dict = {"page": f"{png.parent.name[:2]}/{png.stem}", "path": str(png)}
    try:
        for mode in modes:
            name, _, rules = mode.partition(":")
            analysis = analyze(png, F.FlushSplit(name), P.RuleBarriers(rules or "off"))
            rec[mode] = summary(analysis)
            rec["width"] = analysis.width
    except Exception as exc:  # noqa: BLE001 — сбой одной вырезки не роняет прогон
        rec["error"] = repr(exc)
    rec["seconds"] = round(time.monotonic() - t, 1)
    return rec
