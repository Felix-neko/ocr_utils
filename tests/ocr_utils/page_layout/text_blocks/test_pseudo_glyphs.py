"""Тесты фильтра псевдоглифов ядра (``text_blocks.pseudo_glyphs``) и его режима в ``analyse_gray``: черта и крючки у края снимаются, буквы остаются, ``OFF`` ничего не меняет."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
from ocr_utils.page_layout.text_blocks.page import analyse_gray
from ocr_utils.page_layout.text_blocks.pseudo_glyphs import (
    PseudoMode,
    PseudoParams,
    PseudoReason,
    doubtful,
    erase_mask,
    reasons,
)


def _page() -> np.ndarray:
    """Синтетическая страница 300 dpi: строки текста и серая линия сгиба у левого края с крючками к двум строкам."""
    gray = np.full((900, 1100), 255, np.uint8)
    for i in range(14):
        cv2.putText(gray, "the quick brown fox jumps over lazy dog", (40, 80 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0,
                    3, cv2.LINE_AA)  # fmt: skip
    gray[40:760, 0:2] = 100
    gray[255:262, 0:30] = 100
    gray[505:512, 0:30] = 100
    return gray


def test_off_is_plain_analysis() -> None:
    """Режим ``OFF`` — прежний разбор: отчёта нет, оси те же."""
    gray = _page()
    plain = analyse_gray(gray, InkEngine())
    off = analyse_gray(gray, InkEngine(), pseudo=PseudoMode.OFF)
    assert off.pseudo_glyphs is None
    assert len(off.axes) == len(plain.axes)
    assert all(np.array_equal(a.points, b.points) for a, b in zip(off.axes, plain.axes))


def test_rules_drop_fold_line_only() -> None:
    """Режим ``RULES``: закрашено только у левого края, отчёт с причинами, строк столько же."""
    gray = _page()
    result = analyse_gray(gray, InkEngine(), pseudo=PseudoMode.RULES, raw300=gray)
    report = result.pseudo_glyphs
    assert report is not None and report.mode == PseudoMode.RULES.value
    assert report.drops and all(d.box[0] <= 2 for d in report.drops)
    assert {d.reason for d in report.drops} <= {r.value for r in PseudoReason}
    assert report.craft_calls == 0
    assert len(result.axes) == len(analyse_gray(gray, InkEngine()).axes)


def test_erase_mask_keeps_neighbours() -> None:
    """Ореол закраса не задевает пиксели соседних компонент."""
    labels = np.zeros((10, 10), np.int32)
    labels[4:6, 2:4] = 1
    labels[4:6, 5:7] = 2
    mask = erase_mask(labels, np.array([1]), halo_px=2)
    assert mask[4:6, 2:4].all() and not mask[4:6, 5:7].any() and mask[3, 2]


def test_doubtful_needs_grey_and_not_bar() -> None:
    """CRAFT спрашивают только про серые не-бруски у края или вдали, где правила промолчали."""
    frame = pd.DataFrame({
        "edge_mm": [2.0, 2.0, 2.0, 10.0], "tone_rel": [0.25, 0.0, 0.25, 0.25], "w_mm": [0.5, 0.5, 1.5, 0.5],
        "h_mm": [1.0, 1.0, 0.5, 1.0], "gap_rel": [0.1] * 4, "gap_mm": [0.3] * 4,
    })  # fmt: skip
    out = pd.Series([""] * 4)
    assert doubtful(frame, out, PseudoParams()).tolist() == [True, False, False, False]


def test_reasons_default_has_no_craft() -> None:
    """Без порога CRAFT голос CRAFT не подаётся, даже если колонка оценок есть."""
    frame = pd.DataFrame([dict(kind="row_left", edge_mm=3.0, stroke_mm=0.45, w_mm=0.6, h_mm=1.0, aspect=1.6,
                               tone_rel=0.25, h_rel_block=0.6, gap_mm=0.3, gap_rel=0.1, craft=0.0)])  # fmt: skip
    assert reasons(frame).iloc[0] == ""
    assert reasons(frame, PseudoParams(craft_doubt=0.4)).iloc[0] == PseudoReason.CRAFT.value
