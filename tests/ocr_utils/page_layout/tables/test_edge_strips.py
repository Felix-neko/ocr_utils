"""Тесты снятия полос тени у края вырезки из штриховой краски роста схемы (``tables.detector.edge_strips``, TABLES_VERSION 7)."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.geometry import Box
from ocr_utils.page_layout.tables.detector import EDGE_STRIP_MM, edge_strips, grow_diagram, line_art_mask
from ocr_utils.units import mm_to_px

DPI = 150


def _page_with_fold_shadow() -> np.ndarray:
    """Бинарная полоса 150 dpi (краска 255): рамка рисунка у корешка и 1-px тень сгиба у правого края, как ХиП 1991/06 IMG_0419."""
    binary = np.zeros((600, 400), np.uint8)
    # Рамка рисунка 273×200 px, толщина 3 px: правая сторона в 26 px от края — ближе склейки 5 мм (29 px).
    binary[100:300, 100:103] = 255
    binary[100:300, 370:373] = 255
    binary[100:103, 100:373] = 255
    binary[297:300, 100:373] = 255
    # Тень сгиба: столбец у самого края выше и ниже рисунка (5–66 мм), «крупнее буквы» по высоте.
    binary[60:500, 399] = 255
    return binary


def test_edge_strips_marks_thin_strip_only():
    """Полоса у края толщиной 1 px — полоса; рисунок у края толще 1 мм и фон — нет."""
    binary = np.zeros((300, 300), np.uint8)
    binary[20:280, 299] = 255  # полоса у правого края
    binary[50:250, 260:300] = 255  # рисунок, упёршийся в край (40 px ≈ 6.8 мм)
    binary[100:110, 0:200] = 0
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    strips = edge_strips(stats, binary.shape, mm_to_px(EDGE_STRIP_MM, DPI))
    assert not strips[0]
    # Полоса и рисунок касаются друг друга — одна компонента толщиной 40 px: не полоса.
    assert not strips[1:].any()
    lone = np.zeros((300, 300), np.uint8)
    lone[20:280, 299] = 255
    _, _, stats, _ = cv2.connectedComponentsWithStats(lone, 8)
    assert edge_strips(stats, lone.shape, mm_to_px(EDGE_STRIP_MM, DPI)).tolist() == [False, True]


def test_line_art_mask_drops_fold_shadow():
    """В штриховой краске нет тени сгиба, рисунок на месте."""
    binary = _page_with_fold_shadow()
    art = line_art_mask(binary, DPI)
    assert art[:, 399].max() == 0
    assert art[100:300, 100:103].min() == 255


def test_grow_diagram_stays_at_drawing():
    """Рост схемы по краске без тени сгиба остаётся рамкой рисунка (раньше уходил по тени на всю её высоту)."""
    binary = _page_with_fold_shadow()
    seed = Box(100, 100, 373, 300)
    grown = grow_diagram(line_art_mask(binary, DPI), seed, DPI, [])
    assert (grown.y0, grown.y1, grown.x1) == (100, 300, 373)
