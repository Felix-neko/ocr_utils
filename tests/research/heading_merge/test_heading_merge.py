"""Тесты стенда heading_merge: мера стыков ряда на синтетических боксах глифов (запреты — в тестах text_blocks/test_typeset.py)."""

from __future__ import annotations

import numpy as np

from research.heading_merge.measure import JointKind, joints_of


def word(x0: float, count: int, height: float, bottom: float, width: float = 0.0, gap: float = 0.0) -> np.ndarray:
    """Слово из ``count`` одинаковых глифов высоты ``height`` на базовой линии ``bottom``: боксы ``x0, y0, x1, y1``."""
    width = width or 0.6 * height
    gap = gap or 0.15 * height
    lefts = x0 + np.arange(count) * (width + gap)
    return np.column_stack([lefts, np.full(count, bottom - height), lefts + width, np.full(count, bottom)])


def test_joints_of_finds_kegl_jump() -> None:
    """Подпись (кегль 10) и заголовок (кегль 30) через 40 px: стык-кандидат одной оси."""
    glyphs = np.vstack([word(0, 8, 10, 100), word(8 * 7.5 + 40, 6, 30, 110)])
    joints = [j for j in joints_of(glyphs) if j.candidate]
    assert len(joints) == 1
    assert joints[0].kind is JointKind.AXIS
    assert joints[0].ratio > 2.5


def test_joints_of_marks_row_kind() -> None:
    """Глифы двух осей ряда: стык между ними — ``ROW``."""
    glyphs = np.vstack([word(0, 8, 10, 100), word(100, 6, 30, 110)])
    owners = np.array([0] * 8 + [1] * 6)
    assert any(j.kind is JointKind.ROW for j in joints_of(glyphs, owners))
