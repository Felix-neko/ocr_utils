"""Тесты стенда heading_merge на синтетических боксах глифов: стыки, разный набор, резка строки, продление оси."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocr_utils.page_layout.text_blocks.segment import Segment
from research.heading_merge.gates import clip_extension, different_sets, split_segment
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


def test_different_sets_same_baseline_is_one_line() -> None:
    """«50 лет»: цифры вдвое выше слова, но на одной базовой линии — не разный набор."""
    digits, letters = word(0, 2, 32, 140), word(80, 3, 16, 140)
    assert not different_sets(digits, letters, gap=41.0)


def test_different_sets_masthead_and_small_line() -> None:
    """Шапка (кегль 54) и «Год» (17) с низами на 14 px выше через 30 px — разный набор."""
    masthead, small = word(0, 6, 54, 290), word(400, 3, 17, 276)
    assert different_sets(masthead, small, gap=30.0)


def test_different_sets_body_digits_are_not_cut() -> None:
    """Цифры корпуса («1968», 18 px) среди строчных (10 px) на одной линии через широкий пробел — не резать."""
    letters, digits = word(0, 6, 10, 100), word(100, 4, 18, 100)
    assert not different_sets(letters, digits, gap=30.0)


def _segment(glyphs: np.ndarray) -> Segment:
    """Строка из боксов глифов: центр-линия по серединам глифов с шагом в пиксель."""
    xs = np.arange(glyphs[:, 0].min(), glyphs[:, 2].max(), 1.0)
    ys = np.interp(xs, (glyphs[:, 0] + glyphs[:, 2]) / 2.0, (glyphs[:, 1] + glyphs[:, 3]) / 2.0)
    return Segment(
        x0=int(glyphs[:, 0].min()),
        y0=int(glyphs[:, 1].min()),
        x1=int(glyphs[:, 2].max()),
        y1=int(glyphs[:, 3].max()),
        height=float(np.median(glyphs[:, 3] - glyphs[:, 1])),
        xs=xs,
        ys=ys,
        weights=np.ones_like(xs),
        glyphs=glyphs,
    )


def test_split_segment_cuts_masthead_from_month() -> None:
    """Шапка (кегль 43) и месяц (22, низ на 23 px выше) через 56 px — две строки, у каждой свои глифы."""
    glyphs = np.vstack([word(0, 8, 43, 190), word(8 * 32.25 + 56, 5, 22, 167)])
    parts = split_segment(_segment(glyphs))
    assert len(parts) == 2
    assert len(parts[0].glyphs) == 8 and len(parts[1].glyphs) == 5
    assert parts[0].x1 < parts[1].x0


def test_split_segment_keeps_plain_line() -> None:
    """Строка корпуса одного кегля не режется."""
    glyphs = np.vstack([word(0, 5, 10, 100), word(60, 5, 10, 100), word(120, 5, 10, 101)])
    assert len(split_segment(_segment(glyphs))) == 1


@dataclass
class _Axis:
    """Ось-заглушка: только то, что читает ``clip_extension``."""

    glyphs: np.ndarray
    body_points: np.ndarray


def test_clip_extension_stops_at_foreign_glyph() -> None:
    """Продление оси шапки, задевшее глиф соседней строки, отменяется; в пустоту — остаётся."""
    glyphs = word(0, 6, 43, 190)
    axis = _Axis(glyphs, np.array([[0.0, 180.0], [190.0, 180.0]]))
    others = word(240, 4, 22, 190)
    assert clip_extension(axis, others, 300.0) == 190.0
    assert clip_extension(axis, np.zeros((0, 4)), 300.0) == 300.0
