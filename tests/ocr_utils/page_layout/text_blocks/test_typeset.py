"""Тесты запретов сращивания строк разного набора на синтетических боксах глифов: разный набор, резка строки, ряды, продление оси, линейка."""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.text_blocks.barriers import BarrierLines
from ocr_utils.page_layout.text_blocks.baseline_axis import _foreign_glyphs, _hits_foreign
from ocr_utils.page_layout.text_blocks.lines import LineAxis
from ocr_utils.page_layout.text_blocks.segment import Segment, split_segment
from ocr_utils.page_layout.text_blocks.typeset import barrier_between, different_sets


def word(x0: float, count: int, height: float, bottom: float) -> np.ndarray:
    """Слово из ``count`` одинаковых глифов высоты ``height`` на базовой линии ``bottom``: боксы ``x0, y0, x1, y1``."""
    width, gap = 0.6 * height, 0.15 * height
    lefts = x0 + np.arange(count) * (width + gap)
    return np.column_stack([lefts, np.full(count, bottom - height), lefts + width, np.full(count, bottom)])


def axis_of(glyphs: np.ndarray) -> LineAxis:
    """Ось по боксам глифов: прямая по серединам от первого глифа до последнего."""
    xs = np.array([glyphs[:, 0].min(), glyphs[:, 2].max()])
    y = float(np.median((glyphs[:, 1] + glyphs[:, 3]) / 2.0))
    return LineAxis(
        points=np.column_stack([xs, [y, y]]),
        height=float(np.median(glyphs[:, 3] - glyphs[:, 1])),
        column=0,
        dpi=150,
        cross=False,
        sagitta_mm=0.0,
        slope_deg=0.0,
        bend_mm=0.0,
        resid_parabola_mm=0.0,
        glyphs=glyphs,
    )


def test_same_baseline_logo_is_one_line() -> None:
    """«50 лет»: цифры вдвое выше слова, но на одной базовой линии — не разный набор."""
    assert not different_sets(word(0, 2, 32, 140), word(80, 3, 16, 140), gap=41.0)


def test_masthead_and_small_line_differ() -> None:
    """Шапка (кегль 54) и «Год» (17) с низами на 14 px выше через 30 px — разный набор."""
    assert different_sets(word(0, 6, 54, 290), word(400, 3, 17, 276), gap=30.0)


def test_body_text_is_never_cut() -> None:
    """Корпус (крупная сторона меньше 18 px) не режется, даже с разными низами: старинные цифры уходят под строку."""
    assert not different_sets(word(0, 4, 17, 1031), word(100, 3, 7, 1026), gap=10.0)


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
    assert [len(part.glyphs) for part in parts] == [8, 5]
    assert parts[0].x1 < parts[1].x0
    assert parts[1].height < parts[0].height


def test_split_segment_keeps_heading_with_wide_spaces() -> None:
    """Заголовок одного кегля с широкими пробелами на одной линии не режется."""
    glyphs = np.vstack([word(0, 5, 30, 100), word(200, 5, 30, 101), word(400, 5, 30, 100)])
    assert len(split_segment(_segment(glyphs))) == 1


def test_extension_stops_at_glyph_of_another_line() -> None:
    """Продление оси шапки задевает «Год издания» на другой линии — отменяется; половинка той же строки — не чужая."""
    masthead, small, half = word(0, 6, 43, 190), word(240, 4, 22, 167), word(240, 4, 43, 190)
    foreign = _foreign_glyphs([masthead, small], 0)
    assert _hits_foreign(foreign, 190.0, 300.0, 170.0, 0.6 * 43)
    assert len(_foreign_glyphs([masthead, half], 0)) == 0


def test_barrier_between_ignores_short_stroke_and_finds_toc_rule() -> None:
    """Линейка оглавления, вошедшая «глифом» в строку, разделяет; короткий штрих буквы — нет."""
    left, right = word(100, 6, 16, 300), word(260, 6, 10, 300)
    rule_glyph = np.array([[250.0, 0.0, 252.0, 900.0]])
    left_axis, right_axis = axis_of(np.vstack([left, rule_glyph])), axis_of(right)
    long_rule = BarrierLines.of([np.array([[251.0, 0.0], [251.0, 900.0]])])
    short_stroke = BarrierLines.of([np.array([[251.0, 280.0], [251.0, 320.0]])])
    assert barrier_between([left_axis], [right_axis], 292.0, 295.0, long_rule)
    assert not barrier_between([left_axis], [right_axis], 292.0, 295.0, short_stroke)
