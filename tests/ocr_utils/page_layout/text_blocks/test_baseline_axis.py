"""Вторая ось строки — по базовой линии: не прыгает на прописных и индексах, держит плотный набор и изгиб."""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.text_blocks.baseline_axis import line_x_height
from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
from ocr_utils.page_layout.text_blocks.metrics import crossings_of
from ocr_utils.page_layout.text_blocks.page import analyse_gray
from tests.ocr_utils.page_layout.text_blocks.synthetic import LETTER_X_H, bow_dy, glyph_line_page

# Рендер синтетики — 300 dpi, разбор — на рабочей копии 150 dpi.
K = 2.0


def _errors(analysis, tops: list[float], width: int, amplitude: float, which: str) -> np.ndarray:
    """Отклонения оси от истинной (пиксели рабочей копии) по всем строкам, ось — ``points`` или ``body_points``."""
    errors = []
    truth_levels = np.array([top + LETTER_X_H / 2.0 for top in tops]) / K
    for axis in analysis.axes:
        points = axis.points if which == "old" else axis.body_points
        assert points is not None
        xs = points[:, 0]
        bend = bow_dy(xs * K, width, amplitude) / K
        # Строка — та, чья истинная ось ближе к середине найденной.
        level = truth_levels[np.argmin(np.abs(truth_levels + np.median(bend) - np.median(points[:, 1])))]
        errors.append(np.abs(points[:, 1] - (level + bend)))
    return np.concatenate(errors)


def test_body_axis_follows_the_true_axis_through_capitals_and_indices():
    """На строках с прописными, выносными и индексами вторая ось ближе к истинной, чем ось по краске."""
    amplitude = 18.0
    image, tops = glyph_line_page(amplitude=amplitude)
    analysis = analyse_gray(image, InkEngine())
    # Строку сцепка может разрезать на куски — каждая ось сверяется с истинной осью своей строки.
    assert len(analysis.axes) >= len(tops)
    new = _errors(analysis, tops, image.shape[1], amplitude, "new")
    old = _errors(analysis, tops, image.shape[1], amplitude, "old")
    assert np.percentile(new, 95) < 0.6
    assert new.max() <= old.max()
    assert np.percentile(new, 95) < np.percentile(old, 95)


def test_dense_lines_do_not_cross():
    """Плотный набор («б» над «р» почти вплотную): вторые оси строк не скрещиваются и не сливаются."""
    image, tops = glyph_line_page(step=52, amplitude=10.0, seed=5)
    analysis = analyse_gray(image, InkEngine())
    bodies = [axis.body_points for axis in analysis.axes if axis.body_points is not None]
    assert len(bodies) >= len(tops)
    assert crossings_of(bodies) == 0


def test_line_x_height_ignores_capitals_and_descenders():
    """Высота строчной строки — по низшему «этажу» глифов: прописные и выносные её не завышают."""
    lower = [[0, 10, 8, 21]] * 6
    tall = [[0, 5, 8, 21], [0, 10, 8, 26]]
    glyphs = np.array(lower + tall, dtype=float)
    assert line_x_height(glyphs) == 11.0
