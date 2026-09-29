"""Замена осей-выбросов ``axes_fix.fixed_axes``: ось, лежащая на своих буквах, не заменяется; косая ось мимо букв — заменяется."""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.text_blocks import WORK_DPI
from ocr_utils.page_layout.text_blocks.axes_fix import fixed_axes, glyph_fit
from ocr_utils.page_layout.text_blocks.lines import LineAxis

# Корпус (пиксели рабочей копии): высота буквы, ширина, шаг строк, длина строк.
BODY_LETTER = 12.0
LETTER_W = 9.0
PITCH = 23.0
LENGTH = 600.0


def _axis(y0: float, letter: float, curve, axis_curve=None, height: float | None = None) -> LineAxis:
    """Строка из букв высотой ``letter``, середины которых идут по ``y0 + curve(x)``; ось — ``axis_curve`` (по умолчанию по буквам)."""
    xs = np.arange(50.0, 50.0 + LENGTH, LETTER_W + 3.0)
    mids = y0 + curve(xs)
    glyphs = np.column_stack([xs, mids - letter / 2, xs + LETTER_W, mids + letter / 2])
    grid = np.linspace(xs[0], xs[-1] + LETTER_W, 60)
    line = axis_curve or curve
    points = np.column_stack([grid, y0 + line(grid)])
    return LineAxis(
        points=points,
        height=height or letter * 1.4,
        column=0,
        dpi=WORK_DPI,
        cross=False,
        sagitta_mm=0.0,
        slope_deg=0.0,
        bend_mm=0.0,
        resid_parabola_mm=0.0,
        glyphs=glyphs,
        centre_points=points.copy(),
    )


def _flat(xs: np.ndarray) -> np.ndarray:
    return np.zeros_like(xs)


def _page(extra: LineAxis) -> list[LineAxis]:
    """Корпус из восьми прямых строк и проверяемая строка."""
    return [_axis(100.0 + i * PITCH, BODY_LETTER, _flat) for i in range(8)] + [extra]


def test_tilted_heading_on_its_letters_is_kept():
    """Заголовок крупным кеглем с настоящим наклоном 3°: ось по серединам букв не заменяется (1966/02 IMG_0077_1L)."""
    tilt = np.tan(np.radians(3.0))
    heading = _axis(40.0, 20.0, lambda xs: tilt * (xs - 50.0))
    axes, log = fixed_axes(_page(heading), WORK_DPI)
    assert log == []
    assert axes[-1] is heading


def test_curled_body_line_is_kept():
    """Строка корпуса на загибе бумаги (высота оси раздута изгибом, наклон −5°): не крупный набор, не заменяется."""
    curl = lambda xs: -np.clip(150.0 - (xs - 50.0), 0.0, None) * np.tan(np.radians(5.0))  # noqa: E731
    line = _axis(400.0, BODY_LETTER, curl, height=22.0)
    axes, log = fixed_axes(_page(line), WORK_DPI)
    assert log == []
    assert axes[-1] is line


def test_oblique_axis_off_its_letters_is_replaced():
    """Крупный набор, ось которого ушла наискось мимо букв (выносные тянут вторую ось): заменяется и ложится на буквы."""
    wrong = lambda xs: np.tan(np.radians(5.0)) * (xs - 50.0)  # noqa: E731
    heading = _axis(40.0, 20.0, _flat, axis_curve=wrong)
    axes, log = fixed_axes(_page(heading), WORK_DPI)
    assert len(log) == 1
    assert glyph_fit(axes[-1], axes[-1].points) < 0.1
