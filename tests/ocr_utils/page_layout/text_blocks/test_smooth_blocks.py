"""Тесты переключателя способов сборки блоков: ``SMOOTH`` (гладкие стороны, по умолчанию) и ``LEGACY`` (прежний ход) на синтетических колонках."""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.text_blocks.blocks import DEFAULT_BLOCKS_MODE, BlocksMode, Row
from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
from ocr_utils.page_layout.text_blocks.page import analyse_gray
from ocr_utils.page_layout.text_blocks.smooth_envelope import SmoothEnvelope, is_trapezoid
from tests.ocr_utils.page_layout.text_blocks.synthetic import column_page

ENGINE = InkEngine()


def test_default_mode_is_smooth() -> None:
    """Способ по умолчанию — гладкий; прежний выбирается переключателем."""
    assert DEFAULT_BLOCKS_MODE is BlocksMode.SMOOTH


def test_justified_column_has_straight_smooth_sides() -> None:
    """Колонка по формату: у гладкой границы боковые стороны — прямые без ступенек, все концы осей внутри."""
    analysis = analyse_gray(column_page(columns=1, justify="both"), ENGINE)
    block = max(analysis.blocks, key=lambda item: len(item.rows))
    assert isinstance(block.envelope, SmoothEnvelope)
    for side in (block.envelope.left, block.envelope.right):
        inner = side[1:-1]  # без точек стыка с крышками
        assert np.ptp(inner[:, 0]) < 2.0
    polygon = block.envelope.polygon
    for row in block.rows:
        for axis in row.axes:
            x0, x1 = axis.points[0, 0], axis.points[-1, 0]
            assert polygon[:, 0].min() <= x0 + 1.0 and polygon[:, 0].max() >= x1 - 1.0


def test_modes_switch() -> None:
    """``LEGACY`` даёт прежнюю огибающую со справочной кромкой по краске, ``SMOOTH`` — гладкую без неё."""
    page = column_page(columns=2, justify="both")
    legacy = analyse_gray(page, ENGINE, blocks_mode=BlocksMode.LEGACY)
    smooth = analyse_gray(page, ENGINE, blocks_mode=BlocksMode.SMOOTH)
    assert legacy.blocks and smooth.blocks
    assert all(not isinstance(block.envelope, SmoothEnvelope) for block in legacy.blocks)
    assert all(block.envelope_ink is not None for block in legacy.blocks)
    assert all(block.envelope_ink is None for block in smooth.blocks)
    # Колонки не сливаются ни одним способом.
    assert len(smooth.blocks) >= 2


def test_trapezoid_heading_vs_skewed_column() -> None:
    """Стороны расходятся на 30° (заголовок трапецией) — трапеция; обе наклонены на 5° (поворот скана) — нет."""
    rows = [
        Row(y=y, height=12.0, x0=100.0 + 0.27 * (y - 100), x1=700.0 - 0.27 * (y - 100), axes=())
        for y in (100.0, 130.0, 160.0, 190.0)
    ]
    assert is_trapezoid(rows, None, None)
    tilt = np.tan(np.radians(5.0))
    skewed = [
        Row(y=y, height=12.0, x0=100.0 + tilt * y, x1=700.0 + tilt * y, axes=()) for y in (100.0, 130.0, 160.0, 190.0)
    ]
    assert not is_trapezoid(skewed, None, None)


def test_diverging_ragged_sides_are_ragged_not_center() -> None:
    """Рваный набор, чьи края легли на расходящиеся прямые, — ``ragged``, а не ``center``: середины строк гуляют."""
    from ocr_utils.page_layout.text_blocks.alignment import AlignKind, is_centered, verdict

    rng = np.random.default_rng(1)
    rows = []
    for index, y in enumerate(np.arange(100.0, 400.0, 30.0)):
        x0 = 100.0 + 0.27 * (y - 100) + rng.uniform(-30, 30)
        x1 = 700.0 - 0.27 * (y - 100) + rng.uniform(-80, 10) - (200.0 if index % 2 else 0.0)
        rows.append(Row(y=float(y), height=12.0, x0=float(x0), x1=float(x1), axes=()))
    assert is_trapezoid(rows, None, None)
    assert verdict(False, False, is_centered(rows, 150.0)) is AlignKind.RAGGED
