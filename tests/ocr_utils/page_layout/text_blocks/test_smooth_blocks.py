"""Тесты переключателя способов сборки блоков: ``SMOOTH`` (гладкие стороны, по умолчанию) и ``LEGACY`` (прежний ход) на синтетических колонках."""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.text_blocks.blocks import DEFAULT_BLOCKS_MODE, BlocksMode
from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
from ocr_utils.page_layout.text_blocks.page import analyse_gray
from ocr_utils.page_layout.text_blocks.smooth_envelope import SmoothEnvelope
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
