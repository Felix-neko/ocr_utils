"""Сайдкар геометрии разбора (``text_blocks.store``): запись массивов и чтение обратно без потерь."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from ocr_utils.page_layout.text_blocks import sides as sd
from ocr_utils.page_layout.text_blocks import store
from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
from ocr_utils.page_layout.text_blocks.page import analyse_gray
from tests.ocr_utils.page_layout.text_blocks.synthetic import column_page


@pytest.fixture(scope="module")
def analysis():
    """Разбор синтетической страницы в две колонки по формату; у первой оси — участок перескока."""
    result = analyse_gray(column_page(columns=2, justify="both"), InkEngine())
    axis = result.axes[0]
    jumped = replace(axis, jump_spans=((axis.x0 + 5.0, axis.x0 + 15.0),))
    blocks = tuple(
        replace(b, rows=tuple(replace(r, axes=tuple(jumped if a is axis else a for a in r.axes)) for r in b.rows))
        for b in result.blocks
    )
    return replace(result, axes=(jumped, *result.axes[1:]), blocks=blocks)


def _lines(analysis) -> list[dict]:
    """Дополнительные линии сторон, как у разбора пака (``final.side_lines``)."""
    out = []
    for block in analysis.blocks:
        sides = sd.sides_of(block)
        own = {}
        for side in (sd.SideKind.LEFT, sd.SideKind.RIGHT):
            alignment = sd.side_alignment(block, side, sd.AlignMethod.ROBUST)
            own[side] = None if alignment is None else sd.filled_side(sides, alignment, block)
        out.append(own)
    return out


def test_arrays_round_trip(analysis, tmp_path):
    """Оси, их блоки и ряды, перескоки и стороны читаются обратно теми же, что были записаны."""
    lines = _lines(analysis)
    store.write_arrays(tmp_path / "p.npz", store.page_arrays(analysis, lines))
    payload = {
        "page": "синтетика",
        "size": [analysis.width, analysis.height],
        "dpi": analysis.dpi,
        "text_blocks": {
            "dpi": analysis.dpi,
            "arrays": "p.npz",
            "blocks": [
                {"polygon": np.asarray(b.envelope.polygon).tolist(), "rows": len(b.rows),
                 "alignment": {"kind": "both", "left": True, "right": True}}
                for b in analysis.blocks
            ],
        },
    }  # fmt: skip
    (tmp_path / "p.json").write_text(json.dumps(payload))
    page = store.load_page(tmp_path / "p.json")
    assert len(page.axes) == len(analysis.axes)
    for record, axis in zip(page.axes, analysis.axes):
        assert np.allclose(record.points, axis.points, atol=1e-3)
        assert np.allclose(np.reshape(record.jump_spans, (-1, 2)), np.reshape(axis.jump_spans, (-1, 2)))
    assert page.axes[0].jump_spans
    # Каждая ось, вошедшая в ряд, помечена своим блоком и рядом.
    assert all(record.block >= 0 and record.row >= 0 for record in page.axes)
    for block_index, block in enumerate(analysis.blocks):
        record = page.blocks[block_index]
        assert record.rows == len(block.rows) and record.aligned_left and record.aligned_right
        for code, kind in ((store.SideCode.LEFT, sd.SideKind.LEFT), (store.SideCode.RIGHT, sd.SideKind.RIGHT)):
            side = record.sides[code]
            raw = block.envelope.left if kind is sd.SideKind.LEFT else block.envelope.right
            assert np.allclose(side.raw, raw, atol=1e-3)
            line = lines[block_index][kind]
            assert len(side.points) == (0 if line is None else len(line.points))
            assert len(side.kinds) == len(side.points)


def test_point_kinds_marks_patches_over_anomalies():
    """Заплатка в недостоверном участке — ANOMALY, заплатка вне его — GAP, своя точка — ALIGNED."""
    points = np.column_stack([np.zeros(6), np.arange(6, dtype=float) * 10])
    line = sd.FilledSide(sd.SideKind.LEFT, sd.AlignMethod.ROBUST, points,
                         np.array([False, True, True, False, True, False]), 5.0, 0.0, 0.0, 0.0, 0.0)  # fmt: skip
    kinds = store.point_kinds(line, ((15.0, 25.0),))
    assert kinds.tolist() == [store.PointKind.ALIGNED, store.PointKind.GAP, store.PointKind.ANOMALY,
                              store.PointKind.ALIGNED, store.PointKind.GAP, store.PointKind.ALIGNED]  # fmt: skip
