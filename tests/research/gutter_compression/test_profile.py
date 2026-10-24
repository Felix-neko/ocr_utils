"""Тесты поправки на место в строке."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.gutter_compression.profile import LinePosition, correct_position, spine_position


def test_spine_position_by_side() -> None:
    # У корешка левой страницы — концы строк, правой — начала.
    assert spine_position("L") is LinePosition.END
    assert spine_position("R") is LinePosition.START


def test_correct_position_divides_by_bias_of_spine_side() -> None:
    bias = pd.DataFrame({"position": ["start", "start", "end", "end"], "bin": [0.0, 1.0, 0.0, 1.0],
                         "acf_edge": [1.05, 1.0, 0.95, 1.0]})  # fmt: skip
    table = pd.DataFrame({"side": ["L", "R", "R"], "from_edge_mm": [0.5, 0.5, 1.5], "acf_ratio": [0.95, 1.05, 0.8]})
    out = correct_position(table, bias)
    assert np.allclose(out.acf_ratio, [1.0, 1.0, 0.8])
    assert correct_position(table, None) is table
