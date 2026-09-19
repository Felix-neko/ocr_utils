import cv2
import numpy as np

from research.geometry_regression.bend import bend_metrics
from research.geometry_regression.strokes import find_strokes
from tests.research.geometry_regression.synthetic import add_rules, binarize, text_page

RULE = (300, 2500, 1500, 2500)  # линейка 100 мм при 300 dpi


def _wave_page(amplitude_px: float) -> np.ndarray:
    """Та же страница, но линейка волной: середина ушла вниз на ``amplitude_px``."""
    page = text_page(lines=20).copy()
    x0, y, x1 = RULE[0], RULE[1], RULE[2]
    xs = np.arange(x0, x1 + 1, dtype=np.float64)
    ys = y + amplitude_px * np.sin(np.pi * (xs - x0) / (x1 - x0))
    points = np.stack([xs, ys], axis=1).round().astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(page, [points], False, 0, thickness=4, lineType=cv2.LINE_AA)
    return binarize(page)


def test_bent_line_gives_sagitta_in_mm():
    before = binarize(add_rules(text_page(lines=20), [RULE]))
    after = _wave_page(amplitude_px=12)  # ≈ 1 мм при 300 dpi
    strokes_b = find_strokes(before, 8, 300)
    metrics, culprits = bend_metrics(before, after, strokes_b, None, 300, 150)
    assert metrics["stroke_bend_lines"] >= 1
    assert 0.8 < metrics["stroke_bend_dev_mm"] < 1.4
    assert len(culprits["stroke_bend_dev_mm"]["segments_a"]) > 5


def test_straight_line_in_both_versions_is_flat():
    page = binarize(add_rules(text_page(lines=20), [RULE]))
    strokes_b = find_strokes(page, 8, 300)
    metrics, _ = bend_metrics(page, page, strokes_b, None, 300, 150)
    assert metrics["stroke_bend_lines"] >= 1
    assert metrics["stroke_bend_dev_mm"] < 0.2
