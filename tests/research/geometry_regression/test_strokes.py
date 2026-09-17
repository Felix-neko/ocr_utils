import numpy as np

from research.geometry_regression.strokes import find_strokes, match_strokes, stroke_metrics
from tests.research.geometry_regression.synthetic import add_rules, binarize, text_page

RULES = [(300, 2500, 1500, 2500), (300, 2700, 1500, 2700), (1700, 400, 1700, 2600)]


def test_tilted_rule_gives_deviation_in_mm():
    before = binarize(add_rules(text_page(lines=20), RULES))
    tilted = RULES[:1] + [(300, 2700, 1500, 2700 + 42)] + RULES[2:]  # ~2° на 1200 px
    after = binarize(add_rules(text_page(lines=20), tilted))
    strokes_b, strokes_a = find_strokes(before, 8, 300), find_strokes(after, 8, 300)
    pairs = match_strokes(strokes_b, strokes_a, None, 300, 150)
    assert len(pairs) >= 3
    metrics, culprits = stroke_metrics(strokes_b, strokes_a, pairs, 0.0, 300)
    expected = 1200 / 300 * 25.4 * np.sin(np.radians(2.0))  # ≈ 3.5 мм
    assert abs(metrics["hstroke_dev_max_delta_mm"] - expected) < 0.6
    assert abs(metrics["vstroke_dev_max_delta_mm"]) < 0.2
    assert "hstroke_dev_max_delta_mm" in culprits


def test_parallel_group_spread_grows_when_one_line_turns():
    parallel = [(300, 1000 + i * 150, 1500, 1000 + i * 150) for i in range(4)]
    before = binarize(add_rules(text_page(lines=0), parallel))
    turned = parallel[:3] + [(300, 1450, 1500, 1450 + 30)]
    after = binarize(add_rules(text_page(lines=0), turned))
    strokes_b, strokes_a = find_strokes(before, 8, 300), find_strokes(after, 8, 300)
    pairs = match_strokes(strokes_b, strokes_a, None, 300, 150)
    metrics, _ = stroke_metrics(strokes_b, strokes_a, pairs, 0.0, 300)
    assert metrics["parallel_groups"] >= 1
    assert metrics["parallel_spread_delta_max"] > 1.0


def test_unchanged_page_has_zero_deltas():
    page = binarize(add_rules(text_page(lines=10), RULES))
    strokes = find_strokes(page, 8, 300)
    pairs = match_strokes(strokes, strokes, None, 300, 150)
    metrics, _ = stroke_metrics(strokes, strokes, pairs, 0.0, 300)
    assert metrics["strokes_lost_frac"] == 0.0
    assert metrics["hstroke_dev_max_delta_mm"] == 0.0
    assert metrics["parallel_spread_delta_max"] == 0.0
