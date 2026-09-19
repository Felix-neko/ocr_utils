
from ocr_utils.geometry_regression.strokes import find_strokes, match_strokes, stroke_metrics
from tests.ocr_utils.geometry_regression.synthetic import add_rules, binarize, text_page

RULES = [(300, 2500, 1500, 2500), (300, 2700, 1500, 2700), (1600, 400, 1600, 2600)]  # вертикаль в 34 мм от края


def test_tilted_rule_gives_deviation_in_mm():
    before = binarize(add_rules(text_page(lines=20), RULES))
    # ~1.15° на 1200 px (как линейка 50 мм под 1.2° на 1967/01 с.38, только длиннее); при
    # большем наклоне середина линейки уходит дальше допуска сопоставления MATCH_PERP_MM.
    tilted = RULES[:1] + [(300, 2700, 1500, 2700 + 24)] + RULES[2:]
    after = binarize(add_rules(text_page(lines=20), tilted))
    strokes_b, strokes_a = find_strokes(before, 8, 300), find_strokes(after, 8, 300)
    pairs = match_strokes(strokes_b, strokes_a, None, 300, 150)
    assert len(pairs) >= 3
    metrics, culprits = stroke_metrics(strokes_b, strokes_a, pairs, 0.0, 300)
    expected = 24 / 300 * 25.4  # уход конца ≈ 2.0 мм
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


def test_lone_vertical_near_edge_is_dropped_but_frame_is_kept():
    """Одиночная вертикаль у края кадра (тень корешка) — не штрих; рамка с двух сторон — штрихи."""
    page = text_page(lines=10)
    shadow = binarize(add_rules(page, [(120, 300, 122, 2700)], thickness=6))  # 10 мм от левого края
    strokes = find_strokes(shadow, 8, 300)
    assert not [s for s in strokes if abs(s.angle_deg) > 80 and s.length > 1000]
    frame = binarize(add_rules(page, [(120, 300, 122, 2700), (1880, 300, 1882, 2700)], thickness=6))
    strokes = find_strokes(frame, 8, 300)
    assert len([s for s in strokes if abs(s.angle_deg) > 80 and s.length > 1000]) >= 2  # LSD даёт обе кромки линейки


def test_short_pencil_mark_on_margin_is_dropped():
    page = binarize(add_rules(text_page(lines=10), [(140, 1000, 142, 1120)], thickness=5))  # 10 мм на поле
    assert not [s for s in find_strokes(page, 8, 300) if abs(s.angle_deg) > 80]
