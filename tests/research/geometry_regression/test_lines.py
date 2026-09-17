from research.geometry_regression.edges import column_edges, edge_metrics
from research.geometry_regression.lines import line_metrics, match_lines
from research.geometry_regression.regions import text_lines
from tests.research.geometry_regression.synthetic import binarize, text_page, wave_region


def test_waved_line_raises_wobble():
    page = text_page()
    before = binarize(page)
    after = binarize(wave_region(page, 1400, 1460, amplitude_px=6.0, period_px=250.0))
    lines_b, _ = text_lines(before)
    lines_a, _ = text_lines(after)
    pairs = match_lines(lines_b, lines_a, None)
    assert len(pairs) >= 30
    metrics, culprits = line_metrics(lines_b, lines_a, pairs, 25.0)
    assert metrics["line_wobble_delta_max"] > 0.05
    box = culprits["line_wobble_delta_max"]["a"]
    assert 650 <= (box[1] + box[3]) / 2 <= 760  # виновник — строка в волне (150 dpi)


def test_same_page_lines_match_with_zero_delta():
    page = binarize(text_page())
    lines, separators = text_lines(page)
    pairs = match_lines(lines, lines, None)
    assert len(pairs) == len(lines)
    metrics, _ = line_metrics(lines, lines, pairs, 25.0)
    assert metrics["line_dev_max_delta_mm"] == 0.0
    edges = column_edges(lines, separators, page.shape[1] // 2)
    assert any(e.side == "left" for e in edges)
    assert edge_metrics(edges, edges)[0]["edge_dev_max_delta_mm"] == 0.0
