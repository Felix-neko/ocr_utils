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
    from research.geometry_regression.stretch import glyph_line_metrics

    metrics, culprits, _ = glyph_line_metrics(before, after, pairs, None, 150.0, 25.0)
    assert metrics["line_glyph_wobble_max"] > 0.05
    box = culprits["line_glyph_wobble_max"]["a"]
    assert 650 <= (box[1] + box[3]) / 2 <= 760  # виновник — строка в волне (150 dpi)


def test_same_page_lines_match_with_zero_delta():
    page = binarize(text_page())
    lines, separators = text_lines(page)
    pairs = match_lines(lines, lines, None)
    assert len(pairs) == len(lines)
    metrics = line_metrics(lines, lines, pairs)
    assert metrics["text_sag_gain_mm"] == 0.0 and metrics["lines_matched"] == len(lines)
    edges = column_edges(lines, separators, page.shape[1] // 2)
    assert any(e.side == "left" for e in edges)
    assert edge_metrics(edges, edges)[0]["edge_dev_max_delta_mm"] == 0.0


def test_stretched_heading_gives_wedge_in_mm():
    import cv2
    import numpy as np

    from research.geometry_regression.stretch import glyph_line_metrics

    page = text_page(lines=12, font_px=90, x0=200)  # крупный кегль: строки заведомо выше 4 мм
    before = binarize(page)
    # Клин: первая строка растянута по вертикали вокруг своей середины на 6 % к правому краю —
    # середина на месте, чтобы клин не выглядел доворотом строки (тот детектор прощает).
    h, w = page.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    band = (ys >= 280) & (ys < 400)
    stretch = 1.0 + 0.06 * (xs - 200) / 1600
    ys_src = np.where(band, 340 + (ys - 340) / stretch, ys)
    after = binarize(cv2.remap(page, xs, ys_src.astype(np.float32), cv2.INTER_LINEAR, borderValue=255))
    lines_b, _ = text_lines(before)
    lines_a, _ = text_lines(after)
    metrics, culprits, verified = glyph_line_metrics(
        before, after, match_lines(lines_b, lines_a, None), None, 150.0, 25.0
    )
    assert metrics["stretch_lines"] >= 5 and len(verified) >= 5
    assert (
        metrics["line_stretch_mm_max"] > 0.1
    )  # клин 6 % на строке ~5.5 мм ≈ 0.3 мм, часть съедает шаг сетки масштабов
    assert 150 <= culprits["line_stretch_mm_max"]["b"][1] <= 200  # виновник — растянутая строка (150 dpi)
    same, _, _ = glyph_line_metrics(before, before, match_lines(lines_b, lines_b, None), None, 150.0, 25.0)
    assert same["line_stretch_mm_max"] < 0.1
    assert same["line_tilt_dev_max_mm"] < 0.1 and same["line_glyph_wobble_max"] < 0.01
