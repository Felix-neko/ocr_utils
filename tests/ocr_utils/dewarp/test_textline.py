"""Движок textline на синтетике: кривая полоса становится заметно ровнее."""

from __future__ import annotations

import cv2
import pytest

from ocr_utils.dewarp import compare, quality
from ocr_utils.dewarp.engines.textline import TextLineEngine
from tests.ocr_utils.dewarp.synthetic import curl_page, sine_page
from tests.ocr_utils.page_layout.orientation.synthetic import text_page

DPI = 300


def _bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize("maker", [curl_page, sine_page])
def test_textline_straightens_synthetic_warp(maker):
    warped, _ = maker()
    engine = TextLineEngine()
    engine.load("cpu")
    engine.dpi = DPI
    result = engine.dewarp(_bgr(warped))
    assert result is not None and result.shape == (*warped.shape, 3)
    assert engine.last_field is not None and engine.last_field.lines >= 8

    before = quality.measure(_bgr(warped), DPI)
    after = quality.measure(result, DPI)
    assert after["line_fit.sagitta_rel_p90"] < 0.5 * before["line_fit.sagitta_rel_p90"]
    assert after["skew_map.max_dev_deg"] < before["skew_map.max_dev_deg"]


def test_textline_leaves_straight_page_alone():
    page = text_page()
    engine = TextLineEngine()
    engine.load("cpu")
    engine.dpi = DPI
    result = engine.dewarp(_bgr(page))
    field = engine.last_field
    # На прямой полосе поле — почти ноль: сдвиги меньше пикселя копии 150 dpi.
    assert field is not None
    assert abs(field.grid).max() < 1.0
    assert quality.measure(result, DPI)["line_fit.sagitta_rel_p90"] < 0.15


def test_side_by_side_shape():
    page = text_page()
    pair = compare.side_by_side(_bgr(page), _bgr(page), height=400)
    assert pair.shape[0] == 400
    assert pair.shape[1] > 2 * 400 * page.shape[1] / page.shape[0]
