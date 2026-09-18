"""CPU-детекторы на синтетике: прямая полоса ниже порогов, кривые — выше."""

from __future__ import annotations

import json

import pytest

from ocr_utils.scan_markup.curved_lines.detectors import DETECTORS
from ocr_utils.scan_markup.curved_lines.flags import Thresholds
from tests.ocr_utils.scan_markup.curved_lines.synthetic import (
    bow_page,
    curl_page,
    frame_of,
    sine_page,
    straight_page,
    tilted_block_page,
)

CPU = ("skew_map", "line_fit", "strip_shift")


def _measure(name: str, image):
    detector = DETECTORS[name]
    measure = detector.run(frame_of(image), True)
    return Thresholds.from_detectors([detector]).apply(name, measure)


@pytest.mark.parametrize("name", CPU)
def test_straight_page_is_below_thresholds(name):
    measure = _measure(name, straight_page())
    assert not measure.silent, measure.note
    assert not measure.flag, measure.metrics


@pytest.mark.parametrize("name", CPU)
def test_bow_page_is_flagged(name):
    measure = _measure(name, bow_page())
    assert not measure.silent, measure.note
    assert measure.flag, measure.metrics


def test_line_fit_sees_sagitta_of_the_bow():
    curved = _measure("line_fit", bow_page(amplitude_px=30.0))
    straight = _measure("line_fit", straight_page())
    # Дуга 30 px при высоте строки около 32 px (300 dpi): прогиб порядка высоты.
    assert curved.metrics["sagitta_rel_p90"] > 0.5
    assert curved.metrics["sagitta_rel_p90"] > 3 * straight.metrics["sagitta_rel_p90"]


def test_skew_map_sees_tilted_block_without_curvature():
    measure = _measure("skew_map", tilted_block_page(angle_deg=3.0))
    assert measure.flag
    # Половина тайлов под 0°, половина под 3°: медиана между ними, зато размах — почти 3°.
    assert measure.metrics["spread_deg"] >= 2.0


def test_sine_page_is_seen_by_skew_map_but_not_by_line_fit():
    """Волна, одинаковая на всех строках, — предел line_fit: парабола через две волны почти
    прямая, а разброса наклонов между одинаковыми строками нет. Карта углов видит её по
    тайлам, где фаза волны разная."""
    assert _measure("skew_map", sine_page()).flag
    measure = _measure("line_fit", sine_page())
    assert not measure.flag
    assert measure.metrics["resid_lin_rel_p90"] > 2 * _measure("line_fit", straight_page()).metrics["resid_lin_rel_p90"]


def test_curl_page_raises_sagitta():
    measure = _measure("line_fit", curl_page())
    assert measure.metrics["sagitta_rel_max3"] > 0.25


@pytest.mark.parametrize("name", CPU)
def test_raw_is_json_serializable(name):
    measure = DETECTORS[name].run(frame_of(bow_page()), True)
    assert measure.raw is not None
    json.dumps(measure.raw)


@pytest.mark.parametrize("name", CPU)
def test_without_keep_raw_there_is_no_raw(name):
    assert DETECTORS[name].run(frame_of(straight_page()), False).raw is None


def test_end_curl_flags_curled_line_ends_but_not_straight_page():
    curled = _measure("end_curl", curl_page(amplitude_px=40.0, width_px=400.0))
    straight = _measure("end_curl", straight_page())
    assert not curled.silent and not straight.silent
    assert curled.flag, curled.metrics
    assert not straight.flag, straight.metrics
    assert curled.metrics["end_slope_deg"] > 3 * straight.metrics["end_slope_deg"]


def test_end_curl_raw_is_json_serializable():
    measure = DETECTORS["end_curl"].run(frame_of(curl_page()), True)
    json.dumps(measure.raw)
