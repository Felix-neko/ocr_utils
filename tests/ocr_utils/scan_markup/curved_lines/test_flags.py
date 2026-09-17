from __future__ import annotations

import pytest

from ocr_utils.scan_markup.curved_lines import flags
from ocr_utils.scan_markup.curved_lines.detectors.base import Detector, Measure, silent


def _detector(name: str, thresholds: dict[str, float]) -> Detector:
    return Detector(name=name, summary="", stage="cpu", thresholds=thresholds, run=lambda frame, keep_raw: Measure())


def test_score_is_max_ratio_over_flag_metrics_only():
    thresholds = flags.Thresholds.from_detectors([_detector("a", {"x": 2.0, "y": 10.0})])
    measure = thresholds.apply("a", Measure(metrics={"x": 1.0, "y": 25.0, "z": 1000.0}))
    assert measure.score == pytest.approx(2.5)
    assert measure.flag


def test_silent_measure_never_flags():
    thresholds = flags.Thresholds.from_detectors([_detector("a", {"x": 1.0})])
    assert not thresholds.apply("a", silent("мало строк")).flag


def test_override_replaces_default_and_rejects_typos():
    detector = _detector("line_fit", {"sagitta_rel_p90": 0.3})
    thresholds = flags.Thresholds.from_detectors([detector], ["line_fit.sagitta_rel_p90=0.5"])
    assert thresholds.values["line_fit"]["sagitta_rel_p90"] == 0.5
    with pytest.raises(flags.ThresholdError):
        flags.Thresholds.from_detectors([detector], ["line_fit.nope=1"])
    with pytest.raises(flags.ThresholdError):
        flags.Thresholds.from_detectors([detector], ["other.sagitta_rel_p90=1"])
    with pytest.raises(flags.ThresholdError):
        flags.Thresholds.from_detectors([detector], ["line_fit.sagitta_rel_p90=abc"])


def test_combo_votes_or_strong():
    weak_yes = Measure(metrics={}, flag=True, score=1.1)
    no = Measure(metrics={}, flag=False, score=0.3)
    strong_yes = Measure(metrics={}, flag=True, score=2.0)
    assert not flags.combine({"a": weak_yes, "b": no}, votes=2, strong=1.5).flag
    assert flags.combine({"a": weak_yes, "b": weak_yes}, votes=2, strong=1.5).flag
    assert flags.combine({"a": strong_yes, "b": no}, votes=2, strong=1.5).flag
    assert flags.combine({"a": silent("x")}, votes=1, strong=1.5).silent
    # Детектор без права одиночного голоса: сильный score не даёт свод, два голоса — дают.
    assert not flags.combine({"a": strong_yes, "b": no}, votes=2, strong=1.5, solo=["b"]).flag
    assert flags.combine({"a": strong_yes, "b": weak_yes}, votes=2, strong=1.5, solo=["b"]).flag


def test_sufficient_detector_flags_combo_alone():
    weak_yes = Measure(metrics={}, flag=True, score=1.1)
    no = Measure(metrics={}, flag=False, score=0.3)
    assert not flags.combine({"a": weak_yes, "b": no}, votes=2, strong=2.5).flag
    assert flags.combine({"a": weak_yes, "b": no}, votes=2, strong=2.5, sufficient=["a"]).flag
    assert not flags.combine({"a": no, "b": no}, votes=2, strong=2.5, sufficient=["a"]).flag
