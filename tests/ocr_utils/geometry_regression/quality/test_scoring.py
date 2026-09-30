"""Правила вердикта v17 (``ocr_utils.geometry_regression.quality.scoring``): группы, жёсткие пороги, совокупность, гистерезис."""

from __future__ import annotations

from ocr_utils.geometry_regression.quality.scoring import Group, Rule, Thresholds17, Verdict


def test_clean_page_is_ok():
    """Все метрики ниже порогов и сумма групп ниже S — ok."""
    result = Thresholds17().assess({"line_quality_mm": 0.3, "edge_quality_mm": 0.2})
    assert result.verdict is Verdict.OK and result.rule is Rule.CLEAN


def test_related_metrics_count_once_in_their_group():
    """Край блока и вертикальная линейка — одно явление (перекос): в сумму группа идёт максимумом, а не суммой."""
    metrics = {"edge_quality_mm": 0.9, "vstroke_dev_max_delta_mm": 0.63, "vstroke_pairs": 5}
    result = Thresholds17().assess(metrics)
    assert result.groups[Group.VERTICALS] == 0.9
    assert result.total == 0.9


def test_single_line_gain_does_not_forgive_lineart():
    """Выпрямленный заголовок не прощает порчу не-текста: при разошедшихся линейках без другого выигрыша — bad."""
    metrics = {"parallel_spread_other": 2.0, "line_quality_gain_mm": 9.0}
    result = Thresholds17(total=10.0).assess(metrics)
    assert result.gain > 5 and result.gain_other == 0.0
    assert result.verdict is Verdict.BAD and result.rule is Rule.NO_GAIN


def test_total_over_groups_rolls_back_without_any_flag():
    """Сумма групп ≥ S при score каждой метрики < 1 — bad «совокупность»."""
    metrics = {"line_quality_mm": 0.7, "edge_quality_mm": 0.9, "stroke_bend_dev_mm": 0.7}
    result = Thresholds17(total=2.5).assess(metrics)
    assert max(s.score for s in result.scores.values()) < 1.0
    assert result.verdict is Verdict.BAD and result.rule is Rule.TOTAL


def test_individual_hard_threshold():
    """Метрика, дошедшая до своего жёсткого порога, — bad при любом выигрыше."""
    thresholds = Thresholds17(hard_overrides={"edge_quality_mm": 2.0})
    result = thresholds.assess({"edge_quality_mm": 2.1, "edge_quality_gain_mm": 30.0})
    assert result.verdict is Verdict.BAD and result.rule is Rule.HARD and result.culprit == "edge_quality_mm"


def test_hysteresis_keeps_finereader_when_gain_dominates():
    """Порча явно меньше выигрыша — mixed; сравнимая — bad."""
    thresholds = Thresholds17(total=10.0)
    mixed = thresholds.assess({"line_quality_mm": 0.9, "lines_quality_gain_mm": 1.2})  # порча 1.1, выигрыш 3
    assert mixed.verdict is Verdict.MIXED
    bad = thresholds.assess({"line_quality_mm": 0.9, "lines_quality_gain_mm": 0.5})  # порча 1.1, выигрыш 1.25
    assert bad.verdict is Verdict.BAD and bad.rule is Rule.HYSTERESIS


def test_unforgivable_ignores_gain():
    """Непрощаемая метрика ≥ порога — bad при любом выигрыше."""
    result = Thresholds17().assess({"lineart_bend_mm": 0.9, "lines_quality_gain_mm": 10.0})
    assert result.verdict is Verdict.BAD and result.rule is Rule.UNFORGIVABLE


def test_total_is_forgiven_by_a_large_gain():
    """Сумма групп ≥ S, но меньше ratio × выигрыш — совокупность не срабатывает (законная большая правка)."""
    metrics = {"line_quality_mm": 0.7, "edge_quality_mm": 0.9, "stroke_bend_dev_mm": 0.7, "lines_quality_gain_mm": 2.0}
    result = Thresholds17(total=2.5).assess(metrics)
    assert result.total >= 2.5 and result.verdict is Verdict.OK


def test_no_gain_rolls_back():
    """Порча ≥ 1 без выигрыша — bad."""
    result = Thresholds17(total=10.0).assess({"edge_quality_mm": 1.2})
    assert result.verdict is Verdict.BAD and result.rule is Rule.NO_GAIN
