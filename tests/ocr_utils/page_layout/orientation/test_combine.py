"""Сведение мнений: ось голосованием, сторона — первым уверенным."""

from __future__ import annotations

from ocr_utils.page_layout.orientation.analysis import combine
from ocr_utils.page_layout.orientation.detectors import Verdict


def test_axis_wins_by_weight_not_by_count():
    """Два неуверенных голоса не перевешивают один уверенный: иначе слабые меры,
    которых в наборе большинство, диктовали бы ответ."""
    verdict, _, _ = combine(
        {
            "ink_axis": Verdict(90, 0.95, axis_only=True),
            "profile": Verdict(0, 0.30, axis_only=True),
            "surya_lines": Verdict(0, 0.30, axis_only=True),
        }
    )
    assert verdict.rotate_cw == 90


def test_side_is_taken_from_the_arbiter_first():
    verdict, source, disputed = combine(
        {
            "ink_axis": Verdict(270, 0.40),
            "surya_lines": Verdict(90, 0.90, axis_only=True),
            "ocr_vote": Verdict(90, 0.85),
        }
    )
    assert (verdict.rotate_cw, source) == (90, "ocr_vote")
    assert not disputed, "слабое возражение не делает полосу спорной"


def test_the_arbiter_outranks_a_more_confident_heuristic():
    """Арбитр читает полосу на каждом повороте, остальные судят косвенно — поэтому его
    ответ не голосуется наравне с ними, даже когда те увереннее."""
    verdict, source, disputed = combine({"ocr_vote": Verdict(90, 0.8), "osd": Verdict(270, 0.9)})
    assert (verdict.rotate_cw, source) == (90, "ocr_vote")
    assert not disputed


def test_equally_confident_heuristics_disagree_without_an_arbiter():
    verdict, _, disputed = combine({"osd": Verdict(90, 0.8), "doctr": Verdict(270, 0.9)})
    assert disputed


def test_axis_seen_but_side_unknown_stays_axis_only():
    verdict, source, disputed = combine({"surya_lines": Verdict(90, 0.9, axis_only=True)})
    assert verdict.axis_only and verdict.rotate_cw == 90
    assert source == "" and disputed


def test_silence_from_everyone_is_not_a_verdict():
    verdict, _, _ = combine({"ink_axis": Verdict(0, 0.0), "profile": Verdict(0, 0.0)})
    assert verdict.confidence == 0.0
    assert verdict.note


def test_verdict_without_any_found_lines_is_disputed():
    """Ось, назначенная только косвенно, ненадёжна — но лишь когда арбитра не было.

    Замер по паку-1: без арбитра все три ошибки прогона пришлись на полосы, где ни
    ink_axis, ни surya_lines не нашли строк, — там ось домысливал osd, и домыслил неверно.
    """
    verdict, _, disputed = combine(
        {
            "ink_axis": Verdict(0, 0.0, note="ось не различается"),
            "surya_lines": Verdict(0, 0.0, note="строк не нашлось"),
            "osd": Verdict(180, 0.39),
        }
    )
    assert disputed and verdict.rotate_cw == 180
    assert "строк не нашёл никто" in verdict.note


def test_allowed_angles_exclude_impossible_answers():
    """Ответа, которого нет в наборе, детектор дать не может в принципе."""
    verdict, _, _ = combine({"osd": Verdict(180, 0.9)}, allowed=(0, 90))
    assert verdict.rotate_cw != 180


def test_arbiter_verdict_outside_the_allowed_set_is_ignored():
    verdict, source, _ = combine({"ocr_vote": Verdict(270, 0.9), "ink_axis": Verdict(0, 0.9)}, allowed=(0, 90))
    assert source != "ocr_vote"


def test_found_lines_keep_the_verdict_undisputed():
    verdict, _, disputed = combine({"surya_lines": Verdict(90, 0.9, axis_only=True), "ocr_vote": Verdict(90, 0.8)})
    assert not disputed and verdict.rotate_cw == 90
