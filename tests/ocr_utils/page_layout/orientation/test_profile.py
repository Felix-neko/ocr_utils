"""profile: ось по периодичности проекционного профиля."""

from __future__ import annotations

import pytest

from ocr_utils.page_layout.orientation.detectors.base import rotate_cw
from ocr_utils.page_layout.orientation.detectors.profile import detect
from tests.ocr_utils.page_layout.orientation.synthetic import blank_page, text_page
from tests.ocr_utils.page_layout.orientation.test_ink_axis import frame_of


@pytest.mark.parametrize("applied,axis", [(0, 0), (90, 90), (180, 0), (270, 90)])
def test_axis_of_a_rotated_text_page_is_recovered(applied, axis):
    """Различает только ось, поэтому 0 и 180 для него — один и тот же ответ."""
    verdict = detect(frame_of(rotate_cw(text_page(), applied)))
    assert verdict.axis_only
    assert verdict.rotate_cw == axis


def test_blank_page_yields_no_opinion():
    assert detect(frame_of(blank_page())).confidence == 0.0
