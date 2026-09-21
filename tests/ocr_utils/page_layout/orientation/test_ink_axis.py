"""ink_axis: ось текста и сторона поворота."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from ocr_utils.page_layout.orientation.detectors.base import Frame, rotate_cw
from ocr_utils.page_layout.orientation.detectors.ink_axis import detect, glyph_mask, ink_asymmetry
from tests.ocr_utils.page_layout.orientation.synthetic import blank_page, line_art_page, text_page


def half(image: np.ndarray) -> np.ndarray:
    """Копия 150 dpi — то, с чем работают детекторы (см. :func:`frame_of`)."""
    return cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


def frame_of(image: np.ndarray) -> Frame:
    """Кадр из полосы, нарисованной в масштабе 300 dpi.

    Уменьшение вдвое повторяет то, что делает ``read_frame``: детекторы калиброваны по
    копии 150 dpi, и подавать им картинку вдвое крупнее значило бы вывести размер буквы
    за границы отбора — тест мерил бы не то.
    """
    return Frame("x/y/z.tif", Path("z.tif"), image.shape[1], image.shape[0], 300, half(image), image)


def test_upright_text_page_needs_no_rotation():
    assert detect(frame_of(text_page())).rotate_cw == 0


@pytest.mark.parametrize("applied", [0, 90, 180, 270])
def test_rotation_of_a_text_page_is_recovered(applied):
    """Полосу крутим на известный угол — детектор обязан назвать обратный."""
    verdict = detect(frame_of(rotate_cw(text_page(), applied)))
    assert verdict.rotate_cw == (-applied) % 360


def test_blank_page_yields_no_opinion():
    verdict = detect(frame_of(blank_page()))
    assert verdict.confidence == 0.0
    assert verdict.note


def test_line_art_without_text_yields_no_opinion():
    """Чертёж без подписи не должен превращаться в «строки»: там нечего читать."""
    verdict = detect(frame_of(line_art_page()))
    assert verdict.confidence == 0.0


def test_descenders_pull_ink_centre_up():
    """Знак асимметрии — та самая калибровка, на которой держится различение 0 и 180."""
    upright, _ = ink_asymmetry(glyph_mask(half(text_page())))
    flipped, _ = ink_asymmetry(glyph_mask(half(rotate_cw(text_page(), 180))))
    assert upright < 0 < flipped


def test_text_without_descenders_leaves_the_side_undecided():
    verdict = detect(frame_of(text_page(descenders=False)))
    assert verdict.axis_only
    assert verdict.rotate_cw == 0
