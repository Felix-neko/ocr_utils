"""Защитная маска из размеченных областей (``scan_cleanup.protect``)."""

import numpy as np
import pytest

from ocr_utils.scan_cleanup.protect import ProtectOptions, analysis_roi, build_protect, is_full_page, rects_mask
from ocr_utils.scan_cleanup.source import PageMarkup, Rect
from ocr_utils.scan_markup.db.models import KIND_COLOR, KIND_COLOR_TEXT, KIND_GRAYSCALE, KIND_STAMP_SUSPECT

SHAPE = (600, 400)


def markup(*regions: Rect) -> PageMarkup:
    return PageMarkup("1970/01/0010.tif", 400, 600, 600, 8, regions, ())


def rect(kind: str, box=(50, 50, 150, 150), full_page: bool = False) -> Rect:
    return Rect(*box, kind, full_page)


def test_pictures_are_protected():
    mask, rects = build_protect(SHAPE, markup(rect(KIND_COLOR), rect(KIND_GRAYSCALE, (200, 200, 300, 300))))
    assert len(rects) == 2
    assert mask[100, 100] == 255
    assert mask[250, 250] == 255
    assert mask[0, 0] == 0


def test_color_text_is_not_protected():
    """Цветной набор — это буквы: их ловит бинаризация, а прямоугольник вокруг
    оставил бы неразмытым весь фон между ними."""
    mask, rects = build_protect(SHAPE, markup(rect(KIND_COLOR_TEXT)))
    assert rects == ()
    assert not mask.any()


def test_stamp_suspect_follows_the_flag():
    page = markup(rect(KIND_STAMP_SUSPECT))
    assert not build_protect(SHAPE, page)[0].any()
    assert build_protect(SHAPE, page, ProtectOptions(protect_stamp_suspect=True))[0][100, 100] == 255


def test_full_page_region_is_recognised():
    assert is_full_page(markup(rect(KIND_COLOR, (0, 0, 400, 600), full_page=True))) is not None
    assert is_full_page(markup(rect(KIND_COLOR))) is None


def test_full_page_by_area_even_without_the_flag():
    """Флаг ставит детекция, а полосу могли обвести руками уже после."""
    assert is_full_page(markup(rect(KIND_COLOR, (0, 0, 400, 580)))) is not None
    assert is_full_page(markup(rect(KIND_COLOR, (0, 0, 400, 300)))) is None


def test_color_text_never_makes_a_page_full():
    """Даже цветной заголовок во всю полосу — это не полосная иллюстрация."""
    assert is_full_page(markup(rect(KIND_COLOR_TEXT, (0, 0, 400, 600), full_page=True))) is None


def test_analysis_roi_excludes_all_regions():
    """Из области анализа вычитаются ВСЕ растровые области, а не только защищаемые.

    Средние тона фотографии тянут порог Оцу вверх независимо от того, защищаем мы
    её потом или нет.
    """
    roi = analysis_roi(SHAPE, markup(rect(KIND_STAMP_SUSPECT), rect(KIND_GRAYSCALE, (200, 200, 300, 300))))
    assert roi[100, 100] == 0
    assert roi[250, 250] == 0
    assert roi[0, 0] == 255


def test_analysis_roi_is_none_without_regions():
    assert analysis_roi(SHAPE, markup()) is None


def test_rects_mask_clips_to_the_frame():
    mask = rects_mask(SHAPE, [Rect(-50, -50, 100, 100, KIND_COLOR, False)])
    assert mask.shape == SHAPE
    assert mask[0, 0] == 255
    assert mask[150, 150] == 0
