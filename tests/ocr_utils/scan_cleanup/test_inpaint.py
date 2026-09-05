"""Группировка зон закраса по всем видам разметки сразу (``scan_cleanup.inpaint``).

Сеть заменена заглушкой: проверяется, ЧТО и СКОЛЬКИМИ операциями подаётся на закрас,
а не качество заливки.
"""

import numpy as np

from ocr_utils.scan_cleanup.inpaint import (
    DEFAULT_LAMA_ROI_MAX_SIDE,
    InpaintOptions,
    inpaint_page,
    page_masks,
    zone_kinds,
)
from ocr_utils.scan_cleanup.source import MaskRow, PageMarkup
from ocr_utils.scan_markup.db.models import MASK_HANDWRITING, MASK_LIBRARY_STAMP

from .conftest import mask_rle, text_page

W, H = 400, 600
FILL = 111


class StubModels:
    """Заглушка ``GpuModels``: красит ROI ровным цветом и запоминает маски."""

    def __init__(self):
        self.masks: "list[np.ndarray]" = []

    def lama_fill_roi(self, roi, roi_mask, max_side=512):
        self.masks.append(roi_mask.copy())
        return np.full_like(roi, FILL)


def markup_with(*boxes_kinds) -> PageMarkup:
    """Полоса с масками: ``(kind, (x1, y1, x2, y2))`` для каждой."""
    rows = tuple(
        MaskRow(kind, x1, y1, x2 - x1, y2 - y1, mask_rle((H, W), (x1, y1, x2, y2)))
        for kind, (x1, y1, x2, y2) in boxes_kinds
    )
    return PageMarkup("1970/01/0010.tif", W, H, 600, 8, (), rows)


def run(markup, **kw):
    models = StubModels()
    opts = InpaintOptions(group_min_dilate_px=kw.pop("group_min_dilate_px", 0), **kw)
    out, report = inpaint_page(text_page(), markup, opts, models)
    return out, report, models


def test_neighbouring_kinds_are_painted_in_one_operation():
    """Печать и надпись рядом идут одной операцией.

    Ради этого группировка и перестала делиться по видам: закрашивая печать
    отдельно, сеть видит соседнюю надпись в контекстном поле и затягивает её штрихи
    в дыру.
    """
    markup = markup_with(
        (MASK_LIBRARY_STAMP, (100, 100, 160, 140)),
        (MASK_HANDWRITING, (170, 100, 230, 140)),  # зазор 10 при ширине 60 — склеиваются
    )
    _out, report, models = run(markup)

    assert report.zones == 1
    assert len(models.masks) == 1
    assert report.rois[0][0] == f"{MASK_LIBRARY_STAMP}+{MASK_HANDWRITING}"


def test_distant_kinds_stay_separate():
    markup = markup_with((MASK_LIBRARY_STAMP, (10, 10, 70, 50)), (MASK_HANDWRITING, (300, 500, 360, 540)))
    _out, report, _models = run(markup)

    assert report.zones == 2
    assert {label for label, _roi in report.rois} == {MASK_LIBRARY_STAMP, MASK_HANDWRITING}


def test_painted_mask_is_the_original_areas_not_the_expanded_one():
    """Раздутая версия решает только, что с чем объединять; красится исходное.

    Между двумя объединёнными областями остаётся нетронутая полоса — если бы
    закрашивалась раздутая маска, она была бы залита.
    """
    markup = markup_with((MASK_LIBRARY_STAMP, (100, 100, 160, 140)), (MASK_HANDWRITING, (170, 100, 230, 140)))
    out, _report, _models = run(markup)

    assert (out[110, 130] == FILL).all()  # внутри первой области
    assert (out[110, 200] == FILL).all()  # внутри второй
    assert (out[110, 165] != FILL).any()  # зазор между ними не тронут


def test_one_kind_still_works():
    markup = markup_with((MASK_LIBRARY_STAMP, (100, 100, 160, 140)))
    _out, report, _models = run(markup)
    assert report.zones == 1
    assert report.counts() == {MASK_LIBRARY_STAMP: 1}


def test_page_without_masks_is_returned_as_is():
    markup = PageMarkup("1970/01/0010.tif", W, H, 600, 8, (), ())
    src = text_page()
    out, report = inpaint_page(src, markup, InpaintOptions(), StubModels())
    assert out is src
    assert report.zones == 0


def test_page_masks_skips_empty_kinds():
    markup = markup_with((MASK_LIBRARY_STAMP, (100, 100, 160, 140)))
    assert list(page_masks(markup, (MASK_LIBRARY_STAMP, MASK_HANDWRITING))) == [MASK_LIBRARY_STAMP]


def test_zone_kinds_sorted_by_area():
    zone = np.zeros((H, W), np.uint8)
    zone[100:140, 100:230] = 255
    masks = {MASK_LIBRARY_STAMP: np.zeros((H, W), np.uint8), MASK_HANDWRITING: np.zeros((H, W), np.uint8)}
    masks[MASK_LIBRARY_STAMP][100:140, 100:120] = 255  # 20 px шириной
    masks[MASK_HANDWRITING][100:140, 130:230] = 255  # 100 px шириной

    assert zone_kinds(zone, masks) == [MASK_HANDWRITING, MASK_LIBRARY_STAMP]


def test_default_roi_side_is_1024():
    """512 калибровался на пальцах; на разметке при нём оставались артефакты."""
    assert InpaintOptions().lama_roi_max_side == DEFAULT_LAMA_ROI_MAX_SIDE == 1024


def test_network_gets_the_mask_without_any_dilation():
    """В сеть уходит ровно обведённое: припуск группировки маску не раздувает.

    ``group_min_dilate_px`` участвует ТОЛЬКО в проверке пересечения раздутых рамок
    (``grouping.expand_box``); маски групп собираются из карты меток исходной,
    нераздутой маски.
    """
    from ocr_utils.scan_cleanup.source import decode_mask_rows

    markup = markup_with(
        (MASK_LIBRARY_STAMP, (100, 100, 160, 140)),
        (MASK_HANDWRITING, (200, 100, 260, 140)),  # зазор 40 px — сольются только припуском
    )
    drawn = decode_mask_rows(markup.masks, W, H)

    models = StubModels()
    out, report = inpaint_page(text_page(), markup, InpaintOptions(group_min_dilate_px=40), models)

    assert report.zones == 1, "припуск 40 px должен слить их в одну операцию"
    for kind, mask in report.masks.items():
        assert np.array_equal(mask > 0, decode_mask_rows(markup.masks_of(kind), W, H))
    # Сеть красит ROI целиком, но вклеивается только под маской: вне неё кадр не тронут.
    assert not ((out[..., 0] == FILL) & ~drawn).any()
    # И зазор между слитыми областями остался нетронутым.
    assert (out[110, 180] != FILL).any()
