"""Способы зоны обрезки на синтетическом контенте: метка у корешка — сор, номер под колонкой — нет; гибрид не режет контент за отступом; обложка не обрезается."""

import numpy as np

from ocr_utils.scan_markup.content_box.content import Item, PageContent, Side
from ocr_utils.scan_markup.content_box.methods import Method, TrimParams, box_of, measures, strays

DPI = 254.0  # 10 px/мм


def content(items, cut_side=Side.LEFT) -> PageContent:
    """Страница 1500×2000 px (150×200 мм) без краски."""
    return PageContent(1500, 2000, DPI, cut_side, tuple(items), np.zeros((200, 150), bool), 10.0)


BODY = Item((300, 300, 1300, 1800), "block")
NUMBER_UNDER = Item((780, 1850, 820, 1880), "line", short=True)  # номер страницы под колонкой — в охвате тела
MARK_AT_CUT = Item((20, 1700, 45, 1760), "line", short=True)  # «49» у корешка — вне тела


def test_strays_only_outside_body():
    page = content([BODY, NUMBER_UNDER, MARK_AT_CUT])
    assert strays(page, TrimParams()) == [MARK_AT_CUT]


def test_hybrid_keeps_content_beyond_fixed_margin():
    # Блок заходит за отступ сверху (2 мм от верха при отступе 4 мм).
    near_top = Item((300, 20, 1300, 400), "block")
    page = content([near_top, MARK_AT_CUT])
    box = box_of(Method.HYBRID, page, TrimParams(), cover=False)
    assert box[1] <= 20 and box[0] == 50  # слева — отступ у разреза 5 мм, метка за ним
    m = measures(box, page, TrimParams())
    assert m.lost_items == 0 and m.strays_in == 0


def test_cover_is_whole_page():
    page = content([BODY])
    for method in Method:
        assert box_of(method, page, TrimParams(), cover=True) == (0, 0, 1500, 2000)
