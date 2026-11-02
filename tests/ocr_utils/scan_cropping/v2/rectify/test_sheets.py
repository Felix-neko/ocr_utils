"""Тесты панели итога листа 08_rectify (``sheets.pages_panel``): страницы разворота по отдельности в своих прямоугольниках через черту, зона докраса — цветом."""

from types import SimpleNamespace

import numpy as np

from ocr_utils.scan_cropping.v2.fill import FillMethod
from ocr_utils.scan_cropping.v2.rectify import sheets
from ocr_utils.scan_cropping.v2.rectify.coons import Layout, PageRect
from ocr_utils.scan_cropping.v2.rectify.pages import PagePart, cut_pages


def result(single: bool = False) -> SimpleNamespace:
    """Итог 200×100 без разбора текста: поле по X от −6 до +6 px сверху вниз (кривая линия разреза)."""
    rects = (PageRect(10, 10, 190, 90),) if single else (PageRect(10, 10, 100, 90), PageRect(100, 10, 190, 90))
    lay = Layout(200, 100, rects, 100.0)
    dx = np.repeat(np.linspace(-6, 6, 100, dtype=np.float32)[:, None], 200, axis=1)
    image = np.full((100, 200, 3), 230, np.uint8)
    pages = [SimpleNamespace(after=None) for _ in rects]
    return SimpleNamespace(layout=lay, canvas=lay, dense=(dx, np.zeros_like(dx)), final=image, pages=pages)


def test_spread_pages_side_by_side_with_divider(monkeypatch):
    monkeypatch.setattr(sheets, "SHEET_HEIGHT", 100)
    res = result()
    written = cut_pages(res, [PagePart.LEFT, PagePart.RIGHT], 300.0, FillMethod.PULL_PUSH, 0.0)
    panel = sheets.pages_panel(res, written)
    widths = [p.box[2] - p.box[0] for p in written]
    assert panel.shape == (100, sum(widths) + sheets.PAGE_DIVIDER_PX, 3)
    # Черта — сразу за левой страницей.
    assert (panel[:, widths[0] : widths[0] + sheets.PAGE_DIVIDER_PX] == sheets.COLOUR_PAGE_DIVIDER).all()
    # Зона докраса (низ левой страницы у кривого разреза: там разрез левее корешка) подкрашена, середина — нет.
    x0, y0, x1, y1 = written[0].box
    corner = panel[y1 - 2, widths[0] - 1].astype(int)
    assert abs(corner[0] - corner[2]) > 20
    middle = panel[50, widths[0] // 2].astype(int)
    assert abs(middle[0] - middle[2]) < 3


def test_single_page_has_no_divider(monkeypatch):
    monkeypatch.setattr(sheets, "SHEET_HEIGHT", 100)
    res = result(single=True)
    panel = sheets.pages_panel(res, None)
    assert panel.shape[0] == 100
    assert not (panel == 0).all(axis=(0, 2)).any()
