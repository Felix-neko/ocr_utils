"""Тесты «в кадре одна страница» на синтетике: правило вида кадра, склейка ложно разрезанного листа и отброс полоски в ``page_quads``, части выхода страниц, флаги ``single-page``/``spread``, Кунс, поле и нарезка одной области."""

from types import SimpleNamespace

import numpy as np
import pytest

from ocr_utils.frame_flags.flags import FlagsError, FrameFlag, FrameFlags, normalize_flags
from ocr_utils.scan_cropping.v2.fill import FillMethod
from ocr_utils.scan_cropping.v2.page_frames import PageFrame
from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM, PagePose, page_quads
from ocr_utils.scan_cropping.v2.rectify.coons import page_patch, plan_layout, rectify_maps
from ocr_utils.scan_cropping.v2.rectify.pages import PagePart, cut_line, cut_pages, page_areas, wanted_parts
from ocr_utils.scan_cropping.v2.single_page import FrameKind, classify_frame

MM = SEG_PX_PER_MM


def rect_quad(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Трапеция-прямоугольник по часовой стрелке от левого верхнего угла."""
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)


def paper_of(w_mm: float, h_mm: float) -> np.ndarray:
    """Маска бумаги — прямоугольник ``w_mm × h_mm`` с полями 10 мм."""
    pad = int(10 * MM)
    mask = np.zeros((int(h_mm * MM) + 2 * pad, int(w_mm * MM) + 2 * pad), bool)
    mask[pad:-pad, pad:-pad] = True
    return mask


def test_classify_frame():
    sheet = paper_of(180, 255)
    # Одна трапеция — одна страница; бумага низкая (ярлык) — страницы нет.
    assert classify_frame([rect_quad(0, 0, 180 * MM, 255 * MM)], sheet, 1, MM) is FrameKind.SINGLE
    assert classify_frame([rect_quad(0, 0, 80 * MM, 60 * MM)], paper_of(80, 60), 1, MM) is FrameKind.NO_PAGE
    assert classify_frame([], sheet, 0, MM) is FrameKind.NO_PAGE
    # Лист 180 мм, разрезанный на 150 + 30 мм: при одном экземпляре SAM 3 — ложный корешок, при двух — полоска.
    halves = [rect_quad(0, 0, 150 * MM, 255 * MM), rect_quad(150 * MM, 0, 180 * MM, 255 * MM)]
    assert classify_frame(halves, sheet, 1, MM) is FrameKind.SINGLE
    assert classify_frame(halves, sheet, 2, MM) is FrameKind.SLIVER
    # Косой разрез по печати при одном экземпляре: куски почти равны, но бумага узкая — лист одной страницы.
    oblique = [rect_quad(0, 0, 100 * MM, 255 * MM), rect_quad(100 * MM, 0, 180 * MM, 255 * MM)]
    assert classify_frame(oblique, sheet, 1, MM) is FrameKind.SINGLE
    assert classify_frame(oblique, sheet, 2, MM) is FrameKind.SPREAD
    # Разворот 2 × 160 мм — разворот; узкая поднятая страница при широкой бумаге — тоже разворот.
    spread = [rect_quad(0, 0, 160 * MM, 255 * MM), rect_quad(160 * MM, 0, 320 * MM, 255 * MM)]
    assert classify_frame(spread, paper_of(320, 255), 2, MM) is FrameKind.SPREAD
    raised = [rect_quad(0, 0, 160 * MM, 255 * MM), rect_quad(160 * MM, 0, 230 * MM, 255 * MM)]
    assert classify_frame(raised, paper_of(230, 255), 2, MM) is FrameKind.SPREAD


def test_page_quads_merges_and_drops():
    paper = paper_of(180, 255)
    h, w = paper.shape
    gray = np.full((h, w), 200, np.uint8)
    paint = np.zeros((h, w), bool)
    book = paper.copy()
    hulls = np.zeros((h, w), bool)
    cut = int((10 + 150) * MM)
    # Один экземпляр SAM 3 на весь лист, но печать — два пятна: запасной разрез по провалу печати.
    inst = paper.astype(np.uint8)
    hulls[int(30 * MM) : int(240 * MM), int(20 * MM) : cut - int(5 * MM)] = True
    hulls[int(30 * MM) : int(240 * MM), cut + int(5 * MM) : int(185 * MM)] = True
    pages = page_quads(paper, book, hulls, inst, gray, paint)
    assert len(pages) == 1 and pages[0].frame_kind is FrameKind.SINGLE and pages[0].pose is PagePose.FLAT
    assert pages[0].mask.sum() == paper.sum()
    # Два экземпляра — настоящий корешок: страница — широкая (левая), полоска справа отброшена.
    inst2 = inst.copy()
    inst2[:, cut:] *= 2
    pages = page_quads(paper, book, hulls, inst2, gray, paint)
    assert len(pages) == 1 and pages[0].frame_kind is FrameKind.SLIVER and pages[0].spread_index == 0
    assert not pages[0].mask[:, cut + 5 :].any()
    # Флаг «разворот» выключает правило: остаются две страницы.
    pages = page_quads(paper, book, hulls, inst2, gray, paint, FrameKind.SPREAD)
    assert len(pages) == 2 and all(p.frame_kind is FrameKind.SPREAD for p in pages)


def test_wanted_parts_by_kind():
    none = FrameFlags()
    assert wanted_parts(none, FrameKind.SINGLE) == [PagePart.SINGLE]
    assert wanted_parts(none, FrameKind.NO_PAGE) == []
    assert wanted_parts(FrameFlags(frozenset({FrameFlag.IGNORE_BOTH})), FrameKind.SINGLE) == []
    assert wanted_parts(none, FrameKind.SLIVER, 1) == [PagePart.RIGHT]
    assert wanted_parts(FrameFlags(frozenset({FrameFlag.IGNORE_RIGHT})), FrameKind.SLIVER, 1) == []


def test_single_page_flags():
    assert normalize_flags(["single-page", "cover-both"]) == {FrameFlag.SINGLE_PAGE, FrameFlag.COVER_BOTH}
    assert FrameFlags(normalize_flags(["spread"])).spread
    for bad in (["single-page", "spread"], ["single-page", "ignore-left"], ["single-page", "no-splitting"],
                ["single-page", "cover-right"]):  # fmt: skip
        with pytest.raises(FlagsError):
            normalize_flags(bad)


def single_frame_rect(x0: float, y0: float, x1: float, y1: float) -> PageFrame:
    """Область одной страницы-прямоугольника (как ``page_frames.single_frame``: ``far`` слева, ``spine`` справа)."""
    n = 50
    top = np.stack([np.linspace(x0, x1, n), np.full(n, y0)], axis=1)
    bottom = np.stack([np.linspace(x0, x1, n), np.full(n, y1)], axis=1)
    far = np.stack([np.full(n, x0), np.linspace(y0, y1, n)], axis=1)
    spine = np.stack([np.full(n, x1), np.linspace(y0, y1, n)], axis=1)
    return PageFrame(top, bottom, far, spine, np.zeros((0, 2)))


def test_single_layout_identity_and_cut():
    patch = page_patch(single_frame_rect(40, 30, 240, 330), True, lambda p: np.asarray(p, np.float64), 20.0)
    layout = plan_layout((patch,), (360, 300))
    assert layout.single and (layout.width, layout.height) == (300, 360)
    # Прямоугольник — тождество, «корешок» — правый край холста.
    map_x, map_y = rectify_maps((patch,), layout)
    rows, cols = np.mgrid[0:360, 0:300]
    np.testing.assert_allclose(map_x, cols, atol=1e-3)
    np.testing.assert_allclose(map_y, rows, atol=1e-3)
    zero = (np.zeros((360, 300), np.float32), np.zeros((360, 300), np.float32))
    (area,) = page_areas(layout, zero)
    assert area[30:330, 40:240].all() and not area[:, 240:].any()
    assert len(cut_line(layout, zero)) == 0
    result = SimpleNamespace(layout=layout, dense=zero, final=np.full((360, 300, 3), 220, np.uint8))
    pages = cut_pages(result, [PagePart.SINGLE], 300.0, FillMethod.PULL_PUSH, 0.0)
    assert [p.part for p in pages] == [PagePart.SINGLE] and pages[0].image.shape[:2] == (300, 200)


def test_single_text_frame(monkeypatch):
    from ocr_utils.scan_cropping.v2.rectify import text_run
    from ocr_utils.scan_cropping.v2.rectify.geometry import SpreadGeometry

    image = np.full((360, 300, 3), 220, np.uint8)
    geometry = SpreadGeometry([single_frame_rect(40, 30, 240, 330)], (0, 0, 300, 360), (1.0, 1.0))
    monkeypatch.setattr(text_run, "analyze_page", lambda *a, **k: None)
    monkeypatch.setattr(text_run, "page_regions", lambda *a, **k: ())
    result = text_run.text_frame(image, geometry, 300.0, text_run.TextParams(passes=1), (text_run.PageControl(),))
    assert result.layout.single and len(result.pages) == 1
    # Без опор поле нулевое: итог — та же страница.
    assert np.abs(result.dense[0]).max() < 1e-3 and result.final.shape[:2] == (360, 300)
