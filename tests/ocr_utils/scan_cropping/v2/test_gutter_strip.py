"""Тесты полосы-вклейки и подгоночных областей разворота на синтетике: щель SAM 3 (без островков страниц на полосе), края полосы, линия корешка, области страниц с отрезком через полосу, мостик склейки только у щели."""

import numpy as np

from ocr_utils.scan_cropping.v2.glue import glue_hull_bridge
from ocr_utils.scan_cropping.v2.gutter_strip import Strip, find_strip, spine_line
from ocr_utils.scan_cropping.v2.page_frames import StripJoin, spread_frames
from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM, PagePose, PageQuad
from ocr_utils.scan_cropping.v2.spread_lines import Line

H, W = 900, 1400
# Страницы: левая x 100–640, правая x 760–1300 (щель 120 px ≈ 17 мм — полоса), y 100–800.
LEFT_X, RIGHT_X, TOP, BOT = (100, 640), (760, 1300), 100, 800


def spread_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Кадр (белая полоса между сероватыми страницами со складками по кромкам), экземпляры SAM 3, книга, страницы."""
    img = np.full((H, W, 3), 20, np.uint8)
    img[TOP:BOT, LEFT_X[0] : RIGHT_X[1]] = 232
    img[TOP:BOT, LEFT_X[1] : RIGHT_X[0]] = 250  # полоса белее страниц
    for x in (LEFT_X[1], RIGHT_X[0]):
        img[TOP:BOT, x - 1 : x + 2] = 190  # складки по кромкам
    inst = np.zeros((H, W), np.uint8)
    inst[TOP:BOT, LEFT_X[0] : LEFT_X[1]] = 1
    inst[TOP:BOT, RIGHT_X[0] : RIGHT_X[1]] = 2
    book = np.zeros((H, W), bool)
    book[TOP - 20 : BOT + 20, LEFT_X[0] - 20 : RIGHT_X[1] + 20] = True
    return img, inst, book, inst > 0


def page(x0: int, x1: int, top: int = TOP) -> PageQuad:
    """Плоская страница-прямоугольник."""
    mask = np.zeros((H, W), bool)
    mask[top:BOT, x0:x1] = True
    quad = np.array([[x0, top], [x1, top], [x1, BOT], [x0, BOT]], np.float64)
    return PageQuad(mask, quad, None, PagePose.FLAT)


def test_find_strip_between_sam3_instances():
    img, inst, book, _ = spread_masks()
    divider = Line(700.0, 450.0, 0.0, TOP, BOT)
    strip, sam3 = find_strip(img, inst, book, divider)
    assert abs(sam3.extra["gap_mm"] - 120 / SEG_PX_PER_MM) < 1.0
    assert strip is not None
    assert abs(strip.left.x_at(450.0) - LEFT_X[1]) < 3 and abs(strip.right.x_at(450.0) - RIGHT_X[0]) < 3
    assert 15 < strip.width_mm < 19


def test_no_strip_for_narrow_gap():
    img, inst, book, _ = spread_masks()
    inst[:, 640:720] = 1  # щель 40 px ≈ 6 мм — стык тетрадей, не полоса
    strip, _ = find_strip(img, inst, book, Line(740.0, 450.0, 0.0, TOP, BOT))
    assert strip is None


def test_islands_on_strip_do_not_narrow_gap():
    """Клочки бумаги правой страницы на полосе (островки экземпляра в половине строк) щель не сужают."""
    img, inst, book, _ = spread_masks()
    for y in range(TOP, BOT, 40):
        inst[y : y + 20, 650:670] = 2  # островок в 10 px от левой страницы
    _, sam3 = find_strip(img, inst, book, Line(700.0, 450.0, 0.0, TOP, BOT))
    assert abs(sam3.extra["gap_mm"] - 120 / SEG_PX_PER_MM) < 1.0


def strip_with(k_left: float, k_right: float) -> Strip:
    """Полоса 640–760 px в строке 450 с краями заданных наклонов."""
    empty = np.zeros((0, 2))
    left = Line(640.0, 450.0, k_left, TOP, BOT)
    right = Line(760.0, 450.0, k_right, TOP, BOT)
    return Strip(left, right, np.zeros((H, W), bool), 17.0, empty, empty)


def test_spine_line_rules():
    divider = Line(700.0, 450.0, 0.05, TOP, BOT)
    assert spine_line(None, divider, "gap_mid")[0] is divider
    spine, source = spine_line(strip_with(0.010, 0.016), divider, "gap_mid")
    assert abs(spine.k - np.tan((np.arctan(0.010) + np.arctan(0.016)) / 2)) < 1e-9
    assert abs(spine.x_at(450.0) - 700.0) < 1e-9 and source.startswith("средний наклон краёв полосы")
    spine, source = spine_line(strip_with(0.013, -0.022), divider, "gap_mid")  # края расходятся на ~2°
    assert spine.k == divider.k and "расходятся" in source


def test_frames_without_strip_end_on_spine():
    pages = [page(*LEFT_X), page(*RIGHT_X)]
    spine = Line(700.0, 450.0, 0.0, TOP, BOT)
    empty = np.zeros((H, W), bool)
    frames = spread_frames(pages, spine, None, empty, empty, [np.zeros((0, 2))] * 2)
    for f in frames:
        assert abs(f.spine[0, 0] - 700) < 2 and abs(f.spine[1, 0] - 700) < 2
        assert abs(f.top[-1, 1] - TOP) < 3 and abs(f.bottom[-1, 1] - BOT) < 3


def test_frames_with_strip_interpolate_across():
    """Верх левой страницы у полосы на 20 px ниже, чем у правой: угол на корешке — посередине отрезка, всплеск маски у
    корешка (между линиями полосы) в подгонку не идёт."""
    left, right = page(*LEFT_X, top=TOP + 20), page(*RIGHT_X)
    left.mask[TOP - 30 : TOP + 20, 625:640] = True
    strip = Strip(Line(620.0, 450.0, 0.0, TOP, BOT), Line(780.0, 450.0, 0.0, TOP, BOT), np.zeros((H, W), bool), 23.0,
                  np.zeros((0, 2)), np.zeros((0, 2)))  # fmt: skip
    spine = Line(700.0, 450.0, 0.0, TOP, BOT)
    empty = np.zeros((H, W), bool)
    frames = spread_frames([left, right], spine, strip, empty, empty, [np.zeros((0, 2))] * 2, StripJoin.SEGMENT)
    corner_l, corner_r = frames[0].top[-1], frames[1].top[-1]
    assert np.allclose(corner_l, corner_r)
    assert abs(corner_l[0] - 700) < 1
    assert abs(corner_l[1] - (TOP + 10)) < 3  # середина между 120 и 100
    assert abs(frames[0].top[-2][0] - 620) < 2  # конец верха левой страницы — на вспомогательной линии


def test_glue_bridge_stays_in_rows_of_both_pages():
    """Низ левой страницы у корешка выше, дальше — ниже (кадр косой): мостик не уходит ниже конца щели."""
    paper = np.zeros((H, W), bool)
    book = np.zeros((H, W), bool)
    book[50:880, 50:1350] = True
    paper[TOP:780, 100:660] = True  # левая страница до y 780 у корешка
    paper[TOP:850, 100:300] = True  # дальше от корешка — ниже
    paper[TOP:780, 700:1300] = True  # правая до y 780
    out = glue_hull_bridge(paper, book).paper
    assert out[450, 660:700].all()  # щель склеена
    assert not out[800:850, 660:700].any()  # под концом щели — не бумага


def test_strip_paper_fills_wide_strip_between_pages():
    """Полоса 17 мм шире предела склейки: в бумагу идёт вся, но только в строках обеих страниц."""
    from ocr_utils.scan_cropping.v2.gutter_strip import strip_paper

    img, inst, book, _ = spread_masks()
    pages = [page(*LEFT_X), page(*RIGHT_X)]
    paper = pages[0].mask | pages[1].mask
    added = strip_paper(img, inst, book, pages, paper)
    assert added[450, LEFT_X[1] + 5 : RIGHT_X[0] - 5].all()
    assert not added[TOP - 10].any() and not added[BOT + 5].any()
    assert not (added & paper).any()


def test_frames_with_strip_tangent_each_page():
    """Касательная: верх каждой страницы — до линии корешка на своей высоте (левая 120, правая 100), без излома."""
    left, right = page(*LEFT_X, top=TOP + 20), page(*RIGHT_X)
    strip = Strip(Line(620.0, 450.0, 0.0, TOP, BOT), Line(780.0, 450.0, 0.0, TOP, BOT), np.zeros((H, W), bool), 23.0,
                  np.zeros((0, 2)), np.zeros((0, 2)))  # fmt: skip
    spine = Line(700.0, 450.0, 0.0, TOP, BOT)
    empty = np.zeros((H, W), bool)
    frames = spread_frames([left, right], spine, strip, empty, empty, [np.zeros((0, 2))] * 2)
    assert abs(frames[0].top[-1][0] - 700) < 1 and abs(frames[0].top[-1][1] - (TOP + 20)) < 2
    assert abs(frames[1].top[-1][0] - 700) < 1 and abs(frames[1].top[-1][1] - TOP) < 2
