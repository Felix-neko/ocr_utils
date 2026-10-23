"""Тесты рваного края на синтетике: мера извилистости ровной и зубчатой стороны, зоны у кромок книги и слияние двух
проходов SAM 3 с откатом при промахах, край по маске ленты, выбор способа, строки пальца у края, заполнение пропусков
края и замена бумаги в ленте."""

import numpy as np

from ocr_utils.scan_cropping.v2.edge_refine import (
    Band,
    Engine,
    Method,
    MethodResult,
    apply_edge,
    choose,
    edge_of,
    fill_edge,
    finger_rows_of,
)
from ocr_utils.scan_cropping.v2.edge_zones import ZoneStatus, merge_levels, outer_zones
from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM
from ocr_utils.scan_cropping.v2.ragged_edge import is_ragged, measure_far_side, page_raw_mask
from ocr_utils.scan_cropping.v2.side_fit import Side

H, W = 1800, 2400
TOP, BOT = 200, 1600
LEFT_X, RIGHT_X = (300, 1180), (1220, 2100)  # левая и правая страницы


def page_mask(x0: int, x1: int, teeth_mm: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Страница-прямоугольник; ``teeth_mm`` — глубина зубцов на левой стороне (рваный край)."""
    mask = np.zeros((H, W), bool)
    mask[TOP:BOT, x0:x1] = True
    if teeth_mm:
        depth = int(teeth_mm * SEG_PX_PER_MM)
        for y in range(TOP, BOT, 40):
            mask[y : y + 20, x0 : x0 + depth] = False
    quad = np.array([[x0, TOP], [x1, TOP], [x1, BOT], [x0, BOT]], np.float64)
    return mask, quad


def test_measure_smooth_and_ragged():
    smooth, quad = page_mask(*LEFT_X)
    ragged, _ = page_mask(*LEFT_X, teeth_mm=3.0)
    m_smooth = measure_far_side(smooth, quad, True, None, SEG_PX_PER_MM)
    m_ragged = measure_far_side(ragged, quad, True, None, SEG_PX_PER_MM)
    assert m_smooth.tortuosity < 1.01 and not is_ragged(m_smooth)
    assert m_ragged.tortuosity > 1.2 and is_ragged(m_ragged)


def test_page_raw_mask_keeps_tongues_and_drops_neighbour():
    page, _ = page_mask(*LEFT_X)
    other, _ = page_mask(*RIGHT_X)
    raw = page | other
    raw[800:820, LEFT_X[0] - 60 : LEFT_X[0]] = True  # язык наружу на ~9 мм
    out = page_raw_mask(raw, page, other, SEG_PX_PER_MM)
    assert out[810, LEFT_X[0] - 50] and not out[810, RIGHT_X[0] + 10]


def book_and_pages():
    """Книга (кромки x 250 и 2150), две страницы базы и растянутый проход."""
    book = np.zeros((H, W), bool)
    book[TOP - 30 : BOT + 30, 250:2150] = True
    left, _ = page_mask(*LEFT_X, teeth_mm=3.0)
    right, _ = page_mask(*RIGHT_X)
    inst = np.zeros((H, W), np.uint8)
    inst[left] = 1
    inst[right] = 2
    left_lv, _ = page_mask(*LEFT_X)  # растянутый проход: левая сторона ровная
    inst_lv = np.zeros((H, W), np.uint8)
    inst_lv[left_lv] = 2  # метки растянутого прохода свои
    inst_lv[right] = 1
    return book, inst, inst_lv


def test_outer_zones_follow_book_edges():
    book, _, _ = book_and_pages()
    zones = outer_zones(book, np.zeros_like(book))
    assert zones[Side.LEFT][900, 260] and zones[Side.LEFT][900, 250 + int(25 * SEG_PX_PER_MM)]
    assert not zones[Side.LEFT][900, 1200] and zones[Side.RIGHT][900, 2140]


def test_merge_takes_stretched_in_zone_with_base_labels():
    book, inst, inst_lv = book_and_pages()
    zones = outer_zones(book, np.zeros_like(book))
    m = merge_levels(inst > 0, inst, inst_lv > 0, inst_lv, zones)
    assert m.status[Side.LEFT] is ZoneStatus.USED
    # Зубец базы закрыт растянутым проходом, метка — левой страницы базы (1).
    assert m.raw[TOP + 5, LEFT_X[0] + 3] and m.inst[TOP + 5, LEFT_X[0] + 3] == 1
    # Вне зон — база.
    assert (m.inst[:, 1150:1250] == inst[:, 1150:1250]).all()


def test_merge_falls_back_on_misses():
    book, inst, inst_lv = book_and_pages()
    zones = outer_zones(book, np.zeros_like(book))
    empty = merge_levels(inst > 0, inst, np.zeros_like(book), np.zeros_like(inst), zones)
    assert empty.frame_status and (empty.raw == (inst > 0)).all()
    lost = inst_lv.copy()
    lost[:, : LEFT_X[0] + 120] = 0  # растянутый проход потерял полосу у левой кромки
    m = merge_levels(inst > 0, inst, lost > 0, lost, zones)
    assert m.status[Side.LEFT] is ZoneStatus.LOSS
    single = merge_levels(inst == 1, np.where(inst == 1, 1, 0).astype(np.uint8), inst_lv > 0, inst_lv, zones)
    assert single.frame_status  # у базы одна страница — кадр на базе


def test_edge_of_first_exit_outward():
    mask = np.zeros((4, 10), bool)
    mask[0, 3:] = True
    mask[1, :] = True  # маска до наружного края ленты
    mask[2, 6:] = True
    out = edge_of(mask, 7)
    assert out[0] == 3 and out[1] == 0 and out[2] == 6 and np.isnan(out[3])


def test_choose_prefers_agreeing_smooth_and_inner():
    edge = np.full(100, 50.0)
    good = MethodResult(Method(Engine.SAM3, 4), edge, rough_mm=0.1, accepted=True)
    inner = MethodResult(Method(Engine.SAM2, 4), edge + 3.0, rough_mm=0.12, accepted=True)
    far = MethodResult(Method(Engine.SAM3, 1), edge - 200.0, rough_mm=0.05, accepted=True)
    rejected = MethodResult(Method(Engine.SAM2, 1), edge, rough_mm=0.01, accepted=False)
    best = choose([good, inner, far, rejected], ppm=17.0)
    # Дальний от медианы не годен для выбора; из согласных — наименьшая рваность, при равенстве — ближний внутрь.
    assert best is inner


def test_apply_edge_replaces_band_only():
    paper, quad = page_mask(*LEFT_X, teeth_mm=3.0)
    band = Band(quad, True, SEG_PX_PER_MM, ratio=2.0)
    edge = np.full(band.length, float(band.col(0.0)))  # край — по стороне трапеции
    out = apply_edge(paper, band, edge)
    mid = (TOP + BOT) // 2
    assert out[mid, LEFT_X[0] + 3]  # зубец залит
    assert not out[mid, LEFT_X[0] - 5]  # снаружи стороны — не бумага
    assert (out[:, 1000:] == paper[:, 1000:]).all()  # вне ленты без изменений


def test_finger_rows_only_where_core_touches_edge():
    paper, quad = page_mask(*RIGHT_X)
    band = Band(quad, False, SEG_PX_PER_MM, ratio=2.0)
    page_band = band.warp(paper.astype(np.uint8), seg=True, nearest=True) > 0
    core = np.zeros((band.length, band.width), bool)
    side = band.col(0.0)
    core[100:400, : side - int(8 * band.ppm)] = True  # ладонь снаружи книги, 8 мм и дальше за краем
    core[1000:1200, side - 20 : side + 20] = True  # палец на кромке
    core[1500:1700, band.col(20.0) :] = True  # палец глубоко на странице
    rows = finger_rows_of(core, page_band, band)
    assert not rows[100:400].any() and rows[1000:1200].all() and not rows[1500:1700].any()


def test_fill_edge_bridges_gap_with_line():
    """Пропуск под пальцем — прямой по выбранному краю, не по грубой маске (она огибает палец выемкой)."""
    n, ppm = 2000, 17.0
    page_band = np.zeros((n, 700), bool)
    page_band[:, 300:] = True
    page_band[800:1200, 300:360] = False  # выемка грубой маски у пальца
    edge = np.linspace(300.0, 320.0, n)
    edge[800:1200] = np.nan
    out = fill_edge(MethodResult(Method(Engine.SAM3, 4), edge, accepted=True), page_band, 400, ppm)
    assert np.isfinite(out).all() and np.abs(out - np.linspace(300.0, 320.0, n)).max() < 0.5


def test_union_levels_edges_fills_top_notch_only_near_edges():
    """Растянутый проход заделывает выемку у верхнего края страницы; дыра в глубине страницы и бумага вне книги — нет."""
    from ocr_utils.scan_cropping.v2.edge_zones import union_levels_edges

    h, w = 600, 400
    inst = np.zeros((h, w), np.uint8)
    inst[50:550, 20:190] = 1
    inst[50:550, 210:380] = 2
    inst[50:80, 150:190] = 0  # выемка у верха левой страницы, у корешка (30 px ≈ 4.3 мм)
    inst[300:320, 60:80] = 0  # дыра в глубине страницы
    raw = inst > 0
    raw_lv = np.zeros((h, w), bool)
    raw_lv[50:550, 20:380] = True  # растянутый видит всё, в том числе щель у корешка
    raw_lv[10:30, 20:380] = True  # и полосу над страницами вне книги
    book = np.zeros((h, w), bool)
    book[40:560, 10:390] = True
    inst_lv = np.zeros((h, w), np.uint8)
    inst_lv[:, :200] = 1  # экземпляры растянутого прохода — по страницам
    inst_lv[:, 200:] = 2
    inst_lv[~raw_lv] = 0
    out = union_levels_edges(raw, inst, raw_lv, inst_lv, book)
    assert out.added_top[50:80, 150:190].all() and (out.inst[50:80, 150:190] == 1).all()
    assert not out.raw[300:320, 60:80].any()  # глубже EDGES_IN_MM — не трогается
    assert not out.raw[10:30].any()  # вне книги — не трогается
