"""Тесты уточнения ленты у корешка (``research.gutter_strip.refine``) на синтетике: конец участка маски, прямая края с
выбросами, отклик тонкой линии, выбор края с вытеснением сильной линией ближе к своей странице."""

from types import SimpleNamespace

import numpy as np

from research.gutter_strip.refine import (
    EdgeMeasures,
    Engine,
    Method,
    MethodResult,
    choose_edge,
    line_fit,
    line_response,
    run_end,
)

PPM = 13.8


def fake_band(length: int, width: int = 691) -> SimpleNamespace:
    """Лента-заглушка: пролёт страниц — вся длина, 13.8 px/мм."""
    return SimpleNamespace(ppm=PPM, page_rows=(0, length - 1), length=length, width=width)


def test_run_end_both_directions():
    """Конец сплошного участка: вправо — первый столбец вне маски, влево — последний столбец маски; вне маски — NaN."""
    mask = np.zeros((3, 20), bool)
    mask[0, 2:10] = True
    mask[1, 12:18] = True
    right = run_end(mask, 5, 1)
    assert right[0] == 10 and np.isnan(right[1]) and np.isnan(right[2])
    left = run_end(mask, 15, -1)
    assert left[1] == 12 and np.isnan(left[0])


def test_line_fit_ignores_chunk_on_tongue_tips():
    """Треть строк ушла на кончики языков (на 7 мм к корешку) — прямая по остальным, они «на прямой», выбросы — нет."""
    n = 3000
    rows = np.arange(n)
    edge = 400 + 0.02 * rows
    edge[1000:2000] -= 7 * PPM
    on, share, p90 = line_fit(edge.astype(float), np.ones(n, bool), fake_band(n))
    cut = int(15 * PPM)
    assert on[cut:1000].all() and not on[1000:2000].any() and on[2000 : n - cut].all()
    assert 0.6 < share < 0.7
    assert p90 < 0.05


def test_line_response_sees_thin_dotted_line():
    """Пунктирная тёмная линия шириной 4 px на белой бумаге даёт отклик складки, ровная бумага — нет."""
    h, w = 600, 300
    img = np.full((h, w, 3), 245, np.uint8)
    img[::4, 150:154] = 200  # пунктир: тёмная каждая четвёртая строка
    img[1::4, 150:154] = 200
    resp = line_response(img, PPM)
    assert resp[100:500, 151].mean() > 8
    assert resp[100:500, 60].mean() < 1


def result(name: str, engine: Engine, right_mm: float, support: float, rough: float, n: int = 400) -> MethodResult:
    """Принятый способ с правым краем на ``right_mm`` от разделителя (столбцы ленты с ``half`` = 345)."""
    edge = np.full(n, 345 + right_mm * PPM)
    m = EdgeMeasures(cover=1.0, rough_mm=rough, support_l=support, ink_out_mm2=0.0, accepted=True)
    res = MethodResult(Method(engine, 4), edge, edge, right_m=m, left_m=EdgeMeasures())
    res.right_clean = edge
    res.left_clean = edge
    return res


def test_choose_consensus_among_nets_not_straight_line():
    """Прямая без сети с умеренной опорой не выигрывает у сети только тем, что «ровнее» (ПХ 1975/10 IMG_0067)."""
    net = result("sam3", Engine.SAM3, 0.65, support=11.6, rough=0.07)
    solid = result("solid", Engine.SOLID, 4.0, support=6.4, rough=0.0)
    assert choose_edge([net, solid], page_left=False, ppm=PPM) is net


def test_strong_line_ahead_overrides_consensus():
    """Сильная линия на 13 мм ближе к правой странице вытесняет согласный выбор сетей у сгиба (ПХ 1975/04 IMG_0083)."""
    nets = [result("sam3", Engine.SAM3, 3.3, support=17.0, rough=0.07), result("sam2", Engine.SAM2, 3.8, 15.0, 0.2)]
    solid = result("solid", Engine.SOLID, 16.3, support=13.7, rough=0.0)
    assert choose_edge([*nets, solid], page_left=False, ppm=PPM) is solid


def fake_region_frame(h: int = 600, w: int = 300):
    """Кадр-заглушка для области полосы: страницы — по сторонам от столбцов 100 / 160 в строках 20…580, книга вся."""
    left = np.zeros((h, w), bool)
    right = np.zeros((h, w), bool)
    left[20:580, :100] = True
    right[20:580, 100:] = True
    pages = [SimpleNamespace(mask=left), SimpleNamespace(mask=right)]
    frame = SimpleNamespace(img=np.zeros((h, w, 3), np.uint8), pages=pages, book=np.ones((h, w), bool))
    band = SimpleNamespace(
        ys_seg=np.arange(h, dtype=float), ratio=1.0, x_full=lambda e, rows=None: np.asarray(e, float)
    )
    return SimpleNamespace(frame=frame), band


def test_region_on_lower_part_of_spine():
    """Полоса только в нижней половине: одна серия, выше неё полосы нет, ширина — по строкам."""
    from research.gutter_strip.refine_region import build_region

    sf, band = fake_region_frame()
    left = np.full(600, 100.0)
    right = np.full(600, 101.0)
    right[300:] = 160.0  # 60 px ≈ 8.7 мм на 6.9 px/мм
    region = build_region(sf, band, left, right)
    assert region is not None and len(region.runs) == 1
    a, b = region.runs[0]
    assert 295 <= a <= 305 and b == 580
    assert not region.mask[:290].any() and region.mask[400, 100:160].all()


def test_snap_end_through_bridge():
    """Полоса кончается в 1.5 мм под верхом страниц, над ней — бумага (перемычка): конец доводится до края листа."""
    from research.gutter_strip.refine_region import build_region, snap_ends

    sf, band = fake_region_frame()
    left = np.full(600, 100.0)
    right = np.full(600, 101.0)
    right[30:] = 160.0
    region = build_region(sf, band, left, right)
    paper = np.zeros((600, 300), bool)
    paper[20:580] = True
    region = snap_ends(region, sf.frame.pages, paper)
    assert "top" in region.snapped and region.touch_top
    assert region.mask[20:30, 100:160].all()


def test_spine_follows_junction_for_partial_widening_strip():
    """Полоса в нижней половине расширяется вправо (левый край — продолжение стыка страниц): линия корешка — по стыку и
    левому краю полосы, а не по середине полосы (ПХ 1975/10 IMG_0081)."""
    from ocr_utils.scan_cropping.v2.spread_lines import Line
    from research.gutter_strip.refine_region import build_region, region_spine

    sf, band = fake_region_frame()
    left = np.full(600, 100.0)
    right = np.full(600, 101.0)
    right[300:] = np.linspace(140.0, 190.0, 300)  # 6…13 мм, расширяется книзу
    region = build_region(sf, band, left, right)
    lmask = np.zeros((600, 300), bool)
    rmask = np.zeros((600, 300), bool)
    for y in range(20, 580):
        lmask[y, :100] = True
        rmask[y, int(np.ceil(right[y])) :] = True
    pages = [SimpleNamespace(mask=lmask), SimpleNamespace(mask=rmask)]
    divider = Line(100.0, 300.0, 0.0, 20.0, 580.0)
    spine, source = region_spine(region, divider, pages)
    assert "стык" in source
    assert abs(spine.x_at(40.0) - 100.0) < 2 and abs(spine.x_at(560.0) - 100.0) < 2
