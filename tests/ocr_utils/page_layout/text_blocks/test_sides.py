"""Стороны границы блока и выравнивание по ним (``text_blocks.sides``) на синтетических страницах."""

from __future__ import annotations

import numpy as np
import pytest

from ocr_utils.page_layout.text_blocks import sides as sd
from ocr_utils.page_layout.text_blocks.blocks import _row_axis
from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
from ocr_utils.page_layout.text_blocks.page import analyse_gray
from tests.ocr_utils.page_layout.text_blocks.synthetic import bowed, column_page, sheared

ENGINE = InkEngine()


def _block(image: np.ndarray):
    """Самый многострочный блок разбора синтетической страницы."""
    analysis = analyse_gray(image, ENGINE)
    return max(analysis.blocks, key=lambda block: len(block.rows))


@pytest.fixture(scope="module")
def pages():
    """Одна колонка по формату: прямая, перекошенная и выгнутая аркой."""
    plain = column_page(columns=1, justify="both")
    return {"plain": _block(plain), "sheared": _block(sheared(plain)), "bowed": _block(bowed(plain))}


def _changes(labels: tuple) -> list[tuple]:
    """Смены метки по кругу: пары (было, стало)."""
    return [(labels[i - 1], labels[i]) for i in range(len(labels)) if labels[i - 1] is not labels[i]]


@pytest.mark.parametrize("method", list(sd.SidesMethod))
@pytest.mark.parametrize("name", ["plain", "sheared", "bowed"])
def test_four_arcs_in_cycle_order(pages, method, name):
    """Ровно четыре дуги в порядке обхода: в каждом углу одна смена, непомеченных звеньев нет."""
    sides = sd.sides_of(pages[name], method)
    assert len(sides.labels) == len(sides.polygon) == len(sides.corner)
    changes = _changes(sides.labels)
    assert len(changes) == 4
    for before, after in changes:
        assert sd.CYCLE[(sd.CYCLE.index(before) + 1) % 4] is after


@pytest.mark.parametrize("name", ["plain", "sheared", "bowed"])
def test_methods_agree_on_a_clean_column(pages, name):
    """На чистой колонке методы расходятся только у углов — не больше 5 % длины контура."""
    result = {method: sd.sides_of(pages[name], method) for method in sd.SidesMethod}
    for first, second in ((sd.SidesMethod.CONSTRUCT, sd.SidesMethod.RAYS), (sd.SidesMethod.RAYS, sd.SidesMethod.FRAME)):
        assert sd.label_agreement(result[first], result[second]) >= 0.95


def test_vertical_sides_are_vertical_and_corners_are_excluded(pages):
    """Вертикальные стороны прямой колонки стоят вертикально, углы — малая доля длины."""
    for method in sd.SidesMethod:
        measures = {item.side: item for item in sd.side_measures(sd.sides_of(pages["plain"], method), 150.0)}
        for side in (sd.SideKind.LEFT, sd.SideKind.RIGHT):
            assert abs(measures[side].tilt_deg) < 1.0, (method, side)
            assert measures[side].corner_share < 0.2, (method, side)


def test_corner_split_is_a_single_cut():
    """Пила в углу: дуга угла делится одним разрезом с наименьшей ценой."""
    # Шесть звеньев: цена «верха» (метка 3) и «лева» (метка 0) чередуется, как на зубчатом углу.
    costs = np.zeros((6, 4))
    costs[:, 3] = [0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    costs[:, 0] = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    cut = sd._split_corner(costs, 3, 0)
    totals = [costs[:k, 3].sum() + costs[k:, 0].sum() for k in range(7)]
    assert totals[cut] == min(totals)
    assert cut == 3  # середина диапазона лучших разрезов {1, 3}: 1 + 0 = 1 у обоих, берётся средний


def test_single_indent_does_not_break_a_run():
    """Абзацный отступ между рядами на кривой серию не рвёт, два мимо подряд — рвут."""
    on, indent, off = sd.RowStatus.ON, sd.RowStatus.INDENT, sd.RowStatus.OFF
    assert sd.aligned_runs([on, on, indent, on, on]).all()
    runs = sd.aligned_runs([on, on, off, off, on, on, on])
    assert list(runs) == [False, False, False, False, True, True, True]


@pytest.mark.parametrize("method", [sd.AlignMethod.TANGENT, sd.AlignMethod.ROBUST])
@pytest.mark.parametrize("name", ["plain", "sheared", "bowed"])
def test_justified_column_is_aligned_on_both_sides(pages, method, name):
    """Выключка по формату: обе стороны выровнены и на наклоне, и на изгибе."""
    for side in (sd.SideKind.LEFT, sd.SideKind.RIGHT):
        result = sd.side_alignment(pages[name], side, method)
        assert result is not None and result.on_share >= 0.8, (name, side, result.on_share)


@pytest.mark.parametrize("method", list(sd.AlignMethod))
def test_ragged_right_side_is_not_aligned(method):
    """Выключка влево: правая сторона рваная, левая выровнена.

    Мерой служит доля рядов в выровненных СЕРИЯХ: доля «на кривой» на рваном крае обманчива — ряды,
    ушедшие внутрь дальше ``INDENT_MIN_MM``, из неё исключены как отступы (замер: 0.62 у ``trend``
    при 0.22 в сериях).
    """
    block = _block(column_page(columns=1, justify="left"))
    left = sd.side_alignment(block, sd.SideKind.LEFT, method)
    right = sd.side_alignment(block, sd.SideKind.RIGHT, method)
    assert left.aligned_share >= 0.8
    assert right.aligned_share < 0.5


def _whitened(image: np.ndarray, rows: list[tuple[int, float, float]]) -> np.ndarray:
    """Забелить у рядов разбора полосу краски по x: ``(номер ряда, доля слева, доля справа)``.

    Координаты разбора — рабочая копия (150 dpi), картинка — вдвое крупнее. Так из полной строки
    получается подпись справа (забелена левая часть) или абзацный отступ (забелено начало).
    """
    block = _block(image)
    out = image.copy()
    width = block.envelope.polygon[:, 0].max() - block.envelope.polygon[:, 0].min()
    left = block.envelope.polygon[:, 0].min()
    for index, start, stop in rows:
        row = block.rows[index]
        y0, y1 = int((row.y - row.height) * 2), int((row.y + row.height) * 2)
        x0, x1 = int((left + start * width) * 2) - 4, int((left + stop * width) * 2) + 4
        out[max(0, y0) : y1, max(0, x0) : x1] = 255
    return out


def _extent(sides: sd.BlockSides, side: sd.SideKind) -> float:
    """Размах по x звеньев стороны."""
    xs = sides.polygon[[label is side for label in sides.labels], 0]
    return float(np.ptp(xs)) if xs.size else 0.0


def test_short_last_row_is_extended_and_bottom_spans_the_block():
    """Подпись справа в последней строке: ось продлена по предпоследней, низ — во всю ширину блока."""
    block = _block(_whitened(column_page(columns=1, justify="both"), [(-1, 0.0, 0.7)]))
    edge = sd.edge_axis(block, top=False)
    assert edge.extended and edge.reference == len(block.rows) - 2
    assert edge.gap == pytest.approx(block.pitch_px, rel=0.25)
    sides = sd.sides_rays(block)
    width = float(np.ptp(block.envelope.polygon[:, 0]))
    assert _extent(sides, sd.SideKind.BOTTOM) >= 0.8 * width
    measures = {item.side: item for item in sd.side_measures(sides, block.dpi)}
    assert abs(measures[sd.SideKind.LEFT].tilt_deg) < 1.0


def test_indented_first_row_is_extended_and_top_spans_the_block():
    """Абзацный отступ первой строки: ось продлена влево по второй, верх — во всю ширину."""
    block = _block(_whitened(column_page(columns=1, justify="both"), [(0, 0.0, 0.3)]))
    edge = sd.edge_axis(block, top=True)
    assert edge.extended and edge.reference == 1 and edge.gap < 0
    width = float(np.ptp(block.envelope.polygon[:, 0]))
    assert _extent(sd.sides_rays(block), sd.SideKind.TOP) >= 0.8 * width


def test_full_edge_rows_are_not_extended(pages):
    """Полная последняя строка не продлевается; первая строка синтетики начинается с абзацного
    отступа (x 84 при крае 64 — 3.4 мм, больше ``TRIM_MM``) и законно продлевается влево по второй."""
    assert not sd.edge_axis(pages["plain"], top=False).extended
    first = sd.edge_axis(pages["plain"], top=True)
    assert first.extended and first.reference == 1 and first.points[-1, 0] == pytest.approx(first.real_x1, abs=6.0)


def test_short_penultimate_row_is_skipped_for_the_nearest_full_one():
    """Предпоследняя тоже короткая (конец абзаца над подписью) — опорой берётся полная выше."""
    image = _whitened(column_page(columns=1, justify="both"), [(-2, 0.4, 1.0), (-1, 0.0, 0.7)])
    block = _block(image)
    edge = sd.edge_axis(block, top=False)
    # Опора — ближайшая ПОЛНАЯ строка выше: все ряды между ней и последней — короткие (забеленная
    # предпоследняя и, в синтетике, ещё начало абзаца с отступом).
    assert edge.extended and edge.reference <= len(block.rows) - 3
    assert sd._is_full(block, _row_axis(block.rows[edge.reference]))
    assert not any(sd._is_full(block, _row_axis(row)) for row in block.rows[edge.reference + 1 :])
    assert edge.gap == pytest.approx((len(block.rows) - 1 - edge.reference) * block.pitch_px, rel=0.15)


def test_default_method_is_construct(pages):
    """По умолчанию стороны размечаются по построению огибающей; остальные методы доступны явно."""
    assert sd.DEFAULT_SIDES_METHOD is sd.SidesMethod.CONSTRUCT
    block = pages["bowed"]
    assert sd.sides_of(block).labels == sd.sides_construct(block).labels
    assert sd.sides_of(block, sd.SidesMethod.RAYS).method is sd.SidesMethod.RAYS


def test_uncertain_ends_are_on_top_and_bottom_only(pages):
    """Неуверенные концы — только у верха и низа, по букве крайнего ряда с каждого конца; в меры не входят."""
    block = pages["plain"]
    sides = sd.sides_of(block)
    assert sides.uncertain is not None and sides.uncertain.any()
    labels = np.array([label.value for label in sides.labels])
    assert set(labels[sides.uncertain]) <= {"top", "bottom"}
    _, _, lengths = sd.outward_normals(sides.polygon)
    for side, row in ((sd.SideKind.TOP, block.rows[0]), (sd.SideKind.BOTTOM, block.rows[-1])):
        own = labels == side.value
        letter = max(row.glyph_w, row.glyph_h) or max(block.glyph_size)
        assert float(lengths[own & sides.uncertain].sum()) == pytest.approx(2 * letter, abs=4 * 3.0)


def test_fill_side_gaps_drops_ragged_ends_and_patches_the_middle():
    """Невыровненные концы отброшены, невыровненная середина — заплатка, выровненные точки на месте."""
    v = np.arange(8, dtype=np.float64) * 10.0
    u = np.zeros(8)
    aligned = np.array([False, True, True, False, False, True, True, False])
    line_v, line_u, filled = sd.fill_side_gaps(v, u, aligned, step=3.0)
    assert line_v[0] == pytest.approx(10.0) and line_v[-1] == pytest.approx(60.0)
    assert np.all(np.diff(line_v) > 0)
    # Заплатка — строго внутри разрыва 20…50, с узлами не реже шага.
    assert np.all((line_v[filled] > 20.0) & (line_v[filled] < 50.0))
    assert filled.sum() >= 9
    assert not filled[line_v <= 20.0].any() and not filled[line_v >= 50.0].any()


def test_fill_side_gaps_ignores_a_bump_in_the_ragged_middle():
    """Выступ на невыровненном участке прямой стороны заплатка не повторяет: линия прямая, изгиб ≈ 0."""
    v = np.arange(20, dtype=np.float64) * 10.0
    u = 0.1 * v
    aligned = np.ones(20, dtype=bool)
    u[8:11] += 15.0  # выступ, например кромка у короткого конца абзаца
    aligned[8:11] = False
    line_v, line_u, filled = sd.fill_side_gaps(v, u, aligned, step=3.0)
    assert filled.any()
    assert np.allclose(line_u, 0.1 * line_v, atol=1e-6)
    raw = sd._tilt_bend(np.column_stack([u, v]), True, 150.0)[1]
    fixed = sd._tilt_bend(np.column_stack([line_u, line_v]), True, 150.0)[1]
    assert raw > 2.0 and fixed < 0.01


def test_fill_side_gaps_needs_two_aligned_levels():
    """Меньше двух выровненных точек на разных высотах — линии нет; всё выровнено — линия и есть сторона."""
    v = np.arange(5, dtype=np.float64)
    u = np.zeros(5)
    assert sd.fill_side_gaps(v, u, np.array([False, False, True, False, False]), step=1.0) is None
    line_v, line_u, filled = sd.fill_side_gaps(v, u + v, np.ones(5, dtype=bool), step=0.3)
    assert not filled.any()
    assert np.array_equal(line_v, v) and np.array_equal(line_u, u + v)


@pytest.mark.parametrize("method", list(sd.AlignMethod))
def test_filled_side_on_a_justified_column_matches_the_side(pages, method):
    """У колонки по формату дополнительная линия — сама сторона: меры те же, что у всей стороны."""
    block = pages["plain"]
    sides = sd.sides_of(block)
    measures = {item.side: item for item in sd.side_measures(sides, block.dpi)}
    for side in (sd.SideKind.LEFT, sd.SideKind.RIGHT):
        line = sd.filled_side(sides, sd.side_alignment(block, side, method), block)
        assert line is not None and line.side is side and line.method is method
        assert line.raw_tilt_deg == pytest.approx(measures[side].tilt_deg, abs=1e-9)
        assert line.raw_bend_mm == pytest.approx(measures[side].bend_mm, abs=1e-9)
        assert abs(line.tilt_deg - line.raw_tilt_deg) < 0.2


def test_robust_side_follows_changing_slope():
    """Колонка по формату, край которой меняет наклон по высоте (S-образно): стороны выровнены.

    Одна прямая или парабола на весь блок такой край не описывает; местная кривая
    (``sides._fit_local``) — да.
    """
    import cv2

    from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
    from ocr_utils.page_layout.text_blocks.page import analyse_gray
    from ocr_utils.page_layout.text_blocks.sides import AlignMethod, SideKind, side_alignment
    from tests.ocr_utils.page_layout.text_blocks.synthetic import column_page
    from tests.ocr_utils.scan_markup.synthetic import PAPER

    page = column_page(columns=1)
    height, width = page.shape
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    shift = (40.0 * np.sin(2 * np.pi * ys / height)).astype(np.float32)
    page = cv2.remap(page, xs - shift, ys, cv2.INTER_LINEAR, borderValue=int(PAPER))
    analysis = analyse_gray(page, InkEngine())
    block = max(analysis.blocks, key=lambda item: item.lines)
    for side in (SideKind.LEFT, SideKind.RIGHT):
        assert side_alignment(block, side, AlignMethod.ROBUST).aligned_share >= 0.9, side


def test_filled_side_patches_an_unreliable_span(pages):
    """Недостоверный участок в середине стороны закрывается заплаткой, концы и остальное — как без него."""
    block = pages["plain"]
    sides = sd.sides_of(block)
    alignment = sd.side_alignment(block, sd.SideKind.LEFT, sd.AlignMethod.ROBUST)
    plain = sd.filled_side(sides, alignment, block)
    top, bottom = float(plain.points[0, 1]), float(plain.points[-1, 1])
    middle = (top + bottom) / 2
    span = (middle - 10.0, middle + 10.0)
    patched = sd.filled_side(sides, alignment, block, (span,))
    assert patched is not None
    inside = (patched.points[:, 1] > span[0]) & (patched.points[:, 1] < span[1])
    assert inside.any() and patched.filled[inside].all()
    assert not patched.filled[~inside].any() or not plain.filled.any()
    # Концы линии и её меры на ровной колонке не меняются: заплатка идёт по той же прямой.
    assert patched.points[0, 1] == pytest.approx(top) and patched.points[-1, 1] == pytest.approx(bottom)
    assert patched.tilt_deg == pytest.approx(plain.tilt_deg, abs=0.1)
