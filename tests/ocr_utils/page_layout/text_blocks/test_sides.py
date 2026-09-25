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
