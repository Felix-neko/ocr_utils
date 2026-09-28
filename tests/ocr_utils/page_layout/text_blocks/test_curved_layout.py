"""Разбор текста с кривыми строками: оси строк, колонки, огибающие блоков и выключка."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from ocr_utils.page_layout.text_blocks import WORK_DPI
from ocr_utils.page_layout.text_blocks.alignment import AlignKind
from ocr_utils.page_layout.text_blocks.blocks import (
    BlocksMode,
    EXTEND_SLOPE_LIMIT_DEG,
    BODY_QUANTILE,
    CapKind,
    Row,
    _axis_slope,
    _same_row,
    _style_break,
    envelope_of,
)
from ocr_utils.page_layout.text_blocks import blocks
from ocr_utils.page_layout.text_blocks import segment as seg
from ocr_utils.page_layout.text_blocks.segment import _edge_marks
from ocr_utils.page_layout.text_blocks.columns import Gutter, Zone, gutters_of, inside_gutter, zones_of
from ocr_utils.page_layout.text_blocks.lines import LineAxis
from ocr_utils.page_layout.text_blocks.leaders import Leader, leaders_of
from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
from ocr_utils.page_layout.text_blocks.page import analyse_gray
from ocr_utils.page_layout import px_to_mm
from tests.ocr_utils.page_layout.text_blocks.synthetic import (
    bowed,
    column_page,
    inset_page,
    leader_page,
    sheared,
    single_line_page,
)

ENGINE = InkEngine()


def test_axes_follow_bow():
    """Ось строки повторяет заданную дугу: прогиб оси совпадает с заданным с точностью 15 %."""
    amplitude = 30.0  # пиксели рендера 300 dpi
    page = bowed(column_page(columns=1), amplitude)
    analysis = analyse_gray(page, ENGINE)
    long_axes = [axis for axis in analysis.axes if axis.length_mm > 80]
    assert len(long_axes) > 10

    def bow(x: float) -> float:
        """Заданное поле смещения в пикселях рабочей копии (деформация задана в 300 dpi)."""
        half = page.shape[1] / 2.0
        return amplitude / 2.0 * (1.0 - ((x * 2.0 - half) / half) ** 2)

    # Ожидаемая сагитта считается по КОНЦАМ ОСИ: строка не доходит до краёв страницы, где
    # деформация нулевая, поэтому её прогиб меньше амплитуды поля.
    errors = []
    for axis in long_axes:
        chord = (bow(axis.x0) + bow(axis.x1)) / 2.0
        expected = px_to_mm(bow((axis.x0 + axis.x1) / 2.0) - chord, axis.dpi)
        errors.append(abs(axis.sagitta_mm - expected) / max(expected, 1e-6))
    assert float(np.median(errors)) < 0.2


def test_two_columns_two_blocks():
    """Страница в две колонки: два блока, огибающие не пересекают межколонник."""
    analysis = analyse_gray(column_page(columns=2), ENGINE)
    assert len(analysis.blocks) == 2
    left, right = analysis.blocks
    assert left.envelope.right[:, 0].max() < right.envelope.left[:, 0].min()


def test_three_columns():
    """Страница в три колонки: три блока и два межколонника."""
    analysis = analyse_gray(column_page(columns=3), ENGINE)
    assert len(analysis.blocks) == 3
    assert len(analysis.gutters) == 2


def test_alignment_kinds():
    """Выключка: по формату — ``both``, рваный правый край — ``left``, рваный левый — ``right``."""
    for justify, expected in (("both", AlignKind.BOTH), ("left", AlignKind.LEFT), ("right", AlignKind.RIGHT)):
        analysis = analyse_gray(column_page(columns=1, justify=justify), ENGINE)
        assert analysis.alignments, justify
        assert analysis.alignments[0].kind is expected, justify


def test_alignment_survives_bow():
    """Выключка по формату не теряется на изогнутой странице: меры считаются от огибающей."""
    analysis = analyse_gray(bowed(column_page(columns=1), 30.0), ENGINE)
    assert analysis.alignments[0].kind is AlignKind.BOTH
    # Огибающая изогнутого блока не обязана быть прямой, но должна быть ГЛАДКОЙ: соседние узлы
    # сетки отстоят друг от друга меньше чем на десятую долю миллиметра.
    left = analysis.blocks[0].envelope.left
    assert px_to_mm(float(np.abs(np.diff(left[:, 0])).max()), analysis.dpi) < 0.1


def test_inset_gutter_lives_on_part_of_height():
    """Врезка сверху справа: межколонник живёт на части высоты, зон становится две."""
    page = inset_page()
    work = page[::2, ::2]
    gutters = gutters_of(work, WORK_DPI)
    assert gutters, "межколонник врезки не найден"
    height, width = work.shape
    assert max(g.y1 for g in gutters) < height * 0.9
    zones = zones_of(gutters, height, width, WORK_DPI)
    assert len(zones) >= 2
    assert max(len(zone.columns) for zone in zones) == 2


def test_envelope_ignores_paragraph_indent():
    """Абзацный отступ не тянет огибающую внутрь блока: он считается отдельно, как отступ."""
    analysis = analyse_gray(column_page(columns=1, indent=60), ENGINE)
    block = analysis.blocks[0]
    alignment = analysis.alignments[0]
    assert alignment.left.indent_rows >= 3
    left_edges = np.array([row.x0 for row in block.rows])
    envelope = np.interp([row.y for row in block.rows], block.envelope.left[:, 1], block.envelope.left[:, 0])
    # Огибающая идёт по телу блока, а не по отступам: медиана остатка близка к нулю.
    assert abs(float(np.median(left_edges - envelope))) < 3.0


def test_dilated_outline_contains_envelope():
    """Раздутая граница блока содержит исходную и не рвётся на куски."""
    analysis = analyse_gray(column_page(columns=1), ENGINE)
    block = analysis.blocks[0]
    outline = block.envelope.polygon_dilated
    assert outline is not None and len(outline) >= 4
    contour = outline.astype(np.float32).reshape(-1, 1, 2)
    inside = [cv2.pointPolygonTest(contour, (float(x), float(y)), False) for x, y in block.envelope.polygon]
    assert min(inside) >= 0.0


def test_dilated_outline_is_smoother():
    """Дилатация делает границу ровнее: разброс правой кромки по y падает (прежняя граница, ``BlocksMode.LEGACY``).

    У гладкой границы (``SMOOTH``) сторона выровненной колонки — прямая, разброс уже нулевой, и дилатации
    выравнивать нечего.
    """
    analysis = analyse_gray(column_page(columns=1), ENGINE, blocks_mode=BlocksMode.LEGACY)
    block = analysis.blocks[0]
    outline = block.envelope.polygon_dilated
    assert outline is not None

    def spread(points: np.ndarray) -> float:
        """Разброс абсцисс правой половины контура: чем больше, тем кромка зубчатее."""
        right = points[points[:, 0] > np.median(points[:, 0])]
        return float(np.std(right[:, 0])) if right.size else 0.0

    assert spread(outline) <= spread(block.envelope.polygon) + 1e-6


def test_dilation_keeps_columns_apart():
    """Дилатация на полсимвола не сводит соседние колонки: контуры не пересекаются."""
    analysis = analyse_gray(column_page(columns=2), ENGINE)
    left, right = analysis.blocks
    assert left.envelope.polygon_dilated is not None and right.envelope.polygon_dilated is not None
    assert left.envelope.polygon_dilated[:, 0].max() < right.envelope.polygon_dilated[:, 0].min()


def test_leaders_found_and_not_a_gutter():
    """Отточия собираются в цепочки, а поле точек не становится межколонником."""
    page = leader_page()
    work = page[::2, ::2]
    found, _ = leaders_of(work, WORK_DPI)
    assert len(found) >= 10, "цепочки точек не найдены"
    assert not gutters_of(work, WORK_DPI, found), "поле отточий принято за межколонник"


def test_leader_row_keeps_metrics():
    """Точки отточия не идут в меры набора: штрих и размер символа как у строки без отточий."""
    analysis = analyse_gray(leader_page(), ENGINE)
    rows = [row for block in analysis.blocks for row in block.rows if row.stroke > 0]
    assert rows, "рядов не нашлось"
    plain = analyse_gray(column_page(columns=1), ENGINE)
    plain_rows = [row for block in plain.blocks for row in block.rows if row.stroke > 0]
    dotted = float(np.median([row.stroke for row in rows]))
    plain_stroke = float(np.median([row.stroke for row in plain_rows]))
    # Точка отточия вдвое толще штриха набора: если её не исключать, медиана ряда таблицы уезжает
    # в полтора-два раза. Допуск в 25 % оставлен на разброс самой синтетики.
    assert dotted <= 1.25 * plain_stroke, f"штрих ряда с отточием {dotted} против {plain_stroke}"


def test_single_row_envelope_has_no_whiskers():
    """Контур однострочного блока не торчит за краску ряда (вертикальных «усов» нет)."""
    analysis = analyse_gray(single_line_page(), ENGINE)
    assert len(analysis.blocks) == 1 and len(analysis.blocks[0].rows) == 1
    block = analysis.blocks[0]
    row = block.rows[0]
    top, bottom = float(row.top_edge[:, 1].min()), float(row.bottom_edge[:, 1].max())
    margin = 3.0  # пиксели рабочей копии: запас огибающей наружу
    assert block.envelope.left[:, 1].min() >= top - margin
    assert block.envelope.left[:, 1].max() <= bottom + margin


def _half_rows(analysis) -> int:
    """Ряды-половинки: шаг до соседнего ряда меньше 0.7 медианного шага блока."""
    count = 0
    for block in analysis.blocks:
        ys = np.sort(np.array([row.y for row in block.rows], dtype=np.float64))
        if ys.size < 3:
            continue
        steps = np.diff(ys)
        median = float(np.median(steps))
        if median > 0:
            count += int((steps < 0.7 * median).sum())
    return count


def test_sheared_page_has_no_half_rows():
    """Перекошенная страница в две колонки: строка не разваливается на две половинки."""
    analysis = analyse_gray(sheared(column_page(columns=2), 26.0), ENGINE)
    assert len(analysis.blocks) == 2
    assert _half_rows(analysis) == 0


def test_sheared_page_keeps_lines_whole():
    """На перекосе ось идёт по всей строке: длинных осей столько же, сколько на ровной странице."""
    straight = analyse_gray(column_page(columns=2), ENGINE)
    skewed = analyse_gray(sheared(column_page(columns=2), 26.0), ENGINE)
    long_straight = sum(1 for axis in straight.axes if axis.length_mm > 40)
    long_skewed = sum(1 for axis in skewed.axes if axis.length_mm > 40)
    assert long_skewed >= 0.9 * long_straight


def test_single_line_page_falls_back():
    """Страница без якорных строк разбирается прежним ходом и не падает."""
    analysis = analyse_gray(single_line_page(), ENGINE)
    assert len(analysis.axes) >= 1


def _fragment(x0: float, x1: float, y: float, height: float, jitter: float = 0.0) -> LineAxis:
    """Короткий обрывок строки: восемь узлов на высоте ``y``, крайний сбит на ``jitter`` px.

    Сбитый крайний узел — это шум формы буквы; на семимиллиметровом обрывке он и даёт мнимый
    наклон конца в десятки градусов.
    """
    xs = np.linspace(x0, x1, 8)
    ys = np.full(xs.shape, float(y))
    ys[0] += jitter
    return LineAxis(
        points=np.column_stack([xs, ys]),
        height=height,
        column=0,
        dpi=WORK_DPI,
        cross=False,
        sagitta_mm=0.0,
        slope_deg=0.0,
        bend_mm=0.0,
        resid_parabola_mm=0.0,
    )


def test_row_joins_fragments_across_long_gap():
    """Два обрывка строки на одной высоте — один ряд, даже если между ними 18 мм пустоты.

    Так выглядит последняя строка сноски «ч. II,» + «с. 421» (1973/08 с.85): у правого обрывка
    длиной 7 мм наклон начала выходит шумовым (+0.7, то есть 35 градусов), и продолжение осей
    навстречу без ограничений расходилось на 24 px при допуске 8.
    """
    left = _fragment(520, 610, 1396, height=13.0)
    right = _fragment(827, 902, 1397, height=15.5, jitter=-3.0)
    # Сбитый крайний узел даёт мнимый наклон начала круче правдоподобного, и он обрезается.
    assert _axis_slope(right, True) == pytest.approx(math.tan(math.radians(EXTEND_SLOPE_LIMIT_DEG)))
    assert _same_row([left], right)


def _row(x0: float, x1: float, ys: list[float], height: float) -> Row:
    """Ряд блока с профилями краски: ось по заданным ординатам, краска — полвысоты вокруг неё."""
    xs = np.linspace(x0, x1, len(ys))
    points = np.column_stack([xs, np.asarray(ys, dtype=float)])
    axis = LineAxis(
        points=points,
        height=height,
        column=0,
        dpi=WORK_DPI,
        cross=False,
        sagitta_mm=0.0,
        slope_deg=0.0,
        bend_mm=0.0,
        resid_parabola_mm=0.0,
    )
    return Row(
        y=float(np.median(points[:, 1])),
        height=height,
        x0=x0,
        x1=x1,
        axes=(axis,),
        top_edge=np.column_stack([xs, points[:, 1] - height / 2.0]),
        bottom_edge=np.column_stack([xs, points[:, 1] + height / 2.0]),
    )


def _line_with_glyphs(first_width: float, last_width: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Прямая центр-линия и боксы букв строки: крайние буквы заданной ширины, между ними мелкие.

    Абсциссы центр-линии — пиксели рендера от начала сегмента (k = 2, начало строки x0 = 100
    рабочей копии), боксы букв — пиксели рабочей копии, как их отдаёт ``_glyph_boxes``.
    """
    xs = np.arange(240, dtype=float)
    ys = np.full(240, 20.0)
    boxes = [(100.0, 100.0 + first_width, first_width, 15.0)]
    boxes += [(x, x + 8.0, 8.0, 15.0) for x in np.arange(100.0 + first_width + 4, 220.0 - last_width - 4, 12.0)]
    boxes.append((220.0 - last_width, 220.0, last_width, 15.0))
    return xs, ys, np.array(boxes, dtype=float)


def test_flatten_ends_clamps_a_wide_first_letter():
    """Крупная первая буква не утаскивает ось, хотя краски в её крайних столбцах много.

    1975/05 с.97, «Нам пишут…»: первый глиф 37 px при высоте строки 32, и ось уезжала на 18.7 px
    рабочей копии. Ворота по количеству краски такой случай пропускают — столбцы крупной буквы
    полны; ворота по отклонению ловят.
    """
    xs, ys, boxes = _line_with_glyphs(first_width=20.0, last_width=8.0)
    zone = xs <= (100.0 + 20.0 + seg.END_GLYPH_MARGIN * 20.0 - 100.0) * 2.0
    ys[zone] = 8.0  # угол буквы: центр масс уехал на полвысоты строки
    out = seg._flatten_ends(xs, ys, boxes, height_px=30.0, k=2.0, x0=100.0)
    assert out[zone] == pytest.approx(20.0 - seg.END_FLAT_TOLERANCE_HEIGHTS * 30.0)
    assert out[~zone] == pytest.approx(ys[~zone])


def test_flatten_ends_keeps_a_column_that_follows_the_line():
    """Столбец в зоне конца, идущий по строке, не трогается."""
    xs, ys, boxes = _line_with_glyphs(first_width=20.0, last_width=8.0)
    out = seg._flatten_ends(xs, ys, boxes, height_px=30.0, k=2.0, x0=100.0)
    assert out == pytest.approx(ys)


def test_flatten_ends_skips_a_three_glyph_logo():
    """У строки из трёх крупных знаков опорой стала бы вторая буква — концы не прижимаем.

    Числа с логотипа рубрики «Нам пишут» (1975/05 с.97): строка x 112..209 рабочей копии при
    высоте 32, три глифа шириной 37, 23 и 34 px. После вычета обеих зон середины остаётся 14 px
    рабочей копии при опорном окне в 48 — опереться не на что.
    """
    xs = np.arange(194, dtype=float)
    ys = np.full(194, 20.0)
    ys[:20] = 8.0
    boxes = np.array([(112.0, 149.0, 37.0, 35.0), (151.0, 174.0, 23.0, 20.0), (174.0, 208.0, 34.0, 20.0)])
    out = seg._flatten_ends(xs, ys, boxes, height_px=64.0, k=2.0, x0=112.0)
    assert out == pytest.approx(ys)


def _marked_line(span: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Наклонная центр-линия, у которой столбцы внутри ``span`` держат ординату низкой метки."""
    xs = np.arange(120, dtype=float)
    ys = 20.0 + 0.1 * xs
    ys[(xs >= span[0]) & (xs <= span[1])] = 35.0  # центр масс точки — на базовой линии
    return xs, ys, np.full(120, 12.0)


def test_edge_mark_shortens_the_axis_to_the_middle_of_the_mark():
    """Запятая в конце строки: ось доводится до её середины и идёт по ходу строки.

    1973/08 с.85, сноска «…Энгельс Ф. Соч., т. 25,»: ``np.interp`` за концом опоры отдаёт
    концевое значение, и ось шла от запятой горизонтальной полкой до её дальнего края.
    """
    # Метка доходит до последнего столбца строки — за запятой краски уже нет.
    xs, ys, weights = _marked_line((112.0, 119.0))
    out_xs, out_ys, out_weights, inner = _edge_marks(xs, ys, weights, [(112.0, 119.0)], height_px=30.0)
    assert inner == []  # метка краевая, в интерполяцию по соседям она не уходит
    assert out_xs[-1] == pytest.approx(115.0)  # ось доведена до СЕРЕДИНЫ метки (115.5)
    assert out_weights.size == out_xs.size
    inside = out_xs >= 112.0
    assert out_ys[inside] == pytest.approx(20.0 + 0.1 * out_xs[inside], abs=0.5)  # ход строки


def test_mark_in_the_middle_of_a_line_is_not_an_edge_mark():
    """Точка между буквами ось не укорачивает — её обходит интерполяция по соседям."""
    xs, ys, weights = _marked_line((60.0, 64.0))
    out_xs, _, _, inner = _edge_marks(xs, ys, weights, [(60.0, 64.0)], height_px=30.0)
    assert inner == [(60.0, 64.0)]
    assert out_xs.size == xs.size


def test_mark_before_a_leader_run_is_not_an_edge_mark():
    """За меткой идёт отточие — строка на ней не кончается, ось не обрезаем.

    1971/10 с.93: первая точка отточия обрезала ось по своей середине, и продление по отточию
    уходило с уровня строки.
    """
    xs, ys, weights = _marked_line((112.0, 119.0))
    out_xs, _, _, inner = _edge_marks(xs, ys, weights, [(112.0, 119.0)], height_px=30.0, leaders=[(120.0, 300.0)])
    assert inner == [(112.0, 119.0)]
    assert out_xs.size == xs.size


def _style_row(height: float, stroke: float, glyph: tuple[float, float]) -> Row:
    """Ряд только с мерами набора: высота бокса, толщина штриха и медианный размер глифа."""
    return Row(y=0.0, height=height, x0=0.0, x1=100.0, axes=(), stroke=stroke, glyph_w=glyph[0], glyph_h=glyph[1])


def test_style_break_ignores_a_tall_box_when_the_glyphs_are_the_same():
    """Раздутая высота ряда без смены кегля границей блока не становится.

    1971/10 с.87 nogeo: три верхние строки корпуса дали боксы 26.5, 37.0 и 21.0 при 14–18 ниже
    (отношение 1.56) и штрих 2.50 против 2.00 (1.25 — штрих квантован шагом 0.25 px). Произведение
    1.95 при пороге 1.9, и колонка рвалась пополам, хотя глифы с обеих сторон одинаковы.
    """
    before = [_style_row(height, 2.50, (7.5, 10.0)) for height in (26.5, 37.0, 21.0)]
    after = [_style_row(height, 2.00, (7.5, 10.0)) for height in (17.0, 14.0, 17.0)]
    assert not _style_break(before + after, 2)


def test_style_break_still_finds_a_heading():
    """Настоящая смена кегля границей остаётся: глифы заголовка вдвое крупнее корпусных."""
    # 1973/06 с.65 geo: заголовок «ОСНОВЫ ЭКОНОМИКИ…» над подзаголовком «Тема 11…».
    before = [_style_row(32.0, 4.00, (14.4, 32.5)) for _ in range(2)]
    after = [_style_row(16.0, 2.40, (8.5, 17.2)) for _ in range(2)]
    assert _style_break(before + after, 1)


def test_style_break_still_finds_a_bold_heading_of_the_same_size():
    """Заголовок того же кегля, но жирный, границей остаётся: гасится только признак кегля."""
    before = [_style_row(17.0, 4.00, (7.5, 10.0)) for _ in range(4)]
    after = [_style_row(16.0, 2.00, (7.5, 10.0)) for _ in range(4)]
    assert _style_break(before + after, 3)


def test_block_contour_swallows_a_strongly_curved_row():
    """Контур блока охватывает краску строки, чей собственный изгиб больше межстрочного шага.

    Сноска 1973/08 с.85: первая строка — дуга 1382 → 1368 → 1403 (размах 35 px при шаге между
    рядами 20.5), вторая — короткая, ниже неё. Рамка «ордината → края» на таком блоке обязана
    обрушить правую кромку с x = 905 до x = 610 за несколько пикселей высоты и срезает хвост
    первой строки; контур это чинит, проглатывая краску рядов.
    """
    long_row = _row(557, 905, [1382, 1374, 1368, 1370, 1384, 1403], height=17.0)
    short_row = _row(520, 610, [1401, 1396], height=13.0)
    envelope = envelope_of([long_row, short_row], pitch=20.5, dpi=WORK_DPI, smooth_pitches=2.5, cap=CapKind.INK)
    hull = envelope.polygon.astype(np.float32).reshape(-1, 1, 2)
    for row in (long_row, short_row):
        for x, y in np.vstack([row.top_edge, row.bottom_edge]):
            assert cv2.pointPolygonTest(hull, (float(x), float(y)), True) >= -0.5


def test_body_cap_does_not_follow_ascenders():
    """Кромка-полоса идёт мимо выносных элементов, а кромка по краске — по ним.

    У ряда пять столбцов из двадцати задраны на высоту прописной; полоса вокруг оси обязана
    остаться у корпуса, а кромка по краске — подняться к ним.
    """
    xs = np.linspace(100.0, 400.0, 20)
    axis_y = np.full(xs.size, 200.0)
    tops = np.full(xs.size, 195.0)
    tops[::4] = 185.0  # выносные элементы вверх: четверть столбцов
    row = _row(100.0, 400.0, [200.0, 200.0], height=14.0)
    row = Row(
        **{
            **row.__dict__,
            "top_edge": np.column_stack([xs, tops]),
            "bottom_edge": np.column_stack([xs, np.full(xs.size, 205.0)]),
        }
    )
    body = blocks._cap_body(row, 100.0, 400.0, -1.0)
    ink = blocks._cap(row, 100.0, 400.0, -1.0)
    assert body[:, 1].max() - body[:, 1].min() < 0.5  # полоса ровная
    assert ink[:, 1].max() - ink[:, 1].min() > 5.0  # краска вихляет на выносных
    # Полоса стоит у корпуса (195), а не у выносных (185): квантиль 0.8 оставляет их снаружи.
    assert float(np.median(body[:, 1])) > float(np.median(axis_y)) - 8.0


def test_row_keeps_neighbouring_lines_apart():
    """Обрывки соседних строк (разнесены на межстрочный шаг) в один ряд не сливаются."""
    left = _fragment(520, 610, 1396, height=13.0)
    right = _fragment(827, 902, 1396 + 22, height=13.0)
    assert not _same_row([left], right)


def _glyph_ink(spans: list[tuple[float, float, float, float]], shape: tuple[int, int] = (1600, 1800)) -> np.ndarray:
    """Краска рендера 300 dpi: по каждому отрезку ``(x0, x1, y, глиф)`` рабочей копии — ряд квадратных «букв».

    Буквы — квадраты со стороной ``глиф`` px рабочей копии через пробел в полбуквы, по центру ``y``.
    """
    ink = np.zeros(shape, dtype=np.uint8)
    k = 2
    for x0, x1, y, glyph in spans:
        x = x0
        while x + glyph <= x1:
            top = int((y - glyph / 2.0) * k)
            ink[top : top + int(glyph * k), int(x * k) : int((x + glyph) * k)] = 255
            x += 1.5 * glyph
    return ink


def test_row_splits_a_small_signature_from_a_large_heading():
    """«доцент,» (глиф 9 px) и «в новых условиях» (18 px) на одной высоте через 32 мм — два ряда.

    1967/05, IMG_0059_1L: подпись автора слева и заголовок справа сливались в один ряд, после чего
    ни подпись, ни заголовок нельзя было отделить друг от друга.
    """
    small = _fragment(152, 206, 655, height=18.0)
    large = _fragment(398, 750, 658, height=31.5)
    assert _same_row([small], large)  # по одной ординате они сливаются
    ink = _glyph_ink([(152, 206, 655, 9.0), (398, 750, 658, 18.0)])
    pieces = blocks._split_by_glyph([small, large], ink, 2.0, WORK_DPI)
    assert pieces == [[small], [large]]


def test_row_keeps_a_wide_spaced_heading_of_one_size():
    """Заголовок одного кегля с широким пробелом (бокс слова вдвое выше) остаётся одним рядом.

    1970/10 IMG_0040_1L: «Механизация работ на базе» — высоты осей 25 и 45 px через 4.7 мм, а глиф
    у обеих 24 px.
    """
    first = _fragment(150, 400, 700, height=45.0)
    second = _fragment(430, 700, 700, height=25.0)
    ink = _glyph_ink([(150, 400, 700, 24.0), (430, 700, 700, 24.0)])
    group = [first, second]
    assert blocks._split_by_glyph(group, ink, 2.0, WORK_DPI) == [group]


def test_row_keeps_a_small_gap_between_sizes():
    """Разный глиф через обычный пробел (цифры рядом со строчными) — один ряд."""
    first = _fragment(150, 300, 700, height=14.0)
    second = _fragment(310, 500, 700, height=14.0)
    ink = _glyph_ink([(150, 300, 700, 9.0), (310, 500, 700, 14.0)])
    group = [first, second]
    assert blocks._split_by_glyph(group, ink, 2.0, WORK_DPI) == [group]


def _sized(row: Row, glyph_h: float) -> Row:
    """Тот же ряд с заданной медианной высотой глифа (мера кегля)."""
    return Row(**{**row.__dict__, "glyph_h": glyph_h})


def test_heading_and_signature_side_by_side_are_two_streams():
    """Заголовок справа и подпись слева перемежаются по y — два потока по три ряда; корпус ниже — третий.

    1967/05, IMG_0059_1L: глиф заголовка 23 px, подписи 10–14 px (у «доцент,» не мерится).
    """
    heading = [
        _sized(_row(355, 784, [567.0] * 4, 31.0), 23.0),
        _sized(_row(354, 635, [613.0] * 4, 31.0), 23.0),
        _sized(_row(363, 750, [658.0] * 4, 31.5), 23.0),
    ]
    signature = [
        _sized(_row(110, 248, [635.0] * 4, 16.0), 14.0),
        _row(151, 206, [655.0] * 4, 18.0),
        _sized(_row(62, 296, [673.0] * 4, 17.0), 10.0),
    ]
    body = [_sized(_row(60, 786, [752.0 + 22.0 * index] * 4, 16.0), 11.5) for index in range(3)]
    rows = sorted([*heading, *signature, *body], key=lambda row: row.y)
    streams = blocks._streams(rows, 22.0, WORK_DPI)
    assert [[row.y for row in stream] for stream in streams] == [
        [567.0, 613.0, 658.0],
        [635.0, 655.0, 673.0],
        [752.0, 774.0, 796.0],
    ]


def test_same_size_rows_side_by_side_stay_one_stream():
    """Полу-ряды таблицы одного кегля бок о бок (текст графы и число) потоков не заводят."""
    rows = [
        _sized(_row(187, 560, [1037.0] * 4, 14.0), 10.0),
        _sized(_row(609, 684, [1036.0] * 4, 10.0), 11.0),
        _sized(_row(188, 560, [1055.0] * 4, 10.0), 10.0),
        _sized(_row(600, 684, [1058.0] * 4, 10.0), 11.0),
    ]
    streams = blocks._streams(rows, 22.0, WORK_DPI)
    assert len(streams) == 1 and streams[0] is rows


def test_plain_column_with_a_paragraph_end_is_one_stream():
    """Колонка с концом абзаца в одно слово и красной строкой — один поток, тот же список рядов."""
    rows = [
        _row(60, 786, [100.0] * 4, 15.0),
        _row(60, 786, [122.0] * 4, 15.0),
        _row(60, 100, [144.0] * 4, 15.0),
        _row(110, 786, [166.0] * 4, 15.0),
        _row(60, 786, [188.0] * 4, 15.0),
    ]
    streams = blocks._streams(rows, 22.0, WORK_DPI)
    assert len(streams) == 1 and streams[0] is rows


def test_body_contour_holds_all_its_axes():
    """Контур-ПОЛОСА охватывает оси всех своих рядов, даже когда изгиб строки больше шага рядов.

    Та же сноска 1973/08 с.85, что и у кромки по краске: рамка «ордината → края» обязана обрушить
    правую кромку за несколько пикселей высоты и срезает конец первой строки («Соч., т. 25,» — на
    10.4 px снаружи). Полоса вокруг оси букв не обещает, но СВОИ ОСИ обязана держать внутри.
    """
    long_row = _row(557, 905, [1382, 1374, 1368, 1370, 1384, 1403], height=17.0)
    short_row = _row(520, 610, [1401, 1396], height=13.0)
    envelope = envelope_of([long_row, short_row], pitch=20.5, dpi=WORK_DPI, smooth_pitches=2.5, cap=CapKind.BODY)
    hull = envelope.polygon.astype(np.float32).reshape(-1, 1, 2)
    for row in (long_row, short_row):
        for axis in row.axes:
            for x, y in np.asarray(axis.points, dtype=np.float64):
                assert cv2.pointPolygonTest(hull, (float(x), float(y)), True) >= -1.0


def test_body_contour_covers_an_axis_longer_than_the_ink():
    """У блока из одного ряда контур-полоса охватывает ось и там, где краски уже нет.

    Край ряда ищется по заполненности столбца и тонкие элементы отбрасывает: у заголовка-вензеля
    1973/06 с.65 край ряда x = 220 при оси от x = 170, и ось торчала из контура на полсантиметра.
    """
    row = _row(220.0, 748.0, [213.0, 225.0, 240.0], height=121.0)
    long_axis = _fragment(170.0, 746.0, 226.0, height=121.0)
    row = Row(**{**row.__dict__, "axes": (long_axis,)})
    envelope = envelope_of([row], pitch=40.0, dpi=WORK_DPI, smooth_pitches=2.5, cap=CapKind.BODY)
    hull = envelope.polygon.astype(np.float32).reshape(-1, 1, 2)
    for x, y in np.asarray(long_axis.points, dtype=np.float64):
        assert cv2.pointPolygonTest(hull, (float(x), float(y)), True) >= -1.0


def test_body_contour_still_leaves_ascenders_outside():
    """Полоса и после подмешивания лент остаётся полосой: выносные элементы наружу.

    Иначе вернулось бы вихляние границы амплитудой в полвысоты строчной, ради которого полосу и
    заводили: контур-полоса обязан быть уже контура по краске.
    """
    xs = np.linspace(100.0, 400.0, 20)
    tops = np.full(xs.size, 195.0)
    tops[::4] = 175.0  # выносные элементы вверх: четверть столбцов
    row = _row(100.0, 400.0, [200.0, 200.0, 200.0], height=14.0)
    row = Row(
        **{
            **row.__dict__,
            "top_edge": np.column_stack([xs, tops]),
            "bottom_edge": np.column_stack([xs, np.full(xs.size, 205.0)]),
        }
    )
    other = _row(100.0, 400.0, [240.0, 240.0, 240.0], height=14.0)
    body = envelope_of([row, other], pitch=40.0, dpi=WORK_DPI, smooth_pitches=2.5, cap=CapKind.BODY)
    ink = envelope_of([row, other], pitch=40.0, dpi=WORK_DPI, smooth_pitches=2.5, cap=CapKind.INK)
    assert body.polygon[:, 1].min() > ink.polygon[:, 1].min() + 5.0


# Межколонник, колонки зоны и строка правой графы, заехавшая в него, — числа с 1971/10 с.93.
_GUTTER = Gutter(points=((500.0, 605.0, 726.0), (1500.0, 605.0, 726.0)))
_ZONE = Zone(y0=500, y1=1500, columns=((0, 618), (723, 1009)))


def _toc_axes() -> tuple[LineAxis, LineAxis, LineAxis]:
    """Строка оглавления с отточием слева, её продолжение в правой графе и обрывок между ними."""
    left = _fragment(128.0, 608.0, 1198.0, height=12.5)  # «Кабели дальней связи . . . . . .»
    right = _fragment(719.0, 863.0, 1190.0, height=12.5)  # «825, 1000 м в зави-»
    orphan = _fragment(641.0, 665.0, 1193.0, height=17.0)  # «425», целиком в межколоннике
    return left, right, orphan


def test_an_axis_inside_the_gutter_belongs_to_no_column():
    """Ось, целиком лежащая в межколоннике, не в колонке: ``bounds_at`` её попросту не ограничивал."""
    _, _, orphan = _toc_axes()
    assert inside_gutter([_GUTTER], orphan.x0, orphan.x1, orphan.cy)
    assert not blocks._inside(orphan, _ZONE.columns[0], [_GUTTER], 1009)
    assert not blocks._inside(orphan, _ZONE.columns[1], [_GUTTER], 1009)


def test_a_gutter_orphan_goes_to_the_nearest_text_not_to_the_nearest_leader():
    """Обрывок в межколоннике уходит в графу справа: отточия соседа слева текстом не считаются.

    По концам осей левая строка ближе (33 px против 54), но её хвост — отточие; до её ТЕКСТА
    311 px. Без отточия правило переворачивается — это и проверяется вторым случаем.
    """
    left, right, orphan = _toc_axes()
    leader = Leader(x0=330.0, x1=608.0, y=1198.0, thickness=2.0, dots=20)
    homes = blocks._column_homes([left, right, orphan], _ZONE, [_GUTTER], 1009, [leader])
    assert homes[blocks._axis_key(left)] == 0 and homes[blocks._axis_key(right)] == 1
    assert homes[blocks._axis_key(orphan)] == 1
    without = blocks._column_homes([left, right, orphan], _ZONE, [_GUTTER], 1009, [])
    assert without[blocks._axis_key(orphan)] == 0


def test_gutter_orphans_chain_to_each_other():
    """Сироты цепляются цепочкой: «425» идёт за «500,», а тот — за «825, 1000 м» в правой графе."""
    left, right, orphan = _toc_axes()
    second = _fragment(681.0, 705.0, 1191.0, height=17.0)  # «500,», тоже целиком в межколоннике
    leader = Leader(x0=330.0, x1=608.0, y=1198.0, thickness=2.0, dots=20)
    homes = blocks._column_homes([left, right, orphan, second], _ZONE, [_GUTTER], 1009, [leader])
    assert homes[blocks._axis_key(second)] == 1 and homes[blocks._axis_key(orphan)] == 1


# Хвост последней строки: предпоследняя строка выгнута аркой (как на 1976/09 с.92), последняя —
# на треть колонки ниже на шаг строк. Шаг 22.7 px и символ 8 px — замер той же полосы.
_TAIL_PITCH = 22.7
_TAIL_GLYPH = 8.0


def _arch(x0: float, x1: float, y: float, rise: float, count: int = 60) -> list[float]:
    """Ординаты арки: концы на ``y``, середина на ``rise`` выше."""
    t = np.linspace(-1.0, 1.0, count)
    return list(y - rise * (1.0 - t**2))


def _tail_pair() -> tuple[Row, Row]:
    """Предпоследний (полный, аркой) и последний (короткий, та же арка ниже на шаг) ряды."""
    reference = _row(112.0, 476.0, _arch(112.0, 476.0, 1416.0, 4.0), height=14.0)
    xs = np.linspace(112.0, 476.0, 60)
    ys = np.asarray(_arch(112.0, 476.0, 1416.0 + _TAIL_PITCH, 4.0))
    short = xs <= 254.0
    last = _row(112.0, float(xs[short][-1]), list(ys[short]), height=14.0)
    return reference, last


def test_tail_follows_the_previous_line_to_the_cut():
    """Хвост идёт по изгибу предпоследней строки на шаг ниже и кончается на линии отсечки."""
    reference, last = _tail_pair()
    tail, cut = blocks._tail_of(last, reference, _TAIL_PITCH, _TAIL_GLYPH, WORK_DPI)
    assert tail[0, 0] > last.x1 and tail[-1, 0] == pytest.approx(476.0, abs=2.0 + abs(cut[1, 0] - cut[0, 0]))
    ref_xs, ref_ys = blocks._smoothed_axis(blocks._row_axis(reference), WORK_DPI)
    # За зоной сращивания хвост — сглаженная опора, сдвинутая на шаг строк.
    steady = tail[:, 0] >= last.x1 + blocks.TAIL_BLEND_GLYPHS * _TAIL_GLYPH
    expected = np.interp(tail[steady, 0], ref_xs, ref_ys) + _TAIL_PITCH
    assert np.abs(tail[steady, 1] - expected).max() < 0.5
    # Ступеньки на стыке с настоящей осью нет.
    assert abs(tail[0, 1] - last.axes[0].points[-1, 1]) < 1.0


def test_tail_cut_is_perpendicular_to_a_sloping_line():
    """На наклонной опоре отсечка на высоте хвоста сдвинута на −наклон·интервал."""
    slope = 0.05
    xs = np.linspace(112.0, 476.0, 60)
    reference = _row(112.0, 476.0, list(1400.0 + slope * (xs - 112.0)), height=14.0)
    short = xs <= 254.0
    last = _row(112.0, 254.0, list(1400.0 + _TAIL_PITCH + slope * (xs[short] - 112.0)), height=14.0)
    tail, cut = blocks._tail_of(last, reference, _TAIL_PITCH, _TAIL_GLYPH, WORK_DPI)
    assert tail[-1, 0] == pytest.approx(476.0 - slope * _TAIL_PITCH, abs=1.0)
    direction = cut[1] - cut[0]
    # Отсечка перпендикулярна опоре с точностью до градуса (наклон меряется по сглаженной оси).
    angle = math.degrees(math.atan2(direction[0], direction[1])) + math.degrees(math.atan(slope))
    assert abs(angle) < 1.0


def test_no_tail_for_a_full_row_a_single_row_or_distant_rows():
    """Полная последняя строка, блок из одного ряда и несоседние ряды хвоста не получают."""
    reference, _ = _tail_pair()
    xs = np.linspace(112.0, 476.0, 60)
    full = _row(112.0, 474.0, list(1416.0 + _TAIL_PITCH + 0.0 * xs), height=14.0)
    assert blocks._tail_of(full, reference, _TAIL_PITCH, _TAIL_GLYPH, WORK_DPI) is None
    assert blocks._with_tail([reference], _TAIL_PITCH, _TAIL_GLYPH, WORK_DPI) == [reference]
    far = _row(112.0, 254.0, [1416.0 + 4.0 * _TAIL_PITCH] * 25, height=14.0)
    assert blocks._tail_of(far, reference, _TAIL_PITCH, _TAIL_GLYPH, WORK_DPI) is None


def test_envelope_encloses_the_tail():
    """Главная граница блока обводит хвост, а справочная по краске его не касается."""
    reference, last = _tail_pair()
    rows = blocks._with_tail(
        [_row(112.0, 476.0, _arch(112.0, 476.0, 1416.0 - k * _TAIL_PITCH, 4.0), 14.0) for k in (3, 2, 1)]
        + [reference, last],
        _TAIL_PITCH,
        _TAIL_GLYPH,
        WORK_DPI,
    )
    tail = rows[-1].tail
    assert tail is not None
    envelope = envelope_of(rows, _TAIL_PITCH, WORK_DPI, 2.5, (_TAIL_GLYPH, 9.5), cap=CapKind.BODY)
    hull = envelope.polygon.astype(np.float32).reshape(-1, 1, 2)
    assert all(cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0 for x, y in tail)
    # Низ блока правее конца строки идёт по хвосту (плюс отступ полосы), а не горизонталью.
    bottom = envelope.bottom
    at = np.interp(tail[:, 0], bottom[:, 0], bottom[:, 1]) - tail[:, 1]
    assert np.ptp(at) < 1.0


def test_no_tail_across_foreign_text():
    """Справа от короткой строки краска чужого текста — хвост не строится; пустая бумага — строится."""
    reference, last = _tail_pair()
    k = 2  # рендер RENDER_DPI вдвое крупнее рабочей копии WORK_DPI
    ink = np.zeros((1600 * k, 600 * k), dtype=bool)
    assert blocks._tail_of(last, reference, _TAIL_PITCH, _TAIL_GLYPH, WORK_DPI, ink) is not None
    # Чужая строка на высоте последней, от середины колонки до правого края.
    y = int((1416.0 + _TAIL_PITCH) * k)
    ink[y - 8 : y + 8, 300 * k : 470 * k] = True
    assert blocks._tail_of(last, reference, _TAIL_PITCH, _TAIL_GLYPH, WORK_DPI, ink) is None


def test_top_edge_follows_the_baseline_not_a_capital_word():
    """Слово «из прописных» в первой строке поднимает ось, а верх блока — нет.

    Слово синтетики — сплошной прямоугольник; «прописное» слово — тот же прямоугольник, выше на 40 %
    при той же базовой линии. Ось (середина краски) над ним поднимается, линия середины строчной и
    отмеренный от неё верх блока остаются на месте (1975/05 с.97, «В УМТС Башкирского»). Свойство
    прежней кромки-полосы (``BlocksMode.LEGACY``): гладкая граница берёт верх по постоянному отступу от
    сглаженной оси.
    """
    from tests.ocr_utils.page_layout.orientation.synthetic import GLYPH_H, INK

    plain = column_page(columns=1, justify="both")
    before = max(analyse_gray(plain, ENGINE, blocks_mode=BlocksMode.LEGACY).blocks, key=lambda block: len(block.rows))
    row = before.rows[0]
    y = int(round(row.y * 2))
    band = (plain[y - GLYPH_H : y + GLYPH_H] < 128).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(band, 8)
    words = sorted(
        (stats[i] for i in range(1, count) if stats[i, cv2.CC_STAT_HEIGHT] >= GLYPH_H // 2), key=lambda s: s[0]
    )
    capital = plain.copy()
    raise_px = int(0.4 * GLYPH_H)
    chosen = words[len(words) // 2 : len(words) // 2 + 2]  # два слова подряд посередине строки
    for left, top, width, _, _ in chosen:
        top += y - GLYPH_H
        capital[top - raise_px : top, left : left + width] = INK
    after = max(analyse_gray(capital, ENGINE, blocks_mode=BlocksMode.LEGACY).blocks, key=lambda block: len(block.rows))
    x0 = chosen[0][0] / 2.0 + 4
    x1 = (chosen[-1][0] + chosen[-1][2]) / 2.0 - 4
    xs = np.linspace(x0, x1, 12)
    axis_before, axis_after = blocks._row_axis(before.rows[0]), blocks._row_axis(after.rows[0])
    axis_shift = np.interp(xs, axis_after[:, 0], axis_after[:, 1]) - np.interp(xs, axis_before[:, 0], axis_before[:, 1])
    top_shift = np.interp(xs, after.envelope.top[:, 0], after.envelope.top[:, 1]) - np.interp(
        xs, before.envelope.top[:, 0], before.envelope.top[:, 1]
    )
    assert axis_shift.min() <= -1.0  # ось над «прописным» словом поднялась
    assert np.abs(top_shift).max() < 0.6  # верх блока остался на месте
