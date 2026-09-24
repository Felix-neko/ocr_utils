"""Разбор текста с кривыми строками: оси строк, колонки, огибающие блоков и выключка."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from ocr_utils.curved_layout import WORK_DPI
from ocr_utils.curved_layout.alignment import AlignKind
from ocr_utils.curved_layout.blocks import (
    EXTEND_SLOPE_LIMIT_DEG,
    Row,
    _axis_slope,
    _same_row,
    _style_break,
    envelope_of,
)
from ocr_utils.curved_layout import segment as seg
from ocr_utils.curved_layout.columns import gutters_of, zones_of
from ocr_utils.curved_layout.lines import LineAxis
from ocr_utils.curved_layout.leaders import leaders_of
from ocr_utils.curved_layout.engines.ink import InkEngine
from ocr_utils.curved_layout.page import analyse_gray
from ocr_utils.page_layout import px_to_mm
from tests.ocr_utils.curved_layout.synthetic import (
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
    """Дилатация делает границу ровнее: разброс правой кромки по y падает."""
    analysis = analyse_gray(column_page(columns=1), ENGINE)
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


def test_flatten_ends_clamps_the_corner_of_an_italic_letter():
    """Столбцы-углы на конце строки не утаскивают ось вверх.

    У курсива нет вертикальных штрихов: справа дальше всего выступает ВЕРХНИЙ-правый угол
    последней буквы, и центр масс в этих столбцах уходит вверх (1971/10 с.87, «комиссии ВАК»:
    ось поднималась на 2.4 px рабочей копии при высоте строчной 10).
    """
    xs = np.arange(120, dtype=float)
    ys = np.full(120, 20.0)
    weights = np.full(120, 12.0)
    ys[-4:] = (14.0, 10.0, 8.0, 7.0)  # угол буквы: центр масс уехал вверх
    weights[-4:] = 4.0  # и краски в этих столбцах втрое меньше обычного
    out = seg._flatten_ends(xs, ys, weights, height_px=30.0)
    assert out[:-4] == pytest.approx(ys[:-4])
    assert out[-4:] == pytest.approx(20.0 - seg.END_FLAT_TOLERANCE_HEIGHTS * 30.0)


def test_flatten_ends_keeps_a_full_column():
    """Полный столбец на конце строки не трогается: его центр масс верен."""
    xs = np.arange(120, dtype=float)
    ys = np.full(120, 20.0)
    weights = np.full(120, 12.0)
    ys[-4:] = (14.0, 10.0, 8.0, 7.0)  # ординаты те же, но краски в столбцах обычное количество
    out = seg._flatten_ends(xs, ys, weights, height_px=30.0)
    assert out == pytest.approx(ys)


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
    envelope = envelope_of([long_row, short_row], pitch=20.5, dpi=WORK_DPI, smooth_pitches=2.5)
    hull = envelope.polygon.astype(np.float32).reshape(-1, 1, 2)
    for row in (long_row, short_row):
        for x, y in np.vstack([row.top_edge, row.bottom_edge]):
            assert cv2.pointPolygonTest(hull, (float(x), float(y)), True) >= -0.5


def test_row_keeps_neighbouring_lines_apart():
    """Обрывки соседних строк (разнесены на межстрочный шаг) в один ряд не сливаются."""
    left = _fragment(520, 610, 1396, height=13.0)
    right = _fragment(827, 902, 1396 + 22, height=13.0)
    assert not _same_row([left], right)
