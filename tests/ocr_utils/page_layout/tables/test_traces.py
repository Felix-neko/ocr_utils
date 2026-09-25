"""Трассы линеек: сплайн по изогнутой линейке против истинной кривой синтетики, концы, JSON, помехи."""

from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw

from ocr_utils.page_layout.tables.traces import Axis, RuleTrace, trace_rules
from tests.ocr_utils.page_layout.tables import synthetic

DPI = synthetic.DPI


def _nearest(centres: list[float], trace: RuleTrace, truth) -> float:
    """Середина истинной линейки, ближайшей к трассе (по медиане расхождения)."""
    along = trace.along_grid(4.0)
    return min(centres, key=lambda centre: abs(float(np.median(trace.across_at(along) - truth(centre, along)))))


def test_spline_follows_s_bend_within_a_pixel() -> None:
    """S-изгиб с наклоном до 4,6°: положение — в пределах 2 px (0,17 мм, в среднем 0,35 px), наклон — 1,5° вдали от концов.

    Порог — компромисс шума и смещения: полоса сглаживания выбирается по остатку не выше 0,5 px, и на
    крутом изгибе синтетики (кривизна 4·10⁻⁴ 1/px) грубая полоса смещает кривую на доли пикселя.
    """
    table = synthetic.make_curved_table()
    traces = trace_rules(table.image, DPI)
    assert len(traces.horizontal) == len(table.row_centres)
    assert len(traces.vertical) == len(table.col_centres)
    ends = int(3 * DPI / 25.4)
    for trace in traces.horizontal:
        along = trace.along_grid(1.0)
        centre = _nearest(table.row_centres, trace, table.horizontal_truth)
        error = np.abs(trace.across_at(along) - table.horizontal_truth(centre, along))
        assert error.max() < 2.0 and error.mean() < 0.35
        slope_error = np.abs(trace.angle_deg_at(along) - np.degrees(np.arctan(table.bend_slope(along))))
        assert slope_error[ends:-ends].max() < 1.5
        assert slope_error.max() < 2.0
    for trace in traces.vertical:
        along = trace.along_grid(1.0)
        centre = _nearest(table.col_centres, trace, table.vertical_truth)
        assert np.abs(trace.across_at(along) - table.vertical_truth(centre, along)).max() < 2.0


def test_chord_deviation_matches_truth() -> None:
    """Отклонение от отрезка «начало–конец» — как у истинной кривой, до пикселя."""
    table = synthetic.make_curved_table(amplitude_px=8.0)
    traces = trace_rules(table.image, DPI)
    for trace in traces.horizontal:
        along = trace.along_grid(1.0)
        centre = _nearest(table.row_centres, trace, table.horizontal_truth)
        truth = table.horizontal_truth(centre, along)
        chord = truth[0] + (truth[-1] - truth[0]) * (along - along[0]) / (along[-1] - along[0])
        expected = float(np.abs(truth - chord).max())
        measured, _where = trace.max_chord_deviation_px()
        assert abs(measured - expected) < 1.5
        assert measured > 4.0  # изгиб настоящий: отрезок ошибся бы больше чем на 4 px


def test_outside_the_rule_is_nan_unless_extended() -> None:
    """За концами кривая не определена; продолжение — только явное и только на заданную длину."""
    table = synthetic.make_curved_table()
    trace = trace_rules(table.image, DPI).horizontal[0]
    beyond = trace.end_out + 5.0
    assert np.isnan(trace.across_at(beyond)[0])
    assert np.isfinite(trace.across_at(beyond, extend_px=10.0)[0])
    assert np.isnan(trace.across_at(beyond, extend_px=2.0)[0])


def test_json_round_trip_keeps_the_curve() -> None:
    """Компактный сплайн JSON воспроизводит кривую до десятой пикселя, ломаная — через 1 мм."""
    table = synthetic.make_curved_table()
    trace = trace_rules(table.image, DPI).horizontal[1].mapped(2.0, 100.0, 50.0)
    payload = json.loads(json.dumps(trace.to_json()))
    restored = RuleTrace.from_json(payload)
    along = trace.along_grid(1.0)
    assert np.abs(restored.across_at(along) - trace.across_at(along)).max() < 0.1
    step = np.diff(np.array(payload["polyline"])[:, 0])
    assert abs(float(np.median(step)) - DPI * 2 / 25.4) < 1.0
    assert restored.axis == Axis.HORIZONTAL


def test_mapping_moves_the_curve_exactly() -> None:
    """Перенос в другую картинку — масштаб и сдвиг, без перестройки сплайна."""
    table = synthetic.make_curved_table()
    trace = trace_rules(table.image, DPI).horizontal[2]
    moved = trace.mapped(2.0, 30.0, -7.0)
    along = trace.along_grid(8.0)
    assert np.allclose(moved.across_at(along * 2.0 + 30.0), trace.across_at(along) * 2.0 - 7.0)
    assert np.allclose(moved.angle_deg_at(along * 2.0 + 30.0), trace.angle_deg_at(along))


def _straight_rule(extra=None) -> np.ndarray:
    """Горизонтальная линейка 4 px на y = 100..103 длиной 600 px и, по желанию, помеха ``extra(draw)``."""
    canvas = Image.new("L", (700, 200), synthetic.PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([50, 100, 649, 103], fill=synthetic.INK)
    if extra is not None:
        extra(draw)
    return np.asarray(canvas)


def _letter_on_rule(draw: ImageDraw.ImageDraw) -> None:
    """Буква «Т», стоящая на линейке, и перекладина, касающаяся её снизу."""
    draw.rectangle([300, 70, 330, 73], fill=synthetic.INK)
    draw.rectangle([313, 70, 317, 99], fill=synthetic.INK)
    draw.rectangle([420, 104, 432, 110], fill=synthetic.INK)


def test_letter_touching_the_rule_does_not_bend_it() -> None:
    """Буква на линейке и касание снизу сдвигают кривую меньше чем на полпикселя."""
    clean = trace_rules(_straight_rule(), DPI).horizontal
    noisy = trace_rules(_straight_rule(_letter_on_rule), DPI).horizontal
    assert len(clean) == len(noisy) == 1
    along = clean[0].along_grid(1.0)
    assert np.abs(noisy[0].across_at(along) - 101.5).max() < 0.5


def _dashed(draw: ImageDraw.ImageDraw) -> None:
    """Пунктирная линейка: штрихи 5 мм через просветы 5 мм.

    Просветы уже 3 мм разрывами не станут вовсе: их закрывает ещё открытие-закрытие фрагментов
    (ядро 3 мм), и трасса идёт по ним сплошной. Сшивка же цепочки берёт просвет до 8 мм, если оба
    штриха не короче 5 мм.
    """
    for x in range(50, 650, 120):
        draw.rectangle([x, 150, x + 59, 153], fill=synthetic.INK)


def _leader(draw: ImageDraw.ImageDraw) -> None:
    """Отточие «……»: точки 3 px через 12 px."""
    for x in range(50, 650, 12):
        draw.rectangle([x, 150, x + 2, 152], fill=synthetic.INK)


def test_dashed_rule_is_one_trace_with_gaps_and_leader_is_no_rule() -> None:
    """Пунктир — одна трасса с разрывами; отточие линейкой не становится."""
    dashed = [
        t for t in trace_rules(_straight_rule(_dashed), DPI).horizontal if abs(t.across_at(t.start_out)[0] - 151.5) < 3
    ]
    assert len(dashed) == 1
    assert len(dashed[0].gaps) == 4
    leader = [
        t for t in trace_rules(_straight_rule(_leader), DPI).horizontal if abs(t.across_at(t.start_out)[0] - 151) < 3
    ]
    assert leader == []


def _hooked_rules() -> np.ndarray:
    """Горизонтали, за последней вертикалью уходящие вниз под 13° (загиб у края полосы, 1976/08).

    До вертикали x = 560 линейки ровные и обрываются за 14 px до неё; за ней, через просвет в 6 px,
    идёт кусок 90 px с наклоном 13° — фрагмент 3 мм при штрихе 3 px такого наклона не переживает.
    """
    canvas = np.full((420, 720), synthetic.PAPER, np.uint8)
    import cv2

    cv2.rectangle(canvas, (60, 40), (62, 380), synthetic.INK, -1)
    cv2.rectangle(canvas, (560, 40), (562, 380), synthetic.INK, -1)
    for y in (80, 180, 280):
        cv2.line(canvas, (62, y), (546, y), synthetic.INK, 3)
        rise = int(round(90 * np.tan(np.radians(13))))
        cv2.line(canvas, (569, y), (659, y + rise), synthetic.INK, 3)
    return canvas


def test_steep_hook_beyond_a_crossing_is_followed() -> None:
    """Трасса продолжается за вертикаль в крутой загиб и доходит до его конца с наклоном около 13°."""
    traces = trace_rules(_hooked_rules(), DPI, 4.0)
    long = [t for t in traces.horizontal if t.length_mm > 40]
    assert len(long) == 3
    for trace in long:
        assert trace.end_out >= 650
        assert abs(float(trace.angle_deg_at(trace.end_out - 20)[0]) - 13.0) < 3.0
        assert abs(float(trace.angle_deg_at(300.0)[0])) < 1.0
