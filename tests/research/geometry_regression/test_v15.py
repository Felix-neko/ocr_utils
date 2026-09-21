"""Стенд v15 на синтетике: перекос блока, довернутый заголовок, черта без пары LSD, разрыв линии, два снимка в рамке."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.geometry_regression.field import estimate_field
from ocr_utils.geometry_regression.lines import match_lines
from ocr_utils.geometry_regression.regions import text_lines
from ocr_utils.geometry_regression.render import RENDER_DPI, to_work
from ocr_utils.geometry_regression.strokes import Stroke, find_strokes
from research.geometry_regression.v15.blocks import block_edges, edge_metrics
from research.geometry_regression.v15.field_shear import shear_metrics
from research.geometry_regression.v15.lines import glyph_line_metrics, tilt_summary
from research.geometry_regression.v15.raster import raster_edge_metrics
from research.geometry_regression.v15.ridge import bend_metrics, trace_strokes
from research.geometry_regression.v15.scoring import Thresholds15
from tests.ocr_utils.geometry_regression.synthetic import add_rules, rotate, text_page
from tests.ocr_utils.geometry_regression.test_raster import PHOTO, _halftone, _page_with_photo

DPI = 150


def shear(page: np.ndarray, shear_deg: float) -> np.ndarray:
    """Сдвиг: x уезжает пропорционально y (блок становится параллелограммом, строки остаются горизонтальными)."""
    h, w = page.shape
    k = np.tan(np.radians(shear_deg))
    matrix = np.array([[1.0, k, -k * h / 2.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    return cv2.warpAffine(page, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


def _edges(before: np.ndarray, after: np.ndarray):
    b, a = to_work(before), to_work(after)
    warp = estimate_field(b, a, DPI, [])
    lines_b, _ = text_lines(before, DPI)
    lines_a, _ = text_lines(after, DPI)
    pairs = match_lines(lines_b, lines_a, warp, DPI)
    return warp, lines_b, block_edges(lines_b, pairs, DPI)


def test_sheared_block_is_caught_and_rotation_is_not():
    page = text_page(lines=30)
    warp, lines_b, edges = _edges(page, shear(page, 1.0))
    metrics, culprits = edge_metrics(edges, DPI)
    assert metrics["edges_matched"] >= 1
    # Колонка ~130 мм под 1° — уход кромки ~2.3 мм.
    assert metrics["edge_shear_delta_mm"] > 1.5
    assert "segments_a" in culprits["edge_shear_delta_mm"]
    field, _ = shear_metrics(warp, [line.box for line in lines_b], [], DPI)
    assert field["field_shear_p90_deg"] > 0.7

    warp, lines_b, edges = _edges(page, rotate(page, 1.0))
    metrics, _ = edge_metrics(edges, DPI)
    assert abs(metrics["edge_shear_delta_mm"]) < 0.5
    field, _ = shear_metrics(warp, [line.box for line in lines_b], [], DPI)
    assert field["field_shear_p90_deg"] < 0.3


def test_heading_tilt_by_projection():
    """Крупный заголовок довернут на 1° при неизменном корпусе: порча ≈ длина·sin 1°, выигрыша нет."""
    page = text_page(lines=25).copy()
    y0, y1, x0, x1 = 150, 260, 200, 1500
    cv2.putText(page, "ZAGOLOVOK GAZETY", (x0, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 0, 8, cv2.LINE_AA)
    after = page.copy()
    crop = page[y0:y1, x0:x1]
    matrix = cv2.getRotationMatrix2D(((x1 - x0) / 2.0, (y1 - y0) / 2.0), -1.0, 1.0)
    after[y0:y1, x0:x1] = cv2.warpAffine(crop, matrix, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=255)
    b, a = to_work(page), to_work(after)
    warp = estimate_field(b, a, DPI, [])
    lines_b, _ = text_lines(page, DPI)
    lines_a, _ = text_lines(after, DPI)
    pairs = match_lines(lines_b, lines_a, warp, DPI)
    _, _, _, tilts = glyph_line_metrics(page, after, pairs, lines_b, warp, DPI, 25.0)
    heading = [t for t in tilts if t.heading]
    assert heading, "заголовок должен быть отдельной строкой"
    metrics, _ = tilt_summary(tilts)
    assert 1.2 < metrics["line_tilt_dev_max_mm"] < 2.5
    assert metrics["line_tilt_gain_deg"] < 0.3


def test_short_bar_tilt_by_ink_without_lsd_pair():
    """Черта 5 мм над текстом повёрнута на 3°: след краски даёт наклон черты в A без LSD-пары."""
    page = text_page(lines=25).copy()
    y, x0, x1 = 150, 1500, 1500 + int(5 * RENDER_DPI / 25.4)
    before = add_rules(page, [(x0, y, x1, y)], thickness=4)
    # Числитель и знаменатель — иначе короткая черта считается тире и отсеивается.
    for yy in (y - 24, y + 24):
        cv2.putText(before, "12", (x0 + 4, yy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA)
    # Поворот только области формулы на 3° (страница на месте — поле почти тождественное).
    after = before.copy()
    ry0, ry1, rx0, rx1 = y - 60, y + 60, x0 - 40, x1 + 40
    matrix = cv2.getRotationMatrix2D(((rx1 - rx0) / 2.0, (ry1 - ry0) / 2.0), -3.0, 1.0)
    after[ry0:ry1, rx0:rx1] = cv2.warpAffine(
        before[ry0:ry1, rx0:rx1], matrix, (rx1 - rx0, ry1 - ry0), flags=cv2.INTER_LINEAR, borderValue=255
    )
    strokes_b = [s for s in find_strokes(before, 4.0, RENDER_DPI, [], DPI) if s.length < 100]
    assert strokes_b, "черта в B должна найтись"
    warp = estimate_field(to_work(before), to_work(after), DPI, [])
    assert warp is not None
    tilts = trace_strokes(before, after, strokes_b, warp, RENDER_DPI, DPI)
    assert tilts and tilts[0].source == "ridge"
    assert 2.0 < abs(tilts[0].angle_a - tilts[0].angle_b) < 4.0


def test_broken_line_with_neighbour_is_not_bent():
    """Вертикаль с разрывом и параллельная соседка в 4 мм: след не перескакивает, изгиб ≈ 0."""
    page = np.full((1800, 900), 255, np.uint8)
    x, gap = 450, int(4 * RENDER_DPI / 25.4)
    before = add_rules(page, [(x, 200, x, 1600)], thickness=4)
    after = add_rules(page, [(x, 200, x, 800), (x, 860, x, 1600), (x - gap, 700, x - gap, 1000)], thickness=4)
    strokes_b = find_strokes(before, 4.0, RENDER_DPI, [], DPI)
    long = [s for s in strokes_b if s.length > 1000]
    assert long
    metrics, _ = bend_metrics(
        before, after, long, estimate_field(to_work(before), to_work(after), DPI, []), RENDER_DPI, DPI
    )
    assert metrics["stroke_bend_dev_mm"] < 0.3


def test_two_photos_in_one_box_one_rotated():
    """Два снимка в одной рамке растра: поворот одного ловится по его собственным кромкам."""
    second = (PHOTO[0], PHOTO[3] + 120, PHOTO[2], PHOTO[3] + 500)
    before = _page_with_photo()
    before[second[1] - 30 : second[3] + 30, second[0] - 30 : second[2] + 30] = 255
    before = _halftone(before, second, seed=1)
    after = before.copy()
    y0, y1, x0, x1 = second[1] - 40, second[3] + 40, second[0] - 40, second[2] + 40
    crop = before[y0:y1, x0:x1]
    matrix = cv2.getRotationMatrix2D(((x1 - x0) / 2.0, (y1 - y0) / 2.0), -2.0, 1.0)
    after[y0:y1, x0:x1] = cv2.warpAffine(crop, matrix, (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST, borderValue=255)
    union = (PHOTO[0], PHOTO[1], PHOTO[2], second[3])
    metrics, culprits = raster_edge_metrics(before, after, [union], None, DPI)
    assert metrics["raster_photos"] == 2
    assert metrics["raster_photo_tilt_mm"] > 1.5
    assert culprits["raster_photo_tilt_mm"]["b"][1] >= second[1] - 20


def test_scoring_lone_stroke_is_soft_and_uniform_shift_is_soft():
    thr = Thresholds15()
    lone = {"hstroke_dev_max_delta_mm": 1.2, "hstroke_pairs": 1.0, "text_sag_gain_mm": 0.5}
    assert thr.apply(lone).verdict == "mixed"
    lone_no_gain = {"hstroke_dev_max_delta_mm": 1.2, "hstroke_pairs": 1.0}
    assert thr.apply(lone_no_gain).verdict == "bad"
    uniform = {
        "vstroke_dev_max_delta_mm": 1.5,
        "vstroke_pairs": 5.0,
        "vstroke_uniform": 0.9,
        "vstroke_tilt_wmean_delta": 0.5,
        "text_spread_gain_deg": 1.0,
    }
    assert thr.apply(uniform).verdict == "mixed"
    ragged = {**uniform, "vstroke_uniform": 0.3}
    assert thr.apply(ragged).verdict == "bad"
    shear_alone = {"field_shear_p90_deg": 0.8, "edges_matched": 2.0, "edge_shear_delta_mm": 0.3}
    assert thr.apply(shear_alone).verdict == "ok"
    shear_confirmed = {**shear_alone, "edge_shear_delta_mm": 1.2}
    assert thr.apply(shear_confirmed).verdict == "bad"
