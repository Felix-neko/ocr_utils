"""Стенд v16 на синтетике: кромка по краске с отступами и разрезанной строкой, порча line art, дробные черты, форма заголовка."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_utils.geometry_regression.field import estimate_field
from ocr_utils.geometry_regression.lines import match_lines
from ocr_utils.geometry_regression.regions import text_lines
from ocr_utils.geometry_regression.render import RENDER_DPI, to_work
from ocr_utils.geometry_regression.strokes import find_strokes
from research.geometry_regression.v15.blocks import block_edges, text_rows
from research.geometry_regression.v15.lineart import lineart_metrics
from research.geometry_regression.v15.lines import glyph_line_metrics
from research.geometry_regression.v15.ridge import fraction_metrics, lsd_fallback, trace_strokes
from research.geometry_regression.v15.scoring import Thresholds15
from tests.ocr_utils.geometry_regression.synthetic import FONT_PATH, WORDS, add_rules, rotate, text_page
from tests.research.geometry_regression.test_v15 import shear

DPI = 150


def _indented_page(width: int = 2000, height: int = 3000, lines: int = 36, font_px: int = 40) -> np.ndarray:
    """Страница 300 dpi: абзацы с красной строкой, одна строка с широким пробелом (разрез сегментации), одиночная буква «и»."""
    rng = np.random.default_rng(3)
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, font_px)
    y, x0 = 300, 200
    for i in range(lines):
        words = " ".join(rng.choice(WORDS, size=int(rng.integers(6, 9))))[:64]
        if i % 7 == 0:
            draw.text((x0 + 120, y), words, fill=0, font=font)  # красная строка
        elif i % 7 == 3:
            draw.text((x0, y), "и", fill=0, font=font)  # одиночная буква, дальше широкий пробел
            draw.text((x0 + 220, y), words[:50], fill=0, font=font)
        else:
            draw.text((x0, y), words, fill=0, font=font)
        y += int(font_px * 1.6)
    return np.asarray(image)


def test_ink_edge_ignores_indents_and_split_lines():
    page = _indented_page()
    warp = estimate_field(to_work(page), to_work(page), DPI, [])
    lines_b, separators = text_lines(page, DPI)
    rows = text_rows([line for line in lines_b if line.column >= 0])
    edges = block_edges(lines_b, separators, to_work(page).shape[1], page, page, warp, DPI)
    left = [e for e in edges if e.side == "left"]
    assert left, "левая кромка должна найтись"
    xs = left[0].points_b[:, 0]
    # Все точки кромки — на видимом крае колонки (x0 = 200 px рендера = 100 на копии), отступов нет.
    assert np.all(np.abs(xs - 100) <= 3), xs
    assert left[0].lines >= 20
    # Разрезанная строка («и» + широкий пробел) вошла в кромку одним рядом.
    assert len(rows) <= 36


def test_sheared_page_edge_shift_by_ink():
    page = _indented_page()
    after = shear(page, 1.0)
    warp = estimate_field(to_work(page), to_work(after), DPI, [])
    lines_b, separators = text_lines(page, DPI)
    edges = block_edges(lines_b, separators, to_work(page).shape[1], page, after, warp, DPI)
    left = [e for e in edges if e.side == "left"]
    assert left and left[0].shear_delta_mm > 1.5


def _drawing(width: int = 1800, height: int = 1800) -> np.ndarray:
    """Чертёж: прямоугольник 100×80 мм с осями и подписями."""
    page = np.full((height, width), 255, np.uint8)
    x0, y0, x1, y1 = 300, 300, 300 + int(100 * RENDER_DPI / 25.4), 300 + int(80 * RENDER_DPI / 25.4)
    page = add_rules(
        page,
        [
            (x0, y0, x1, y0),
            (x0, y1, x1, y1),
            (x0, y0, x0, y1),
            (x1, y0, x1, y1),
            (x0, (y0 + y1) // 2, x1, (y0 + y1) // 2),
        ],
        thickness=4,
    )
    return page, (x0 // 2 - 10, y0 // 2 - 10, x1 // 2 + 10, y1 // 2 + 10)


def _lineart(before: np.ndarray, after: np.ndarray, box) -> dict:
    warp = estimate_field(to_work(before), to_work(after), DPI, [box])
    sb = find_strokes(before, 4.0, RENDER_DPI, [box], DPI)
    sa = find_strokes(after, 4.0, RENDER_DPI, [box], DPI, drop_lone=False)
    rot = warp.rot_deg if warp is not None else 0.0
    metrics, _ = lineart_metrics(before, after, sb, sa, [box], warp, rot, RENDER_DPI, DPI)
    return metrics


def test_lineart_shear_changes_axis_angle_but_rotation_does_not():
    before, box = _drawing()
    sheared = shear(before, 1.5)
    metrics = _lineart(before, sheared, box)
    assert metrics["lineart_strokes"] >= 4
    assert 1.0 < metrics["lineart_axis_delta_deg"] < 2.2
    rotated = rotate(before, 1.5)
    metrics = _lineart(before, rotated, box)
    assert metrics["lineart_axis_delta_deg"] < 0.4
    assert metrics["lineart_rot_max_deg"] < 0.6


def test_lineart_bent_line():
    before, box = _drawing()
    h, w = before.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    y_mid = (300 + 300 + int(80 * RENDER_DPI / 25.4)) // 2
    inside = (ys > y_mid - 40) & (ys < y_mid + 40)
    ys_src = ys + inside * 12.0 * np.sin(np.pi * (xs - 300) / (100 * RENDER_DPI / 25.4))
    after = cv2.remap(before, xs, ys_src.astype(np.float32), cv2.INTER_LINEAR, borderValue=255)
    metrics = _lineart(before, after, box)
    assert metrics["lineart_bend_mm"] > 0.6


def test_fraction_bars_tilt():
    """Две дробные черты по 12 мм с числителем и знаменателем, повёрнутые на 2.5°, при ровном тексте вокруг."""
    page = text_page(lines=20).copy()
    bars = []
    for x0 in (1300, 1550):
        y = 150
        x1 = x0 + int(12 * RENDER_DPI / 25.4)
        cv2.line(page, (x0, y), (x1, y), 0, 4)
        for yy in (y - 26, y + 26):
            cv2.putText(page, "123", (x0 + 6, yy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA)
        bars.append((x0, y, x1))
    after = page.copy()
    ry0, ry1, rx0, rx1 = 60, 240, 1250, 1750
    matrix = cv2.getRotationMatrix2D(((rx1 - rx0) / 2.0, (ry1 - ry0) / 2.0), -2.5, 1.0)
    after[ry0:ry1, rx0:rx1] = cv2.warpAffine(
        page[ry0:ry1, rx0:rx1], matrix, (rx1 - rx0, ry1 - ry0), flags=cv2.INTER_LINEAR, borderValue=255
    )
    warp = estimate_field(to_work(page), to_work(after), DPI, [])
    sb = find_strokes(page, 4.0, RENDER_DPI, [], DPI)
    sa = find_strokes(after, 4.0, RENDER_DPI, [], DPI, drop_lone=False)
    from ocr_utils.geometry_regression.strokes import match_strokes

    tilts = lsd_fallback(
        trace_strokes(page, after, sb, warp, RENDER_DPI, DPI), match_strokes(sb, sa, warp, RENDER_DPI, DPI)
    )
    metrics, culprits = fraction_metrics(tilts, page < 128, RENDER_DPI, [], DPI / RENDER_DPI)
    assert metrics["fraction_bars"] >= 2
    assert 1.5 < metrics["fraction_tilt_mean_delta_deg"] < 3.5
    assert len(culprits["fraction_tilt_mean_delta_deg"]["segments_b"]) >= 2


def test_heading_step_and_bend_ratios():
    """Заголовок: правая треть поднята на 1 мм (ступенька) — растёт ступенька к высоте и кривизна к длине."""
    page = text_page(lines=25).copy()
    cv2.putText(page, "ZAGOLOVOK STATI GAZETY", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 2.6, 0, 7, cv2.LINE_AA)
    after = page.copy()
    shift = int(1.0 * RENDER_DPI / 25.4)
    after[150 - shift : 280 - shift, 900:1400] = page[150:280, 900:1400]
    after[280 - shift : 280, 900:1400] = 255
    warp = estimate_field(to_work(page), to_work(after), DPI, [])
    lines_b, _ = text_lines(page, DPI)
    lines_a, _ = text_lines(after, DPI)
    pairs = match_lines(lines_b, lines_a, warp, DPI)
    metrics, _, _, _ = glyph_line_metrics(page, after, pairs, lines_b, warp, DPI, 25.0)
    assert metrics["line_step_ratio"] > 0.04
    assert metrics["line_bend_ratio"] > 3.0e-3


def test_scoring_v16_rules():
    thr = Thresholds15()
    lineart = {"lineart_axis_delta_deg": 1.2, "text_sag_gain_mm": 2.0, "text_spread_gain_deg": 3.0}
    assert thr.apply(lineart).verdict == "bad"  # непрощаемая
    fraction = {"fraction_bars": 2.0, "fraction_tilt_mean_delta_deg": 1.0, "text_sag_gain_mm": 2.0}
    assert thr.apply(fraction).verdict == "mixed"  # обычная порча с гистерезисом
    one_bar = {"fraction_bars": 1.0, "fraction_tilt_mean_delta_deg": 3.0}
    assert thr.apply(one_bar).verdict == "ok"
    heading = {"line_bend_ratio": 4.0e-3}
    assert thr.apply(heading).verdict == "bad"
