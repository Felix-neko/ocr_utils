"""Тесты защиты печати и наклона по содержимому: охват четырёхугольником, зазор без дорисовки, точки схода по строкам."""

from __future__ import annotations

import cv2
import numpy as np

from research.newspaper_crop.content_tilt import (
    ContentModel,
    Family,
    Segment,
    SegmentKind,
    border_share,
    fit_vanishing_point,
    local_direction,
)
from research.newspaper_crop.fill import CopyMode, copy_rectangle
from research.newspaper_crop.text_guard import expand_quad, guard_mask, side_index
from research.newspaper_crop.zone import quad_mask

SQUARE = np.array([[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]])


def test_expand_quad_covers_points_and_keeps_frozen_side():
    """Точка за правой стороной раздвигает правую сторону; замороженная сторона не двигается."""
    pts = np.array([[95.0, 50.0]])
    quad, shift = expand_quad(SQUARE, pts)
    assert abs(shift - 5.0) < 1e-6
    assert quad_mask(quad, (120, 120))[50, 94]
    right = side_index(SQUARE, "right")
    frozen, shift = expand_quad(SQUARE, pts, (right,))
    assert shift == 0.0
    assert np.allclose(frozen, SQUARE)


def test_side_index_names():
    """Имена сторон квадрата по положению середин."""
    assert side_index(SQUARE, "top") == 0
    assert side_index(SQUARE, "right") == 1
    assert side_index(SQUARE, "bottom") == 2
    assert side_index(SQUARE, "left") == 3


def test_guard_keeps_text_near_edge():
    """Печать у края листа: зазор вне листа — ровный цвет бумаги, Telea не тянет туда краску букв."""
    h, w = 200, 200
    img = np.full((h, w, 3), 230, np.uint8)
    img[:, 150:] = 90  # не бумага (тень, чужой лист)
    paper = np.zeros((h, w), bool)
    paper[:, :150] = True
    own = np.zeros((h, w), bool)
    own[90:100, 130:146] = True  # строка у самого края листа
    img[own] = 20
    keep = guard_mask(own, 8, 0) & ~paper
    quad = np.array([[0.0, 0.0], [199.0, 0.0], [199.0, 199.0], [0.0, 199.0]])
    out = copy_rectangle(img, paper, quad, CopyMode.IN_PLACE, keep)
    assert np.abs(out[95, 151:154].astype(int) - 230).max() <= 2  # у букв за краем — цвет бумаги
    plain = copy_rectangle(img, paper, quad, CopyMode.IN_PLACE)
    assert np.abs(plain[95, 151:154].astype(int) - 230).max() > 10  # без защиты заливка темнее (тянет краску и тень)


def _segment(p0, p1, family=Family.ALONG):
    """Сегмент по двум точкам.

    Args:
        p0: Начало.
        p1: Конец.
        family: Семейство.

    Returns:
        :class:`Segment`.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    return Segment((p0 + p1) / 2, d / length, length, family, SegmentKind.TEXT_LINE)


def test_vanishing_point_of_converging_lines():
    """Строки, сходящиеся в точку (5000, 300), дают эту точку схода; выброс не мешает."""
    vp_true = np.array([5000.0, 300.0])
    segs = []
    for y in range(100, 900, 50):
        p0 = np.array([100.0, float(y)])
        d = (vp_true - p0) / np.linalg.norm(vp_true - p0)
        segs.append(_segment(p0, p0 + 300 * d))
    segs.append(_segment([100, 500], [400, 560]))  # косой выброс
    vp, inl = fit_vanishing_point(segs, np.random.default_rng(0))
    assert vp is not None
    assert np.allclose(vp[:2] / vp[2], vp_true, atol=30)
    assert inl.sum() == len(segs) - 1


def test_local_direction_follows_nearby_edges():
    """Боковая сторона вертикальна, края колонок у неё наклонены на 2° — локальная оценка даёт 2°."""
    tilt = np.radians(2.0)
    d = np.array([np.sin(tilt), np.cos(tilt)])
    segs = [
        Segment(np.array([80.0 + k, 400.0]), d, 400.0, Family.ACROSS, SegmentKind.COLUMN_EDGE) for k in (0, 30, 60)
    ]
    model = ContentModel(segs, np.array([1.0, 0.0]))
    quad = np.array([[0.0, 0.0], [600.0, 0.0], [600.0, 800.0], [0.0, 800.0]])
    left = 3
    direction, spread, count = local_direction(model, quad, left)
    assert count == 3 and spread < 1e-6
    angle = np.degrees(np.arccos(abs(direction @ d)))
    assert angle < 0.05


def test_border_share():
    """Сторона по рамке кадра — доля 1, внутри — 0."""
    quad = np.array([[0.0, 10.0], [50.0, 10.0], [50.0, 90.0], [0.0, 90.0]])
    assert border_share(quad, 3, (100, 100)) == 1.0
    assert border_share(quad, 1, (100, 100)) == 0.0


def test_copy_rectangle_without_keep_unchanged():
    """Без зазора выход прежний (обратная совместимость)."""
    img = np.full((50, 50, 3), 200, np.uint8)
    paper = np.ones((50, 50), bool)
    quad = np.array([[0.0, 0.0], [49.0, 0.0], [49.0, 49.0], [0.0, 49.0]])
    out = copy_rectangle(img, paper, quad, CopyMode.IN_PLACE)
    assert out.shape[:2] == (49, 49) or out.shape[:2] == (50, 50)
    assert cv2.absdiff(out, img[: out.shape[0], : out.shape[1]]).max() == 0


def test_finger_protect_keeps_tilted_line_and_drops_blob():
    """Наклонная строка (5°: прямоугольник по осям толще LINE_MAX_THICKNESS_PX) под защитой целиком; крупное тёмное пятно (тень, мрамор) — нет."""
    from research.newspaper_crop.text_guard import finger_protect

    img = np.full((800, 1600, 3), 225, np.uint8)
    # Строка из 40 «букв» 14×20 px через 12 px с подъёмом 5° (в половинном масштабе — глифы 7×10, зазор 6 < RLSA).
    for i in range(40):
        x = 100 + i * 26
        y = int(400 - i * 26 * np.tan(np.radians(5.0)))
        cv2.rectangle(img, (x, y), (x + 13, y + 19), (30, 30, 30), 3)
    cv2.circle(img, (800, 650), 70, (40, 40, 40), -1)
    guard = finger_protect(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    first, last = (107, 410), (int(100 + 39 * 26 + 7), int(400 - 39 * 26 * np.tan(np.radians(5.0)) + 10))
    assert guard[first[1], first[0]] and guard[last[1], last[0]]
    assert not guard[650, 800]
