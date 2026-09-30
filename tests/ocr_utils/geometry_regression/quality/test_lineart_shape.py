"""Форма рисунка сверх подобия, наклон по прямым и перенос оси строки полем (``ocr_utils.geometry_regression.quality.lineart_flow``, v18)."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.geometry_regression.quality.lineart_flow import lsd_angle, shape_measures, transfer_axis

# Сетка точек краски рисунка 60×40 мм при 150 dpi: 360×240 px.
XS, YS = np.meshgrid(np.linspace(0.0, 360.0, 37), np.linspace(0.0, 240.0, 25))
GRID = np.column_stack([XS.ravel(), YS.ravel()])


def _rotate_scale(points: np.ndarray, angle_deg: float, scale: float) -> np.ndarray:
    """Подобие: поворот вокруг центра сетки и масштаб."""
    theta = np.radians(angle_deg)
    rotation = scale * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    centre = points.mean(axis=0)
    return (points - centre) @ rotation.T + centre + np.array([4.0, -3.0])


def test_rotation_and_scale_are_not_shape_damage():
    """Поворот на 2° и масштаб 1.03 рисунка целиком — подобие: все меры формы около нуля."""
    out = shape_measures(GRID, _rotate_scale(GRID, 2.0, 1.03))
    assert out["shape_mm"] < 0.05
    assert out["shape_shear_deg"] < 0.05
    assert out["shape_aniso"] < 1e-3


def test_shear_and_stretch_are_nonsimilar():
    """Перекос осей на 1° и вытяжка по вертикали на 3 % (1972/10 с.79) — уход аффинной модели от подобия, без изгиба."""
    shear = np.tan(np.radians(1.0))
    moved = GRID @ np.array([[1.0, shear], [0.0, 1.03]]).T
    out = shape_measures(GRID, moved)
    assert abs(out["shape_shear_deg"] - 1.0) < 0.1
    assert abs(out["shape_aniso"] - np.log(1.03)) < 0.005
    assert out["nonsim_mm"] > 0.5
    assert out["curve_mm"] < 0.05


def test_bent_bottom_is_curve():
    """Низ рисунка прогнут дугой (кривые стали кривее): квадратичная модель уходит от аффинной.

    Прогиб нижней кромки 6 px (1 мм) у краёв; лучшая аффинная модель забирает линейную часть, и уход на контуре —
    около трети прогиба.
    """
    moved = GRID.copy()
    moved[:, 1] += 6.0 * ((GRID[:, 0] - 180.0) / 180.0) ** 2 * (GRID[:, 1] / 240.0)
    out = shape_measures(GRID, moved)
    assert 0.25 < out["curve_mm"] < 0.5
    assert out["shape_mm"] >= out["curve_mm"] - 1e-6


def test_outliers_are_trimmed():
    """Промахи потока на штриховке (5 % точек, сдвиг на 20 px) не делают из подобия порчу формы."""
    moved = _rotate_scale(GRID, 1.0, 1.0)
    rng = np.random.default_rng(1)
    bad = rng.choice(len(GRID), len(GRID) // 20, replace=False)
    moved[bad] += 20.0
    assert shape_measures(GRID, moved)["shape_mm"] < 0.3


def _grid_picture(angle_deg: float) -> np.ndarray:
    """Картинка «рисунка»: прямоугольная решётка, повёрнутая на угол (белый фон, чёрные линии)."""
    image = np.full((400, 500), 255, np.uint8)
    for x in range(60, 460, 40):
        cv2.line(image, (x, 40), (x, 360), 0, 2)
    for y in range(40, 380, 40):
        cv2.line(image, (60, y), (440, y), 0, 2)
    matrix = cv2.getRotationMatrix2D((250.0, 200.0), -angle_deg, 1.0)
    return cv2.warpAffine(image, matrix, (500, 400), borderValue=255)


def test_lsd_angle_reads_rotation_of_straight_lines():
    """Наклон по прямым рисунка: повёрнутая на 1.5° решётка даёт 1.5° (знак — как ``atan2(dy, dx)`` в пикселях)."""
    assert abs(lsd_angle(_grid_picture(1.5)) - 1.5) < 0.3
    assert abs(lsd_angle(_grid_picture(0.0))) < 0.3


def test_lsd_angle_without_lines_is_none():
    """Без околоосевых прямых наклон не определяется."""
    assert lsd_angle(np.full((200, 200), 255, np.uint8)) is None


def test_transfer_axis_follows_tilted_line():
    """Строка, наклонённая в A на 2° при тождественной странице: перенесённая ось повторяет наклон."""
    gray_b = np.full((200, 600), 255, np.uint8)
    for x in range(60, 540, 14):
        cv2.rectangle(gray_b, (x, 90), (x + 9, 110), 0, -1)
    matrix = cv2.getRotationMatrix2D((300.0, 100.0), -2.0, 1.0)
    gray_a = cv2.warpAffine(gray_b, matrix, (600, 200), borderValue=255)
    points = np.column_stack([np.arange(70.0, 530.0, 10.0), np.full(46, 100.0)])
    moved = transfer_axis(gray_b, gray_a, np.eye(3), points, 20.0)
    assert moved is not None
    in_a, ncc, confidence = moved
    assert ncc > 0.3
    assert np.median(confidence) == 1.0
    slope = np.polyfit(in_a[:, 0], in_a[:, 1], 1)[0]
    assert abs(np.degrees(np.arctan(slope)) - 2.0) < 0.5


def test_turned_half_is_part_turn():
    """Правая половина рисунка довёрнута на 2° вокруг своего центра, левая на месте: разность поворотов частей — 2°."""
    moved = GRID.copy()
    right = GRID[:, 0] > 180.0
    moved[right] = _rotate_scale(GRID[right], 2.0, 1.0) - np.array([4.0, -3.0])
    out = shape_measures(GRID, moved)
    assert abs(out["part_turn_deg"] - 2.0) < 0.3
    assert out["part_turn_mm"] > 0.5
    assert shape_measures(GRID, _rotate_scale(GRID, 2.0, 1.0))["part_turn_deg"] < 0.05


def test_gap_ink_tells_text_line_from_photo():
    """Строка текста — белый просвет над или под ней; полоса полутона фото — краска с обеих сторон."""
    from ocr_utils.geometry_regression.quality.lines import _gap_ink

    points = np.column_stack([np.arange(50.0, 350.0, 5.0), np.full(60, 100.0)])
    text = np.full((200, 400), 255, np.uint8)
    for x in range(50, 350, 12):
        cv2.rectangle(text, (x, 90), (x + 8, 110), 0, -1)
    photo = np.where(np.indices((200, 400)).sum(axis=0) % 2 == 0, 0, 255).astype(np.uint8)
    assert _gap_ink(text, points, 20.0) < 0.06
    assert _gap_ink(photo, points, 20.0) > 0.3
