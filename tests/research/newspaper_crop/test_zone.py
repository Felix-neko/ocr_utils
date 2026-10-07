"""Тесты хвоста стенда зоны копирования: четырёхугольник по маске листа, выбор маски SAM, дорисовка, выборка."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from research.newspaper_crop.engines import pick_sheet_mask
from research.newspaper_crop.fill import CopyMode, copy_rectangle, fill_non_paper
from research.newspaper_crop.frames import even_pick
from research.newspaper_crop.zone import SideFit, build_zone, fit_enclosing_quad, keystone_deg, quad_mask

# Синтетический «лист»: трапеция в кадре 600×400 (H×W).
TRAPEZOID = np.array([[60, 50], [340, 70], [350, 540], [40, 560]], np.float64)


def _mask(poly: np.ndarray, shape: tuple[int, int] = (600, 400)) -> np.ndarray:
    """Булева маска многоугольника.

    Args:
        poly: Вершины (N, 2).
        shape: (H, W).

    Returns:
        Маска.
    """
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m.astype(bool)


def _match(quad: np.ndarray, expected: np.ndarray) -> float:
    """Наибольшее расстояние от вершины ожидаемого четырёхугольника до ближайшей найденной.

    Args:
        quad: Найденные вершины.
        expected: Ожидаемые вершины.

    Returns:
        Расстояние, px.
    """
    return float(max(np.linalg.norm(quad - e, axis=1).min() for e in expected))


@pytest.mark.parametrize("side_fit", list(SideFit))
def test_quad_recovers_trapezoid(side_fit: SideFit) -> None:
    """На чистой трапеции оба способа находят её вершины с точностью до пары пикселей."""
    quad = fit_enclosing_quad(_mask(TRAPEZOID), side_fit=side_fit)
    assert quad is not None
    assert _match(quad, TRAPEZOID) < 3.0


def test_notch_does_not_pull_side_inward() -> None:
    """Выемка (палец, надрыв) лежит внутри: сторона остаётся на краю листа, выемка — в дорисовке."""
    mask = _mask(TRAPEZOID)
    mask[500:600, 150:220] = False
    zone = build_zone(mask)
    assert zone.quad is not None
    assert _match(zone.quad, TRAPEZOID) < 4.0
    assert zone.fill_frac > 0.01
    assert zone.outside_frac < 0.002


def test_bulged_edge_stays_inside() -> None:
    """Выгнутый край листа (горб у корешка) не срезается: доля листа за четырёхугольником ничтожна."""
    xs = np.linspace(40, 350, 60)
    bottom = np.stack([xs, 550 + 30 * np.sin(np.linspace(0, np.pi, 60))], axis=1)
    poly = np.vstack([[[60, 50], [340, 70]], bottom[::-1]])
    zone = build_zone(_mask(poly))
    assert zone.quad is not None
    assert zone.outside_frac < 0.003


def test_frame_border_side_uses_frame_edge() -> None:
    """Лист, уходящий за рамку справа: правая сторона — рамка кадра, а не выдуманная прямая."""
    mask = np.zeros((600, 400), bool)
    mask[50:560, 40:] = True
    quad = fit_enclosing_quad(mask)
    assert quad is not None
    assert np.abs(quad[:, 0].max() - 399) < 2.0


def test_keystone_zero_for_rectangle() -> None:
    """У прямоугольника мера трапеции нулевая, у трапеции — положительная."""
    rect = np.array([[0, 0], [10, 0], [10, 20], [0, 20]], float)
    assert keystone_deg(rect) == pytest.approx(0.0, abs=1e-3)
    assert keystone_deg(TRAPEZOID) > 1.0


def test_pick_sheet_mask_prefers_largest_confident() -> None:
    """Из кандидатов SAM берётся самый крупный с оценкой не ниже порога, а не самый уверенный."""
    masks = np.zeros((3, 10, 10), bool)
    masks[0, :2, :2] = True
    masks[1, :, :] = True
    masks[2, :5, :5] = True
    assert pick_sheet_mask(masks, np.array([0.9, 0.6, 0.8])).sum() == 100
    assert pick_sheet_mask(masks, np.array([0.9, 0.3, 0.8])).sum() == 25


def test_fill_keeps_paper_pixels() -> None:
    """Дорисовка не трогает пиксели листа и закрашивает не-бумагу цветом, близким к бумаге."""
    img = np.full((200, 160, 3), 200, np.uint8)
    paper = np.ones((200, 160), bool)
    paper[:, :30] = False
    img[:, :30] = (0, 0, 255)
    out = fill_non_paper(img, paper)
    assert np.array_equal(out[paper], img[paper])
    assert np.abs(out[~paper].astype(int) - 200).mean() < 20


def test_copy_rectangle_warp_size() -> None:
    """Выпрямление даёт прямоугольник по длинам сторон четырёхугольника."""
    img = np.full((600, 400, 3), 180, np.uint8)
    paper = _mask(TRAPEZOID)
    out = copy_rectangle(img, paper, TRAPEZOID, CopyMode.WARP)
    assert abs(out.shape[1] - 310) < 3 and abs(out.shape[0] - 510) < 3
    inplace = copy_rectangle(img, paper, TRAPEZOID, CopyMode.IN_PLACE)
    assert inplace.shape[:2] == (510, 310)


def test_quad_mask_area() -> None:
    """Маска четырёхугольника по площади совпадает с маской исходного многоугольника."""
    assert abs(int(quad_mask(TRAPEZOID, (600, 400)).sum()) - int(_mask(TRAPEZOID).sum())) < 5


def test_even_pick_keeps_ends() -> None:
    """Равномерная выборка берёт первый и последний элементы и нужное число."""
    paths = [Path(f"{i}.RAF") for i in range(100)]
    pick = even_pick(paths, 5)
    assert len(pick) == 5 and pick[0] == paths[0] and pick[-1] == paths[-1]
    assert even_pick(paths[:3], 5) == paths[:3]


def test_mask_touching_frame_falls_back_to_box() -> None:
    """Маска во весь кадр с выемкой у рамки (как у scan_cropping): четырёхугольник не вырождается в диагональ."""
    mask = np.ones((600, 400), bool)
    mask[:40, 300:] = False
    mask[560:, :80] = False
    zone = build_zone(mask)
    assert zone.quad is not None
    assert zone.keystone_deg < 25.0
    assert zone.outside_frac < 0.01
