"""Пересчёт координат между кадром CVAT и оригиналом.

Числа взяты с реальной полосы пака-1: 3492x6051 при 600 dpi, делитель 8.
"""

import numpy as np
import pytest

from ocr_utils.scan_markup.geometry import (
    crop_size,
    cvat_size,
    divisor_for_dpi,
    mask_to_original,
    point_to_original,
    rect_to_original,
    to_cvat_rect,
)

W, H, D = 3492, 6051, 8
CROP_W, CROP_H = 3488, 6048
CVAT_W, CVAT_H = 436, 756


@pytest.mark.parametrize("dpi,expected", [(600, 8), (450, 6), (300, 4), (75, 1), (40, 1)])
def test_divisor_for_dpi(dpi: int, expected: int) -> None:
    """600 dpi даёт 8, 450 — 6; ниже 75 dpi делитель не опускается ниже единицы."""
    assert divisor_for_dpi(dpi) == expected


def test_crop_and_cvat_size() -> None:
    """Обрезка съедает не больше divisor-1 пикселя по каждой стороне."""
    assert crop_size(W, H, D) == (CROP_W, CROP_H)
    assert cvat_size(W, H, D) == (CVAT_W, CVAT_H)
    assert W - CROP_W < D and H - CROP_H < D


def test_rect_touching_right_edge_expands() -> None:
    """Прямоугольник, доведённый до края кадра, растягивается на всю ширину оригинала.

    Разметчик не видел полоски, обрезанной при уменьшении, но объект в неё продолжается —
    иначе у фотографии, свёрстанной в обрез, останется незакрытая кромка.
    """
    assert rect_to_original(100, 10, CVAT_W, CVAT_H, D, W, H) == (800, 80, W, H)


def test_rect_inside_frame_is_plain_scaling() -> None:
    """Прямоугольник вдали от края просто умножается на делитель, без всякой добавки."""
    assert rect_to_original(100, 10, 400, 700, D, W, H) == (800, 80, 3200, 5600)


def test_round_trip_through_cvat_coordinates() -> None:
    """Оригинал -> CVAT -> оригинал: промах не больше делителя (это цена уменьшения)."""
    x1, y1, x2, y2 = 800, 1600, 3000, 5000
    back = rect_to_original(*to_cvat_rect(x1, y1, x2, y2, D, CVAT_W, CVAT_H), D, W, H)
    assert all(abs(a - b) <= D for a, b in zip(back, (x1, y1, x2, y2)))


def test_to_cvat_rect_clamps_to_frame() -> None:
    """Область, найденная в обрезанной полоске, не должна вылезать за кадр CVAT.

    Иначе CVAT отвергает шейп целиком, и предразметка теряется молча.
    """
    assert to_cvat_rect(0, 0, W, H, D, CVAT_W, CVAT_H) == (0.0, 0.0, float(CVAT_W), float(CVAT_H))


def test_mask_upscale_is_exact_nearest() -> None:
    """Каждый пиксель разметки становится квадратом divisor x divisor."""
    mask = np.zeros((CVAT_H, CVAT_W), bool)
    mask[10:20, 30:40] = True
    full = mask_to_original(mask, D, W, H)
    assert full.shape == (H, W)
    assert full[80:160, 240:320].all()
    assert full.sum() == 10 * D * 10 * D


def test_mask_touching_edges_fills_cropped_strip() -> None:
    """Маска у правого-нижнего угла достаёт до самого края оригинала, включая угол."""
    mask = np.zeros((CVAT_H, CVAT_W), bool)
    mask[CVAT_H - 6 :, CVAT_W - 6 :] = True
    full = mask_to_original(mask, D, W, H)
    assert full[H - 1, W - 1]
    assert full[:, W - 1].sum() == 6 * D + (H - CROP_H)
    assert full[H - 1, :].sum() == 6 * D + (W - CROP_W)


def test_mask_inside_frame_leaves_strip_empty() -> None:
    """Маска вдали от края обрезанную полоску не заполняет."""
    mask = np.zeros((CVAT_H, CVAT_W), bool)
    mask[10:20, 30:40] = True
    full = mask_to_original(mask, D, W, H)
    assert not full[:, CROP_W:].any()
    assert not full[CROP_H:, :].any()


def test_mask_of_wrong_size_is_rejected() -> None:
    """Маска не от того кадра — ошибка, а не молча съехавшая разметка."""
    with pytest.raises(ValueError):
        mask_to_original(np.zeros((100, 100), bool), D, W, H)


def test_point_is_plain_scaling() -> None:
    """Точка просто умножается на делитель."""
    assert point_to_original(120, 300, D, W, H) == (120 * D, 300 * D)


def test_point_at_frame_edge_stays_inside_original() -> None:
    """Точка у края кадра НЕ дотягивается до края оригинала, но и за него не уходит.

    Рамку и маску мы распространяем в обрезанную полоску: у них есть граница, и упёршаяся
    в край кадра граница означает «объект продолжается». У точки границы нет — она сама
    себе положение, и сдвигать её на 4 пикселя вправо значило бы указать не то место.
    """
    x, y = point_to_original(CVAT_W - 1, CVAT_H - 1, D, W, H)
    assert (x, y) == ((CVAT_W - 1) * D, (CVAT_H - 1) * D)
    assert x < CROP_W and y < CROP_H


def test_point_outside_frame_is_clamped() -> None:
    """Координата за кадром обрезается по оригиналу, а не даёт индекс за массивом."""
    assert point_to_original(10**6, 10**6, D, W, H) == (W - 1, H - 1)
    assert point_to_original(-5, -5, D, W, H) == (0, 0)
