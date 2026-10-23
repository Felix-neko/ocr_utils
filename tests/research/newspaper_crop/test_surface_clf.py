"""Тесты участков классификатора «свой лист / чужое»: стороны трапеции, участок вдоль стороны, глубина эталонной прямой."""

from __future__ import annotations

import numpy as np

from research.newspaper_crop.surface_clf import CROP_ALONG, CROP_DEPTH, color_features, crop_at, gt_depth, side_frames

QUAD = np.array([[100.0, 50.0], [500.0, 50.0], [500.0, 650.0], [100.0, 650.0]])


def test_side_frames_point_inward():
    """Нормаль каждой стороны смотрит к центру трапеции."""
    frames = side_frames(QUAD)
    centre = QUAD.mean(axis=0)
    for frame in frames.values():
        assert (centre - frame.p) @ frame.inward > 0
    assert abs(frames["top"].length - 400) < 1e-6 and abs(frames["left"].length - 600) < 1e-6


def test_crop_at_reads_inside_band():
    """Участок у верхней стороны на глубине 30 px берёт полосу картинки, лежащую на 30 px ниже стороны."""
    img = np.zeros((700, 600, 3), np.uint8)
    img[72:88, :] = 200  # полоса на глубине 22–38 px от верха (y=50)
    frame = side_frames(QUAD)["top"]
    crop = crop_at(img, frame, 200.0, 30.0)
    assert crop.shape == (CROP_DEPTH, CROP_ALONG, 3)
    rows = crop[:, :, 0].mean(axis=1)
    assert rows[CROP_DEPTH // 2] == 200


def test_gt_depth_of_tilted_line():
    """Эталон от глубины 0 до 40 вдоль верхней стороны: глубина посередине — 20."""
    frame = side_frames(QUAD)["top"]
    p = frame.p
    a = p
    b = p + frame.along * frame.length + frame.inward * 40
    depth = gt_depth(frame, a[0], a[1], b[0], b[1], np.array([0.0, frame.length / 2, frame.length]))
    assert np.allclose(depth, [0, 20, 40], atol=1e-6)


def test_color_features_shape():
    """Девять цветовых признаков на участок."""
    crops = np.random.default_rng(0).integers(0, 255, (3, CROP_DEPTH, CROP_ALONG, 3), dtype=np.uint8)
    assert color_features(crops).shape == (3, 9)
