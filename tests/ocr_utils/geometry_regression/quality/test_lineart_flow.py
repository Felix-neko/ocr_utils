"""Выравнивание и рамки мер line art по плотному полю (``ocr_utils.geometry_regression.quality.lineart_flow``)."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.geometry_regression.field import Field
from ocr_utils.geometry_regression.quality.lineart_flow import homography, merge_boxes


def test_overlapping_boxes_merge_and_distant_stay():
    """Рамки одной схемы от v16 и v6 сливаются в одну; далёкая рамка остаётся отдельной."""
    boxes = [(100, 100, 400, 300), (110, 90, 420, 310), (600, 600, 700, 700)]
    merged = merge_boxes(boxes)
    assert len(merged) == 2
    assert (100.0, 90.0, 420.0, 310.0) in merged


def test_homography_recovers_keystone():
    """Трапеция (проективное искажение) по тайлам поля восстанавливается, остатки тайлов — около нуля."""
    truth = np.array([[1.0, 0.02, 5.0], [0.0, 1.0, -3.0], [2e-5, 0.0, 1.0]])
    xs, ys = np.meshgrid(np.arange(50, 1200, 100.0), np.arange(50, 1700, 100.0))
    centres = np.column_stack([xs.ravel(), ys.ravel()])
    moved = cv2.perspectiveTransform(centres.reshape(-1, 1, 2), truth).reshape(-1, 2)
    tiles = np.column_stack([centres, moved - centres, np.ones(len(centres))])
    field = Field(1240, 1754, 150.0, tiles, np.eye(2, 3), np.zeros((len(centres), 2)), np.ones(len(centres)), np.zeros((0, 3)))
    matrix, resid = homography(field)
    assert matrix is not None
    assert np.abs(resid).max() < 0.05
