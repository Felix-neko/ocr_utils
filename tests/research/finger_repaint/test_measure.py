"""Тесты меры тёмного пятна на месте пальца (синтетика): серое пятно в зоне ловится, буквы и чистая зона — нет."""

import cv2
import numpy as np

from research.finger_repaint.measure import spot_score

PX_PER_MM = 4.0
PAPER = (215, 230, 238)


def page() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Страница 600×600 целиком бумага, зона пальца — круг у правого края."""
    img = np.full((600, 600, 3), PAPER, np.uint8)
    paint = np.zeros((600, 600), np.uint8)
    cv2.circle(paint, (560, 300), 120, 1, -1)
    return img, paint > 0, np.ones((600, 600), bool)


def test_grey_spot_found() -> None:
    """Размытое серое пятно в зоне пальца — пятно, площадь порядка самого пятна."""
    img, paint, visible = page()
    spot = np.zeros((600, 600), np.float32)
    cv2.circle(spot, (540, 300), 60, 1.0, -1)
    spot = cv2.GaussianBlur(spot, (0, 0), 10)
    img = np.clip(img.astype(np.float32) - 45 * spot[..., None], 0, 255).astype(np.uint8)
    score = spot_score(img, paint, visible, PX_PER_MM)
    assert score.area_mm2 > 300
    assert score.mean_drop > 8


def test_letters_and_clean_zone_not_spot() -> None:
    """Чистая зона и буквы (тонкие тёмные штрихи) в зоне пальца — не пятно."""
    img, paint, visible = page()
    assert spot_score(img, paint, visible, PX_PER_MM).area_mm2 == 0
    for y in range(200, 420, 24):
        cv2.putText(img, "TEXT TEXT", (440, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1)
    assert spot_score(img, paint, visible, PX_PER_MM).area_mm2 < 5
