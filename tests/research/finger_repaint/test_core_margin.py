"""Тесты стенда «кайма кожи в copy back» (синтетика): припуск ядра не возвращает кайму, буквы блока за припуском возвращаются."""

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.scale import Scale
from research.finger_repaint.core_margin import MarginInputs, copy_back_with_margin, grow_core, rim_redness

H, W = 400, 400
PAPER = (215, 230, 238)
SKIN = (150, 160, 215)


def frame() -> MarginInputs:
    """Кадр: бумага, ядро пальца-круг, кайма кожи 3 px вокруг ядра, буквы в блоке Table рядом; закрас — ровная бумага."""
    original = np.full((H, W, 3), PAPER, np.uint8)
    core = np.zeros((H, W), np.uint8)
    cv2.circle(core, (200, 200), 60, 1, -1)
    rim = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    original[rim] = SKIN  # кайма кожи 3 px за ядром и само ядро
    for y in range(110, 300, 20):
        cv2.putText(original, "TEXT", (275, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
    paint = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (121, 121))) > 0
    clean = np.full((H, W, 3), PAPER, np.uint8)
    block = np.array([[100, 90], [380, 90], [380, 320], [100, 320]], np.float32)
    return MarginInputs("synthetic", original, clean, paint, core > 0, np.ones((H, W), bool), [block], ["Table"])


def test_grow_core() -> None:
    """Раздутие ядра на 4 px: радиус круга растёт на 4."""
    core = np.zeros((100, 100), bool)
    core[40:60, 40:60] = True
    grown = grow_core(core, 4)
    assert grown[36:64, 40:60].all() and not grown[34, 50]
    assert (grow_core(core, 0) == core).all()


def test_margin_drops_rim_keeps_letters() -> None:
    """Без припуска кайма кожи возвращается с оригинала, с припуском 4 px — нет; буквы блока за припуском — на месте."""
    inputs = frame()
    scale = Scale()
    before, returned0 = copy_back_with_margin(inputs, 0, scale)
    after, returned4 = copy_back_with_margin(inputs, 4, scale)
    assert rim_redness(before, inputs.core, inputs.paint, returned0) > 10
    assert rim_redness(after, inputs.core, inputs.paint, returned0) < 2
    letters = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)[90:320, 270:380] < 100
    letters0 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)[90:320, 270:380] < 100
    assert letters.sum() >= 0.95 * letters0.sum() > 0
