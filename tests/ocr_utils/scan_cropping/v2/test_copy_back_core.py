"""Тест copy back с припуском ядра пальца (синтетика): кайма кожи 3 px за ядром не возвращается с оригинала, буквы блока за припуском — возвращаются (МТС 1990/04 IMG_1197)."""

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.copy_back import copy_back_blocks

H, W = 400, 400
PAPER = (215, 230, 238)
SKIN = (150, 160, 215)


def test_core_grow_drops_rim_keeps_letters() -> None:
    """Без припуска кайма кожи возвращается, с припуском 4 px — нет; буквы блока Table — на месте."""
    original = np.full((H, W, 3), PAPER, np.uint8)
    core = np.zeros((H, W), np.uint8)
    cv2.circle(core, (200, 200), 60, 1, -1)
    rim = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    original[rim] = SKIN
    for y in range(110, 300, 20):
        cv2.putText(original, "TEXT", (275, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
    paint = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (121, 121))) > 0
    clean = np.full((H, W, 3), PAPER, np.uint8)
    block = [np.array([[100, 90], [380, 90], [380, 320], [100, 320]], np.float32)]
    book = np.ones((H, W), bool)
    skin_rim = rim & (core == 0)
    before, _, _ = copy_back_blocks(original, clean, block, paint, book, (18, 36), ["Table"], core > 0)
    after, _, _ = copy_back_blocks(original, clean, block, paint, book, (18, 36), ["Table"], core > 0, 4)
    assert (before[skin_rim] == SKIN).all(axis=1).mean() > 0.9
    assert (after[skin_rim] == SKIN).all(axis=1).mean() < 0.05
    letters = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)[90:320, 270:380] < 100
    letters0 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)[90:320, 270:380] < 100
    assert letters.sum() >= 0.95 * letters0.sum() > 0
