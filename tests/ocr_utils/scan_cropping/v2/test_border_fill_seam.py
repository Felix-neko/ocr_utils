"""Тесты мягкого шва закраса pull-push (``border_fill.seam_ramp``, ``FillMethod.PULL_PUSH_SEAM``) на синтетике."""

from __future__ import annotations

import numpy as np

from ocr_utils.scan_cropping.v2.border_fill import fill_pull_push
from ocr_utils.scan_cropping.v2.fill import FillMethod, fill_to_rectangle

# px/мм синтетической ¼ копии (как пак-6: 13.8 px/мм кадра × ¼).
MM = 3.45
# Цвета BGR: бумага, тёмная кайма завала у края, синяя печать у края, коврик.
PAPER = (205, 225, 230)
RIM = (180, 198, 203)
STAMP = (170, 60, 30)
MAT = (20, 20, 20)


def synthetic(stamp: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Страница слева (300 px) с тёмной каймой 1.2 мм у правого края, справа — коврик.

    Args:
        stamp: Положить синюю печать у края в строках 150–200.

    Returns:
        ``(картинка BGR uint8, маска бумаги)``.
    """
    img = np.full((400, 400, 3), PAPER, np.uint8)
    paper = np.zeros((400, 400), bool)
    paper[:, :300] = True
    rim = int(1.2 * MM)
    img[:, 300 - rim : 300] = RIM
    if stamp:
        img[150:200, 280:300] = STAMP
    img[:, 300:] = MAT
    return img, paper


def gray(bgr: np.ndarray) -> np.ndarray:
    """Серое по весам BT.601 (как ``cv2.COLOR_BGR2GRAY``)."""
    return bgr.astype(np.float32) @ np.array([0.114, 0.587, 0.299], np.float32)


def test_seam_starts_from_rim_and_converges():
    """У края закрас со швом ближе к кайме, чем без шва; за шириной шва — совпадает с прежним pull-push."""
    img, paper = synthetic()
    old = fill_pull_push(img, paper, MM, None, 0.0)
    new = fill_pull_push(img, paper, MM, None, 5.0)
    rim_gray = gray(np.array(RIM))
    near, far = 301, 300 + int(6 * MM)
    assert abs(gray(new[200, near]) - rim_gray) <= abs(gray(old[200, near]) - rim_gray)
    assert abs(gray(new[200, far]) - gray(old[200, far])) < 1.0


def test_seam_does_not_drag_stamp():
    """Синяя печать у края не уходит в закрас: у печати закрас не синее прежнего pull-push."""
    img, paper = synthetic(stamp=True)
    old = fill_pull_push(img, paper, MM, None, 0.0).astype(np.float32)
    new = fill_pull_push(img, paper, MM, None, 5.0).astype(np.float32)
    # Синева = B − R; печать даёт +140, бумага −25.
    blue_old = (old[160:190, 301:310, 0] - old[160:190, 301:310, 2]).mean()
    blue_new = (new[160:190, 301:310, 0] - new[160:190, 301:310, 2]).mean()
    assert blue_new <= blue_old + 3.0


def test_zero_seam_is_old_pull_push():
    """Ширина шва 0 — прежний pull-push байт в байт."""
    img, paper = synthetic()
    assert np.array_equal(fill_pull_push(img, paper, MM, None, 0.0), fill_pull_push(img, paper, MM, None))


def test_fill_method_switch_keeps_paper():
    """``PULL_PUSH`` и ``PULL_PUSH_SEAM`` различаются только вне бумаги; бумага не тронута."""
    small, paper_small = synthetic()
    # Полный масштаб: ×4 (fill_non_paper_pull_push сам уменьшает до ¼).
    img = np.kron(small, np.ones((4, 4, 1), np.uint8))
    paper = np.kron(paper_small, np.ones((4, 4), bool))
    old = fill_to_rectangle(img, paper, FillMethod.PULL_PUSH, box=(0, 0, 1600, 1600), px_per_mm=MM * 4)
    new = fill_to_rectangle(img, paper, FillMethod.PULL_PUSH_SEAM, box=(0, 0, 1600, 1600), px_per_mm=MM * 4)
    assert np.array_equal(old[paper], img[paper])
    assert np.array_equal(new[paper], img[paper])
    assert not np.array_equal(old[~paper], new[~paper])
