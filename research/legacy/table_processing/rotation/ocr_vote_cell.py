"""Арбитр стороны: распознавание ячейки на каждом допустимом повороте.

ЗАЧЕМ ИМЕННО ЗДЕСЬ. Ось ячейки надёжно дают три дешёвые меры, а вот СТОРОНУ (90 против 270)
не даёт ни одна: буква, повёрнутая влево, ровно так же широка, как повёрнутая вправо.
Сторону может назвать только тот, кто текст читает.

ПОЧЕМУ TESSERACT, А НЕ SURYA. Замерено в ``orientation/README.md``: surya сам выпрямляет
вход и на перевёрнутом тексте отдаёт уверенность 1.00, то есть для голосования по стороне
бесполезен. Tesseract читает то, что дали, и на неверной стороне выдаёт мусор — что и
требуется. Счёт — число букв кириллицы в словах с уверенностью не ниже порога; та же
мера, что у арбитра полос, и по той же причине: она не зависит ни от длины строки, ни от
языковой модели.

ЦЕНА. Прогон распознавания на каждый допустимый угол. Для ячейки это доли секунды, но
ячеек в таблице десятки, поэтому арбитр вызывается только по тем ячейкам, которые дешёвые
меры уже назвали боковыми.
"""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.scan_markup.orientation.detectors.ocr_vote import letters
from ocr_utils.scan_markup.orientation.detectors.osd import tesseract_available

from research.legacy.table_processing.rotation.base import CellCrop, CellDetector, Verdict, rotate_cw

# Ниже этого числа букв на лучшем повороте считаем, что читать нечего.
MIN_LETTERS = 3

# Поля вокруг ячейки перед распознаванием: tesseract плохо читает текст, прижатый к краю.
PAD_PX = 12

# Ниже этой высоты строки (в пикселях) кроп увеличивается: у tesseract точность падает,
# когда знак ниже 20 px, а в шапке таблицы петит при 300 dpi даёт как раз около 20.
MIN_HEIGHT_PX = 300
UPSCALE = 2


def _prepared(gray: np.ndarray, rotation: int) -> np.ndarray:
    turned = rotate_cw(gray, rotation)
    if min(turned.shape[:2]) < MIN_HEIGHT_PX:
        turned = cv2.resize(turned, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(turned, PAD_PX, PAD_PX, PAD_PX, PAD_PX, cv2.BORDER_CONSTANT, value=255)


def detect(crop: CellCrop) -> Verdict:
    scores = {rotation: letters(_prepared(crop.gray, rotation)) for rotation in crop.allowed}
    metrics = {f"letters_{rotation}": float(value) for rotation, value in scores.items()}
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_rotation, best_score = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else 0

    if best_score < MIN_LETTERS:
        return Verdict(0, 0.0, metrics=metrics, note="букв не нашлось ни под каким углом")
    confidence = min(1.0, (best_score - second) / best_score)
    return Verdict(best_rotation, confidence, metrics=metrics)


ALGORITHM = CellDetector(
    name="ocr_vote",
    summary="арбитр: tesseract на каждом допустимом повороте, побеждает угол с большим числом букв",
    stage="cpu",
    gives_sign=True,
    run=detect,
    available=tesseract_available,
)
