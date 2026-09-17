"""Пара «было | стало» для просмотра глазами.

Обе половины уменьшены до одной высоты, между ними серая перемычка, поперёк обеих —
тонкие горизонтальные реперные линии через равные промежутки: на фоне ровной линейки
остаточный прогиб строки виден сразу, а без неё глаз «выпрямляет» строки сам.
"""

from __future__ import annotations

import cv2
import numpy as np

# Высота пары; ширина получается по пропорциям полос (портретная полоса ~ 0.58 высоты,
# вдвоём с перемычкой — около 1650 px).
PAIR_HEIGHT = 1400
GAP_PX = 12
REFERENCE_LINES = 12
LINE_COLOUR = (0, 0, 220)


def _fit_height(image: np.ndarray, height: int) -> np.ndarray:
    scale = height / image.shape[0]
    size = (max(1, round(image.shape[1] * scale)), height)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def side_by_side(before: np.ndarray, after: np.ndarray, height: int = PAIR_HEIGHT) -> np.ndarray:
    """BGR-картинка: слева исходник, справа результат, с реперными линиями."""
    left = _fit_height(before, height)
    right = _fit_height(after, height)
    if left.ndim == 2:
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    if right.ndim == 2:
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    gap = np.full((height, GAP_PX, 3), 128, np.uint8)
    canvas = np.hstack([left, gap, right])
    for index in range(1, REFERENCE_LINES):
        y = round(height * index / REFERENCE_LINES)
        cv2.line(canvas, (0, y), (canvas.shape[1] - 1, y), LINE_COLOUR, 1, cv2.LINE_AA)
    cv2.putText(canvas, "before", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(
        canvas, "after", (left.shape[1] + GAP_PX + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA
    )
    return canvas
