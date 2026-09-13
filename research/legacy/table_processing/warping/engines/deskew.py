"""Поворот на медианный угол линеек — точка отсчёта для всех остальных.

Самое дешёвое, что можно сделать с перекошенной таблицей, и часто достаточное: замер по
паку показал, что у пропущенных таблиц медианный разброс углов 0.19°, а сагитта 0.21 мм,
то есть перекос и кривизна одного порядка. Если поворота хватает, городить поле смещений
незачем.
"""

from __future__ import annotations

import cv2
import numpy as np

from research.legacy.table_processing.detection.ruling import find_lines
from research.legacy.table_processing.warping.engines.base import Warped, Warper
from research.legacy.table_processing.warping.trace import trace

# Меньше этого угла не поворачиваем: интерполяция размоет штрих сильнее, чем выправит
# геометрию. 0.05° на таблице шириной 130 мм — это 0.11 мм на краю, вдвое меньше толщины
# линейки.
MIN_ANGLE_DEG = 0.05


def run(gray: np.ndarray, dpi: int) -> Warped:
    horizontal, vertical = trace(find_lines(gray, dpi), dpi)
    angles = [line.angle_deg for line in horizontal] or [line.angle_deg for line in vertical]
    if not angles:
        return Warped(gray, note="линеек не нашлось")
    angle = float(np.median(angles))
    if abs(angle) < MIN_ANGLE_DEG:
        return Warped(gray, note=f"угол {angle:+.3f}° — поворачивать нечего")

    height, width = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    turned = cv2.warpAffine(gray, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    moved_x = matrix[0, 0] * grid_x + matrix[0, 1] * grid_y + matrix[0, 2]
    moved_y = matrix[1, 0] * grid_x + matrix[1, 1] * grid_y + matrix[1, 2]
    field = np.stack([moved_x - grid_x, moved_y - grid_y])
    return Warped(turned, field, note=f"поворот на {angle:+.3f}°", changed=True)


ALGORITHM = Warper(name="deskew", summary="поворот на медианный угол линеек", run=run)
