"""Гомография по четырём углам таблицы: лечит трапецию.

КОГДА НУЖНА. Полосу снимали не строго сверху, и прямоугольная таблица стала трапецией:
верхняя линейка короче нижней, вертикали сходятся. Линейки при этом ОСТАЮТСЯ ПРЯМЫМИ, и
раздельное поле смещений такую беду видит как перекос каждой линейки по отдельности —
выправит, но ценой неравномерного растяжения. Гомография описывает ровно эту деформацию
четырьмя числами и потому честнее.

УГЛЫ БЕРУТСЯ ИЗ ЛИНЕЕК, а не из внешней рамки: рамки у половины таблиц пака нет вовсе.
Верх — самая верхняя горизонталь, низ — самая нижняя, лево и право — крайние вертикали;
углы получаются их пересечениями, продлёнными при необходимости.
"""

from __future__ import annotations

import cv2
import numpy as np

from research.legacy.table_processing.detection.ruling import find_lines
from research.legacy.table_processing.warping.engines.base import Warped, Warper
from research.legacy.table_processing.warping.trace import Polyline, trace

# Ниже этой разницы длин верхней и нижней линейки трапеции нет: 0.5%.
MIN_TAPER = 0.005


def _line_at(line: Polyline, coordinate: float) -> float:
    """Значение ломаной в точке, с продлением по прямой за её концами."""
    slope, intercept = np.polyfit(line.along, line.across, 1)
    if line.along[0] <= coordinate <= line.along[-1]:
        return float(np.interp(coordinate, line.along, line.across))
    return float(slope * coordinate + intercept)


def run(gray: np.ndarray, dpi: int) -> Warped:
    height, width = gray.shape[:2]
    horizontal, vertical = trace(find_lines(gray, dpi), dpi)
    if len(horizontal) < 2 or len(vertical) < 2:
        return Warped(gray, note="для гомографии нужны две горизонтали и две вертикали")

    top = min(horizontal, key=lambda line: float(np.mean(line.across)))
    bottom = max(horizontal, key=lambda line: float(np.mean(line.across)))
    left = min(vertical, key=lambda line: float(np.mean(line.across)))
    right = max(vertical, key=lambda line: float(np.mean(line.across)))

    top_y = float(np.mean(top.across))
    bottom_y = float(np.mean(bottom.across))
    corners = np.float32(
        [
            [_line_at(left, top_y), _line_at(top, _line_at(left, top_y))],
            [_line_at(right, top_y), _line_at(top, _line_at(right, top_y))],
            [_line_at(right, bottom_y), _line_at(bottom, _line_at(right, bottom_y))],
            [_line_at(left, bottom_y), _line_at(bottom, _line_at(left, bottom_y))],
        ]
    )
    top_width = abs(corners[1][0] - corners[0][0])
    bottom_width = abs(corners[2][0] - corners[3][0])
    if max(top_width, bottom_width) <= 0:
        return Warped(gray, note="углы не сошлись")
    taper = abs(top_width - bottom_width) / max(top_width, bottom_width)
    if taper < MIN_TAPER:
        return Warped(gray, note=f"трапеции нет: сужение {taper:.1%}")

    side = max(top_width, bottom_width)
    left_x = min(corners[0][0], corners[3][0])
    target = np.float32([[left_x, top_y], [left_x + side, top_y], [left_x + side, bottom_y], [left_x, bottom_y]])
    matrix = cv2.getPerspectiveTransform(corners, target)
    fixed = cv2.warpPerspective(gray, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    ones = np.ones_like(grid_x)
    stacked = np.stack([grid_x, grid_y, ones]).reshape(3, -1)
    moved = matrix @ stacked
    moved = (moved[:2] / np.maximum(1e-6, moved[2])).reshape(2, height, width)
    field = np.stack([moved[0] - grid_x, moved[1] - grid_y])
    return Warped(fixed, field, note=f"сужение {taper:.1%}", changed=True)


ALGORITHM = Warper(name="perspective", summary="гомография по четырём углам таблицы", run=run)
