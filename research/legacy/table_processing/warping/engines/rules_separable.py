"""Раздельное поле смещений: вертикальное по горизонтальным линейкам, горизонтальное — по вертикальным.

ЗАМЫСЕЛ. Горизонтальная линейка обязана быть горизонтальной прямой. Значит, зная её
ломаную, мы знаем, на сколько сдвинуть каждую точку по вертикали, чтобы она стала прямой.
Между линейками смещение интерполируется линейно, за крайними — продолжается постоянным.
Ровно то же по второй оси с вертикальными линейками. Два поля складываются и применяются
одним ``cv2.remap``.

ПОЧЕМУ РАЗДЕЛЬНО, А НЕ ТОНКОЙ ПЛАСТИНОЙ ПО ПЕРЕСЕЧЕНИЯМ. Опор мало: у половины таблиц пака
три-четыре горизонтальные линейки и пять-шесть вертикальных, а пересечений и того меньше
(у трети таблиц их вовсе нет — линейки не доходят друг до друга). Тонкая пластина на таком
числе опор ведёт себя тем хуже, чем дальше от них; раздельная схема на тех же данных
устойчива, потому что вдоль линейки опора есть В КАЖДОЙ точке.

ПОБОЧНО СНИМАЕТСЯ ПЕРЕКОС: наклон линейки — это тоже отклонение от горизонтали, и
выпрямление его убирает. Отдельный поворот после этого не нужен.
"""

from __future__ import annotations

import cv2
import numpy as np

from research.legacy.table_processing.detection.ruling import find_lines
from research.legacy.table_processing.warping.engines.base import Warped, Warper
from research.legacy.table_processing.warping.trace import Polyline, trace

# Меньше этого числа линеек по оси — поле по этой оси не строим.
MIN_RULES = 1

# Смещение больше этого считаем ошибкой прослеживания и поле не строим: 3 мм на таблице —
# это уже не кривизна, а слипшиеся линейки соседних строк.
MAX_SHIFT_MM = 3.0


def _axis_field(lines: list[Polyline], width: int, height: int, horizontal: bool) -> "np.ndarray | None":
    """Поле смещений вдоль одной оси: (H, W), в пикселях."""
    if len(lines) < MIN_RULES:
        return None

    # Для каждой линейки: её цель (прямая на средней координате) и отклонение в каждой точке.
    axis = np.arange(width if horizontal else height, dtype=float)
    targets: list[float] = []
    deviations: list[np.ndarray] = []
    for line in sorted(lines, key=lambda item: float(np.mean(item.across))):
        target = float(np.mean(line.across))
        sampled = np.interp(axis, line.along, line.across, left=line.across[0], right=line.across[-1])
        targets.append(target)
        deviations.append(sampled - target)
    stack = np.stack(deviations)  # (линеек, длина оси)

    if len(targets) == 1:
        field = np.repeat(stack, height if horizontal else width, axis=0)
        return field if horizontal else field.T

    # Между линейками — линейная интерполяция по их целевым координатам.
    knots = np.array(targets, dtype=float)
    grid = np.arange(height if horizontal else width, dtype=float)
    out = np.empty((grid.size, axis.size), dtype=np.float32)
    for index in range(axis.size):
        out[:, index] = np.interp(grid, knots, stack[:, index])
    return out if horizontal else out.T


def run(gray: np.ndarray, dpi: int) -> Warped:
    height, width = gray.shape[:2]
    horizontal, vertical = trace(find_lines(gray, dpi), dpi)
    vertical_field = _axis_field(horizontal, width, height, True)
    horizontal_field = _axis_field(vertical, width, height, False)
    if vertical_field is None and horizontal_field is None:
        return Warped(gray, note="линеек не нашлось")

    limit = MAX_SHIFT_MM * dpi / 25.4
    shift_y = np.zeros((height, width), np.float32) if vertical_field is None else vertical_field.astype(np.float32)
    shift_x = np.zeros((height, width), np.float32) if horizontal_field is None else horizontal_field.astype(np.float32)
    if float(np.abs(shift_y).max()) > limit or float(np.abs(shift_x).max()) > limit:
        return Warped(gray, note="смещение больше 3 мм — похоже на ошибку прослеживания")

    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    # map(x, y) говорит, ОТКУДА взять пиксель: чтобы линейка встала на своё место, берём её
    # из того места, где она сейчас, то есть прибавляем отклонение.
    straightened = cv2.remap(gray, grid_x + shift_x, grid_y + shift_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Warped(
        straightened,
        np.stack([shift_x, shift_y]),
        note=f"по {len(horizontal)} горизонталям и {len(vertical)} вертикалям",
        changed=True,
    )


ALGORITHM = Warper(name="rules_separable", summary="раздельные поля смещений по линейкам обеих осей", run=run)
