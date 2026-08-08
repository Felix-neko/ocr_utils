"""Энергия растрового муара, нормированная на контраст тайла.

ИДЕЯ. Газетный полутоновый растр — это готовая мира, отпечатанная прямо на странице.
Если уменьшить резкий кадр БЕЗ сглаживания, растр даёт сильный муар (именно так расфокус
ловят глазами, листая превью в просмотрщике); если кадр мягкий, растр уже смазан оптикой
и муара не будет. Меряем муар как ``std(NEAREST − AREA)``: структура текста есть в обоих
уменьшениях и сокращается, остаётся чистая энергия растра.

Сырой муар растёт вместе с количеством краски на полосе (корреляция ≈ +0.5 с количеством
контента — замерено в ``defocus_moire_improvement_plan.md``), поэтому делим его на контраст
тайла ``std(AREA)``. После нормировки зависимость от заполнения текстом пропадает (≈ −0.1).

Ограничение: метрика требует наличия типографского растра. На обложках, картонных
переплётах и рукописных этикетках растра нет — такие файлы она ранжирует бессмысленно.
"""

import cv2
import numpy as np

from ocr_utils.defocus_detection.metrics.base import Algorithm
from ocr_utils.defocus_detection.tiles import Grid

# Во сколько раз уменьшать кадр перед замером муара. При 2× растр газетного превью
# попадает точно в зону алиасинга.
DEFAULT_FACTOR = 2.0


def _tile_sharpness(gray: np.ndarray, grid: Grid, factor: float = DEFAULT_FACTOR) -> np.ndarray:
    """Карта нормированного муара по тайлам (больше = резче).

    Args:
        gray: Полутоновый кадр.
        grid: Сетка тайлов.
        factor: Коэффициент уменьшения кадра перед замером муара.

    Returns:
        Массив (ny, nx) со значениями std(NEAREST−AREA) / std(AREA).
    """
    g = gray.astype(np.float32)
    h, w = g.shape
    nw, nh = max(1, int(w / factor)), max(1, int(h / factor))
    nearest = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_NEAREST)
    area = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    diff = nearest - area

    out = np.full((grid.ny, grid.nx), np.nan)
    for iy in range(grid.ny):
        for ix in range(grid.nx):
            y1, y2, x1, x2 = grid.bounds(iy, ix)
            # Границы тайла пересчитываем в координаты уменьшенного кадра.
            sy1, sy2 = int(y1 / factor), max(int(y1 / factor) + 1, int(y2 / factor))
            sx1, sx2 = int(x1 / factor), max(int(x1 / factor) + 1, int(x2 / factor))
            structure = float(area[sy1:sy2, sx1:sx2].std())
            if structure < 1e-3:
                continue
            out[iy, ix] = float(diff[sy1:sy2, sx1:sx2].std()) / structure
    return out


ALGORITHM = Algorithm(
    name="moire",
    summary="муар типографского растра при уменьшении без сглаживания, нормированный на контраст тайла",
    tile_sharpness=_tile_sharpness,
    unit="муар/контраст",
)
