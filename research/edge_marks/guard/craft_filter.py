"""Фильтр глифов у аномальных выровненных сторон: краска, в которой голосование детекторов (CRAFT, pero, docTR) видит сор, закрашивается перед повторным разбором."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from ocr_utils.page_layout.text_blocks import RENDER_DPI
from ocr_utils.page_layout.text_blocks.page import PageAnalysis
from research.edge_marks.guard.anomaly import SideBump, robust_line
from research.edge_marks.guard.glyph_vote import Engine, component_scores, is_junk

# Зона фильтра у аномальной стороны: начинается в ``ZONE_FROM_MM`` мм НАРУЖУ от устойчивой прямой стороны и идёт
# ещё на ``ZONE_OUT_MM`` мм (не дальше края страницы и не внутрь других блоков). Компонента проверяется, только если
# задевает зону, то есть выходит за сторону: краска внутри колонки выступа не даёт. Зона, начинавшаяся на 3 мм
# ВНУТРИ стороны, выбросила обломок разорванной бинаризацией «щ» в начале строки (1975/01 IMG_0026_1L). Наружу до
# края страницы было нельзя: зона цепляла текст вне блоков — на 1975 IMG_0146_1L выбросила тире посреди строк.
ZONE_FROM_MM = 0.3
ZONE_OUT_MM = 12.0
# Компоненты длиннее этого (мм) не трогаются: линейки, рамки.
MAX_DROP_MM = 20.0
# Горизонтальный штрих (ширина не меньше стольких высот) не трогается: дефис, тире, линейка — CRAFT их почти не
# видит (на 55 полосах выброшены висячий дефис «Глав-» и тире посреди строки).
BAR_ASPECT = 2.0
# Запас по высоте зоны сверху и снизу блока, шагов строк.
ZONE_PAD_PITCHES = 1.0
# Порог краски рендера.
INK_LEVEL = 128


def side_zones(analysis: PageAnalysis, bumps: list[SideBump], shape300: tuple[int, int]) -> np.ndarray:
    """Маска зоны фильтра в пикселях рендера 300 dpi: полосы у аномальных выровненных сторон.

    По каждой стороне блока, где есть выступ: по высоте — весь блок с запасом ``ZONE_PAD_PITCHES`` шага; по
    ширине — от ``ZONE_FROM_MM`` до ``ZONE_FROM_MM + ZONE_OUT_MM`` мм наружу от устойчивой прямой стороны. Контуры ДРУГИХ
    блоков страницы из зоны вычитаются: соседняя колонка не фильтруется.

    Args:
        analysis: Разбор страницы (пиксели рабочей копии).
        bumps: Выступы (:func:`anomaly.page_bumps`).
        shape300: Размер рендера ``(высота, ширина)``.

    Returns:
        Булева маска ``shape300``.
    """
    height, width = shape300
    k = RENDER_DPI / analysis.dpi
    zone = np.zeros(shape300, dtype=bool)
    sides = {(bump.block, bump.side) for bump in bumps}
    for number, side in sorted(sides):
        block = analysis.blocks[number]
        curve = np.asarray(getattr(block.envelope, side), dtype=np.float64)
        a, b = robust_line(curve)
        pad = ZONE_PAD_PITCHES * block.pitch_px
        y0 = max(0, int((curve[:, 1].min() - pad) * k))
        y1 = min(height, int((curve[:, 1].max() + pad) * k) + 1)
        ys = np.arange(y0, y1)
        # Абсцисса стороны по высоте (прямая в пикселях рабочей копии → рендер).
        side_x = (a * (ys / k) + b) * k
        start, outside = (value * RENDER_DPI / 25.4 for value in (ZONE_FROM_MM, ZONE_FROM_MM + ZONE_OUT_MM))
        for y, x in zip(ys, side_x):
            if side == "right":
                zone[y, max(0, int(x + start)) : min(width, int(x + outside) + 1)] = True
            else:
                zone[y, max(0, int(x - outside)) : max(0, min(width, int(x - start) + 1))] = True
    # Другие блоки не трогаем: их контуры (со своими строками) вычитаются из зоны.
    others = np.zeros(shape300, dtype=np.uint8)
    for number, block in enumerate(analysis.blocks):
        if any(number == n for n, _ in sides):
            continue
        polygon = np.asarray(block.envelope.polygon, dtype=np.float64) * k
        cv2.fillPoly(others, [polygon.astype(np.int32)], 1)
    return zone & (others == 0)


def keep_gray(
    gray300: np.ndarray, zone: np.ndarray, maps: dict[Engine, np.ndarray], row_of: Callable[[float], tuple[float, float]]
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Закрасить в зоне краску, которую голосование детекторов считает сором.

    Компоненты краски (8-связность), задевающие зону, оцениваются детекторами ``maps``
    (:func:`glyph_vote.component_scores`) и закрашиваются белым, если хотя бы один видит в них сор
    (:func:`glyph_vote.is_junk`). Компоненты длиннее ``MAX_DROP_MM`` и горизонтальные штрихи (ширина не меньше
    ``BAR_ASPECT`` высот: дефис, тире, линейка) не трогаются.

    Args:
        gray300: Серая полоса 300 dpi.
        zone: Маска зоны (:func:`side_zones`).
        maps: Карты голосующих детекторов того же размера, 0…1.
        row_of: Середина строки и x-высота по ординате компоненты (пиксели рендера) — для окна базовой линии pero.

    Returns:
        ``(полоса с закрашенным сором, рамки выброшенных компонент x0, y0, x1, y1 в пикселях рендера)``.
    """
    ink = (gray300 < INK_LEVEL).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    max_px = MAX_DROP_MM * RENDER_DPI / 25.4
    # Кандидаты: компоненты, задевшие зону, не штрихи и не длинные линейки.
    touched = np.zeros(count, dtype=bool)
    touched[np.unique(labels[zone & (ink > 0)])] = True
    touched[0] = False
    bar = stats[:, 2] >= BAR_ASPECT * stats[:, 3]
    candidates = np.nonzero(touched & ~bar & (np.maximum(stats[:, 2], stats[:, 3]) <= max_px))[0]
    drop = np.zeros(count, dtype=bool)
    for index in candidates:
        centre_y = stats[index, 1] + stats[index, 3] / 2.0
        drop[index] = is_junk(component_scores(labels, stats, int(index), maps, row_of(centre_y)))
    out = gray300.copy()
    out[drop[labels]] = 255
    boxes = [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h, _ in stats[np.nonzero(drop)[0]]]
    return out, boxes


__all__ = ["BAR_ASPECT", "MAX_DROP_MM", "ZONE_FROM_MM", "ZONE_OUT_MM", "keep_gray", "side_zones"]
