"""Мера тёмного пятна на месте пальца: провал яркости закраса пальца на бумаге против кольца бумаги вокруг зоны, без букв."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

# Масштаб, на котором считается мера (как мера шлейфа ``research/border_fill``).
MEASURE_SCALE = 0.25
# Окно закрытия, которым из серого убираются буквы (фон бумаги для поиска краски), мм.
INK_CLOSE_MM = 3.0
# Насколько пиксель темнее фона бумаги, чтобы считаться краской (буквы, линейки), уровни серого.
INK_DARK = 20
# Предел насыщенности краски (хрома Lab, 8-битные единицы OpenCV): насыщенное — не буквы.
INK_MAX_CHROMA = 16
# Кольцо бумаги вокруг зоны пальца, по которому берётся опорная яркость: от и до, мм от зоны.
RING_MM = (3.0, 10.0)
# Наименьшее число пикселей кольца у компоненты зоны, чтобы опора считалась.
RING_MIN_PX = 50
# Провал L (Lab, 0..100) пикселя зоны против опоры, с которого пиксель — «пятно».
SPOT_DROP_L = 8.0
# Наименьшая площадь пятна, мм².
MIN_SPOT_MM2 = 4.0


@dataclass(frozen=True)
class SpotScore:
    """Мера тёмного пятна на месте пальца в одном прямоугольнике итога.

    Поля:
        area_mm2: Площадь пятен (пиксели зоны пальца на бумаге темнее опоры на ``SPOT_DROP_L``), мм².
        mean_drop: Средний провал L по пятнам (0 — пятен нет).
        max_drop: 95-й процентиль провала L по пятнам.
        zone_mm2: Площадь зоны пальца на бумаге, мм².
        spots: Маска пятен на масштабе ``MEASURE_SCALE``.
        zone: Маска зоны пальца на бумаге на масштабе ``MEASURE_SCALE``.
    """

    area_mm2: float
    mean_drop: float
    max_drop: float
    zone_mm2: float
    spots: np.ndarray
    zone: np.ndarray


def ink_mask(bgr: np.ndarray, mm: float) -> np.ndarray:
    """Краска (буквы, линейки): темнее закрытия серого окном ``INK_CLOSE_MM`` на ``INK_DARK`` и малонасыщенная.

    Args:
        bgr: Картинка BGR uint8.
        mm: Плотность картинки, px/мм.

    Returns:
        Булева маска краски, расширенная на 1 px.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = max(3, int(round(INK_CLOSE_MM * mm)) | 1)
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    ink = (paper.astype(np.int16) - gray > INK_DARK) & (chroma <= INK_MAX_CHROMA)
    return cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def spot_score(bgr: np.ndarray, paint: np.ndarray, visible: np.ndarray, px_per_mm: float) -> SpotScore:
    """Мера тёмного пятна на месте пальца.

    Зона — закрас пальца на бумаге (``paint ∩ visible``). Опора — медиана L кольца бумаги ``RING_MM`` вокруг своей
    компоненты зоны (без краски). Пятно — пиксели зоны без краски, темнее опоры на ``SPOT_DROP_L``, компоненты от
    ``MIN_SPOT_MM2``.

    Args:
        bgr: Итоговый прямоугольник BGR uint8 (полный масштаб).
        paint: Зона закраса пальца в координатах прямоугольника (bool).
        visible: Бумага ∪ защита контента в координатах прямоугольника (bool): что в итоге — не закрас до рамки.
        px_per_mm: Плотность прямоугольника, px/мм.

    Returns:
        Мера.
    """
    s = MEASURE_SCALE
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    size = (small.shape[1], small.shape[0])
    mm = px_per_mm * s
    paint_s = cv2.resize(paint.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    vis_s = cv2.resize(visible.astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 0.99
    zone = paint_s & vis_s
    empty = np.zeros(zone.shape, bool)
    if not zone.any():
        return SpotScore(0.0, 0.0, 0.0, 0.0, empty, zone)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    light = lab[..., 0] * (100 / 255)
    ink = ink_mask(small, mm)
    # Кольцо бумаги вокруг зоны и номер ближайшей компоненты зоны для каждого его пикселя.
    n, labels = cv2.connectedComponents(zone.astype(np.uint8), connectivity=8)
    dist, (iy, ix) = ndimage.distance_transform_edt(~zone, return_indices=True)
    nearest = labels[iy, ix]
    ring = vis_s & ~paint_s & ~ink & (dist >= RING_MM[0] * mm) & (dist <= RING_MM[1] * mm)
    drop = np.zeros(zone.shape, np.float32)
    for k in range(1, n):
        own = ring & (nearest == k)
        if own.sum() < RING_MIN_PX:
            continue
        ref = float(np.median(light[own]))
        part = labels == k
        drop[part] = ref - light[part]
    spots = zone & ~ink & (drop > SPOT_DROP_L)
    # Без мелочи: пятна меньше MIN_SPOT_MM2 (после открытия 3×3 — без нитей вдоль букв).
    spots = cv2.morphologyEx(spots.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    m, comp, stats, _ = cv2.connectedComponentsWithStats(spots.astype(np.uint8), connectivity=8)
    small_ids = np.nonzero(stats[:, cv2.CC_STAT_AREA] < MIN_SPOT_MM2 * mm * mm)[0]
    spots &= ~np.isin(comp, small_ids)
    area = float(spots.sum() / (mm * mm))
    if not spots.any():
        return SpotScore(0.0, 0.0, 0.0, float(zone.sum() / (mm * mm)), spots, zone)
    d = drop[spots]
    return SpotScore(area, float(d.mean()), float(np.percentile(d, 95)), float(zone.sum() / (mm * mm)), spots, zone)
