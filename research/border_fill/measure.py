"""Мера шлейфа закраса: отклонение цвета закраса (Lab) от робастного цвета бумаги у ближайшего участка края, площадь и пятна."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from ocr_utils.scan_cropping.v2.border_fill import to_lab

# Масштаб, на котором считается мера (как закрас Telea в ``v2/fill.py``).
MEASURE_SCALE = 0.25
# Глубина полосы бумаги у края, по которой берётся её цвет, мм.
BAND_MM = 10.0
# Размер участка (клетки сетки), по которому берётся медиана цвета бумаги, мм.
CELL_MM = 25.0
# Наименьшее число пикселей полосы в клетке, чтобы её медиана считалась.
CELL_MIN_PX = 30
# Отклонение ΔE76 (Lab, L 0..100) закраса от бумаги, выше которого пиксель — «шлейф».
TRAIL_DELTA_E = 15.0
# Отступ от края бумаги, ближе которого пиксели не меряются (смешанный край), мм.
SKIP_MM = 2.0
# Наименьшая площадь пятна шлейфа, мм².
MIN_SPOT_MM2 = 4.0


@dataclass(frozen=True)
class TrailScore:
    """Мера шлейфа одного прямоугольника.

    Поля:
        area_mm2: Площадь пикселей закраса с ΔE > порога (пятна не меньше ``MIN_SPOT_MM2``), мм².
        reach_mm: Наибольшая удалённость такого пикселя от края бумаги, мм (длина самого длинного шлейфа).
        mean_de: Среднее ΔE по всему закрасу дальше ``SKIP_MM`` от края.
        fill_mm2: Площадь закраса, мм².
        spots: Маска шлейфа на масштабе ``MEASURE_SCALE`` (для листов).
    """

    area_mm2: float
    reach_mm: float
    mean_de: float
    fill_mm2: float
    spots: np.ndarray


def cell_medians(lab: np.ndarray, band: np.ndarray, cell_px: int) -> np.ndarray:
    """Медиана Lab полосы бумаги по клеткам сетки, клетки без полосы — значением ближайшей клетки с полосой.

    Args:
        lab: Картинка Lab (H, W, 3).
        band: Булева маска полосы бумаги у края.
        cell_px: Сторона клетки, px.

    Returns:
        Массив (Hc, Wc, 3) медиан; если полосы нет вовсе — заполнен NaN.
    """
    h, w = band.shape
    hc, wc = -(-h // cell_px), -(-w // cell_px)
    out = np.full((hc, wc, 3), np.nan, np.float32)
    ys, xs = np.nonzero(band)
    if len(ys) == 0:
        return out
    # Номер клетки каждого пикселя полосы, сортировка по нему — медиана по группам.
    cell = (ys // cell_px) * wc + xs // cell_px
    order = np.argsort(cell, kind="stable")
    cell, vals = cell[order], lab[ys[order], xs[order]]
    starts = np.r_[0, np.nonzero(np.diff(cell))[0] + 1]
    ends = np.r_[starts[1:], len(cell)]
    for s, e in zip(starts, ends):
        if e - s >= CELL_MIN_PX:
            out.flat[cell[s] * 3 : cell[s] * 3 + 3] = np.median(vals[s:e], axis=0)
    valid = ~np.isnan(out[..., 0])
    if not valid.any():
        return out
    # Пустые клетки — значением ближайшей заполненной.
    _, (iy, ix) = ndimage.distance_transform_edt(~valid, return_indices=True)
    return out[iy, ix]


def trail_score(bgr: np.ndarray, paper: np.ndarray, keep: np.ndarray, px_per_mm: float) -> TrailScore:
    """Мера шлейфа закраса на итоговом прямоугольнике.

    Опорный цвет пикселя закраса — медиана Lab полосы бумаги (``BAND_MM`` внутрь от края) в клетке ``CELL_MM``, где
    лежит ближайший к нему пиксель бумаги. Шлейф — пиксели закраса дальше ``SKIP_MM`` от бумаги с ΔE > порога.

    Args:
        bgr: Итоговый прямоугольник BGR uint8 (полный масштаб).
        paper: Маска бумаги прямоугольника (полный масштаб).
        keep: Маска сохранённого сверх бумаги (защита контента): не закрас, не меряется.
        px_per_mm: Плотность прямоугольника, px/мм.

    Returns:
        Мера шлейфа.
    """
    s = MEASURE_SCALE
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    size = (small.shape[1], small.shape[0])
    p = cv2.resize(paper.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    k = cv2.resize(keep.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    mm = px_per_mm * s
    empty = TrailScore(0.0, 0.0, 0.0, 0.0, np.zeros(p.shape, bool))
    if not p.any() or p.all():
        return empty
    lab = to_lab(small)
    # Полоса бумаги у края и медианы по клеткам.
    inside = ndimage.distance_transform_edt(p)
    band = p & (inside <= BAND_MM * mm)
    medians = cell_medians(lab, band, max(int(round(CELL_MM * mm)), 4))
    if np.isnan(medians).all():
        return empty
    # Ближайший пиксель бумаги для каждого пикселя не-бумаги и его клетка.
    dist, (iy, ix) = ndimage.distance_transform_edt(~p, return_indices=True)
    cell_px = max(int(round(CELL_MM * mm)), 4)
    ref = medians[iy // cell_px, ix // cell_px]
    fill = ~p & ~k
    measured = fill & (dist > SKIP_MM * mm)
    de = np.linalg.norm(lab - ref, axis=2)
    spots = measured & (de > TRAIL_DELTA_E)
    # Без мелочи: пятна меньше MIN_SPOT_MM2.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(spots.astype(np.uint8), connectivity=8)
    small_spots = np.nonzero(stats[:, cv2.CC_STAT_AREA] < MIN_SPOT_MM2 * mm * mm)[0]
    spots &= ~np.isin(labels, small_spots)
    area = float(spots.sum() / (mm * mm))
    reach = float(dist[spots].max() / mm) if spots.any() else 0.0
    mean_de = float(de[measured].mean()) if measured.any() else 0.0
    return TrailScore(area, reach, mean_de, float(fill.sum() / (mm * mm)), spots)
