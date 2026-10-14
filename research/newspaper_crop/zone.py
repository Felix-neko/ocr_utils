"""Общий хвост движков: чистка маски бумаги, охватывающий четырёхугольник с перспективой, доля дорисовки, меры зоны."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Ядро подгонки четырёхугольника переехало в ``scan_cropping.v2.trapezoid``; здесь — реэкспорт для стенда.
from ocr_utils.scan_cropping.v2.trapezoid import (  # noqa: F401
    DEFAULT_SIDE_FIT,
    RANSAC_TOL,
    SIDE_BAND,
    SIDE_QUANTILE,
    SideFit,
    fit_enclosing_quad,
    keystone_deg,
    quad_mask,
)

# Рабочий масштаб маски относительно превью 2944×4416: половина (1472×2208) — край бумаги на нём
# точен до ~2 px превью, а морфология и контуры идут быстро.
WORK_SCALE = 0.5
# Признаки вырожденного четырёхугольника: трапеция круче (градусы — на выборке ЭГ настоящая не больше 11.6°) или
# за ним осталась доля листа больше порога.
MAX_KEYSTONE, MAX_OUTSIDE = 25.0, 0.01
# Наименьшая доля кадра у маски бумаги: меньше — движок промахнулся.
MIN_PAPER_FRAC = 0.2


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Заливает дыры маски (всё, что не связано с краем кадра через фон).

    Args:
        mask: Булева маска (H, W).

    Returns:
        Булева маска без внутренних дыр.
    """
    h, w = mask.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = mask.astype(np.uint8)
    flood = pad.copy()
    cv2.floodFill(flood, None, (0, 0), 2)
    return (flood[1:-1, 1:-1] != 2) | mask


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Оставляет крупнейшую связную область маски.

    Args:
        mask: Булева маска.

    Returns:
        Булева маска одной области (пустая, если на входе пусто).
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, bool)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best


def clean_paper_mask(mask: np.ndarray, open_px: int = 5) -> np.ndarray:
    """Приводит сырую маску движка к «листу»: открытие от ниток, крупнейшая область, без дыр.

    Args:
        mask: Булева маска бумаги на рабочем масштабе.
        open_px: Радиус открытия, px рабочего масштаба (снимает перемычки к соседней полосе в пару пикселей).

    Returns:
        Булева маска листа.
    """
    m = mask.astype(np.uint8)
    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_px + 1, 2 * open_px + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return fill_holes(largest_component(m.astype(bool)))


@dataclass(frozen=True)
class Zone:
    """Итог по кадру: маска листа, охватывающий четырёхугольник и меры (доли — от площади кадра или четырёхугольника)."""

    paper: np.ndarray
    quad: np.ndarray | None
    paper_frac: float
    fill_frac: float
    outside_frac: float
    keystone_deg: float

    @property
    def ok(self) -> bool:
        """Зона построена: есть четырёхугольник и лист не крохотный."""
        return self.quad is not None and self.paper_frac >= MIN_PAPER_FRAC


def build_zone(raw_mask: np.ndarray, quantile: float = SIDE_QUANTILE) -> Zone:
    """Собирает зону из сырой маски движка.

    Args:
        raw_mask: Булева маска бумаги на рабочем масштабе.
        quantile: Квантиль сторон (см. :func:`fit_enclosing_quad`).

    Returns:
        :class:`Zone`: доля дорисовки — часть четырёхугольника вне листа; ``outside_frac`` — часть листа
        вне четырёхугольника (обрезано квантилем).
    """
    paper = clean_paper_mask(raw_mask)
    if not paper.any():
        return zone_of(paper, None)
    quad = fit_enclosing_quad(paper, quantile)
    zone = zone_of(paper, quad) if quad is not None else None
    if zone is None or zone.keystone_deg > MAX_KEYSTONE or zone.outside_frac > MAX_OUTSIDE:
        # Вырожденная подгонка (маска упёрлась в рамку кадра с нескольких сторон — две стороны легли на одну
        # рамку, диагональ через лист): запасной четырёхугольник — повёрнутый прямоугольник листа.
        contours, _ = cv2.findContours(paper.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        box = cv2.boxPoints(cv2.minAreaRect(max(contours, key=cv2.contourArea))).astype(np.float64)
        zone = zone_of(paper, box)
    return zone


def zone_of(paper: np.ndarray, quad: np.ndarray | None) -> Zone:
    """Зона с мерами по готовым маске листа и четырёхугольнику (после правки — например, отрезки за складкой).

    Args:
        paper: Булева маска листа на рабочем масштабе.
        quad: Вершины четырёхугольника в пикселях маски или None.

    Returns:
        :class:`Zone`.
    """
    area = paper.size
    if quad is None:
        return Zone(paper, None, float(paper.mean()), 0.0, 0.0, 0.0)
    qm = quad_mask(quad, paper.shape)
    q_area = max(int(qm.sum()), 1)
    return Zone(
        paper=paper,
        quad=quad,
        paper_frac=float(paper.sum() / area),
        fill_frac=float((qm & ~paper).sum() / q_area),
        outside_frac=float((paper & ~qm).sum() / max(int(paper.sum()), 1)),
        keystone_deg=keystone_deg(quad),
    )


def ink_mask(rgb: np.ndarray, shape: tuple[int, int], block: int = 31, offset: int = 18) -> np.ndarray:
    """Краска печати на рабочем масштабе: адаптивный порог (тёмнее локального среднего на ``offset``).

    Args:
        rgb: Кадр RGB.
        shape: Рабочий (H, W).
        block: Окно адаптивного порога, px рабочего масштаба (≈ три строки текста превью).
        offset: Насколько темнее локального среднего, уровни серого.

    Returns:
        Булева маска краски.
    """
    gray = cv2.cvtColor(cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    return cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, offset) > 0


def ink_in_fill(zone: Zone, ink: np.ndarray, erode_px: int = 6) -> float:
    """Доля краски четырёхугольника, попавшая в зону дорисовки вглубь от края листа дальше ``erode_px``.

    Высокое значение — движок недобрал бумагу и дорисовка затрёт печать. Полоса у самого края листа
    не считается: там краской кажутся тени и торцы нижних листов.

    Args:
        zone: Зона.
        ink: Маска краски.
        erode_px: Ширина полосы у края, px рабочего масштаба.

    Returns:
        Доля (0, если четырёхугольника нет).
    """
    if zone.quad is None:
        return 0.0
    qm = quad_mask(zone.quad, zone.paper.shape)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
    deep = cv2.erode((qm & ~zone.paper).astype(np.uint8), k) > 0
    total = int((ink & qm).sum())
    return float((ink & deep).sum() / total) if total else 0.0


def work_shape_of(rgb: np.ndarray) -> tuple[int, int]:
    """Размер рабочей маски (H, W) для кадра на масштабе :data:`WORK_SCALE`.

    Args:
        rgb: Кадр превью.

    Returns:
        (H, W).
    """
    return int(round(rgb.shape[0] * WORK_SCALE)), int(round(rgb.shape[1] * WORK_SCALE))
