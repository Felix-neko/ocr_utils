"""Уточнение зоны склейки страниц у края книги: в полосе до D px от фона остаётся только бумага (не торец, не обложка, не фон)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from scipy import ndimage

from ocr_utils.scan_cropping.v2.glue_refine import (  # noqa: F401 — ядро, ре-экспорт для стенда
    L_MARGIN,
    OPEN_PX,
    SEED_FAR,
    SEED_MIN,
    SEED_NEAR,
    RefineResult,
    background_like,
    classify_colour,
    colour_model,
    finish_refine,
    refine_band,
)
from ocr_utils.scan_cropping.v2.paper import smooth_lab

# R2: ширина слоя по глубине от фона и полуокно вдоль края для опорного цвета бумаги, px; порог ΔE (Lab, 8 бит).
RING_HALF = 5
ALONG_HALF = 150
LOCAL_DE = 6.0
# R3: запас вокруг полосы для рамки GrabCut, px; итераций.
GRABCUT_PAD = 30
GRABCUT_ITERS = 4


class RefineMethod(str, Enum):
    """Способ классификации пикселя полосы уточнения."""

    # Цветовая модель бумаги по затравкам страниц у корешка.
    COLOUR = "colour"
    # Сравнение с бумагой той же страницы на той же глубине от фона.
    LOCAL_TONE = "local_tone"
    # GrabCut в рамке полосы.
    GRABCUT = "grabcut"
    # Большинство из трёх.
    VOTE = "vote"










def classify_local_tone(
    bgr: np.ndarray, base: np.ndarray, depth: np.ndarray, band: np.ndarray, max_depth: int
) -> np.ndarray:
    """R2: пиксель полосы — бумага, если его Lab близок к медиане (здесь — среднему) бумаги той же глубины рядом.

    Опорный цвет по слоям глубины: сумма Lab бумаги страниц слоя в окне ±ALONG_HALF, делённая на число пикселей слоя
    (бокс-фильтр без нормировки), — на каждый слой глубины ±RING_HALF.

    Args:
        bgr: Кадр BGR.
        base: Маска бумаги до склейки (страницы).
        depth: Расстояние от фона внутри книги.
        band: Полоса уточнения.
        max_depth: Глубина полосы D.

    Returns:
        Булева карта «бумага» (вне полосы — True).
    """
    lab = smooth_lab(bgr)
    out = np.ones(base.shape, bool)
    k = (2 * ALONG_HALF + 1, 2 * ALONG_HALF + 1)
    for d0 in range(0, max_depth + 1, RING_HALF):
        layer = base & (np.abs(depth - d0) <= RING_HALF)
        target = band & (np.abs(depth - d0) <= RING_HALF / 2 + 0.5)
        if not target.any():
            continue
        cnt = cv2.boxFilter(layer.astype(np.float32), -1, k, normalize=False)
        ref = np.stack([cv2.boxFilter(lab[..., c] * layer, -1, k, normalize=False) for c in range(3)], -1)
        ref = ref / np.maximum(cnt, 1)[..., None]
        # Только оттенок (a, b): тень складки у корешка темнее бумаги, но того же оттенка; обложка — другого.
        de = np.linalg.norm(lab[..., 1:] - ref[..., 1:], axis=-1)
        ok = (de <= LOCAL_DE) & (cnt > 50)
        out[target] = ok[target]
    return out


def classify_grabcut(
    bgr: np.ndarray, base: np.ndarray, band: np.ndarray, bgmask: np.ndarray, rough: np.ndarray
) -> np.ndarray:
    """R3: GrabCut в рамке полосы: страницы — бумага, фон и «не бумага» по R1 — фон.

    Args:
        bgr: Кадр BGR.
        base: Бумага до склейки.
        band: Полоса уточнения.
        bgmask: Фон.
        rough: Разметка R1 («бумага»).

    Returns:
        Булева карта «бумага» (вне полосы — True).
    """
    out = np.ones(base.shape, bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(band.astype(np.uint8))
    h, w = base.shape
    for i in range(1, count):
        x, y, bw, bh, _ = stats[i]
        x0, y0 = max(0, x - GRABCUT_PAD), max(0, y - GRABCUT_PAD)
        x1, y1 = min(w, x + bw + GRABCUT_PAD), min(h, y + bh + GRABCUT_PAD)
        sl = (slice(y0, y1), slice(x0, x1))
        part = labels[sl] == i
        mask = np.where(rough[sl], cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
        mask[base[sl] & ~part] = cv2.GC_FGD
        mask[bgmask[sl]] = cv2.GC_BGD
        if not ((mask == cv2.GC_FGD).any() and ((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD)).any()):
            continue
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(np.ascontiguousarray(bgr[sl]), mask, None, bgd, fgd, GRABCUT_ITERS, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            continue
        fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
        out[sl][part] = fg[part]
    return out


def refine(
    bgr: np.ndarray,
    book: np.ndarray,
    base: np.ndarray,
    glued: np.ndarray,
    hulls: np.ndarray,
    pages: list[np.ndarray],
    method: RefineMethod,
    max_depth: int,
) -> RefineResult:
    """Уточнение склейки: в полосе добавленного не глубже ``max_depth`` от фона остаётся только бумага.

    Args:
        bgr: Кадр BGR (½), после закраса пальцев.
        book: Маска книги.
        base: Бумага до склейки (``sam3_1288_plain``).
        glued: Бумага после склейки.
        hulls: Оболочки своей печати.
        pages: Страницы (``glue.split_pages``); сейчас не используются — уточнение только по сегментации.
        method: Способ классификации.
        max_depth: Глубина полосы уточнения D, px.

    Returns:
        Маски.
    """
    added, bgmask, depth, band = refine_band(bgr, book, base, glued, max_depth)
    if not band.any():
        return RefineResult(glued, added, np.zeros_like(added), band)
    r1 = classify_colour(bgr, book, hulls, added)
    if method == RefineMethod.COLOUR:
        paper = r1
    elif method == RefineMethod.LOCAL_TONE:
        paper = classify_local_tone(bgr, base, depth, band, max_depth)
    elif method == RefineMethod.GRABCUT:
        paper = classify_grabcut(bgr, base, band, bgmask, r1)
    else:
        votes = (
            r1.astype(int)
            + classify_local_tone(bgr, base, depth, band, max_depth)
            + classify_grabcut(bgr, base, band, bgmask, r1)
        )
        paper = votes >= 2
    return finish_refine(base, added, band, paper, bgmask)
