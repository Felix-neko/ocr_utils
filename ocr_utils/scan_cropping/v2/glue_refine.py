"""Уточнение зоны склейки страниц у края книги: в полосе до D px от фона остаётся только бумага (не торец, не обложка, не фон)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from ocr_utils.scan_cropping.v2.background import MAT_MAX_CHROMA, MAT_SAMPLE_MAX_GRAY, chroma
from ocr_utils.scan_cropping.v2.book import disk
from ocr_utils.scan_cropping.v2.paper import MAHAL_THR, PaperModel, paper_seeds, smooth_lab

# Глубина полосы уточнения от фона, px масштаба сегментации (~6.9 px/мм): 35 px ≈ 5 мм. Классификатор «бумага / не
# бумага» запросто принимает бумагу с текстом за не бумагу; при прежних 100 px (≈ 1.5 см) уточнение съело номер
# страницы у корешка (МТС 1991/03 IMG_0752, отсмотр прогона пака-6 2026-10-12). 5 мм — только торец и обложка у края.
REFINE_DEPTH = 35
# Затравки бумаги для модели цвета берутся в этом диапазоне расстояний от зоны склейки, px: бумага у корешка
# темнее и желтее середины страницы.
SEED_NEAR, SEED_FAR = 40, 150
# Если затравок у корешка меньше этого, берутся все затравки страниц.
SEED_MIN = 500
# Запас по яркости ниже 5-го перцентиля L затравок. Тень складки у корешка темнее бумаги на 60–80 уровней L, а
# оттенок у неё тот же: решает оттенок (Махаланобис по ab), яркость отсекает только почти чёрное.
L_MARGIN = 90
# Открытие оставленного добавленного, px.
OPEN_PX = 3


@dataclass(frozen=True)
class RefineResult:
    """Итог уточнения.

    Поля:
        paper: Маска бумаги после уточнения.
        added: Оставшееся добавленное склейкой.
        cut: Срезанное уточнением (было в склейке, оказалось не бумагой).
        band: Полоса уточнения (добавленное не глубже D от фона).
    """

    paper: np.ndarray
    added: np.ndarray
    cut: np.ndarray
    band: np.ndarray


def background_like(lab: np.ndarray, gray: np.ndarray, book: np.ndarray) -> np.ndarray:
    """Фон: вне книги или тёмный бесцветный пиксель (коврик, не попавший в фон).

    Args:
        lab: Кадр Lab (uint8).
        gray: Кадр в сером.
        book: Маска книги.

    Returns:
        Булева маска.
    """
    return ~book | ((gray < MAT_SAMPLE_MAX_GRAY) & (chroma(lab) < MAT_MAX_CHROMA))


def colour_model(bgr: np.ndarray, book: np.ndarray, hulls: np.ndarray, added: np.ndarray) -> tuple[PaperModel, float]:
    """Модель цвета бумаги по затравкам у корешка и нижний порог яркости.

    Args:
        bgr: Кадр BGR (масштаб сегментации).
        book: Маска книги.
        hulls: Оболочки своей печати.
        added: Добавленное склейкой.

    Returns:
        ``(модель, нижний порог L)``.
    """
    lab = smooth_lab(bgr)
    seeds = paper_seeds(bgr, hulls, book)
    # Затравки поближе к зоне склейки: бумага у корешка по цвету не такая, как в середине страницы.
    d = cv2.distanceTransform((~added).astype(np.uint8), cv2.DIST_L2, 5)
    near = seeds & (d >= SEED_NEAR) & (d <= SEED_FAR)
    use = near if near.sum() >= SEED_MIN else seeds
    model = PaperModel(lab, use)
    return model, float(np.percentile(lab[..., 0][use], 5)) - L_MARGIN


def classify_colour(bgr: np.ndarray, book: np.ndarray, hulls: np.ndarray, added: np.ndarray) -> np.ndarray:
    """Бумага по цветовой модели (Махаланобис по ab и нижний порог L).

    Args:
        bgr: Кадр BGR.
        book: Маска книги.
        hulls: Оболочки печати.
        added: Добавленное склейкой.

    Returns:
        Булева карта «бумага».
    """
    lab = smooth_lab(bgr)
    model, l_low = colour_model(bgr, book, hulls, added)
    return (model.distance(lab) < MAHAL_THR) & (lab[..., 0] >= l_low)


def refine_band(bgr: np.ndarray, book: np.ndarray, base: np.ndarray, glued: np.ndarray, max_depth: int):
    """Полоса уточнения и фон для неё.

    Args:
        bgr: Кадр BGR.
        book: Маска книги.
        base: Бумага до склейки.
        glued: Бумага после склейки.
        max_depth: Глубина полосы D от фона, px.

    Returns:
        ``(добавленное склейкой, фон, глубина от фона внутри книги, полоса)``.
    """
    added = glued & ~base
    lab8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bgmask = background_like(lab8, gray, book)
    depth = cv2.distanceTransform(book.astype(np.uint8), cv2.DIST_L2, 5)
    return added, bgmask, depth, added & (depth <= max_depth)


def finish_refine(
    base: np.ndarray, added: np.ndarray, band: np.ndarray, paper: np.ndarray, bgmask: np.ndarray
) -> RefineResult:
    """Сборка итога уточнения: из добавленного в полосе остаётся только «бумага» и не фон.

    Args:
        base: Бумага до склейки.
        added: Добавленное склейкой.
        band: Полоса уточнения.
        paper: Карта «бумага» классификатора.
        bgmask: Фон.

    Returns:
        Маски.
    """
    keep = added & ~(band & (~paper | bgmask))
    keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_OPEN, disk(OPEN_PX)) > 0
    out = base | keep
    # Дыры внутри щели (запертые мостиками) — бумага, но только внутри прежней склейки и не фон.
    out |= ndimage.binary_fill_holes(out) & added & ~bgmask
    kept = out & ~base
    return RefineResult(out, kept, added & ~kept, band)


def refine_colour(
    bgr: np.ndarray,
    book: np.ndarray,
    base: np.ndarray,
    glued: np.ndarray,
    hulls: np.ndarray,
    max_depth: int = REFINE_DEPTH,
) -> RefineResult:
    """Уточнение склейки цветовой моделью бумаги: в полосе добавленного не глубже ``max_depth`` от фона остаётся бумага.

    Args:
        bgr: Кадр BGR (масштаб сегментации), после закраса пальцев.
        book: Маска книги.
        base: Бумага до склейки.
        glued: Бумага после склейки.
        hulls: Оболочки своей печати.
        max_depth: Глубина полосы уточнения D, px.

    Returns:
        Маски.
    """
    added, bgmask, _, band = refine_band(bgr, book, base, glued, max_depth)
    if not band.any():
        return RefineResult(glued, added, np.zeros_like(added), band)
    return finish_refine(base, added, band, classify_colour(bgr, book, hulls, added), bgmask)
