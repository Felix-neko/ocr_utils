"""Фон на коврике у кадра без печати: порог Оцу по одному коврику (одна мода) режет освещённый коврик в «не фон» — порог принимается, только если он заметно выше коврика (МТС 1990/04 IMG_1108)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.background import (
    MAT_MAX_CHROMA,
    MAT_TAIL_MARGIN,
    MAT_TAIL_PCT,
    book_seeds,
    chroma,
    clean_background,
    mat_seeds,
)
from ocr_utils.scan_cropping.v2.book import book_close_open, content_hulls

# Порог Оцу принимается, только если он выше медианы коврика хотя бы на столько уровней серого. Без печати (закрытая
# книга с пустой обложкой) выборка Оцу — один коврик, одна мода, и порог ложится внутрь неё: у IMG_1108 Оцу 29 при
# медиане коврика 25, освещённые части коврика (серое 30–40) уходят в «не фон». Пак-6: у 2473 кадров с печатью Оцу
# 120–140 (выше медианы коврика на ~100), у 7 без печати — не выше медианы + 8.
MAT_OTSU_MIN_GAP = 15.0


def bg_dark_gap(bgr: np.ndarray, content: np.ndarray) -> np.ndarray:
    """``background.bg_dark`` с правилом ``MAT_OTSU_MIN_GAP``: Оцу внутри коврика — порог по хвосту коврика.

    Args:
        bgr: Кадр BGR (масштаб сегментации).
        content: Своя печать (bool).

    Returns:
        Маска фона.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    mat = mat_seeds(bgr)
    paper = cv2.dilate(content.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    samples = np.r_[gray[mat], gray[paper]].astype(np.uint8)
    thr, _ = cv2.threshold(samples.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if mat.any():
        tail = float(np.percentile(gray[mat], MAT_TAIL_PCT)) + MAT_TAIL_MARGIN
        # Оцу ниже медианы коврика + зазор — он делит сам коврик (печати нет): берётся хвост коврика.
        thr = tail if thr < float(np.median(gray[mat])) + MAT_OTSU_MIN_GAP else min(thr, tail)
    bg = (gray < thr) & (chroma(lab) < MAT_MAX_CHROMA)
    return clean_background(bg)


def bg_flood_gap(bgr: np.ndarray, content: np.ndarray) -> np.ndarray:
    """``background.bg_flood`` на пороге ``bg_dark_gap``: водораздел между «коврик у рамки» и «книга».

    Args:
        bgr: Кадр BGR (масштаб сегментации).
        content: Своя печать (bool).

    Returns:
        Маска фона.
    """
    rough = bg_dark_gap(bgr, content)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    sure_bg = cv2.erode(rough.astype(np.uint8), k) > 0
    sure_book = (cv2.erode((~rough).astype(np.uint8), k) > 0) | book_seeds(content)
    markers = np.zeros(rough.shape, np.int32)
    markers[sure_bg] = 1
    markers[sure_book] = 2
    markers = cv2.watershed(cv2.GaussianBlur(bgr, (0, 0), 1.0), markers)
    return clean_background(markers == 1)


@dataclass(frozen=True)
class BookChange:
    """Маска книги кадра до и после правила.

    Поля:
        key: Ключ кадра.
        before: Книга из кэша (bool).
        after: Книга по ``bg_flood_gap`` (bool).
        border_before: Доля периметра кадра, которой касается книга до правила.
        border_after: То же после.
    """

    key: str
    before: np.ndarray
    after: np.ndarray
    border_before: float
    border_after: float


def border_share(mask: np.ndarray) -> float:
    """Доля периметра кадра, которой касается маска."""
    edge = np.r_[mask[0], mask[-1], mask[:, 0], mask[:, -1]]
    return float(edge.mean())


def book_change(cache_dir: Path) -> BookChange | None:
    """Пересчёт книги кадра по кэшу (``seg.png``, ``book.npz``) с правилом ``MAT_OTSU_MIN_GAP``.

    Args:
        cache_dir: Папка кадра в кэше v2.

    Returns:
        До и после или None, если нет файлов.
    """
    if not (cache_dir / "seg.png").exists() or not (cache_dir / "book.npz").exists():
        return None
    img = cv2.imread(str(cache_dir / "seg.png"))
    with np.load(cache_dir / "book.npz") as data:
        content, before = data["content"], data["book"]
    bg = bg_flood_gap(img, content)
    after = book_close_open(bg, content_hulls(content, ~bg))
    return BookChange(cache_dir.name, before, after, border_share(before), border_share(after))
