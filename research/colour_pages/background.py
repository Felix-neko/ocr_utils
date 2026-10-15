"""Фон на коврике у кадра без печати (МТС 1990/04 IMG_1108): пересчёт маски книги по кэшу правилом пакета ``background.MAT_OTSU_MIN_GAP`` и сравнение с кэшем."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ocr_utils.scan_cropping.v2.background import bg_dark, bg_flood
from ocr_utils.scan_cropping.v2.book import book_close_open, content_hulls

# Правило перенесено в пакет (``background.MAT_OTSU_MIN_GAP``, ``bg_dark``); здесь — прежние имена стенда.
bg_dark_gap = bg_dark
bg_flood_gap = bg_flood


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
