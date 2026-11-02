"""Проба порога краски для кандидатов: глобальный Оцу (как у разбора, ``segment.component_mask``) против Саволы с окном в мм — какая доля пикселей размеченных компонент остаётся краской и сколько компонент распадается."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.filters import threshold_sauvola

from ocr_utils.page_layout.text_blocks import RENDER_DPI
from ocr_utils.units import mm_to_px
from research.pseudo_glyphs.candidates import components_of

# Окна Саволы, мм бумаги (при 300 dpi: 1.5 мм ≈ 17 px, 3 мм ≈ 35 px, 6 мм ≈ 71 px), и коэффициент k.
WINDOWS_MM = (1.5, 3.0, 6.0)
SAUVOLA_K = 0.2


def sauvola_ink(gray300: np.ndarray, window_mm: float, k: float = SAUVOLA_K) -> np.ndarray:
    """Маска краски по Саволе.

    Args:
        gray300: Серый 300 dpi.
        window_mm: Окно, мм (нечётное число пикселей).
        k: Коэффициент Саволы.

    Returns:
        Булева маска «краска».
    """
    size = mm_to_px(window_mm, RENDER_DPI)
    size += 1 - size % 2
    return gray300 < threshold_sauvola(gray300, window_size=size, k=k)


def page_rows(gray300: np.ndarray, part: pd.DataFrame) -> list[dict]:
    """Меры размеченных компонент одной страницы при Саволе с разными окнами.

    Args:
        gray300: Серый 300 dpi страницы (как у разбора).
        part: Размеченные кандидаты страницы (``comp``, ``x0..y1``, ``label``, ``group``).

    Returns:
        Строки: класс, окно, доля пикселей компоненты Оцу, оставшихся краской, число кусков, на которые она распалась.
    """
    labels = components_of(gray300).labels
    rows = []
    for window in WINDOWS_MM:
        ink = sauvola_ink(gray300, window)
        for r in part.itertuples():
            mask = labels[r.y0 : r.y1, r.x0 : r.x1] == r.comp
            kept = ink[r.y0 : r.y1, r.x0 : r.x1] & mask
            pieces = cv2.connectedComponents(kept.astype(np.uint8), connectivity=8)[0] - 1
            rows.append({"label": r.label, "group": r.group, "window_mm": window,
                         "kept": float(kept.sum()) / max(int(mask.sum()), 1), "pieces": int(pieces)})  # fmt: skip
    return rows


def compare(frame: pd.DataFrame, gray_dir: Path) -> pd.DataFrame:
    """Сводка по классам: медиана доли оставшихся пикселей и доля компонент, распавшихся больше чем на 2 куска.

    Args:
        frame: Размеченные кандидаты (``evaluate.labelled``).
        gray_dir: Папка серого 300 dpi (``<ключ>.png``).

    Returns:
        Таблица «класс × окно».
    """
    rows = []
    for key, part in frame.groupby("key"):
        gray = cv2.imread(str(gray_dir / f"{key}.png"), cv2.IMREAD_GRAYSCALE)
        rows.extend(page_rows(gray, part))
    table = pd.DataFrame(rows)
    return table.groupby(["window_mm", "label"]).agg(
        n=("kept", "size"),
        kept_median=("kept", "median"),
        vanished=("kept", lambda s: float((s < 0.2).mean())),
        split=("pieces", lambda s: float((s > 2).mean())),
    )


__all__ = ["compare", "page_rows", "sauvola_ink"]
