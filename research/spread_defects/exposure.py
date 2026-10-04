"""Пересвет: бледная и истончённая краска по плиткам кадра — толщина штриха, светлота краски, доля краски.

Что видно на примерах пользователя (1990/06 IMG_0067 правая полоса, 1981/11 IMG_0046): буквы тоньше и светлее,
тонкие элементы «съедены» светом, у соседних кадров того же выпуска тот же набор жирнее и темнее. Выбитая в 255
бумага признаком не служит — её 18–40 % почти на любом кадре пака (замер 2026-10-03). Поэтому мера — по
краске: толщина штриха ``2·площадь/периметр`` маски краски на уровне половины отражения бумаги и светлота
краски (медиана отражения пикселей краски) по текстовым плиткам. Сравнение — с медианой выпуска: печать,
гарнитура и бумага в выпуске одни, а свет от кадра к кадру меняется.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from research.spread_defects.motion import TILE, line_share, tile_ink

# Уровень бумаги: максимум в окне (больше межстрочья) и сглаживание, px полного кадра.
PAPER_WINDOW, PAPER_SIGMA = 41, 15
# Краска — пиксели с отражением ниже этого (доля уровня бумаги).
INK_REFL = 0.55
# Плитка годится, если пикселей краски в ней не меньше и прямых линий мало.
MIN_INK_PIXELS, MAX_LINES = 1500, 0.05


@dataclass(frozen=True)
class ExposureTiles:
    """Сетка признаков пересвета по плиткам ``TILE`` кадра.

    Attributes:
        text: bool[gy, gx] — текстовая плитка (``motion.tile_ink``, без линеек и рамок).
        stroke: float32[gy, gx] — толщина штриха, px (2·площадь/периметр маски краски).
        refl: float32[gy, gx] — медиана отражения пикселей краски (0 — чёрная, 1 — как бумага).
        cover: float32[gy, gx] — доля краски в плитке на уровне ``INK_REFL``.
        paper: float32[gy, gx] — медиана уровня бумаги в плитке, 0..255.
    """

    text: np.ndarray
    stroke: np.ndarray
    refl: np.ndarray
    cover: np.ndarray
    paper: np.ndarray


def exposure_tiles(rgb: np.ndarray) -> ExposureTiles:
    """Признаки пересвета по плиткам кадра полного разрешения.

    Args:
        rgb: Кадр H×W×3 uint8.

    Returns:
        ``ExposureTiles`` с сеткой ``H // TILE`` × ``W // TILE`` (та же, что у ``motion_tiles``).
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    paper = cv2.dilate(gray, np.ones((PAPER_WINDOW, PAPER_WINDOW), np.uint8)).astype(np.float32)
    paper = cv2.GaussianBlur(paper, (0, 0), PAPER_SIGMA)
    refl = gray.astype(np.float32) / np.maximum(paper, 1.0)
    ink = (refl < INK_REFL).astype(np.uint8)
    # Граница маски — пиксели краски, у которых есть сосед-не-краска (4-связность).
    border = ink & (1 - cv2.erode(ink, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)))
    gy, gx = gray.shape[0] // TILE, gray.shape[1] // TILE
    text = np.zeros((gy, gx), bool)
    stroke, light, cover, level = (np.zeros((gy, gx), np.float32) for _ in range(4))
    for row in range(gy):
        for col in range(gx):
            ys, xs = slice(row * TILE, (row + 1) * TILE), slice(col * TILE, (col + 1) * TILE)
            if tile_ink(hsv[ys, xs]) is None:
                continue
            value = hsv[ys, xs][..., 2]
            if line_share(value, float(np.percentile(value, 90))) > MAX_LINES:
                continue
            mask = ink[ys, xs].astype(bool)
            area = int(mask.sum())
            if area < MIN_INK_PIXELS:
                continue
            text[row, col] = True
            stroke[row, col] = 2.0 * area / max(int(border[ys, xs].sum()), 1)
            light[row, col] = float(np.median(refl[ys, xs][mask]))
            cover[row, col] = area / float(TILE * TILE)
            level[row, col] = float(np.median(paper[ys, xs]))
    return ExposureTiles(text, stroke, light, cover, level)


def side_summary(tiles: ExposureTiles, fold_x: float) -> dict:
    """Сводка пересвета по полосам и четвертям высоты.

    Args:
        tiles: Сетка кадра.
        fold_x: x сгиба посередине высоты, px полного кадра (делит полосы).

    Returns:
        Словарь: для полос ``L``/``R`` — медианы толщины штриха, светлоты и доли краски по полосе
        (``<сторона>_stroke`` и т. д.), худшая четверть по толщине (``<сторона>_stroke_q``) и число плиток.
    """
    out = {}
    gy, gx = tiles.text.shape
    centers = (np.arange(gx) + 0.5) * TILE
    for side, cols in (("L", centers < fold_x), ("R", centers > fold_x)):
        mask = tiles.text & cols[None, :]
        out[f"{side}_tiles"] = int(mask.sum())
        if mask.sum() < 8:
            continue
        out[f"{side}_stroke"] = float(np.median(tiles.stroke[mask]))
        out[f"{side}_refl"] = float(np.median(tiles.refl[mask]))
        out[f"{side}_cover"] = float(np.median(tiles.cover[mask]))
        quarters = []
        for q in range(4):
            rows = np.zeros(gy, bool)
            rows[q * gy // 4 : (q + 1) * gy // 4] = True
            part = mask & rows[:, None]
            if part.sum() >= 4:
                quarters.append(float(np.median(tiles.stroke[part])))
        out[f"{side}_stroke_q"] = float(min(quarters)) if quarters else np.nan
    return out
