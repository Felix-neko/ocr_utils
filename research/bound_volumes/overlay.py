"""Оверлеи маски верхних страниц: книга SAM2, отсечённые торцы и крышка, барьер слоёв, врезки нормалей и глубины."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed
from research.bound_volumes.layers import Layers

# Палитра (BGR): верхние страницы — контур синий, отсечённое от книги — красная заливка, барьер — оранжевый.
COLOUR_TOP = (220, 90, 20)
COLOUR_CUT = (0, 0, 220)
COLOUR_BARRIER = (0, 165, 255)
COLOUR_SEED = (40, 170, 40)
COLOUR_GT = (180, 105, 255)
AREA_ALPHA = 0.45


def normal_rgb(normal: np.ndarray) -> np.ndarray:
    """Нормали в цвет (x, y, z → R, G, B), BGR uint8.

    Args:
        normal: Нормали ``(H, W, 3)``.

    Returns:
        Картинка BGR.
    """
    return ((np.clip(normal, -1, 1) * 0.5 + 0.5) * 255).astype(np.uint8)[..., ::-1].copy()


def depth_rgb(depth: np.ndarray) -> np.ndarray:
    """Глубина в цвет (turbo по 2–98 перцентилям), BGR uint8.

    Args:
        depth: Глубина с NaN.

    Returns:
        Картинка BGR.
    """
    lo, hi = np.nanpercentile(depth, 2), np.nanpercentile(depth, 98)
    norm = np.clip((np.nan_to_num(depth, nan=hi) - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def layers_overlay(bgr_work: np.ndarray, layers: Layers, title: str, gt: np.ndarray | None = None) -> np.ndarray:
    """Оверлей движка слоёв: кадр с разметкой и рядом — нормали, глубина, барьер на уменьшенных врезках.

    Args:
        bgr_work: Кадр BGR рабочего масштаба (без пальцев).
        layers: Итог движка.
        title: Шапка (кадр, движок, числа).
        gt: Эталон верхних страниц (bool) или None.

    Returns:
        Картинка BGR со шапкой и легендой в полях.
    """
    canvas = bgr_work.copy()
    layer = canvas.copy()
    cut = layers.book & ~layers.top
    layer[cut] = COLOUR_CUT
    layer[layers.barrier & layers.book] = COLOUR_BARRIER
    cv2.addWeighted(layer, AREA_ALPHA, canvas, 1 - AREA_ALPHA, 0, canvas)
    contours, _ = cv2.findContours(layers.top.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(canvas, contours, -1, COLOUR_TOP, 2)
    seeds, _ = cv2.findContours(layers.seed.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, seeds, -1, COLOUR_SEED, 1)
    if gt is not None:
        gtc, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, gtc, -1, COLOUR_GT, 1)
    h, w = canvas.shape[:2]
    small = (w // 2, h // 2)
    barrier_vis = np.full((h, w, 3), 255, np.uint8)
    barrier_vis[layers.book] = (225, 225, 225)
    barrier_vis[layers.barrier] = COLOUR_BARRIER
    side = np.vstack(
        [
            cv2.resize(normal_rgb_safe(layers, bgr_work), small, interpolation=cv2.INTER_AREA),
            cv2.resize(barrier_vis, small, interpolation=cv2.INTER_NEAREST),
        ]
    )
    pad = h - side.shape[0]
    side = np.pad(side, ((0, max(pad, 0)), (0, 0), (0, 0)), constant_values=255)[:h]
    legend = [
        LegendEntry("верхние страницы (итог)", COLOUR_TOP),
        LegendEntry("отсечено от книги SAM2 (торцы, крышка)", COLOUR_CUT, alpha=AREA_ALPHA, style=SampleStyle.BOX),
        LegendEntry(
            "барьер: ступенька глубины / излом нормалей", COLOUR_BARRIER, alpha=AREA_ALPHA, style=SampleStyle.BOX
        ),
        LegendEntry("затравка роста", COLOUR_SEED),
    ]
    if gt is not None:
        legend.append(LegendEntry("эталон CVAT", COLOUR_GT))
    return framed(
        np.hstack([canvas, np.full((h, 6, 3), 255, np.uint8), side]),
        [title, "справа: нормали MoGe-2; барьер на книге"],
        legend,
    )


def normal_rgb_safe(layers: Layers, bgr_work: np.ndarray) -> np.ndarray:
    """Врезка излома нормалей (если нормалей нет в итоге — серый кадр).

    Args:
        layers: Итог движка (карта излома).
        bgr_work: Кадр рабочего масштаба.

    Returns:
        Картинка BGR: излом нормалей в оттенках серого, тёмное — сильный излом.
    """
    bend = np.clip(layers.bend / 20.0, 0, 1)
    gray = (255 * (1 - bend)).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def raycut_overlay(bgr_work: np.ndarray, cut, title: str, gt: np.ndarray | None = None) -> np.ndarray:
    """Оверлей движка лучей: книга SAM2, срезанное от неё, итог, точки кромки по лучам; справа — излом нормалей.

    Args:
        bgr_work: Кадр BGR рабочего масштаба (без пальцев).
        cut: Итог ``layers.RayCut``.
        title: Шапка.
        gt: Эталон (bool) или None.

    Returns:
        Картинка BGR со шапкой и легендой в полях.
    """
    canvas = bgr_work.copy()
    layer = canvas.copy()
    layer[cut.book & ~cut.top] = COLOUR_CUT
    cv2.addWeighted(layer, AREA_ALPHA, canvas, 1 - AREA_ALPHA, 0, canvas)
    contours, _ = cv2.findContours(cut.top.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(canvas, contours, -1, COLOUR_TOP, 2)
    # Сырые пики по лучам — точки; где пика нет — не рисуется.
    for (x, y), (nx, ny), d in zip(cut.points, cut.normals, cut.raw_depth):
        if d > 0:
            cv2.circle(canvas, (int(x + nx * d), int(y + ny * d)), 1, COLOUR_BARRIER, -1)
    if gt is not None:
        gtc, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, gtc, -1, COLOUR_GT, 1)
    h, w = canvas.shape[:2]
    bend = (255 * (1 - np.clip(cut.bend / 8.0, 0, 1))).astype(np.uint8)
    side = cv2.resize(cv2.cvtColor(bend, cv2.COLOR_GRAY2BGR), (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    side = np.pad(side, ((0, h - side.shape[0]), (0, 0), (0, 0)), constant_values=255)
    legend = [
        LegendEntry("верхние страницы (итог)", COLOUR_TOP),
        LegendEntry("срезано от книги SAM2", COLOUR_CUT, alpha=AREA_ALPHA, style=SampleStyle.BOX),
        LegendEntry("самый внутренний пик излома по лучу", COLOUR_BARRIER),
    ]
    if gt is not None:
        legend.append(LegendEntry("эталон CVAT", COLOUR_GT))
    body = np.hstack([canvas, np.full((h, 6, 3), 255, np.uint8), side])
    return framed(body, [title, "справа: излом нормалей MoGe-2 (тёмное — сильнее)"], legend)


# Оверлей фона (BGR): фон — тёмно-синий, руки — розовый, спорное — оранжевый, контур книги — синий.
COLOUR_BG = (120, 40, 0)
COLOUR_HAND = (180, 105, 255)
COLOUR_DISPUTE = (0, 165, 255)
COLOUR_BOOK = (220, 90, 20)
BG_ALPHA = 0.5


def background_overlay(
    bgr: np.ndarray, bg: np.ndarray, hands: np.ndarray, book: np.ndarray, title: list[str], dispute: np.ndarray | None = None
) -> np.ndarray:
    """Оверлей сегментации фона: фон, руки и спорные пиксели полупрозрачно, контур книги линией; шапка и легенда в полях.

    Args:
        bgr: Кадр BGR (масштаб сегментации).
        bg: Маска фона.
        hands: Маска рук.
        book: Маска книги (не фон − руки).
        title: Строки шапки.
        dispute: Спорные пиксели (способы не согласны) или None.

    Returns:
        Картинка BGR.
    """
    canvas = bgr.copy()
    layer = canvas.copy()
    layer[bg] = COLOUR_BG
    layer[hands] = COLOUR_HAND
    if dispute is not None:
        layer[dispute] = COLOUR_DISPUTE
    cv2.addWeighted(layer, BG_ALPHA, canvas, 1 - BG_ALPHA, 0, canvas)
    contours, _ = cv2.findContours(book.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(canvas, contours, -1, COLOUR_BOOK, 2)
    legend = [
        LegendEntry("фон (коврик)", COLOUR_BG, alpha=BG_ALPHA, style=SampleStyle.BOX),
        LegendEntry("руки", COLOUR_HAND, alpha=BG_ALPHA, style=SampleStyle.BOX),
        LegendEntry("граница книги (не фон − руки)", COLOUR_BOOK),
    ]
    if dispute is not None:
        legend.append(LegendEntry("спорно: способы не согласны", COLOUR_DISPUTE, alpha=BG_ALPHA, style=SampleStyle.BOX))
    return framed(canvas, title, legend)
