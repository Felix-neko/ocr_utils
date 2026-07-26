"""Геометрия ROI вокруг маски пальца и растушёванное вклеивание результата.

Чистый CPU-модуль: сам инпейнтинг (LaMa) живёт в ``GpuModels.inpaint``, а здесь —
то, что ему нужно вокруг сети и что удобно тестировать без GPU.

Почему инпейнтинг идёт «по ROI», а не по кадру. Палец всегда входит с края книги,
где в кадре доминирует ЧЁРНЫЙ фон. Если гнать сеть по всему снимку 5696×4272,
LaMa «затягивает» дыру этим доминирующим чёрным. В ТЕСНОМ ROI вокруг маски (с
контекстным полем ``padding``, растянутым в ``roi_scale`` раз) сеть видит
локальный контекст — кромку переплёта, поле страницы — и достраивает именно его.
Результат вклеивается обратно ТОЛЬКО внутри маски с растушёвкой краёв, поэтому
остальная часть кадра не трогается, а шов незаметен.

ROI считается ПОКОМПОНЕНТНО: несколько разнесённых пальцев иначе слились бы в
один гигантский ROI на всю полосу кадра, и вся выгода от тесного контекста пропала бы.
"""

from typing import Optional

import cv2
import numpy as np


# ROI вокруг маски увеличиваем в 1.5 раза — сети нужен контекст кромки/фона, иначе
# дыра «заливается» доминирующим цветом.
DEFAULT_ROI_SCALE = 1.5


def mask_bbox(mask: np.ndarray) -> "Optional[tuple[int, int, int, int]]":
    """Возвращает (x1, y1, x2, y2) — bbox ненулевых пикселей маски, либо None."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def roi_bounds(
    mask: np.ndarray, padding: int, roi_scale: float, shape: "tuple[int, int]"
) -> "Optional[tuple[int, int, int, int]]":
    """ROI вокруг маски: bbox + ``padding``, затем масштаб ``roi_scale`` от центра.

    Итоговый прямоугольник обрезается границами кадра ``shape`` (h, w). Возвращает
    (x1, y1, x2, y2) или None, если маска пустая.
    """
    bbox = mask_bbox(mask)
    if bbox is None:
        return None
    h, w = shape
    x1, y1, x2, y2 = bbox
    # Поле контекста
    x1, y1, x2, y2 = x1 - padding, y1 - padding, x2 + padding, y2 + padding
    # Масштаб вокруг центра
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1) * roi_scale, (y2 - y1) * roi_scale
    x1, x2 = int(round(cx - bw / 2)), int(round(cx + bw / 2))
    y1, y2 = int(round(cy - bh / 2)), int(round(cy + bh / 2))
    # Обрезаем по кадру
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def mask_components(mask: np.ndarray) -> "list[np.ndarray]":
    """Разбивает маску на отдельные связные компоненты (список масок uint8 0/255)."""
    num, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return [((labels == i).astype(np.uint8) * 255) for i in range(1, num)]


def roi_bounds_list(
    mask: np.ndarray, padding: int = 64, roi_scale: float = DEFAULT_ROI_SCALE
) -> "list[tuple[int, int, int, int]]":
    """ROI каждой связной компоненты маски (список (x1, y1, x2, y2)) — для отладки.

    Покомпонентно, чтобы несколько разнесённых пальцев не сливались в один
    гигантский ROI на всю полосу кадра.
    """
    rois = []
    for comp in mask_components(mask):
        b = roi_bounds(comp, padding, roi_scale, mask.shape[:2])
        if b is not None:
            rois.append(b)
    return rois


def blend_roi(orig: np.ndarray, filled: np.ndarray, mask: np.ndarray, feather: int) -> np.ndarray:
    """Вклеивает ``filled`` в ``orig`` по маске с мягким спадом краёв (alpha=1 внутри)."""
    m = (mask > 0).astype(np.float32)
    if feather > 0:
        k = 2 * feather + 1
        a = cv2.GaussianBlur(m, (k, k), 0)
        a = np.maximum(a, m)  # внутри маски заполнение всегда полное
    else:
        a = m
    a = a[..., None]
    return (a * filled.astype(np.float32) + (1.0 - a) * orig.astype(np.float32)).astype(np.uint8)
