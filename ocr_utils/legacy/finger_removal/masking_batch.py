"""Батчевая детекция пальцев и отладочный оверлей маски (помойка).

Вынесено из ``finger_removal/masking.py``, когда рабочий пайплайн
(``ocr_utils.scan_cropping``) перешёл на покадровую обработку: батч по нескольким
кадрам сразу использовался только самостоятельной CLI ``legacy.finger_removal.cli``.
Отличия от покадровой ветки: YOLO-World прогоняется одним ``predict`` по списку
кадров, дилатация круговая (без асимметрии), нет ``fill_holes``-таймингов и
``mask_predilate`` для оверлея.
"""

import cv2
import numpy as np

from ocr_utils.scan_cropping.finger_removal.masking import (
    MAX_FINGER_AREA_FRAC,
    _select_finger_boxes,
    fill_holes,
    keep_border_components,
    skin_edge_mask,
)


def neural_hand_mask_batch(
    rgb_list: "list[np.ndarray]",
    models,
    conf: float = 0.05,
    max_box_frac: float = 0.30,
    max_area_frac: float = MAX_FINGER_AREA_FRAC,
    containment_thresh: float = 0.8,
) -> "list[np.ndarray]":
    """Батчевая версия ``neural_hand_mask``. Возвращает список масок uint8 0/255.

    ``models`` — ``ocr_utils.scan_cropping.gpu_models.GpuModels``. Батчевого метода
    у класса нет (рабочему пайплайну он не нужен), поэтому кадры прогоняются по
    одному — выигрыш батча здесь потерян, но код остаётся рабочим.
    """
    masks = []
    for rgb in rgb_list:
        h, w = rgb.shape[:2]
        img_area = h * w
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        boxes, confs, cls = models.detect_hand_boxes(bgr, conf=conf)
        if len(boxes) > 0:
            bw = boxes[:, 2] - boxes[:, 0]
            bh = boxes[:, 3] - boxes[:, 1]
            keep = (bw * bh) <= (max_box_frac * img_area)
            boxes, confs, cls = boxes[keep], confs[keep], cls[keep]
        if len(boxes) > 0:
            boxes = _select_finger_boxes(boxes, confs, cls, containment_thresh)
        if len(boxes) == 0:
            masks.append(np.zeros((h, w), dtype=np.uint8))
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        for m_bin in models.segment_boxes(bgr, boxes):
            if m_bin.sum() > max_area_frac * img_area:
                continue
            mask[m_bin] = 255
        masks.append(mask)
    return masks


def build_finger_mask_batch(
    rgb_list: "list[np.ndarray]",
    models,
    method: str = "auto",
    edge_frac: float = 0.12,
    dilate_px: int = 12,
    min_area_frac: float = 0.0015,
    conf: float = 0.05,
) -> "list[tuple[np.ndarray, str]]":
    """Батчевая версия ``build_finger_mask``. Возвращает список (mask, info)."""
    if not rgb_list:
        return []

    results = []
    if method == "skin":
        for rgb in rgb_list:
            mask = skin_edge_mask(rgb, edge_frac=edge_frac, min_area_frac=min_area_frac)
            results.append((mask, method))
    elif method in ("neural", "auto"):
        neural_masks = neural_hand_mask_batch(rgb_list, models, conf=conf)
        for rgb, nm in zip(rgb_list, neural_masks):
            h, w = rgb.shape[:2]
            if method == "neural":
                mask = keep_border_components(nm, edge_frac=edge_frac, min_area_frac=min_area_frac)
                info = method
            elif int(np.count_nonzero(nm)) == 0:
                mask = np.zeros((h, w), dtype=np.uint8)
                info = "auto(пусто)"
            else:
                mask, info = nm, "auto(neural)"

            if int(np.count_nonzero(mask)) > 0:
                mask = fill_holes(mask)
                if dilate_px > 0:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
                    mask = cv2.dilate(mask, kernel, iterations=1)
            results.append((mask, info))
    else:
        raise ValueError(f"Неизвестный метод маскирования: {method}")
    return results


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Возвращает RGB с полупрозрачной красной заливкой по маске (для отладки)."""
    out = rgb.copy()
    red = np.zeros_like(rgb)
    red[:, :, 0] = 255
    m = mask > 0
    out[m] = (alpha * red[m] + (1.0 - alpha) * rgb[m]).astype(np.uint8)
    # Контур маски для наглядности
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 3)
    return out
