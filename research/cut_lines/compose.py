"""Вклейка донора в полном разрешении: отображение страницы поднимается с масштаба совмещения, цвет бумаги и сила краски подгоняются под основу."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

# Окно фона бумаги (закрытие — буквы пропадают) и медиана, px полного кадра (1) (~13.8 px/мм).
BG_CLOSE_PX = 31
BG_MEDIAN_PX = 41
# Сглаживание продлённого фона (1) за кромку листа, px полного кадра.
BG_EXTEND_BLUR = 12.0
# Краска для подгонки силы: глубина чернил донора больше этого; допуск множителя силы краски.
INK_DEPTH_MIN = 0.25
INK_GAIN_RANGE = (0.8, 1.8)
# Запас вокруг зоны вклейки при вырезке окна, px полного кадра.
ROI_PAD = 80


@dataclass(frozen=True)
class Composite:
    """Итог вклейки страницы.

    Поля:
        image: Кадр (1) с вклейкой, BGR uint8 полного размера.
        donor: Донор после совмещения и подгонки цвета в окне вклейки (для листов), BGR.
        box: Окно вклейки ``(x0, y0, x1, y1)`` в px полного кадра (1).
        ink_gain: Множитель силы краски донора.
    """

    image: np.ndarray
    donor: np.ndarray
    box: tuple[int, int, int, int]
    ink_gain: float


def paper_background(img: np.ndarray) -> np.ndarray:
    """Гладкий фон бумаги: закрытие окном больше штриха (буквы пропадают) и медиана, по каналам.

    Args:
        img: BGR uint8.

    Returns:
        Фон float32 той же формы.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BG_CLOSE_PX, BG_CLOSE_PX))
    return cv2.medianBlur(cv2.morphologyEx(img, cv2.MORPH_CLOSE, k), BG_MEDIAN_PX).astype(np.float32)


def extend_background(bg: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Продление фона за пределы ``valid``: значение ближайшего надёжного пикселя, затем размытие.

    Args:
        bg: Фон float32 (H, W, 3).
        valid: Где фон надёжен (bool).

    Returns:
        Фон, продлённый на всё окно.
    """
    if not valid.any():
        return bg
    _, (iy, ix) = ndimage.distance_transform_edt(~valid, return_indices=True)
    out = bg[iy, ix]
    smooth = cv2.GaussianBlur(out, (0, 0), BG_EXTEND_BLUR)
    # Внутри надёжной зоны — свой фон, снаружи — сглаженное продление.
    return np.where(valid[..., None], bg, smooth)


def full_map(
    map_seg: np.ndarray, shape: tuple[int, int], box: tuple[int, int, int, int], seg: float, donor_scale: float
):
    """Отображение масштаба совмещения → окно полного кадра (1), координаты в полном кадре (2).

    Args:
        map_seg: Отображение (по x или y) на масштабе совмещения, координаты в уменьшенном (2).
        shape: ``(H, W)`` полного кадра (1).
        box: Окно ``(x0, y0, x1, y1)`` в полном кадре (1).
        seg: Масштаб совмещения относительно (1).
        donor_scale: Масштаб уменьшенного (2) относительно полного (2).

    Returns:
        Карта float32 размера окна.
    """
    x0, y0, x1, y1 = box
    gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    # Значение поля в точке окна: билинейно по карте масштаба совмещения.
    val = cv2.remap(map_seg, gx * seg, gy * seg, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return val / donor_scale


def upsample(mask: np.ndarray, shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    """Маска масштаба совмещения → окно полного кадра (билинейно).

    Args:
        mask: Маска (bool или float).
        shape: ``(H, W)`` полного кадра.
        box: Окно ``(x0, y0, x1, y1)``.

    Returns:
        float32 размера окна.
    """
    x0, y0, x1, y1 = box
    return cv2.resize(mask.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)[y0:y1, x0:x1]


def compose(
    base: np.ndarray,
    donor: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    alpha_seg: np.ndarray,
    paper_seg: np.ndarray,
    cut_seg: np.ndarray,
    clean_seg: np.ndarray,
    seg: float,
    donor_scale: float,
) -> Composite | None:
    """Вклейка донора в окно вокруг альфы: совмещение в полном разрешении, фон бумаги основы, сила краски.

    Цвет: глубина чернил донора ``D = 1 − I/фон донора`` переносится на фон основы, продлённый за кромку листа:
    ``I = фон основы · (1 − k·D)``. Бумага вклейки получает тон соседней бумаги основы, буквы — силу краски
    основы (``k`` — по видимой части срезанных строк).

    Args:
        base: Кадр (1) BGR полного размера.
        donor: Кадр (2) BGR полного размера.
        map_x: Отображение по x на масштабе совмещения (координаты уменьшенного (2)).
        map_y: Отображение по y.
        alpha_seg: Альфа вклейки на масштабе совмещения.
        paper_seg: Видимая бумага (1) на масштабе совмещения.
        cut_seg: Срезанные строки на масштабе совмещения.
        clean_seg: Где краску донора не брать (продление листа вне бумаги донора: торец, коврик) — там чистая
            бумага основы.
        seg: Масштаб совмещения относительно (1).
        donor_scale: Масштаб уменьшенного (2) относительно полного (2).

    Returns:
        Итог или None, если вклеивать нечего.
    """
    ys, xs = np.nonzero(alpha_seg > 0.01)
    if len(xs) == 0:
        return None
    h, w = base.shape[:2]
    x0, y0 = max(0, int(xs.min() / seg) - ROI_PAD), max(0, int(ys.min() / seg) - ROI_PAD)
    x1, y1 = min(w, int(xs.max() / seg) + ROI_PAD), min(h, int(ys.max() / seg) + ROI_PAD)
    box = (x0, y0, x1, y1)
    fx = full_map(map_x, (h, w), box, seg, donor_scale)
    fy = full_map(map_y, (h, w), box, seg, donor_scale)
    warped = cv2.remap(donor, fx, fy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    roi = base[y0:y1, x0:x1]
    size = (x1 - x0, y1 - y0)
    alpha = np.clip(upsample(alpha_seg, (h, w), box), 0, 1)
    paper = upsample(paper_seg, (h, w), box) > 0.5
    cut = upsample(cut_seg, (h, w), box) > 0.5
    # Фон донора и основы; фон основы надёжен в её бумаге с отступом от кромки.
    bg2 = paper_background(warped)
    bg1 = paper_background(roi)
    valid = cv2.erode(paper.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    bg1 = extend_background(bg1, valid)
    depth2 = 1 - warped.astype(np.float32) / np.maximum(bg2, 1)
    depth2[upsample(clean_seg, (h, w), box) > 0.5] = 0.0
    depth1 = 1 - roi.astype(np.float32) / np.maximum(bg1, 1)
    # Сила краски: отношение глубин в видимой части срезанных строк (где краска есть в обоих).
    # Краска для подгонки — во всей видимой части окна вклейки (строки над срезанной тоже), где она есть в обоих.
    ink = (alpha > 0.01) & valid & (depth2.mean(axis=2) > INK_DEPTH_MIN) & (depth1.mean(axis=2) > INK_DEPTH_MIN / 2)
    gain = 1.0
    if ink.sum() > 200:
        gain = float(np.median(depth1[ink].mean(axis=1) / np.maximum(depth2[ink].mean(axis=1), 1e-3)))
    gain = float(np.clip(gain, *INK_GAIN_RANGE))
    patch = np.clip(bg1 * (1 - gain * depth2), 0, 255)
    out = base.copy()
    out[y0:y1, x0:x1] = (roi * (1 - alpha[..., None]) + patch * alpha[..., None]).astype(np.uint8)
    assert out[y0:y1, x0:x1].shape[:2] == (size[1], size[0])
    return Composite(out, patch.astype(np.uint8), box, gain)
