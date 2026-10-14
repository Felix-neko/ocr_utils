"""Фон (тёмный коврик) вокруг книги водоразделом и маска рук по коже; книга = не фон (грубая маска книги)."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

# Масштаб сегментации относительно кадра подшивок МТО 1990: ½ (5875×4570 → 2938×2285, ~6.9 px/мм). Все размеры
# в px ниже — на этом масштабе; для других кадров масштаб сегментации подбирается так, чтобы px/мм совпали
# (``v2.scale``).
BG_SCALE = 0.5
# Полоса у рамки кадра, откуда берутся образцы коврика, px масштаба сегментации.
BORDER_PX = 40
# Порог «тёмного» для образцов коврика (серое): отсекает в полосе у рамки руки и край книги.
MAT_SAMPLE_MAX_GRAY = 70
# Предел хромы коврика (Lab, 8 бит OpenCV): коврик почти бесцветный, кожа и голубая обложка — нет.
MAT_MAX_CHROMA = 18.0
# Порог яркости фона: перцентиль серого коврика у рамки и запас над ним.
MAT_TAIL_PCT = 99.5
MAT_TAIL_MARGIN = 20.0
# Минимальная площадь дыры в фоне, которая остаётся дырой (меньше — заливается фоном), px².
MIN_HOLE_PX = 2000
# Окно закрытия маски фона, px.
CLOSE_PX = 5
# Кожа в YCrCb: диапазоны Cr и Cb (классика для светлой кожи при дневном свете) и минимальная яркость.
SKIN_CR = (137, 185)
# Разность Cr − Cb кожи не меньше этого (замер 06/IMG_1296: кожа p50 26, бумага p95 17).
SKIN_MIN_CR_MINUS_CB = 19
SKIN_CB = (80, 130)
SKIN_MIN_Y = 60
# Раздутие ядер пальцев при построении маски рук, px.
HAND_CORE_DILATE_PX = 8


def border_band(shape: tuple[int, int], width: int = BORDER_PX) -> np.ndarray:
    """Полоса шириной ``width`` вдоль рамки кадра.

    Args:
        shape: Размер ``(H, W)``.
        width: Ширина, px.

    Returns:
        Булева маска.
    """
    band = np.zeros(shape, bool)
    band[:width] = band[-width:] = True
    band[:, :width] = band[:, -width:] = True
    return band


def chroma(lab: np.ndarray) -> np.ndarray:
    """Хрома Lab (8 бит OpenCV): расстояние (a, b) от нейтрали.

    Args:
        lab: Картинка Lab uint8.

    Returns:
        Карта float32.
    """
    return np.hypot(lab[..., 1].astype(np.float32) - 128, lab[..., 2].astype(np.float32) - 128)


def mat_seeds(bgr: np.ndarray) -> np.ndarray:
    """Уверенный коврик: тёмные бесцветные пиксели полосы у рамки.

    Args:
        bgr: Кадр BGR.

    Returns:
        Булева маска затравок фона.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return border_band(gray.shape) & (gray < MAT_SAMPLE_MAX_GRAY) & (chroma(lab) < MAT_MAX_CHROMA)


def book_seeds(content: np.ndarray) -> np.ndarray:
    """Уверенная книга: своя печать (защита глифов), чуть ужатая.

    Args:
        content: Своя печать (bool) в масштабе сегментации.

    Returns:
        Булева маска затравок книги.
    """
    return cv2.erode(content.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def hand_mask(bgr: np.ndarray, finger_core: np.ndarray) -> np.ndarray:
    """Руки: кожа по цвету (YCrCb), связная с ядрами пальцев SAM или с рамкой кадра, плюс сами ядра.

    Args:
        bgr: Кадр BGR (исходный, с пальцами).
        finger_core: Ядра пальцев SAM до дилатации (bool) в масштабе сегментации.

    Returns:
        Булева маска рук.
    """
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    cr16, cb16 = cr.astype(np.int16), cb.astype(np.int16)
    skin = (cr >= SKIN_CR[0]) & (cr <= SKIN_CR[1]) & (cb >= SKIN_CB[0]) & (cb <= SKIN_CB[1]) & (y >= SKIN_MIN_Y)
    # Жёлтая бумага по Cr близка к коже; отделяет разность Cr − Cb (кожа 26, бумага ≤ 17; 06/IMG_1296).
    skin &= (cr16 - cb16) >= SKIN_MIN_CR_MINUS_CB
    skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    # Ядра пальцев чуть раздуваются: кромка пальца по цвету может не пройти порог кожи.
    k = 2 * HAND_CORE_DILATE_PX + 1
    core = cv2.dilate(finger_core.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    # Рука — компонента «кожа или ядро», задевающая ядро пальца или рамку кадра (рука входит с края снимка).
    labels, _ = ndimage.label(skin | core)
    anchors = core | border_band(skin.shape, 3)
    keep = np.unique(labels[anchors & (labels > 0)])
    return np.isin(labels, keep[keep > 0])


def clean_background(bg: np.ndarray) -> np.ndarray:
    """Чистка маски фона: закрытие, только компоненты, связные с рамкой, мелкие дыры (блики, пылинки) — в фон.

    Args:
        bg: Сырая маска фона.

    Returns:
        Булева маска фона.
    """
    k = 2 * CLOSE_PX + 1
    bg = cv2.morphologyEx(bg.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    labels, _ = ndimage.label(bg > 0)
    # Фон — компоненты, задевающие полосу образцов коврика у рамки (BORDER_PX), а не только саму рамку.
    keep = np.unique(labels[border_band(bg.shape, BORDER_PX) & (labels > 0)])
    bg = np.isin(labels, keep[keep > 0])
    # Мелкие «не фон» внутри фона (пылинки, блики коврика) — тоже фон.
    labels, count = ndimage.label(~bg)
    if count:
        sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, count + 1))
        small = np.r_[False, sizes < MIN_HOLE_PX]
        bg |= small[labels]
    return bg


def bg_dark(bgr: np.ndarray, content: np.ndarray) -> np.ndarray:
    """Фон порогом: серое ниже порога Оцу между образцами коврика и печатью-бумагой, хрома мала.

    Порог считается по гистограмме объединения затравок коврика и бумаги вокруг печати, а не всего кадра: так
    тени на странице и светлый торец не сдвигают его.

    Args:
        bgr: Кадр BGR.
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
    # Оцу между ковриком и бумагой (~130) отдаёт фону серо-сиреневый край обложки (~100–120, 06/IMG_1296):
    # порог не выше верхнего хвоста самого коврика плюс запас.
    if mat.any():
        thr = min(thr, float(np.percentile(gray[mat], MAT_TAIL_PCT)) + MAT_TAIL_MARGIN)
    bg = (gray < thr) & (chroma(lab) < MAT_MAX_CHROMA)
    return clean_background(bg)


def bg_flood(bgr: np.ndarray, content: np.ndarray) -> np.ndarray:
    """Фон водоразделом: маркеры «коврик у рамки» и «печать», граница — по самому сильному перепаду между ними.

    Маркер книги дополняется всем, что заметно светлее порога ``bg_dark`` (бумага, обрез), маркер фона — тёмным
    у рамки; неразмеченная полоса вокруг их границы отдаётся водоразделу.

    Args:
        bgr: Кадр BGR (масштаб сегментации).
        content: Своя печать (bool).

    Returns:
        Маска фона.
    """
    rough = bg_dark(bgr, content)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    sure_bg = cv2.erode(rough.astype(np.uint8), k) > 0
    sure_book = (cv2.erode((~rough).astype(np.uint8), k) > 0) | book_seeds(content)
    markers = np.zeros(rough.shape, np.int32)
    markers[sure_bg] = 1
    markers[sure_book] = 2
    blurred = cv2.GaussianBlur(bgr, (0, 0), 1.0)
    markers = cv2.watershed(blurred, markers)
    # Линии водораздела (−1) — отдаём книге, чтобы не терять кромку.
    return clean_background(markers == 1)


def book_from_background(bg: np.ndarray, hands: np.ndarray) -> np.ndarray:
    """Книга = не фон − руки; остаются компоненты не меньше 5% крупнейшей, дыры залиты.

    Args:
        bg: Маска фона.
        hands: Маска рук.

    Returns:
        Булева маска книги.
    """
    book = ~bg & ~hands
    book = cv2.morphologyEx(book.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    labels, count = ndimage.label(book)
    if count == 0:
        return book
    sizes = ndimage.sum(book, labels, index=range(1, count + 1))
    keep = np.r_[False, sizes >= 0.05 * sizes.max()]
    return ndimage.binary_fill_holes(keep[labels])
