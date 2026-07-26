"""Коррекция теневой зоны вокруг пальца несколькими методами (выбор в CLI).

Все методы правят только ЛОКАЛЬНУЮ зону = ``dilate(finger_mask) ∩ book_mask`` с мягким
feather — так тень у пальца компенсируется, а корешок/центр/виньетка и текст вне зоны не
трогаются (текст внутри зоны сохраняется — правится только освещённость, не контент).

Методы:
  - ``none``            — без коррекции;
  - ``classic``         — аддитивный подъём низкочастотного дефицита освещённости (L-канал);
  - ``retinex``         — мультипликативная gain-коррекция освещённости (гомоморфная/Retinex);
  - ``docshadow-sd7k``  — нейросеть DocShadow (FSENet), веса SD7K (документные тени);
  - ``docshadow-kligler`` / ``docshadow-jung`` — та же сеть, веса других датасетов.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SHADOW_METHODS = ("none", "classic", "retinex", "docshadow-sd7k", "docshadow-kligler", "docshadow-jung")


def parse_dilate_px(value) -> "tuple[int, int]":
    """Парсит ``dilate_px``: число / 'N' → (N, N); (x, y) / 'X,Y' → (X, Y).

    По Y обычно задают вдвое больше, чем по X (тень тянется вдоль края книги): '450,900'.
    """
    if isinstance(value, (tuple, list)):
        return (int(value[0]), int(value[1]))
    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) == 1:
        return (int(parts[0]), int(parts[0]))
    return (int(parts[0]), int(parts[1]))


# ============================================================
# Локализация зоны и смешивание
# ============================================================


def _shadow_band(finger_mask: np.ndarray, book_mask: np.ndarray, dilate_px, feather_sigma: float):
    """Зона-кандидат тени (дилатация пальца ∩ книга) и её мягкая маска смешивания.

    ``dilate_px`` — число (симметрично) либо пара ``(x, y)`` для анизотропной дилатации
    (напр. ``(450, 900)`` — по Y вдвое сильнее, чем по X: тень тянется вдоль края больше).
    """
    dx, dy = (dilate_px, dilate_px) if isinstance(dilate_px, (int, float)) else dilate_px
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(dx) + 1, 2 * int(dy) + 1))
    band = cv2.bitwise_and(cv2.dilate(finger_mask, k), book_mask)
    alpha = cv2.GaussianBlur((band > 0).astype(np.float32), (0, 0), feather_sigma)
    alpha *= (book_mask > 0).astype(np.float32)  # не выходим за книгу даже после feather
    return band, alpha


def _blend(orig: np.ndarray, corrected: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Смешивает исходник и скорректированный кадр по мягкой маске ``alpha`` (0..1)."""
    a = alpha[..., None]
    return np.clip(orig.astype(np.float32) * (1.0 - a) + corrected.astype(np.float32) * a, 0, 255).astype(np.uint8)


def _illumination(L: np.ndarray, illum_k: int) -> np.ndarray:
    """Низкочастотная освещённость L: morph-close (убирает тёмный текст) + сглаживание."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (illum_k, illum_k))
    return cv2.GaussianBlur(cv2.morphologyEx(L, cv2.MORPH_CLOSE, k), (0, 0), 25)


def _paper_ref(illum: np.ndarray, book_mask: np.ndarray, band: np.ndarray) -> float:
    """Робастный опорный уровень чистой бумаги — МЕДИАНА освещённости по бумаге вне зоны тени.

    Медиана (а не высокий перцентиль) устойчива к зональному ПЕРЕсвету, из-за которого
    перцентиль вырождается в 255 (см. отчёт: `ref L=255`).
    """
    inbook = book_mask > 0
    ref_region = inbook & (band == 0)
    src = illum[ref_region] if np.count_nonzero(ref_region) > 1000 else illum[inbook]
    return float(np.median(src)) if src.size else 230.0


# ============================================================
# Классические методы
# ============================================================


def _classic(bgr: np.ndarray, book_mask: np.ndarray, band: np.ndarray, illum_k: int, cap: float) -> np.ndarray:
    """Аддитивный подъём дефицита освещённости: L' = L + clip(ref - illum, 0, cap).

    Дефицит низкочастотный → бумага и текст поднимаются на одну и ту же локальную величину:
    градиент тени исчезает, контраст «текст↔бумага» сохраняется.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    illum = _illumination(L, illum_k)
    ref = _paper_ref(illum, book_mask, band)
    deficit = np.clip(ref - illum, 0, cap)
    lab[:, :, 0] = np.clip(L + deficit, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _retinex(bgr: np.ndarray, book_mask: np.ndarray, band: np.ndarray, illum_k: int, gain_max: float) -> np.ndarray:
    """Мультипликативная gain-коррекция (гомоморфная/Retinex): L' = L * clip(ref/illum, 1, gain_max)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    illum = np.maximum(_illumination(L, illum_k), 1.0)
    ref = _paper_ref(illum, book_mask, band)
    gain = np.clip(ref / illum, 1.0, gain_max)
    lab[:, :, 0] = np.clip(L * gain, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ============================================================
# Диспетчер
# ============================================================


def shadow_variant(method: str) -> "Optional[str]":
    """Вариант весов DocShadow из значения ``--shadow-method`` (``None`` для не-сетевых).

    Нужен ``GpuModels``, чтобы понять, грузить ли DocShadow: ``docshadow-sd7k`` →
    ``"sd7k"``, а ``none``/``classic``/``retinex`` сети не требуют вовсе.
    """
    return method.split("-", 1)[1] if method.startswith("docshadow-") else None


def correct_finger_shadow(
    bgr: np.ndarray,
    finger_mask: Optional[np.ndarray],
    book_mask: np.ndarray,
    method: str,
    models=None,
    dilate_px=(450, 900),
    feather_sigma: float = 60.0,
    illum_k: int = 51,
    cap: float = 45.0,
    gain_max: float = 1.6,
    max_side: int = 2048,
) -> np.ndarray:
    """Корректирует теневую зону вокруг пальца выбранным ``method``.

    Все методы применяются только в feathered-зоне ``dilate(finger_mask) ∩ book_mask``
    (текст и остальной кадр сохраняются). Если пальца нет или method='none' — возвращает
    исходный кадр без изменений.

    ``models`` (``scan_cropping.gpu_models.GpuModels``) нужен только методам
    ``docshadow-*``; ``classic``/``retinex`` считаются на CPU и обходятся без него.
    """
    if method == "none" or finger_mask is None or int(np.count_nonzero(finger_mask)) == 0:
        return bgr
    band, alpha = _shadow_band(finger_mask, book_mask, parse_dilate_px(dilate_px), feather_sigma)
    if int(np.count_nonzero(band)) == 0:
        return bgr
    if method == "classic":
        corrected = _classic(bgr, book_mask, band, illum_k, cap)
    elif method == "retinex":
        corrected = _retinex(bgr, book_mask, band, illum_k, gain_max)
    elif method.startswith("docshadow-"):
        corrected = models.remove_shadow(bgr, max_side=max_side)
    else:
        raise ValueError(f"неизвестный метод коррекции тени: {method}")
    return _blend(bgr, corrected, alpha)
