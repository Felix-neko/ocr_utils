"""Защита контента при закрасе пальцев: блоки surya, задетые зоной закраса, возвращаются с оригинала только внутри грубой маски книги."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.scan_cropping.finger_removal.text_protection import intersecting_polygons, polygons_to_mask

# Припуск блоков surya, px калибровочного кадра (по X, по Y; пересчитывается по ``Scale.frame_px``) — один и тот же для
# copy back в зоне пальца и для защиты контента от вырезки (решение пользователя 2026-10-14: на оверлее рисуется ровно
# зона, которую возвращает copy back). По X 18 px ≈ 1.3 мм — как в боевом профиле МТС v1. По Y 36 px ≈ 2.6 мм: хвосты
# букв и полстроки; 72 v1 подбирались под тень пальца (её теперь снимает карта тени, ``finger_shadow``), и полоса 5 мм
# под блоком возвращала с оригинала кончик пальца (МТС 1990/01 IMG_0842) и торец под последней строкой.
LAYOUT_PAD_PX = (18, 36)
# Классы блоков surya, которые при copy back возвращаются целиком, в том числе поверх ядра пальца: на рисунках и фото
# детектор пальцев ошибается (лицо на портрете — «кожа»). Остальные классы — только зона дилатации пальца в блоке.
FULL_COPY_LABELS = frozenset({"Picture", "Figure"})
# Припуск ядра пальца, px калибровочного кадра (пересчёт через ``Scale.frame_px``): зона дилатации, которую copy back
# возвращает с оригинала, отстоит от ядра SAM 3 на столько. Ядро чуть уже пальца, и без припуска с оригинала
# возвращалась розовая кайма кожи у кончика (МТС 1990/04 IMG_1197). Замер: превышение a* над бумагой у края ядра
# 6 → 5 → 4 на 1, 3, 5 px; решение пользователя 2026-10-15 — 4 px (кайма сильно слабеет, буквы у пальца на месте;
# стенд research/finger_repaint core-margin, 53 кадра: a* в 1–3 px за ядром 6 → 0).
CORE_GROW_PX = 4


def copy_back_blocks(
    original: np.ndarray,
    clean: np.ndarray,
    polygons: list[np.ndarray],
    paint: np.ndarray,
    book: np.ndarray,
    pad_px: int | tuple[int, int] = LAYOUT_PAD_PX,
    labels: list[str] | None = None,
    core: np.ndarray | None = None,
    core_grow_px: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Возвращает с оригинала пиксели блоков surya, задетых зоной закраса пальцев, — только внутри книги.

    Блок задет, если он (с припуском ``pad_px``) хоть сколько-нибудь пересекается с зоной закраса. Что возвращается —
    по классу блока (решение пользователя 2026-10-14):

    - ``FULL_COPY_LABELS`` (Picture, Figure) — блок с припуском целиком, в том числе поверх ядра пальца: детектор
      пальцев принимает напечатанное лицо или фото рук за кожу, и такой ложный «палец» затирал портреты (МТС 1967/12
      IMG_0149, 1975/10 IMG_0011);
    - остальные классы (текст, заголовки, таблицы, колонтитулы…) — только зона дилатации пальца внутри блока: зона
      закраса без ядра. Текст с пальцем детектор путает редко, а ядро на тексте — настоящий палец: возвращать его — значит
      вернуть в итог кожу (на pack6_v3 ядро ≥ 100 мм² в итоге у 723 кадров).

    Без типов блоков (кэш ``layout.json`` версии 3) или без ядра — прежнее поведение: все задетые блоки целиком.
    В любом случае возврат только внутри ``book`` (весь «не фон» или «не фон» без рук, см. ``--copy-back-over-fingers``).

    Args:
        original: Исходный кадр (с пальцами), BGR uint8.
        clean: Кадр после закраса пальцев того же размера.
        polygons: Полигоны блоков surya в координатах кадра (``text_protection.layout_polygons``).
        paint: Зона закраса пальцев (uint8 0/255 или bool) размера кадра.
        book: Где возврат разрешён (bool) размера кадра: весь «не фон» (по умолчанию, ``--copy-back-over-fingers``) или
            «не фон» без рук.
        pad_px: Припуск блоков, px кадра: скаляр или пара (по X, по Y).
        labels: Типы блоков surya по порядку ``polygons`` или None.
        core: Ядро пальца (bool) размера кадра или None.
        core_grow_px: Припуск ядра, px кадра: из возврата блоков «только зона дилатации» вычитается ядро, раздутое на
            столько (0 — само ядро).

    Returns:
        ``(кадр, маска возвращённого bool, число задетых блоков)``.
    """
    paint_b = paint > 0
    paint_u8 = paint_b.astype(np.uint8) * 255
    empty = np.zeros(paint.shape[:2], bool)
    if not polygons or not paint_u8.any():
        return clean, empty, 0
    # Блоки, задетые зоной закраса, — порознь «целиком» и «только зона дилатации пальца».
    by_class = labels is not None and core is not None and len(labels) == len(polygons)
    full, ring = [], []
    for k, poly in enumerate(polygons):
        if not intersecting_polygons([poly], paint_u8, pad_px):
            continue
        (full if not by_class or labels[k] in FULL_COPY_LABELS else ring).append(poly)
    if not full and not ring:
        return clean, empty, 0
    # Область возврата: целые блоки с припуском ∩ книга; у остальных — ещё ∩ зона закраса без ядра. Вне зоны закраса
    # оригинал и так совпадает с закрасом, поэтому целые блоки зоной закраса не ограничиваются.
    region = np.zeros(paint.shape[:2], bool)
    if full:
        region |= polygons_to_mask(paint.shape, full, pad_px) > 0
    if ring:
        core_b = core > 0
        if core_grow_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * core_grow_px + 1, 2 * core_grow_px + 1))
            core_b = cv2.dilate(core_b.astype(np.uint8), k) > 0
        region |= (polygons_to_mask(paint.shape, ring, pad_px) > 0) & paint_b & ~core_b
    region &= book
    out = clean.copy()
    out[region] = original[region]
    return out, region & paint_b, len(full) + len(ring)
