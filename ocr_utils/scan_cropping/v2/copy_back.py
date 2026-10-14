"""Защита контента при закрасе пальцев: блоки surya, задетые зоной закраса, возвращаются с оригинала только внутри грубой маски книги."""

from __future__ import annotations

import numpy as np

from ocr_utils.scan_cropping.finger_removal.text_protection import intersecting_polygons, polygons_to_mask

# Припуск блоков surya, px калибровочного кадра (по X, по Y; пересчитывается по ``Scale.frame_px``) — один и тот же для
# copy back в зоне пальца и для защиты контента от вырезки (решение пользователя 2026-10-14: на оверлее рисуется ровно
# зона, которую возвращает copy back). По X 18 px ≈ 1.3 мм — как в боевом профиле МТС v1. По Y 36 px ≈ 2.6 мм: хвосты
# букв и полстроки; 72 v1 подбирались под тень пальца (её теперь снимает карта тени, ``finger_shadow``), и полоса 5 мм
# под блоком возвращала с оригинала кончик пальца (МТС 1990/01 IMG_0842) и торец под последней строкой.
LAYOUT_PAD_PX = (18, 36)


def copy_back_blocks(
    original: np.ndarray,
    clean: np.ndarray,
    polygons: list[np.ndarray],
    paint: np.ndarray,
    book: np.ndarray,
    pad_px: int | tuple[int, int] = LAYOUT_PAD_PX,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Возвращает с оригинала пиксели блоков surya, задетых зоной закраса пальцев, — только внутри книги.

    Как ``copy-back-layout-zones`` v1: блок, хоть сколько-нибудь пересекающийся с зоной закраса (с припуском
    ``pad_px``), копируется с оригинала жёстко, в том числе поверх ядра пальца (настоящий палец входит с края по чистой
    бумаге и в блок не попадает). Отличие: копируется только часть блока внутри грубой маски книги (не фон и
    не руки). Куски блоков, вылезающие за книгу — на коврик, на палец, на руку у края кадра, — остаются закрашенными.

    Args:
        original: Исходный кадр (с пальцами), BGR uint8.
        clean: Кадр после закраса пальцев того же размера.
        polygons: Полигоны блоков surya в координатах кадра (``text_protection.layout_polygons``).
        paint: Зона закраса пальцев (uint8 0/255 или bool) размера кадра.
        book: Где возврат разрешён (bool) размера кадра: весь «не фон» (по умолчанию, ``--copy-back-over-fingers``) или
            «не фон» без рук.
        pad_px: Припуск блоков, px кадра: скаляр или пара (по X, по Y).

    Returns:
        ``(кадр, маска возвращённого bool, число задетых блоков)``.
    """
    paint_u8 = (paint > 0).astype(np.uint8) * 255
    empty = np.zeros(paint.shape[:2], bool)
    if not polygons or not paint_u8.any():
        return clean, empty, 0
    # Блоки, задетые зоной закраса.
    hits = intersecting_polygons(polygons, paint_u8, pad_px)
    if not hits:
        return clean, empty, 0
    # Область возврата: блоки с припуском ∩ книга. Вне зоны закраса оригинал и так совпадает с закрасом,
    # поэтому ограничивать ещё и зоной закраса не нужно — но маска «возвращено» считается только по ней.
    region = (polygons_to_mask(paint.shape, hits, pad_px) > 0) & book
    out = clean.copy()
    out[region] = original[region]
    return out, region & (paint_u8 > 0), len(hits)
