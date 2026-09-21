"""Кромки фотографий по каждому снимку в рамке; наклон — по снимку целиком.

Две правки к ``raster.py`` ядра по разбору 2026-09-22:

* рамка растра от ``page_layout`` бывает одна на два снимка (1969/12 с.99: два фото, оба
  довернуты FineReader'ом на ~2°), а ядро берёт одну компоненту краски с наибольшим
  перекрытием и кромки общей рамки не находит — ``raster_edges`` 0. Здесь внутри рамки
  берутся все компоненты замкнутой краски площадью от ``MIN_COMPONENT_FRAC`` рамки, и кромки
  меряются по каждой;
* наклон снимка — среднее ухода по найденным кромкам A минус то же в B: трапеция, ставшая
  параллелограммом того же уровня (1971/10 с.66: правая кромка +2.5 мм, левая на столько же
  выпрямилась), по худшей кромке была порчей 2.5 мм, а по снимку целиком — ноль. Худшая
  кромка остаётся как контекст; изгиб кромки — по-прежнему максимум.
"""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.geometry_regression import mm_to_px
from ocr_utils.geometry_regression.field import Field
from ocr_utils.geometry_regression.raster import Edge, _box_in_a, _edge_profile, _ink_mask, _polyline

Box = tuple[int, int, int, int]

# Компонента краски внутри рамки растра меньше этой доли площади рамки — не снимок (подпись, пыль).
MIN_COMPONENT_FRAC = 0.15
# Припуск рамки компоненты: кромка на бинарном рендере осыпана, bbox компоненты её чуть режет.
COMPONENT_PAD_MM = 2.0


def photo_boxes(mask: np.ndarray, box: Box, dpi: float) -> list[tuple[Box, np.ndarray]]:
    """Снимки внутри рамки: рамка компоненты (с припуском, обрезана по кадру) и её маска."""
    h, w = mask.shape
    x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
    if x1 <= x0 or y1 <= y0:
        return []
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    inside = labels[y0:y1, x0:x1]
    areas = np.bincount(inside.ravel(), minlength=count)
    areas[0] = 0
    min_area = MIN_COMPONENT_FRAC * (x1 - x0) * (y1 - y0)
    pad = mm_to_px(COMPONENT_PAD_MM, dpi)
    out: list[tuple[Box, np.ndarray]] = []
    for label in np.flatnonzero(areas >= min_area):
        cx, cy, cw, ch = stats[label, :4]
        photo = (max(0, int(cx) - pad), max(0, int(cy) - pad), min(w, int(cx + cw) + pad), min(h, int(cy + ch) + pad))
        out.append((photo, (labels == label).astype(np.uint8)))
    return out


def _component_in_a(mask_a: np.ndarray, box_a: Box) -> np.ndarray:
    """Компонента A, больше всех перекрывающая рамку снимка (как ``_photo_component`` ядра)."""
    count, labels = cv2.connectedComponents(mask_a, connectivity=8)
    x0, y0, x1, y1 = box_a
    inside = labels[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
    if inside.size == 0 or count <= 1:
        return np.zeros_like(mask_a)
    hist = np.bincount(inside.ravel(), minlength=count)
    hist[0] = 0
    return (labels == int(hist.argmax())).astype(np.uint8)


def raster_edge_metrics(
    gray_b: np.ndarray, gray_a: np.ndarray, raster_b: list[Box], warp: Field | None, dpi: float
) -> tuple[dict[str, float], dict]:
    """Порча кромок фотографий: изгиб (максимум по кромкам) и наклон (среднее по снимку) A − B.

    Args:
        gray_b, gray_a: Серые рабочие копии обеих версий одного dpi.
        raster_b: Рамки растра на B в пикселях ``dpi``.
        warp: Поле смещений B → A или ``None``.
        dpi: Разрешение копий.

    Returns:
        ``raster_edge_bend_mm`` — рост сагитты худшей кромки; ``raster_photo_tilt_mm`` — рост
        среднего ухода кромок от оси по худшему снимку; ``raster_edge_tilt_mm`` — рост по худшей
        кромке (контекст); ``raster_edges``, ``raster_photos``; виновники с профилями кромок.
    """
    metrics = {
        "raster_edge_bend_mm": 0.0,
        "raster_photo_tilt_mm": 0.0,
        "raster_edge_tilt_mm": 0.0,
        "raster_edges": 0.0,
        "raster_photos": 0.0,
    }
    culprits: dict = {}
    if not raster_b:
        return metrics, culprits
    mask_b = _ink_mask(gray_b, dpi)
    mask_a = _ink_mask(gray_a, dpi)
    for box in raster_b:
        for photo, component_b in photo_boxes(mask_b, box, dpi):
            photo_a = _box_in_a(photo, warp, gray_a.shape)
            component_a = _component_in_a(mask_a, photo_a)
            edges: list[tuple[Edge, Edge]] = []
            for side in ("top", "bottom", "left", "right"):
                edge_b = _edge_profile(component_b, photo, side, dpi)
                edge_a = _edge_profile(component_a, photo_a, side, dpi)
                if edge_b is not None and edge_a is not None:
                    edges.append((edge_b, edge_a))
            if not edges:
                continue
            metrics["raster_photos"] += 1.0
            metrics["raster_edges"] += float(len(edges))
            tilt_delta = float(np.mean([a.tilt_mm for _, a in edges]) - np.mean([b.tilt_mm for b, _ in edges]))
            worst_tilt = max(edges, key=lambda pair: pair[1].tilt_mm - pair[0].tilt_mm)
            if tilt_delta > metrics["raster_photo_tilt_mm"]:
                metrics["raster_photo_tilt_mm"] = tilt_delta
                culprits["raster_photo_tilt_mm"] = {"b": photo, "a": photo_a, **_polyline(*worst_tilt)}
            for edge_b, edge_a in edges:
                for name, delta in (
                    ("raster_edge_bend_mm", edge_a.sag_mm - edge_b.sag_mm),
                    ("raster_edge_tilt_mm", edge_a.tilt_mm - edge_b.tilt_mm),
                ):
                    if delta > metrics[name]:
                        metrics[name] = float(delta)
                        culprits[name] = {"b": photo, "a": photo_a, **_polyline(edge_b, edge_a)}
    return metrics, culprits


__all__ = ["photo_boxes", "raster_edge_metrics"]
