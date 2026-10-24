"""Меры стенда: IoU рамки с эталоном, промах каждой стороны в мм со знаком (наружу/внутрь), сопоставление эталона с находками, сериализация фото в JSON."""

from __future__ import annotations

import numpy as np

from research.photo_bounds.bounds import PhotoBound, Side, SideFit

Box = tuple[float, float, float, float]

# Порядок сторон в мерах и CSV.
SIDES = (Side.TOP, Side.BOTTOM, Side.LEFT, Side.RIGHT)


def iou(a: Box, b: Box) -> float:
    """Пересечение над объединением двух рамок ``(x0, y0, x1, y1)``.

    Args:
        a, b: Рамки.

    Returns:
        IoU 0..1.
    """
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def side_deltas_mm(box: Box, ref: Box, mm_per_px: float) -> dict[Side, float]:
    """Промах сторон рамки против эталона, мм со знаком: плюс — сторона снаружи эталона (лишнее), минус — внутри
    (срезано).

    Args:
        box: Рамка.
        ref: Эталон.
        mm_per_px: мм на px.

    Returns:
        Промах по сторонам.
    """
    return {
        Side.TOP: (ref[1] - box[1]) * mm_per_px,
        Side.BOTTOM: (box[3] - ref[3]) * mm_per_px,
        Side.LEFT: (ref[0] - box[0]) * mm_per_px,
        Side.RIGHT: (box[2] - ref[2]) * mm_per_px,
    }


def best_match(ref: Box, boxes: list[Box]) -> tuple[int, float]:
    """Находка с наибольшим IoU против эталона.

    Args:
        ref: Эталон.
        boxes: Находки.

    Returns:
        ``(индекс, IoU)``; ``(-1, 0.0)`` — находок нет или ни одна не пересекает эталон.
    """
    best, best_iou = -1, 0.0
    for i, b in enumerate(boxes):
        v = iou(ref, b)
        if v > best_iou:
            best, best_iou = i, v
    return best, best_iou


def scale_box(box, factor: float) -> Box:
    """Рамка × коэффициент (смена разрешения)."""
    return tuple(float(v) * factor for v in box)


def side_json(fit: SideFit) -> dict:
    """Подгонка стороны → JSON (точки, годность, коэффициенты, меры)."""
    return {
        "side": fit.side.value,
        "along": np.round(fit.along, 2).tolist(),
        "across": np.round(fit.across, 2).tolist(),
        "contrast": np.round(fit.contrast, 1).tolist(),
        "inlier": fit.inlier.astype(int).tolist(),
        "coef": None if fit.coef is None else fit.coef.to_json(),
        "span": [float(fit.span[0]), float(fit.span[1])],
        "samples": fit.samples,
        "cover": round(fit.cover, 3),
        "resid_p90_mm": round(fit.resid_p90_mm, 3) if np.isfinite(fit.resid_p90_mm) else None,
        "resid_rms_mm": round(fit.resid_rms_mm, 3) if np.isfinite(fit.resid_rms_mm) else None,
        "contrast_median": round(fit.contrast_median, 1),
        "width_median_mm": round(fit.width_median_mm, 3) if np.isfinite(fit.width_median_mm) else None,
        "bow_mm": round(fit.bow_mm, 3),
    }


def photo_json(photo: PhotoBound) -> dict:
    """Фото → JSON."""
    return {
        "box": [round(v, 2) for v in photo.box],
        "mask_box": [int(v) for v in photo.mask_box],
        "seed": [int(v) for v in photo.seed],
        "edge_class": photo.edge_class.value,
        "sharp_sides": [bool(v) for v in photo.sharp_sides],
        "quad_reason": photo.quad_reason,
        "outside_mm2": photo.outside_mm2,
        "corners": None if photo.corners is None else np.round(photo.corners, 2).tolist(),
        "polygon": np.round(photo.polygon(), 1).tolist(),
        "sides": [side_json(s) for s in photo.sides],
    }


# Полоса у стороны рамки для мер тесноты, мм: от ``BAND_NEAR_MM`` до ``BAND_FAR_MM`` внутрь и наружу.
BAND_NEAR_MM = 0.3
BAND_FAR_MM = 1.3
# Сторона «срезает снимок», если снаружи полоса покрыта светлым растром больше чем на ``CUT_COVER``; «с лишним полем»,
# если внутри покрыта меньше чем на ``SPARE_COVER``.
CUT_COVER = 0.5
SPARE_COVER = 0.5


def band_cover(light: np.ndarray, origin: tuple[int, int], box: Box, side: Side, inner: bool, dpi: float) -> float:
    """Доля светлого растра в полосе у стороны рамки (внутри или снаружи) по всей длине стороны.

    Мера тесноты рамки без эталона: у рамки точно по снимку полоса внутри — растр (~1), снаружи — бумага (~0).

    Args:
        light: Светлый растр окна (``bounds.light_map``).
        origin: Начало окна на полосе ``(x, y)``.
        box: Рамка, px полосы.
        side: Сторона.
        inner: Полоса внутри рамки (иначе — снаружи).
        dpi: Разрешение.

    Returns:
        Доля 0..1; ``nan`` — полоса вне окна.
    """
    near, far = BAND_NEAR_MM * dpi / 25.4, BAND_FAR_MM * dpi / 25.4
    ox, oy = origin
    x0, y0, x1, y1 = box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy
    sign = -1 if inner else 1
    if side is Side.TOP:
        a, b = sorted((y0 - sign * near, y0 - sign * far))
        part = light[int(max(a, 0)) : int(max(b, 0)), int(max(x0, 0)) : int(max(x1, 0))]
    elif side is Side.BOTTOM:
        a, b = sorted((y1 + sign * near, y1 + sign * far))
        part = light[int(max(a, 0)) : int(max(b, 0)), int(max(x0, 0)) : int(max(x1, 0))]
    elif side is Side.LEFT:
        a, b = sorted((x0 - sign * near, x0 - sign * far))
        part = light[int(max(y0, 0)) : int(max(y1, 0)), int(max(a, 0)) : int(max(b, 0))]
    else:
        a, b = sorted((x1 + sign * near, x1 + sign * far))
        part = light[int(max(y0, 0)) : int(max(y1, 0)), int(max(a, 0)) : int(max(b, 0))]
    return float(part.mean()) if part.size else float("nan")
