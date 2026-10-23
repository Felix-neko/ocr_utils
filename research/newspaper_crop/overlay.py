"""Оверлеи зоны копирования: лист, четырёхугольник копирования, дорисовка, отрезанное квантилем; шапка и легенда в полях."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed
from research.newspaper_crop.zone import Zone, quad_mask

# Цвета BGR (палитра проекта, навык draw-overlay): не бумага — красный «отвергнуто», четырёхугольник
# копирования — синий «граница блока», дорисовка — оранжевый, лист за четырёхугольником — фиолетовый.
COLOUR_NOT_PAPER = (0, 0, 220)
COLOUR_QUAD = (220, 90, 20)
COLOUR_FILL = (0, 165, 255)
COLOUR_CUT = (200, 60, 200)
COLOUR_FINGER = (180, 105, 255)
# Складка корешка — зелёный «принято» палитры (линия, по которой отрезано).
COLOUR_FOLD = (0, 150, 0)
# Трапеция листа до отрезки — фиолетовый «крупная огибающая» палитры, пунктиром.
COLOUR_DETECTED = (200, 60, 200)
AREA_ALPHA = 0.45


def _tint(canvas: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int], alpha: float) -> None:
    """Полупрозрачно закрашивает область маски на холсте (на месте).

    Args:
        canvas: BGR-холст.
        mask: Булева маска того же размера.
        colour: Цвет BGR.
        alpha: Непрозрачность.
    """
    if not mask.any():
        return
    canvas[mask] = (canvas[mask] * (1 - alpha) + np.array(colour) * alpha).astype(np.uint8)


def zone_overlay(
    bgr: np.ndarray,
    zone: Zone,
    header: list[str],
    finger: np.ndarray | None = None,
    max_height: int = 2208,
    folds: list[np.ndarray] | None = None,
    detected_quad: np.ndarray | None = None,
) -> np.ndarray:
    """Оверлей зоны поверх кадра на рабочем масштабе.

    Args:
        bgr: Кадр BGR (любой масштаб — приводится к размеру маски зоны).
        zone: Зона кадра.
        header: Строки шапки.
        finger: Маска пальцев на рабочем масштабе или None.
        max_height: Наибольшая высота картинки (ужимается с сохранением пропорций).
        folds: Принятые складки корешка — ломаные в пикселях рабочей маски — или None.
        detected_quad: Трапеция листа, найденная до отрезки за складкой (пиксели маски), — рисуется пунктиром
            под зоной копирования; None — не рисовать.

    Returns:
        BGR-картинка с шапкой и легендой в полях.
    """
    h, w = zone.paper.shape
    canvas = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    qm = quad_mask(zone.quad, (h, w)) if zone.quad is not None else np.zeros((h, w), bool)
    _tint(canvas, ~zone.paper & ~qm, COLOUR_NOT_PAPER, AREA_ALPHA)
    _tint(canvas, qm & ~zone.paper, COLOUR_FILL, AREA_ALPHA)
    _tint(canvas, zone.paper & ~qm, COLOUR_CUT, AREA_ALPHA)
    if finger is not None:
        contours, _ = cv2.findContours(finger.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, COLOUR_FINGER, 2)
    for line in folds or []:
        cv2.polylines(canvas, [np.round(line).astype(np.int32)], False, COLOUR_FOLD, 2)
    if detected_quad is not None:
        _dashed_poly(canvas, detected_quad, COLOUR_DETECTED, 2)
    if zone.quad is not None:
        cv2.polylines(canvas, [np.round(zone.quad).astype(np.int32)], True, COLOUR_QUAD, 3)
    if h > max_height:
        canvas = cv2.resize(canvas, (int(w * max_height / h), max_height), interpolation=cv2.INTER_AREA)
    legend = [
        LegendEntry("не бумага (вне четырёхугольника)", COLOUR_NOT_PAPER, AREA_ALPHA, SampleStyle.BOX),
        LegendEntry("дорисовать: в четырёхугольнике, но не бумага", COLOUR_FILL, AREA_ALPHA, SampleStyle.BOX),
        LegendEntry("бумага за четырёхугольником (отрезано)", COLOUR_CUT, AREA_ALPHA, SampleStyle.BOX),
        LegendEntry("зона копирования" if detected_quad is not None else "четырёхугольник копирования", COLOUR_QUAD),
    ]
    if detected_quad is not None:
        legend.append(LegendEntry("трапеция листа (до отрезки за складкой)", COLOUR_DETECTED, style=SampleStyle.DASHED))
    if finger is not None:
        legend.append(LegendEntry("палец (детектор)", COLOUR_FINGER))
    if folds:
        legend.append(LegendEntry("складка корешка: за ней — соседняя полоса, отрезано", COLOUR_FOLD))
    return framed(canvas, header, legend)


def _dashed_poly(
    canvas: np.ndarray, poly: np.ndarray, colour: tuple[int, int, int], thickness: int, dash: int = 18
) -> None:
    """Замкнутый многоугольник пунктиром (на месте).

    Args:
        canvas: BGR-холст.
        poly: Вершины (N, 2).
        colour: Цвет BGR.
        thickness: Толщина, px.
        dash: Длина штриха и пропуска, px.
    """
    for i in range(len(poly)):
        a, b = np.asarray(poly[i], float), np.asarray(poly[(i + 1) % len(poly)], float)
        length = float(np.linalg.norm(b - a))
        steps = max(int(length // dash), 1)
        for k in range(0, steps, 2):
            p = a + (b - a) * (k / steps)
            q = a + (b - a) * (min(k + 1, steps) / steps)
            cv2.line(canvas, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), colour, thickness)


# Оверлей классификатора «свой лист / чужое»: участки — от зелёного «принято» (свой лист) к красному «отвергнуто»
# (чужое) по вероятности; эталон пользователя — чёрный с белой обводкой (читается и на бумаге, и на тени); граница
# по классификатору — розовый «линия отсечки»; отрезка конвейера (складка и защита печати) — оранжевый пунктир;
# трапеция листа до правки — синий «граница блока».
COLOUR_OWN = (0, 150, 0)
COLOUR_OTHER = (0, 0, 220)
COLOUR_GT = (20, 20, 20)
COLOUR_CLF = (180, 105, 255)
COLOUR_PIPELINE = (0, 165, 255)
# Непрозрачность заливки участков.
CROP_ALPHA = 0.4


def surface_overlay(
    bgr: np.ndarray,
    crops: pd.DataFrame,
    quad: np.ndarray,
    final_quad: np.ndarray | None,
    gt: pd.DataFrame,
    boundaries: dict[str, tuple[float, float]],
    header: list[str],
    crop_along: int,
    zoom_sides: list[str],
    max_height: int = 1800,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Оверлей классификатора у сторон листа: участки по вероятности «чужого», эталон, граница по классификатору,
    отрезка конвейера; плюс вырезки крупно вдоль выбранных сторон.

    Каждый участок закрашивается полосой в шаг глубин :data:`surface_clf.DEPTHS` вокруг своего центра (участки по
    глубине перекрываются, полоса в шаг — сплошная и без наложений) цветом от «своего» к «чужому» по вероятности ``p``.

    Args:
        bgr: Превью кадра BGR (пиксели превью — те же, что у трапеций и эталона).
        crops: Участки кадра (``side, u, depth, p``; эталонные участки ``ref`` не рисуются).
        quad: Трапеция листа до правки (вершины (4, 2), px превью), от неё отсчитаны участки.
        final_quad: Трапеция конвейера после отрезки по складке и охвата печати, или None.
        gt: Эталон кадра (``side, x0, y0, x1, y1``).
        boundaries: ``{сторона: (k, c)}`` — граница по классификатору как глубина ``k·u + c`` от стороны;
            (0, 0) — сторона не сдвигается (не рисуется).
        header: Строки шапки.
        crop_along: Длина участка вдоль стороны, px превью.
        zoom_sides: Стороны, для которых нужна вырезка крупно.
        max_height: Высота картинки кадра целиком, px.

    Returns:
        (кадр целиком со шапкой и легендой, ``{сторона: вырезка со шапкой и легендой}``).
    """
    from research.newspaper_crop.surface_clf import DEPTHS, side_frames

    canvas = bgr.copy()
    frames = side_frames(quad)
    step = DEPTHS[1] - DEPTHS[0]
    layer = canvas.copy()
    # Участки: полоса в шаг глубин вокруг центра, цвет — смесь «своего» и «чужого» по p.
    for r in crops[~crops.ref].itertuples():
        f = frames[r.side]
        du = np.array([-crop_along / 2, crop_along / 2, crop_along / 2, -crop_along / 2]) + r.u
        dv = np.array([-1, -1, 1, 1]) * step / 2 + r.depth
        pts = f.p + du[:, None] * f.along + dv[:, None] * f.inward
        colour = tuple(int(a * (1 - r.p) + b * r.p) for a, b in zip(COLOUR_OWN, COLOUR_OTHER))
        cv2.fillPoly(layer, [np.round(pts).astype(np.int32)], colour)
    cv2.addWeighted(layer, CROP_ALPHA, canvas, 1 - CROP_ALPHA, 0, canvas)
    # Трапеция до правки — тонко, конвейер — пунктиром.
    cv2.polylines(canvas, [np.round(quad).astype(np.int32)], True, COLOUR_QUAD, 5)
    if final_quad is not None:
        _dashed_poly(canvas, final_quad, COLOUR_PIPELINE, 7, dash=40)
    # Эталон: чёрная линия с белой обводкой (под границей классификатора: совпадение видно по обводке).
    for r in gt.itertuples():
        a, b = (int(round(r.x0)), int(round(r.y0))), (int(round(r.x1)), int(round(r.y1)))
        cv2.line(canvas, a, b, (255, 255, 255), 13)
        cv2.line(canvas, a, b, COLOUR_GT, 6)
    # Граница по классификатору.
    for side, (k, c) in boundaries.items():
        if k == 0 and c == 0:
            continue
        f = frames[side]
        u = np.array([0.0, f.length])
        v = np.maximum(k * u + c, 0.0)
        a, b = f.p + u[0] * f.along + v[0] * f.inward, f.p + u[1] * f.along + v[1] * f.inward
        cv2.line(canvas, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), COLOUR_CLF, 7)
    legend = [
        LegendEntry("участок: свой лист (p≈0)", COLOUR_OWN, alpha=CROP_ALPHA, style=SampleStyle.BOX),
        LegendEntry("участок: чужое (p≈1)", COLOUR_OTHER, alpha=CROP_ALPHA, style=SampleStyle.BOX),
        LegendEntry("эталон пользователя (CVAT)", COLOUR_GT),
        LegendEntry("граница по классификатору", COLOUR_CLF),
        LegendEntry("трапеция листа (SAM2.1), от неё участки", COLOUR_QUAD),
        LegendEntry("конвейер: складка + охват печати", COLOUR_PIPELINE, style=SampleStyle.DASHED),
    ]
    h, w = canvas.shape[:2]
    scale = max_height / h
    whole = framed(cv2.resize(canvas, (round(w * scale), max_height), interpolation=cv2.INTER_AREA), header, legend)
    # Вырезки: прямоугольник вокруг полосы стороны (от 120 px снаружи до 320 px внутрь), повёрнутый вдоль стороны.
    zooms = {}
    for side in zoom_sides:
        f = frames[side]
        out_h, depth0 = 440, -120
        out_w = int(f.length)
        uu, vv = np.meshgrid(np.arange(out_w, dtype=np.float32), np.arange(out_h, dtype=np.float32) + depth0)
        mx = f.p[0] + uu * f.along[0] + vv * f.inward[0]
        my = f.p[1] + uu * f.along[1] + vv * f.inward[1]
        strip = cv2.remap(canvas, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))  # fmt: skip
        # Базис (вдоль, внутрь) с отрицательным определителем даёт зеркальную картинку — отражаем обратно.
        if f.along[0] * f.inward[1] - f.along[1] * f.inward[0] < 0:
            strip = strip[:, ::-1]
        # Длинную полосу режем на две строки, чтобы вырезка была читаемой по ширине.
        half = (out_w + 1) // 2
        left, right = strip[:, :half], strip[:, half:]
        right = np.pad(right, ((0, 0), (0, half - right.shape[1]), (0, 0)), constant_values=255)
        sheet = np.vstack([left, np.full((12, half, 3), 255, np.uint8), right])
        zooms[side] = framed(sheet, header + [f"сторона {side}: полоса вдоль стороны, снаружи сверху (−120…+320 px)"],
                             legend)  # fmt: skip
    return whole, zooms
