"""Кромки фотографии «было | стало»: погнул ли FineReader край растра.

Фотография на бинарном рендере — россыпь точек, и обычные метрики к ней неприменимы: LSD
находит в ней случайные «штрихи», строки текста в ней нет, а тайлы поля смещений остаются без
пары и у целой фотографии (растровая сетка: 1970/12 с.76, 1975/04 с.2 — половина тайлов без
пары при целом снимке). Что видит глаз, когда FineReader портит снимок, — его кромка: ровный
край становится волной (1970/10 с.71, 1971/07 с.43 — верхняя кромка) или прямоугольник —
трапецией (1971/04 с.44). Поэтому мерятся четыре кромки каждой растровой области: маска краски
замыкается на ``CLOSE_MM`` (точки растра сливаются в сплошное пятно), вдоль кромки берётся
профиль — для верхней кромки первая строка краски в каждом столбце, и так далее, — к профилю
подгоняется прямая, и берутся её сагитта (размах остатка между ``SAG_PERCENTILES``, мм) и
наклон к оси (в мм ухода конца кромки: длина × sin). Метрика — разность A − B: кромка,
кривая уже в B (снимок вклеен криво), не считается порчей.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.geometry_regression import mm_to_px, px_to_mm
from ocr_utils.geometry_regression.field import Field

Box = tuple[int, int, int, int]

# Замыкание маски краски: точки растра в 1–1.5 мм друг от друга сливаются в пятно; подпись под
# снимком стоит дальше 2 мм и в пятно не входит.
CLOSE_MM = 1.2
# Припуск вокруг рамки, в котором ищется кромка: рамка ``page_layout`` лежит на кромке с
# точностью до клетки растрового детектора, а в A погнутая кромка уходит от прогноза поля на
# миллиметры (1971/07 с.43: верхняя кромка в A задрана на 13 мм). Строка текста в припуске не
# страшна: кромка ищется по компоненте снимка (``_photo_component``).
PAD_MM = 15.0
# Края кромки не мерятся: углы снимка на бинарном рендере скруглены и осыпаны.
EDGE_MARGIN_FRAC = 0.08
# Кромка короче не мерится: сагитта короткой кромки тонет в осыпи точек.
MIN_EDGE_MM = 30.0
# Профиль должен найтись хотя бы на этой доле столбцов (строк): иначе кромки нет (снимок
# без чёткого края, виньетка).
MIN_COVERAGE = 0.7
# Сагитта — размах остатка от прямой между этими перцентилями (одиночный выброс не задирает).
SAG_PERCENTILES = (2.0, 98.0)
# Кромка непрерывна: между соседними столбцами профиль (после медианного фильтра) меняется на
# сотые миллиметра даже у погнутой кромки. Скачок больше — профиль ушёл внутрь снимка по
# светлому месту (небо, белая стена у края кадра: 1970/10 с.71, правая кромка) или наружу, на
# строку текста, прижатую к снимку (та же страница, верхняя кромка в A). Профиль режется по
# скачкам, берётся самый длинный непрерывный кусок; короче ``MIN_COVERAGE`` — кромки нет.
# Плечо — MEDIAN_WIDTH столбцов (4 мм при 150 dpi): 3 мм на плече — наклон ~37°; перекошенная
# FineReader'ом кромка доходит до 25° (1971/04 с.44, нижняя кромка второго снимка), буква под
# 60° даёт 7 мм. Медиана той же ширины гасит осыпь кромки (недостающие точки растра у края —
# ступеньки в 1 мм на 1–2 мм длины).
MAX_STEP_MM = 3.0
MEDIAN_WIDTH = 25
# Робастная подгонка прямой: точки дальше стольких MAD от первой прямой отбрасываются.
OUTLIER_MADS = 4.0


@dataclass(frozen=True)
class Edge:
    """Одна кромка снимка: сагитта и наклон к оси (мм), координаты профиля (пиксели копии)."""

    sag_mm: float
    tilt_mm: float
    points: np.ndarray  # N × 2, (x, y) точек профиля


def _ink_mask(gray: np.ndarray, dpi: float) -> np.ndarray:
    """Маска краски с замкнутыми точками растра: снимок — одно сплошное пятно."""
    ink = (gray < 128).astype(np.uint8)
    k = max(3, mm_to_px(CLOSE_MM, dpi) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)


def _median_filter(values: np.ndarray, width: int) -> np.ndarray:
    """Скользящая медиана нечётной ширины с повтором краёв."""
    if len(values) < width:
        return values.copy()
    half = width // 2
    padded = np.pad(values, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    return np.median(windows, axis=1)


def _longest_run(t: np.ndarray, v: np.ndarray, max_step: float) -> tuple[np.ndarray, np.ndarray]:
    """Самый длинный кусок профиля без скачков больше ``max_step`` (по сглаженному профилю)."""
    smooth = _median_filter(v, MEDIAN_WIDTH)
    # Скачок мерится на плече MEDIAN_WIDTH столбцов, а не между соседями: уход профиля на
    # прижатую к снимку букву — наклонная в 60° (1970/10 с.71, «металлическими»), от столбца к
    # столбцу она даёт меньше max_step, а на плече — больше; кромка снимка и перекошенная круче 37° не бывает.
    lag = MEDIAN_WIDTH
    if len(smooth) <= lag:
        return t, v
    jumps = np.abs(smooth[lag:] - smooth[:-lag]) > max_step
    breaks = np.nonzero(jumps)[0] + lag
    best = (0, 0)
    for start, stop in zip(np.r_[0, breaks], np.r_[breaks, len(t)]):
        if stop - start > best[1] - best[0]:
            best = (int(start), int(stop))
    # Кусок обрезается на плечо с обеих сторон: сам скачок лежит внутри плеча.
    start, stop = best[0] + (lag if best[0] > 0 else 0), best[1] - (lag if best[1] < len(t) else 0)
    return t[start:stop], v[start:stop]


def _fit_line(t: np.ndarray, v: np.ndarray) -> tuple[float, float, np.ndarray] | None:
    """Прямая v = a·t + b по точкам с отбросом выбросов; возвращает (a, b, остатки) или None."""
    if len(t) < 5:
        return None
    a, b = np.polyfit(t, v, 1)
    resid = v - (a * t + b)
    mad = np.median(np.abs(resid - np.median(resid))) * 1.4826
    keep = np.abs(resid) <= max(OUTLIER_MADS * mad, 1.0)
    if keep.sum() < 5:
        return None
    a, b = np.polyfit(t[keep], v[keep], 1)
    return float(a), float(b), v[keep] - (a * t[keep] + b)


def _photo_component(mask: np.ndarray, box: Box) -> np.ndarray:
    """Маска одной компоненты замкнутой краски — той, что больше всех перекрывает рамку снимка.

    Строки текста рядом со снимком — свои компоненты (между ними и снимком бумага шире
    ``CLOSE_MM``) и в маску не входят; светлые места внутри снимка (небо, стена) на его
    внешний контур не влияют. Текст, прижатый к снимку вплотную, остаётся в компоненте —
    его срезает :func:`_longest_run`.
    """
    count, labels = cv2.connectedComponents(mask, connectivity=8)
    x0, y0, x1, y1 = box
    inside = labels[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
    if inside.size == 0 or count <= 1:
        return np.zeros_like(mask)
    hist = np.bincount(inside.ravel(), minlength=count)
    hist[0] = 0
    return (labels == int(hist.argmax())).astype(np.uint8)


def _edge_profile(mask: np.ndarray, box: Box, side: str, dpi: float) -> Edge | None:
    """Профиль одной кромки снимка внутри рамки с припуском и прямая по нему.

    Args:
        mask: Маска компоненты снимка (:func:`_photo_component`), вся страница.
        box: Рамка снимка ``(x0, y0, x1, y1)`` в пикселях маски.
        side: ``top`` / ``bottom`` / ``left`` / ``right``.
        dpi: Разрешение маски.

    Returns:
        :class:`Edge` или ``None``, если кромка короткая или непрерывный профиль нашёлся меньше
        чем на ``MIN_COVERAGE`` столбцов.
    """
    h, w = mask.shape
    pad = mm_to_px(PAD_MM, dpi)
    x0, y0, x1, y1 = box
    horizontal = side in ("top", "bottom")
    length = (x1 - x0) if horizontal else (y1 - y0)
    if px_to_mm(length, dpi) < MIN_EDGE_MM:
        return None
    margin = int(round(length * EDGE_MARGIN_FRAC))
    # Полоса поиска — ±pad вокруг самой кромки; кромка — первая краска компоненты снаружи.
    if horizontal:
        cols = np.arange(max(0, x0 + margin), min(w, x1 - margin))
        edge_y = y0 if side == "top" else y1
        lo, hi = max(0, edge_y - pad), min(h, edge_y + pad)
        band = mask[lo:hi, cols]
        if side == "bottom":
            band = band[::-1, :]
        found = band.any(axis=0)
        edge = band.argmax(axis=0)
        v = (lo + edge) if side == "top" else (hi - 1 - edge)
        t = cols
    else:
        rows = np.arange(max(0, y0 + margin), min(h, y1 - margin))
        edge_x = x0 if side == "left" else x1
        lo, hi = max(0, edge_x - pad), min(w, edge_x + pad)
        band = mask[rows, lo:hi].T  # строки полосы — координата x
        if side == "right":
            band = band[::-1, :]
        found = band.any(axis=0)
        edge = band.argmax(axis=0)
        v = (lo + edge) if side == "left" else (hi - 1 - edge)
        t = rows
    total = len(t)
    t, v = t[found].astype(np.float64), v[found].astype(np.float64)
    if total == 0 or len(t) / total < MIN_COVERAGE:
        return None
    t, v = _longest_run(t, v, mm_to_px(MAX_STEP_MM, dpi))
    if len(t) / total < MIN_COVERAGE:
        return None
    fit = _fit_line(t, v)
    if fit is None:
        return None
    a, _, resid = fit
    lo_r, hi_r = np.percentile(resid, SAG_PERCENTILES)
    sag_mm = px_to_mm(float(hi_r - lo_r), dpi)
    tilt_mm = px_to_mm(float(len(t)) * abs(float(np.sin(np.arctan(a)))), dpi)
    points = np.column_stack([t, v]) if horizontal else np.column_stack([v, t])
    return Edge(sag_mm, tilt_mm, points)


def _box_in_a(box: Box, warp: Field | None, shape: tuple) -> Box:
    """Рамка B в кадре A через поле смещений (без поля — как есть), обрезанная по кадру."""
    x0, y0, x1, y1 = box
    if warp is not None:
        corners = warp.transform(np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], dtype=np.float64))
        x0, y0 = corners[:, 0].min(), corners[:, 1].min()
        x1, y1 = corners[:, 0].max(), corners[:, 1].max()
    h, w = shape[:2]
    return (int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1)))


def raster_edge_metrics(
    gray_b: np.ndarray, gray_a: np.ndarray, raster_b: list[Box], warp: Field | None, dpi: float
) -> tuple[dict[str, float], dict]:
    """Порча кромок фотографий: рост сагитты и наклона кромки A − B, максимум по всем кромкам.

    Args:
        gray_b: Серая рабочая копия B (без коррекции).
        gray_a: Серая рабочая копия A (с коррекцией), тот же dpi.
        raster_b: Рамки растра на B в пикселях ``dpi``.
        warp: Поле смещений B → A (рамки в A берутся через него) или ``None``.
        dpi: Разрешение копий.

    Returns:
        Метрики ``raster_edge_bend_mm`` (рост сагитты, мм), ``raster_edge_tilt_mm`` (рост ухода
        конца кромки от оси, мм), ``raster_edges`` (сколько кромок померено) и виновников —
        рамки худшей кромки в B и A с её профилем как ломаной.
    """
    metrics = {"raster_edge_bend_mm": 0.0, "raster_edge_tilt_mm": 0.0, "raster_edges": 0.0}
    culprits: dict = {}
    if not raster_b:
        return metrics, culprits
    mask_b = _ink_mask(gray_b, dpi)
    mask_a = _ink_mask(gray_a, dpi)
    for box in raster_b:
        box_a = _box_in_a(box, warp, gray_a.shape)
        photo_b = _photo_component(mask_b, box)
        photo_a = _photo_component(mask_a, box_a)
        for side in ("top", "bottom", "left", "right"):
            edge_b = _edge_profile(photo_b, box, side, dpi)
            edge_a = _edge_profile(photo_a, box_a, side, dpi)
            if edge_b is None or edge_a is None:
                continue
            metrics["raster_edges"] += 1.0
            for name, delta in (
                ("raster_edge_bend_mm", edge_a.sag_mm - edge_b.sag_mm),
                ("raster_edge_tilt_mm", edge_a.tilt_mm - edge_b.tilt_mm),
            ):
                if delta > metrics[name]:
                    metrics[name] = float(delta)
                    culprits[name] = {"b": box, "a": box_a, **_polyline(edge_b, edge_a)}
    return metrics, culprits


def _polyline(edge_b: Edge, edge_a: Edge, step: int = 8) -> dict:
    """Профили кромок как ломаные для оверлея (каждая ``step``-я точка)."""
    out = {}
    for key, edge in (("segments_b", edge_b), ("segments_a", edge_a)):
        pts = edge.points[::step]
        out[key] = [[*pts[i], *pts[i + 1]] for i in range(len(pts) - 1)]
    return out


__all__ = ["Edge", "raster_edge_metrics"]
