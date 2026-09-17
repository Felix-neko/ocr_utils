"""Кривизна строк по полигонам детектора Surya — нейросетевая сегментация вместо RLSA.

То же измерение, что в ``line_fit`` (центр-линия по краске, прямая и парабола), но строки
находит обученная сеть. Ошибки у неё другие: там, где смыкание слипает строку с линейкой
таблицы или рвёт заголовок вразрядку, сеть уверенно выделяет именно текст. Полигон Surya
четырёхугольный, то есть даёт наклон строки, но не её изгиб — изгиб дочитывается по краске
внутри полигона.

Полигоны ищутся тайлами через ``defocus_detection.lines.detect.LineDetector`` и лежат в
его дисковом кэше: сеть — самая дорогая часть прогона, и повторный прогон с другой
агрегацией её трогать не должен. Работает в родительском процессе, пачками (CLAUDE.md:
GPU в пул не заворачивать).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from ocr_utils.scan_markup.curved_lines.detectors.base import Detector, GpuPage, Measure
from ocr_utils.scan_markup.curved_lines.fitting import (
    LineFit,
    centreline,
    fit_line,
    long_mask,
    page_stats,
    slope_field_stats,
    smooth_median,
)

# Размеры — в пикселях картинки, ушедшей на GPU (длинная сторона ``--gpu-side``, то есть
# около 150 dpi при 1536).
MIN_HEIGHT_PX = 6
MIN_LENGTH_HEIGHTS = 8
SMOOTH_HEIGHTS = 2.0
MIN_LINES = 8
MIN_LONG_LINES = 30  # см. line_fit.MIN_LONG_LINES
RAW_STEP = 16

DEFAULT_TILE_SIDE = 1100
DEFAULT_TILE_OVERLAP = 300
DEFAULT_MIN_CONF = 0.5


def surya_available() -> bool:
    try:
        import surya.detection  # noqa: F401
    except Exception:
        return False
    return True


def _polygon_slope(tl: np.ndarray, tr: np.ndarray, br: np.ndarray, bl: np.ndarray) -> float:
    """Наклон полигона в градусах (среднее по верхнему и нижнему ребру), знак как у строк."""
    slopes = []
    for left, right in ((tl, tr), (bl, br)):
        span = right[0] - left[0]
        if abs(span) > 1e-9:
            slopes.append((right[1] - left[1]) / span)
    return float(np.degrees(np.arctan(np.mean(slopes)))) if slopes else 0.0


def measure_regions(gray: np.ndarray, regions: Sequence, keep_raw: bool) -> Measure:
    """Метрики по найденным полигонам и краске внутри них."""
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink = gray <= threshold
    height, width = gray.shape
    fits: list[LineFit] = []
    heights: list[float] = []
    poly_slopes: list[float] = []
    raw_lines: list[dict] = []
    for region in regions:
        tl, tr, br, bl = region.corners()
        line_h = region.height
        line_w = float((tr[0] + br[0]) / 2.0 - (tl[0] + bl[0]) / 2.0)
        if line_h < MIN_HEIGHT_PX or line_w < MIN_LENGTH_HEIGHTS * line_h:
            continue
        polygon = np.array([tl, tr, br, bl], dtype=np.float64)
        x0, y0 = int(max(0, np.floor(polygon[:, 0].min()))), int(max(0, np.floor(polygon[:, 1].min())))
        x1, y1 = int(min(width, np.ceil(polygon[:, 0].max()))), int(min(height, np.ceil(polygon[:, 1].max())))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        inside = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(inside, [np.round(polygon - [x0, y0]).astype(np.int32)], 255)
        xs, ys, weights = centreline(ink[y0:y1, x0:x1] & (inside > 0))
        ys = smooth_median(ys, int(SMOOTH_HEIGHTS * line_h))
        fit = fit_line(xs, ys, weights)
        if fit is None:
            continue
        fit = LineFit(
            fit.slope_deg,
            fit.curvature,
            fit.sagitta,
            fit.resid_lin,
            fit.resid_quad,
            fit.length,
            fit.n,
            x0 + fit.x0,
            y0 + fit.y0,
            x0 + fit.x1,
            y0 + fit.y1,
        )
        fits.append(fit)
        heights.append(float(line_h))
        poly_slopes.append(_polygon_slope(tl, tr, br, bl))
        if keep_raw:
            points = [[round(x0 + px, 1), round(y0 + py, 1)] for px, py in zip(xs[::RAW_STEP], ys[::RAW_STEP])]
            raw_lines.append(
                {
                    "poly": [[round(float(px), 1), round(float(py), 1)] for px, py in polygon],
                    "slope": round(fit.slope_deg, 3),
                    "sag_rel": round(abs(fit.sagitta) / line_h, 3),
                    "pts": points,
                }
            )

    raw = {"w": width, "h": height, "lines": raw_lines} if keep_raw else None
    if len(fits) < MIN_LINES:
        return Measure(metrics={"lines": float(len(fits))}, note="мало строк", silent=True, raw=raw)
    metrics = page_stats(fits, heights)
    long = long_mask(fits)
    if keep_raw:
        for line, is_long in zip(raw_lines, long):
            line["long"] = int(is_long)
    if int(long.sum()) < MIN_LONG_LINES:
        return Measure(metrics=metrics, note="мало длинных строк", silent=True, raw=raw)
    centres_x = np.array([(fit.x0 + fit.x1) / 2.0 for fit in fits])[long] / width
    centres_y = np.array([(fit.y0 + fit.y1) / 2.0 for fit in fits])[long] / height
    slopes = np.array([fit.slope_deg for fit in fits])[long]
    lengths = np.array([fit.length for fit in fits])[long]
    metrics.update(slope_field_stats(centres_x, centres_y, slopes, lengths))
    poly = np.array(poly_slopes)[long]
    metrics["poly_slope_spread_deg"] = float(np.percentile(poly, 90) - np.percentile(poly, 10)) if poly.size else 0.0
    return Measure(metrics=metrics, raw=raw)


class SuryaCurves:
    """GPU-детектор: полигоны Surya (через LineDetector с кэшем) плюс краска внутри них."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        tile_side: int = DEFAULT_TILE_SIDE,
        tile_overlap: int = DEFAULT_TILE_OVERLAP,
        min_conf: float = DEFAULT_MIN_CONF,
        batch_size: int | None = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._tile_side = tile_side
        self._tile_overlap = tile_overlap
        self._min_conf = min_conf
        self._batch_size = batch_size
        self._detector = None

    def _load(self):
        if self._detector is None:
            from ocr_utils.defocus_detection.lines.detect import DetectCache, DetectParams, LineDetector

            params = DetectParams(
                mode="tiles",
                tile_side=self._tile_side,
                tile_overlap=self._tile_overlap,
                min_conf=self._min_conf,
                batch_size=self._batch_size,
            )
            cache = DetectCache(self._cache_dir / "surya") if self._cache_dir is not None else None
            self._detector = LineDetector(params, cache)
        return self._detector

    def __call__(self, pages: Sequence[GpuPage], keep_raw: bool = False) -> list[Measure]:
        detector = self._load()
        measures: list[Measure] = []
        for page in pages:
            gray = np.asarray(page.image.convert("L"))
            try:
                regions = detector.detect(page.path, gray)
            except Exception as error:
                measures.append(Measure(note=f"surya: {error}", silent=True))
                continue
            measures.append(measure_regions(gray, regions, keep_raw))
        return measures


def draw(canvas: np.ndarray, raw: dict, scale: float) -> None:
    """Полигоны Surya серым, центр-линии — цветом по прогибу."""
    for line in raw.get("lines", []):
        polygon = np.array([[px * scale, py * scale] for px, py in line["poly"]], np.int32)
        cv2.polylines(canvas, [polygon.reshape(-1, 1, 2)], True, (180, 180, 180), 1, cv2.LINE_AA)
        points = np.array([[px * scale, py * scale] for px, py in line["pts"]], np.int32)
        if len(points) < 2:
            continue
        sag = line["sag_rel"]
        colour = (0, 170, 0) if sag < 0.15 else (0, 200, 255) if sag < 0.3 else (0, 0, 255)
        cv2.polylines(canvas, [points.reshape(-1, 1, 2)], False, colour, 2 if line.get("long") else 1, cv2.LINE_AA)


ALGORITHM = Detector(
    name="surya_lines",
    summary="то же, что line_fit, но строки размечает сеть Surya (GPU, тайлами, с кэшем полигонов)",
    stage="gpu",
    # Пороги — по 14 эталонным полосам при --gpu-side 1536: p90 прогиба у прямых 0.05-0.10
    # высоты, у кривых 0.13-0.20 (две «лёгкие» — 0.07); среднее трёх больших прогибов у прямых
    # до 0.16, у кривых 0.19-0.32 (лёгкая — 0.08); размах наклонов полигонов у прямых до
    # 0.37°, у кривых 0.44-3.8°.
    # Разброс наклонов ПОЛИГОНОВ (poly_slope_spread_deg) из флаговых убран после прогона по
    # паку: углы полигонов surya квантованы, и при пороге 0.4° метрика флаговала 30% полос.
    # Остальные пороги подняты к p95 распределения по паку.
    thresholds={"sagitta_rel_p90": 0.15, "sagitta_rel_max3": 0.22, "slope_spread_deg": 0.9, "slope_resid_deg": 0.35},
    version=3,
    make_batch=SuryaCurves,
    draw=draw,
    available=surya_available,
)
