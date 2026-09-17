"""Движок ``textline`` — выпрямление по центр-линиям строк, как dewarp в Leptonica.

ИДЕЯ (Bloomberg, leptonica ``dewarp``). Каждая строка текста на плоской странице прямая;
на скане с прогибом у корешка она изгибается. Если через центр-линию каждой строки
провести параболу и вычесть из неё прямую, останется чистый прогиб d(x) — на сколько
пикселей строка ушла вниз или вверх от своей прямой в каждой точке. Прогибы всех строк —
это разреженные отсчёты гладкого поля вертикальной диспаратности V(x, y); по ним поле
восстанавливается сглаживающей интерполяцией и применяется ремапом ``dst(x, y) =
src(x, y + V(x, y))``. Строки становятся прямыми, а размер полосы, поля и общий перекос
не меняются: перекос FineReader выправляет сам и делает это хорошо, портит он именно
кривые строки.

Только вертикальная составляющая. Leptonica умеет и горизонтальную (растяжение текста
у корешка), но на флэтбед-скане журнала прогиб мал, и главное искажение — вертикальное.
Горизонтальная поправка требует надёжных краёв колонок, а их у полосы с таблицей и
заголовками во всю ширину нет; вариант ``textline_h`` отложен.

Сегментация строк — та же, что у детектора ``line_fit`` (маска глифов, RLSA, сборка
кусков в строки с запретом склейки через межколонник), поэтому движок выпрямляет ровно
то, что детектор считает кривым.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

from ocr_utils.dewarp.engines.base import DewarpEngine
from ocr_utils.scan_markup.curved_lines.detectors.line_fit import (
    LINE_MAX_THICKNESS_PX,
    MIN_LENGTH_HEIGHTS,
    SMOOTH_HEIGHTS,
    _line_blobs,
    column_separators,
    link_spans,
)
from ocr_utils.scan_markup.curved_lines.fitting import LineFit, centreline, fit_line, long_mask, smooth_median
from ocr_utils.scan_markup.orientation.detectors.ink_axis import LINE_ASPECT, LINE_MIN_LENGTH_PX, glyph_mask

logger = logging.getLogger(__name__)

# Рабочие разрешения: сегментация на 150 dpi (константы детектора заданы для него),
# центр-линия на 300 dpi (точность прогиба).
SEG_DPI = 150
FINE_DPI = 300

# Меньше стольких длинных строк — поле не строится, кадр возвращается как есть: по
# горстке строк интерполяция нарисует поле из ничего и «выправит» ровную полосу.
MIN_LINES = 8

# Сколько отсчётов прогиба брать вдоль каждой строки.
SAMPLES_PER_LINE = 12

# Сглаживание тонкопластинного сплайна (RBF thin plate). Координаты отсчётов нормированы
# на размер полосы (0..1), поэтому ядро r²·ln r имеет порядок сотых, и параметр надо
# держать той же малости: при 30 сплайн вырождался в плоскость и поле выходило нулевым
# (замер на синтетике: ошибка поля 6 px при истинном прогибе 12). При 0.01 синтетический
# прогиб у корешка снимается с 0.38 высоты строки до 0.015, а рябь от формы букв не
# проходит.
SMOOTHING = 0.01

# Сетка, на которой вычисляется поле; дальше — билинейный апскейл до полного кадра.
GRID_COLS = 32
GRID_ROWS = 48

# Прогиб больше стольких высот строки — не прогиб, а слипшаяся с чем-то строка; отсчёт
# отбрасывается, чтобы не тащить поле за собой.
MAX_SAG_HEIGHTS = 1.5


@dataclass(frozen=True)
class Field:
    """Поле вертикальной диспаратности на сетке — для отладки и тестов.

    Attributes:
        grid: Массив (GRID_ROWS, GRID_COLS) прогибов в пикселях копии 150 dpi.
        samples: Число отсчётов, по которым построено поле.
        lines: Число длинных строк.
    """

    grid: np.ndarray
    samples: int
    lines: int


Segment = tuple[LineFit, float, np.ndarray, np.ndarray]


def _segment(gray150: np.ndarray, gray300: np.ndarray) -> list[Segment]:
    """Строки полосы: аппроксимация, высота и сглаженная центр-линия (xs, ys) — всё в
    координатах копии 150 dpi."""
    threshold, _ = cv2.threshold(gray150, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    mask150 = glyph_mask(gray150)
    smeared, labels, stats = _line_blobs(mask150)
    separators = column_separators(smeared)
    ink300 = gray300 <= threshold
    segments: list[Segment] = []
    for span in link_spans(stats, separators):
        members = stats[span]
        x = int(members[:, cv2.CC_STAT_LEFT].min())
        y = int(members[:, cv2.CC_STAT_TOP].min())
        x_end = int((members[:, cv2.CC_STAT_LEFT] + members[:, cv2.CC_STAT_WIDTH]).max())
        y_end = int((members[:, cv2.CC_STAT_TOP] + members[:, cv2.CC_STAT_HEIGHT]).max())
        w, h = x_end - x, y_end - y
        h_line = float(np.median(members[:, cv2.CC_STAT_HEIGHT]))
        if (
            h > LINE_MAX_THICKNESS_PX
            or w < LINE_ASPECT * h_line
            or w < max(LINE_MIN_LENGTH_PX, MIN_LENGTH_HEIGHTS * h_line)
        ):
            continue
        own = np.isin(labels[y:y_end, x:x_end], span).astype(np.uint8)
        own300 = cv2.resize(own, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
        crop = ink300[2 * y : 2 * y_end, 2 * x : 2 * x_end]
        own300 = own300[: crop.shape[0], : crop.shape[1]]
        xs, ys, weights = centreline(crop & (own300 > 0))
        ys = smooth_median(ys, int(SMOOTH_HEIGHTS * 2 * h_line))
        fit = fit_line(xs, ys, weights)
        if fit is None:
            continue
        segments.append(
            (
                LineFit(
                    slope_deg=fit.slope_deg,
                    curvature=fit.curvature * 2.0,
                    sagitta=fit.sagitta / 2.0,
                    resid_lin=fit.resid_lin / 2.0,
                    resid_quad=fit.resid_quad / 2.0,
                    length=fit.length / 2.0,
                    n=fit.n,
                    x0=x + fit.x0 / 2.0,
                    y0=y + fit.y0 / 2.0,
                    x1=x + fit.x1 / 2.0,
                    y1=y + fit.y1 / 2.0,
                ),
                h_line,
                x + xs / 2.0,
                y + ys / 2.0,
            )
        )
    return segments


def measure_field(gray150: np.ndarray, gray300: np.ndarray) -> Field | None:
    """Поле прогибов по строкам полосы; None — строк мало.

    Прогиб берётся не из параболы, а из самой сглаженной центр-линии: разность между ней
    и её прямой. Парабола хороша для книжного корешка, где прогиб плавный по всей строке,
    а у журнальной полосы прогиб часто сидит в последней четверти строки, и парабола его
    размазывает по всей длине. Шум центр-линии (форма букв) гасится медианой в две высоты
    строки при измерении и сглаживанием сплайна при интерполяции.
    """
    segments = _segment(gray150, gray300)
    if not segments:
        return None
    fits = [segment[0] for segment in segments]
    long = long_mask(fits)
    chosen = [segment for segment, is_long in zip(segments, long) if is_long]
    if len(chosen) < MIN_LINES:
        return None

    height150, width150 = gray150.shape
    points: list[tuple[float, float]] = []
    values: list[float] = []
    for fit, h, xs, ys in chosen:
        if abs(fit.sagitta) > MAX_SAG_HEIGHTS * h or xs.size < SAMPLES_PER_LINE:
            continue
        y_lin = fit.y0 + (fit.y1 - fit.y0) * (xs - fit.x0) / max(fit.length, 1e-9)
        d = ys - y_lin
        # Отсчёты — средние по SAMPLES_PER_LINE равным отрезкам строки: и шум меньше, и
        # сплайну не тысячи точек на полосу.
        for chunk in np.array_split(np.arange(xs.size), SAMPLES_PER_LINE):
            if chunk.size == 0:
                continue
            points.append((float(xs[chunk].mean()) / width150, float(y_lin[chunk].mean()) / height150))
            values.append(float(d[chunk].mean()))
    if len(points) < MIN_LINES * SAMPLES_PER_LINE // 2:
        return None

    pts = np.array(points)
    vals = np.array(values)
    interpolator = RBFInterpolator(pts, vals, kernel="thin_plate_spline", smoothing=SMOOTHING)
    # Сетка вычисляется внутри охвата отсчётов: за его пределами (поля, колонтитул) поле
    # продолжается постоянным — тонкопластинный сплайн вне данных уходит куда угодно.
    gx = np.clip((np.arange(GRID_COLS) + 0.5) / GRID_COLS, pts[:, 0].min(), pts[:, 0].max())
    gy = np.clip((np.arange(GRID_ROWS) + 0.5) / GRID_ROWS, pts[:, 1].min(), pts[:, 1].max())
    mesh = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)
    grid = interpolator(mesh).reshape(GRID_ROWS, GRID_COLS)
    return Field(grid=grid, samples=len(points), lines=len(chosen))


def apply_field(img: np.ndarray, field: Field, scale: float) -> np.ndarray:
    """Ремап кадра полем прогибов; ``scale`` — пикселей кадра на пиксель копии 150 dpi."""
    height, width = img.shape[:2]
    full = cv2.resize(field.grid.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC) * scale
    map_x = np.tile(np.arange(width, dtype=np.float32), (height, 1))
    map_y = np.arange(height, dtype=np.float32)[:, None] + full
    return cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _scaled_gray(img_bgr: np.ndarray, dpi: int, target_dpi: int) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    factor = target_dpi / dpi
    if abs(factor - 1.0) < 1e-6:
        return gray
    size = (max(1, round(gray.shape[1] * factor)), max(1, round(gray.shape[0] * factor)))
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


class TextLineEngine(DewarpEngine):
    name = "textline"

    def __init__(self) -> None:
        self.last_field: Field | None = None

    def load(self, device: str) -> None:
        pass  # чистая классика: ни модели, ни весов

    def dewarp(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        gray150 = _scaled_gray(img_bgr, self.dpi, SEG_DPI)
        gray300 = _scaled_gray(img_bgr, self.dpi, FINE_DPI)
        # Копия 300 dpi должна быть ровно вдвое крупнее копии 150: сегментация меряет её
        # координаты умножением на два.
        gray300 = cv2.resize(gray300, (gray150.shape[1] * 2, gray150.shape[0] * 2), interpolation=cv2.INTER_AREA)
        field = measure_field(gray150, gray300)
        self.last_field = field
        if field is None:
            logger.info("textline: строк мало, кадр оставлен как есть")
            return img_bgr.copy()
        return apply_field(img_bgr, field, self.dpi / SEG_DPI)
