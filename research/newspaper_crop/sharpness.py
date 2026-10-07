"""Цена выпрямления перспективы для резкости текста: DOM и крутизна края на участках печати до и после гомографии, на превью, превью ×2 и RAW."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import rawpy

from ocr_utils.defocus_detection.metrics.dom import gradient_threshold, sharpness_maps
from research.newspaper_crop.fill import quad_homography
from research.newspaper_crop.ink_guard import print_cluster, text_lines
from research.newspaper_crop.zone import quad_mask, work_shape_of

# Сторона участка печати на масштабе превью, px: ~6 строк текста ЭГ (строка ~32 px превью).
TILE_PREVIEW = 160
# Сколько участков на кадр: самые густые по строкам печати.
TILES_PER_FRAME = 30
# Наименьшая доля строк печати в участке, иначе это поле или иллюстрация.
MIN_LINE_SHARE = 0.35
# Отступ участка от стороны четырёхугольника, px превью: у края лист выгнут, тени и дорисовка.
EDGE_MARGIN = 120


class Variant(Enum):
    """Вариант выпрямления: где (превью, превью ×2, RAW), какой интерполяцией и в каком масштабе сравнивается."""

    PREVIEW_LINEAR = "превью, linear"
    PREVIEW_CUBIC = "превью, cubic"
    PREVIEW_LANCZOS = "превью, lanczos4"
    UP2_CUBIC_DOWN = "превью ×2 lanczos → cubic → ×0.5 area"
    UP2_CUBIC_KEEP = "превью ×2 lanczos → cubic, выход ×2"
    PREVIEW_SHIFT_HALF = "контроль: превью, сдвиг на ½ px, cubic"
    RAW_CUBIC = "RAW, cubic"
    RAW_LANCZOS = "RAW, lanczos4"
    RAW_SHIFT_HALF = "контроль: RAW, сдвиг на ½ px, cubic"


@dataclass(frozen=True)
class TileScore:
    """Замер одного участка: резкость DOM и крутизна края до и после, локальный наклон строк в источнике."""

    dom_src: float
    dom_out: float
    steep_src: float
    steep_out: float
    tilt_deg: float


def steepness(tile: np.ndarray) -> float:
    """Крутизна края: p95 модуля градиента Собеля, делённый на размах яркости участка (p95 − p5).

    Почти не зависит от поворота (модуль градиента), контраст сокращается.

    Args:
        tile: Полутоновый участок uint8.

    Returns:
        Безразмерная крутизна (больше — резче); NaN на пустом участке.
    """
    f = tile.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    rng = float(np.percentile(f, 95) - np.percentile(f, 5))
    if rng < 10:
        return float("nan")
    return float(np.percentile(np.hypot(gx, gy), 95)) / rng


def dom_score(tile: np.ndarray, threshold: float) -> float:
    """Средний DOM по краевым пикселям участка с общим для пары порогом края.

    Args:
        tile: Полутоновый участок uint8.
        threshold: Порог первой разности «это край» (берётся с участка-источника, чтобы пара мерилась одинаково).

    Returns:
        DOM (больше — резче); NaN, если краёв нет.
    """
    sharp, edges = sharpness_maps(tile, edge_rel=0.0, edge_min=threshold)
    return float(sharp[edges].mean()) if edges.any() else float("nan")


def crop(gray: np.ndarray, centre: np.ndarray, side: int) -> np.ndarray | None:
    """Квадратный участок с центром ``centre`` (None, если не помещается в кадр).

    Args:
        gray: Полутоновый кадр.
        centre: (x, y).
        side: Сторона, px.

    Returns:
        Участок или None.
    """
    x0, y0 = int(round(centre[0] - side / 2)), int(round(centre[1] - side / 2))
    if x0 < 0 or y0 < 0 or x0 + side > gray.shape[1] or y0 + side > gray.shape[0]:
        return None
    return gray[y0 : y0 + side, x0 : x0 + side]


def apply_h(m: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """Образ точки под гомографией.

    Args:
        m: Матрица 3×3.
        pt: Точка (x, y).

    Returns:
        Точка (x, y).
    """
    v = m @ np.array([pt[0], pt[1], 1.0])
    return v[:2] / v[2]


def tile_centres(rgb: np.ndarray, quad: np.ndarray) -> list[np.ndarray]:
    """Центры участков печати на масштабе превью: сетка с шагом в участок, густые по строкам, внутри четырёхугольника с отступом.

    Args:
        rgb: Превью RGB.
        quad: Четырёхугольник в пикселях превью.

    Returns:
        До :data:`TILES_PER_FRAME` центров (x, y), самые густые первыми.
    """
    shape = work_shape_of(rgb)
    lines = print_cluster(text_lines(rgb, shape))
    scale = rgb.shape[0] / shape[0]
    inner = quad_mask(quad / scale, shape)
    k = int(EDGE_MARGIN / scale)
    inner = cv2.erode(inner.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
    side = int(TILE_PREVIEW / scale)
    cands = []
    for y in range(0, shape[0] - side, side):
        for x in range(0, shape[1] - side, side):
            if not inner[y : y + side, x : x + side].all():
                continue
            share = float(lines[y : y + side, x : x + side].mean())
            if share >= MIN_LINE_SHARE:
                cands.append((share, np.array([(x + side / 2) * scale, (y + side / 2) * scale])))
    cands.sort(key=lambda c: -c[0])
    return [c for _, c in cands[:TILES_PER_FRAME]]


def local_tilt(m: np.ndarray, pt: np.ndarray) -> float:
    """Наклон строки в источнике относительно выхода в точке: угол образа горизонтального отрезка, градусы.

    Args:
        m: Гомография источник → выход.
        pt: Точка источника.

    Returns:
        Модуль угла, градусы.
    """
    d = apply_h(m, pt + np.array([10.0, 0.0])) - apply_h(m, pt)
    return abs(float(np.degrees(np.arctan2(d[1], d[0]))))


def score_pairs(
    src: np.ndarray,
    out: np.ndarray,
    centres_src: list[np.ndarray],
    centres_out: list[np.ndarray],
    side: int,
    tilts: list[float],
) -> list[TileScore]:
    """Замеры пар участков «источник — выход».

    Args:
        src: Источник (полутоновый).
        out: Выход (полутоновый).
        centres_src: Центры в источнике.
        centres_out: Соответствующие центры в выходе.
        side: Сторона участка, px.
        tilts: Локальный наклон в каждой точке, градусы.

    Returns:
        Замеры по участкам, поместившимся в оба кадра.
    """
    scores = []
    for cs, co, tilt in zip(centres_src, centres_out, tilts):
        a, b = crop(src, cs, side), crop(out, co, side)
        if a is None or b is None:
            continue
        thr = gradient_threshold(a)
        scores.append(TileScore(dom_score(a, thr), dom_score(b, thr), steepness(a), steepness(b), tilt))
    return scores


def warp_variant(gray: np.ndarray, quad: np.ndarray, flags: int) -> tuple[np.ndarray, np.ndarray]:
    """Выпрямление кадра гомографией четырёхугольника.

    Args:
        gray: Полутоновый кадр.
        quad: Четырёхугольник в пикселях кадра.
        flags: Интерполяция OpenCV.

    Returns:
        (выпрямленный кадр, гомография).
    """
    m, w, h = quad_homography(quad)
    return cv2.warpPerspective(gray, m, (w, h), flags=flags, borderMode=cv2.BORDER_REPLICATE), m


def measure_frame(preview_path: Path, raf_path: Path | None, quad_norm: np.ndarray) -> dict[Variant, list[TileScore]]:
    """Все варианты выпрямления одного кадра (воркер пула).

    Args:
        preview_path: Превью JPEG (вход стенда; при закрашенных пальцах — версия ``nofinger``).
        raf_path: RAF для вариантов на RAW или None.
        quad_norm: Четырёхугольник в нормированных координатах кадра.

    Returns:
        Замеры по вариантам.
    """
    bgr = cv2.imread(str(preview_path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h1, w1 = gray.shape
    quad1 = quad_norm * [w1, h1]
    centres = tile_centres(rgb, quad1)
    side = TILE_PREVIEW
    out: dict[Variant, list[TileScore]] = {}

    # Превью: три интерполяции на исходном масштабе.
    for variant, flags in (
        (Variant.PREVIEW_LINEAR, cv2.INTER_LINEAR),
        (Variant.PREVIEW_CUBIC, cv2.INTER_CUBIC),
        (Variant.PREVIEW_LANCZOS, cv2.INTER_LANCZOS4),
    ):
        warped, m = warp_variant(gray, quad1, flags)
        tilts = [local_tilt(m, c) for c in centres]
        out[variant] = score_pairs(gray, warped, centres, [apply_h(m, c) for c in centres], side, tilts)

    # Контроль: чистая цена интерполяции — сдвиг на полпикселя по обеим осям, без поворота.
    shift = np.float32([[1, 0, 0.5], [0, 1, 0.5]])
    shifted = cv2.warpAffine(gray, shift, (w1, h1), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    out[Variant.PREVIEW_SHIFT_HALF] = score_pairs(
        gray, shifted, centres, [c + 0.5 for c in centres], side, [0.0] * len(centres)
    )

    # Апскейл ×2 до выпрямления: выход сравнивается и после возврата к ×1, и на ×2 с превью ×2 без выпрямления.
    up = cv2.resize(gray, (w1 * 2, h1 * 2), interpolation=cv2.INTER_LANCZOS4)
    warped2, m2 = warp_variant(up, quad1 * 2, cv2.INTER_CUBIC)
    tilts = [local_tilt(m2, c * 2) for c in centres]
    down = cv2.resize(warped2, (warped2.shape[1] // 2, warped2.shape[0] // 2), interpolation=cv2.INTER_AREA)
    out[Variant.UP2_CUBIC_DOWN] = score_pairs(
        gray, down, centres, [apply_h(m2, c * 2) / 2 for c in centres], side, tilts
    )
    out[Variant.UP2_CUBIC_KEEP] = score_pairs(
        up, warped2, [c * 2 for c in centres], [apply_h(m2, c * 2) for c in centres], side * 2, tilts
    )

    if raf_path is not None:
        with rawpy.imread(str(raf_path)) as raw:
            raw_rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
        raw_gray = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2GRAY)
        hr, wr = raw_gray.shape
        s = wr / w1
        quad_r = quad_norm * [wr, hr]
        centres_r = [c * [wr / w1, hr / h1] for c in centres]
        side_r = int(round(side * s))
        for variant, flags in ((Variant.RAW_CUBIC, cv2.INTER_CUBIC), (Variant.RAW_LANCZOS, cv2.INTER_LANCZOS4)):
            warped, m = warp_variant(raw_gray, quad_r, flags)
            tilts = [local_tilt(m, c) for c in centres_r]
            out[variant] = score_pairs(raw_gray, warped, centres_r, [apply_h(m, c) for c in centres_r], side_r, tilts)
        shifted = cv2.warpAffine(raw_gray, shift, (wr, hr), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        out[Variant.RAW_SHIFT_HALF] = score_pairs(
            raw_gray, shifted, centres_r, [c + 0.5 for c in centres_r], side_r, [0.0] * len(centres_r)
        )
    return out
