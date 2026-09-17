"""Загнутые кончики строк: усреднённый по колонке профиль отклонения концов строк.

ЗАЧЕМ. Основные детекторы меряют строку или тайл целиком, и загиб последних 5-10 мм
строки (единицы пикселей при 600 dpi) тонет в шуме формы букв: на размеченной выборке
1976/12 они ловили 3 из 20 таких полос. Здесь шум давится усреднением по всем строкам
колонки: у настоящего загиба концы всех строк уходят в одну сторону, у шума — в разные.

КАК. Строки — те же, что у ``line_fit`` (``line_samples``). В каждой колонке берутся
длинные строки, ДОХОДЯЩИЕ до края колонки (иначе загиб у края измерять не по чему); у
каждой прямая проводится по средней половине, и отклонение центр-линии от неё
раскладывается по расстоянию до края. Медиана по строкам в каждом бине — профиль загиба
c(d). Две метрики: наклон профиля на первых 200 px (300 dpi) в градусах — угол кончика
относительно середины строки — и, независимо, разница угла текста в узкой полосе у края
колонки и в её центре (сдвиг профилей, как в ``skew_map``).

Замер на выборке 1976/12 (20 полос с загибами, 12 без): по отдельности каждая метрика
слаба (AUC 0.75), базовая линия по нижней огибающей вместо центра масс не работает вовсе
(AUC 0.5). В объединении с щедрыми порогами основных детекторов при 0.8° ловится 14 из 20
при одной ложной из 12. Поэтому детектор ВСПОМОГАТЕЛЬНЫЙ и ДОСТАТОЧНЫЙ: его флаг сам по
себе даёт сводный флаг (``Detector.sufficient``). Глобальные искажения (волны, поворот
блока) он видит плохо — это дело остальных.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from ocr_utils.scan_markup.curved_lines.detectors import skew_map
from ocr_utils.scan_markup.curved_lines.detectors.skew_map import MIN_COLUMN_PX
from ocr_utils.scan_markup.curved_lines.detectors.base import Detector, Frame, Measure
from ocr_utils.scan_markup.curved_lines.detectors.line_fit import LineSample, line_samples
from ocr_utils.scan_markup.curved_lines.fitting import long_mask
from ocr_utils.scan_markup.orientation.detectors.ink_axis import glyph_mask

# Размеры — в пикселях копии 300 dpi, если не сказано иное.

# Строка «доходит до края колонки», если её конец в стольких высотах от края (край —
# p90 правых концов / p10 левых начал длинных строк колонки).
EDGE_TOL_HEIGHTS = 1.5
# Опорная прямая — по средней половине строки: кончики в неё не входят по построению.
MID = (0.25, 0.75)
# Профиль по расстоянию от края: бин и глубина (2.5 см).
BIN_PX = 8
DEPTH_PX = 600
# Наклон профиля считается по первым стольким пикселям от края (7 мм).
SLOPE_DEPTH_PX = 200
# «Кончик» для отклонения и знака — первые столько пикселей от края.
NEAR_PX = 60
# Меньше стольких доходящих до края строк — профиль не строится.
MIN_LINES = 8

# Полоса у края колонки для сравнения углов (копия 150 dpi): 120 px это 2 см.
EDGE_BAND_PX = 120
MIN_BAND_INK = 0.02

# СТРУКТУРНЫЙ ТЕНЗОР. Третья мера того же — угол строк в полосе 2 см у края против центра,
# но не по сдвигу профилей, а по ориентации градиента анизотропно сглаженной краски
# (ориентационное поле, как в дьюорпе по локальным ориентациям): сглаживание вдоль строки
# на 0.8 высоты стирает буквы и оставляет «бруски» строк, чей градиент перпендикулярен
# строке. На выборке 1976/12 это лучшая из проверенных мер (AUC 0.82 против 0.74 у сдвига
# профилей): при 0.5° — 11 из 20 полос с загибами и 1 из 12 без.
TENSOR_BAND_PX = 240  # копия 300 dpi, те же 2 см
TENSOR_SIGMA_Y = 1.5
TENSOR_SIGMA_X_HEIGHTS = 0.8
MIN_COHERENCE = 0.2


def _columns(separators: list[tuple[int, int]], width150: int) -> list[tuple[int, int]]:
    edges = [0] + [v for pair in separators for v in pair] + [width150]
    columns = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2) if edges[i + 1] - edges[i] >= MIN_COLUMN_PX]
    return columns or [(0, width150)]


def end_profile(lines: list[LineSample], side: str, edge: float, h300: float) -> dict | None:
    """Профиль отклонения концов строк от края колонки, в высотах строки."""
    bins: dict[int, list[float]] = {}
    signs: list[float] = []
    used = 0
    for line in lines:
        xs, ys = line.xs, line.ys
        end = xs.max() if side == "right" else xs.min()
        if abs(end - edge) > EDGE_TOL_HEIGHTS * h300:
            continue
        n = xs.size
        a, b = int(MID[0] * n), int(MID[1] * n)
        if b - a < 12:
            continue
        coef = np.polyfit(xs[a:b], ys[a:b], 1)
        resid = (ys - np.polyval(coef, xs)) / h300
        distance = (edge - xs) if side == "right" else (xs - edge)
        ok = (distance >= 0) & (distance <= DEPTH_PX)
        for index, value in zip((distance[ok] // BIN_PX).astype(int), resid[ok]):
            bins.setdefault(int(index), []).append(float(value))
        near = resid[ok & (distance <= NEAR_PX)]
        if near.size:
            signs.append(float(np.sign(np.median(near))))
        used += 1
    if used < MIN_LINES:
        return None
    count = DEPTH_PX // BIN_PX
    profile = np.array([np.median(bins[k]) if k in bins else np.nan for k in range(count)])
    centres = np.arange(count) * BIN_PX + BIN_PX / 2.0
    near_value = (
        float(np.nanmean(profile[: NEAR_PX // BIN_PX])) if np.isfinite(profile[: NEAR_PX // BIN_PX]).any() else 0.0
    )
    chosen = np.isfinite(profile) & (centres <= SLOPE_DEPTH_PX)
    slope = float(np.polyfit(centres[chosen], profile[chosen] * h300, 1)[0]) if chosen.sum() >= 5 else 0.0
    sign_array = np.array(signs)
    return {
        "side": side,
        "lines": used,
        "near_rel": abs(near_value),
        "slope_deg": abs(float(np.degrees(np.arctan(slope)))),
        "consistency": float(abs(sign_array.mean())) if sign_array.size else 0.0,
        "profile": [None if not np.isfinite(v) else round(float(v), 4) for v in profile],
    }


def edge_angle_diff(mask150: np.ndarray, column: tuple[int, int], lines: list[LineSample]) -> float | None:
    """Разница угла текста у края колонки и в её центре, по двум вертикальным половинам."""
    x0, x1 = column
    if x1 - x0 < 3 * EDGE_BAND_PX:
        return None
    y0 = min(line.y for line in lines)
    y1 = max(line.y_end for line in lines)
    if y1 - y0 < 200:
        return None
    mid = (x0 + x1) // 2
    centre = (mid - EDGE_BAND_PX // 2, mid + EDGE_BAND_PX // 2)
    best = None
    for band in ((x0, x0 + EDGE_BAND_PX), (x1 - EDGE_BAND_PX, x1)):
        for ya, yb in ((y0, (y0 + y1) // 2), ((y0 + y1) // 2, y1)):
            edge_tile = mask150[ya:yb, band[0] : band[1]]
            centre_tile = mask150[ya:yb, centre[0] : centre[1]]
            if edge_tile.mean() / 255.0 < MIN_BAND_INK or centre_tile.mean() / 255.0 < MIN_BAND_INK:
                continue
            e_angle, e_conf, e_period = skew_map.tile_angle(edge_tile)
            c_angle, c_conf, c_period = skew_map.tile_angle(centre_tile)
            if min(e_conf, c_conf) < skew_map.MIN_CONFIDENCE or min(e_period, c_period) < skew_map.MIN_PERIODICITY:
                continue
            diff = abs(e_angle - c_angle)
            best = diff if best is None else max(best, diff)
    return best


def tensor_angle(gray300: np.ndarray, x0: int, x1: int, y0: int, y1: int, sigma_x: float) -> tuple[float, float] | None:
    """Угол строк в окне по структурному тензору (градусы) и когерентность ориентации."""
    crop = 255.0 - gray300[y0:y1, x0:x1].astype(np.float32)
    if crop.size == 0:
        return None
    smooth = gaussian_filter(crop, sigma=(TENSOR_SIGMA_Y, sigma_x))
    gy, gx = np.gradient(smooth)
    jxx, jyy, jxy = float((gx * gx).sum()), float((gy * gy).sum()), float((gx * gy).sum())
    if jxx + jyy <= 0.0:
        return None
    theta = 0.5 * np.degrees(np.arctan2(2.0 * jxy, jxx - jyy))  # ориентация градиента
    coherence = float(np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy**2) / (jxx + jyy))
    line_angle = (theta - 90.0 + 90.0) % 180.0 - 90.0  # строки перпендикулярны градиенту
    return float(line_angle), coherence


def tensor_edge_diff(
    gray300: np.ndarray, column: tuple[int, int], lines: list[LineSample], h300: float
) -> float | None:
    """Разница угла строк у края колонки и в центре по структурному тензору, по двум половинам высоты."""
    x0, x1 = 2 * column[0], 2 * column[1]
    if x1 - x0 < 3 * TENSOR_BAND_PX:
        return None
    y0 = 2 * min(line.y for line in lines)
    y1 = 2 * max(line.y_end for line in lines)
    if y1 - y0 < 400:
        return None
    mid = (x0 + x1) // 2
    sigma_x = TENSOR_SIGMA_X_HEIGHTS * h300
    best = None
    for ya, yb in ((y0, (y0 + y1) // 2), ((y0 + y1) // 2, y1)):
        centre = tensor_angle(gray300, mid - TENSOR_BAND_PX // 2, mid + TENSOR_BAND_PX // 2, ya, yb, sigma_x)
        if centre is None or centre[1] < MIN_COHERENCE:
            continue
        for bx0, bx1 in ((x0, x0 + TENSOR_BAND_PX), (x1 - TENSOR_BAND_PX, x1)):
            edge = tensor_angle(gray300, bx0, bx1, ya, yb, sigma_x)
            if edge is None or edge[1] < MIN_COHERENCE:
                continue
            diff = abs(edge[0] - centre[0])
            best = diff if best is None else max(best, diff)
    return best


def measure(frame: Frame, keep_raw: bool = False) -> Measure:
    samples, separators = line_samples(frame)
    width150 = frame.gray150.shape[1]
    mask150 = glyph_mask(frame.gray150)
    profiles: list[dict] = []
    angle_diffs: list[float] = []
    tensor_diffs: list[float] = []
    for column in _columns(separators, width150):
        lines = [s for s in samples if s.x >= column[0] - 2 and s.x_end <= column[1] + 2]
        if len(lines) < MIN_LINES:
            continue
        lengths = np.array([s.x_end - s.x for s in lines], dtype=np.float64)
        long = lengths >= 0.6 * np.percentile(lengths, 90)
        lines = [s for s, ok in zip(lines, long) if ok]
        if len(lines) < MIN_LINES:
            continue
        h300 = 2.0 * float(np.median([s.h_line for s in lines]))
        right_edge = 2.0 * float(np.percentile([s.x_end for s in lines], 90))
        left_edge = 2.0 * float(np.percentile([s.x for s in lines], 10))
        for side, edge in (("right", right_edge), ("left", left_edge)):
            profile = end_profile(lines, side, edge, h300)
            if profile is not None:
                profile.update(
                    {"column": [int(column[0]), int(column[1])], "edge150": round(edge / 2.0, 1), "h150": h300 / 2.0}
                )
                profiles.append(profile)
        diff = edge_angle_diff(mask150, column, lines)
        if diff is not None:
            angle_diffs.append(diff)
        diff = tensor_edge_diff(frame.gray300, column, lines, h300)
        if diff is not None:
            tensor_diffs.append(diff)

    raw = (
        {"w": width150, "h": frame.gray150.shape[0], "profiles": profiles, "angle_diffs": angle_diffs}
        if keep_raw
        else None
    )
    if not profiles and not angle_diffs:
        return Measure(metrics={"end_lines": 0.0}, note="нет строк, доходящих до края колонки", silent=True, raw=raw)
    metrics = {
        "end_slope_deg": max((p["slope_deg"] for p in profiles), default=0.0),
        "end_dev_rel": max((p["near_rel"] for p in profiles), default=0.0),
        "end_consistency": max((p["consistency"] for p in profiles), default=0.0),
        "end_lines": float(sum(p["lines"] for p in profiles)),
        "edge_angle_diff_deg": max(angle_diffs, default=0.0),
        "tensor_edge_diff_deg": max(tensor_diffs, default=0.0),
    }
    return Measure(metrics=metrics, raw=raw)


def draw(canvas: np.ndarray, raw: dict, scale: float) -> None:
    """Профиль загиба у каждого края колонки, усиленный в 20 раз, поверх середины колонки."""
    for profile in raw.get("profiles", []):
        x_edge = profile["edge150"] * scale
        h150 = profile["h150"]
        y_mid = canvas.shape[0] / 2.0
        points = []
        for index, value in enumerate(profile["profile"]):
            if value is None:
                continue
            d150 = (index * BIN_PX + BIN_PX / 2.0) / 2.0
            x = x_edge - d150 * scale if profile["side"] == "right" else x_edge + d150 * scale
            points.append([x, y_mid + 20.0 * value * h150 * scale])
        if len(points) < 2:
            continue
        colour = (
            (0, 0, 255)
            if profile["slope_deg"] >= 0.8
            else (0, 200, 255) if profile["slope_deg"] >= 0.5 else (0, 170, 0)
        )
        cv2.polylines(canvas, [np.array(points, np.int32).reshape(-1, 1, 2)], False, colour, 2, cv2.LINE_AA)
        cv2.line(canvas, (int(x_edge), int(y_mid - 40)), (int(x_edge), int(y_mid + 40)), (255, 120, 0), 1)
        cv2.putText(
            canvas,
            f"{profile['slope_deg']:.2f}deg",
            (int(x_edge) - 40, int(y_mid) - 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            colour,
            1,
            cv2.LINE_AA,
        )


ALGORITHM = Detector(
    name="end_curl",
    summary="загнутые кончики строк: усреднённый по колонке профиль концов и угол кромки против центра",
    stage="cpu",
    # Пороги — по выборке 1976/12 (20 полос с загибами, 12 без). Наклон профиля и сдвиг
    # профилей при 0.8° вместе с щедрыми порогами основных детекторов дают 14 из 20 при
    # 2 ложных из 12 и 26% пака. Структурный тензор сам по себе — лучшая из трёх мер
    # (AUC 0.82), но в объединении НЕ добавляет ни одной полосы выборки, а при 0.5° поднимает
    # долю пака с 26% до 35%; поэтому его порог 1.0° — метрика остаётся в CSV, флагует редко.
    thresholds={"end_slope_deg": 0.8, "edge_angle_diff_deg": 0.8, "tensor_edge_diff_deg": 1.0},
    version=2,
    run=measure,
    draw=draw,
    sufficient=True,
)
