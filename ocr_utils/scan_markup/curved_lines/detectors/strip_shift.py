"""Сдвиг строк между соседними вертикальными полосами колонки — кросс-корреляция профилей.

Ещё один взгляд на ту же геометрию, но без сегментации строк и без перебора углов.
Колонка режется на узкие вертикальные полосы; профиль проекции каждой (сумма краски по
строкам развёртки) — это гребёнка строк. У прямой строки гребёнка соседней полосы — та
же, сдвинутая на ``ширина полосы · tg наклона``; кросс-корреляция находит этот сдвиг с
точностью до долей пикселя. Накопленный сдвиг вдоль колонки — это форма строк: прямая
у прямой (пусть и перекошенной) колонки, дуга у прогнутой. Метрика — остаток
накопленного сдвига от прямой, в долях межстрочного шага.

Колонка режется ещё и по вертикали на несколько лент, потому что прогиб у корешка сидит
внизу полосы, а вверху строки прямые: корреляция по всей высоте колонки нашла бы
доминирующий нулевой сдвиг и прогиб бы проглотила.

Экспериментальный детектор: у него есть слепое пятно — там, где полоса не находит
периодичности (таблица, заголовок, две строки), — и он сильнее других зависит от
разбиения на колонки.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

from ocr_utils.scan_markup.curved_lines.detectors.base import Detector, Frame, Measure, silent
from ocr_utils.scan_markup.curved_lines.detectors.line_fit import column_separators
from ocr_utils.scan_markup.orientation.detectors.ink_axis import _smear, glyph_mask
from ocr_utils.scan_markup.orientation.detectors.profile import PITCH_MAX_PX, PITCH_MIN_PX

# Все размеры — в пикселях копии 150 dpi.

# Колонка уже этой не режется на полосы: 120 px это 2 см, шесть полос по 20 px.
MIN_COLUMN_PX = 120
N_STRIPS = 6
# Лент по вертикали и минимальная высота ленты в межстрочных шагах: меньше четырёх строк —
# гребёнке не с чем коррелировать.
N_BANDS = 4
MIN_BAND_PITCHES = 4.0

# Ниже этого пика автокорреляции у колонки нет межстрочного шага — не текст.
MIN_PITCH_PEAK = 0.2
# Ниже этой нормированной кросс-корреляции сдвиг между полосами не найден (в одной из
# полос картинка, заголовок или пусто).
MIN_CORR = 0.3

# Меньше стольких единиц (колонка × лента) с найденными сдвигами — сводку не строить.
MIN_UNITS = 3


def _detrended(profile: np.ndarray, window: int) -> np.ndarray:
    profile = profile.astype(np.float64)
    profile -= uniform_filter1d(profile, size=int(window) | 1, mode="nearest")
    return profile


def column_pitch(profile: np.ndarray) -> tuple[float, float]:
    """Межстрочный шаг колонки по автокорреляции профиля и высота пика."""
    profile = _detrended(profile, PITCH_MAX_PX * 2)
    energy = float((profile * profile).sum())
    if energy <= 0.0 or profile.size <= PITCH_MAX_PX * 2:
        return 0.0, 0.0
    correlation = np.correlate(profile, profile, mode="full")[profile.size - 1 :] / energy
    window = correlation[PITCH_MIN_PX : PITCH_MAX_PX + 1]
    if window.size == 0:
        return 0.0, 0.0
    lag = int(window.argmax())
    return float(PITCH_MIN_PX + lag), float(window[lag])


def strip_shift(previous: np.ndarray, current: np.ndarray, max_lag: int) -> tuple[float, float]:
    """Сдвиг ``current`` относительно ``previous`` (положительный — строки ниже) и пик корреляции.

    Пик уточняется параболой по трём точкам — сдвиг нужен с точностью до долей пикселя,
    потому что на ширине полосы в 20 px наклон в градус даёт треть пикселя.
    """
    norm = float(np.sqrt((previous * previous).sum() * (current * current).sum()))
    if norm <= 0.0:
        return 0.0, 0.0
    lags = np.arange(-max_lag, max_lag + 1)
    scores = np.array([float((current * np.roll(previous, lag)).sum()) / norm for lag in lags])
    best = int(scores.argmax())
    lag = float(lags[best])
    if 0 < best < scores.size - 1:
        left, mid, right = scores[best - 1], scores[best], scores[best + 1]
        denominator = left - 2.0 * mid + right
        if denominator < 0.0:
            lag += 0.5 * (left - right) / denominator
    # np.roll(previous, lag) сдвигает гребёнку вниз на lag, значит current ниже previous на lag.
    return lag, float(scores[best])


def measure(frame: Frame, keep_raw: bool = False) -> Measure:
    mask = glyph_mask(frame.gray150)
    height, width = mask.shape
    smeared = _smear(mask, horizontal=True)
    separators = column_separators(smeared)

    # Колонки — промежутки между межколонниками достаточной ширины.
    edges = [0] + [x for pair in separators for x in pair] + [width]
    columns = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2) if edges[i + 1] - edges[i] >= MIN_COLUMN_PX]

    units: list[dict] = []
    for x0, x1 in columns:
        column = mask[:, x0:x1]
        pitch, peak = column_pitch(column.sum(axis=1))
        if peak < MIN_PITCH_PEAK or pitch <= 0.0:
            continue
        band_h = max(int(MIN_BAND_PITCHES * pitch), height // N_BANDS)
        strip_w = (x1 - x0) // N_STRIPS
        max_lag = max(2, int(pitch // 2))
        for y0 in range(0, height - band_h + 1, band_h):
            y1 = y0 + band_h
            profiles = [
                _detrended(mask[y0:y1, x0 + k * strip_w : x0 + (k + 1) * strip_w].sum(axis=1), pitch * 2)
                for k in range(N_STRIPS)
            ]
            shifts = [0.0]
            ok = True
            for k in range(1, N_STRIPS):
                lag, corr = strip_shift(profiles[k - 1], profiles[k], max_lag)
                if corr < MIN_CORR:
                    ok = False
                    break
                shifts.append(shifts[-1] + lag)
            if not ok:
                continue
            xs = x0 + (np.arange(N_STRIPS) + 0.5) * strip_w
            shifts_arr = np.array(shifts)
            lin = np.polyfit(xs, shifts_arr, 1)
            resid = shifts_arr - np.polyval(lin, xs)
            units.append(
                {
                    "x0": int(x0),
                    "x1": int(x1),
                    "y0": int(y0),
                    "y1": int(y1),
                    "pitch": round(pitch, 1),
                    "xs": [round(float(v), 1) for v in xs],
                    "shift": [round(float(v), 2) for v in shifts_arr],
                    "angle": float(np.degrees(np.arctan(lin[0]))),
                    "resid_rel": float(np.abs(resid).max() / pitch),
                    "resid_rms_rel": float(np.sqrt((resid * resid).mean()) / pitch),
                }
            )

    raw = {"w": width, "h": height, "units": units} if keep_raw else None
    if len(units) < MIN_UNITS:
        return Measure(metrics={"units": float(len(units))}, note="мало колонок с периодичностью", silent=True, raw=raw)

    angles = np.array([unit["angle"] for unit in units])
    median = float(np.median(angles))
    metrics = {
        "units": float(len(units)),
        "columns": float(len(columns)),
        "resid_max_rel": float(max(unit["resid_rel"] for unit in units)),
        "resid_rms_rel": float(np.sqrt(np.mean([unit["resid_rms_rel"] ** 2 for unit in units]))),
        "angle_spread_deg": float(np.percentile(angles, 90) - np.percentile(angles, 10)),
        "angle_max_dev_deg": float(np.abs(angles - median).max()),
    }
    return Measure(metrics=metrics, raw=raw)


def draw(canvas: np.ndarray, raw: dict, scale: float) -> None:
    """Ленты колонок и кривая накопленного сдвига (усилена втрое), цвет — остаток."""
    for unit in raw.get("units", []):
        colour = (0, 170, 0) if unit["resid_rel"] < 0.15 else (0, 200, 255) if unit["resid_rel"] < 0.3 else (0, 0, 255)
        cv2.rectangle(
            canvas,
            (int(unit["x0"] * scale), int(unit["y0"] * scale)),
            (int(unit["x1"] * scale), int(unit["y1"] * scale)),
            (210, 210, 210),
            1,
        )
        y_mid = (unit["y0"] + unit["y1"]) / 2.0
        points = np.array([[x * scale, (y_mid + 3.0 * s) * scale] for x, s in zip(unit["xs"], unit["shift"])], np.int32)
        cv2.polylines(canvas, [points.reshape(-1, 1, 2)], False, colour, 2, cv2.LINE_AA)


ALGORITHM = Detector(
    name="strip_shift",
    summary="сдвиг гребёнки строк между соседними вертикальными полосами колонки (кросс-корреляция профилей)",
    stage="cpu",
    # Пороги — по 14 эталонным полосам: остаток накопленного сдвига у прямых 0.015-0.025
    # шага, у кривых 0.08-0.22 (одна «лёгкая» — 0.02); отклонение угла ленты от медианы
    # у прямых до 0.29°, у кривых 0.55-4.0° (та же лёгкая — 0.03°).
    # После прогона по паку пороги подняты (p95 по паку 0.09 и 1.0): при 0.05 / 0.5 детектор
    # в одиночку отправлял в свод таблицы и оглавления с «гребёнкой» из ячеек.
    thresholds={"resid_max_rel": 0.12, "angle_max_dev_deg": 1.0},
    version=1,
    run=measure,
    draw=draw,
    # Замер по паку: 74 полосы, где детектор в одиночку набрал «сильный» score, глазами
    # оказались ровными (гребёнка ловит соседнюю строку). Голос считается только с чужим.
    solo=False,
)
