"""Линия по краске вдоль штриха: наклон и изгиб «было | стало» без пары LSD в A.

Ядро меряет наклон штриха в A по отрезку LSD, спаренному с отрезком B. Разбор 2026-09-22:
дробные черты формул 1966/05 с.70 (4 мм) LSD в A не находит вовсе — метрика 0 при видном
наклоне; линейке 1974/08 с.96 LSD в A дал отрезок 489 px с подъёмом 4 px, тогда как краска той
же линии ровная (157–158 px по всей ширине) — ложный ``htilt`` 1.17 мм. Здесь, как в
``bend.py`` ядра, линия в A ищется по самой краске вдоль штриха B, перенесённого полем:
поперечные профили через ``step_mm``, в них тонкий прогон краски, прямая по центрам даёт
наклон, остаток — сагитту. Пара LSD не нужна; шаг проб для коротких черт — 1 мм.

Ограничение шага (``max_jump_mm``): центр краски от пробы к пробе у линии, даже погнутой,
смещается на сотые миллиметра; скачок больше — линия кончилась, а окно нашло соседнюю
(1971/08 с.30: на разрыве вертикали трассировщик ядра перескочил на стрелку в 3.9 мм левее и
дал «изгиб» 3.6 мм). След режется по скачкам, берётся самый длинный кусок.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.geometry_regression import px_to_mm
from ocr_utils.geometry_regression.bend import BEND_MIN_MM, SAG_PERCENTILES, _continuous, _despike, _ink_fraction
from ocr_utils.geometry_regression.field import Field
from ocr_utils.geometry_regression.strokes import (
    AXIS_TOL_DEG,
    RULER_AXIS_TOL_DEG,
    Stroke,
    _is_fraction_bar,
    _weighted_tilt,
)

# Пробы для наклона коротких штрихов (дробные черты 4–8 мм): шаг 1 мм, окно ±1.5 мм.
TILT_STEP_MM = 1.0
TILT_WINDOW_MM = 1.5
TILT_MIN_SAMPLES = 4
# Пробы для изгиба длинных линий — как в ядре.
BEND_STEP_MM = 2.0
BEND_WINDOW_MM = 2.0
BEND_MIN_SAMPLES = 5
# Прогон краски толще — буквы или клякса, не линия.
MAX_THICK_MM = 1.5
# Скачок центра краски между соседними пробами больше — линия кончилась.
MAX_JUMP_MM = 0.5
# Кусок следа короче этой доли исходной длины — линии нет.
MIN_COVERAGE = 0.6
# Излом: соседний кусок следа отстоит от продолжения прямой главного куска не меньше чем на
# столько (мм), в куске не меньше JOG_MIN_SAMPLES проб, а вдоль продолжения прямой краски нет
# (доля ниже JOG_MAX_CONTINUATION) — линия в этом месте разорвана и сдвинута (1974/10 с.30:
# рамка со сдвигом 3.9 мм). Если краска вдоль продолжения есть, линия продолжается прямо, а
# кусок в стороне — соседняя линия, на которую соскочило окно (1971/08 с.30).
JOG_MIN_MM = 1.0
JOG_MIN_SAMPLES = 3
JOG_MAX_CONTINUATION = 0.3
# Средний наклон штрихов ориентации считается от такой суммарной длины (мм): в ядре 20 мм (две
# черты по 5 мм давали 0.6° «из ничего» по LSD); след краски точнее, и три дробные черты по 4 мм
# (1966/05 с.70, наклон +1.6° каждая) должны считаться.
WMEAN_MIN_TOTAL_MM = 8.0
# Дробные черты формул — отдельное семейство: горизонтальные штрихи до FRACTION_MAX_MM вне таблиц и
# рисунков с краской над и под чертой (``strokes._is_fraction_bar``). Средний по длине наклон всех
# горизонталей их не видит: линейки колонтитула и таблицы по 120–130 мм съедают черты 9–18 мм
# (1973/08 с.32, с.35, 1974/10 с.52, 1975/06 с.50 — черты в A под 1–4°). Считаются от FRACTION_MIN_BARS
# черт, средний наклон — по числу черт, без веса по длине.
FRACTION_MAX_MM = 25.0
FRACTION_MIN_BARS = 2
# Числитель и знаменатель: краска в полосах от FRACTION_NEAR_MM до FRACTION_FAR_MM над и под чертой
# (в ядре — до 1.2 мм, для тире; у крупных формул с индексами зазор до 3 мм: 1973/08 с.35) покрывает
# не меньше FRACTION_COVER длины черты с обеих сторон. Подчёркивание рубрики — краска только над
# чертой; линейки между строками длиннее FRACTION_MAX_MM; строки рядом с дробью в формуле (0,5·S/E·K)
# краску в стороны дают, поэтому по сторонам не проверяется.
FRACTION_NEAR_MM = 0.2
FRACTION_FAR_MM = 3.0
FRACTION_COVER = 0.25


@dataclass(frozen=True)
class Trace:
    """След линии: центры краски, направление прямой по ним, сагитта, покрытие."""

    points: np.ndarray  # N × 2, пиксели рендера
    angle_deg: float  # направление прямой в (−90, 90]; 0 — горизонталь
    sag_mm: float
    coverage: float
    length_px: float
    jog_mm: float = 0.0  # излом: боковой сдвиг куска линии относительно её главной части

    @property
    def axis_tilt(self) -> float:
        """Отклонение от ближайшей оси (градусы)."""
        a = abs(self.angle_deg)
        return a if a <= 45.0 else 90.0 - a


def _profile_runs(gray, p0, u, n, length, dpi, step_mm, window_mm):
    """Центры тонких прогонов краски в поперечных профилях вдоль линии (см. ``bend._profile_runs``)."""
    mm = dpi / 25.4
    ts = np.arange(0.0, length + 1e-6, step_mm * mm)
    ss = np.arange(-window_mm * mm, window_mm * mm + 1.0)
    xs = (p0[0] + ts[:, None] * u[0] + ss[None, :] * n[0]).astype(np.float32)
    ys = (p0[1] + ts[:, None] * u[1] + ss[None, :] * n[1]).astype(np.float32)
    values = cv2.remap(gray, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    ink = values < 128
    max_thick = MAX_THICK_MM * mm
    runs_per_sample: list[list[float]] = []
    for i in range(len(ts)):
        idx = np.flatnonzero(ink[i])
        if idx.size == 0:
            runs_per_sample.append([])
            continue
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        runs_per_sample.append([float(ss[run].mean()) if len(run) <= max_thick else np.nan for run in runs])
        runs_per_sample[-1] = [c for c in runs_per_sample[-1] if not np.isnan(c)]
    return ts, runs_per_sample


def _pick_nearest(runs_per_sample, target: float) -> np.ndarray:
    return np.array([min(runs, key=lambda c: abs(c - target)) if runs else np.nan for runs in runs_per_sample])


def _pieces(ts: np.ndarray, offsets: np.ndarray, valid: np.ndarray, max_jump: float) -> list[list[int]]:
    """Куски следа без скачков центра и без длинных пропусков (индексы проб), от длинного к короткому."""
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return []
    pieces: list[list[int]] = [[int(idx[0])]]
    for prev, cur in zip(idx[:-1], idx[1:]):
        # Пропуск проб след не рвёт (штриховка, подписи поперёк полки шкафа — 1966/01 с.78):
        # скачок мерится на пропуск, допуск растёт с его длиной.
        if abs(offsets[cur] - offsets[prev]) > max_jump * (cur - prev):
            pieces.append([])
        pieces[-1].append(int(cur))
    return sorted(pieces, key=lambda p: -(ts[p[-1]] - ts[p[0]] + 1))


def _jog(gray, p0, u, n, ts, offsets, pieces: list[list[int]], coef: np.ndarray, dpi: float) -> float:
    """Излом (мм): сдвиг соседнего куска от продолжения прямой главного, если вдоль продолжения краски нет."""
    mm = dpi / 25.4
    jog = 0.0
    for piece in pieces[1:]:
        if len(piece) < JOG_MIN_SAMPLES:
            continue
        t_piece = ts[piece]
        predicted = np.polyval(coef, t_piece)
        shift = float(abs(np.median(offsets[piece] - predicted)))
        if shift < JOG_MIN_MM * mm:
            continue
        start = p0 + t_piece[0] * u + predicted[0] * n
        stop = p0 + t_piece[-1] * u + predicted[-1] * n
        if _ink_fraction(gray, start, stop) < JOG_MAX_CONTINUATION:
            jog = max(jog, shift / mm)
    return jog


def trace_line(
    gray: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    dpi: float,
    field: Field | None,
    k: float,
    step_mm: float,
    window_mm: float,
    min_samples: int,
) -> Trace | None:
    """След линии краски вдоль отрезка (x0, y0)–(x1, y1) рендера ``gray``; при ``field`` — перенесённого полем.

    Args:
        gray: Серый рендер.
        x0, y0, x1, y1: Отрезок в пикселях ``gray``.
        dpi: Разрешение рендера.
        field: Поле смещений B → A в пикселях ``dpi × k`` или ``None``.
        k: Множитель из пикселей ``gray`` в пиксели поля.
        step_mm: Шаг проб вдоль линии.
        window_mm: Полуширина поперечного окна.
        min_samples: Проб с краской меньше — следа нет.

    Returns:
        :class:`Trace` или ``None``.
    """
    p0 = np.array([x0, y0], dtype=np.float64)
    p1 = np.array([x1, y1], dtype=np.float64)
    if field is not None:
        p0 = field.transform(p0[None] * k)[0] / k
        p1 = field.transform(p1[None] * k)[0] / k
    d = p1 - p0
    length = float(np.hypot(*d))
    if length < 1.0:
        return None
    u = d / length
    n = np.array([-u[1], u[0]])
    ts, runs = _profile_runs(gray, p0, u, n, length, dpi, step_mm, window_mm)
    # Сдвиг прогноза — по пробам с единственным прогоном (как в ядре: рядом идущая кривая
    # графика иначе тянет медиану), окно переносится на него, второй проход.
    offsets = _pick_nearest(runs, 0.0)
    lone = np.array([c[0] if len(c) == 1 else np.nan for c in runs])
    anchor = lone if np.isfinite(lone).sum() >= min_samples else offsets
    if np.isfinite(anchor).sum() >= min_samples:
        p0 = p0 + float(np.nanmedian(anchor)) * n
        ts, runs = _profile_runs(gray, p0, u, n, length, dpi, step_mm, window_mm)
        lone = np.array([c[0] if len(c) == 1 else np.nan for c in runs])
        anchor = lone if np.isfinite(lone).sum() >= min_samples else _pick_nearest(runs, 0.0)
        target = float(np.nanmedian(anchor)) if np.isfinite(anchor).sum() >= min_samples else 0.0
        offsets = _pick_nearest(runs, target)
    valid = ~np.isnan(offsets)
    if valid.sum() >= 2:
        valid &= _continuous(gray, p0, u, n, ts, offsets)
    pieces = _pieces(ts, offsets, valid, MAX_JUMP_MM * dpi / 25.4)
    if not pieces:
        return None
    keep = np.zeros_like(valid)
    keep[pieces[0]] = True
    coverage = float(keep.mean()) if len(ts) else 0.0
    if keep.sum() < min_samples or coverage < MIN_COVERAGE:
        return None
    ts_ok, off_ok = ts[keep], offsets[keep]
    if len(off_ok) >= 5:
        off_ok = _despike(off_ok)
    coef = np.polyfit(ts_ok, off_ok, 1)
    resid = off_ok - np.polyval(coef, ts_ok)
    lo, hi = np.percentile(resid, SAG_PERCENTILES)
    direction = u + coef[0] * n
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
    angle = (angle + 90.0) % 180.0 - 90.0
    points = p0[None, :] + ts_ok[:, None] * u[None, :] + off_ok[:, None] * n[None, :]
    jog = _jog(gray, p0, u, n, ts, offsets, pieces, coef, dpi)
    return Trace(points, angle, px_to_mm(float(hi - lo), dpi), coverage, float(ts_ok[-1] - ts_ok[0]), jog)


@dataclass(frozen=True)
class StrokeTilt:
    """Штрих B с направлениями в обеих версиях: по следам краски или (запасной ход) по паре LSD."""

    stroke: Stroke
    angle_b: float
    angle_a: float
    segment_b: list[float]  # отрезок для оверлея (пиксели рендера)
    segment_a: list[float]
    source: str  # "ridge" | "lsd"


def trace_strokes(
    gray_b: np.ndarray, gray_a: np.ndarray, strokes_b: list[Stroke], field: Field | None, dpi: float, field_dpi: float
) -> list[StrokeTilt]:
    """Направления околоосевых штрихов B по следам краски в B и в A; без следа в любой версии штрих пропускается."""
    k = field_dpi / dpi
    out: list[StrokeTilt] = []
    if field is None:
        # Без поля прогноз положения линии в A — сама линия B: на сдвинутой странице след ляжет
        # на чужую краску (1971/08 с.30 без поля давал средний наклон граф 1.8° из ничего).
        return out
    for stroke in strokes_b:
        if stroke.axis_tilt is None:
            continue
        args = (stroke.x0, stroke.y0, stroke.x1, stroke.y1, dpi)
        trace_b = trace_line(gray_b, *args, None, k, TILT_STEP_MM, TILT_WINDOW_MM, TILT_MIN_SAMPLES)
        if trace_b is None:
            continue
        trace_a = trace_line(gray_a, *args, field, k, TILT_STEP_MM, TILT_WINDOW_MM, TILT_MIN_SAMPLES)
        if trace_a is None:
            continue
        out.append(
            StrokeTilt(stroke, trace_b.angle_deg, trace_a.angle_deg, _segment(trace_b), _segment(trace_a), "ridge")
        )
    return out


def lsd_fallback(tilts: list[StrokeTilt], pairs: list[tuple[Stroke, Stroke]]) -> list[StrokeTilt]:
    """Штрихи с парой LSD, но без следа краски — по углам отрезков LSD (как в ядре)."""
    traced = {id(t.stroke) for t in tilts}
    extra = [
        StrokeTilt(b, b.angle_deg, a.angle_deg, [b.x0, b.y0, b.x1, b.y1], [a.x0, a.y0, a.x1, a.y1], "lsd")
        for b, a in pairs
        if id(b) not in traced and b.axis_tilt is not None
    ]
    return tilts + extra


def _segment(trace: Trace) -> list[float]:
    pts = trace.points
    return [float(pts[0, 0]), float(pts[0, 1]), float(pts[-1, 0]), float(pts[-1, 1])]


def _axis_tilt(angle_deg: float, orient: str) -> float:
    a = abs(angle_deg)
    return a if orient == "h" else 90.0 - a


def _seg_box(segment: list[float]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = segment
    return (int(min(x0, x1)), int(min(y0, y1)), int(max(x0, x1)) + 1, int(max(y0, y1)) + 1)


def ridge_stroke_metrics(
    tilts: list[StrokeTilt], dpi: float, rot_adjust_deg: float = 0.0
) -> tuple[dict[str, float], dict]:
    """Наклон к оси: уход конца в мм и средний наклон, порознь для горизонталей и вертикалей.

    Args:
        tilts: Штрихи с направлениями (:func:`trace_strokes` + :func:`lsd_fallback`).
        dpi: Разрешение рендера.
        rot_adjust_deg: Подтверждённый доворот страницы (``deskew``): вычитается из направления
            в A — штрих, повернувшийся вместе со всей страницей, порчей не считается.

    Returns:
        Метрики ``{h,v}stroke_pairs``, ``{h,v}stroke_dev_max_delta_mm``,
        ``{h,v}stroke_tilt_wmean_{b,a,delta}``, ``{h,v}stroke_gain_mm``, ``strokes_traced`` и
        виновники с отрезками (``segments_*``).
    """
    metrics: dict[str, float] = {"strokes_traced": float(sum(1 for t in tilts if t.source == "ridge"))}
    culprits: dict = {}
    for orient, low, high in (("h", 0.0, AXIS_TOL_DEG), ("v", 90.0 - AXIS_TOL_DEG, 90.0)):
        # Линейка — то, что и в B стояло почти на оси (как в ядре: RULER_AXIS_TOL_DEG), с перпендикуляром.
        own = [
            t
            for t in tilts
            if low <= abs(t.angle_b) <= high and t.stroke.ruled and _axis_tilt(t.angle_b, orient) <= RULER_AXIS_TOL_DEG
        ]
        key = f"{orient}stroke"
        metrics[f"{key}_pairs"] = float(len(own))
        tilt_b = [_axis_tilt(t.angle_b, orient) for t in own]
        tilt_a = [_axis_tilt(t.angle_a, orient) for t in own]
        # Поправка на доворот — только если семейство от неё становится прямее: линейка, ушедшая
        # против доворота страницы (врезка 1970/04 с.26 при снятой трапеции), от поправки хуже.
        adjusted = [_axis_tilt(t.angle_a - rot_adjust_deg, orient) for t in own]
        weights = [t.stroke.length for t in own]
        if rot_adjust_deg and _weighted_tilt(list(zip(adjusted, weights))) < _weighted_tilt(list(zip(tilt_a, weights))):
            tilt_a = adjusted
        total_mm = px_to_mm(sum(t.stroke.length for t in own), dpi)
        metrics[f"{key}_tilt_wmean_b"] = _weighted_tilt([(tb, t.stroke.length) for tb, t in zip(tilt_b, own)])
        metrics[f"{key}_tilt_wmean_a"] = _weighted_tilt([(ta, t.stroke.length) for ta, t in zip(tilt_a, own)])
        metrics[f"{key}_tilt_wmean_delta"] = (
            metrics[f"{key}_tilt_wmean_a"] - metrics[f"{key}_tilt_wmean_b"] if total_mm >= WMEAN_MIN_TOTAL_MM else 0.0
        )
        if not own:
            metrics[f"{key}_dev_max_delta_mm"] = 0.0
            metrics[f"{key}_gain_mm"] = 0.0
            metrics[f"{key}_uniform"] = 0.0
            continue
        deltas = [
            px_to_mm(t.stroke.length, dpi) * (np.sin(np.radians(ta)) - np.sin(np.radians(tb)))
            for t, tb, ta in zip(own, tilt_b, tilt_a)
        ]
        worst = int(np.argmax(deltas))
        metrics[f"{key}_dev_max_delta_mm"] = float(deltas[worst])
        metrics[f"{key}_gain_mm"] = float(max(0.0, -min(deltas)))
        # Равномерность: 1 − коэффициент вариации прироста наклона по длинным линейкам (не короче
        # половины самой длинной). Около 1 — все ушли на один угол (графы таблицы сдвинуты целиком:
        # 1975/08 с.39 — пять граф по 0.4–0.5°), около 0.3–0.5 — разнобой (1967/01 с.38: 0.07–1.26°,
        # 1970/11 с.65: 0.2–1.2°, «границы поплыли»).
        metrics[f"{key}_uniform"] = _uniformity(
            np.array([ta - tb for tb, ta in zip(tilt_b, tilt_a)]), np.array([t.stroke.length for t in own])
        )
        shown = [i for i, d in enumerate(deltas) if d >= 0.5] or [worst]
        culprits[f"{key}_dev_max_delta_mm"] = {
            "b": own[worst].stroke.box,
            "a": _seg_box(own[worst].segment_a),
            "segments_b": [own[i].segment_b for i in shown],
            "segments_a": [own[i].segment_a for i in shown],
        }
    return metrics, culprits


def _uniformity(deltas: np.ndarray, lengths: np.ndarray) -> float:
    """1 − коэффициент вариации приростов по линейкам не короче половины самой длинной; 0, если их меньше двух или прирост мал."""
    long = lengths >= 0.5 * lengths.max()
    if long.sum() < 2:
        return 0.0
    values = deltas[long]
    mean = float(values.mean())
    if mean <= 0.1:
        return 0.0
    return float(np.clip(1.0 - values.std() / mean, 0.0, 1.0))


def _cover(ink: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> float:
    """Доля столбцов [x0, x1) с краской в полосе строк [y0, y1); вне кадра — 0."""
    h, w = ink.shape
    x0, x1, y0, y1 = max(0, x0), min(w, x1), max(0, y0), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(ink[y0:y1, x0:x1].any(axis=0).mean())


def _is_fraction_bar_wide(ink: np.ndarray, stroke: Stroke, dpi: float) -> bool:
    """Дробная черта: краска и над, и под чертой в пределах её длины."""
    mm = dpi / 25.4
    x0, x1 = int(min(stroke.x0, stroke.x1)), int(max(stroke.x0, stroke.x1)) + 1
    y = (stroke.y0 + stroke.y1) / 2.0
    near, far = FRACTION_NEAR_MM * mm, FRACTION_FAR_MM * mm
    return all(
        _cover(ink, x0, x1, y_lo, y_hi) >= FRACTION_COVER
        for y_lo, y_hi in ((int(y - far), int(y - near) + 1), (int(y + near), int(y + far) + 1))
    )


def fraction_metrics(
    tilts: list[StrokeTilt], ink_b: np.ndarray, dpi: float, exclude: list, k: float
) -> tuple[dict[str, float], dict]:
    """Наклон дробных черт формул: средний и худший прирост |наклона| A − B (градусы).

    Args:
        tilts: Штрихи с направлениями (:func:`trace_strokes` + :func:`lsd_fallback`).
        ink_b: Краска рендера B (``серый < 128``) для проверки числителя и знаменателя.
        dpi: Разрешение рендера.
        exclude: Рамки таблиц и рисунков в пикселях ``dpi / k`` — черты внутри них не считаются.
        k: Множитель из пикселей рендера в пиксели рамок.

    Returns:
        ``fraction_bars``, ``fraction_tilt_mean_delta_deg``, ``fraction_tilt_max_delta_deg`` и виновник
        (все черты, сегменты B и A).
    """
    metrics = {"fraction_bars": 0.0, "fraction_tilt_mean_delta_deg": 0.0, "fraction_tilt_max_delta_deg": 0.0}
    culprits: dict = {}
    max_len = FRACTION_MAX_MM * dpi / 25.4
    bars: list[StrokeTilt] = []
    for t in tilts:
        s = t.stroke
        if s.length > max_len or abs(t.angle_b) > AXIS_TOL_DEG or s.in_lineart:
            continue
        mx, my = s.mid
        if any(x0 <= mx * k < x1 and y0 <= my * k < y1 for x0, y0, x1, y1 in exclude):
            continue
        if not _is_fraction_bar_wide(ink_b, s, dpi):
            continue
        bars.append(t)
    metrics["fraction_bars"] = float(len(bars))
    if len(bars) < FRACTION_MIN_BARS:
        return metrics, culprits
    deltas = [_axis_tilt(t.angle_a, "h") - _axis_tilt(t.angle_b, "h") for t in bars]
    metrics["fraction_tilt_mean_delta_deg"] = float(np.mean(deltas))
    metrics["fraction_tilt_max_delta_deg"] = float(max(deltas))
    worst = int(np.argmax(deltas))
    culprits["fraction_tilt_mean_delta_deg"] = {
        "b": bars[worst].stroke.box,
        "a": _seg_box(bars[worst].segment_a),
        "segments_b": [t.segment_b for t in bars],
        "segments_a": [t.segment_a for t in bars],
    }
    culprits["fraction_tilt_max_delta_deg"] = culprits["fraction_tilt_mean_delta_deg"]
    return metrics, culprits


def _trace_box(trace: Trace) -> tuple[int, int, int, int]:
    pts = trace.points
    return (int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()) + 1, int(pts[:, 1].max()) + 1)


def bend_metrics(
    gray_b: np.ndarray, gray_a: np.ndarray, strokes_b: list[Stroke], field: Field | None, dpi: float, field_dpi: float
) -> tuple[dict[str, float], dict]:
    """Прирост сагитты и излома A − B по длинным (≥ ``BEND_MIN_MM``) штрихам B, след с ограничением шага."""
    metrics = {"stroke_bend_dev_mm": 0.0, "stroke_bend_lines": 0.0, "stroke_jog_dev_mm": 0.0}
    culprits: dict = {}
    k = field_dpi / dpi
    min_len = BEND_MIN_MM * dpi / 25.4
    for stroke in strokes_b:
        if stroke.length < min_len:
            continue
        args = (stroke.x0, stroke.y0, stroke.x1, stroke.y1, dpi)
        trace_b = trace_line(gray_b, *args, None, k, BEND_STEP_MM, BEND_WINDOW_MM, BEND_MIN_SAMPLES)
        trace_a = trace_line(gray_a, *args, field, k, BEND_STEP_MM, BEND_WINDOW_MM, BEND_MIN_SAMPLES)
        if trace_b is None or trace_a is None:
            continue
        metrics["stroke_bend_lines"] += 1.0
        pts = trace_a.points
        culprit = {
            "b": stroke.box,
            "a": _trace_box(trace_a),
            "segments_b": [[stroke.x0, stroke.y0, stroke.x1, stroke.y1]],
            "segments_a": [[*pts[i], *pts[i + 1]] for i in range(len(pts) - 1)],
        }
        for name, delta in (
            ("stroke_bend_dev_mm", trace_a.sag_mm - trace_b.sag_mm),
            ("stroke_jog_dev_mm", trace_a.jog_mm - trace_b.jog_mm),
        ):
            if delta > metrics[name]:
                metrics[name] = delta
                culprits[name] = culprit
    return metrics, culprits


__all__ = [
    "Trace",
    "StrokeTilt",
    "trace_line",
    "trace_strokes",
    "lsd_fallback",
    "ridge_stroke_metrics",
    "fraction_metrics",
    "bend_metrics",
]
