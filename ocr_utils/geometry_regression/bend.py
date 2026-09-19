"""Изгиб длинной линии: сагитта краски вдоль штриха «было | стало».

Кусочный вариант по LSD (штрих B в A распался на отрезки под разными углами) на паке дал
296 ложных `bad`: LSD режет ровную толстую линейку на параллельные куски со своими углами,
и разность углов ничего не говорит о форме линии (1969/01 с.85, 1973/03 с.86 — ровные линейки).
Здесь линия мерится по самой краске: вдоль штриха B через ``BEND_STEP_MM`` берётся поперечный
профиль, в нём — ближайший к оси прогон краски не толще ``BEND_MAX_THICK_MM`` (чтобы не зацепить
текст рядом), и его центр даёт поперечное смещение линии в этой точке. То же — в A по той же
линии, перенесённой полем смещений. Остаток смещений от прямой (p98 − p2, мм) — сагитта;
метрика — разность сагитт A − B: ровная линейка в обеих версиях даёт ≈ 0.1–0.3 мм, погнутая
FineReader'ом (1966/01 с.95 — 1.7, с.78 — 3.5, 1970/06 с.37 — 0.9) заметно больше. Пара в A
штриху не нужна: линия, которую FineReader порвал так, что LSD её не нашёл, всё равно мерится.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.geometry_regression import px_to_mm
from ocr_utils.geometry_regression.field import Field
from ocr_utils.geometry_regression.strokes import Stroke

# Линия короче — не мерится: сагитта короткой линии тонет в шуме толщины краски.
BEND_MIN_MM = 50.0
# Шаг проб вдоль линии и полуширина поперечного окна (в нём линия ищется после переноса полем:
# поле ошибается на доли миллиметра, окно должно её перекрывать).
BEND_STEP_MM = 2.0
BEND_WINDOW_MM = 2.0
# Прогон краски толще — не линия, а буквы или клякса рядом с ней.
BEND_MAX_THICK_MM = 1.5
# Проб с найденной линией должно быть не меньше этой доли от всех (и не меньше BEND_MIN_SAMPLES):
# пунктир и линия, порванная в A на короткие куски, иначе дают случайные центры.
BEND_MIN_COVERAGE = 0.6
BEND_MIN_SAMPLES = 5
# Линия, а не текст: соседние пробы (через BEND_STEP_MM) у линии, даже погнутой, смещены друг
# от друга на сотые миллиметра; если поле смещений увело прогноз на строку текста (1973/01 с.56:
# верхняя половина страницы без тайлов, A растянута FineReader'ом на 7–12 %), центры краски
# прыгают по штрихам букв на ~1 мм от пробы к пробе. Медиана шага больше — линии тут нет.
BEND_MAX_STEP_MM = 0.4
# Ширина медианного фильтра по пробам (нечётная): убирает скачки до двух проб подряд.
BEND_MEDIAN_WIDTH = 5
# Сагитта — размах остатка между этими перцентилями: одиночная проба, зацепившая засечку
# буквы, размах не задирает.
SAG_PERCENTILES = (2.0, 98.0)


@dataclass(frozen=True)
class Ridge:
    """Поперечные смещения краски линии в пробах вдоль неё (в пикселях рендера).

    ``points`` — координаты найденных центров краски (N × 2), ``sag_mm`` — размах остатка от
    прямой, ``coverage`` — доля проб, где линия нашлась.
    """

    points: np.ndarray
    sag_mm: float
    coverage: float


def _profile_runs(
    gray: np.ndarray, p0: np.ndarray, u: np.ndarray, n: np.ndarray, length: float, dpi: float
) -> tuple[np.ndarray, list[list[float]]]:
    """Центры тонких прогонов краски в поперечных профилях вдоль линии.

    Аргументы: ``gray`` — серый рендер; ``p0`` — начало линии; ``u``/``n`` — единичные вектор
    вдоль и нормаль; ``length`` — длина в пикселях; ``dpi`` — разрешение рендера.
    Возвращает положения проб вдоль линии и для каждой пробы список поперечных смещений
    прогонов краски не толще ``BEND_MAX_THICK_MM`` (пустой — краски в пробе нет).
    """
    mm = dpi / 25.4
    ts = np.arange(0.0, length, BEND_STEP_MM * mm)
    ss = np.arange(-BEND_WINDOW_MM * mm, BEND_WINDOW_MM * mm + 1.0)
    xs = (p0[0] + ts[:, None] * u[0] + ss[None, :] * n[0]).astype(np.float32)
    ys = (p0[1] + ts[:, None] * u[1] + ss[None, :] * n[1]).astype(np.float32)
    values = cv2.remap(gray, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    ink = values < 128
    max_thick = BEND_MAX_THICK_MM * mm
    runs_per_sample: list[list[float]] = []
    for i in range(len(ts)):
        idx = np.flatnonzero(ink[i])
        if idx.size == 0:
            runs_per_sample.append([])
            continue
        # Прогоны краски в профиле — по разрывам индексов.
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        runs_per_sample.append([float(ss[run].mean()) for run in runs if len(run) <= max_thick])
    return ts, runs_per_sample


def _pick_nearest(runs_per_sample: list[list[float]], target: float) -> np.ndarray:
    """В каждой пробе — прогон, ближайший к ``target``; NaN, где прогонов нет."""
    return np.array([min(runs, key=lambda c: abs(c - target)) if runs else np.nan for runs in runs_per_sample])


def _despike(offsets: np.ndarray) -> np.ndarray:
    """Медианный фильтр шириной ``BEND_MEDIAN_WIDTH`` по пробам: одиночные скачки убираются.

    Проба, где ближайший прогон — точка отточия или засечка буквы рядом с линейкой (1975/07
    с.44), выпадает на миллиметры на одну-две пробы; погнутая линия уходит от прямой плавно и
    на десяток проб подряд (1971/08 с.79: конец линейки на 1.2 мм) — её фильтр не трогает.
    Отбраковка по остатку от прямой (MAD) такой изгиб на конце принимала за выбросы.
    """
    half = BEND_MEDIAN_WIDTH // 2
    padded = np.pad(offsets, half, mode="edge")
    return np.array([np.median(padded[i : i + BEND_MEDIAN_WIDTH]) for i in range(len(offsets))])


def ridge_along(
    gray: np.ndarray, x0: float, y0: float, x1: float, y1: float, dpi: float, field: Field | None, k: float
) -> Ridge | None:
    """Линия краски вдоль отрезка (x0, y0)–(x1, y1), при ``field`` — перенесённого полем.

    ``k`` — множитель из пикселей ``gray`` в пиксели поля. Возвращает None, если линия найдена
    меньше чем в ``BEND_MIN_COVERAGE`` проб.
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
    ts, runs_per_sample = _profile_runs(gray, p0, u, n, length, dpi)
    # Первый проход — прогон, ближайший к прогнозу; медиана найденного даёт сдвиг прогноза
    # (поле ошибается на 1–2 мм у края тайлов), второй проход — прогон, ближайший к этому сдвигу.
    offsets = _pick_nearest(runs_per_sample, 0.0)
    if np.isfinite(offsets).sum() >= BEND_MIN_SAMPLES:
        offsets = _pick_nearest(runs_per_sample, float(np.nanmedian(offsets)))
    ok = ~np.isnan(offsets)
    coverage = float(ok.mean()) if len(ts) else 0.0
    if ok.sum() < BEND_MIN_SAMPLES or coverage < BEND_MIN_COVERAGE:
        return None
    # Скачки центров от пробы к пробе — это текст под прогнозом, а не линия.
    steps = np.abs(np.diff(offsets[ok]))
    if px_to_mm(float(np.median(steps)), dpi) > BEND_MAX_STEP_MM:
        return None
    ts, offsets = ts[ok], _despike(offsets[ok])
    coef = np.polyfit(ts, offsets, 1)
    resid = offsets - np.polyval(coef, ts)
    lo, hi = np.percentile(resid, SAG_PERCENTILES)
    points = p0[None, :] + ts[:, None] * u[None, :] + offsets[:, None] * n[None, :]
    return Ridge(points, px_to_mm(float(hi - lo), dpi), coverage)


def bend_metrics(
    gray_b: np.ndarray, gray_a: np.ndarray, strokes_b: list[Stroke], field: Field | None, dpi: float, field_dpi: float
) -> tuple[dict[str, float], dict]:
    """Наибольший прирост сагитты A − B по длинным штрихам B.

    Аргументы: ``gray_b``/``gray_a`` — рендеры обеих версий в ``dpi``; ``strokes_b`` — штрихи B
    в тех же пикселях; ``field`` — поле смещений B→A в пикселях ``field_dpi``.
    Возвращает метрики (``stroke_bend_dev_mm`` — max(sag_A − sag_B, 0); ``stroke_bend_lines`` —
    сколько линий померено) и виновника: рамки в обеих версиях и ломаная найденной линии A.
    """
    metrics = {"stroke_bend_dev_mm": 0.0, "stroke_bend_lines": 0.0}
    culprits: dict = {}
    k = field_dpi / dpi
    min_len = BEND_MIN_MM * dpi / 25.4
    for stroke in strokes_b:
        if stroke.length < min_len:
            continue
        ridge_b = ridge_along(gray_b, stroke.x0, stroke.y0, stroke.x1, stroke.y1, dpi, None, k)
        ridge_a = ridge_along(gray_a, stroke.x0, stroke.y0, stroke.x1, stroke.y1, dpi, field, k)
        if ridge_b is None or ridge_a is None:
            continue
        metrics["stroke_bend_lines"] += 1.0
        delta = ridge_a.sag_mm - ridge_b.sag_mm
        if delta > metrics["stroke_bend_dev_mm"]:
            metrics["stroke_bend_dev_mm"] = delta
            pts = ridge_a.points
            culprits["stroke_bend_dev_mm"] = {
                "b": stroke.box,
                "a": (int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()) + 1, int(pts[:, 1].max()) + 1),
                "segments_b": [[stroke.x0, stroke.y0, stroke.x1, stroke.y1]],
                "segments_a": [[*pts[i], *pts[i + 1]] for i in range(len(pts) - 1)],
            }
    return metrics, culprits
