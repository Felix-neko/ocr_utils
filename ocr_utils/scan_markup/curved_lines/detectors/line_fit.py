"""Аппроксимация каждой строки прямой и параболой — по образцу dewarp из Leptonica.

Строки собираются классикой: компоненты размера глифа смыкаются по горизонтали (RLSA),
вытянутые сгустки — строки. У каждой берётся центр масс краски по столбцам (уже на копии
300 dpi, ради точности прогиба), сглаживается медианой шириной в две высоты строки и
аппроксимируется прямой и параболой. Метрики полосы — перцентили прогибов, остатков и
разброс наклонов по длинным строкам (см. ``fitting.page_stats``).

Отдельно меряется КРАЙ КОЛОНКИ: начала (и концы) строк, выровненные по одной вертикали,
собираются в кластер, и через них проводится парабола ``x = f(y)``. Прогиб у корешка
сдвигает концы строк по горизонтали тем сильнее, чем ближе к корешку, и край становится
дугой — это второй, независимый от кривизны самих строк признак (ScanTailor строит модель
dewarp в том числе по краям).

Слабое место — сегментация: таблица, заголовок вразрядку, линейка под колонкой дают
«строки», которых нет, и они портят максимумы. Поэтому в сводке нет голого максимума
(только среднее трёх больших), а кривизна считается по длинным строкам. Нейросетевая
сегментация того же самого — ``surya_lines``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.scan_markup.curved_lines.detectors.base import Detector, Frame, Measure, silent
from ocr_utils.scan_markup.curved_lines.fitting import (
    LineFit,
    centreline,
    clusters_1d,
    edge_fit,
    fit_line,
    long_mask,
    page_stats,
    slope_field_stats,
    smooth_median,
)
from ocr_utils.page_layout.orientation.detectors.ink_axis import (
    LINE_ASPECT,
    LINE_MAX_THICKNESS_PX,
    LINE_MIN_LENGTH_PX,
    _smear,
    glyph_mask,
)

# Все размеры — в пикселях копии 150 dpi, если не сказано иное.

# Строка короче стольких своих высот в аппроксимацию не идёт: прогиб растёт с квадратом
# длины, и по короткой подписи кривизну не увидеть, а вот форму букв — запросто.
MIN_LENGTH_HEIGHTS = 8
MIN_HEIGHT_PX = 4

# СБОРКА СТРОКИ ИЗ КУСКОВ. Смыкание RLSA с зазором 8 px (1.35 мм) рвёт выключенную строку
# по широким пробелам: в наборе журнала пробел растягивается до 2 мм, и строка распадается
# на два-три куска, каждый вдвое короче — а прогиб куска вчетверо меньше прогиба строки.
# Замер на эталонных полосах: без сборки p90 прогиба у кривых полос 0.08 высоты, у прямых
# 0.09 — ничего не разделяет. Поэтому куски на одной базовой линии сцепляются обратно:
# зазор по x не больше стольких высот, разница ординат центров — не больше такой доли
# высоты, высоты похожи.
LINK_GAP_HEIGHTS = 2.5
LINK_DY_HEIGHTS = 0.35
LINK_HEIGHT_RATIO = 1.6

# Сцеплять через межколонник нельзя: строки соседних колонок стоят на одной базовой линии
# и зазор между ними (4-5 мм) меньше, чем разрешённый зазор в высотах. Поэтому перед
# сборкой ищутся вертикальные полосы без строк — межколонники и поля — и сцепка через
# них запрещена. Межколонник: доля краски меньше такой от медианы по столбцам и ширина
# не меньше SEPARATOR_MIN_PX.
SEPARATOR_FRAC = 0.12
SEPARATOR_MIN_PX = 10
# Межколонники по лентам высоты (``column_separators_banded``): заголовок или подпись на всю
# ширину перекрывает пустую полосу, и по профилю всей страницы межколонник пропадает
# (1976/02 с.84 в geometry_regression). Лента — BAND_PX по высоте (40 мм при 150 dpi) с шагом
# в половину; межколонник страницы — полоса, пустая не меньше чем в BAND_SHARE лент с текстом.
BAND_PX = 236
BAND_SHARE = 0.6

# Окно медианного сглаживания центр-линии, в высотах строки (на копии 300 dpi).
SMOOTH_HEIGHTS = 2.0

# Меньше стольких пригодных строк — сводку не строить.
MIN_LINES = 8
# И меньше стольких ДЛИННЫХ строк — тоже: таблица, оглавление и выходные данные дают
# десяток «строк» из ячеек и подписей, и по ним прогиб и разброс наклонов — шум. Замер по
# паку: ложные флаги на таблицах и оглавлениях имели 10-25 длинных строк, текстовые полосы — 40-100.
MIN_LONG_LINES = 30

# Край колонки: начала строк в пределах стольких медианных высот друг от друга по x —
# один край; не меньше стольких строк; парабола проводится не через сами начала (абзацный
# отступ — выброс в полторы-три высоты), а через медианы по горизонтальным полосам.
EDGE_TOL_HEIGHTS = 1.5
EDGE_MIN_LINES = 8
EDGE_BANDS = 6

# Шаг выборки центр-линии в raw (для оверлея), px копии 150 dpi.
RAW_STEP = 16


def _line_blobs(mask150: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Смыкание, метки и статистика сгустков."""
    smeared = _smear(mask150, horizontal=True)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(smeared, 8)
    if count <= 1:
        return smeared, labels, np.empty((0, 5), int)
    return smeared, labels, stats


def column_separators(smeared: np.ndarray) -> list[tuple[int, int]]:
    """Вертикальные полосы без строк — межколонники и поля, как ``(x0, x1)``."""
    profile = smeared.sum(axis=0).astype(np.float64)
    if profile.max() <= 0:
        return [(0, smeared.shape[1])]
    positive = profile[profile > 0]
    level = SEPARATOR_FRAC * float(np.median(positive))
    empty = profile < level
    separators: list[tuple[int, int]] = []
    start = None
    for x, flag in enumerate(empty):
        if flag and start is None:
            start = x
        if not flag and start is not None:
            if x - start >= SEPARATOR_MIN_PX:
                separators.append((start, x))
            start = None
    if start is not None and smeared.shape[1] - start >= SEPARATOR_MIN_PX:
        separators.append((start, smeared.shape[1]))
    return separators


def column_separators_banded(smeared: np.ndarray) -> list[tuple[int, int]]:
    """Межколонники по лентам высоты: полоса пуста в большинстве лент с текстом.

    В отличие от :func:`column_separators`, заголовок на всю ширину страницы межколонник не
    ломает: он занимает одну-две ленты, а колонки — остальные. Ленты без краски (поля,
    пустой низ) в голосовании не участвуют.

    Args:
        smeared: маска строк после горизонтального смыкания (как для ``column_separators``).

    Returns:
        Полосы ``(x0, x1)`` по всей высоте страницы — межколонники и поля.
    """
    height, width = smeared.shape
    if height <= BAND_PX:
        return column_separators(smeared)
    votes = np.zeros(width, dtype=np.int32)
    bands = 0
    for y0 in range(0, height - BAND_PX + 1, BAND_PX // 2):
        band = smeared[y0 : y0 + BAND_PX]
        profile = band.sum(axis=0).astype(np.float64)
        positive = profile[profile > 0]
        if positive.size < width * 0.1:  # лента почти без краски — не голосует
            continue
        bands += 1
        votes += profile < SEPARATOR_FRAC * float(np.median(positive))
    if bands == 0:
        return [(0, width)]
    empty = votes >= max(1, int(np.ceil(BAND_SHARE * bands)))
    separators: list[tuple[int, int]] = []
    start = None
    for x, flag in enumerate(empty):
        if flag and start is None:
            start = x
        if not flag and start is not None:
            if x - start >= SEPARATOR_MIN_PX:
                separators.append((start, x))
            start = None
    if start is not None and width - start >= SEPARATOR_MIN_PX:
        separators.append((start, width))
    return separators


def _crosses(separators: list[tuple[int, int]], x0: float, x1: float) -> bool:
    """Лежит ли между x0 и x1 межколонник."""
    return any(s0 >= x0 - 1 and s1 <= x1 + 1 for s0, s1 in separators)


def link_spans(stats: np.ndarray, separators: list[tuple[int, int]]) -> list[list[int]]:
    """Сцепляет куски одной строки в цепочки; возвращает списки индексов сгустков."""
    candidates = [
        index
        for index in range(1, stats.shape[0])
        if MIN_HEIGHT_PX <= stats[index, cv2.CC_STAT_HEIGHT] <= LINE_MAX_THICKNESS_PX
        and stats[index, cv2.CC_STAT_WIDTH] >= stats[index, cv2.CC_STAT_HEIGHT]
    ]
    candidates.sort(key=lambda index: stats[index, cv2.CC_STAT_LEFT])
    used: set[int] = set()
    spans: list[list[int]] = []
    for head in candidates:
        if head in used:
            continue
        chain = [head]
        used.add(head)
        while True:
            x, y, w, h = (int(stats[chain[-1], k]) for k in (0, 1, 2, 3))
            cy, right = y + h / 2.0, x + w
            best = None
            for other in candidates:
                if other in used:
                    continue
                ox, oy, ow, oh = (int(stats[other, k]) for k in (0, 1, 2, 3))
                if ox < right - 2 or ox - right > LINK_GAP_HEIGHTS * max(h, oh):
                    continue
                if abs(oy + oh / 2.0 - cy) > LINK_DY_HEIGHTS * max(h, oh):
                    continue
                if max(h, oh) > LINK_HEIGHT_RATIO * min(h, oh) or _crosses(separators, right, ox):
                    continue
                if best is None or ox < int(stats[best, cv2.CC_STAT_LEFT]):
                    best = other
            if best is None:
                break
            chain.append(best)
            used.add(best)
        spans.append(chain)
    return spans


def _edge_sagitta(values: np.ndarray, y_mid: np.ndarray, median_h: float) -> tuple[float, float, list, list] | None:
    """Прогиб края по медианам полос: (sag_rel, resid_rel, точки полос, кривая)."""
    order = np.argsort(y_mid)
    values, y_mid = values[order], y_mid[order]
    bands = np.array_split(np.arange(values.size), min(EDGE_BANDS, values.size // 3 or 1))
    band_x = np.array([np.median(values[band]) for band in bands if band.size])
    band_y = np.array([np.median(y_mid[band]) for band in bands if band.size])
    result = edge_fit(band_x, band_y)
    if result is None:
        return None
    sagitta, a, b, c = result
    lin = np.polyfit(band_y, band_x, 1)
    resid = float(np.sqrt(np.mean((band_x - np.polyval(lin, band_y)) ** 2)))
    centre = float(band_y.mean())
    curve = [
        [round(float(a + b * (yy - centre) + c * (yy - centre) ** 2), 1), round(float(yy), 1)]
        for yy in np.linspace(band_y[0], band_y[-1], 24)
    ]
    points = [[round(float(v), 1), round(float(yy), 1)] for v, yy in zip(band_x, band_y)]
    return abs(sagitta) / median_h, resid / median_h, points, curve


@dataclass(frozen=True)
class LineSample:
    """Строка после сегментации: бокс на копии 150 dpi и центр-линия на копии 300 dpi.

    ``xs``/``ys`` — в координатах КОПИИ 300 dpi (не выреза), ``ys`` уже сглажены медианой.
    Одна выборка кормит два детектора: ``line_fit`` (аппроксимации целиком) и ``end_curl``
    (кончики строк), поэтому сегментация делается один раз и лежит здесь.
    """

    x: int
    y: int
    x_end: int
    y_end: int
    h_line: float
    xs: np.ndarray
    ys: np.ndarray
    weights: np.ndarray


def line_samples(frame: Frame, banded: bool = False) -> tuple[list[LineSample], list[tuple[int, int]]]:
    """Строки полосы (RLSA + сборка кусков) с центр-линиями и межколонники.

    Args:
        frame: подготовленный кадр с копиями 150 и 300 dpi.
        banded: искать межколонники по лентам высоты (``column_separators_banded``), а не по
            профилю всей страницы; для ``curved_lines`` остаётся прежний способ.

    Returns:
        Строки и полосы-межколонники ``(x0, x1)`` на копии 150 dpi.
    """
    gray150, gray300 = frame.gray150, frame.gray300
    threshold, _ = cv2.threshold(gray150, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    mask150 = glyph_mask(gray150)
    smeared, labels, stats = _line_blobs(mask150)
    separators = column_separators_banded(smeared) if banded else column_separators(smeared)
    ink300 = gray300 <= threshold
    samples: list[LineSample] = []
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
        # Краска строки на копии 300 dpi, ограниченная СВОИМИ сгустками: соседняя строка
        # или линейка в тот же прямоугольник попасть не должны.
        own = np.isin(labels[y:y_end, x:x_end], span).astype(np.uint8)
        own300 = cv2.resize(own, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
        crop = ink300[2 * y : 2 * y_end, 2 * x : 2 * x_end]
        own300 = own300[: crop.shape[0], : crop.shape[1]]
        xs, ys, weights = centreline(crop & (own300 > 0))
        if xs.size == 0:
            continue
        ys = smooth_median(ys, int(SMOOTH_HEIGHTS * 2 * h_line))
        samples.append(LineSample(x, y, x_end, y_end, h_line, xs + 2 * x, ys + 2 * y, weights))
    return samples, separators


def measure(frame: Frame, keep_raw: bool = False) -> Measure:
    gray150 = frame.gray150
    samples, separators = line_samples(frame)

    fits: list[LineFit] = []
    heights: list[float] = []
    raw_lines: list[dict] = []
    for sample in samples:
        x, y, w, h, h_line = sample.x, sample.y, sample.x_end - sample.x, sample.y_end - sample.y, sample.h_line
        xs, ys = sample.xs - 2 * x, sample.ys - 2 * y
        fit = fit_line(xs, ys, sample.weights)
        if fit is None:
            continue
        # Переводим в координаты копии 150 dpi: там же живут сетка тайлов и оверлей.
        fit = LineFit(
            slope_deg=fit.slope_deg,
            curvature=fit.curvature * 2.0,  # px⁻¹ масштабируется вместе с длиной
            sagitta=fit.sagitta / 2.0,
            resid_lin=fit.resid_lin / 2.0,
            resid_quad=fit.resid_quad / 2.0,
            length=fit.length / 2.0,
            n=fit.n,
            x0=x + fit.x0 / 2.0,
            y0=y + fit.y0 / 2.0,
            x1=x + fit.x1 / 2.0,
            y1=y + fit.y1 / 2.0,
        )
        fits.append(fit)
        heights.append(h_line)
        if keep_raw:
            step = RAW_STEP * 2
            points = [[round(x + px / 2.0, 1), round(y + py / 2.0, 1)] for px, py in zip(xs[::step], ys[::step])]
            raw_lines.append(
                {
                    "box": [x, y, w, h],
                    "slope": round(fit.slope_deg, 3),
                    "sag_rel": round(abs(fit.sagitta) / h_line, 3),
                    "resid_rel": round(fit.resid_lin / h_line, 3),
                    "pts": points,
                }
            )

    raw = (
        {"w": gray150.shape[1], "h": gray150.shape[0], "lines": raw_lines, "edges": [], "separators": separators}
        if keep_raw
        else None
    )
    if len(fits) < MIN_LINES:
        return Measure(metrics={"lines": float(len(fits))}, note="мало строк", silent=True, raw=raw)

    metrics = page_stats(fits, heights)
    long = long_mask(fits)
    if keep_raw:
        for line, is_long in zip(raw_lines, long):
            line["long"] = int(is_long)
    if int(long.sum()) < MIN_LONG_LINES:
        return Measure(metrics=metrics, note="мало длинных строк", silent=True, raw=raw)

    # Наклон как поле по полосе — по длинным строкам, с весом длины.
    centres_x = np.array([(fit.x0 + fit.x1) / 2.0 for fit in fits])[long] / gray150.shape[1]
    centres_y = np.array([(fit.y0 + fit.y1) / 2.0 for fit in fits])[long] / gray150.shape[0]
    slopes = np.array([fit.slope_deg for fit in fits])[long]
    lengths = np.array([fit.length for fit in fits])[long]
    metrics.update(slope_field_stats(centres_x, centres_y, slopes, lengths))

    # Края колонок: начала и концы длинных строк.
    median_h = float(np.median(np.array(heights)[long])) if long.any() else float(np.median(heights))
    edge_sag, edge_resid, edge_count = 0.0, 0.0, 0
    for side, coordinate in (("start", "x0"), ("end", "x1")):
        values = np.array([getattr(fit, coordinate) for fit in fits])[long]
        y_mid = np.array([(fit.y0 + fit.y1) / 2.0 for fit in fits])[long]
        for cluster in clusters_1d(values, EDGE_TOL_HEIGHTS * median_h, EDGE_MIN_LINES):
            result = _edge_sagitta(values[cluster], y_mid[cluster], median_h)
            if result is None:
                continue
            sag_rel, resid_rel, points, curve = result
            edge_count += 1
            edge_sag = max(edge_sag, sag_rel)
            edge_resid = max(edge_resid, resid_rel)
            if keep_raw:
                raw["edges"].append({"side": side, "sag_rel": round(sag_rel, 3), "pts": points, "fit": curve})
    metrics.update({"edge_sag_rel_max": edge_sag, "edge_resid_rel_max": edge_resid, "edge_clusters": float(edge_count)})
    return Measure(metrics=metrics, raw=raw)


def _colour(sag_rel: float) -> tuple[int, int, int]:
    return (0, 170, 0) if sag_rel < 0.15 else (0, 200, 255) if sag_rel < 0.3 else (0, 0, 255)


def draw(canvas: np.ndarray, raw: dict, scale: float) -> None:
    """Центр-линии строк (цвет — прогиб), края колонок синим, межколонники серым."""
    for x0, x1 in raw.get("separators", []):
        cv2.rectangle(canvas, (int(x0 * scale), 0), (int(x1 * scale), canvas.shape[0]), (235, 235, 235), -1)
    for line in raw.get("lines", []):
        points = np.array([[px * scale, py * scale] for px, py in line["pts"]], np.int32)
        if len(points) < 2:
            continue
        thickness = 2 if line.get("long") else 1
        cv2.polylines(canvas, [points.reshape(-1, 1, 2)], False, _colour(line["sag_rel"]), thickness, cv2.LINE_AA)
    for edge in raw.get("edges", []):
        curve = np.array([[px * scale, py * scale] for px, py in edge["fit"]], np.int32)
        cv2.polylines(canvas, [curve.reshape(-1, 1, 2)], False, (255, 120, 0), 2, cv2.LINE_AA)
        for px, py in edge["pts"]:
            cv2.circle(canvas, (int(px * scale), int(py * scale)), 3, (255, 120, 0), 1, cv2.LINE_AA)


ALGORITHM = Detector(
    name="line_fit",
    summary="прямая и парабола через центр-линию каждой строки (классическая сегментация RLSA), край колонки",
    stage="cpu",
    # Пороги — по 14 эталонным полосам (8 кривых, 6 прямых), см. README: у прямых p90
    # прогиба 0.04-0.12 высоты, у кривых 0.19-0.33 (две «лёгкие» — 0.07 и 0.10); остаток
    # наклона от плоскости у прямых до 0.17°, у кривых 0.23-1.0° (одна лёгкая — 0.09).
    # Край колонки (edge_*) классы не разводит (абзацные отступы и висячие строки шумят
    # сильнее прогиба) и флага не ставит — остаётся в CSV.
    # После прогона по паку пороги подняты к p95-p97 распределения: при p90-порогах
    # (0.15 / 0.6 / 0.2) флаговалось 13% полос, и выборка глазами показала, что слабые
    # флаги — шум сегментации на ровном тексте.
    thresholds={"sagitta_rel_p90": 0.18, "sagitta_rel_max3": 0.28, "slope_spread_deg": 0.9, "slope_resid_deg": 0.3},
    version=3,
    run=measure,
    draw=draw,
)
