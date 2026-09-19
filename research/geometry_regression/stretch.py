"""Строки по одним и тем же глифам: наклон, волна и вертикальное растяжение куска за куском.

Центр-линия строки (``line_fit``) на бинарном рендере иногда цепляет мусор: 1966/05 с.70 —
3 мм «наклона» на ровной строке; а сопоставление строк по предсказанию поля на почти пустой
странице промахивалось на строку (1967/02 с.97: «Корректор» против «Техн. редактор»).
Поэтому здесь всё меряется ПО ОДНИМ И ТЕМ ЖЕ ГЛИФАМ: строка B режется на куски вдоль x,
каждый кусок ищется в полосе A вокруг предсказанного места (нормированная корреляция), и
пара строк считается ПОДТВЕРЖДЁННОЙ, только если нашлись не меньше трёх кусков. Дальше по
найденным кускам:

* наклон — прямая через положения кусков: в B по центр-линии ``line_fit`` (она строится по
  своим сгусткам), в A — та же плюс сдвиги кусков (субпиксельно, парабола по отклику
  корреляции); уход конца строки в мм (длина × sin), «стало − было» — порча, обратное — выигрыш;
* волна — остаток положений кусков от прямой в A минус то же в B; положение куска в B —
  центр-линия ``line_fit`` (по своим сгусткам: центроид полосы вокруг строки цеплял соседей),
  в A — то же плюс найденный сдвиг; распрямление волнистой строки даёт минус, а не плюс;
* растяжение — для высоких строк (≥ 4 мм) перебор вертикального масштаба куска
  0.90–1.10 с уточнением параболой; клин масштаба вдоль строки × высота = мм ухода верха
  относительно низа (1967/03 с.36: 0.24 мм при ровном низе).

Одинаковые глифы в B и A — шум формы букв сокращается, остаётся то, что сделал FineReader.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from research.geometry_regression import mm_to_px, px_to_mm
from research.geometry_regression.field import Field
from research.geometry_regression.regions import TextLine
from research.geometry_regression.render import RENDER_DPI

# Кусок строки вдоль x (мм) и минимальное число найденных кусков, чтобы пара строк считалась
# подтверждённой (идёт в сводки страницы).
CHUNK_MM = 8.0
MIN_CHUNKS = 3
# Для попарных метрик самой строки (наклон, волна, растяжение) кусков нужно больше и они должны
# покрывать заметную долю строки: шапка акцидентным шрифтом на 1966/04 с.2 дала 3 куска из 15,
# и центр-линия в трёх точках «наклонилась» на 7.7°.
MIN_CHUNKS_METRIC = 5
MIN_CHUNK_COVERAGE = 0.4
# Короткие заголовки рубрик (15-25 мм) идут только в ВЫИГРЫШ по наклону, в градусах: рубрика
# «ИНФОРМАЦИЯ» на 1968/02 с.92 выровнялась, но по длине в мм выигрыша не набирала.
GAIN_MIN_LENGTH_MM = 15.0
# Честная пара: длины строк B и A совпадают не хуже этой доли, и все найденные куски лежат в
# боксе A (с припуском в высоту строки). Иначе строка в A распалась на куски и куски B
# цепляются за соседнюю строку — ложная «волна» (1968/08 с.86).
MIN_LENGTH_MATCH = 0.85
# Растяжение меряется только по высоким строкам: на корпусе 1 % масштаба меньше полупикселя.
STRETCH_MIN_HEIGHT_MM = 4.0
STRETCH_MIN_LENGTH_MM = 40.0
# Полоса вокруг строки (в долях высоты) и окна поиска: широкое при масштабе 1, узкое при переборе.
BAND_HEIGHTS = 0.5
SEARCH_DY_HEIGHTS = 0.3
SEARCH_DX_MM = 1.0
FINE_PX = 3
SCALES = np.round(np.arange(0.90, 1.1001, 0.01), 2)
MIN_PEAK = 0.6
MIN_INK_FRAC = 0.02


@dataclass(frozen=True)
class Chunk:
    x: float  # середина куска в B, px рендера
    dy: float  # сдвиг найденного места в A относительно предсказанного, px, субпиксельно
    peak: float
    scale: float | None  # вертикальный масштаб A/B, если мерился


def _match(template: np.ndarray, region: np.ndarray):
    """Пик корреляции ``(peak, px, py, py_subpx)``; ``py_subpx`` уточнён параболой по y."""
    if region.shape[0] < template.shape[0] or region.shape[1] < template.shape[1]:
        return None
    response = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
    _, peak, _, (px, py) = cv2.minMaxLoc(response)
    sub = float(py)
    if 0 < py < response.shape[0] - 1:
        up, mid, down = response[py - 1, px], response[py, px], response[py + 1, px]
        denom = up - 2 * mid + down
        if abs(denom) > 1e-9:
            sub += float(np.clip(0.5 * (up - down) / denom, -0.5, 0.5))
    return float(peak), px, py, sub


def _scale(chunk: np.ndarray, band_a: np.ndarray, x_hit: int, y_hit: int) -> float | None:
    """Вертикальный масштаб куска в A относительно B: перебор с уточнением параболой."""
    h, w = chunk.shape
    peaks = []
    for s in SCALES:
        hs = max(2, int(round(h * s)))
        scaled = cv2.resize(chunk, (w, hs), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        yc = y_hit + (h - hs) // 2
        pad = FINE_PX + abs(h - hs) // 2 + 1
        found = _match(scaled, band_a[max(0, yc - pad) : yc + hs + pad, max(0, x_hit - FINE_PX) : x_hit + w + FINE_PX])
        peaks.append(found[0] if found is not None else -1.0)
    peaks = np.array(peaks)
    i = int(peaks.argmax())
    if peaks[i] < MIN_PEAK:
        return None
    s = float(SCALES[i])
    if 0 < i < len(SCALES) - 1:
        left, mid, right = peaks[i - 1], peaks[i], peaks[i + 1]
        denom = left - 2 * mid + right
        if abs(denom) > 1e-9:
            s += float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5)) * float(SCALES[1] - SCALES[0])
    return s


def line_chunks(
    ink_b: np.ndarray, ink_a: np.ndarray, line_b: TextLine, field: Field | None, dpi: float, want_scale: bool
) -> list[Chunk]:
    """Куски строки B, найденные в A (пиксели рендера ``RENDER_DPI``)."""
    k = RENDER_DPI / dpi
    h = line_b.height * k
    margin = int(BAND_HEIGHTS * h)
    by0, by1 = int(line_b.y0 * k) - margin, int(line_b.y1 * k) + margin
    bx0, bx1 = int(line_b.x0 * k), int(line_b.x1 * k)
    chunk_px = mm_to_px(CHUNK_MM, RENDER_DPI)
    if by0 < 0 or by1 > ink_b.shape[0] or bx1 - bx0 < MIN_CHUNKS * chunk_px:
        return []
    corner = np.array([[line_b.x0, line_b.y0]], dtype=np.float64)
    moved = (field.transform(corner)[0] if field is not None else corner[0]) * k
    shift_x, shift_y = moved[0] - bx0, moved[1] - int(line_b.y0 * k)
    dx, dy = mm_to_px(SEARCH_DX_MM, RENDER_DPI), int(SEARCH_DY_HEIGHTS * h) + 4
    chunks: list[Chunk] = []
    for i in range((bx1 - bx0) // chunk_px):
        cx0 = bx0 + i * chunk_px
        chunk = ink_b[by0:by1, cx0 : cx0 + chunk_px]
        if chunk.mean() / 255.0 < MIN_INK_FRAC:
            continue
        x_pred, y_pred = int(round(cx0 + shift_x)), int(round(by0 + shift_y))
        if x_pred < 0 or y_pred < 0:
            continue
        x0, y0 = max(0, x_pred - dx), max(0, y_pred - dy)
        found = _match(chunk, ink_a[y0 : y_pred + chunk.shape[0] + dy, x0 : x_pred + chunk_px + dx])
        if found is None or found[0] < MIN_PEAK:
            continue
        peak, px, py, sub = found
        x_hit, y_hit = x0 + px, y0 + py
        scale = _scale(chunk, ink_a, x_hit, y_hit) if want_scale else None
        chunks.append(Chunk(cx0 + chunk_px / 2.0, y0 + sub - (by0 + shift_y), peak, scale))
    return chunks


def _resid(xs: np.ndarray, ys: np.ndarray) -> float:
    """RMS остаток точек от прямой."""
    return float(np.sqrt(np.mean((ys - np.polyval(np.polyfit(xs, ys, 1), xs)) ** 2)))


def _chunks_inside(chunks: list[Chunk], line_b: TextLine, line_a: TextLine, field: Field | None, dpi: float) -> bool:
    """Все найденные куски лежат в боксе строки A по вертикали (припуск — высота строки)."""
    k = RENDER_DPI / dpi
    corner = np.array([[line_b.x0, line_b.y0]], dtype=np.float64)
    moved = (field.transform(corner)[0] if field is not None else corner[0]) * k
    shift_y = moved[1] - line_b.y0 * k
    # Припуск — две высоты: у сильно изогнутой строки B куски после распрямления законно
    # уходят за бокс A на высоту строки (1968/05 с.55).
    y_lo, y_hi = (line_a.y0 - 2 * line_a.height) * k, (line_a.y1 + 2 * line_a.height) * k
    for chunk in chunks:
        # dy куска — сдвиг относительно предсказанного места; предсказанное = строка B + сдвиг поля.
        y_found = line_b.cy * k + shift_y + chunk.dy
        if not y_lo <= y_found <= y_hi:
            return False
    return True


def _tilt_gain_deg(metrics: dict[str, float], chunks: list[Chunk], line_b: TextLine) -> None:
    """Выигрыш по наклону короткой строки в градусах — в ``metrics["line_tilt_gain_deg"]``."""
    xs = np.array([c.x for c in chunks])
    dys = np.array([c.dy for c in chunks])
    cy_b = np.interp(xs, line_b.xs, line_b.ys)
    tilt_b = float(np.degrees(np.arctan(np.polyfit(xs, cy_b, 1)[0])))
    tilt_a = float(np.degrees(np.arctan(np.polyfit(xs, cy_b + dys, 1)[0])))
    metrics["line_tilt_gain_deg"] = max(metrics["line_tilt_gain_deg"], abs(tilt_b) - abs(tilt_a))


def glyph_line_metrics(
    gray300_b: np.ndarray,
    gray300_a: np.ndarray,
    pairs: list[tuple[TextLine, TextLine]],
    field: Field | None,
    dpi: float,
    min_len_mm: float,
) -> tuple[dict[str, float], dict, list[tuple[TextLine, TextLine]]]:
    """Метрики строк по глифам, рамки виновников (пиксели ``dpi``) и подтверждённые пары."""
    ink_b, ink_a = 255 - gray300_b, 255 - gray300_a
    k = RENDER_DPI / dpi
    min_len = mm_to_px(min_len_mm, dpi)
    stretch_h, stretch_len = mm_to_px(STRETCH_MIN_HEIGHT_MM, dpi), mm_to_px(STRETCH_MIN_LENGTH_MM, dpi)
    metrics = {
        "lines_verified": 0.0,
        "stretch_lines": 0.0,
        "line_tilt_dev_max_mm": 0.0,
        "line_tilt_gain_mm": 0.0,
        "line_tilt_gain_deg": 0.0,
        "line_glyph_wobble_max": 0.0,
        "line_stretch_mm_max": 0.0,
    }
    gain_min_len = mm_to_px(GAIN_MIN_LENGTH_MM, dpi)
    culprits: dict = {}
    best: dict[str, tuple[float, TextLine, TextLine] | None] = {key: None for key in metrics if key.startswith("line_")}
    verified: list[tuple[TextLine, TextLine]] = []
    for line_b, line_a in pairs:
        if line_b.length < gain_min_len:
            continue
        if min(line_b.length, line_a.length) < MIN_LENGTH_MATCH * max(line_b.length, line_a.length):
            continue
        want_scale = line_b.height >= stretch_h and line_b.length >= stretch_len
        chunks = line_chunks(ink_b, ink_a, line_b, field, dpi, want_scale)
        if len(chunks) < MIN_CHUNKS or not _chunks_inside(chunks, line_b, line_a, field, dpi):
            continue
        possible = max(1, int(line_b.length * k) // mm_to_px(CHUNK_MM, RENDER_DPI))
        covered = len(chunks) >= MIN_CHUNKS_METRIC and len(chunks) / possible >= MIN_CHUNK_COVERAGE
        # Пара подтверждена (идёт в сводки и выигрыш) уже при MIN_CHUNKS кусках; в попарные
        # метрики самой строки — только длинная и хорошо покрытая.
        verified.append((line_b, line_a))
        metrics["lines_verified"] += 1.0
        if line_b.length < min_len or not covered:
            if covered:
                _tilt_gain_deg(metrics, chunks, line_b)
            continue
        xs = np.array([c.x for c in chunks])
        dys = np.array([c.dy for c in chunks])
        cy_b = np.interp(xs, line_b.xs, line_b.ys)  # положение строки B на середине куска
        length_mm = px_to_mm(xs[-1] - xs[0] + mm_to_px(CHUNK_MM, RENDER_DPI), RENDER_DPI)
        line_fit = np.polyfit(xs, dys, 1)
        wobble_px = _resid(xs, cy_b + dys) - _resid(xs, cy_b)
        wobble = wobble_px / (line_b.height * k)
        values = {"line_glyph_wobble_max": float(wobble)}
        # Наклон обеих версий — прямые по положениям одних и тех же кусков: B по центр-линии,
        # A — она же плюс найденные сдвиги. Не «наклон B + доворот»: у изогнутой строки B
        # сдвиги кусков — это её распрямление, а не поворот (1966/05 с.86).
        tilt_b = float(np.degrees(np.arctan(np.polyfit(xs, cy_b, 1)[0])))
        tilt_a = float(np.degrees(np.arctan(np.polyfit(xs, cy_b + dys, 1)[0])))
        dev = length_mm * (np.sin(np.radians(abs(tilt_a))) - np.sin(np.radians(abs(tilt_b))))
        values.update({"line_tilt_dev_max_mm": dev, "line_tilt_gain_mm": -dev})
        metrics["line_tilt_gain_deg"] = max(metrics["line_tilt_gain_deg"], abs(tilt_b) - abs(tilt_a))
        scales = np.array([c.scale for c in chunks if c.scale is not None])
        if len(scales) >= MIN_CHUNKS:
            metrics["stretch_lines"] += 1.0
            xs_s = np.array([c.x for c in chunks if c.scale is not None])
            trend = abs(np.polyfit(xs_s, scales, 1)[0] * (xs_s[-1] - xs_s[0]))
            # Клин — порча лишь сверх того, на сколько FineReader довернул и выпрямил строку (по
            # сдвигам кусков): выравнивая наклонный или изогнутый заголовок, он оставляет клин
            # высоты букв (1968/02 с.92, 1968/03 с.89), и глаз видит выпрямление, а не клин; на
            # ровном заголовке (1967/03 с.36) доворота нет и клин — порча. Цена правила — заголовок,
            # который FineReader и довернул, и растянул (1967/05 с.26): наклон B по центр-линии на
            # жирном кегле ненадёжен, знак доворота не проверить, клин прощается.
            turn_mm = length_mm * abs(np.sin(np.arctan(line_fit[0])))
            straightened_mm = max(0.0, px_to_mm(-wobble_px, RENDER_DPI))
            values["line_stretch_mm_max"] = max(
                0.0, float(trend * px_to_mm(line_b.height * k, RENDER_DPI)) - turn_mm - straightened_mm
            )
        for name, value in values.items():
            if best[name] is None or value > best[name][0]:
                best[name] = (float(value), line_b, line_a)
    for name, item in best.items():
        if item is not None:
            metrics[name] = item[0]
            culprits[name] = {"b": item[1].box, "a": item[2].box}
    return metrics, culprits, verified
