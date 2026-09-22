"""Строки по тем же глифам: наклон по проекции, волна, растяжение — и выигрыш только там, где он виден.

Отличия от ``stretch.glyph_line_metrics`` ядра:

* **наклон** строки в B и в A — по проекции полосы строки (угол, при котором горизонтальный
  профиль краски самый резкий, перебор грубо/точно), а не по центр-линии ``line_fit``: на
  акцидентном шрифте центр-линия скачет на ±10 px, и довернутый на 0.86° заголовок 1969/06
  с.79 выходил «выигрышем» 0.08°. Разность наклонов сверяется с доворотом по сдвигам кусков
  глифов (``polyfit(xs, dys)``): расходятся больше ``CROSS_TOL_DEG`` — строка не мерится;
* **выигрыш по наклону** (``line_tilt_gain_*``) — только по отдельным строкам (заголовок:
  выше корпуса в ``HEADING_HEIGHT_RATIO`` раз или с пустотой в ``HEADING_GAP_HEIGHTS`` высот
  сверху и снизу): по одной корпусной строке из 60 с шумной центр-линией набирался «выигрыш»
  0.7° (1968/03 с.8); выигрыш корпуса — сводки ``line_metrics`` по колонкам;
* **форма строки — относительными мерами** (решение пользователя 2026-09-22): базовая линия по
  центроидам краски кусков (B) и та же плюс сдвиги (A); ``line_bend_ratio`` — рост размаха остатка от
  прямой относительно длины строки (дуга заголовка 1973/10 с.32: +4.4·10⁻³, выпрямленный 1968/02 с.92:
  −3.9·10⁻³), ``line_step_ratio`` — рост наибольшего скачка между соседними кусками относительно
  высоты букв (ступенька «Севера» 1968/09 с.52: +0.05), ``line_wedge_ratio`` — линейный тренд
  масштаба кусков (клин), ``line_stretch_ratio`` — размах масштабов p10–p90 (неравномерное
  растяжение); клин при настоящем выпрямлении строки режется до ``WEDGE_FORGIVEN_MAX``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.geometry_regression import mm_to_px, px_to_mm
from ocr_utils.geometry_regression.field import Field
from ocr_utils.geometry_regression.regions import TextLine
from ocr_utils.geometry_regression.render import RENDER_DPI
from ocr_utils.geometry_regression.stretch import (
    BAND_HEIGHTS,
    CHUNK_MM,
    GAIN_MIN_LENGTH_MM,
    MIN_CHUNK_COVERAGE,
    MIN_CHUNKS,
    MIN_CHUNKS_METRIC,
    MIN_LENGTH_MATCH,
    SEARCH_DY_HEIGHTS,
    STRETCH_MIN_HEIGHT_MM,
    STRETCH_MIN_LENGTH_MM,
    _chunks_inside,
    line_chunks,
)

Box = tuple[int, int, int, int]

# Перебор угла проекции: грубый шаг по всему диапазону, точный — вокруг лучшего.
TILT_RANGE_DEG = 3.0
TILT_COARSE_DEG = 0.25
TILT_FINE_DEG = 0.05
# Полоса строки для проекции — бокс с припуском в долях высоты (соседняя строка дальше 0.5 h).
TILT_BAND_HEIGHTS = 0.15
# Размытие полосы перед перебором углов (px рендера): снимает преимущество нулевого угла.
TILT_BLUR_PX = 1.0
# Разность наклонов по проекции и доворот по кускам глифов расходятся сильнее — наклон не мерится.
CROSS_TOL_DEG = 0.35
# Отдельная строка (заголовок): выше медианы корпуса во столько раз ИЛИ пустота сверху и снизу.
HEADING_HEIGHT_RATIO = 1.5
HEADING_GAP_HEIGHTS = 1.5
# Кусок, чей сдвиг упёрся в край окна поиска (``SEARCH_DY_HEIGHTS`` высот + 4 px, с таким запасом до
# края), найден не там: заголовок, повёрнутый сильнее окна (1972/01 с.92, −2.25° → 0°), на дальнем
# конце давал одинаковые «сдвиги» у последних кусков, и прямая по ним ломалась — ложная дуга.
CLIP_MARGIN_PX = 0.5
# Форма базовой линии по кускам: припуск окна центроида (px рендера) и перцентили размаха остатка.
CENTROID_PAD_PX = 10
# Выпрямление («строка была кривой и стала прямее») мерится полным размахом остатка базовой: здесь
# вопрос в самой кривизне, а не в приросте, и обе версии считаются по одним и тем же глифам.
GAIN_PERCENTILES = (2.0, 98.0)
# Размах остатка берётся усечённым (p10–p90): один кусок, найденный не там, не должен сойти за дугу.
SHAPE_PERCENTILES = (10.0, 90.0)
# Ступенькой считается сдвиг уровня, подтверждённый не менее чем этим числом кусков с каждой стороны.
STEP_MIN_SIDE = 2
# Строка «стала прямее», если кривизна относительно длины упала не меньше чем на столько; тогда клин
# и неравномерность масштаба режутся до WEDGE_FORGIVEN_MAX (побочный эффект выпрямления).
STRAIGHTENED_BEND_RATIO = 1.5e-3
WEDGE_FORGIVEN_MAX = 0.04
# Клин от WEDGE_HARD не прощается и при выпрямлении (1976/08 с.73 — 0.117 при стало-прямее на 0.15 мм).
WEDGE_HARD = 0.10
# Форма считается только по надёжно найденным кускам: медиана пиков корреляции не ниже этой (на
# страницах со снятой трапецией короткие строки в 2–4 мм совпадают на 0.6–0.85 и рисуют дугу из
# промахов: 1966/06 с.62, 1968/05 с.95); худший пик не ограничивается — кусок на самой ступеньке
# совпадает плохо по природе.
SHAPE_MIN_PEAK_MEDIAN = 0.9
# Строка, слипшаяся с соседней глубже этой доли высоты, в форму не идёт.
OVERLAP_MAX_HEIGHTS = 0.2
# Кусков в строке для формы (после медианного сглаживания по трём); строки ниже SHAPE_MIN_HEIGHT_MM не
# меряются; прирост кривизны и ступеньки засчитывается только от SHAPE_MIN_DELTA_MM — у корпусной
# строки в 2 мм промах куска на четверть миллиметра даёт «ступеньку» 0.1 высоты (1968/03 с.8).
SHAPE_MIN_CHUNKS = 5
SHAPE_MIN_HEIGHT_MM = 1.8
SHAPE_MIN_DELTA_MM = 0.3
# Клин прощается и при выпрямлении наклона на STRAIGHTENED_TILT_DEG. Наклон по центроидам кусков,
# когда проекция не сошлась с кусками, отвергнут: на трёхстрочной рубрике 1966/06 с.62 центроиды
# давали +1° из ничего; разрядка 1967/03 с.11 (проекция слепа к разреженным капителям) — известный промах.
STRAIGHTENED_TILT_DEG = 0.3


@dataclass(frozen=True)
class LineTilt:
    """Наклон одной пары строк по проекции (градусы, положительный — вниз направо)."""

    line_b: TextLine
    line_a: TextLine
    tilt_b: float
    tilt_a: float
    length_mm: float
    heading: bool  # отдельная строка (заголовок) — идёт в выигрыш по наклону
    measurable: bool  # длинная и хорошо покрытая кусками — идёт в порчу по наклону


def tilt_summary(
    tilts: list[LineTilt], rot_adjust_deg: float = 0.0, gain: bool = True
) -> tuple[dict[str, float], dict]:
    """Порча и выигрыш по наклону строк с поправкой на подтверждённый доворот страницы.

    Args:
        tilts: Наклоны пар строк (:func:`glyph_line_metrics`).
        rot_adjust_deg: Подтверждённый доворот страницы: вычитается из наклона A — строка,
            повернувшаяся вместе со всей страницей, порчей не считается.
        gain: Считать ли выигрыш (для строк внутри таблиц и рисунков — только он).

    Returns:
        ``line_tilt_dev_max_mm``, ``line_tilt_gain_mm``, ``line_tilt_gain_deg`` и виновники.
    """
    metrics = {"line_tilt_dev_max_mm": 0.0, "line_tilt_gain_mm": 0.0, "line_tilt_gain_deg": 0.0}
    best: dict[str, tuple[float, LineTilt] | None] = {name: None for name in metrics}
    # Поправка на доворот — только если строки от неё в среднем прямее.
    if rot_adjust_deg and tilts:
        plain = float(np.median([abs(t.tilt_a) for t in tilts]))
        moved = float(np.median([abs(t.tilt_a - rot_adjust_deg) for t in tilts]))
        if moved >= plain:
            rot_adjust_deg = 0.0
    for item in tilts:
        tilt_a = item.tilt_a - rot_adjust_deg
        dev = item.length_mm * (np.sin(np.radians(abs(tilt_a))) - np.sin(np.radians(abs(item.tilt_b))))
        values = {}
        if item.measurable:
            values["line_tilt_dev_max_mm"] = float(dev)
        if gain and item.heading:
            values["line_tilt_gain_mm"] = float(-dev)
            values["line_tilt_gain_deg"] = float(abs(item.tilt_b) - abs(tilt_a))
        for name, value in values.items():
            if best[name] is None or value > best[name][0]:
                best[name] = (value, item)
    culprits: dict = {}
    for name, found in best.items():
        if found is not None:
            metrics[name] = found[0]
            culprits[name] = {"b": found[1].line_b.box, "a": found[1].line_a.box}
    return metrics, culprits


def projection_tilt(ink: np.ndarray, x0: int, y0: int, x1: int, y1: int, height: float) -> float | None:
    """Наклон строки (градусы, положительный — вниз направо) по резкости горизонтальной проекции.

    Args:
        ink: Краска (255 − серый) рендера, в котором заданы координаты.
        x0, y0, x1, y1: Бокс строки в пикселях ``ink``.
        height: Высота строки в тех же пикселях.

    Returns:
        Угол или ``None``, если полоса выходит за кадр или почти пуста.
    """
    margin = int(TILT_BAND_HEIGHTS * height)
    y0, y1 = max(0, y0 - margin), min(ink.shape[0], y1 + margin)
    x0, x1 = max(0, x0), min(ink.shape[1], x1)
    band = ink[y0:y1, x0:x1]
    if band.size == 0 or band.mean() < 1.0 or (x1 - x0) < 4 * (y1 - y0):
        return None
    # На бинарном рендере при нулевом угле сдвига нет и интерполяция ничего не размывает —
    # профиль там всегда резче (так «профильный deskew» и был отвергнут в docs/status.md).
    # Размытие полосы до перебора уравнивает углы: интерполяция размытого не меняет резкость.
    band = cv2.GaussianBlur(band.astype(np.float32), (0, 0), TILT_BLUR_PX)
    h, w = band.shape
    cx = w / 2.0

    def sharpness(angle: float) -> float:
        # Сдвиг строк по вертикали пропорционально x: полоса «раскручивается» на угол.
        shift = np.tan(np.radians(angle))
        matrix = np.array([[1.0, 0.0, 0.0], [-shift, 1.0, cx * shift]], dtype=np.float32)
        warped = cv2.warpAffine(band, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        profile = warped.sum(axis=1)
        return float((profile**2).sum())

    coarse = np.arange(-TILT_RANGE_DEG, TILT_RANGE_DEG + 1e-6, TILT_COARSE_DEG)
    best = float(coarse[int(np.argmax([sharpness(a) for a in coarse]))])
    fine = np.arange(best - TILT_COARSE_DEG, best + TILT_COARSE_DEG + 1e-6, TILT_FINE_DEG)
    return float(fine[int(np.argmax([sharpness(a) for a in fine]))])


def is_heading(line: TextLine, lines: list[TextLine]) -> bool:
    """Отдельная строка: крупнее корпуса или с пустотой сверху и снизу в своей колонке."""
    # Строка через межколонник (column −1) сверяется со ВСЕМИ строками: это либо заголовок на всю
    # ширину, либо две корпусные строки, слипшиеся через межколонник (``line_fit`` сшивает их по
    # общей y-полосе). У слипшейся высота корпусная, а над и под ней в колонках стоят соседи —
    # сравнение только с такими же «через межколонник» делало её заголовком, и дыра над межколонником
    # давала ложную дугу (1967/08 с.88, 1966/02 с.65).
    others = [other for other in lines if other is not line and (line.column < 0 or other.column == line.column)]
    if not others:
        return True
    body = float(np.median([other.height for other in others]))
    if line.height >= HEADING_HEIGHT_RATIO * body:
        return True
    gap = HEADING_GAP_HEIGHTS * line.height
    # Сосед — любая строка той же колонки, чья полоса заходит в зазор над или под строкой, в том
    # числе перекрывающая её по y (строка через межколонник рядом с корпусом, 1966/06 с.78: сосед
    # заходил на 2 px, и корпусная строка сходила за заголовок).
    near = [
        other
        for other in others
        if min(line.x1, other.x1) > max(line.x0, other.x0) and other.y1 >= line.y0 - gap and other.y0 <= line.y1 + gap
    ]
    return not near


def glyph_line_metrics(
    gray300_b: np.ndarray,
    gray300_a: np.ndarray,
    pairs: list[tuple[TextLine, TextLine]],
    lines_b: list[TextLine],
    field: Field | None,
    dpi: float,
    min_len_mm: float,
    underlines: list[Box] | None = None,
) -> tuple[dict[str, float], dict, list[tuple[TextLine, TextLine]], list[LineTilt]]:
    """Метрики строк по глифам (кроме наклона), виновники, подтверждённые пары и наклоны строк.

    Наклон (порча и выигрыш) сводится отдельно — :func:`tilt_summary`: ему нужна поправка на
    доворот страницы, которая известна только после всех строк и штрихов.

    Args:
        gray300_b, gray300_a: Рендеры обеих версий в ``RENDER_DPI``.
        pairs: Пары строк B → A (``lines.match_lines``), боксы в пикселях ``dpi``.
        lines_b: Все строки B той же группы (для признака «отдельная строка»).
        field: Поле смещений B → A.
        dpi: Разрешение боксов строк.
        min_len_mm: Строка короче в попарные метрики не идёт.
        underlines: Рамки горизонтальных штрихов B (пиксели ``dpi``): у строки с линейкой в полосе
            кусков форма не меряется — куски цепляются за подчёркивание, и его наклон (порча
            штриха, ловится отдельно) выглядит дугой заголовка (1967/01 с.71).

    Returns:
        Метрики, виновники (пиксели ``dpi``), подтверждённые пары и наклоны по проекции
        (:class:`LineTilt`) тех пар, где проекция и куски глифов согласны.
    """
    ink_b, ink_a = 255 - gray300_b, 255 - gray300_a
    k = RENDER_DPI / dpi
    min_len = mm_to_px(min_len_mm, dpi)
    stretch_h, stretch_len = mm_to_px(STRETCH_MIN_HEIGHT_MM, dpi), mm_to_px(STRETCH_MIN_LENGTH_MM, dpi)
    metrics = {
        "lines_verified": 0.0,
        "stretch_lines": 0.0,
        "line_bend_ratio": 0.0,
        "line_step_ratio": 0.0,
        "line_wedge_ratio": 0.0,
        "line_stretch_ratio": 0.0,
        "line_tilt_skipped": 0.0,
    }
    gain_min_len = mm_to_px(GAIN_MIN_LENGTH_MM, dpi)
    culprits: dict = {}
    best: dict[str, tuple[float, TextLine, TextLine] | None] = {
        key: None for key in metrics if key.startswith("line_") and key != "line_tilt_skipped"
    }
    verified: list[tuple[TextLine, TextLine]] = []
    tilts: list[LineTilt] = []
    half = mm_to_px(CHUNK_MM, RENDER_DPI) // 2
    for line_b, line_a in pairs:
        if line_b.length < gain_min_len:
            continue
        if min(line_b.length, line_a.length) < MIN_LENGTH_MATCH * max(line_b.length, line_a.length):
            continue
        want_scale = line_b.height >= stretch_h and line_b.length >= stretch_len
        chunks = line_chunks(ink_b, ink_a, line_b, field, dpi, want_scale)
        # Куски у края окна поиска по y выбрасываются: их сдвиг — предел окна, а не место в A.
        dy_limit = int(SEARCH_DY_HEIGHTS * line_b.height * k) + 4 - CLIP_MARGIN_PX
        chunks = [c for c in chunks if abs(c.dy) < dy_limit]
        if len(chunks) < MIN_CHUNKS or not _chunks_inside(chunks, line_b, line_a, field, dpi):
            continue
        possible = max(1, int(line_b.length * k) // mm_to_px(CHUNK_MM, RENDER_DPI))
        covered = len(chunks) >= MIN_CHUNKS_METRIC and len(chunks) / possible >= MIN_CHUNK_COVERAGE
        verified.append((line_b, line_a))
        metrics["lines_verified"] += 1.0
        xs = np.array([c.x for c in chunks])
        dys = np.array([c.dy for c in chunks])
        # Доворот по кускам — прямая по их сдвигам: шум формы букв B и A одинаков и сокращается.
        turn_deg = float(np.degrees(np.arctan(np.polyfit(xs, dys, 1)[0])))
        tilt_b = projection_tilt(ink_b, *(int(v * k) for v in line_b.box), line_b.height * k)
        tilt_a = projection_tilt(ink_a, *(int(v * k) for v in line_a.box), line_a.height * k)
        tilt_ok = tilt_b is not None and tilt_a is not None and abs((tilt_a - tilt_b) - turn_deg) <= CROSS_TOL_DEG
        length_px = float(xs[-1] - xs[0] + 2 * half)
        length_mm = px_to_mm(length_px, RENDER_DPI)
        heading = is_heading(line_b, lines_b)
        measurable = line_b.length >= min_len and covered
        # Форма строки — только у отдельных строк (заголовков) от SHAPE_MIN_HEIGHT_MM: у корпуса промах
        # одного куска даёт ложную ступеньку (1968/03 с.8, отсекается порогом SHAPE_MIN_DELTA_MM), а
        # волну корпуса FineReader убирает — её выигрыш считают сводки ``line_metrics``.
        peaks = [c.peak for c in chunks]
        shape_ok = (
            measurable
            and heading
            and not _underlined(line_b, underlines or [])
            and not _overlapped(line_b, lines_b)
            and line_b.height >= mm_to_px(SHAPE_MIN_HEIGHT_MM, dpi)
            and len(chunks) >= SHAPE_MIN_CHUNKS
            and float(np.median(peaks)) >= SHAPE_MIN_PEAK_MEDIAN
        )
        if not tilt_ok:
            metrics["line_tilt_skipped"] += 1.0
        else:
            tilts.append(LineTilt(line_b, line_a, tilt_b, tilt_a, length_mm, heading, measurable))
        if not shape_ok:
            continue
        # Форма мерится по САМИМ СДВИГАМ кусков (A − B по одним и тем же глифам): из них вычитается
        # прямая, то есть общий сдвиг и доворот строки. Что осталось — то, что FineReader сделал со
        # строкой сверх переноса и поворота: дуга и ступенька. Абсолютные базовые линии (центроид
        # краски куска) для этого не годились: шум формы букв в B ±5 px входил в размах и в скачок
        # по-разному в A и B, а выправленный наклон (1972/01 с.92, −2.25° → 0°) сам по себе давал
        # «ступеньку» 3.7 px на кусок.
        resid = dys - np.polyval(np.polyfit(xs, dys, 1), xs)
        lo, hi = np.percentile(resid, SHAPE_PERCENTILES)
        bend_px = float(hi - lo)
        step_px = _level_step(resid)
        height_px = line_b.height * k
        floor = mm_to_px(SHAPE_MIN_DELTA_MM, RENDER_DPI)
        values: dict[str, float] = {
            # Кривизна относительно длины строки и ступенька относительно высоты букв; меньше
            # SHAPE_MIN_DELTA_MM — шум кусков, не считается.
            "line_bend_ratio": float(bend_px / length_px) if bend_px >= floor else 0.0,
            "line_step_ratio": float(step_px / height_px) if step_px >= floor else 0.0,
        }
        scales = np.array([c.scale for c in chunks if c.scale is not None])
        if len(scales) >= MIN_CHUNKS:
            metrics["stretch_lines"] += 1.0
            xs_s = np.array([c.x for c in chunks if c.scale is not None])
            # Клин — линейный тренд масштаба от начала к концу; неравномерность — размах p10–p90:
            # оба безразмерные (растяжение относительно высоты букв).
            values["line_wedge_ratio"] = float(abs(np.polyfit(xs_s, scales, 1)[0] * (xs_s[-1] - xs_s[0])))
            values["line_stretch_ratio"] = float(np.percentile(scales, 90) - np.percentile(scales, 10))
            # Клин при настоящем выпрямлении (строка стала заметно прямее или ровнее — 1968/02 с.92,
            # 1968/03 с.89) глаз прощает; прямее не стала — клин порча целиком (1968/09 с.52, 1976/07 с.31).
            straightened = _bend_gain(ink_b, ink_a, line_b, line_a, xs, dys, k, half) >= STRAIGHTENED_BEND_RATIO or (
                tilt_ok and abs(tilt_b) - abs(tilt_a) >= STRAIGHTENED_TILT_DEG
            )
            if straightened and values["line_wedge_ratio"] < WEDGE_HARD:
                values["line_wedge_ratio"] = min(values["line_wedge_ratio"], WEDGE_FORGIVEN_MAX)
                values["line_stretch_ratio"] = min(values["line_stretch_ratio"], WEDGE_FORGIVEN_MAX)
        for name, value in values.items():
            if best[name] is None or value > best[name][0]:
                best[name] = (float(value), line_b, line_a)
    for name, item in best.items():
        if item is not None:
            metrics[name] = item[0]
            culprits[name] = {"b": item[1].box, "a": item[2].box}
    return metrics, culprits, verified, tilts


def _overlapped(line: TextLine, lines: list[TextLine]) -> bool:
    """Слиплась ли строка с соседней по вертикали (перекрытие боксов больше ``OVERLAP_MAX_HEIGHTS`` её высоты).

    Заголовок, в бокс которого въехала соседняя строка (1967/01 с.71: 1164–1189 и 1183–1209), по
    кускам меряет чужие буквы — «дуга» 0.012 длины из ничего.
    """
    for other in lines:
        if other is line or other.column != line.column or other.x1 <= line.x0 or other.x0 >= line.x1:
            continue
        overlap = min(line.y1, other.y1) - max(line.y0, other.y0)
        if overlap > OVERLAP_MAX_HEIGHTS * line.height:
            return True
    return False


def _underlined(line: TextLine, underlines: list[Box]) -> bool:
    """Есть ли горизонтальный штрих в полосе кусков строки (±``BAND_HEIGHTS`` высоты) на половину её длины."""
    margin = BAND_HEIGHTS * line.height
    for x0, y0, x1, y1 in underlines:
        cy = (y0 + y1) / 2.0
        if line.y0 - margin <= cy <= line.y1 + margin and min(x1, line.x1) - max(x0, line.x0) >= 0.5 * line.length:
            return True
    return False


def _bend_gain(
    ink_b: np.ndarray,
    ink_a: np.ndarray,
    line_b: TextLine,
    line_a: TextLine,
    xs: np.ndarray,
    dys: np.ndarray,
    k: float,
    half: int,
) -> float:
    """Насколько строка стала ПРЯМЕЕ: (размах остатка базовой B − то же в A) в долях длины.

    Здесь абсолютные базовые линии нужны: вопрос не «что изменилось», а «была ли строка кривой и
    выпрямилась». Базовая B — центроид краски куска, A — то же плюс сдвиг куска: шум формы букв
    одинаков в обеих версиях и в разности размахов сокращается.
    """
    y0, y1 = int(line_b.y0 * k) - CENTROID_PAD_PX, int(line_b.y1 * k) + CENTROID_PAD_PX
    base_b = _median3(np.array([_centroid_y(ink_b, int(x - half), int(x + half), y0, y1) for x in xs]))
    base_a = _median3(base_b + dys)
    length_px = float(xs[-1] - xs[0] + 2 * half)
    return (_bend(xs, base_b) - _bend(xs, base_a)) / length_px


def _bend(xs: np.ndarray, ys: np.ndarray) -> float:
    """Размах (``GAIN_PERCENTILES``) остатка ряда от прямой — «насколько ряд кривой» в пикселях."""
    resid = ys - np.polyval(np.polyfit(xs, ys, 1), xs)
    lo, hi = np.percentile(resid, GAIN_PERCENTILES)
    return float(hi - lo)


def _median3(values: np.ndarray) -> np.ndarray:
    """Скользящая медиана по трём точкам с повтором краёв."""
    if len(values) < 3:
        return values
    padded = np.pad(values, 1, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, 3), axis=1)


def _level_step(resid: np.ndarray) -> float:
    """Наибольший СДВИГ УРОВНЯ ряда: |медиана слева − медиана справа| по всем разбиениям.

    Ступенька — это когда часть строки уехала и там осталась (``STEP_MIN_SIDE`` кусков с каждой
    стороны), а не когда один кусок нашёлся не там: одиночный промах даёт два скачка подряд (туда
    и обратно) и медианы половин почти не двигает (1968/02 с.92 — первый кусок мимо на 3.8 px).
    """
    if len(resid) < 2 * STEP_MIN_SIDE:
        return 0.0
    return max(
        abs(float(np.median(resid[:i])) - float(np.median(resid[i:])))
        for i in range(STEP_MIN_SIDE, len(resid) - STEP_MIN_SIDE + 1)
    )


def _centroid_y(ink: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> float:
    """Вертикальный центроид краски в окне (пиксели рендера); середина окна, если краски нет."""
    h, w = ink.shape
    x0, x1, y0, y1 = max(0, x0), min(w, x1), max(0, y0), min(h, y1)
    band = ink[y0:y1, x0:x1].astype(np.float64)
    weights = band.sum(axis=1)
    if weights.sum() <= 0:
        return (y0 + y1) / 2.0
    return float((weights * np.arange(y0, y1)).sum() / weights.sum())


__all__ = ["LineTilt", "tilt_summary", "projection_tilt", "is_heading", "glyph_line_metrics"]
