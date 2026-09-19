"""Длинные прямые штрихи (LSD): наклон к оси, поворот и параллельность попарно «было | стало».

Отрезки ищет LSD (``cv2.createLineSegmentDetector``) на рендере 300 dpi — он находит
штрих любой ориентации (наклонную дробную черту 1967/01 с.85, полки шкафа под −5° на
1966/01 с.78) и не требует морфологии, которая рвёт наклонённую линейку. Отрезки короче
``min_len_mm`` отбрасываются: у текста длинных прямых кромок нет.

Три вещи, которые FineReader портит и которые видны только на штрихах:

* **Наклон к оси.** Линейка таблицы или рамка, бывшая вертикальной, наклонилась (с.38,
  с.95): у околоосевых отрезков сравнивается |угол от оси|, взвешенно по длине.
* **Параллельность.** Полки шкафа в перспективе были параллельны, а после «распрямления»
  разошлись (с.78); рамки блок-схемы — то же (с.80). Отрезки B группируются по углу, и у
  группы сравнивается разброс углов до и после: законная правка (поворот, сдвиг) разброс
  не меняет, местная — увеличивает.
* **Поворот отрезка сверх общего.** ``|Δугла − поворот страницы|`` — сколько FineReader
  довернул именно этот штрих; для контекста, флага не ставит.

Сопоставление B → A: середина отрезка B переносится полем смещений, ищется отрезок A
близкого угла и длины, чья середина лежит у прямой B′.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from research.geometry_regression import px_to_mm
from research.geometry_regression.field import Field

# LSD на слегка размытом бинарном рендере: без размытия ступеньки бинаризации дробят отрезки.
BLUR_SIGMA_PX = 2.0
# Кляксы недосвета у корешка (1966/01 с.91): длинные, но конусные и щербатые; сплошная масса
# толще 2.5 мм — кайма. Толщина краски меряется поперёк отрезка в нескольких
# точках: медиана больше ``MAX_INK_MM`` или разброс больше ``MAX_RAGGED`` долей медианы — не штрих.
MAX_INK_MM = 2.5
# Вне рисунка толстый штрих — клякса (тень корешка 2 мм на 1976/07 с.92 и 1976/08 с.85);
# внутри рамки line art допустимы бруски чертежа до ``MAX_INK_MM`` (шкаф на 1966/01 с.78).
MAX_INK_MM_TEXT = 1.2
LINEART_PAD_MM = 2.0
# Толстый штрих ближе этого к левому или правому краю кадра — тень корешка, даже если детектор
# штриха включил её в «рисунок» (1976/08 с.85: клякса плюс логотип рубрики — одна рамка).
# Та же полоса — для одиночных вертикалей (кромка кадра на 1975/08 с.95 — в 26 мм от края).
EDGE_MM = 30.0
MAX_RAGGED = 0.6
# Конус: клякса недосвета сужается к концу (1966/01 с.91: 0.15 → 0.7 мм по длине), линейка и
# линия чертежа ровные; порог — изменение толщины вдоль отрезка в долях медианы.
MAX_TAPER = 0.7
INK_PROBE_MM = 5.0
INK_SAMPLES = 15
# Склейка коллинеарных кусков: наклонная черта на бинарном рендере — лестница, LSD режет её
# на 2-3 отрезка (1967/01 с.85). Кандидаты в склейку — от этой длины (мм); склеиваются при
# близких углах, малом перпендикулярном зазоре и разрыве вдоль не больше ``MERGE_GAP_MM``.
MERGE_MIN_MM = 4.0
MERGE_ANGLE_DEG = 4.0
MERGE_PERP_MM = 0.25
MERGE_GAP_MM = 1.3
# Двойники: линия толще ~2 px даёт у LSD два отрезка — по верхней и нижней кромке (линейка под
# сноской на 1968/10 с.20 считалась за две черты и вместе с одним тире набирала «≥ 3 черт формул»).
# Параллельные (угол в пределах TWIN_ANGLE_DEG) отрезки не дальше TWIN_PERP_MM друг от друга с
# перекрытием вдоль не меньше TWIN_OVERLAP от короткого — одна линия, остаётся длинный.
TWIN_ANGLE_DEG = 3.0
TWIN_PERP_MM = 2.5
TWIN_OVERLAP = 0.7
# Околоосевой отрезок — в пределах этого угла от горизонтали или вертикали. Для ВЫИГРЫША —
# строже: наклонные линии чертежа (полки шкафа под 5°, 1966/01 с.78), повёрнутые FineReader
# к оси, — порча, а не выправленная линейка.
AXIS_TOL_DEG = 10.0
RULER_AXIS_TOL_DEG = 3.0
# Средний наклон горизонтальных штрихов считается, если их суммарная длина не меньше этого:
# на нетронутой странице с двумя чёрточками по 5 мм (1967/01 с.17) он давал 0.6° из ничего.
WMEAN_MIN_TOTAL_MM = 20.0
# Сопоставление: допуск перпендикулярного расстояния до прямой B′ (мм), по углу и по длине.
MATCH_PERP_MM = 1.3
# Допуски сопоставления не ужимать: наклонённая на 6° дробная черта (1967/01 с.85) и линейка,
# укороченная FineReader на треть (1971/08 с.79), — та самая порча, которую надо спарить и
# измерить. Росчерки логотипов отсеиваются раньше, в ``_drop_lone``.
MATCH_ANGLE_DEG = 8.0
MATCH_LENGTH_FRAC = 0.4
# Одиночный вертикальный штрих без перпендикулярного соседа (конец другого штриха ближе
# PERP_MM) вне рамок line art — не линейка, если он в полосе EDGE_MM от края кадра (тень
# корешка, 1975/07 с.78) или короче LONE_MAX_MM (карандашная пометка на поле, 1967/07 с.75).
# У рамок таблиц и колонок перпендикуляры есть, черты формул горизонтальны — их правило не трогает.
PERP_MM = 3.0
LONE_MAX_MM = 15.0
# Тень корешка — одна линия; рамка колонки у края (1966/01 с.95: 6 вертикалей в 19-24 мм от
# края) — несколько. Вертикаль у края не считается тенью, если рядом (в пределах SIBLING_MM
# по x, с перекрытием по y) есть другая вертикаль от SIBLING_MIN_MM длиной.
SIBLING_MM = 8.0
SIBLING_MIN_MM = 20.0
# Рамка колонки замыкается с двух сторон: длинной (≥ FRAME_MIN_MM) вертикали у одного края
# отвечает вертикаль у противоположного; тень корешка бывает только с одной стороны.
FRAME_MIN_MM = 100.0
EDGE_MIN_PERPS = 2
# Росчерк логотипа: в полосе ±SCRIBBLE_BAND_MM по y лежат ≥ SCRIBBLE_MIN штрихов с разбросом
# углов больше SCRIBBLE_SPREAD_DEG — это волна, а не линейки (1975/04 с.95).
SCRIBBLE_BAND_MM = 3.0
SCRIBBLE_MIN = 3
SCRIBBLE_SPREAD_DEG = 2.0
SCRIBBLE_MAX_MM = 30.0
# Наклонный штрих (|наклон к оси| > LOGO_TILT_DEG) без перпендикуляра в верхних LOGO_TOP_FRAC
# страницы — росчерк логотипа рубрики, не штрих.
LOGO_TILT_DEG = 3.0
LOGO_TOP_FRAC = 0.15
# Короткий горизонтальный штрих вне рисунка — либо дробная черта формулы, либо тире в тексте
# (1968/10 с.20: три тире по 4-5 мм дали «наклон черт формул»). Дробную черту отличает краска
# числителя и знаменателя: в полосах от FRACTION_NEAR_MM до FRACTION_FAR_MM над и под чертой
# краска покрывает не меньше FRACTION_COVER долей её длины; у тире там пусто (соседние строки
# дальше: при кегле 8 и интерлиньяже ~3.5 мм до них ≥ 2 мм).
SHORT_MM = 8.0
FRACTION_NEAR_MM = 0.2
FRACTION_FAR_MM = 1.2
FRACTION_COVER = 0.15
FRACTION_SEARCH_MM = 0.5
# Штрихи с уходом конца от оси не меньше этого (мм) обводятся на оверлее все.
SHOW_DEV_MM = 0.5
# Группа параллельных: отрезки B от 8 мм в пределах этого угла от самого длинного; меньше
# стольких — не группа. Короткие штрихи (дробные черты 4-8 мм) в группы не идут: их углы шумят.
GROUP_TOL_DEG = 1.0
GROUP_MIN = 3
GROUP_MIN_MM = 8.0


@dataclass(frozen=True)
class Stroke:
    x0: float
    y0: float
    x1: float
    y1: float
    length: float  # px рендера
    angle_deg: float  # в (−90, 90]; 0 — горизонталь, ±90 — вертикаль
    in_lineart: bool = False  # лежит в рамке рисунка
    ruled: bool = True  # есть перпендикулярный сосед (линейка таблицы/рамки), а не росчерк логотипа

    @property
    def mid(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0

    @property
    def axis_tilt(self) -> float | None:
        """Отклонение от ближайшей оси, если отрезок околоосевой; иначе None."""
        a = abs(self.angle_deg)
        if a <= AXIS_TOL_DEG:
            return a
        if a >= 90.0 - AXIS_TOL_DEG:
            return 90.0 - a
        return None

    def tilt_from(self, orient: str) -> float:
        """Отклонение от заданной оси («h»/«v») без допуска — для двойника в A."""
        a = abs(self.angle_deg)
        return a if orient == "h" else 90.0 - a

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (
            int(min(self.x0, self.x1)),
            int(min(self.y0, self.y1)),
            int(max(self.x0, self.x1)) + 1,
            int(max(self.y0, self.y1)) + 1,
        )


def _wrap(angle: np.ndarray | float):
    """Разность направлений в (−90, 90]."""
    return (np.asarray(angle) + 90.0) % 180.0 - 90.0


def find_strokes(
    gray: np.ndarray,
    min_len_mm: float,
    dpi: float,
    lineart_boxes: list | None = None,
    boxes_dpi: float | None = None,
    drop_lone: bool = True,
) -> list[Stroke]:
    """Отрезки LSD длиной от ``min_len_mm`` на рендере ``gray``; толстые и щербатые кляксы отсеяны.

    Args:
        gray: серый рендер страницы.
        min_len_mm: короче — не штрих.
        dpi: разрешение рендера.
        lineart_boxes: рамки рисунков ``(x0, y0, x1, y1)`` в пикселях ``boxes_dpi``; внутри них
            допустимы толстые бруски чертежа, вне — толстый штрих считается кляксой.
        boxes_dpi: разрешение, в котором заданы рамки (по умолчанию ``dpi``).
        drop_lone: отсеивать одиночные вертикали у края и росчерки (``_drop_lone``). Для версии
            A выключается: пара ищется от штрихов B, и рамка, которую FineReader в A разрезал
            на куски у края (1966/01 с.95), должна остаться, чтобы кускам было с чем спариться.

    Returns:
        Штрихи, прошедшие фильтр толщины, щербатости и конуса.
    """
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    blurred = cv2.GaussianBlur(gray, (0, 0), BLUR_SIGMA_PX)
    found = detector.detect(blurred)[0]
    if found is None:
        return []
    segments = found.reshape(-1, 4).astype(np.float64)
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    segments = _merge_collinear(segments[lengths >= MERGE_MIN_MM * dpi / 25.4], dpi)
    segments = _drop_twins(segments, dpi)
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
    angles = _wrap(np.degrees(np.arctan2(segments[:, 3] - segments[:, 1], segments[:, 2] - segments[:, 0])))
    min_len = min_len_mm * dpi / 25.4
    ink = gray < 128
    k = dpi / (boxes_dpi or dpi)
    # Рамки рисунков с припуском: кромка фотографии лежит на самой границе рамки (1968/04
    # с.86), и без припуска её штрихи попадают то внутрь, то наружу.
    pad = LINEART_PAD_MM * dpi / 25.4
    boxes = [(x0 * k - pad, y0 * k - pad, x1 * k + pad, y1 * k + pad) for x0, y0, x1, y1 in (lineart_boxes or [])]
    edge_px = EDGE_MM * dpi / 25.4
    short_px = SHORT_MM * dpi / 25.4
    strokes: list[Stroke] = []
    for i in np.nonzero(lengths >= min_len)[0]:
        mx, my = (segments[i][0] + segments[i][2]) / 2.0, (segments[i][1] + segments[i][3]) / 2.0
        in_lineart = any(x0 <= mx < x1 and y0 <= my < y1 for x0, y0, x1, y1 in boxes)
        near_edge = mx < edge_px or mx > gray.shape[1] - edge_px
        thick_ok = MAX_INK_MM if in_lineart and not near_edge else MAX_INK_MM_TEXT
        short_h = lengths[i] < short_px and abs(angles[i]) <= AXIS_TOL_DEG and not in_lineart
        if short_h and not _is_fraction_bar(ink, segments[i], dpi):
            continue
        if _is_thin(ink, segments[i], dpi, thick_ok):
            strokes.append(Stroke(*segments[i], float(lengths[i]), float(angles[i]), in_lineart))
    strokes = _mark_ruled(strokes, dpi)
    strokes = _unrule_logo(strokes, gray.shape[0])
    return _drop_lone(strokes, gray.shape, boxes, dpi) if drop_lone else strokes


def _unrule_logo(strokes: list[Stroke], height: int) -> list[Stroke]:
    """Штрихи внутри рисунка в шапке страницы — росчерки логотипа рубрики, не линейки.

    Логотип («Письма читателей» 1973/05 с.91, «Экономическое образование кадров» 1973/06 с.65)
    — рамка line art в верхних ``LOGO_TOP_FRAC`` страницы; его росчерк, размеченный буквами
    с перпендикулярами, иначе проходит за линейку, и FineReader, довернув логотип, «наклоняет
    черты формул». Таблица в шапке так не теряется: её линейки в B стоят на оси и ловятся
    параллельностью и изгибом, а порчу схем — поле смещений.
    """
    top = LOGO_TOP_FRAC * height
    return [
        (
            Stroke(s.x0, s.y0, s.x1, s.y1, s.length, s.angle_deg, s.in_lineart, False)
            if s.in_lineart and s.mid[1] < top
            else s
        )
        for s in strokes
    ]


def _mark_ruled(strokes: list[Stroke], dpi: float) -> list[Stroke]:
    """Штрихи внутри рисунков: с перпендикулярным соседом — линейка таблицы, без него — росчерк."""
    perp_px = PERP_MM * dpi / 25.4
    return [
        Stroke(
            s.x0,
            s.y0,
            s.x1,
            s.y1,
            s.length,
            s.angle_deg,
            s.in_lineart,
            not s.in_lineart or _has_perpendicular(s, strokes, perp_px),
        )
        for s in strokes
    ]


def _count_perpendicular(stroke: Stroke, others: list[Stroke], perp_px: float) -> int:
    """Сколько РАЗНЫХ мест на штрихе, куда упираются концы штрихов другой ориентации.

    Контакты ближе ``3 * perp_px`` друг к другу вдоль штриха — одно место: угол рамки на
    бинарном рендере даёт два-три коротких обломка (1975/08 с.95), а не два перпендикуляра.
    """
    x0, y0, x1, y1 = stroke.x0, stroke.y0, stroke.x1, stroke.y1
    length = max(1e-6, stroke.length)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    contacts: list[float] = []
    for other in others:
        if other is stroke or abs(_wrap(other.angle_deg - stroke.angle_deg)) < 45.0:
            continue
        for px, py in ((other.x0, other.y0), (other.x1, other.y1)):
            t = (px - x0) * ux + (py - y0) * uy
            if -perp_px <= t <= length + perp_px and abs((px - x0) * uy - (py - y0) * ux) <= perp_px:
                contacts.append(t)
                break
    count = 0
    last = None
    for t in sorted(contacts):
        if last is None or t - last > 3.0 * perp_px:
            count += 1
        last = t
    return count


def _has_perpendicular(stroke: Stroke, others: list[Stroke], perp_px: float) -> bool:
    """Есть ли штрих другой ориентации, конец которого лежит у этого штриха."""
    return _count_perpendicular(stroke, others, perp_px) > 0


def _has_sibling(stroke: Stroke, others: list[Stroke], dpi: float) -> bool:
    """Есть ли рядом другая длинная вертикаль с перекрытием по y — признак рамки, а не тени корешка."""
    sib_px, min_px = SIBLING_MM * dpi / 25.4, SIBLING_MIN_MM * dpi / 25.4
    y0, y1 = min(stroke.y0, stroke.y1), max(stroke.y0, stroke.y1)
    for other in others:
        if other is stroke or abs(other.angle_deg) < 90.0 - AXIS_TOL_DEG or other.length < min_px:
            continue
        dx = abs(other.mid[0] - stroke.mid[0])
        if dx < 2.0 * dpi / 25.4 or dx > sib_px:
            continue
        oy0, oy1 = min(other.y0, other.y1), max(other.y0, other.y1)
        if min(y1, oy1) - max(y0, oy0) > 0:
            return True
    return False


def _has_opposite(stroke: Stroke, others: list[Stroke], width: int, dpi: float) -> bool:
    """Есть ли длинная вертикаль у противоположного края с перекрытием по y (рамка, а не тень)."""
    edge_px, min_px = EDGE_MM * dpi / 25.4, FRAME_MIN_MM * dpi / 25.4
    left = stroke.mid[0] < width / 2
    y0, y1 = min(stroke.y0, stroke.y1), max(stroke.y0, stroke.y1)
    for other in others:
        if other is stroke or abs(other.angle_deg) < 90.0 - AXIS_TOL_DEG or other.length < min_px:
            continue
        ox = other.mid[0]
        if (ox > width - edge_px) != left or (ox < edge_px) == left:
            continue
        oy0, oy1 = min(other.y0, other.y1), max(other.y0, other.y1)
        if min(y1, oy1) - max(y0, oy0) > 0.5 * (y1 - y0):
            return True
    return False


def _in_scribble(stroke: Stroke, others: list[Stroke], dpi: float, height: int) -> bool:
    """Околоосевой горизонтальный штрих в шапке страницы из группы соседей по y с разбросом углов — росчерк.

    Только в верхних ``LOGO_TOP_FRAC`` страницы и только короткие (< ``SCRIBBLE_MAX_MM``):
    буквы крупного заголовка тоже дают пучок горизонталей под разными углами (1968/04 с.86).
    """
    if abs(stroke.angle_deg) > AXIS_TOL_DEG or stroke.mid[1] > LOGO_TOP_FRAC * height:
        return False
    if stroke.length > SCRIBBLE_MAX_MM * dpi / 25.4:
        return False
    band = SCRIBBLE_BAND_MM * dpi / 25.4
    # Рядом по y есть наклонные отрезки (дуги той же волны, |угол| > LOGO_TILT_DEG) — росчерк.
    tilted = [
        other
        for other in others
        if other is not stroke
        and LOGO_TILT_DEG < abs(other.angle_deg) < 90.0 - AXIS_TOL_DEG
        and abs(other.mid[1] - stroke.mid[1]) <= band
    ]
    if tilted:
        return True
    angles = [
        other.angle_deg
        for other in others
        if abs(other.angle_deg) <= AXIS_TOL_DEG and abs(other.mid[1] - stroke.mid[1]) <= band
    ]
    return len(angles) >= SCRIBBLE_MIN and (max(angles) - min(angles)) > SCRIBBLE_SPREAD_DEG


def _drop_lone(strokes: list[Stroke], shape: tuple, boxes: list, dpi: float) -> list[Stroke]:
    """Отсев штрихов, которые не линейки: одиночные вертикали у края/на поле и росчерки логотипов."""
    height, width = shape[:2]
    perp_px, edge_px, lone_px = PERP_MM * dpi / 25.4, EDGE_MM * dpi / 25.4, LONE_MAX_MM * dpi / 25.4
    kept: list[Stroke] = []
    for stroke in strokes:
        mx, my = stroke.mid
        in_lineart = any(x0 <= mx < x1 and y0 <= my < y1 for x0, y0, x1, y1 in boxes)
        if in_lineart:
            kept.append(stroke)
            continue
        tilt = stroke.axis_tilt
        vertical = abs(stroke.angle_deg) >= 90.0 - AXIS_TOL_DEG
        perps = _count_perpendicular(stroke, strokes, perp_px)
        lone = perps == 0
        near_edge = mx < edge_px or mx > width - edge_px
        framed = stroke.length >= FRAME_MIN_MM * dpi / 25.4 and _has_opposite(stroke, strokes, width, dpi)
        # Вертикаль у края без пары напротив и без соседки — тень корешка или кромка кадра, даже
        # если в неё упирается одна линейка (1975/08 с.95: край страницы под шапкой). Линейку
        # таблицы у края держат её строки — перпендикуляров не меньше EDGE_MIN_PERPS.
        if vertical and near_edge and not _has_sibling(stroke, strokes, dpi) and not framed and perps < EDGE_MIN_PERPS:
            continue
        if vertical and lone and stroke.length < lone_px:
            continue
        if _in_scribble(stroke, strokes, dpi, height):
            continue
        if tilt is not None and tilt > LOGO_TILT_DEG and lone and my < LOGO_TOP_FRAC * height:
            continue
        if tilt is None and lone and my < LOGO_TOP_FRAC * height:
            continue
        kept.append(stroke)
    return kept


def _is_thin(ink: np.ndarray, segment: np.ndarray, dpi: float, max_ink_mm: float = MAX_INK_MM) -> bool:
    """Толщина краски поперёк отрезка в ``INK_SAMPLES`` точках: тонко и ровно — штрих, иначе клякса."""
    x0, y0, x1, y1 = segment
    length = max(1e-6, float(np.hypot(x1 - x0, y1 - y0)))
    nx, ny = -(y1 - y0) / length, (x1 - x0) / length
    probe = int(INK_PROBE_MM * dpi / 25.4)
    h, w = ink.shape
    widths = []
    for t in np.linspace(0.1, 0.9, INK_SAMPLES):
        px, py = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        offsets = np.arange(-probe, probe + 1)
        xs = np.clip(np.round(px + nx * offsets).astype(int), 0, w - 1)
        ys = np.clip(np.round(py + ny * offsets).astype(int), 0, h - 1)
        profile = ink[ys, xs]
        # Отрезок LSD лежит на кромке штриха: берём краску, примыкающую к нему (в пределах 2 px).
        centre = probe
        near = np.nonzero(profile[centre - 2 : centre + 3])[0]
        if near.size == 0:
            continue
        start = centre - 2 + int(near[0])
        lo = start
        while lo > 0 and profile[lo - 1]:
            lo -= 1
        hi = start
        while hi < profile.size - 1 and profile[hi + 1]:
            hi += 1
        widths.append(hi - lo + 1)
    if len(widths) < 3:
        return True  # краски рядом нет (кромка чужого объекта) — судить не по чему, не отсеиваем
    widths = np.array(widths, dtype=np.float64)
    median = float(np.median(widths))
    thick = median > max_ink_mm * dpi / 25.4
    # Разброс — межквартильный: пара точек, где линейку пересекает текст или стрелка, не в счёт.
    ragged = float((np.percentile(widths, 75) - np.percentile(widths, 25)) / max(median, 1.0)) > MAX_RAGGED
    # Конус — по медиане попарных наклонов (одна стрелка на конце линии его не создаёт).
    i, j = np.triu_indices(len(widths), 1)
    slopes = (widths[j] - widths[i]) / (j - i)
    taper = abs(float(np.median(slopes))) * (len(widths) - 1) / max(median, 1.0) > MAX_TAPER
    return not (thick or ragged or taper)


def _fraction_side_cover(ink: np.ndarray, x0: int, x1: int, y_lo: int, y_hi: int) -> float:
    """Доля столбцов в [x0, x1), где в полосе строк [y_lo, y_hi) есть краска; вне кадра — 0."""
    h, w = ink.shape
    x0, x1 = max(0, x0), min(w, x1)
    y_lo, y_hi = max(0, y_lo), min(h, y_hi)
    if x1 <= x0 or y_hi <= y_lo:
        return 0.0
    return float(ink[y_lo:y_hi, x0:x1].any(axis=0).mean())


def _is_fraction_bar(ink: np.ndarray, segment: np.ndarray, dpi: float) -> bool:
    """Короткая горизонтальная черта с краской и над, и под собой — дробная черта, а не тире.

    Отрезок LSD лежит на КРОМКЕ черты, поэтому сначала по столбцам находится сама краска черты
    (её середина и толщина) в пределах ±FRACTION_SEARCH_MM от отрезка, и полосы над и под
    отсчитываются от её краёв. Черта короткая — наклон в полосах не важен, берётся средний y.
    """
    x0, y0, x1, y1 = segment
    left, right = int(min(x0, x1)), int(max(x0, x1)) + 1
    h, w = ink.shape
    left, right = max(0, left), min(w, right)
    y = (y0 + y1) / 2.0
    search = int(FRACTION_SEARCH_MM * dpi / 25.4)
    lo, hi = max(0, int(y) - search), min(h, int(y) + search + 1)
    if right <= left or hi <= lo:
        return False
    window = ink[lo:hi, left:right]
    # Середина и толщина краски черты — медианы по столбцам, где краска есть.
    rows = np.arange(lo, hi)[:, None]
    counts = window.sum(axis=0)
    filled = counts > 0
    if filled.mean() < FRACTION_COVER:
        return False
    centre = float(np.median((window * rows).sum(axis=0)[filled] / counts[filled]))
    thickness = float(np.median(counts[filled]))
    near, far = FRACTION_NEAR_MM * dpi / 25.4, FRACTION_FAR_MM * dpi / 25.4
    top, bottom = centre - thickness / 2.0, centre + thickness / 2.0
    above = _fraction_side_cover(ink, left, right, int(top - far), int(top - near) + 1)
    below = _fraction_side_cover(ink, left, right, int(bottom + near), int(bottom + far) + 1)
    return above >= FRACTION_COVER and below >= FRACTION_COVER


def _drop_twins(segments: np.ndarray, dpi: float) -> np.ndarray:
    """Пару отрезков по двум кромкам одной линии свести к одному по её середине.

    Обход от длинных к коротким: короткий двойник сдвигает длинного на половину зазора к себе
    (так отрезок ложится на ось краски, и в A и B он один и тот же при любой кромке), сам
    отбрасывается. Возвращает отрезки без двойников.
    """
    if len(segments) < 2:
        return segments
    order = np.argsort(-np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1]))
    perp_tol = TWIN_PERP_MM * dpi / 25.4
    kept: list[np.ndarray] = []
    shifted: set[int] = set()
    for row in segments[order]:
        x0, y0, x1, y1 = row
        length = max(1e-6, float(np.hypot(x1 - x0, y1 - y0)))
        angle = np.degrees(np.arctan2(y1 - y0, x1 - x0))
        twin = False
        for j, base in enumerate(kept):
            bx0, by0, bx1, by1 = base
            base_len = max(1e-6, float(np.hypot(bx1 - bx0, by1 - by0)))
            if abs(float(_wrap(angle - np.degrees(np.arctan2(by1 - by0, bx1 - bx0))))) > TWIN_ANGLE_DEG:
                continue
            ux, uy = (bx1 - bx0) / base_len, (by1 - by0) / base_len
            # Середина короткого — не дальше perp_tol от прямой длинного; перекрытие вдоль — по
            # проекциям концов короткого на длинный.
            mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            perp = (mx - bx0) * uy - (my - by0) * ux
            if abs(perp) > perp_tol:
                continue
            t0, t1 = sorted(((x0 - bx0) * ux + (y0 - by0) * uy, (x1 - bx0) * ux + (y1 - by0) * uy))
            overlap = min(t1, base_len) - max(t0, 0.0)
            if overlap >= TWIN_OVERLAP * length:
                twin = True
                # У линии две кромки: первый двойник сдвигает длинного на середину, дальше —
                # только отбрасываются (третий отрезок — уже соседняя линия или мусор).
                if j not in shifted:
                    kept[j] = base + np.array([uy, -ux, uy, -ux]) * (perp / 2.0)
                    shifted.add(j)
                break
        if not twin:
            kept.append(row)
    return np.array(kept) if kept else segments[:0]


def _merge_collinear(segments: np.ndarray, dpi: float) -> np.ndarray:
    """Склейка коллинеарных отрезков с малым разрывом; обход от длинных к коротким.

    Погнутая черта (1967/01 с.85: −6.9° слева, −1.6° справа) НЕ склеивается — углы
    кусков расходятся сильнее допуска, и это правильно: она и есть порча.
    """
    if len(segments) < 2:
        return segments
    perp_tol, gap_tol = MERGE_PERP_MM * dpi / 25.4, MERGE_GAP_MM * dpi / 25.4
    segs = sorted((row.copy() for row in segments), key=lambda r: -np.hypot(r[2] - r[0], r[3] - r[1]))
    i = 0
    while i < len(segs):
        a = segs[i]
        j = i + 1
        while j < len(segs):
            b = segs[j]
            da, db = a[2:] - a[:2], b[2:] - b[:2]
            la = np.hypot(*da)
            ua = da / la
            na = np.array([-ua[1], ua[0]])
            angle = abs(_wrap(np.degrees(np.arctan2(db[1], db[0]) - np.arctan2(da[1], da[0]))))
            perp = max(abs((b[:2] - a[:2]) @ na), abs((b[2:] - a[:2]) @ na))
            if angle < MERGE_ANGLE_DEG and perp <= perp_tol:
                t = np.array([0.0, la, (b[:2] - a[:2]) @ ua, (b[2:] - a[:2]) @ ua])
                gap = max(t[2:].min() - la, -t[2:].max())
                if gap <= gap_tol:
                    a = np.r_[a[:2] + ua * t.min(), a[:2] + ua * t.max()]
                    segs[i] = a
                    del segs[j]
                    continue
            j += 1
        i += 1
    return np.array(segs).reshape(-1, 4)


def match_strokes(
    before: list[Stroke], after: list[Stroke], field: Field | None, dpi: float, field_dpi: float
) -> list[tuple[Stroke, Stroke]]:
    """Пары «отрезок B — тот же отрезок в A». Поле смещений задано в пикселях ``field_dpi``."""
    if not before or not after:
        return []
    k = field_dpi / dpi
    mids_b = np.array([s.mid for s in before])
    moved = field.transform(mids_b * k) / k if field is not None else mids_b
    mids_a = np.array([s.mid for s in after])
    len_a = np.array([s.length for s in after])
    ang_a = np.array([s.angle_deg for s in after])
    perp_tol = MATCH_PERP_MM * dpi / 25.4
    pairs: list[tuple[Stroke, Stroke]] = []
    taken: set[int] = set()
    for stroke, mid in zip(before, moved):
        theta = np.radians(stroke.angle_deg)
        normal = np.array([-np.sin(theta), np.cos(theta)])
        along = np.array([np.cos(theta), np.sin(theta)])
        delta = mids_a - mid
        ok = (
            (np.abs(_wrap(ang_a - stroke.angle_deg)) < MATCH_ANGLE_DEG)
            & (np.abs(len_a - stroke.length) < MATCH_LENGTH_FRAC * stroke.length)
            & (np.abs(delta @ along) < 0.5 * stroke.length)
            & (np.abs(delta @ normal) <= perp_tol)
        )
        candidates = [j for j in np.nonzero(ok)[0] if j not in taken]
        if not candidates:
            continue
        j = min(candidates, key=lambda j: abs(float(delta[j] @ normal)))
        taken.add(j)
        pairs.append((stroke, after[j]))
    return pairs


def _weighted_tilt(items: list[tuple[float, float]]) -> float:
    """Средний |наклон к оси|, взвешенный по длине; ``items`` — (наклон, длина)."""
    if not items:
        return 0.0
    tilt = np.array([t for t, _ in items])
    weight = np.array([w for _, w in items])
    return float((tilt * weight).sum() / weight.sum())


def stroke_metrics(
    before: list[Stroke], after: list[Stroke], pairs: list[tuple[Stroke, Stroke]], rot_deg: float, dpi: float
) -> tuple[dict[str, float], dict]:
    """Метрики по парам отрезков и рамки-виновники для оверлея (в пикселях ``dpi``)."""
    metrics: dict[str, float] = {
        "strokes_b": float(len(before)),
        "strokes_a": float(len(after)),
        "strokes_matched": float(len(pairs)),
        "strokes_lost_frac": 0.0 if not before else 1.0 - len(pairs) / len(before),
    }
    culprits: dict = {}

    # Самый длинный штрих B, которому в A не нашлось прямого двойника: он погнулся или порвался.
    matched_b = {id(b) for b, _ in pairs}
    lost = [b for b in before if id(b) not in matched_b]
    if lost:
        worst = max(lost, key=lambda s: s.length)
        metrics["stroke_lost_max_mm"] = px_to_mm(worst.length, dpi)
        culprits["stroke_lost_max_mm"] = {"b": worst.box, "a": worst.box}
    else:
        metrics["stroke_lost_max_mm"] = 0.0

    # Наклон к оси. Что видно глазу — не угол, а на сколько миллиметров конец штриха ушёл
    # от оси: короткий штрих под 2° (с.2 в 1967/02, 9 мм → 0.3 мм) незаметен, линейка
    # таблицы в 50 мм под 1.2° (с.38 в 1967/01, 1 мм) бросается в глаза. Поэтому максимум
    # берётся по отклонению ``длина × sin(наклон)``, а средний наклон по длине — контекст.
    # Наклон к оси — только по штрихам вне рисунков: у логотипа рубрики (1975/04 с.95,
    # рамка line art) волнистый росчерк не линейка, а порчу настоящих схем ловят поле
    # смещений (доля тайлов без пары) и параллельность.
    for orient, low, high in (("h", 0.0, AXIS_TOL_DEG), ("v", 90.0 - AXIS_TOL_DEG, 90.0)):
        # Линейки (с перпендикулярами: таблицы 1967/01 с.38, рамки) — да; росчерки внутри
        # рамки рисунка (логотип рубрики 1975/04 с.95) — нет: порчу схем ловят поле и параллельность.
        # Линейка — то, что и в B стояло почти на оси (после кадрирования перекос страницы < 1°):
        # росчерк логотипа под 5–6° (1973/05 с.91, 1973/06 с.65) и полки шкафа под −5° (1966/01
        # с.78) — не линейки; их порчу ловят параллельность и изгиб.
        own = [
            (b, a)
            for b, a in pairs
            if low <= abs(b.angle_deg) <= high
            and b.axis_tilt is not None
            and b.ruled
            and b.axis_tilt <= RULER_AXIS_TOL_DEG
        ]
        key = f"{orient}stroke"
        metrics[f"{key}_pairs"] = float(len(own))
        total_mm = px_to_mm(sum(b.length for b, _ in own), dpi)
        metrics[f"{key}_tilt_wmean_b"] = _weighted_tilt([(b.axis_tilt, b.length) for b, _ in own])
        metrics[f"{key}_tilt_wmean_a"] = _weighted_tilt([(a.tilt_from(orient), a.length) for _, a in own])
        metrics[f"{key}_tilt_wmean_delta"] = (
            metrics[f"{key}_tilt_wmean_a"] - metrics[f"{key}_tilt_wmean_b"] if total_mm >= WMEAN_MIN_TOTAL_MM else 0.0
        )
        if own:
            deltas = [
                px_to_mm(b.length, dpi) * (np.sin(np.radians(a.tilt_from(orient))) - np.sin(np.radians(b.axis_tilt)))
                for b, a in own
            ]
            worst = int(np.argmax(deltas))
            metrics[f"{key}_dev_max_delta_mm"] = float(deltas[worst])
            # На оверлей — все штрихи, ушедшие от оси заметно, не только худший: на странице
            # с формулами наклоняются все дробные черты (1967/06 с.39).
            shown = [i for i, d in enumerate(deltas) if d >= SHOW_DEV_MM] or [worst]
            culprits[f"{key}_dev_max_delta_mm"] = {
                "b": own[worst][0].box,
                "a": own[worst][1].box,
                "segments_b": [[own[i][0].x0, own[i][0].y0, own[i][0].x1, own[i][0].y1] for i in shown],
                "segments_a": [[own[i][1].x0, own[i][1].y0, own[i][1].x1, own[i][1].y1] for i in shown],
            }
            # Выигрыш: линейка, которую коррекция поставила на ось (в мм ухода конца).
            metrics[f"{key}_gain_mm"] = float(max(0.0, -min(deltas)))
        else:
            metrics[f"{key}_dev_max_delta_mm"] = 0.0
            metrics[f"{key}_gain_mm"] = 0.0

    # Поворот сверх общего.
    if pairs:
        extra = np.abs(_wrap(np.array([a.angle_deg - b.angle_deg for b, a in pairs]) - rot_deg))
        metrics["stroke_rot_p90"] = float(np.percentile(extra, 90))
        metrics["stroke_rot_max"] = float(extra.max())
    else:
        metrics["stroke_rot_p90"] = metrics["stroke_rot_max"] = 0.0

    # Параллельность: группы отрезков B по углу вокруг самого длинного, разброс до и после.
    metrics["parallel_groups"] = 0.0
    metrics["parallel_spread_delta_max"] = 0.0
    group_min_px = GROUP_MIN_MM * dpi / 25.4
    long_pairs = [(b, a) for b, a in pairs if b.length >= group_min_px]
    if long_pairs:
        pairs = long_pairs
        ang_b = np.array([b.angle_deg for b, _ in pairs])
        ang_a = np.array([a.angle_deg for _, a in pairs])
        order = np.argsort([-b.length for b, _ in pairs])
        used = np.zeros(len(pairs), dtype=bool)
        best = None
        for k in order:
            if used[k]:
                continue
            group = np.nonzero((np.abs(_wrap(ang_b - ang_b[k])) < GROUP_TOL_DEG) & ~used)[0]
            if len(group) < GROUP_MIN:
                continue
            used[group] = True
            spread_b = float(np.ptp(_wrap(ang_b[group] - ang_b[k])))
            spread_a = float(np.ptp(_wrap(ang_a[group] - ang_a[k])))
            metrics["parallel_groups"] += 1.0
            delta = spread_a - spread_b
            # Худшая группа по разности разбросов — по всем группам, не по первой попавшейся.
            if best is None or delta > best[0]:
                best = (delta, group)
        if best is not None:
            delta, group = best
            metrics["parallel_spread_delta_max"] = float(delta)
            xs = [v for i in group for v in (pairs[i][1].x0, pairs[i][1].x1)]
            ys = [v for i in group for v in (pairs[i][1].y0, pairs[i][1].y1)]
            xb = [v for i in group for v in (pairs[i][0].x0, pairs[i][0].x1)]
            yb = [v for i in group for v in (pairs[i][0].y0, pairs[i][0].y1)]
            # Рамка — вся группа, но на оверлее рисуются сами отрезки («segments»): рамка
            # группы накрывает и невредимые линии рядом (1966/06 с.58 — формулы целиком).
            culprits["parallel_spread_delta_max"] = {
                "b": (int(min(xb)), int(min(yb)), int(max(xb)) + 1, int(max(yb)) + 1),
                "a": (int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1),
                "segments_b": [[pairs[i][0].x0, pairs[i][0].y0, pairs[i][0].x1, pairs[i][0].y1] for i in group],
                "segments_a": [[pairs[i][1].x0, pairs[i][1].y0, pairs[i][1].x1, pairs[i][1].y1] for i in group],
            }
    return metrics, culprits
