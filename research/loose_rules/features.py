"""Признаки линейки-сироты: чья краска под трассой (буквы или вытянутый штрих), покрытие, разрывы на межстрочье, тень, формула.

ГЛАВНЫЙ ПРИЗНАК — ``glyph_share``. Ложная сирота сшита из кусков БУКВ: стволы «Н», «И» двух строк
заголовка через межстрочье, низы засечек или перекладины жирных букв одной строки. Каждая такая буква
в полной бинарной картинке — отдельная связная компонента, КОМПАКТНАЯ: вдоль линейки она не длиннее
пары своих поперечников. Настоящая линейка — компонента вытянутая, даже если к ней прилипли хвосты
букв (подчёркивание под словом): вдоль она в разы длиннее, чем поперёк. Поэтому для каждой точки
трассы берётся компонента, которой принадлежит краска под ней, и считается доля точек, чья
компонента компактна.

Все расчёты — в рабочем разрешении детектора таблиц (``ruling.WORK_DPI`` = 150 dpi), на той же
серой копии и той же бинаризации (``ruling.binarize``), по которым линейку нашёл детектор.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from ocr_utils.page_layout.geometry import LooseRule
from ocr_utils.page_layout.pack_analysis.final import PageClass
from ocr_utils.page_layout.tables.rules import side_ink
from ocr_utils.page_layout.tables.ruling import binarize, mm_to_px

# Сетка порогов «компактности» компоненты: протяжённость вдоль линейки не больше стольких поперечников.
# Порог выбирается на разметке, поэтому доля считается сразу для нескольких.
ASPECT_GRID = (1.5, 2.0, 3.0, 4.0)
# Опорный порог для признаков, которым нужен один порог (самый длинный пробег «не букв»).
BASE_ASPECT = 2.0
# Компонента длиннее этого вдоль линейки — не буква, сколь угодно толстая (самые крупные буквы
# заголовков пака — около 12 мм, разведка 2026-09-29).
GLYPH_MAX_MM = 15.0
# Разрыв краски вдоль трассы короче этого — царапина скана или неровность штриха, не разрыв.
MIN_GAP_MM = 0.5
# Окно поперёк линейки для поиска межстрочья в разрыве: по 5 мм в обе стороны от трассы.
INTERLINE_WINDOW_MM = 5.0
# Полосы серости по бокам трассы для признака тени: от 1 до 3 мм от оси.
SHADOW_NEAR_MM = 1.0
SHADOW_FAR_MM = 3.0
# Окно поперёк трассы для профиля ядра штриха: по 1 мм в обе стороны.
CORE_WINDOW_MM = 1.0
# Бумага — этот перцентиль серой копии полосы (фон выровнен, почти вся полоса — бумага).
PAPER_PERCENTILE = 90


@dataclass(frozen=True)
class PageInk:
    """Всё, что признакам нужно о полосе: бинарка, метки компонент, их габариты, серая копия, яркость бумаги."""

    binary: np.ndarray
    labels: np.ndarray
    stats: np.ndarray
    gray: np.ndarray
    paper: float


@dataclass(frozen=True)
class RuleFeatures:
    """Признаки одной сироты; длины — в миллиметрах, доли — от 0 до 1."""

    length_mm: float
    thickness_mm: float
    coverage: float
    glyph_share_15: float
    glyph_share_20: float
    glyph_share_30: float
    glyph_share_40: float
    rule_run_mm: float
    gaps: int
    gap_max_mm: float
    interline_gaps: int
    side_ink_max: float
    side_ink_min: float
    shadow: float
    band_ink: float
    band_dark: float
    core_gray: float
    core_width_mm: float
    edge_mm: float
    in_formula: bool

    def to_row(self) -> dict:
        """Признаки строкой CSV: числа округлены до тысячных, флаг — 0/1."""
        row = {}
        for name, value in asdict(self).items():
            row[name] = (
                int(value)
                if isinstance(value, bool)
                else (round(float(value), 3) if isinstance(value, float) else value)
            )
        return row


def page_ink(gray: np.ndarray) -> PageInk:
    """Бинарка и компоненты полосы — один раз на все её линейки.

    Args:
        gray: Серая копия полосы в рабочем разрешении детектора таблиц.

    Returns:
        :class:`PageInk`: краска белым (``ruling.binarize``), метки 8-связных компонент, их габариты
        (``cv2.CC_STAT_*``), серая копия и яркость бумаги.
    """
    # Та же бинаризация, что у детектора: иначе компоненты разойдутся с его линейками.
    binary = binarize(gray)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return PageInk(binary, labels, stats, gray, float(np.percentile(gray, PAPER_PERCENTILE)))


def _trace(rule: LooseRule, along_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Точки трассы с шагом в пиксель вдоль оси линейки (в «горизонтальной» системе: вдоль — x).

    Args:
        rule: Линейка в пикселях рабочей копии.
        along_size: Размер картинки вдоль оси линейки (ширина для горизонтали, высота для вертикали).

    Returns:
        Пара массивов: координаты вдоль оси (целые, в кадре) и поперёк (дробные, интерполяция ломаной).
    """
    points = np.asarray(rule.points, dtype=np.float64)
    if not rule.horizontal:
        # Вертикаль сводится к горизонтали: вдоль — y, поперёк — x (как в ``rules.side_ink``).
        points = points[:, ::-1]
    points = points[np.argsort(points[:, 0])]
    along = np.arange(int(round(points[0, 0])), int(round(points[-1, 0])) + 1)
    along = along[(along >= 0) & (along < along_size)]
    return along, np.interp(along, points[:, 0], points[:, 1])


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Пробеги ``True`` в булевом ряду: пары ``(начало, конец)``, конец не включительно."""
    padded = np.concatenate([[False], flags, [False]]).astype(np.int8)
    steps = np.diff(padded)
    return list(zip(np.flatnonzero(steps == 1), np.flatnonzero(steps == -1)))


def _interline(binary_t: np.ndarray, centre: np.ndarray, gap: tuple[int, int], band: int, window: int) -> bool:
    """Лежит ли разрыв трассы на межстрочье: в окне поперёк есть сплошь белый ряд, а по краям разрыва — краска.

    Всё в «горизонтальной» системе (вдоль линейки — второй индекс ``binary_t``).

    Args:
        binary_t: Бинарка в системе линейки.
        centre: Поперечная координата трассы в каждой точке вдоль (округлённая).
        gap: Разрыв ``(начало, конец)`` в индексах вдоль.
        band: Полуширина полосы самой линейки поперёк — она из окна исключается.
        window: Полуширина окна поперёк.

    Returns:
        ``True`` — разрыв проходит через просвет между строками соседнего текста.
    """
    height = binary_t.shape[0]
    start, end = gap
    # Поперечная середина разрыва: на длине разрыва трасса почти прямая, берём среднее.
    mid = int(np.mean(centre))
    rows = [(max(0, mid - window), max(0, mid - band)), (min(height, mid + band + 1), min(height, mid + window + 1))]
    ink_per_column = np.zeros(end - start, dtype=np.int64)
    edge_ink = 0
    reach = max(1, end - start)
    for top, bottom in rows:
        if bottom <= top:
            continue
        ink_per_column += (binary_t[top:bottom, start:end] > 0).sum(axis=0)
        # Краска по обе стороны разрыва вдоль (на длину самого разрыва) — рядом стоит текст.
        edge_ink += int((binary_t[top:bottom, max(0, start - reach) : start] > 0).sum())
        edge_ink += int((binary_t[top:bottom, end : end + reach] > 0).sum())
    return bool(ink_per_column.size and ink_per_column.min() == 0 and edge_ink > 0)


def rule_features(
    ink: PageInk, rule: LooseRule, dpi: int, formula_boxes: list[tuple[int, int, int, int]]
) -> RuleFeatures:
    """Признаки одной линейки-сироты.

    Args:
        ink: Полоса (:func:`page_ink`) в рабочем разрешении.
        rule: Линейка в пикселях той же копии.
        dpi: Разрешение копии.
        formula_boxes: Рамки объектов «формула» в пикселях копии ``(x0, y0, x1, y1)``.

    Returns:
        :class:`RuleFeatures`.
    """
    binary, labels, gray = ink.binary, ink.labels, ink.gray
    if not rule.horizontal:
        binary, labels, gray = binary.T, labels.T, gray.T
    height, width = binary.shape
    along, across = _trace(rule, width)
    centre = np.clip(np.round(across).astype(int), 0, height - 1)
    band = max(1, int(np.ceil(rule.thickness_px / 2.0)) + 1)
    # Габариты компоненты вдоль и поперёк оси линейки.
    along_extent = ink.stats[:, cv2.CC_STAT_WIDTH if rule.horizontal else cv2.CC_STAT_HEIGHT]
    across_extent = ink.stats[:, cv2.CC_STAT_HEIGHT if rule.horizontal else cv2.CC_STAT_WIDTH]
    glyph_max = mm_to_px(GLYPH_MAX_MM, dpi)

    # Для каждой точки трассы — самая частая компонента в поперечном окне полосы линейки (0 — бумага).
    dominant = np.zeros(along.size, dtype=np.int64)
    for index, (x, y) in enumerate(zip(along, centre)):
        column = labels[max(0, y - band) : min(height, y + band + 1), x]
        column = column[column > 0]
        if column.size:
            dominant[index] = np.bincount(column).argmax()
    inked = dominant > 0
    ink_count = max(1, int(inked.sum()))

    # Доли точек, чья компонента компактна как буква, при нескольких порогах «вдоль / поперёк».
    ratio = along_extent[dominant] / np.maximum(1, across_extent[dominant])
    short = along_extent[dominant] <= glyph_max
    shares = [float((inked & short & (ratio <= limit)).sum()) / ink_count for limit in ASPECT_GRID]
    # Самый длинный пробег краски, которая НЕ буква (при опорном пороге): у настоящей линейки — почти вся длина.
    not_glyph = inked & ~(short & (ratio <= BASE_ASPECT))
    runs = _runs(not_glyph)
    rule_run = max((end - start for start, end in runs), default=0)

    # Разрывы краски вдоль трассы: сколько, самый длинный, сколько из них на межстрочье соседнего текста.
    gaps = [(s, e) for s, e in _runs(~inked) if e - s >= mm_to_px(MIN_GAP_MM, dpi) and s > 0 and e < along.size]
    window = mm_to_px(INTERLINE_WINDOW_MM, dpi)
    interline = sum(
        _interline(binary, centre[s:e], (int(along[s]), int(along[e - 1]) + 1), band, window) for s, e in gaps
    )

    # Тень: средняя серость полос 1–3 мм по сторонам трассы относительно бумаги, по более тёмной стороне.
    # Берутся только пиксели БУМАГИ (не краски по Otsu): буквы рядом с линейкой — не тень, а тень
    # кромки и корешка — серая муть светлее порога краски.
    near, far = mm_to_px(SHADOW_NEAR_MM, dpi), mm_to_px(SHADOW_FAR_MM, dpi)
    # В тех же полосах — доля краски и серость по ВСЕМ пикселям: тёмная кромка листа проходит порог
    # Otsu и в «бумагу» не попадает, зато даёт сплошь тёмную полосу с одной стороны (у текста рядом с
    # линейкой краски в полосе от силы треть).
    sides, band_inks, band_darks = [], [], []
    for sign in (-1, 1):
        total, count, inked_px, all_total, all_count = 0.0, 0, 0, 0.0, 0
        for x, y in zip(along, centre):
            lo, hi = sorted((y + sign * near, y + sign * far))
            lo, hi = max(0, lo), min(height, hi)
            if hi > lo:
                values = gray[lo:hi, x].astype(np.float64)
                is_ink = binary[lo:hi, x] > 0
                total += float(values[~is_ink].sum())
                count += int((~is_ink).sum())
                inked_px += int(is_ink.sum())
                all_total += float(values.sum())
                all_count += values.size
        sides.append(1.0 - total / count / max(1.0, ink.paper) if count else 0.0)
        band_inks.append(inked_px / all_count if all_count else 0.0)
        band_darks.append(1.0 - all_total / all_count / max(1.0, ink.paper) if all_count else 0.0)

    # Ядро штриха поперёк трассы: минимум серого в окне ±1 мм и ширина по полуспаду между этим минимумом
    # и бумагой. Настоящая линейка — узкое чёрное ядро (серый 5–80, 2–3 px на 150 dpi); кромка листа и
    # тень корешка — серая полка (130–165) шириной 4–10 px (профили, разведка 2026-09-29).
    reach = mm_to_px(CORE_WINDOW_MM, dpi)
    minima, widths = [], []
    for x, y in zip(along, centre):
        profile = gray[max(0, y - reach) : min(height, y + reach + 1), x].astype(np.float64)
        if profile.size == 0:
            continue
        low = float(profile.min())
        minima.append(low)
        widths.append(int((profile < (low + ink.paper) / 2.0).sum()))
    core_gray = float(np.median(minima)) if minima else 255.0
    core_width = float(np.median(widths)) if widths else 0.0

    # Расстояние до ближайшего края кадра (в исходной, не транспонированной системе — симметрично).
    box = rule.box
    frame_h, frame_w = ink.binary.shape
    edge = min(box.x0, box.y0, frame_w - box.x1, frame_h - box.y1)
    cx, cy = (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2
    in_formula = any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in formula_boxes)

    first, second = side_ink(ink.binary, rule, dpi)
    to_mm = 25.4 / dpi
    return RuleFeatures(
        length_mm=float(along.size) * to_mm,
        thickness_mm=rule.thickness_px * to_mm,
        coverage=float(inked.mean()) if along.size else 0.0,
        glyph_share_15=shares[0],
        glyph_share_20=shares[1],
        glyph_share_30=shares[2],
        glyph_share_40=shares[3],
        rule_run_mm=rule_run * to_mm,
        gaps=len(gaps),
        gap_max_mm=max((e - s for s, e in gaps), default=0) * to_mm,
        interline_gaps=int(interline),
        side_ink_max=max(first, second),
        side_ink_min=min(first, second),
        shadow=max(sides),
        band_ink=max(band_inks),
        band_dark=max(band_darks),
        core_gray=core_gray,
        core_width_mm=core_width * to_mm,
        edge_mm=max(0, edge) * to_mm,
        in_formula=in_formula,
    )


def formula_boxes(objects: list[dict], scale: float) -> list[tuple[int, int, int, int]]:
    """Рамки объектов «формула» итогового JSON в пикселях рабочей копии.

    Args:
        objects: ``objects`` итогового JSON разбора пака (рамки — в пикселях скана).
        scale: Рабочее разрешение к разрешению скана.

    Returns:
        Рамки ``(x0, y0, x1, y1)``.
    """
    return [
        tuple(int(round(v * scale)) for v in item["box"])  # type: ignore[misc]
        for item in objects
        if item["class"] == PageClass.FORMULA.value
    ]
