"""Сегментация строк по краске в ДВУХ масштабах: корпус и крупный набор (заголовки, подписи).

Готовый конвейер ``curved_lines.detectors.line_fit`` настроен на корпус: маска глифов берёт
компоненты 5–42 px (копия 150 dpi), смыкание RLSA — 8 px. Крупный заголовок туда не попадает
совсем (буква выше 42 px), а разрядка между словами шире зазора смыкания, поэтому строка
логотипа «ЭКОНОМИЧЕСКОЕ ОБРАЗОВАНИЕ КАДРОВ» (1973/06 с.65) распадалась и теряла ось. Здесь тот
же конвейер прогоняется двумя наборами размеров, и результаты сливаются: крупный сегмент
принимается, если корпусные его не покрыли.

Межколонники приходят снаружи (:mod:`ocr_utils.curved_layout.columns`): иначе строка двух
колонок сшивается через межколонник, ведь ``link_spans`` пускает разрыв до 2.5 высот.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.curved_layout import RENDER_DPI, WORK_DPI
from ocr_utils.page_layout import mm_to_px
from ocr_utils.scan_markup.curved_lines.detectors.line_fit import _crosses
from ocr_utils.scan_markup.curved_lines.fitting import centreline, smooth_median

# Окно медианного сглаживания центр-линии в высотах строки (как в ``line_fit``).
SMOOTH_HEIGHTS = 2.0
# Доля краски в боксе, выше которой «строка» — на самом деле сплошная черта (линейка сноски,
# подчёркивание колонтитула): у набора заполнение 0.15–0.45, у черты — выше 0.55 при высоте 1–2 px.
RULE_MAX_HEIGHT_MM = 1.2
RULE_MIN_FILL = 0.55
# Доля краски в полосе межколонника, выше которой строка и правда набрана через него (заголовок).
GUTTER_INK_SHARE = 0.01
# Крупный сегмент, накрытый корпусными на эту долю длины, — дубль и не берётся.
COVER_SHARE = 0.6
# Заполнение бокса краской, ниже которого длинный компонент — слипшиеся буквы, а не черта.
LETTER_MAX_FILL = 0.55
# Черта короче этого (мм) разделителем не считается: это дефис или тире. Разделитель сноски в
# журнале — черта примерно в 6–8 мм (1973/06 с.65), поэтому порог низкий.
RULE_MIN_LENGTH_MM = 5.0
# Плотность краски в боксе черты: на бинарном рендере сплошная линейка даёт 0.5–0.9.
RULE_MIN_DENSITY = 0.4


@dataclass(frozen=True)
class Scale:
    """Набор размеров одного масштаба сегментации (пиксели копии ``WORK_DPI``).

    Args:
        name: Имя масштаба для отладки.
        min_height: Компонента ниже — не буква (пыль, точки).
        max_height: Компонента выше — не буква (росчерк логотипа, рамка, рисунок).
        gap: Зазор смыкания RLSA: у крупного набора пробел шире.
        aspect: Строка должна быть во столько раз длиннее своей высоты.
        min_length_mm: И не короче этого.
        max_height_mm: Медианная высота сгустков выше этой — не строка текста.
    """

    name: str
    min_height: int
    max_height: int
    gap: int
    aspect: float
    min_length_mm: float
    max_height_mm: float
    max_thickness_mm: float  # сгусток толще — не строка (рисунок, рамка)
    link_gap_heights: float  # разрыв между кусками строки, в высотах
    link_dy_heights: float  # расхождение центров кусков, в высотах


# Корпус — числа готового конвейера (``ink_axis``); крупный набор — заголовки, логотипы рубрик,
# подписи авторов: буквы до 25 мм, разрядка между словами до 4 мм, длина от 12 мм.
SCALES = (
    Scale(
        "корпус",
        min_height=5,
        max_height=42,
        gap=8,
        aspect=6.0,
        min_length_mm=10.0,
        max_height_mm=8.0,
        max_thickness_mm=7.6,
        link_gap_heights=2.5,
        link_dy_heights=0.35,
    ),
    Scale(
        "крупный",
        min_height=16,
        max_height=150,
        gap=24,
        aspect=2.2,
        min_length_mm=12.0,
        max_height_mm=25.0,
        max_thickness_mm=30.0,
        link_gap_heights=2.0,
        link_dy_heights=0.7,
    ),
)
# Куски одной строки отличаются по высоте не больше чем во столько раз.
LINK_HEIGHT_RATIO = 1.8


def link_spans(stats: np.ndarray, separators: list[tuple[int, int]], scale: Scale, dpi: float) -> list[list[int]]:
    """Сцепить куски одной строки в цепочки (индексы сгустков) по размерам ``scale``.

    Повторяет ``line_fit.link_spans``, но пороги приходят из масштаба: готовый отбрасывает всё
    выше 45 px, а строка наклонного логотипа (буквы 22 px, разбег по высоте 30 px) выше — и
    заголовок терял ось целиком (1973/06 с.65).
    """
    max_thickness = mm_to_px(scale.max_thickness_mm, dpi)
    candidates = [
        index
        for index in range(1, stats.shape[0])
        if scale.min_height <= stats[index, cv2.CC_STAT_HEIGHT] <= max_thickness
        and stats[index, cv2.CC_STAT_WIDTH] >= 0.5 * stats[index, cv2.CC_STAT_HEIGHT]
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
                if ox < right - 2 or ox - right > scale.link_gap_heights * max(h, oh):
                    continue
                if abs(oy + oh / 2.0 - cy) > scale.link_dy_heights * max(h, oh):
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


@dataclass(frozen=True)
class Rule:
    """Сплошная горизонтальная черта: разделитель сноски, подчёркивание, линейка колонтитула."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


@dataclass(frozen=True)
class Segment:
    """Строка после сегментации: бокс на рабочей копии и центр-линия на копии ``RENDER_DPI``."""

    x0: int
    y0: int
    x1: int
    y1: int
    height: float
    xs: np.ndarray
    ys: np.ndarray
    weights: np.ndarray
    scale: str = "корпус"

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0


def component_mask(work: np.ndarray, scale: Scale) -> np.ndarray:
    """Маска компонент, похожих на буквы этого масштаба (пиксели копии ``work``).

    Длинный компонент — либо линейка, либо слипшиеся жирные буквы заголовка. Различает их
    ЗАПОЛНЕНИЕ бокса краской: у черты оно около единицы, у слова из букв — 0.3–0.5. Без этого
    «ОБРАЗОВАНИЕ КАДРОВ» (слиплось в один компонент 700 × 26 px) выпадало целиком.
    """
    _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    keep = np.zeros(count, dtype=bool)
    for index in range(1, count):
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        if not (scale.min_height <= height <= scale.max_height):
            continue
        fill = stats[index, cv2.CC_STAT_AREA] / max(1, width * height)
        if width <= 5 * max(height, 1) or fill < LETTER_MAX_FILL:
            keep[index] = True
    return keep[labels].astype(np.uint8)


def rules_of(work: np.ndarray, dpi: float) -> list[Rule]:
    """Сплошные горизонтальные черты страницы: разделитель сноски, подчёркивание, линейка.

    Ищутся отдельно от строк: в маску букв они не попадают (слишком длинные при малой высоте),
    а блоку нужны как граница — текст под разделителем сноски к блоку не относится.
    """
    _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    max_height = mm_to_px(RULE_MAX_HEIGHT_MM, dpi)
    min_width = mm_to_px(RULE_MIN_LENGTH_MM, dpi)
    out: list[Rule] = []
    for index in range(1, count):
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        fill = stats[index, cv2.CC_STAT_AREA] / max(1, width * height)
        # Черта узнаётся по аспекту (длиннее шести своих высот) и плотности: у разделителя сноски
        # на бинарном рендере заполнение падает до 0.5, поэтому порог ниже, чем у «строки-черты».
        if height <= max_height and width >= min_width and width >= 6 * height and fill >= RULE_MIN_DENSITY:
            x0 = int(stats[index, cv2.CC_STAT_LEFT])
            y0 = int(stats[index, cv2.CC_STAT_TOP])
            out.append(Rule(x0=x0, y0=y0, x1=x0 + width, y1=y0 + height))
    return out


def _segments_at_scale(
    gray300: np.ndarray,
    work: np.ndarray,
    ink300: np.ndarray,
    separators: list[tuple[int, int]],
    scale: Scale,
    dpi: float,
) -> list[Segment]:
    """Строки одного масштаба."""
    mask = component_mask(work, scale)
    smeared = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, scale.gap), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(smeared, 8)
    if count <= 1:
        return []
    k = RENDER_DPI / dpi
    rule_height = mm_to_px(RULE_MAX_HEIGHT_MM, dpi)
    min_length = mm_to_px(scale.min_length_mm, dpi)
    max_height = mm_to_px(scale.max_height_mm, dpi)
    out: list[Segment] = []
    for whole in link_spans(stats, separators, scale, dpi):
        for span in _split_at_gutters(whole, stats, separators, ink300, k):
            members = stats[span]
            x0 = int(members[:, cv2.CC_STAT_LEFT].min())
            y0 = int(members[:, cv2.CC_STAT_TOP].min())
            x1 = int((members[:, cv2.CC_STAT_LEFT] + members[:, cv2.CC_STAT_WIDTH]).max())
            y1 = int((members[:, cv2.CC_STAT_TOP] + members[:, cv2.CC_STAT_HEIGHT]).max())
            width = x1 - x0
            h_line = float(np.median(members[:, cv2.CC_STAT_HEIGHT]))
            if h_line > max_height or h_line < scale.min_height:
                continue
            if width < scale.aspect * h_line or width < min_length:
                continue
            own = np.isin(labels[y0:y1, x0:x1], span).astype(np.uint8)
            crop = ink300[int(y0 * k) : int(y1 * k), int(x0 * k) : int(x1 * k)]
            own300 = cv2.resize(own, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
            ink = crop & (own300 > 0)
            if h_line <= rule_height and ink.mean() >= RULE_MIN_FILL:
                continue  # сплошная черта: она ищется отдельно (``rules_of``)
            xs, ys, weights = centreline(ink)
            if xs.size == 0:
                continue
            ys = smooth_median(ys, int(SMOOTH_HEIGHTS * h_line * k))
            out.append(
                Segment(
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    height=h_line,
                    xs=xs / k + x0,
                    ys=ys / k + y0,
                    weights=weights,
                    scale=scale.name,
                )
            )
    return out


def _split_at_gutters(
    span: list[int], stats: np.ndarray, separators: list[tuple[int, int]], ink300: np.ndarray, k: float
) -> list[list[int]]:
    """Разрезать строку по межколонникам, в полосе которых нет краски.

    Сборка кусков строки (``link_spans``) иногда сшивает соседние колонки: между последним
    сгустком одной колонки и первым сгустком другой попадает случайная краска (надстрочный знак,
    точка), и строка тянется через межколонник (1975/05 с.97). Настоящий заголовок через
    межколонник отличается тем, что краска в межколоннике ЕСТЬ, — такую строку не режем, её
    пометит :func:`columns.mark_cut_lines`.
    """
    members = stats[span]
    y0 = int(members[:, cv2.CC_STAT_TOP].min())
    y1 = int((members[:, cv2.CC_STAT_TOP] + members[:, cv2.CC_STAT_HEIGHT]).max())
    centres = members[:, cv2.CC_STAT_LEFT] + members[:, cv2.CC_STAT_WIDTH] / 2.0
    cuts: list[float] = []
    for gx0, gx1 in separators:
        if gx0 <= 0 or not (centres.min() < gx0 and centres.max() > gx1):
            continue
        band = ink300[int(y0 * k) : int(y1 * k) + 1, int(gx0 * k) : int(gx1 * k) + 1]
        if band.size and band.mean() > GUTTER_INK_SHARE:
            continue  # краска в межколоннике: строка и правда набрана через него
        cuts.append((gx0 + gx1) / 2.0)
    if not cuts:
        return [span]
    groups: dict[int, list[int]] = {}
    for index, centre in zip(span, centres):
        key = int(np.searchsorted(sorted(cuts), centre))
        groups.setdefault(key, []).append(index)
    return list(groups.values())


def _covered(segment: Segment, others: list[Segment]) -> bool:
    """Накрыт ли сегмент другими: та же полоса по y и больше ``COVER_SHARE`` длины."""
    covered = 0.0
    for other in others:
        if min(segment.y1, other.y1) - max(segment.y0, other.y0) <= 0.4 * (segment.y1 - segment.y0):
            continue
        covered += max(0, min(segment.x1, other.x1) - max(segment.x0, other.x0))
    return covered >= COVER_SHARE * max(1, segment.x1 - segment.x0)


def segments_of(
    gray300: np.ndarray, separators: list[tuple[int, int]], dpi: float = WORK_DPI
) -> tuple[list[Segment], list[Rule]]:
    """Строки страницы с центр-линиями по краске, в обоих масштабах, и сплошные черты.

    Args:
        gray300: Серый рендер страницы в ``RENDER_DPI``.
        separators: Межколонники ``(x0, x1)`` в пикселях рабочей копии ``dpi``.
        dpi: Разрешение рабочей копии (боксы и высоты отдаются в нём).

    Returns:
        Строки сверху вниз (сначала корпус, затем крупные строки, не накрытые корпусными) и
        сплошные черты страницы.
    """
    work = cv2.resize(
        gray300,
        (max(1, round(gray300.shape[1] * dpi / RENDER_DPI)), max(1, round(gray300.shape[0] * dpi / RENDER_DPI))),
        interpolation=cv2.INTER_AREA,
    )
    threshold, _ = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink300 = gray300 <= threshold
    out: list[Segment] = []
    for scale in SCALES:
        found = _segments_at_scale(gray300, work, ink300, separators, scale, dpi)
        out.extend(segment for segment in found if not _covered(segment, out))
    return sorted(out, key=lambda item: item.cy), rules_of(work, dpi)


__all__ = ["SCALES", "Rule", "Scale", "Segment", "component_mask", "rules_of", "segments_of"]
