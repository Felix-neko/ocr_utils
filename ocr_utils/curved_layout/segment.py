"""Сегментация строк по краске с ЗАДАННЫМИ межколонниками: сгустки → строки → центр-линии.

Повторяет конвейер ``curved_lines.detectors.line_fit.line_samples`` (те же примитивы: маска
глифов, смыкание RLSA, сборка кусков строки), но межколонники приходят снаружи — их ищет
:mod:`ocr_utils.curved_layout.columns`. Иначе строка двух колонок сшивается через межколонник:
``link_spans`` пускает разрыв до 2.5 высот, а межколонник журнала — 3–4 мм.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.curved_layout import RENDER_DPI, WORK_DPI
from ocr_utils.page_layout import mm_to_px
from ocr_utils.page_layout.orientation.detectors.ink_axis import (
    LINE_ASPECT,
    LINE_MAX_THICKNESS_PX,
    LINE_MIN_LENGTH_PX,
    _smear,
    glyph_mask,
)
from ocr_utils.scan_markup.curved_lines.detectors.line_fit import MIN_LENGTH_HEIGHTS, link_spans
from ocr_utils.scan_markup.curved_lines.fitting import centreline, smooth_median

# Окно медианного сглаживания центр-линии в высотах строки (как в ``line_fit``).
SMOOTH_HEIGHTS = 2.0
# Строка тоньше этого — не строка (в пикселях копии 150 dpi размеры ``ink_axis`` заданы для 150 dpi).
MIN_HEIGHT_PX = 4
# Доля краски в боксе, ниже которой «строка» — на самом деле линейка или рамка: у набора
# заполнение бокса краской 0.15–0.45, у сплошной черты — выше 0.6 при высоте в 1–2 px.
RULE_MAX_HEIGHT_MM = 1.2
RULE_MIN_FILL = 0.55
# Доля краски в полосе межколонника, выше которой строка и правда набрана через него (заголовок).
GUTTER_INK_SHARE = 0.01


def _split_at_gutters(
    span: list[int], stats: np.ndarray, separators: list[tuple[int, int]], ink300: np.ndarray, k: float
) -> list[list[int]]:
    """Разрезать строку по межколонникам, в полосе которых нет краски.

    Сборка кусков строки (``link_spans``) иногда сшивает соседние колонки: между последним
    сгустком одной колонки и первым сгустком другой попадает случайная краска (надстрочный знак,
    точка), и строка тянется через межколонник (1975/05 с.97). Настоящий заголовок через
    межколонник отличается тем, что краска в межколоннике ЕСТЬ, — такую строку не режем, её
    пометит :func:`columns.mark_cut_lines`.

    Args:
        span: Индексы сгустков одной строки.
        stats: Статистика связных компонент (``connectedComponentsWithStats``).
        separators: Межколонники и поля в пикселях рабочей копии.
        ink300: Краска рендера ``RENDER_DPI``.
        k: Пикселей рендера на пиксель рабочей копии.

    Returns:
        Один или несколько наборов индексов — куски строки по колонкам.
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


def segments_of(gray300: np.ndarray, separators: list[tuple[int, int]], dpi: float = WORK_DPI) -> list[Segment]:
    """Строки страницы с центр-линиями по краске.

    Args:
        gray300: Серый рендер страницы в ``RENDER_DPI``.
        separators: Межколонники и поля ``(x0, x1)`` в пикселях рабочей копии ``dpi``.
        dpi: Разрешение рабочей копии (боксы и высоты отдаются в нём).

    Returns:
        Строки сверху вниз; линейки, рамки и слишком короткие сгустки отброшены.
    """
    work = cv2.resize(
        gray300,
        (max(1, round(gray300.shape[1] * dpi / RENDER_DPI)), max(1, round(gray300.shape[0] * dpi / RENDER_DPI))),
        interpolation=cv2.INTER_AREA,
    )
    threshold, _ = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    mask = glyph_mask(work)
    smeared = _smear(mask, horizontal=True)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(smeared, 8)
    if count <= 1:
        return []
    ink300 = gray300 <= threshold
    k = RENDER_DPI / dpi
    rule_height = mm_to_px(RULE_MAX_HEIGHT_MM, dpi)
    out: list[Segment] = []
    for whole in link_spans(stats, separators):
        for span in _split_at_gutters(whole, stats, separators, ink300, k):
            members = stats[span]
            x0 = int(members[:, cv2.CC_STAT_LEFT].min())
            y0 = int(members[:, cv2.CC_STAT_TOP].min())
            x1 = int((members[:, cv2.CC_STAT_LEFT] + members[:, cv2.CC_STAT_WIDTH]).max())
            y1 = int((members[:, cv2.CC_STAT_TOP] + members[:, cv2.CC_STAT_HEIGHT]).max())
            width, height = x1 - x0, y1 - y0
            h_line = float(np.median(members[:, cv2.CC_STAT_HEIGHT]))
            if height > LINE_MAX_THICKNESS_PX or h_line < MIN_HEIGHT_PX:
                continue
            if width < LINE_ASPECT * h_line or width < max(LINE_MIN_LENGTH_PX, MIN_LENGTH_HEIGHTS * h_line):
                continue
            own = np.isin(labels[y0:y1, x0:x1], span).astype(np.uint8)
            crop = ink300[int(y0 * k) : int(y1 * k), int(x0 * k) : int(x1 * k)]
            own300 = cv2.resize(own, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
            ink = crop & (own300 > 0)
            if h_line <= rule_height and ink.mean() >= RULE_MIN_FILL:
                continue  # сплошная черта: линейка колонтитула, подчёркивание, разделитель сноски
            xs, ys, weights = centreline(ink)
            if xs.size == 0:
                continue
            ys = smooth_median(ys, int(SMOOTH_HEIGHTS * h_line * k))
            out.append(
                Segment(x0=x0, y0=y0, x1=x1, y1=y1, height=h_line, xs=xs / k + x0, ys=ys / k + y0, weights=weights)
            )
    return sorted(out, key=lambda item: (item.y0 + item.y1) / 2.0)


__all__ = ["Segment", "segments_of"]
