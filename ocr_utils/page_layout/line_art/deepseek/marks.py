"""Пометка на полях — не рисунок: карандашное подчёркивание, скобка у абзаца, дуга через колонку по остатку вырезки с залитыми словами.

ЗАЧЕМ. Классический детектор line art находит пометку кандидатом «штрих» по одним пикселям; DeepSeek на первом
проходе видит только текст, на втором (слова залиты, :func:`pass2.fill_words`) остаются одна-две тонкие линии, и
модель отдаёт ``image`` на всю вырезку, а проверка краски :func:`pass2.has_body` её пропускает — пометка
становилась рисунком (1967/03 с.93, скобка у абзаца). Ложный рисунок — запретная зона для текстовых блоков и лишние
срабатывания метрик line art детектора порчи геометрии.

ПРАВИЛО (стенд ``research/margin_marks``, ``reports/margin_marks.md``; пак-1, 16 пометок в двух вариантах PDF —
15 пойманы, 1 ложное — мусор у края скана; 0 смен у 1306 надписей). Кандидат — пометка, если все условия сразу:

* кандидат «штрих» по одним пикселям (``info.kind = штрих``, ``info.sources = [ink]``);
* исход решения — только рисунки или «неясно» (рисунок с таблицей или формулой не трогается);
* в остатке залитой вырезки (без длинных линеек и крапин, как у :func:`pass2.verdict_pass2`) 1–6 длинных (от
  ``LONG_COMPONENT_MM``) компонент, все тоньше ``MAX_THICKNESS_MM`` и без углов (рамка, коробка схемы);
* длинная сторона рамки кандидата — от ``MIN_LONG_SIDE_MM`` до ``MAX_SIDE_MM``, короткая — не больше ``MAX_SIDE_MM``;
* штрих ручной — волнистость худшей длинной компоненты не меньше ``MIN_WOBBLE_MM`` (печатные линии 0.00–0.05 мм,
  карандаш 0.10–0.81) — или остатка почти нет (штрих лёг на слова и залит вместе с ними);
* краски в рамке не больше ``MAX_INK_SHARE``.

Меры компоненты: толщина ``2·площадь/периметр``, длина ``периметр/2``, волнистость — p95 отклонения пикселей от прямой,
подогнанной по компоненте, минус полтолщины; угол — у компоненты есть и горизонтальный, и вертикальный прямой пробег
от ``STRAIGHT_MM``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.page_layout.line_art.deepseek.pass2 import despeckle, strip_rules
from ocr_utils.page_layout.line_art.deepseek.rules import CROP_DPI

MM_PER_PX = 25.4 / CROP_DPI
# Компонента меньше этой площади, мм², в меры не идёт (крапины и так сняты ``despeckle``).
MIN_COMPONENT_MM2 = 0.5
# Компонента «длинная» — от этой длины, мм: мелкие недозалитые обрывки букв не считаются.
LONG_COMPONENT_MM = 5.0
# Прямой пробег для признака угла, мм.
STRAIGHT_MM = 3.0

# Пороги правила (обоснование — в докстринге модуля и reports/margin_marks.md).
MAX_COMPONENTS = 6
MAX_THICKNESS_MM = 0.5  # карандаш пака 0.22–0.44 мм; рубричная заставка — от 0.65
MIN_LONG_SIDE_MM = 18.0  # мусор у края скана — до 15 мм; пометки — от 22
MAX_SIDE_MM = 50.0  # пометки — до 44 мм; графики, таблицы, заставки — от 60
MIN_WOBBLE_MM = 0.09  # печатные линии — до 0.05 мм
EMPTY_INK_SHARE = 0.005  # остаток почти пуст — штрих залит вместе со словами
MAX_INK_SHARE = 0.12  # пометки — до 9.4 %

# Сведения классического детектора о кандидате-штрихе по одним пикселям.
STROKE_KIND = "штрих"
INK_SOURCE = "ink"


@dataclass(frozen=True)
class Residue:
    """Что осталось в рамке кандидата после заливки слов (без длинных линеек и крапин).

    Attributes:
        long_components: Компонент длиной от ``LONG_COMPONENT_MM``.
        thickness_max_mm: Наибольшая толщина среди длинных компонент.
        wobble_max_mm: Наибольшая волнистость среди длинных компонент.
        corners: Длинных компонент с углом.
        ink_share: Доля краски остатка в рамке.
        box_w_mm: Ширина рамки кандидата.
        box_h_mm: Высота рамки кандидата.
    """

    long_components: int
    thickness_max_mm: float
    wobble_max_mm: float
    corners: int
    ink_share: float
    box_w_mm: float
    box_h_mm: float


def _component(piece: np.ndarray) -> tuple[float, float, float, bool]:
    """Меры одной компоненты (маска 0/1): толщина, длина, волнистость (пиксели) и есть ли угол."""
    area = float(piece.sum())
    contours, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = max(1.0, sum(cv2.arcLength(c, True) for c in contours))
    thickness = 2.0 * area / perimeter
    length = perimeter / 2.0
    # Волнистость: отклонение пикселей от прямой, подогнанной по компоненте (главная ось), p95 минус полтолщины.
    ys, xs = np.nonzero(piece)
    points = np.column_stack([xs, ys]).astype(np.float64)
    centred = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centred, full_matrices=False)
    wobble = max(0.0, float(np.percentile(np.abs(centred @ axes[1]), 95)) - thickness / 2.0)
    # Угол: у компоненты есть и горизонтальный, и вертикальный прямой пробег.
    run = max(3, int(round(STRAIGHT_MM / MM_PER_PX)))
    has_h = bool(cv2.morphologyEx(piece, cv2.MORPH_OPEN, np.ones((1, run), np.uint8)).any())
    has_v = bool(cv2.morphologyEx(piece, cv2.MORPH_OPEN, np.ones((run, 1), np.uint8)).any())
    return thickness, length, wobble, has_h and has_v


def residue(binary: np.ndarray, inner) -> Residue:
    """Меры остатка залитой вырезки в рамке кандидата.

    Args:
        binary: Вырезка с залитыми словами (краска 0, бумага 255), как у второго прохода.
        inner: Рамка кандидата в вырезке ``[x0, y0, x1, y1]``.

    Returns:
        :class:`Residue`.
    """
    stripped = despeckle(strip_rules(binary))
    x0, y0, x1, y1 = inner
    mask = np.zeros(binary.shape, np.uint8)
    mask[y0:y1, x0:x1] = stripped[y0:y1, x0:x1] == 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    mm2 = MM_PER_PX**2
    ink = 0.0
    long_items = []
    for index in range(1, count):
        area = float(stats[index, cv2.CC_STAT_AREA]) * mm2
        if area < MIN_COMPONENT_MM2:
            continue
        ink += area
        thickness, length, wobble, corner = _component((labels == index).astype(np.uint8))
        if length * MM_PER_PX >= LONG_COMPONENT_MM:
            long_items.append((thickness * MM_PER_PX, wobble * MM_PER_PX, corner))
    box_area = max(1.0, float((x1 - x0) * (y1 - y0))) * mm2
    return Residue(
        long_components=len(long_items),
        thickness_max_mm=max((t for t, _, _ in long_items), default=0.0),
        wobble_max_mm=max((w for _, w, _ in long_items), default=0.0),
        corners=sum(1 for _, _, c in long_items if c),
        ink_share=ink / box_area,
        box_w_mm=(x1 - x0) * MM_PER_PX,
        box_h_mm=(y1 - y0) * MM_PER_PX,
    )


def is_stroke_candidate(info: dict | None) -> bool:
    """Кандидат — «штрих», найденный классическим детектором по одним пикселям (без surya и таблиц)."""
    info = info or {}
    return info.get("kind") == STROKE_KIND and list(info.get("sources") or []) == [INK_SOURCE]


def looks_like_mark(found: Residue) -> bool:
    """Остаток — ручная пометка: немного тонких волнистых штрихов без углов, рамка размером с подчёркивание или скобку."""
    long_side = max(found.box_w_mm, found.box_h_mm)
    return (
        1 <= found.long_components <= MAX_COMPONENTS
        and found.thickness_max_mm <= MAX_THICKNESS_MM
        and found.corners == 0
        and MIN_LONG_SIDE_MM <= long_side <= MAX_SIDE_MM
        and (found.wobble_max_mm >= MIN_WOBBLE_MM or found.ink_share < EMPTY_INK_SHARE)
        and found.ink_share <= MAX_INK_SHARE
    )


__all__ = ["Residue", "is_stroke_candidate", "looks_like_mark", "residue"]
