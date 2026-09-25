"""Отточия: ряды точек «. . . . .» в таблицах и оглавлениях — часть строки, а не пустота.

Точка отточия — сгусток около 0.5 × 0.5 мм (3 × 3 px рабочей копии 150 dpi). Маска глифов
(`page_layout.orientation.detectors.ink_axis.glyph_mask`, нижний порог 5 px) её выбрасывает, и
дальше отточие пропадает отовсюду: край ряда останавливается на последнем слове, блок не
накрывает поле точек, ось строки не доходит до конца строки.

Одиночную точку от отточия отличает ЦЕПОЧКА: четыре и больше точек одного размера, стоящих в ряд
с постоянным шагом. Пыль и точечный растр цепочек не образуют, поэтому краем ряда не становятся.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.curved_layout import WORK_DPI
from ocr_utils.page_layout import mm_to_px, px_to_mm

# Размер точки отточия: замер на 1971/10 с.93 — 0.5 × 0.5 мм. Ниже — пыль, выше — буква (её и так
# берёт маска глифов).
DOT_MIN_MM = 0.25
DOT_MAX_MM = 0.9
# Точка круглая: стороны бокса различаются не больше чем во столько раз, и бокс залит краской.
DOT_ASPECT = 2.0
DOT_MIN_FILL = 0.5
# Шаг между точками отточия и допуск на разброс их центров по вертикали. Замер по 1971/10 с.93:
# фактический шаг 3.03 мм (медиана 17.9 px при 150 dpi, p90 — 18.4 px), поэтому порог в 3.0 мм
# рвал половину звеньев — в цепочки входило 23 % точек вместо 61 %. Пять миллиметров ничего не
# добавляют, а склеить через пропущенную точку могут.
LEADER_STEP_MAX_MM = 4.0
LEADER_ROW_TOL_MM = 1.0
# Цепочка короче этого — не отточие: многоточие «…» в тексте состоит из трёх точек.
LEADER_MIN_DOTS = 4
LEADER_MIN_LENGTH_MM = 8.0
# Точки одной цепочки одного кегля; разнокалиберная россыпь цепочкой не считается.
LEADER_SIZE_RATIO = 1.6


@dataclass(frozen=True)
class Leader:
    """Отточие: цепочка точек одной строки (пиксели рабочей копии)."""

    x0: float
    x1: float
    y: float  # медиана ординат точек цепочки
    thickness: float  # медианная высота точки
    dots: int
    points: tuple[tuple[float, float], ...] = ()  # центры точек слева направо

    @property
    def length(self) -> float:
        return self.x1 - self.x0

    def covers(self, x: float) -> bool:
        """Лежит ли абсцисса внутри отточия."""
        return self.x0 <= x <= self.x1


def dot_boxes(work: np.ndarray, dpi: float = WORK_DPI) -> tuple[np.ndarray, np.ndarray]:
    """Мелкие округлые сгустки рабочей копии — кандидаты в точки отточия.

    Args:
        work: Серая рабочая копия страницы.
        dpi: Её разрешение.

    Returns:
        Пара ``(боксы, карта меток)``: боксы ``(n, 5)`` — ``label, cx, cy, w, h``; карта меток от
        ``connectedComponentsWithStats``.
    """
    _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    low, high = mm_to_px(DOT_MIN_MM, dpi), mm_to_px(DOT_MAX_MM, dpi)
    out = []
    for index in range(1, count):
        width = int(stats[index, cv2.CC_STAT_WIDTH])
        height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if not (low <= width <= high and low <= height <= high):
            continue
        if max(width, height) > DOT_ASPECT * max(1, min(width, height)):
            continue
        if stats[index, cv2.CC_STAT_AREA] < DOT_MIN_FILL * width * height:
            continue
        out.append((index, centroids[index][0], centroids[index][1], width, height))
    return (np.asarray(out, dtype=np.float64) if out else np.zeros((0, 5), dtype=np.float64)), labels


def leaders_of(work: np.ndarray, dpi: float = WORK_DPI) -> tuple[list[Leader], np.ndarray]:
    """Отточия страницы и маска их точек.

    Цепочка наращивается ВДОЛЬ СТРОКИ: от каждой точки ищется следующая правее, не дальше
    ``LEADER_STEP_MAX_MM`` по x и ``LEADER_ROW_TOL_MM`` по y ОТ НЕЁ ЖЕ, того же размера. Сравнение
    с предыдущей точкой, а не с медианой всей цепочки, важно на наклонной строке: за десяток точек
    отточие уходит по вертикали дальше собственного допуска, и группировка «по рядам» разваливала
    его на куски (на варианте geo 1971/10 с.93 так находилось 16 цепочек вместо 40).

    Цепочка принимается, если в ней хотя бы ``LEADER_MIN_DOTS`` точек и она длиннее
    ``LEADER_MIN_LENGTH_MM``.

    Args:
        work: Серая рабочая копия страницы.
        dpi: Её разрешение.

    Returns:
        Пара ``(отточия, маска)``: маска — ``uint8`` размером с ``work``, в ней только точки
        принятых цепочек.
    """
    boxes, labels = dot_boxes(work, dpi)
    mask = np.zeros(work.shape, dtype=np.uint8)
    if boxes.shape[0] < LEADER_MIN_DOTS:
        return [], mask
    tolerance = mm_to_px(LEADER_ROW_TOL_MM, dpi)
    step_max = mm_to_px(LEADER_STEP_MAX_MM, dpi)
    boxes = boxes[np.argsort(boxes[:, 1])]
    xs = boxes[:, 1]
    used = np.zeros(boxes.shape[0], dtype=bool)
    out: list[Leader] = []
    kept: list[int] = []
    for head in range(boxes.shape[0]):
        if used[head]:
            continue
        chain = [head]
        used[head] = True
        while True:
            last = boxes[chain[-1]]
            # Кандидаты — только точки в окне по x: список отсортирован, поэтому окно ищется
            # двоичным поиском, а не перебором всех точек страницы.
            lo = int(np.searchsorted(xs, last[1], side="right"))
            hi = int(np.searchsorted(xs, last[1] + step_max, side="right"))
            best = None
            for index in range(lo, hi):
                if used[index]:
                    continue
                item = boxes[index]
                if abs(item[2] - last[2]) > tolerance:
                    continue
                heights = sorted((last[4], item[4]))
                if heights[1] > LEADER_SIZE_RATIO * max(heights[0], 1e-6):
                    continue
                best = index
                break  # точки отсортированы по x: первая подходящая и есть ближайшая
            if best is None:
                break
            chain.append(best)
            used[best] = True
        out.extend(_accept([boxes[index] for index in chain], dpi, kept))
    if kept:
        mask[np.isin(labels, kept)] = 255
    return out, mask


def _accept(chain: list[np.ndarray], dpi: float, kept: list[int]) -> list[Leader]:
    """Принять цепочку точек, если она похожа на отточие; принятые метки копятся в ``kept``."""
    if len(chain) < LEADER_MIN_DOTS:
        return []
    xs = np.array([item[1] for item in chain], dtype=np.float64)
    widths = np.array([item[3] for item in chain], dtype=np.float64)
    x0, x1 = float(xs.min() - widths[0] / 2.0), float(xs.max() + widths[-1] / 2.0)
    if px_to_mm(x1 - x0, dpi) < LEADER_MIN_LENGTH_MM:
        return []
    kept.extend(int(item[0]) for item in chain)
    return [
        Leader(
            x0=x0,
            x1=x1,
            y=float(np.median([item[2] for item in chain])),
            thickness=float(np.median([item[4] for item in chain])),
            dots=len(chain),
            points=tuple((float(item[1]), float(item[2])) for item in chain),
        )
    ]


def leaders_mask(work: np.ndarray, dpi: float, leaders: list[Leader]) -> np.ndarray:
    """Маска точек ГОТОВЫХ отточий: их полосы рисуются толщиной в саму точку.

    Нужна, чтобы не искать отточия второй раз: в разборе они находятся один раз, а краске текста
    (:func:`page.text_ink`) нужна именно маска.

    Args:
        work: Серая рабочая копия (нужен только её размер).
        dpi: Разрешение рабочей копии.
        leaders: Найденные отточия.

    Returns:
        ``uint8`` размером с ``work``: точки цепочек.
    """
    mask = np.zeros(work.shape, dtype=np.uint8)
    for leader in leaders:
        thickness = max(1, int(round(leader.thickness)))
        for x, y in leader.points:
            cv2.circle(mask, (int(round(x)), int(round(y))), max(1, thickness // 2), 255, -1)
    return mask


def spans_at(leaders: list[Leader], y: float, tolerance: float) -> list[tuple[float, float]]:
    """Отрезки отточий, попавших в полосу ``|leader.y - y| <= tolerance``."""
    return [(leader.x0, leader.x1) for leader in leaders if abs(leader.y - y) <= tolerance]


def inside_spans(xs: np.ndarray, spans: list[tuple[float, float]]) -> np.ndarray:
    """Булев вектор: какие абсциссы лежат внутри отточий."""
    out = np.zeros(np.shape(xs), dtype=bool)
    for x0, x1 in spans:
        out |= (xs >= x0) & (xs <= x1)
    return out


def flatten_axis(
    xs: np.ndarray, ys: np.ndarray, weights: np.ndarray, spans: list[tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Выровнять ось строки на участке отточия по соседним словам.

    Центр масс краски в столбце отточия — это ордината точки, то есть базовая линия, на полвысоты
    ниже оси слов. Без поправки ось ныряет вниз на десятки миллиметров, и меры формы строки
    (``bend_mm``, ``sagitta_mm``) становятся фиктивно большими. Ось при этом ДОЛЖНА идти по
    отточию — поэтому точки не выбрасываются, а их ордината берётся интерполяцией по соседям.

    Args:
        xs, ys: Точки центр-линии (пиксели рендера, начало — левый край сегмента).
        weights: Веса точек (краска в столбце).
        spans: Отрезки отточий в тех же координатах.

    Returns:
        Те же массивы с исправленными ординатами и приглушёнными весами внутри отточий.
    """
    if not spans or xs.size == 0:
        return xs, ys, weights
    dotted = inside_spans(xs, spans)
    if not dotted.any() or dotted.all():
        return xs, ys, weights
    fixed = ys.copy()
    fixed[dotted] = np.interp(xs[dotted], xs[~dotted], ys[~dotted])
    light = weights.copy()
    light[dotted] = float(np.min(weights[~dotted])) if np.any(~dotted) else 0.0
    return xs, fixed, light


__all__ = ["Leader", "dot_boxes", "flatten_axis", "inside_spans", "leaders_mask", "leaders_of", "spans_at"]
