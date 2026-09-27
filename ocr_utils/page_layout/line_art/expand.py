"""Достройка рамки наружу до краёв связных пятен краски, которые она режет (рамки surya и DeepSeek срезают края объектов)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr_utils.page_layout.geometry import Box

# Насколько сторона может уйти наружу, мм (пак-1, 600 dpi — пересчитывается от dpi).
# Формула: рамка surya ``Equation`` сдвинута вверх-влево ~0.5 мм, низ срезан у 30 % блоков
# (reports/surya_equations.md) — 3 мм хватает на индексы и дробную черту, но не дотягивает
# до строки соседнего абзаца (шаг строк пака ~3.8 мм).
FORMULA_GROW_MM = 3.0
# Рисунок, схема, бланк: край рамки surya или DeepSeek режет подпись или конец линии на 1–5 мм
# (1968/12 с.138); дальше 6 мм уже соседний текст.
FIGURE_GROW_MM = 6.0

# Сколько кругов достройки: сдвиг одной стороны открывает новые пятна на соседних.
MAX_ROUNDS = 4

SIDES = ("сверху", "снизу", "слева", "справа")


@dataclass(frozen=True)
class Grown:
    """Результат достройки.

    Attributes:
        box: Достроенная рамка.
        failed: Стороны, которые всё ещё режут пятно: оно уходит дальше потолка или за преграду.
    """

    box: Box
    failed: tuple[str, ...]


def grow_to_components(box: Box, ink: np.ndarray, dpi: int, max_grow_mm: float, barriers: list[Box] = ()) -> Grown:
    """Отодвинуть стороны рамки наружу так, чтобы они не резали ни одного связного пятна краски.

    Пятно, которое рамка задевает, но не содержит целиком, тянет пересечённую сторону до своего
    края. Если край пятна дальше ``max_grow_mm`` от исходной стороны (линейка колонки, строка
    соседнего абзаца) или за преградой, сторона пятном не двигается и попадает в ``failed``.
    Рамка только растёт: внутрь стороны не уходят.

    Args:
        box: Исходная рамка в пикселях ``ink``.
        ink: Маска краски всей полосы (ненулевое — краска).
        dpi: Разрешение ``ink``.
        max_grow_mm: Потолок сдвига каждой стороны наружу от исходного положения, мм.
        barriers: Рамки, за которые расти нельзя (растр, таблицы — исключения детектора).

    Returns:
        :class:`Grown` — достроенная рамка и стороны, которые остались резать пятна.
    """
    height, width = ink.shape[:2]
    start = box.clipped(width, height)
    limit = int(round(max_grow_mm / 25.4 * dpi))
    # Окно с запасом на потолок: пятна считаются только в нём, дальше рамка не уйдёт.
    window = Box(start.x0 - limit - 1, start.y0 - limit - 1, start.x1 + limit + 1, start.y1 + limit + 1).clipped(
        width, height
    )
    local = (ink[window.slice] > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(local, 8)
    if count <= 1 or start.area == 0:
        return Grown(start, ())
    # Пятна в координатах полосы: x0, y0, x1, y1.
    lefts = stats[1:, cv2.CC_STAT_LEFT] + window.x0
    tops = stats[1:, cv2.CC_STAT_TOP] + window.y0
    spots = np.stack([lefts, tops, lefts + stats[1:, cv2.CC_STAT_WIDTH], tops + stats[1:, cv2.CC_STAT_HEIGHT]], axis=1)
    # Пятно, упёршееся в край окна, уходит дальше потолка — его край неизвестен, тянуть им нельзя.
    touches_window = (
        ((spots[:, 0] <= window.x0) & (window.x0 > 0))
        | ((spots[:, 1] <= window.y0) & (window.y0 > 0))
        | ((spots[:, 2] >= window.x1) & (window.x1 < width))
        | ((spots[:, 3] >= window.y1) & (window.y1 < height))
    )
    bounds = {
        "сверху": start.y0 - limit,
        "снизу": start.y1 + limit,
        "слева": start.x0 - limit,
        "справа": start.x1 + limit,
    }
    x0, y0, x1, y1 = start.as_tuple()
    failed: set[str] = set()
    for _ in range(MAX_ROUNDS):
        moved = False
        # Пятна, задетые текущей рамкой, но не лежащие в ней целиком.
        hit = (spots[:, 0] < x1) & (spots[:, 2] > x0) & (spots[:, 1] < y1) & (spots[:, 3] > y0)
        outside = (spots[:, 0] < x0) | (spots[:, 1] < y0) | (spots[:, 2] > x1) | (spots[:, 3] > y1)
        failed = set()
        for index in np.flatnonzero(hit & outside):
            sx0, sy0, sx1, sy1 = (int(v) for v in spots[index])
            wanted = {"сверху": sy0, "снизу": sy1, "слева": sx0, "справа": sx1}
            crossed = {"сверху": sy0 < y0, "снизу": sy1 > y1, "слева": sx0 < x0, "справа": sx1 > x1}
            for side in SIDES:
                if not crossed[side]:
                    continue
                target = wanted[side]
                beyond = target < bounds[side] if side in ("сверху", "слева") else target > bounds[side]
                if touches_window[index] or beyond or _blocked(side, target, (x0, y0, x1, y1), barriers):
                    failed.add(side)
                    continue
                if side == "сверху" and target < y0:
                    y0, moved = target, True
                elif side == "снизу" and target > y1:
                    y1, moved = target, True
                elif side == "слева" and target < x0:
                    x0, moved = target, True
                elif side == "справа" and target > x1:
                    x1, moved = target, True
        if not moved:
            break
    return Grown(Box(x0, y0, x1, y1), tuple(side for side in SIDES if side in failed))


def _blocked(side: str, target: int, box: tuple[int, int, int, int], barriers: list[Box]) -> bool:
    """Упрётся ли сторона ``side``, сдвинутая до ``target``, в преграду (пересечёт её край).

    Args:
        side: Сторона рамки.
        target: Новое положение стороны, px.
        box: Текущая рамка ``(x0, y0, x1, y1)``.
        barriers: Преграды.

    Returns:
        ``True``, если между текущим и новым положением стороны лежит преграда, перекрывающая
        рамку по другой оси.
    """
    x0, y0, x1, y1 = box
    for barrier in barriers:
        if side in ("сверху", "снизу"):
            if barrier.x1 <= x0 or barrier.x0 >= x1:
                continue
            if side == "сверху" and target < barrier.y1 <= y0:
                return True
            if side == "снизу" and y1 <= barrier.y0 < target:
                return True
        else:
            if barrier.y1 <= y0 or barrier.y0 >= y1:
                continue
            if side == "слева" and target < barrier.x1 <= x0:
                return True
            if side == "справа" and x1 <= barrier.x0 < target:
                return True
    return False


__all__ = ["FIGURE_GROW_MM", "FORMULA_GROW_MM", "Grown", "grow_to_components"]
