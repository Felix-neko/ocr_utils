"""Линейки-барьеры: ломаные, через которые детектор не сращивает строки и текстовые блоки.

Линейки приходят из детектора таблиц (``tables.detector.detect_all`` → ``PageLayout.loose_rules``)
как ломаные в пикселях рабочей копии и могут быть изогнутыми. Здесь — вся их геометрия:
пересекает ли отрезок ломаную, проходит ли линейка между двумя рядами, полосы-разделители для
сцепки строк, растровая маска для смыкания RLSA и перенос в систему выпрямленной области.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Кусок ломаной считается горизонтальным, если он уходит вдоль x не меньше, чем вдоль y.
# Горизонтальные куски делят ряды (разрез блока), вертикальные — строку (сцепка кусков).
# Линейка между рядами должна накрывать хотя бы такую долю общей ширины двух рядов — как у
# горизонтальной черты при разрезе блока (``blocks.RULE_OVERLAP_SHARE``).
BETWEEN_ROWS_SHARE = 0.15
# Ширина полосы-разделителя вокруг вертикального куска, px рабочей копии: линейка и так идёт по
# краске, широкая полоса съела бы крайнюю букву соседнего слова.
SEPARATOR_PAD_PX = 1


@dataclass(frozen=True)
class BarrierLines:
    """Набор линеек-барьеров: ломаные ``(n, 2)`` в пикселях рабочей копии и их габариты.

    ``lines`` — по одной ломаной на линейку, точки ``(x, y)`` по порядку вдоль неё.
    """

    lines: tuple[np.ndarray, ...] = ()

    @staticmethod
    def of(polylines) -> "BarrierLines":
        """Набор из ломаных любого вида (кортежи точек, списки, массивы).

        Args:
            polylines: Итерируемое ломаных; ломаная — последовательность точек ``(x, y)``.

        Returns:
            Набор; ломаные меньше чем из двух точек выбрасываются.
        """
        out = []
        for points in polylines or ():
            array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
            if len(array) >= 2:
                out.append(array)
        return BarrierLines(tuple(out))

    @property
    def empty(self) -> bool:
        """Нет ни одной линейки."""
        return not self.lines

    def _pieces(self):
        """Все куски всех ломаных: пары точек ``(p, q)``."""
        for line in self.lines:
            for index in range(len(line) - 1):
                yield line[index], line[index + 1]

    def crosses(self, p: tuple[float, float], q: tuple[float, float]) -> bool:
        """Пересекает ли отрезок ``p → q`` хоть одну линейку.

        Args:
            p: Начало отрезка ``(x, y)``, пиксели рабочей копии.
            q: Конец отрезка.

        Returns:
            ``True`` — между точками проходит линейка (касание концом тоже считается).
        """
        if not self.lines:
            return False
        a = np.asarray(p, dtype=np.float64)
        b = np.asarray(q, dtype=np.float64)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        for line in self.lines:
            # Быстрый отсев по габариту ломаной.
            if (
                line[:, 0].max() < lo[0]
                or line[:, 0].min() > hi[0]
                or line[:, 1].max() < lo[1]
                or line[:, 1].min() > hi[1]
            ):
                continue
            for index in range(len(line) - 1):
                if _segments_cross(a, b, line[index], line[index + 1]):
                    return True
        return False

    def between_rows(self, x0: float, x1: float, y_top: float, y_bottom: float) -> bool:
        """Проходит ли линейка между двумя рядами над их общей шириной.

        Для каждого x из ``[x0, x1]`` берётся y линейки (горизонтальные куски); линейка «между»,
        если на доле ширины не меньше ``BETWEEN_ROWS_SHARE`` она лежит строго между ``y_top`` и
        ``y_bottom``. Работает и для изогнутой линейки: сравнение идёт по её y в каждой точке.

        Args:
            x0, x1: Общая ширина двух рядов, пиксели.
            y_top: Уровень верхнего ряда.
            y_bottom: Уровень нижнего ряда.

        Returns:
            ``True`` — ряды разделены линейкой.
        """
        if not self.lines or x1 <= x0 or y_bottom <= y_top:
            return False
        xs = np.arange(int(np.floor(x0)), int(np.ceil(x1)) + 1, dtype=np.float64)
        for line in self.lines:
            covered = np.zeros(len(xs), dtype=bool)
            for a, b in zip(line[:-1], line[1:]):
                dx, dy = b[0] - a[0], b[1] - a[1]
                if abs(dx) < abs(dy) or dx == 0:
                    continue  # вертикальный кусок ряды не делит
                left, right = sorted((a[0], b[0]))
                inside = (xs >= left) & (xs <= right)
                if not inside.any():
                    continue
                ys = a[1] + (xs[inside] - a[0]) * dy / dx
                covered[inside] |= (ys > y_top) & (ys < y_bottom)
            if covered.mean() >= BETWEEN_ROWS_SHARE:
                return True
        return False

    def separators(self) -> list[tuple[int, int, int, int]]:
        """Вертикальные куски линеек как полосы запрета сцепки ``(x0, x1, y0, y1)``.

        Формат — как у межколонников (``columns.separators_for_segmentation``): через такую полосу
        ``segment._crosses`` строку не собирает. Изогнутая вертикаль даёт цепочку коротких полос.
        """
        out: list[tuple[int, int, int, int]] = []
        for a, b in self._pieces():
            if abs(b[1] - a[1]) <= abs(b[0] - a[0]):
                continue
            x_lo = int(np.floor(min(a[0], b[0]))) - SEPARATOR_PAD_PX
            x_hi = int(np.ceil(max(a[0], b[0]))) + SEPARATOR_PAD_PX
            y_lo = int(np.floor(min(a[1], b[1])))
            y_hi = int(np.ceil(max(a[1], b[1])))
            out.append((x_lo, x_hi, y_lo, y_hi))
        return out

    def mask(self, shape: tuple[int, int], scale: float = 1.0, thickness: int = 2) -> np.ndarray:
        """Растровая маска линеек в сетке, крупнее рабочей копии в ``scale`` раз.

        Args:
            shape: Размер маски ``(высота, ширина)``.
            scale: Во сколько раз сетка маски крупнее рабочей копии.
            thickness: Толщина линии маски, пиксели маски.

        Returns:
            Булева маска: ``True`` — пиксель линейки.
        """
        canvas = np.zeros(shape, dtype=np.uint8)
        for line in self.lines:
            points = np.round(line * scale).astype(np.int32)
            cv2.polylines(canvas, [points], False, 1, max(1, int(thickness)))
        return canvas.astype(bool)

    def in_area(self, box: tuple[int, int, int, int], rotate_cw: int, forward) -> "BarrierLines":
        """Линейки в системе ВЫПРЯМЛЕННОЙ вырезки области ``box``.

        Args:
            box: Рамка области ``(x0, y0, x1, y1)`` в пикселях полосы.
            rotate_cw: Поворот области (0, 90, 180, 270).
            forward: Перевод точек вырезки в выпрямленный кадр ``(points, size, rotate_cw)``
                (``page._forward_points``).

        Returns:
            Линейки, задевающие область, сдвинутые и повёрнутые вместе с вырезкой.
        """
        x0, y0, x1, y1 = box
        size = (x1 - x0, y1 - y0)
        out = []
        for line in self.lines:
            if line[:, 0].max() < x0 or line[:, 0].min() > x1 or line[:, 1].max() < y0 or line[:, 1].min() > y1:
                continue
            shifted = line - np.array([x0, y0], dtype=np.float64)
            out.append(forward(shifted, size, rotate_cw) if rotate_cw else shifted)
        return BarrierLines(tuple(out))

    def as_tuples(self) -> tuple[tuple[tuple[float, float], ...], ...]:
        """Ломаные кортежами — для ``LayoutHints.rules``."""
        return tuple(tuple((float(x), float(y)) for x, y in line) for line in self.lines)


def _orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    """Знак поворота ``p → q → r``: > 0 — против часовой, < 0 — по часовой, 0 — на одной прямой."""
    return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))


def _on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
    """Лежит ли точка ``r``, коллинеарная ``p–q``, в габарите отрезка ``p–q``."""
    return min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])


def _segments_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    """Пересекаются ли отрезки ``a–b`` и ``c–d`` (включая касание и наложение на одной прямой).

    Args:
        a, b: Концы первого отрезка.
        c, d: Концы второго.

    Returns:
        ``True`` — у отрезков есть общая точка.
    """
    d1, d2 = _orient(c, d, a), _orient(c, d, b)
    d3, d4 = _orient(a, b, c), _orient(a, b, d)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True
    return (
        (d1 == 0 and _on_segment(c, d, a))
        or (d2 == 0 and _on_segment(c, d, b))
        or (d3 == 0 and _on_segment(a, b, c))
        or (d4 == 0 and _on_segment(a, b, d))
    )


__all__ = ["BarrierLines"]
