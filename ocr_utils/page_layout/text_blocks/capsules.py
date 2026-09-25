"""Капсулы: отрезок с толщиной. На них стоят зоны поиска при сцепке кусков строк.

Зона поиска куска строки — «колбаска»: отрезок, продолжающий ход куска, раздутый на долю высоты
буквы. Проверять пересечение таких зон растеризацией полигонов дорого и неточно, а капсульная
геометрия отвечает точно и в десяток строк: зоны пересекаются тогда и только тогда, когда
расстояние между их осевыми отрезками не больше суммы радиусов.

Несимметричная зона (у точки и запятой она должна тянуться вверх, к своей строке, а не вниз, к
соседней) сводится к обычной симметричной: полоса ``[y − вверх, y + вниз]`` — это капсула
радиуса ``(вверх + вниз) / 2`` с осью, поднятой на ``(вверх − вниз) / 2``. Поэтому во всей
геометрии один радиус и одна формула.

Ординаты растут ВНИЗ (координаты картинки), поэтому «вверх» — это меньшие y.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Меньше этого считаем нулём: длины здесь — пиксели рабочей копии, доли пикселя не значимы.
EPS = 1e-9


@dataclass(frozen=True)
class Capsule:
    """Зона поиска: осевой отрезок ``(ax, ay)–(bx, by)`` с радиусом (пиксели рабочей копии).

    Начало отрезка ``(ax, ay)`` — тот конец, от которого зона выпущена: от него меряются и угол
    подхода, и удалённость точки встречи.
    """

    ax: float
    ay: float
    bx: float
    by: float
    radius: float
    # Радиус на дальнем конце. Зона поиска — КОНУС, а не колбаска: у самого куска направление
    # известно точно, а дальше оценка наклона шумит (замер на 1973/08 с.85: направления концов
    # двух соседних слов разошлись на 8 и 10 градусов, и зоны прошли в 20 px друг от друга при
    # сумме радиусов 12.6, хотя сами куски стояли на одной высоте с точностью 4 px). ``None`` —
    # радиус постоянный, то есть обычная колбаска.
    radius_end: float | None = None

    def radius_at(self, along: float) -> float:
        """Радиус на доле ``along`` (0 — начало отрезка, 1 — конец)."""
        if self.radius_end is None:
            return self.radius
        return self.radius + (self.radius_end - self.radius) * min(max(along, 0.0), 1.0)

    @property
    def length(self) -> float:
        """Длина осевого отрезка."""
        return math.hypot(self.bx - self.ax, self.by - self.ay)

    @property
    def direction(self) -> tuple[float, float]:
        """Единичный вектор от начала отрезка к его концу; у вырожденного — ``(0, 0)``."""
        length = self.length
        if length < EPS:
            return 0.0, 0.0
        return (self.bx - self.ax) / length, (self.by - self.ay) / length


@dataclass(frozen=True)
class Contact:
    """Наибольшее сближение двух зон: где встретились, под какими углами и как далеко от начал."""

    ox: float
    oy: float
    gap: float  # расстояние между осевыми отрезками (0, если они пересекаются)
    angle_first: float  # ∠O A₁A₂ в градусах
    angle_second: float  # ∠O B₁B₂ в градусах
    reach_first: float  # как далеко точка встречи от начала первой зоны
    reach_second: float


def band(x0: float, x1: float, y: float, up: float, down: float) -> Capsule:
    """Капсула из несимметричной горизонтальной полосы ``[y − up, y + down]``.

    Ось поднимается на ``(up − down) / 2``, радиус берётся ``(up + down) / 2`` — полоса
    получается ровно та же, а геометрия остаётся симметричной.

    Args:
        x0, x1: Края полосы по x.
        y: Средняя линия исходной полосы.
        up: На сколько полоса тянется вверх от средней линии.
        down: На сколько вниз.

    Returns:
        Равносильная симметричная капсула.
    """
    shift = (up - down) / 2.0
    return Capsule(ax=x0, ay=y - shift, bx=x1, by=y - shift, radius=(up + down) / 2.0)


def segment_distance(first: Capsule, second: Capsule) -> tuple[float, float, float, float, float]:
    """Наименьшее расстояние между осевыми отрезками и ближайшие точки на каждом.

    Минимум квадрата расстояния по двум параметрам с зажимом обоих в ``[0, 1]``; вырожденный
    случай (параллельные отрезки, нулевая длина) разбирается отдельно.

    Args:
        first, second: Зоны, у которых берутся осевые отрезки.

    Returns:
        ``(расстояние, x на первом, y на первом, x на втором, y на втором)``.
    """
    ux, uy = first.bx - first.ax, first.by - first.ay
    vx, vy = second.bx - second.ax, second.by - second.ay
    wx, wy = first.ax - second.ax, first.ay - second.ay
    a = ux * ux + uy * uy
    b = ux * vx + uy * vy
    c = vx * vx + vy * vy
    d = ux * wx + uy * wy
    e = vx * wx + vy * wy
    denominator = a * c - b * b
    s_top, s_bottom = denominator, denominator
    t_top, t_bottom = denominator, denominator
    if denominator < EPS:
        # Отрезки параллельны или вырождены: первый параметр прижимаем к началу.
        s_top, s_bottom = 0.0, 1.0
        t_top, t_bottom = e, c
    else:
        s_top = b * e - c * d
        t_top = a * e - b * d
        if s_top < 0.0:
            s_top = 0.0
            t_top, t_bottom = e, c
        elif s_top > s_bottom:
            s_top = s_bottom
            t_top, t_bottom = e + b, c
    if t_top < 0.0:
        t_top = 0.0
        if -d < 0.0:
            s_top = 0.0
        elif -d > a:
            s_top = s_bottom
        else:
            s_top, s_bottom = -d, a
    elif t_top > t_bottom:
        t_top = t_bottom
        if (-d + b) < 0.0:
            s_top = 0.0
        elif (-d + b) > a:
            s_top = s_bottom
        else:
            s_top, s_bottom = -d + b, a
    s = 0.0 if abs(s_bottom) < EPS else s_top / s_bottom
    t = 0.0 if abs(t_bottom) < EPS else t_top / t_bottom
    px, py = first.ax + s * ux, first.ay + s * uy
    qx, qy = second.ax + t * vx, second.ay + t * vy
    return math.hypot(px - qx, py - qy), px, py, qx, qy


def angle_at(capsule: Capsule, ox: float, oy: float, lever: float = 0.0) -> float:
    """Угол ``∠O A₁A₂`` в градусах: насколько точка встречи уходит с направления зоны.

    Ноль — точка лежит ровно по ходу зоны, 180° — ровно назад. У двух кусков одной строки зоны
    выпущены навстречу, и обе видят точку встречи почти прямо перед собой.

    Вершина угла отодвигается НАЗАД по оси на ``lever``. Без этого рычага угол при коротком
    зазоре бессмыслен: соседние слова стоят в 12 px друг от друга, точка встречи лежит посредине,
    и законное смещение в 4 px даёт уже 18°, а в 6 px — 27°, то есть верные соединения отбиваются
    (замер на 1973/08 с.85: угол отбивал 197 пар из 573). С рычагом в полторы высоты строчной те
    же 4 px дают 5°, а уход на соседнюю строку (22 px) по-прежнему даёт 46°.

    Args:
        capsule: Зона; вершина угла — её начало, отодвинутое назад на ``lever``.
        ox, oy: Точка встречи.
        lever: На сколько отодвинуть вершину назад по оси зоны.

    Returns:
        Угол от 0 до 180; у вырожденной зоны или совпавшей с вершиной точки — 180.
    """
    dx, dy = capsule.bx - capsule.ax, capsule.by - capsule.ay
    vertex_x, vertex_y = capsule.ax, capsule.ay
    if lever > 0.0:
        unit_x, unit_y = capsule.direction
        vertex_x, vertex_y = capsule.ax - unit_x * lever, capsule.ay - unit_y * lever
    tx, ty = ox - vertex_x, oy - vertex_y
    first = math.hypot(dx, dy)
    second = math.hypot(tx, ty)
    if first < EPS or second < EPS:
        return 180.0
    cosine = max(-1.0, min(1.0, (dx * tx + dy * ty) / (first * second)))
    return math.degrees(math.acos(cosine))


def contact_of(first: Capsule, second: Capsule, lever_first: float = 0.0, lever_second: float = 0.0) -> Contact | None:
    """Соприкосновение двух зон или ``None``, если они не пересекаются.

    Args:
        first, second: Зоны поиска.
        lever_first, lever_second: Насколько отодвинуть назад вершины углов (см. ``angle_at``).

    Returns:
        :class:`Contact` с точкой встречи, углами подхода от обеих зон и удалённостью точки от
        их начал.
    """
    distance, px, py, qx, qy = segment_distance(first, second)
    # Радиусы берутся В ТОЧКАХ СБЛИЖЕНИЯ: у конуса они зависят от того, как далеко от куска
    # зоны встретились.
    first_along = math.hypot(px - first.ax, py - first.ay) / max(first.length, EPS)
    second_along = math.hypot(qx - second.ax, qy - second.ay) / max(second.length, EPS)
    if distance > first.radius_at(first_along) + second.radius_at(second_along):
        return None
    ox, oy = (px + qx) / 2.0, (py + qy) / 2.0
    return Contact(
        ox=ox,
        oy=oy,
        gap=distance,
        angle_first=angle_at(first, ox, oy, lever_first),
        angle_second=angle_at(second, ox, oy, lever_second),
        reach_first=math.hypot(ox - first.ax, oy - first.ay),
        reach_second=math.hypot(ox - second.ax, oy - second.ay),
    )


__all__ = ["Capsule", "Contact", "angle_at", "band", "contact_of", "segment_distance"]
