"""Капсульная геометрия: расстояние между отрезками, углы подхода, несимметричная полоса."""

from __future__ import annotations

import math

import pytest

from ocr_utils.curved_layout.capsules import Capsule, angle_at, band, contact_of, segment_distance


def test_parallel_segments_keep_their_distance():
    """У параллельных отрезков расстояние равно зазору между ними."""
    first = Capsule(ax=0.0, ay=0.0, bx=100.0, by=0.0, radius=3.0)
    second = Capsule(ax=0.0, ay=20.0, bx=100.0, by=20.0, radius=3.0)
    distance, _, _, _, _ = segment_distance(first, second)
    assert distance == pytest.approx(20.0)


def test_crossing_segments_touch():
    """Пересекающиеся отрезки: расстояние ноль, точка встречи — в самом пересечении."""
    first = Capsule(ax=0.0, ay=0.0, bx=100.0, by=100.0, radius=1.0)
    second = Capsule(ax=0.0, ay=100.0, bx=100.0, by=0.0, radius=1.0)
    contact = contact_of(first, second)
    assert contact is not None
    assert contact.gap == pytest.approx(0.0, abs=1e-6)
    assert contact.ox == pytest.approx(50.0, abs=1e-6)
    assert contact.oy == pytest.approx(50.0, abs=1e-6)


def test_capsules_touch_only_within_radii():
    """Зоны соприкасаются, пока зазор не больше суммы радиусов."""
    first = Capsule(ax=0.0, ay=0.0, bx=100.0, by=0.0, radius=6.0)
    near = Capsule(ax=0.0, ay=11.0, bx=100.0, by=11.0, radius=6.0)
    far = Capsule(ax=0.0, ay=13.0, bx=100.0, by=13.0, radius=6.0)
    assert contact_of(first, near) is not None
    assert contact_of(first, far) is None


def test_degenerate_segment_is_a_point():
    """Вырожденный отрезок (кусок из одной буквы) не делит на ноль: это расстояние до точки."""
    point = Capsule(ax=50.0, ay=30.0, bx=50.0, by=30.0, radius=2.0)
    line = Capsule(ax=0.0, ay=0.0, bx=100.0, by=0.0, radius=2.0)
    distance, _, _, _, _ = segment_distance(point, line)
    assert distance == pytest.approx(30.0)


def test_segments_pointing_apart_measure_from_their_ends():
    """Отрезки, направленные в разные стороны, меряются по ближайшим концам."""
    left = Capsule(ax=0.0, ay=0.0, bx=10.0, by=0.0, radius=1.0)
    right = Capsule(ax=40.0, ay=0.0, bx=50.0, by=0.0, radius=1.0)
    distance, px, _, qx, _ = segment_distance(left, right)
    assert distance == pytest.approx(30.0)
    assert px == pytest.approx(10.0)
    assert qx == pytest.approx(40.0)


def test_angle_is_zero_along_the_zone_and_right_across_it():
    """Угол подхода: по ходу зоны — ноль, поперёк — прямой, назад — развёрнутый."""
    zone = Capsule(ax=0.0, ay=0.0, bx=100.0, by=0.0, radius=5.0)
    assert angle_at(zone, 50.0, 0.0) == pytest.approx(0.0)
    assert angle_at(zone, 0.0, 50.0) == pytest.approx(90.0)
    assert angle_at(zone, -50.0, 0.0) == pytest.approx(180.0)


def test_angle_does_not_depend_on_the_side():
    """Знак отклонения не важен: вверх и вниз дают один и тот же угол."""
    zone = Capsule(ax=0.0, ay=0.0, bx=100.0, by=0.0, radius=5.0)
    assert angle_at(zone, 100.0, 40.0) == pytest.approx(angle_at(zone, 100.0, -40.0))
    assert angle_at(zone, 100.0, 100.0) == pytest.approx(45.0)


def test_band_covers_the_asymmetric_stripe():
    """Несимметричная полоса сводится к симметричной капсуле и накрывает ровно себя.

    У точки отточия зона тянется вверх на 0.75 высоты буквы и вниз на 0.25: верхний край должен
    оказаться на уровне оси своей строки, нижний — не доставать до соседней.
    """
    x_height = 10.0
    zone = band(x0=0.0, x1=50.0, y=100.0, up=0.75 * x_height, down=0.25 * x_height)
    assert zone.radius == pytest.approx(5.0)
    top = zone.ay - zone.radius
    bottom = zone.ay + zone.radius
    assert top == pytest.approx(100.0 - 7.5)
    assert bottom == pytest.approx(100.0 + 2.5)


def test_low_mark_zone_reaches_its_own_line_and_not_the_next():
    """Зона точки достаёт до оси своей строки и не дотягивается до соседней.

    Замер 1973/08 с.85: высота буквы 10 px, шаг строк 22.8 px. Точка сидит на базовой линии,
    то есть на 5 px ниже оси своей строки; ось соседней — на 22.8 px ниже неё.
    """
    x_height, pitch = 10.0, 22.8
    dot_y = 100.0
    own_axis = dot_y - 0.5 * x_height
    next_axis = own_axis + pitch
    zone = band(x0=0.0, x1=6.0, y=dot_y, up=0.75 * x_height, down=0.25 * x_height)
    own = Capsule(ax=-30.0, ay=own_axis, bx=0.0, by=own_axis, radius=0.6 * x_height)
    neighbour = Capsule(ax=-30.0, ay=next_axis, bx=0.0, by=next_axis, radius=0.6 * x_height)
    assert contact_of(zone, own) is not None
    assert contact_of(zone, neighbour) is None


def test_neighbouring_line_stays_out_of_a_long_zone():
    """Зона длинного куска не достаёт до куска соседней строки: 12 px толщины против шага 22.8."""
    x_height, pitch = 10.0, 22.8
    reach = Capsule(ax=0.0, ay=0.0, bx=3.0 * x_height, by=0.0, radius=0.6 * x_height)
    same_line = Capsule(ax=40.0, ay=1.0, bx=10.0, by=1.0, radius=0.6 * x_height)
    next_line = Capsule(ax=40.0, ay=pitch, bx=10.0, by=pitch, radius=0.6 * x_height)
    assert contact_of(reach, same_line) is not None
    assert contact_of(reach, next_line) is None


def test_contact_reports_reach_from_the_zone_start():
    """Удалённость точки встречи меряется от НАЧАЛА зоны — по ней выбирается победитель."""
    first = Capsule(ax=0.0, ay=0.0, bx=30.0, by=0.0, radius=6.0)
    second = Capsule(ax=25.0, ay=0.0, bx=5.0, by=0.0, radius=6.0)
    contact = contact_of(first, second)
    assert contact is not None
    assert contact.reach_first == pytest.approx(math.hypot(contact.ox, contact.oy))
    assert contact.reach_first < first.length
