"""Куски строк и сцепка по зонам поиска: разбор на буквы, якоря, правила соединения."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ocr_utils.curved_layout import zones as zn
from ocr_utils.curved_layout.capsules import contact_of
from ocr_utils.curved_layout.pieces import (
    Piece,
    anchors_of,
    axis_residual,
    fit_axis,
    is_low_mark,
    letters_of,
    merged,
    pitch_of,
)

X_HEIGHT = 10.0
PITCH = 22.8


def _piece(x0: float, y: float, letters: int, x_h: float = X_HEIGHT, step: float = 12.0, mark: bool = False) -> Piece:
    """Кусок из ``letters`` букв на уровне ``y``: якоря с постоянным шагом по x."""
    xs = x0 + np.arange(letters) * step
    anchors = np.column_stack([xs, np.full(letters, y)])
    sizes = np.full((letters, 2), [x_h * 0.7, x_h])
    return Piece(
        blobs=(int(x0),),
        anchors=anchors,
        sizes=sizes,
        marks=np.full(letters, mark),
        x0=float(xs[0] - x_h / 2),
        y0=float(y - x_h / 2),
        x1=float(xs[-1] + x_h / 2),
        y1=float(y + x_h / 2),
        x_h=x_h,
        letter_w=x_h * 0.7,
        leader_dots=0,
    )


def test_low_mark_is_a_dot_but_not_a_hyphen():
    """Низкая метка — точка и запятая; дефис не проходит по ширине."""
    assert is_low_mark(width=3.0, height=3.0, x_h=X_HEIGHT)
    assert is_low_mark(width=4.0, height=5.0, x_h=X_HEIGHT)
    assert not is_low_mark(width=9.5, height=2.0, x_h=X_HEIGHT)  # дефис: тонкий, но широкий
    assert not is_low_mark(width=7.0, height=10.0, x_h=X_HEIGHT)  # строчная буква


def test_dot_anchor_rises_to_the_line_level():
    """Якорь точки поднимается от её НИЗА на полвысоты строчной — к оси строки."""
    # Точка 3×3 с центром на 100: низ на 101.5, ось строки должна быть на 96.5.
    boxes = np.array([[50.0, 100.0, 3.0, 3.0], [70.0, 95.0, 7.0, 10.0]])
    anchors, marks = anchors_of(boxes, X_HEIGHT)
    assert marks.tolist() == [True, False]
    assert anchors[0, 1] == pytest.approx(101.5 - 0.5 * X_HEIGHT)
    assert anchors[1, 1] == pytest.approx(95.0)  # обычная буква: якорь остался центром


def test_capital_and_lowercase_anchors_land_on_one_level():
    """Прописная, цифра и строчная на одной базовой линии дают ОДИН уровень якоря."""
    # Базовая линия 100. Строчная 7×10 (центр бокса 95), прописная 6×14 (93), цифра 8×13 (93.5):
    # по центрам боксов уровни разошлись бы на два пикселя, по базовой линии — ни на сколько.
    boxes = np.array([[10.0, 95.0, 7.0, 10.0], [20.0, 93.0, 6.0, 14.0], [30.0, 93.5, 8.0, 13.0]])
    anchors, _ = anchors_of(boxes, X_HEIGHT)
    assert anchors[:, 1] == pytest.approx(100.0 - 0.5 * X_HEIGHT)


def test_descender_does_not_drag_the_anchor_down():
    """Низ «р» лежит ниже базовой линии, но якорь остаётся на уровне строки."""
    # Пять строчных с низом на 100 и одна с выносом вниз на три пикселя.
    boxes = np.array(
        [[10.0, 95.0, 7.0, 10.0], [20.0, 95.0, 7.0, 10.0], [30.0, 95.0, 7.0, 10.0]]
        + [[40.0, 96.5, 7.0, 13.0], [50.0, 95.0, 7.0, 10.0], [60.0, 95.0, 7.0, 10.0]]
    )
    anchors, _ = anchors_of(boxes, X_HEIGHT)
    assert anchors[:, 1] == pytest.approx(95.0, abs=0.2)


def test_letter_is_kept_when_its_box_centre_falls_into_a_hole():
    """Буква приписывается к сгустку по своему пикселю, а не по метке под центром бокса."""
    # Буква вида «С»: просвет открыт вправо и шире ядра смыкания, поэтому под центром её бокса
    # краски нет ни до смыкания, ни после — по центру она потерялась бы.
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[4:20, 2:6] = 1  # левая стойка
    mask[4:8, 2:18] = 1  # верхняя перекладина
    mask[16:20, 2:18] = 1  # нижняя перекладина
    labels = cv2.connectedComponents(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, 8), np.uint8)))[1]
    boxes, owners = letters_of(mask, labels)
    assert boxes.shape[0] == 1
    row, column = int(round(boxes[0, 1])), int(round(boxes[0, 0]))
    assert labels[row, column] == 0  # метка под центром бокса нулевая
    assert owners[0] == labels[12, 3]  # а буква всё равно у своего сгустка


def test_hyphen_is_not_a_long_piece():
    """Дефис: отношение сторон 3, но буква одна — длинным куском он не считается."""
    hyphen = Piece(
        blobs=(1,),
        anchors=np.array([[100.0, 50.0]]),
        sizes=np.array([[9.0, 3.0]]),
        marks=np.array([False]),
        x0=95.0,
        y0=48.0,
        x1=104.0,
        y1=51.0,
        x_h=X_HEIGHT,
        letter_w=9.0,
        leader_dots=0,
    )
    assert not hyphen.long
    assert _piece(100.0, 50.0, letters=4).long


def test_axis_follows_a_bowed_piece():
    """Ось куска — парабола по якорям: заданный прогиб восстанавливается."""
    xs = np.linspace(0.0, 200.0, 9)
    sagitta = 6.0
    ys = 100.0 + sagitta * (1.0 - ((xs - 100.0) / 100.0) ** 2)
    coefficients, centre = fit_axis(np.column_stack([xs, ys]))
    middle = float(np.polyval(coefficients, 100.0 - centre))
    ends = float(np.polyval(coefficients, 0.0 - centre))
    assert middle - ends == pytest.approx(sagitta, rel=0.05)


def test_tangent_at_the_end_follows_the_local_run():
    """Направление на конце берётся по крайним буквам, а не по хорде всего куска."""
    # Кусок идёт горизонтально, а его правый хвост поднимается: касательная должна это увидеть.
    xs = np.arange(9) * 12.0
    ys = np.concatenate([np.full(6, 100.0), [97.0, 94.0, 91.0]])
    piece = _piece(0.0, 100.0, letters=9)
    piece = Piece(**{**piece.__dict__, "anchors": np.column_stack([xs, ys])})
    _, _, dx, dy = piece.tangent(at_start=False)
    assert dy / dx == pytest.approx(-0.25, abs=0.1)


def test_axis_residual_catches_a_jump_to_the_next_line():
    """Кусок, собранный из двух строк, даёт всплеск остатка оси."""
    good = _piece(0.0, 100.0, letters=8)
    jumped = merged(_piece(0.0, 100.0, letters=4), _piece(60.0, 100.0 - PITCH, letters=4))
    assert axis_residual(good) < 0.5
    assert axis_residual(jumped) > zn.AXIS_MAX_RESID_XH * X_HEIGHT


def test_pitch_is_measured_by_overlapping_neighbours():
    """Шаг строк считается по кускам, стоящим друг над другом."""
    pieces = [_piece(0.0, 100.0 + row * PITCH, letters=8) for row in range(6)]
    assert pitch_of(pieces) == pytest.approx(PITCH, abs=1.0)


def test_zones_of_the_same_line_meet_and_of_neighbouring_lines_do_not():
    """Зоны двух кусков одной строки встречаются; куска соседней строки — нет."""
    left = _piece(0.0, 100.0, letters=5)
    right = _piece(80.0, 100.0, letters=5)
    below = _piece(80.0, 100.0 + PITCH, letters=5)
    right_zone = zn.long_zones(left, 0)[1]
    lever = zn.ANGLE_LEVER_XH * X_HEIGHT
    assert contact_of(right_zone.capsule, zn.long_zones(right, 1)[0].capsule, lever, lever) is not None
    assert contact_of(right_zone.capsule, zn.long_zones(below, 2)[0].capsule, lever, lever) is None


def test_dot_zone_reaches_its_own_line():
    """Первичная зона точки достаёт до оси своей строки, а не до соседней."""
    # Точка стоит за концом слова: конец на x = 53, зона слова тянется до 78, точка — с 65.
    dot = _piece(70.0, 100.0, letters=1, mark=True)
    own = _piece(0.0, 100.0, letters=5)
    below = _piece(0.0, 100.0 + PITCH, letters=5)
    zone = zn.primary_zone(dot, 0).capsule
    assert contact_of(zone, zn.long_zones(own, 1)[1].capsule) is not None
    assert contact_of(zone, zn.long_zones(below, 2)[1].capsule) is None


def test_accept_keeps_one_link_per_side():
    """С каждой стороны куска — не больше одного соединения: побеждает ближайшее."""
    links = [
        zn.Link(left=0, right=1, reach_left=30.0, reach_right=30.0, gap=0.0),
        zn.Link(left=0, right=2, reach_left=10.0, reach_right=10.0, gap=0.0),
    ]
    accepted = zn.accept(links, 3)
    # Справа у куска 0 два претендента; берётся тот, чья точка встречи ближе к началу зоны.
    assert {(link.left, link.right) for link in accepted} == {(0, 2)}


def test_accept_requires_a_mutual_choice():
    """Односторонний выбор не принимается: A тянется к B, а B — к C, значит пары нет."""
    links = [
        zn.Link(left=0, right=1, reach_left=10.0, reach_right=30.0, gap=0.0),
        zn.Link(left=2, right=1, reach_left=5.0, reach_right=5.0, gap=0.0),
    ]
    accepted = zn.accept(links, 3)
    # Кусок 1 слева выбирает 2 (ближе), поэтому соединение 0→1 невзаимно и отбрасывается.
    assert {(link.left, link.right) for link in accepted} == {(2, 1)}


def test_a_piece_may_link_on_both_sides():
    """Один кусок может получить соседа и слева, и справа — это разные слоты."""
    links = [
        zn.Link(left=0, right=1, reach_left=10.0, reach_right=10.0, gap=0.0),
        zn.Link(left=1, right=2, reach_left=10.0, reach_right=10.0, gap=0.0),
    ]
    accepted = zn.accept(links, 3)
    assert {(link.left, link.right) for link in accepted} == {(0, 1), (1, 2)}


def test_guard_rejects_a_merge_that_spans_two_lines():
    """Предохранитель отменяет слияние, если после него ось перестаёт описывать кусок."""
    left = _piece(0.0, 100.0, letters=5)
    same = _piece(80.0, 100.0, letters=5)
    below = _piece(80.0, 100.0 + PITCH, letters=5)
    assert zn._guard(left, same, PITCH)
    assert not zn._guard(left, below, PITCH)


def test_comma_counts_as_a_low_mark():
    """Запятая — такая же низкая метка, как точка: она тоже сидит на базовой линии."""
    # Замер по корпусу пака-1 при иксе 10 px: точка 3×3, запятая 3×5, двоеточие 3×8.
    assert is_low_mark(width=3.0, height=3.0, x_h=X_HEIGHT)  # точка
    assert is_low_mark(width=3.0, height=5.0, x_h=X_HEIGHT)  # запятая
    assert is_low_mark(width=3.0, height=6.0, x_h=X_HEIGHT)  # двоеточие
    assert not is_low_mark(width=6.0, height=7.0, x_h=X_HEIGHT)  # строчная «о»


def test_shape_stats_ignore_the_dip_over_a_dot():
    """Провисание оси над точкой не попадает в меры формы строки."""
    from ocr_utils.curved_layout.lines import _shape_stats

    xs = np.arange(0.0, 60.0, 1.0)
    ys = np.full(xs.size, 100.0)
    ys[-4:] = 105.0  # хвост оси нырнул к базовой линии точки
    points = np.column_stack([xs, ys])
    _, _, bend_all, _ = _shape_stats(points, 150.0)
    _, _, bend_kept, _ = _shape_stats(points, 150.0, ((56.0, 60.0),))
    assert bend_all > bend_kept
    assert bend_kept == pytest.approx(0.0, abs=1e-6)
