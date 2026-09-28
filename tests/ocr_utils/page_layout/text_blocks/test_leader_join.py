"""Сращивание строк, сошедшихся на общей точке отточия (``text_blocks.leader_join``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
from ocr_utils.page_layout.text_blocks.leader_join import _piece_of, angle_ok
from ocr_utils.page_layout.text_blocks.page import analyse_gray
from tests.ocr_utils.page_layout.orientation.synthetic import LINE_STEP
from tests.ocr_utils.page_layout.text_blocks.synthetic import stuck_leader_page

ROWS = 12
# Шаг строк синтетики на рабочей копии (150 dpi) — для проверки зон вне страницы.
PITCH = 44.0


def _axes(join: bool, **kwargs) -> list:
    """Оси синтетической таблицы сверху вниз, со сращиванием или без."""
    analysis = analyse_gray(stuck_leader_page(rows=ROWS, **kwargs), InkEngine(join_leaders=join))
    return sorted(analysis.axes, key=lambda axis: axis.cy)


@pytest.mark.parametrize("number_w", [100, 60], ids=["три цифры — длинный кусок", "две цифры — короткий кусок"])
def test_stuck_leader_rows_join(number_w: int):
    """Без шага ряд — две строки над одними точками; со сращиванием — одна строка от слов до числа."""
    before = _axes(False, number_w=number_w)
    after = _axes(True, number_w=number_w)
    assert len(before) == 2 * ROWS, "синтетика не воспроизводит раздвоение ряда"
    assert len(after) == ROWS
    right = max(axis.x1 for axis in before)
    left = min(axis.x0 for axis in before)
    assert all(axis.x0 <= left + 1 and axis.x1 >= right - 1 for axis in after), "сросшаяся строка не накрывает ряд"


def test_number_on_other_row_height_does_not_join():
    """Число с прилипшими точками на полшага ниже ряда — не его продолжение: строки не срастаются."""
    before = _axes(False, shift=LINE_STEP)
    after = _axes(True, shift=LINE_STEP)
    assert len(after) == len(before) == 2 * ROWS


def _row_boxes(letters: int, width: float = 9.0, gap: float = 3.0, top: float = 100.0, height: float = 11.0):
    """Глифы горизонтальной строки ``(n, 4)``: ``letters`` букв ширины ``width`` через ``gap``."""
    x0 = 50.0 + np.arange(letters) * (width + gap)
    return np.column_stack([x0, np.full(letters, top), x0 + width, np.full(letters, top + height)])


def test_angle_long_piece_follows_its_own_tangent():
    """Длинный кусок: точка встречи по ходу его оси проходит, уход вбок на 30° — нет (наклон соседей не влияет)."""
    piece = _piece_of(_row_boxes(8), 11.0)
    x, y = piece.x1 + 120.0, piece.y_at(piece.x1)
    assert angle_ok(piece, True, PITCH, 0.0, (x, y))
    assert not angle_ok(piece, True, PITCH, 0.0, (x, y + 120.0 * math.tan(math.radians(30))))
    # Своя ось важнее наклона соседей: у длинного куска зона идёт по касательной, а не по ``slope``.
    assert not angle_ok(piece, True, PITCH, math.tan(math.radians(30)), (x, y + 120.0 * math.tan(math.radians(30))))


def test_angle_short_piece_follows_neighbour_slope():
    """Короткий кусок (две буквы): зона идёт по местному наклону строк, как вторичная зона сцепки."""
    piece = _piece_of(_row_boxes(2), 11.0)
    assert piece.letters < 3
    x = piece.x0 - 60.0
    rise = 60.0 * math.tan(math.radians(15))
    meeting = (x, piece.cy - rise)
    # При горизонтальных соседях точка на 15° вверх-влево — ещё в допуске; на 30° — уже нет.
    assert angle_ok(piece, False, PITCH, 0.0, meeting)
    assert not angle_ok(piece, False, PITCH, 0.0, (x, piece.cy - 60.0 * math.tan(math.radians(30))))
    # Те же 30°, но соседние строки сами идут с таким наклоном — зона смотрит вдоль них, и угол мал.
    slope = math.tan(math.radians(30))
    assert angle_ok(piece, False, PITCH, slope, (x, piece.cy - 60.0 * slope))
