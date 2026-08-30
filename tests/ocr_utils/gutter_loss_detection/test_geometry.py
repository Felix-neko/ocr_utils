"""Разбор разворота: сгиб, строки, внутренние поля."""

import numpy as np

from ocr_utils.gutter_loss_detection.geometry import analyze_spread
from tests.ocr_utils.gutter_loss_detection.conftest import PITCH, WIDTH, draw_spread


def test_находит_строки_и_шаг(clean_spread):
    result = analyze_spread(clean_spread)
    assert result.problem == ""
    assert result.lines >= 25
    assert abs(result.pitch - PITCH) <= 2


def test_сгиб_на_середине(clean_spread):
    result = analyze_spread(clean_spread)
    assert abs(result.fold_at_middle - WIDTH / 2) <= 4


def test_наклон_сгиба_измеряется():
    result = analyze_spread(draw_spread(inner_margin=55, tilt=40.0))
    assert abs(result.tilt - 40.0) <= 8


def test_поле_растёт_вместе_с_отступом():
    narrow = analyze_spread(draw_spread(inner_margin=6))
    wide = analyze_spread(draw_spread(inner_margin=70))
    assert narrow.sides[0].tight < wide.sides[0].tight


def test_тень_у_корешка_не_считается_краской():
    """С тенью и без неё поле должно получаться одинаковым."""
    with_shadow = analyze_spread(draw_spread(inner_margin=55, shadow=True))
    without = analyze_spread(draw_spread(inner_margin=55, shadow=False))
    assert abs(with_shadow.sides[0].tight - without.sides[0].tight) < 0.3


def test_поворот_не_ломает_замер():
    """Поворот разворота не должен съедать измеренное поле."""
    straight = analyze_spread(draw_spread(inner_margin=55, tilt=0.0))
    tilted = analyze_spread(draw_spread(inner_margin=55, tilt=45.0))
    assert abs(straight.sides[0].tight - tilted.sides[0].tight) < 0.4


def test_одиночная_страница_не_мерится():
    single = np.full((1100, 800), 250.0, np.float32)
    assert analyze_spread(single).problem == "одиночная страница"


def test_пустой_кадр_не_мерится():
    blank = np.full((1100, 1600), 250.0, np.float32)
    assert analyze_spread(blank).problem != ""
