"""Балл ухода текста под корешок и вердикт."""

from ocr_utils.gutter_loss_detection.geometry import analyze_spread
from ocr_utils.gutter_loss_detection.metrics import THRESHOLD, is_tabular, side_bite, spread_bite, verdict
from tests.ocr_utils.gutter_loss_detection.conftest import draw_spread


def test_чистый_разворот_ниже_порога(clean_spread):
    assert spread_bite(analyze_spread(clean_spread)) < THRESHOLD


def test_съеденный_разворот_выше_порога(bitten_spread):
    assert spread_bite(analyze_spread(bitten_spread)) >= THRESHOLD


def test_балл_монотонен_по_полю():
    scores = [spread_bite(analyze_spread(draw_spread(inner_margin=m))) for m in (3, 20, 40, 70)]
    assert scores == sorted(scores, reverse=True)


def test_балл_в_единичном_отрезке():
    for margin in (0, 3, 30, 200):
        score = spread_bite(analyze_spread(draw_spread(inner_margin=margin)))
        assert 0.0 <= score <= 1.0


def test_вердикт_текст(bitten_spread):
    assert verdict(analyze_spread(bitten_spread)).code == "текст"


def test_вердикт_ок(clean_spread):
    assert verdict(analyze_spread(clean_spread)).code == "ок"


def test_вердикт_таблица():
    tabular = analyze_spread(draw_spread(inner_margin=3, rules=True))
    assert any(is_tabular(side) for side in tabular.sides)
    assert verdict(tabular).code == "таблица"


def test_балл_полосы_совпадает_с_отсутствием_поля(bitten_spread):
    for side in analyze_spread(bitten_spread).sides:
        assert side_bite(side) > 0.6


def test_неизмеримый_кадр_даёт_nan():
    import numpy as np

    blank = analyze_spread(np.full((1100, 1600), 250.0, np.float32))
    assert np.isnan(spread_bite(blank))
    assert verdict(blank).code == "ок"
