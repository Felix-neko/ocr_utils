"""Правило «пометка на полях» (``research.margin_marks.rule``) и признаки остатка (``features``) на синтетике."""

from __future__ import annotations

import numpy as np

from research.margin_marks.features import MM_PER_PX, component_measures
from research.margin_marks.rule import Condition, MarkRule

MM = 1.0 / MM_PER_PX


def _row(**changes) -> dict:
    """Строка признаков типичной пометки (волнистое подчёркивание 30 мм) с заменами."""
    row = {
        "info_kind": "штрих", "info_sources": "ink", "outcome": "объекты", "classes": "рисунок",
        "long_components": "1", "long_thickness_max_mm": "0.35", "corners": "0", "box_w_mm": "30.0",
        "box_h_mm": "6.0", "wobble_max_mm": "0.25", "ink_share": "0.05",
    }  # fmt: skip
    row.update({k: str(v) for k, v in changes.items()})
    return row


def test_underline_is_a_mark():
    """Тонкое волнистое подчёркивание размером со строку — пометка."""
    assert MarkRule().is_mark(_row())


def test_printed_rule_is_not_a_mark():
    """Ровная печатная линия (волнистость около нуля) — не пометка."""
    assert Condition.HAND in MarkRule().failed(_row(wobble_max_mm=0.02))


def test_frame_vignette_and_scheme_are_not_marks():
    """Рамка (угол), заставка (толстый штрих), схема (много компонент), большой рисунок — не пометки."""
    rule = MarkRule()
    assert Condition.NO_CORNER in rule.failed(_row(corners=1))
    assert Condition.THIN in rule.failed(_row(long_thickness_max_mm=0.7))
    assert Condition.FEW in rule.failed(_row(long_components=20))
    assert Condition.SIZE in rule.failed(_row(box_w_mm=120, box_h_mm=80))


def test_drawing_with_table_is_left_alone():
    """Рисунок вместе с таблицей и надпись правило не трогает."""
    rule = MarkRule()
    assert Condition.OUTCOME in rule.failed(_row(classes="рисунок|таблица"))
    assert Condition.OUTCOME in rule.failed(_row(outcome="надпись", classes=""))


def test_wobble_separates_straight_and_wavy_lines():
    """Волнистость прямой печатной линии — около нуля, синусоиды с размахом 1 мм — десятые доли мм."""
    straight = np.zeros((int(10 * MM), int(40 * MM)), bool)
    straight[straight.shape[0] // 2 : straight.shape[0] // 2 + 4, 5:-5] = True
    wavy = np.zeros_like(straight)
    xs = np.arange(5, wavy.shape[1] - 5)
    ys = (wavy.shape[0] // 2 + 0.5 * MM * np.sin(xs / (4 * MM))).astype(int)
    for dy in range(4):
        wavy[ys + dy, xs] = True
    gray = np.zeros(straight.shape, np.uint8)
    flat, = component_measures(straight, gray)
    wave, = component_measures(wavy, gray)
    assert flat["wobble_mm"] < 0.05
    assert wave["wobble_mm"] > 0.2
    assert not flat["corner"] and not wave["corner"]


def test_frame_corner_is_detected():
    """Угол рамки (горизонталь и вертикаль в одной компоненте) — ``corner``."""
    mask = np.zeros((int(20 * MM), int(20 * MM)), bool)
    mask[5:9, 5:-5] = True
    mask[5:-5, 5:9] = True
    (item,) = component_measures(mask, np.zeros(mask.shape, np.uint8))
    assert item["corner"]
