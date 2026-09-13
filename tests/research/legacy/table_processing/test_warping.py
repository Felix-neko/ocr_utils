"""Прослеживание линеек, меры геометрии и выпрямители — на синтетике с известным ответом."""

from __future__ import annotations

import numpy as np

from research.legacy.table_processing.detection import ruling
from research.legacy.table_processing.warping import metrics
from research.legacy.table_processing.warping.engines import deskew, rules_separable
from research.legacy.table_processing.warping.trace import trace

from tests.research.legacy.table_processing.synthetic import DPI, bend, make_table

MM_PER_PX = 25.4 / DPI


def _table():
    return make_table(
        upright={(0, 0): "Области", (0, 1): "план", (0, 2): "факт", (1, 0): "Архангельская", (1, 1): "22030"}
    )


# Изгиб для тестов: дуга в полпериода амплитудой 4 px при ширине таблицы 520 px. Выше
# нельзя, и это не каприз теста, а предел ДЕТЕКТОРА: морфологическое открытие ищет 94 px
# краски подряд В ОДНОЙ СТРОКЕ, поэтому линейка видна, пока её уход по вертикали внутри
# этого окна не превышает её собственной толщины. Замер на синтетике: при амплитуде 8 px
# прослеженная сагитта не растёт, а падает — линейка начинает рваться на куски.
BEND_PX = 4.0
BEND_PERIODS = 0.5


def test_trace_sees_the_bend():
    """Прослеженная сагитта у изогнутой линейки заметно больше, чем у прямой."""
    straight, _ = trace(ruling.find_lines(_table().image, DPI), DPI)
    bent, _ = trace(ruling.find_lines(bend(_table().image, BEND_PX, BEND_PERIODS), DPI), DPI)
    assert straight and bent, "линейки обязаны прослеживаться в обоих случаях"
    flat = max(line.sagitta for line in straight)
    curved = max(line.sagitta for line in bent)
    assert curved > flat * 2, f"прямая {flat:.2f}, изогнутая {curved:.2f}"


def test_metrics_see_the_bend():
    straight = metrics.measure(_table().image, DPI)
    bent = metrics.measure(bend(_table().image, BEND_PX, BEND_PERIODS), DPI)
    assert bent.sagitta_max_mm > straight.sagitta_max_mm * 2


def test_separable_straightens_the_bend():
    bent = bend(_table().image, BEND_PX, BEND_PERIODS)
    before = metrics.measure(bent, DPI)
    result = rules_separable.run(bent, DPI)
    assert result.changed, result.note
    after = metrics.measure(result.image, DPI)
    assert after.sagitta_max_mm < before.sagitta_max_mm / 2, f"{before.sagitta_max_mm} -> {after.sagitta_max_mm}"


def test_straight_table_is_not_damaged():
    """Прямую таблицу выпрямитель портить не имеет права."""
    table = _table().image
    before = metrics.measure(table, DPI)
    result = rules_separable.run(table, DPI)
    after = metrics.measure(result.image, DPI)
    assert after.sagitta_max_mm <= before.sagitta_max_mm + 0.05
    ink_before = np.count_nonzero(ruling.binarize(table))
    ink_after = np.count_nonzero(ruling.binarize(result.image))
    assert ink_after >= ink_before * 0.98, "масса краски не должна падать больше чем на 2%"


def test_deskew_removes_the_tilt():
    tilted = make_table(upright={(0, 0): "Области", (1, 1): "22030"}, skew_deg=1.2).image
    before = metrics.measure(tilted, DPI)
    result = deskew.run(tilted, DPI)
    assert result.changed, result.note
    after = metrics.measure(result.image, DPI)
    assert abs(after.angle_median_deg) < abs(before.angle_median_deg) / 3


def test_warper_never_raises_on_a_blank_page():
    blank = np.full((400, 600), 245, np.uint8)
    for module in (deskew, rules_separable):
        result = module.run(blank, DPI)
        assert result.image.shape == blank.shape
        assert result.note
