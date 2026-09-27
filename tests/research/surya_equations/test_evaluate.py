"""Тесты стенда surya_equations: счёт формул DeepSeek, слои, ужатие рамки, сопоставление и пересчёт на пак."""

import numpy as np
import pandas as pd
import pytest

from research.surya_equations.evaluate import box_quality, covers, estimate, match_page, per_page
from research.surya_equations.labels import Block, PageLabels, Verdict, snap_to_ink
from research.surya_equations.overlay import Category, page_cases
from research.surya_equations.sources import MM_PER_UNIT, EquationBox, PageSignals, Stratum, count_deepseek


def test_count_deepseek_separates_display_and_inline():
    """Выносные ``$$`` не считаются парами строчных ``$``; ``\\frac`` у строчных считается отдельно."""
    text = "по формуле <latex>$$a = \\frac{b}{c}$$</latex>, где <latex>$a$</latex> и <latex>$\\frac{1}{2}$</latex>"
    assert count_deepseek(text) == (1, 2, 1)


def test_stratum_order():
    """Слой: выносные и боксы важнее строчных; без признаков — no_signal."""
    box = (EquationBox((0, 0, 1, 1), 0.9),)
    assert PageSignals("p", (10, 10), box, 1, 0, 0).stratum == Stratum.BOTH
    assert PageSignals("p", (10, 10), (), 2, 5, 1).stratum == Stratum.DEEPSEEK_ONLY
    assert PageSignals("p", (10, 10), box, 0, 3, 1).stratum == Stratum.SURYA_ONLY
    assert PageSignals("p", (10, 10), (), 0, 3, 1).stratum == Stratum.INLINE_FRAC_ONLY
    assert PageSignals("p", (10, 10), (), 0, 3, 0).stratum == Stratum.INLINE_ONLY
    assert PageSignals("p", (10, 10), (), 0, 0, 0).stratum == Stratum.NO_SIGNAL


def test_snap_drops_neighbour_text_touching_border():
    """Пятно на краю грубой рамки (соседняя строка) выбрасывается, рамка ужимается до формулы."""
    gray = np.full((400, 400), 255, np.uint8)
    gray[100:140, 120:300] = 0  # формула
    gray[0:30, 0:400] = 0  # строка прозы выше, её низ задевает грубая рамка
    box = snap_to_ink(gray, (5, 5, 90, 50), scale=4.0)  # грубая рамка: 20..360 × 20..200 px
    assert box == (30.0, 25.0, 75.0, 35.0)


def test_covers_block_or_line_of_block():
    """Бокс накрывает блок, если взял половину блока — или сам наполовину лежит в блоке (строка блока)."""
    block = (0, 0, 100, 40)
    assert covers(block, (0, 0, 100, 25))  # верхняя строка многострочного блока
    assert covers(block, (-5, -5, 105, 45))
    assert not covers(block, (90, 30, 300, 200))


def test_box_quality_signs():
    """Плюс — рамка surya шире эталона, минус — срезала формулу."""
    quality = box_quality((10, 10, 110, 30), (8, 9, 110, 28))
    assert quality["left_mm"] == pytest.approx(2 * MM_PER_UNIT)
    assert quality["bottom_mm"] == pytest.approx(-2 * MM_PER_UNIT)
    assert quality["block_cover"] == pytest.approx(0.9)


def _labels() -> PageLabels:
    """Полоса: пойманный блок, пропущенный блок, строчная формула и ложный бокс."""
    blocks = (
        Block((0, 0, 100, 20), (0, 0, 100, 20), 1),
        Block((0, 100, 100, 120), (0, 100, 100, 120), 1),
        Block((0, 200, 50, 220), (0, 200, 50, 220), 1, inline=True),
    )
    verdicts = {0: (Verdict.FORMULA, ""), 1: (Verdict.NOT_FORMULA, "таблица")}
    return PageLabels("1967/01/X", blocks, verdicts)


EQUATIONS = (EquationBox((-2, -2, 100, 19), 0.99), EquationBox((0, 400, 80, 420), 0.5))


def test_match_page_and_categories():
    """Сопоставление полосы и раскладка случаев по трём папкам."""
    blocks, boxes = match_page("1967/01/X", Stratum.BOTH, _labels(), EQUATIONS)
    assert [b["detected"] for b in blocks] == [True, False, False]
    assert [b["n_blocks"] for b in boxes] == [1, 0]
    cases = page_cases(_labels(), EQUATIONS)
    assert [(c.category, c.name) for c in cases] == [
        (Category.FOUND, "E0"),
        (Category.MISSED, "E1"),
        (Category.FALSE, "S1"),
    ]


def test_estimate_weights_strata():
    """Пересчёт на пак: среднее по полосам слоя × число полос слоя; строчные блоки в полноту не входят."""
    blocks, boxes = match_page("1967/01/X", Stratum.BOTH, _labels(), EQUATIONS)
    sample = pd.DataFrame({"page": ["1967/01/X", "1967/01/Y"], "stratum": ["both", "no_signal"]})
    pages = per_page(sample, pd.DataFrame(blocks), pd.DataFrame(boxes))
    pack = estimate(pages, {"both": 10, "no_signal": 1000})
    assert pack.blocks == 20 and pack.detected == 10
    assert pack.recall == 0.5 and pack.precision == 0.5
