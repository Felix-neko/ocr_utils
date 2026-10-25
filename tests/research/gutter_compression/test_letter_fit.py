"""Тесты подгонки модели сжатия по ширине букв (``research.gutter_compression.letter_fit``)."""

import numpy as np
import pandas as pd

from research.gutter_compression.letter_fit import LetterData, fit_letters, letter_prediction, letter_summary
from research.gutter_compression.shape import HeightShape


def synthetic_letters(amplitude: float, length_mm: float, origin_mm: float, noise: float, seed: int = 0) -> LetterData:
    """Буквы 40 строк по 1.2 мм шагом от края строки; ширина на холсте — по модели, плюс шум разных букв."""
    rng = np.random.default_rng(seed)
    centres, halves, ys, lines = [], [], [], []
    for line in range(40):
        # Края строк чередуются: 0 и 0.6 мм дальше начала зоны текста (висячие знаки и буквы).
        edge = origin_mm + (0.6 if line % 2 else 0.0)
        centre = edge + 0.6 + 1.2 * np.arange(25)
        centres.append(centre)
        halves.append(np.full(len(centre), 0.5))
        ys.append(np.full(len(centre), line / 39))
        lines.append(np.full(len(centre), line))
    data = LetterData(np.concatenate(centres), np.concatenate(halves), np.concatenate(ys), np.ones(1000),
                      np.concatenate(lines))  # fmt: skip
    clean = letter_prediction(length_mm, np.full(len(data.r), amplitude), data, origin_mm)
    r = clean * np.exp(rng.normal(0.0, noise, len(clean)))
    return LetterData(data.centre, data.half, data.y, r, data.line)


def test_prediction_is_one_without_compression():
    """Без амплитуды ожидаемое отношение — 1 у любой буквы."""
    data = synthetic_letters(0.0, 5.0, 2.0, 0.0)
    assert np.allclose(letter_prediction(5.0, np.zeros(len(data.r)), data, 2.0), 1.0)


def test_prediction_is_harmonic_mean_over_letter():
    """Ожидаемое отношение буквы у края — между сжатием на её ближнем и дальнем концах."""
    data = LetterData(np.array([2.5]), np.array([0.5]), np.array([0.5]), np.array([1.0]), np.array([0]))
    r = letter_prediction(1.5, np.array([0.4]), data, 2.0)[0]
    c_near, c_far = 1 - 0.4, 1 - 0.4 * np.exp(-1.0 / 1.5)
    assert c_near < r < c_far


def test_fit_recovers_short_strong_decay():
    """Короткий сильный спад (0.4 на 1.5 мм) восстанавливается по буквам при шуме букв 0.12."""
    data = synthetic_letters(0.4, 1.5, 2.0, 0.12)
    fit = fit_letters(data, HeightShape.CONST, 2.0)
    assert abs(fit.coef[0] - 0.4) < 0.08
    assert fit.length_mm < 2.5


def test_fit_separates_weak_long_from_strong_short():
    """Слабый длинный спад (0.1 на 6 мм) не принимается за сильный короткий."""
    data = synthetic_letters(0.1, 6.0, 2.0, 0.12, seed=1)
    fit = fit_letters(data, HeightShape.CONST, 2.0)
    assert fit.coef[0] < 0.2
    assert fit.length_mm > 3.0


def test_letter_summary_takes_nearest_letter_per_line():
    """Последняя буква строки — ближайшая к краю; «у края» — ближе 3 мм."""
    letters = pd.DataFrame({"line": [0, 0, 0, 1, 1], "from_edge_mm": [0.5, 2.0, 10.0, 0.4, 8.0],
                            "ratio": [0.8, 0.9, 1.0, 0.7, 1.0]})  # fmt: skip
    summ = letter_summary(letters)
    assert summ["last_letter"] == 0.75
    assert summ["near_letter"] == 0.8
