"""Тесты сопоставления полос по тексту и нумерации разворотов стенда spread_ocr (синтетика)."""

import numpy as np

from research.spread_ocr.cli import infer_spread_numbers, valid_number, Side
from research.spread_ocr.match import best_page, containment, normalize, sequence_offset, shingles

TEXT_A = (
    "Брокеры хотят участвовать в прибы-\nли — это нормальное явление. Но мало желающих участвовать в убытках фирмы " * 3
)
TEXT_B = (
    "Второй период деятельности биржи как товарной начинается с февраля тысяча девятьсот двадцать второго года " * 3
)


def test_normalize_glues_hyphenation():
    """Перенос в конце строки склеивается, знаки и регистр уходят."""
    assert normalize("Прибы-\nли, Ёж!") == "прибылиеж"


def test_containment_partial_page():
    """Часть полосы целиком входит в полную полосу: вложенность 1, чужой текст — около 0."""
    full = shingles(TEXT_A)
    part = shingles(TEXT_A[:150])
    assert containment(part, full) == 1.0
    assert containment(shingles(TEXT_B), full) < 0.1


def test_best_page_ignores_duplicate_runner_up():
    """Пересъёмка той же полосы в доноре не съедает уверенность."""
    donor = [shingles(TEXT_A), shingles(TEXT_A), shingles(TEXT_B)]
    row = np.array([0.9, 0.85, 0.1], dtype=np.float32)
    donor_self = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.float32)
    found = best_page(row, donor_self, 0.5)
    assert found.donor_index == 0
    assert abs(found.confidence - 0.8) < 1e-6
    assert len(donor) == 3


def test_sequence_offset_mode():
    """Мода сдвига устойчива к обрезанным у сгиба номерам."""
    votes = [(5, 8), (6, 10), (7, 12), (8, 4)]  # последний — «14» прочитан как «4»
    assert sequence_offset(votes) == (-2, 3)


def test_valid_number_parity():
    """Левая — чётная, правая — нечётная, в пределах выпуска."""
    assert valid_number(16, Side.LEFT) == 16
    assert valid_number(17, Side.LEFT) is None
    assert valid_number(17, Side.RIGHT) == 17
    assert valid_number(700, Side.RIGHT) is None


def test_infer_spread_numbers_prefers_right():
    """Несогласованная пара: левая из обрезанного у сгиба номера правится по правой."""
    records = [
        {"set": "a", "frame": "f", "side": "left", "page_number": 6, "page_number_source": "band"},
        {"set": "a", "frame": "f", "side": "right", "page_number": 17, "page_number_source": "page"},
        {"set": "a", "frame": "g", "side": "left", "page_number": None, "page_number_source": ""},
        {"set": "a", "frame": "g", "side": "right", "page_number": 19, "page_number_source": "page"},
    ]
    result = {(r["frame"], r["side"]): r for r in infer_spread_numbers(records)}
    assert result[("f", "left")]["page_number"] == 16
    assert result[("g", "left")]["page_number"] == 18
    assert result[("g", "left")]["page_number_source"] == "spread"
