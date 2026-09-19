"""Приём чтения зоны: пороги уверенности и правило чисел без ведущих нулей."""

from ocr_utils.text_layer_fix.ocr import acceptable, has_leading_zero


def test_leading_zero_detection() -> None:
    assert has_leading_zero("00098")
    assert has_leading_zero("0099 мм")
    # Группы тысяч через пробел и одиночный ноль — не ведущие нули.
    assert not has_leading_zero("24 000")
    assert not has_leading_zero("0")
    assert not has_leading_zero("1 000 000")


def test_numbers_without_letters_by_confidence() -> None:
    # Ниже 0.8 число не принимается вовсе.
    assert not acceptable("12 000", 0.79)[0]
    # В поясе 0.8–0.9 — только без ведущих нулей.
    assert acceptable("24 000", 0.85)[0]
    assert not acceptable("00098", 0.85)[0]
    # От 0.9 — как раньше, любое число.
    assert acceptable("0099", 0.95)[0]


def test_words_need_letters_and_confidence() -> None:
    assert acceptable("Поставщики", 0.7)[0]
    assert not acceptable("Поставщики", 0.5)[0]


def test_surya_hallucination_filter() -> None:
    from ocr_utils.text_layer_fix.second_opinion import hallucination_reason

    assert hallucination_reason("and special to") == "латиница без кириллицы"
    assert hallucination_reason("THE PERSON NAMED IN")
    assert hallucination_reason("35 35 35 35 35 35 31") == "повторяющиеся токены"
    assert hallucination_reason("Million (1997) (1997) (1997) (1997) (1997)")
    # Настоящий текст, числа и короткие обозначения проходят.
    assert hallucination_reason("Наименование материалов") is None
    assert hallucination_reason("24 000") is None
    assert hallucination_reason("кг") is None
    assert hallucination_reason("ГОСТ 12") is None
