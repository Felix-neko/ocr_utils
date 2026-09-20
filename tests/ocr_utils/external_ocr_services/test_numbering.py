"""Оценка номеров страниц по соседям: дыры, вклейка, разворот одной картинкой, опечатка, края выпуска."""

from __future__ import annotations

from ocr_utils.external_ocr_services.numbering import NumberSource, parse_printed, suggest_page_numbers


def _numbers(printed: list[str | None]) -> list[int | None]:
    return [guess.number for guess in suggest_page_numbers(printed)]


def test_parse_printed_takes_first_short_integer():
    assert parse_printed("94") == 94 and parse_printed("1а") == 1 and parse_printed("2 МТС № 5") == 2
    assert parse_printed(None) is None and parse_printed("") is None and parse_printed("нет") is None
    assert parse_printed("12345") is None, "пять цифр — не номер страницы"


def test_gaps_inside_continuous_numbering_are_filled():
    printed = [None, "2", "3", None, None, "6", "7", None]
    guesses = suggest_page_numbers(printed)
    assert [g.number for g in guesses] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert guesses[0].source is NumberSource.SUGGESTED and guesses[1].source is NumberSource.PRINTED
    assert not any(g.suspect for g in guesses)


def test_insert_without_numbers_shifts_offset_and_nearer_side_wins():
    # Страницы 10–13, затем вклейка из двух листов без номеров, затем 14–17: у листов вклейки стороны
    # расходятся на 2; ближняя сторона решает, на равном расстоянии — номера нет.
    printed = ["10", "11", "12", "13", None, None, None, "14", "15", "16", "17"]
    numbers = _numbers(printed)
    assert numbers[:4] == [10, 11, 12, 13] and numbers[7:] == [14, 15, 16, 17]
    assert numbers[4] == 14, "ближе к передней стороне — её смещение"
    assert numbers[6] == 13, "ближе к задней стороне — её смещение"
    assert numbers[5] is None, "равноудалённая — не выводится"


def test_spread_shot_as_one_page_does_not_flag_neighbors():
    # Полоса с индексом 3 — разворот со схемой (страницы 4–5 одной картинкой, номер не читается):
    # дальше смещение растёт на 1, но напечатанные номера остаются доверенными.
    printed = ["1", "2", "3", None, "6", "7", "8"]
    guesses = suggest_page_numbers(printed)
    assert [g.number for g in guesses[:3]] == [1, 2, 3] and [g.number for g in guesses[4:]] == [6, 7, 8]
    assert not any(g.suspect for g in guesses)
    assert guesses[3].number is None, "разворот равноудалён от обеих сторон — номер не выводится"


def test_misread_number_is_suspect_and_replaced():
    printed = ["40", "41", "44", "43", "44", "45"]
    guesses = suggest_page_numbers(printed)
    assert guesses[2].suspect and guesses[2].source is NumberSource.SUGGESTED
    assert guesses[2].printed == 44 and guesses[2].number == 42
    assert [g.number for g in guesses] == [40, 41, 42, 43, 44, 45]
    assert guesses[4].source is NumberSource.PRINTED, "настоящая 44 не подозрительна"


def test_edges_use_the_only_available_side():
    printed = [None, None, "3", "4", "5", None, None]
    assert _numbers(printed) == [1, 2, 3, 4, 5, 6, 7]


def test_too_few_numbered_pages_give_nothing():
    guesses = suggest_page_numbers([None, "2", None])
    assert guesses[0].number is None and guesses[0].source is NumberSource.NONE
    assert guesses[1].number == 2 and guesses[1].source is NumberSource.PRINTED
    assert suggest_page_numbers([]) == []
