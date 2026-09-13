"""Счёт мешанины и привязка таблицы к странице."""

from __future__ import annotations

from research.legacy.table_processing.mining.docx_tables import DocxTable
from research.legacy.table_processing.mining.garbage import cell_is_garbage, cell_is_symbols, score, token_is_garbage
from research.legacy.table_processing.mining.page_match import PageText, match, normalize


def _table(rows, index: int = 0, context: "list[str] | None" = None) -> DocxTable:
    return DocxTable(issue="1966_01", index=index, rows=rows, context_before=context or [])


def test_mangled_header_scores_above_a_clean_one():
    mangled = _table(
        [["Наименование техники", "Поставки, тыс. шт.", "ьц fxo с- o’- ь 8 «2"], ["Тракторы", "1790", "164"]]
    )
    clean = _table(
        [["Группы материально-технических средств", "в млн. руб.", "в % к итогу"], ["Автомобили", "277", "3,5"]]
    )
    assert score(mangled).rank > score(clean).rank


def test_symbol_matrix_is_not_mangled():
    matrix = _table(
        [["Наименование задач", "центральное", "низовое"], ["Сбор информации", "+", "+"], ["Расчёт", "+", "—"]]
    )
    assert score(matrix).rank < 0.2
    assert cell_is_symbols("+")
    assert cell_is_symbols("—")
    assert not cell_is_symbols("164")


def test_token_rules():
    assert token_is_garbage("ьц")
    assert token_is_garbage("fxo"), "латиница без цифр в русском журнале — развал боковой строки"
    assert token_is_garbage("я04")
    assert not token_is_garbage("тракторы")
    assert not token_is_garbage("1790")
    assert not token_is_garbage("гг")
    assert not token_is_garbage("в")


def test_cell_with_short_tokens_is_garbage():
    assert cell_is_garbage("о X X я X ш 2 X о. с")
    assert not cell_is_garbage("норма на изделие")


def test_page_match_prefers_the_page_with_the_words():
    body = "Вывозка древесины 203 фанеры 731 картона 945 плиты 1611 целлюлоза 269 бумага 812 спирт 150"
    table = _table(
        [
            ["Вывозка древесины", "203", "1611"],
            ["фанеры картона плиты", "731", "945"],
            ["целлюлоза бумага спирт", "269", "812"],
            ["итого", "150", "442"],
        ]
    )
    pages = [
        PageText(1, frozenset(normalize("совсем другой текст про снабжение колхозов и совхозов"))),
        PageText(2, frozenset(normalize(body))),
    ]
    found = match(table, pages)
    assert found.page_number == 2
    assert found.source == "cells", "по собственным словам таблицы, раз их достаточно"
    assert not found.ambiguous


def test_page_match_falls_back_to_context():
    table = _table([["о X X", "я X ш"], ["2 X о.", "с"]], context=["Оборотная сторона формы номер один"])
    pages = [
        PageText(1, frozenset(normalize("что-то постороннее"))),
        PageText(2, frozenset(normalize("Оборотная сторона формы номер один, движение материала"))),
    ]
    found = match(table, pages)
    assert found.page_number == 2
    assert found.source == "context"
