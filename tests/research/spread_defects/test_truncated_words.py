"""Тесты словарной проверки крайних слов у сгиба: обрубок ловится, перенос склеивается, пропуск строки не склеивает."""

from research.spread_defects.truncated_words import adjacency, check_lines, clean, is_known


def test_dictionary_basics():
    """Словарь знает полные слова и не знает обрубки."""
    assert is_known("удалось") and is_known("производства")
    assert not is_known("величить")


def test_clean_keeps_hyphen_and_drops_numbers():
    """Перенос сохраняется, цифры и короткие токены отбрасываются."""
    assert clean("произ-") == "произ-"
    assert clean("«плана»,") == "плана"
    assert clean("1979") is None and clean("в") is None


def test_right_page_hyphen_tail_is_not_truncated():
    """Хвост переноса в начале строки правой полосы склеивается с концом предыдущей строки и не считается обрубком."""
    texts = [["рост", "объема", "произ-"], ["водства", "в", "стране"], ["выросло", "почти", "вдвое"]]
    check = check_lines(texts, "R")
    assert check.fold_unknown == 0 and check.fold_checked == 3


def test_right_page_cut_start_is_unknown():
    """Начала строк, ушедшие под переплёт, — незнакомые словарю обрубки.

    Часть обрубков совпадает со словарными словами («словных» — форма слова «словный»), поэтому меряется доля.
    """
    texts = [["величить", "выпуск", "продукции"], ["словных", "банок", "консервов"], ["ределяются", "цены", "рынка"]]
    check = check_lines(texts, "R")
    assert check.fold_unknown >= 2


def test_hyphen_not_joined_across_missing_line():
    """Между строками с пропущенной строкой перенос не склеивается."""
    cys = [100.0, 150.0, 300.0]
    assert adjacency(cys) == [True, False]
