"""Признаки оглавления по словам tesseract и по разметке surya — без самого tesseract."""

from __future__ import annotations

from ocr_utils.page_layout.geometry import Box
from ocr_utils.page_layout.surya.blocks import Block, LayoutBlocks
from ocr_utils.scan_markup.tesseract import Word
from ocr_utils.scan_markup.toc.features import (
    is_contents_word,
    is_index_word,
    keyword_tokens,
    surya_features,
    text_features,
)

WIDTH = 900


def _line(number: int, texts: list[str], right_edge: int | None = None, top: int = 0, conf: float = 90.0) -> list[Word]:
    """Строка из слов шириной 40 px через 10 px; последнее слово можно прижать к ``right_edge``."""
    words = []
    left = 50
    for text in texts:
        words.append(Word(1, 1, number, left, top or number * 20, 40, 14, conf, text))
        left += 50
    if right_edge is not None:
        last = words[-1]
        words[-1] = Word(1, 1, number, right_edge - 30, last.top, 30, 14, conf, last.text)
    return words


def test_toc_lines_count_as_numeric_tails_at_common_edge() -> None:
    words: list[Word] = []
    for number in range(1, 11):
        words += _line(number, ["Иванов", "И.", "Статья", "о", "снабжении", "....", str(number * 5)], right_edge=850)
    words += _line(11, ["Редакционная", "коллегия", "журнала"])
    got = text_features(words, WIDTH)
    assert got["num_tail_lines"] == 10 and got["text_lines"] == 11
    assert got["num_tail_ratio"] == round(10 / 11, 3)
    assert got["leader_words"] == 10


def test_stray_numbers_off_the_column_do_not_count() -> None:
    """Число в конце строки, но не в колонке номеров, — обычный текст («в 1975 г.» без «г.»)."""
    words: list[Word] = []
    for number in range(1, 9):
        words += _line(number, ["Иванов", "Статья", "....", str(number)], right_edge=850)
    words += _line(9, ["план", "выполнен", "на", "105"], right_edge=400)
    got = text_features(words, WIDTH)
    assert got["num_tail_lines"] == 8


def test_body_text_has_no_tails() -> None:
    words: list[Word] = []
    for number in range(1, 20):
        words += _line(number, ["материально-техническое", "снабжение", "предприятий", "и", "строек"])
    got = text_features(words, WIDTH)
    assert got["num_tail_lines"] == 0 and got["num_tail_ratio"] == 0.0 and got["text_lines"] == 19


def test_contents_keyword_survives_letter_spacing_and_one_typo() -> None:
    spaced = _line(1, list("СОДЕРЖАНИЕ"))
    assert [t.text for t in keyword_tokens(spaced)] == ["СОДЕРЖАНИЕ"]
    assert text_features(spaced + _line(2, ["Иванов", "Статья"]), WIDTH)["kw_contents"]
    assert is_contents_word("СОДЕРЖАНИЕ") and is_contents_word("СОДЕРЖАНЕ")
    assert not is_contents_word("СОДЕРЖАТЕЛЬНОСТЬ") and not is_contents_word("СОДЕРЖАНИЯ", upper=False)
    # 1966: рядом с заголовком стоит номер выпуска — строка всё ещё короткая.
    assert text_features(_line(1, ["СОДЕРЖАНИЕ", "6"]) + _line(2, ["Иванов", "Статья"]), WIDTH)["kw_contents"]


def test_contents_word_in_body_text_is_not_a_heading() -> None:
    """«содержание запасов» в тексте и «Содержание» в заголовке статьи — не оглавление."""
    body = _line(1, ["нормативы", "на", "содержание", "запасов", "и", "их", "оборот"])
    assert not text_features(body + _line(2, ["Иванов", "Статья"]), WIDTH)["kw_contents"]
    title = _line(1, ["Содержание", "запасов"])
    assert not text_features(title + _line(2, ["Иванов", "Статья"]), WIDTH)["kw_contents"]
    caps_in_long_line = _line(1, ["СОДЕРЖАНИЕ", "И", "РЕМОНТ", "ПОДЪЁМНЫХ", "МЕХАНИЗМОВ", "НА", "СКЛАДАХ"])
    assert not text_features(caps_in_long_line + _line(2, ["Иванов", "Статья"]), WIDTH)["kw_contents"]


def test_index_heading_requires_capital_and_no_page_number() -> None:
    assert is_index_word("УКАЗАТЕЛЬ") and is_index_word("ПЕРЕЧЕНЬ")
    assert not is_index_word("ПЕРЕМЕННОГО"), "с допуском в одну букву «переменного» становилось «перечнем»"
    assert not is_index_word("ПЕРЕЧЕНЬ", capitalised=False)

    heading = _line(1, ["Указатель", "статей,", "опубликованных", "в", "журнале"])
    assert text_features(heading + _line(2, ["Иванов", "Статья"]), WIDTH)["kw_index"]
    # Контекст может стоять на следующей строке (1975/12: «Указатель статей,» / «опубликованных...»).
    split = _line(1, ["Указатель", "статей,"]) + _line(2, ["опубликованных", "в", "журнале", "в", "1975", "г."])
    assert text_features(split, WIDTH)["kw_index"]
    # Без контекста «Указатель» — просто слово в начале предложения.
    alone = _line(1, ["Указатель", "цен", "утверждён", "в", "прошлом", "году"])
    assert not text_features(alone + _line(2, ["Иванов", "Статья"]), WIDTH)["kw_index"]

    # Строка оглавления, ссылающаяся на указатель, — не заголовок указателя.
    entry = _line(1, ["Указатель", "статей,", "опубликованных", "в", "журнале", "....", "90"], right_edge=850)
    got = text_features(entry + _line(2, ["Иванов", "Статья", "....", "3"], right_edge=850), WIDTH)
    assert not got["kw_index"] and got["num_tail_lines"] == 2

    lower = _line(1, ["не", "указанной", "в", "данном", "перечне"])
    assert not text_features(lower, WIDTH)["kw_index"]


def test_imprint_keyword() -> None:
    imprint = _line(1, ["РЕДКОЛЛЕГИЯ"]) + _line(2, ["Главный", "редактор"])
    assert text_features(imprint, WIDTH)["kw_imprint"]
    assert text_features(_line(1, ["Редколлегия", "журнала"]) + _line(2, ["а", "б"]), WIDTH)["kw_imprint"]
    assert not text_features(_line(1, ["решением", "редколлегии", "журнала"]) + _line(2, ["а", "б"]), WIDTH)[
        "kw_imprint"
    ]


def test_low_confidence_words_do_not_make_keywords() -> None:
    noisy = _line(1, ["СОДЕРЖАНИЕ"], conf=10.0)
    assert not text_features(noisy + _line(2, ["Иванов", "Статья"]), WIDTH)["kw_contents"]


def test_surya_features_sum_toc_area_and_table_area() -> None:
    layout = LayoutBlocks(
        (
            Block("TableOfContents", 0.75, Box(0, 0, 100, 100)),
            Block("TableOfContents", 0.9, Box(0, 100, 100, 200)),
            Block("TableOfContents", 0.1, Box(0, 200, 100, 300)),  # ниже порога уверенности
            Block("Table", 0.95, Box(0, 300, 100, 400)),
            Block("Text", 1.0, Box(0, 400, 100, 500)),
        ),
        100,
        1000,
    )
    got = surya_features(layout)
    assert got == {"surya_toc_conf": 0.9, "surya_toc_area": 0.2, "surya_table_area": 0.1}
    assert surya_features(None) == {"surya_toc_conf": 0.0, "surya_toc_area": 0.0, "surya_table_area": 0.0}
