"""Счёт «мешанины»: какие таблицы выгрузки FineReader испортил.

ЗАЧЕМ СЧЁТ, А НЕ ГЛАЗА. Таблиц в паке 780, глазами их не перебрать; а испорченная таблица
выдаёт себя текстом ещё до того, как мы посмотрели на скан. Счёт нужен, чтобы отобрать
кандидатов на ручную проверку, и только для этого: решает всё равно человек и картинка.

КАК ВЫГЛЯДИТ ПОЛОМКА. FineReader читает боковую строку как последовательность отдельных
знаков, повёрнутых на бок, и выдаёт россыпь мусора вперемешку с латиницей:

    «ьц fxo с- o’- ь 8 «2»                    вместо «1966-1970 гг. в % к 1961-1965 гг.»
    «"3 s о W ХС? О Ч X Я О X .дао 2 о « £ X» вместо «Количество в сутко-комплекте»

Отсюда три признака, которые и считаются: короткие токены, смесь кириллицы с латиницей,
и сосредоточенность того и другого В ШАПКЕ — боковой текст в этих журналах живёт именно там.

ЛОЖНЫЙ ДРУГ — МАТРИЦА ЗНАКОВ. Таблица «какие задачи решаются в каком звене» состоит из
плюсов, и по коротким токенам она неотличима от мешанины. Такие ячейки считаются отдельно
и ШТРАФУЮТ счёт, а не поднимают его.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from research.legacy.table_processing.mining.docx_tables import DocxTable

TOKEN = re.compile(r"\S+")
CYRILLIC = re.compile(r"[А-Яа-яЁё]")
LATIN = re.compile(r"[A-Za-z]")
DIGIT = re.compile(r"\d")
LETTER_OR_DIGIT = re.compile(r"[А-Яа-яЁёA-Za-z0-9]")

# Короткие слова, которые в таблице этих журналов законны: предлоги, союзы, единицы,
# сокращения. Без списка каждая вторая нормальная шапка («в % к итогу») считалась бы мусором.
SHORT_WORDS = {
    "в",
    "и",
    "к",
    "с",
    "у",
    "о",
    "а",
    "на",
    "по",
    "до",
    "за",
    "от",
    "из",
    "не",
    "то",
    "их",
    "им",
    "г",
    "гг",
    "т",
    "кг",
    "шт",
    "м",
    "мм",
    "см",
    "км",
    "мз",
    "м3",
    "м2",
    "л",
    "ц",
    "р",
    "руб",
    "чел",
    "шт",
    "п",
    "пп",
    "№",
    "%",
    "тыс",
    "млн",
    "млрд",
    "год",
    "лет",
    "дн",
    "ч",
    "i",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "vii",
    "viii",
    "ix",
    "x",
    "xi",
    "xii",
}

# Знаки, из которых состоят «матрицы»: плюс, минус, тире, точка, галочка.
SYMBOLS = set("+-—–·•*×✓~=")

# Ячейка считается испорченной, если мусорных токенов в ней не меньше этой доли.
CELL_GARBAGE_SHARE = 0.6

# Второе правило для ячейки: средняя длина токена. Замер по паку: у нормальной шапки
# («Наименование техники и запасных частей») она 6.4, у мешанины («о X X я X ш 2 X о. с»)
# — 1.4. Правило нужно потому, что доля мусорных токенов размывается цифрами: в «ьц fxo
# с- o’- ь 8 «2» четыре токена из семи мусорные, то есть 0.57 — чуть ниже порога.
CELL_MEAN_TOKEN_LEN = 2.4
CELL_MEAN_MIN_TOKENS = 4

# ...и токенов в ней не меньше этого. Одиночный токен слишком часто оказывается сноской
# или номером графы, чтобы судить по нему.
CELL_MIN_TOKENS = 2

# Строка идёт в счёт «испорченной строки», если непустых ячеек в ней хотя бы столько.
# Одна испорченная ячейка в строке из одной — это сноска или обрывок, а не шапка.
ROW_MIN_CELLS = 2


@dataclass(frozen=True)
class GarbageScore:
    """Разложенный счёт: в отчёте нужен не только итог, но и из чего он сложился."""

    rank: float
    garbage_share: float  # доля мусорных токенов по всей таблице
    header_share: float  # доля испорченных ячеек в худшей строке (обычно это и есть шапка)
    symbol_share: float  # доля ячеек-знаков (штраф)
    tokens: int
    garbage_cells: int

    def as_row(self) -> dict[str, float | int]:
        return {
            "rank": round(self.rank, 4),
            "garbage_share": round(self.garbage_share, 4),
            "header_share": round(self.header_share, 4),
            "symbol_share": round(self.symbol_share, 4),
            "tokens": self.tokens,
            "garbage_cells": self.garbage_cells,
        }


def _normalized(token: str) -> str:
    return token.strip(".,;:()[]«»\"'`’ ").lower()


def token_is_garbage(token: str) -> bool:
    """Мусорный ли токен.

    Три правила, каждое поймано на реальных ячейках пака:
    * смесь кириллицы с латиницей в одном токене («я04», «51s», «ХС?») — так не пишут;
    * одиночный знак или обрубок в 1-2 знака, не входящий в список законных сокращений;
    * токен вовсе без букв и цифр («о’-», «£», «»).
    """
    core = _normalized(token)
    if not core:
        return True
    if not LETTER_OR_DIGIT.search(core):
        return True
    if CYRILLIC.search(core) and LATIN.search(core):
        return True
    if DIGIT.search(core) and (CYRILLIC.search(core) or LATIN.search(core)) and len(core) <= 4:
        return True
    if len(core) <= 2 and core not in SHORT_WORDS and not core.isdigit():
        return True
    if LATIN.search(core) and not CYRILLIC.search(core) and not DIGIT.search(core) and core not in SHORT_WORDS:
        # Латиница без цифр в советском отраслевом журнале — почти всегда развал боковой
        # строки. Замер по 20 выпускам: латиница есть в 406 ячейках из 6442, и почти все
        # они — мешанина; законные исключения (римские цифры, «25x25 мм») либо в списке
        # коротких слов, либо содержат цифры и сюда не попадают.
        return True
    return False


def cell_is_symbols(text: str) -> bool:
    """Ячейка из одних знаков: плюс, тире, точка. Ложный друг счёта, см. шапку модуля."""
    stripped = "".join(text.split())
    return bool(stripped) and all(character in SYMBOLS for character in stripped)


def cell_is_garbage(text: str) -> bool:
    tokens = TOKEN.findall(text)
    if len(tokens) < CELL_MIN_TOKENS:
        return False
    if cell_is_symbols(text):
        return False
    garbage = sum(1 for token in tokens if token_is_garbage(token))
    if garbage / len(tokens) >= CELL_GARBAGE_SHARE:
        return True
    mean_length = sum(len(_normalized(token)) for token in tokens) / len(tokens)
    return len(tokens) >= CELL_MEAN_MIN_TOKENS and mean_length <= CELL_MEAN_TOKEN_LEN


def score(table: DocxTable) -> GarbageScore:
    """Счёт мешанины для одной таблицы.

    Итог складывается из двух долей и одного штрафа. Вес шапки выше, потому что боковой
    текст живёт в шапке: испорченная шапка при чистом теле — это ровно наш случай, а
    равномерно испорченная таблица чаще означает плохой скан, и её чинить не этим пакетом.
    """
    cells = table.cells()
    tokens = [token for text in cells for token in TOKEN.findall(text)]
    if not tokens:
        return GarbageScore(0.0, 0.0, 0.0, 0.0, 0, 0)

    garbage_tokens = sum(1 for token in tokens if token_is_garbage(token))
    garbage_share = garbage_tokens / len(tokens)

    # Худшая СТРОКА, а не первые две. Так надёжнее: FineReader нередко сливает две соседние
    # таблицы и боковую колонку текста в одну, и шапка второй таблицы оказывается в середине.
    # Пример — 1966/01, таблица 2: мешанина сидит в четвёртой строке из шести, и «шапка =
    # первые две строки» на ней давала 0.00 при трёх испорченных ячейках.
    header_share = 0.0
    for row in table.rows:
        filled_row = [text for text in row if text.strip()]
        if len(filled_row) < ROW_MIN_CELLS:
            continue
        bad = sum(1 for text in filled_row if cell_is_garbage(text))
        header_share = max(header_share, bad / len(filled_row))

    filled = [text for text in cells if text.strip()]
    symbol_share = sum(1 for text in filled if cell_is_symbols(text)) / len(filled) if filled else 0.0

    garbage_cells = sum(1 for text in cells if cell_is_garbage(text))
    rank = 0.6 * header_share + 0.4 * garbage_share - 0.5 * symbol_share
    return GarbageScore(rank, garbage_share, header_share, symbol_share, len(tokens), garbage_cells)
