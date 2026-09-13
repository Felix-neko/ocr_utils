"""Какой странице PDF соответствует таблица DOCX.

ПОЧЕМУ ЭТО ОТДЕЛЬНАЯ ЗАДАЧА. Казалось бы, у DOCX есть страницы. Но FineReader не пишет
``lastRenderedPageBreak``, а счётчик по разрывам страниц и несплошным секциям расходится
с числом страниц PDF на ±1 у большинства выпусков (проверено на всех 98). Ошибка в одну
страницу означает вырезанную не ту таблицу — и заметить это можно только глазами.

КАК СЧИТАЕМ. У распознанного PDF есть текстовый слой, и он получен из ТОГО ЖЕ прогона
FineReader, что и DOCX. Значит, слова таблицы буквально присутствуют на своей странице —
включая мешанину, если она читается одинаково. Берём редкие токены таблицы (длинные слова
и числа) и смотрим, на какой странице их больше.

ЕСЛИ ТАБЛИЦА — СПЛОШНАЯ МЕШАНИНА, редких токенов в ней нет. Тогда в ход идут соседние
абзацы: они распознаны нормально и стоят на той же странице. Замер на 1966/01: по ячейкам
находятся 9 таблиц из 10 с отрывом от второго места в 3-5 раз, десятая — по абзацам.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from research.legacy.table_processing.mining.docx_tables import DocxTable

# Токены короче не берём: «года» и «всего» есть на каждой странице и ничего не различают.
MIN_WORD_LEN = 5
MIN_DIGITS = 3

# Сколько токенов достаточно, чтобы не звать на помощь соседние абзацы.
ENOUGH_TOKENS = 8

# Отрыв от второго места, ниже которого совпадение считается сомнительным и помечается
# в отчёте. Не отбрасывается: сомнительное совпадение всё равно лучше, чем счётчик DOCX.
MIN_MARGIN = 0.15

WORD = re.compile(r"[А-Яа-яЁё]+")
NUMBER = re.compile(r"\d+")


@dataclass(frozen=True)
class PageText:
    page_number: int  # с единицы
    tokens: frozenset[str]


@dataclass(frozen=True)
class PageMatch:
    page_number: int
    score: float
    margin: float
    source: str  # "cells" | "context" — по чему нашли
    ambiguous: bool

    @property
    def found(self) -> bool:
        return self.page_number > 0


def normalize(text: str) -> set[str]:
    """Редкие токены строки: длинные слова (ё→е) и числа не короче трёх цифр."""
    tokens = {word.lower().replace("ё", "е") for word in WORD.findall(text) if len(word) >= MIN_WORD_LEN}
    tokens |= {number for number in NUMBER.findall(text) if len(number) >= MIN_DIGITS}
    return tokens


def read_pages(pdf_path: Path) -> list[PageText]:
    """Текстовый слой распознанного PDF, по странице на элемент."""
    import fitz

    pages: list[PageText] = []
    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document):
            pages.append(PageText(index + 1, frozenset(normalize(page.get_text()))))
    return pages


def _score(query: set[str], pages: list[PageText], hint: int) -> tuple[int, float, float]:
    """Лучшая страница, её доля найденных токенов и отрыв от второго места."""
    if not query:
        return 0, 0.0, 0.0
    scored = []
    for page in pages:
        hits = len(query & page.tokens) / len(query)
        # Подсказка счётчика DOCX работает мизерной добавкой: она разводит равные ответы
        # (шапка таблицы, повторённая на развороте) и не может перебить настоящее совпадение.
        near = 0.001 if hint and abs(page.page_number - hint) <= 3 else 0.0
        scored.append((hits + near, hits, page.page_number))
    scored.sort(reverse=True)
    best_total, best_hits, best_page = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    return best_page, best_hits, best_total - second


def match(table: DocxTable, pages: list[PageText]) -> PageMatch:
    """Страница таблицы: сперва по её ячейкам, при бедных ячейках — по соседним абзацам."""
    hint = table.docx_page_hint
    cell_tokens = normalize(" ".join(table.cells()))
    if len(cell_tokens) >= ENOUGH_TOKENS:
        page, score, margin = _score(cell_tokens, pages, hint)
        if score > 0:
            return PageMatch(page, score, margin, "cells", margin < MIN_MARGIN)

    context_tokens = normalize(" ".join(table.context_before + table.context_after))
    both = cell_tokens | context_tokens
    page, score, margin = _score(both, pages, hint)
    if score <= 0:
        return PageMatch(0, 0.0, 0.0, "none", True)
    return PageMatch(page, score, margin, "context", margin < MIN_MARGIN)
