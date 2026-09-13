"""Где на полосе стоит таблица: по словам текстового слоя.

ЗАЧЕМ, ЕСЛИ ЕСТЬ ДЕТЕКТОР ЛИНЕЕК. Затем, что на полосе таблиц бывает две и три, а из DOCX
мы знаем конкретную. Детектор линеек скажет «вот три рамки», а какая из них та самая —
скажут слова: у распознанного PDF есть их координаты, и слова таблицы лежат внутри неё.

ПЕРЕСЧЁТ КООРДИНАТ. Промежуточный PDF собран из скана с полями (``pdf_utils.intermediate_pdfs``
добавляет 12.192 мм по горизонтали и 6.096 мм по вертикали, чтобы распрямителю строк
FineReader было куда двигать), поэтому

    px = (pt / 72 - поле_мм / 25.4) * dpi

Проверено на странице 28 выпуска 1966/01: размер страницы 499.8x807.96 pt это 176.3x285.1 мм,
скан 3588x6440 px при 600 dpi это 151.9x272.6 мм, разница ровно в поля. Рамки слов,
наложенные на скан по этой формуле, садятся на свои слова с точностью до пары пикселей.

ЧТО ЭТО НЕ ДЕЛАЕТ. Это грубый локализатор, а не рамка таблицы: слова не знают ни про
линейки, ни про поля таблицы. Окончательную рамку даёт детектор, а якорь только выбирает,
какую из его находок брать.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from research.legacy.table_processing.geometry import Box
from research.legacy.table_processing.mining.docx_tables import DocxTable

POINTS_PER_INCH = 72.0
MM_PER_INCH = 25.4

# Токены короче не берём: односимвольный мусор совпадает с чем угодно на полосе.
MIN_TOKEN_LEN = 3

# Хвосты, которые отрезаются от облака совпавших слов. Слово таблицы почти всегда
# встречается и в тексте вокруг («тракторы»), и без обрезки якорь растягивается на полосу.
TRIM_PERCENTILE = 8.0

# Меньше этого числа совпавших слов — облако считается шумом.
MIN_WORDS = 4

# Меньше этой доли словаря таблицы внутри рамки — рамка не про эту таблицу.
MIN_COVERAGE = 0.15

TOKEN = re.compile(r"[А-Яа-яЁё]{3,}|\d{2,}")


@dataclass(frozen=True)
class Word:
    box: Box  # уже в пикселях скана
    text: str


def page_words(pdf_path: Path, page_number: int, margin_mm: tuple[float, float, float, float], dpi: int) -> list[Word]:
    """Слова страницы распознанного PDF в пикселях скана."""
    import fitz

    left_mm, _, top_mm, _ = margin_mm
    scale = dpi / POINTS_PER_INCH
    left_px = left_mm / MM_PER_INCH * dpi
    top_px = top_mm / MM_PER_INCH * dpi
    words: list[Word] = []
    with fitz.open(pdf_path) as document:
        page = document[page_number - 1]
        for x0, y0, x1, y1, text, *_ in page.get_text("words"):
            box = Box(
                round(x0 * scale - left_px),
                round(y0 * scale - top_px),
                round(x1 * scale - left_px),
                round(y1 * scale - top_px),
            )
            words.append(Word(box, text))
    return words


def _normalize(text: str) -> set[str]:
    return {token.lower().replace("ё", "е") for token in TOKEN.findall(text)}


def anchor_box(table: DocxTable, words: list[Word]) -> Box | None:
    """Облако слов таблицы на полосе, с обрезанными хвостами."""
    wanted: set[str] = set()
    for text in table.cells():
        wanted |= _normalize(text)
    wanted = {token for token in wanted if len(token) >= MIN_TOKEN_LEN}
    if not wanted:
        return None

    hits = [word.box for word in words if _normalize(word.text) & wanted]
    if len(hits) < MIN_WORDS:
        return None

    xs0 = np.array([box.x0 for box in hits], dtype=float)
    ys0 = np.array([box.y0 for box in hits], dtype=float)
    xs1 = np.array([box.x1 for box in hits], dtype=float)
    ys1 = np.array([box.y1 for box in hits], dtype=float)
    return Box(
        int(np.percentile(xs0, TRIM_PERCENTILE)),
        int(np.percentile(ys0, TRIM_PERCENTILE)),
        int(np.percentile(xs1, 100 - TRIM_PERCENTILE)),
        int(np.percentile(ys1, 100 - TRIM_PERCENTILE)),
    )


def table_tokens(table: DocxTable) -> set[str]:
    """Словарь таблицы: всё, по чему её можно узнать среди слов полосы."""
    wanted: set[str] = set()
    for text in table.cells():
        wanted |= _normalize(text)
    return {token for token in wanted if len(token) >= MIN_TOKEN_LEN}


def coverage(box: Box, words: list[Word], wanted: set[str]) -> float:
    """Какая доля словаря таблицы попала внутрь рамки.

    Именно так, а не «облако совпавших слов»: слова таблицы почти всегда встречаются и в
    тексте вокруг неё («тракторы», «поставки»), и облако растягивается на всю полосу — на
    1966/01 оно давало IoU 0.12-0.52 с настоящей рамкой, то есть выбирало неверную таблицу
    на четырёх страницах из десяти. Доля словаря ВНУТРИ рамки этой болезнью не страдает:
    рамка либо накрывает таблицу, либо нет.
    """
    if not wanted:
        return 0.0
    inside: set[str] = set()
    for word in words:
        center_x = (word.box.x0 + word.box.x1) // 2
        center_y = (word.box.y0 + word.box.y1) // 2
        if box.x0 <= center_x <= box.x1 and box.y0 <= center_y <= box.y1:
            inside |= _normalize(word.text) & wanted
    return len(inside) / len(wanted)


def pick_table(tables: "list", table: DocxTable, words: list[Word]) -> "tuple[object | None, float]":
    """Из находок детектора — та, что накрывает больше всего словаря таблицы."""
    wanted = table_tokens(table)
    best, best_score = None, 0.0
    for candidate in tables:
        score = coverage(candidate.box, words, wanted)
        if score > best_score:
            best, best_score = candidate, score
    return (best, best_score) if best_score >= MIN_COVERAGE else (None, best_score)
