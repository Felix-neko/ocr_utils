"""Проверка цитат модели по тексту статьи: нормализация, точный и нечёткий поиск, номер строки в файле выпуска."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from research.article_problems.articles import Article

# Порог нечёткого совпадения (partial_ratio, 0–100): ниже — цитата считается не найденной (выдуманной
# или пересказанной). 90 пропускает мелкие расхождения OCR-разметки и кавычек, но не пересказ.
FOUND_SCORE = 90.0
TAG_RE = re.compile(r"<[^>]{1,40}>")
# Замены символов при нормализации: тире и дефисы → «-», кавычки → «"», ё → е.
CHAR_MAP = str.maketrans(
    {"—": "-", "–": "-", "‑": "-", "«": '"', "»": '"', "„": '"', "“": '"', "”": '"', "ё": "е", "­": ""}
)
ELLIPSIS_RE = re.compile(r"\.\.\.|…")


@dataclass(frozen=True)
class QuoteMatch:
    """Результат поиска цитаты в статье.

    Args:
        found: Совпадение не ниже ``FOUND_SCORE``.
        score: Степень совпадения 0–100 (100 — точное вхождение после нормализации).
        line: Строка файла выпуска, где начинается совпавший фрагмент; ``None`` — не найдено.
    """

    found: bool
    score: float
    line: int | None


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Нормализовать текст для поиска и запомнить, откуда взят каждый символ.

    Нижний регистр, без markdown-выделения (``*``, ``_``) и тегов разметки (``<author>`` и т. п.),
    единые тире и кавычки, ``ё`` → ``е``, любые пробельные последовательности → один пробел.

    Args:
        text: Исходный текст.

    Returns:
        Нормализованный текст и список: i-й символ нормализованного — из позиции ``positions[i]`` исходного.
    """
    # Позиции внутри тегов пропускаются целиком.
    skip = [False] * len(text)
    for tag in TAG_RE.finditer(text):
        for index in range(tag.start(), tag.end()):
            skip[index] = True
    chars: list[str] = []
    positions: list[int] = []
    for index, char in enumerate(text):
        if skip[index] or char in "*_":
            continue
        mapped = char.lower().translate(CHAR_MAP)
        if not mapped:
            continue
        if mapped.isspace():
            # Пробелы схлопываются: второй подряд не добавляется, ведущий тоже.
            if not chars or chars[-1] == " ":
                continue
            mapped = " "
        chars.append(mapped)
        positions.append(index)
    return "".join(chars), positions


def normalize(text: str) -> str:
    """Нормализованный текст без карты позиций (для цитаты).

    Args:
        text: Исходный текст.
    """
    return normalize_with_map(text)[0].strip()


def find_quote(article: Article, quote: str) -> QuoteMatch:
    """Найти цитату модели в тексте статьи.

    Цитата с многоточием внутри проверяется по самому длинному куску между многоточиями. Сначала —
    точное вхождение после нормализации, затем ``partial_ratio`` rapidfuzz (лучшее окно текста).

    Args:
        article: Статья, из которой должна быть цитата.
        quote: Цитата из ответа модели.

    Returns:
        Нашлась ли, насколько похоже и с какой строки файла выпуска начинается фрагмент.
    """
    pieces = [normalize(piece) for piece in ELLIPSIS_RE.split(quote)]
    needle = max(pieces, key=len, default="")
    if len(needle) < 8:
        return QuoteMatch(False, 0.0, None)
    haystack, positions = normalize_with_map(article.text)
    start = haystack.find(needle)
    if start >= 0:
        score = 100.0
    else:
        alignment = fuzz.partial_ratio_alignment(needle, haystack)
        if alignment is None:
            return QuoteMatch(False, 0.0, None)
        score, start = alignment.score, alignment.dest_start
    if score < FOUND_SCORE:
        return QuoteMatch(False, score, None)
    # Номер строки: переводы строк в исходном тексте статьи до начала фрагмента + первая строка статьи.
    offset = positions[min(start, len(positions) - 1)]
    return QuoteMatch(True, score, article.line_start + article.text.count("\n", 0, offset))
