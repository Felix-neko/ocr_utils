"""Сопоставление полос двух съёмок одного выпуска по тексту: шинглы букв, вложенность множеств, лучшая пара кадров и запас уверенности."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# Длина шингла (подряд идущие буквы без пробелов): 6 букв — редкие сочетания, устойчиво к разрывам слов OCR-ом.
SHINGLE = 6
# Меньше шинглов — полоса почти не прочитана (иллюстрация, расфокус), её сходство не считается.
MIN_SHINGLES = 60
# Буквы, которые остаются при нормализации.
_LETTERS = re.compile(r"[^а-яa-z]+")


def normalize(text: str) -> str:
    """Нормализовать текст полосы для сравнения.

    Args:
        text: Сырой текст OCR.

    Returns:
        Строка из одних строчных букв (кириллица и латиница), ё → е, переносы склеены, пробелы и знаки убраны.
    """
    lowered = text.lower().replace("ё", "е")
    # Перенос в конце строки «прибы-\nли» → «прибыли»; остальные пробелы и знаки всё равно уходят.
    lowered = re.sub(r"-\s*\n\s*", "", lowered)
    return _LETTERS.sub("", lowered)


def shingles(text: str) -> set[str]:
    """Множество буквенных шинглов длины ``SHINGLE`` нормализованного текста.

    Args:
        text: Сырой текст OCR.

    Returns:
        Множество подстрок длины ``SHINGLE`` строки ``normalize(text)``.
    """
    letters = normalize(text)
    return {letters[i : i + SHINGLE] for i in range(len(letters) - SHINGLE + 1)}


def containment(first: set[str], second: set[str]) -> float:
    """Сходство двух полос — доля общих шинглов от меньшего множества.

    Вложенность, а не Жаккар: на крупном плане видна только часть полосы, и её текст целиком входит
    в текст полной полосы другой съёмки.

    Args:
        first, second: Множества шинглов.

    Returns:
        ``|A∩B| / min(|A|, |B|)``; 0, если в любой из полос меньше ``MIN_SHINGLES`` шинглов.
    """
    if len(first) < MIN_SHINGLES or len(second) < MIN_SHINGLES:
        return 0.0
    return len(first & second) / min(len(first), len(second))


def similarity_matrix(base: list[set[str]], donor: list[set[str]]) -> np.ndarray:
    """Матрица сходства всех полос базы со всеми полосами донора.

    Args:
        base: Шинглы полос базы.
        donor: Шинглы полос донора.

    Returns:
        Массив ``float32`` формы (len(base), len(donor)) со значениями ``containment``.
    """
    matrix = np.zeros((len(base), len(donor)), dtype=np.float32)
    for i, first in enumerate(base):
        for j, second in enumerate(donor):
            matrix[i, j] = containment(first, second)
    return matrix


@dataclass(frozen=True)
class PageMatch:
    """Лучшая полоса донора для полосы базы.

    Attributes:
        donor_index: Индекс полосы донора (−1 — нет кандидата).
        similarity: Сходство с ней.
        runner_up: Сходство лучшей полосы донора с другим содержимым (не похожей на найденную).
        confidence: Запас ``similarity − runner_up``, обрезанный в 0..1.
    """

    donor_index: int
    similarity: float
    runner_up: float
    confidence: float


def best_page(row: np.ndarray, donor_self: np.ndarray, same_content: float) -> PageMatch:
    """Выбрать лучшую полосу донора и оценить запас уверенности.

    Второе место ищется только среди полос донора с другим содержимым: дубль того же разворота в доноре
    (пересъёмка) не должен съедать уверенность.

    Args:
        row: Сходство полосы базы со всеми полосами донора.
        donor_self: Матрица сходства полос донора между собой.
        same_content: Порог сходства, выше которого две полосы донора считаются одной и той же полосой.

    Returns:
        ``PageMatch``; ``donor_index = −1``, если сходство со всеми нулевое.
    """
    if row.size == 0 or float(row.max()) <= 0.0:
        return PageMatch(-1, 0.0, 0.0, 0.0)
    best = int(np.argmax(row))
    # Полосы донора, совпадающие по содержимому с лучшей, во второе место не идут.
    other = donor_self[best] < same_content
    other[best] = False
    runner_up = float(row[other].max()) if other.any() else 0.0
    similarity = float(row[best])
    return PageMatch(best, similarity, runner_up, float(np.clip(similarity - runner_up, 0.0, 1.0)))


def sequence_offset(votes: list[tuple[int, int]]) -> tuple[int, int]:
    """Сдвиг нумерации подряд снятых разворотов: номер левой полосы = 2·k + сдвиг.

    Каждый прочитанный номер голосует за сдвиг ``номер_левой − 2·k``; ошибки OCR (обрезанная у сгиба цифра,
    «16» → «6») дают разброс, верный сдвиг — мода.

    Args:
        votes: Пары (порядковый номер кадра k, номер левой полосы, выведенный из прочитанного).

    Returns:
        Кортеж (сдвиг-мода, число голосов за него); (0, 0) без голосов.
    """
    if not votes:
        return 0, 0
    offsets = [number - 2 * index for index, number in votes]
    values, counts = np.unique(offsets, return_counts=True)
    best = int(np.argmax(counts))
    return int(values[best]), int(counts[best])
