"""Сравнение распознавателей: расстояние Левенштейна на символах.

CER (character error rate) — отношение числа правок к длине эталона. Считается своей
функцией, а не библиотечной, по той же причине, по которой в проекте свои метрики вообще:
одна формула на десять строк не стоит зависимости, а её поведение на пустом эталоне
(частый случай в таблице) должно быть определено здесь и видно глазами.

СРАВНЕНИЕ НОРМАЛИЗОВАННОЕ. Перенос по дефису склеен, регистр приведён, «ё» сведена к «е»,
пробелы схлопнуты. Иначе замер мерил бы, у кого удачнее склеились строки, а не кто лучше
прочитал буквы.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return WHITESPACE.sub(" ", text.replace("ё", "е").replace("Ё", "Е")).strip().lower()


def edit_distance(first: str, second: str) -> int:
    """Расстояние Левенштейна; память — одна строка таблицы."""
    if first == second:
        return 0
    if not first:
        return len(second)
    if not second:
        return len(first)
    previous = list(range(len(second) + 1))
    for i, left in enumerate(first, start=1):
        current = [i]
        for j, right in enumerate(second, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Доля ошибок на символ. Пустой эталон: 0.0 при пустой гипотезе, иначе 1.0."""
    reference_text = normalize(reference)
    hypothesis_text = normalize(hypothesis)
    if not reference_text:
        return 0.0 if not hypothesis_text else 1.0
    return edit_distance(reference_text, hypothesis_text) / len(reference_text)


@dataclass
class EngineScore:
    """Счёт одного распознавателя по размеченным ячейкам."""

    name: str
    errors: list[float] = field(default_factory=list)
    seconds: float = 0.0
    exact: int = 0

    def add(self, reference: str, hypothesis: str, seconds: float) -> None:
        self.errors.append(cer(reference, hypothesis))
        self.exact += int(normalize(reference) == normalize(hypothesis))
        self.seconds += seconds

    @property
    def mean_cer(self) -> float:
        return sum(self.errors) / len(self.errors) if self.errors else 0.0

    @property
    def median_cer(self) -> float:
        if not self.errors:
            return 0.0
        ordered = sorted(self.errors)
        middle = len(ordered) // 2
        return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2

    def as_row(self) -> list[object]:
        count = max(1, len(self.errors))
        return [
            self.name,
            len(self.errors),
            f"{self.mean_cer:.3f}",
            f"{self.median_cer:.3f}",
            f"{self.exact}/{len(self.errors)}",
            f"{self.seconds / count:.2f}",
        ]


ENGINE_HEADER = ("движок", "ячеек", "CER средний", "CER медианный", "точных", "с/ячейка")
