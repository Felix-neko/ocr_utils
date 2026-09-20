"""Оценка номера страницы у полос без напечатанного номера — по номерам соседних полос выпуска.

Первые полосы статей, обложки, реклама, вклейки с иллюстрациями номера не несут (у МТС — каждая
пятая полоса), а сверка заголовков с оглавлением на сборке (``reconcile``) опирается именно на
номер: статья по «Содержанию» начинается на странице N, и `#` на полосе с номером N — настоящий,
остальные — фантомы. Номер выводится из соседей: у полосы с порядковым номером ``i`` в выпуске и
напечатанным номером ``n`` смещение ``n − i`` постоянно на всём участке сплошной нумерации; сбивают
его вклейка без номеров (после неё смещение падает на число вклеенных листов) и разворот,
снятый одной картинкой (смещение растёт на 1), а ещё модель изредка читает номер неверно
(«44» вместо 41). Поэтому каждая сторона (назад и вперёд) считается надёжной, только если её
ближайшие нумерованные полосы согласны между собой, а обе стороны при расхождении разрешаются в
пользу более близкой.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

# Сколько ближайших нумерованных полос смотреть с каждой стороны.
NEIGHBORS = 3
# Сколько из них должны дать одно и то же смещение, чтобы стороне верить.
AGREEING = 2

# Первое целое число в напечатанном номере: «94», «1а», «2 МТС № 5» (подпись тетради, которую модель
# приняла за номер — ловится проверкой согласия сторон). Пять и больше цифр — не номер страницы.
_NUMBER = re.compile(r"(?<!\d)(\d{1,4})(?!\d)")


class NumberSource(StrEnum):
    """Откуда номер полосы в sidecar: напечатан и согласуется с соседями, выведен из соседей, неизвестен."""

    PRINTED = "printed"
    SUGGESTED = "suggested"
    NONE = "none"


@dataclass(frozen=True)
class PageNumberGuess:
    """Номер одной полосы после оценки.

    Args:
        printed: Напечатанный номер по ответу модели (разобранный); ``None`` — не напечатан или не разобрался.
        suggested: Номер по соседям; ``None`` — вывести не удалось (нет согласной стороны).
        source: Чем считать номер полосы: ``PRINTED`` — напечатанным (сходится с соседями или проверить
            нечем), ``SUGGESTED`` — выведенным (не напечатан или напечатанный подозрителен), ``NONE`` — никаким.
        suspect: Напечатанный номер расходится с обеими согласными сторонами — опечатка модели.
    """

    printed: int | None
    suggested: int | None
    source: NumberSource
    suspect: bool = False

    @property
    def number(self) -> int | None:
        """Номер, которым пользоваться: напечатанный или выведенный по ``source``; ``None`` — нет."""
        if self.source is NumberSource.PRINTED:
            return self.printed
        if self.source is NumberSource.SUGGESTED:
            return self.suggested
        return None


def parse_printed(text: str | None) -> int | None:
    """Напечатанный номер из строки ответа модели: первое целое до четырёх цифр.

    Args:
        text: Поле ``page_number`` ответа (``"94"``, ``"1а"``, ``None``).

    Returns:
        Число или ``None``, если цифр нет.
    """
    if not text:
        return None
    match = _NUMBER.search(str(text))
    return int(match.group(1)) if match else None


def _side_offset(offsets: list[int]) -> int | None:
    """Смещение стороны по её ближайшим нумерованным полосам: значение, на котором сошлись
    хотя бы ``AGREEING`` из них; ``None`` — сторона ненадёжна (мало полос или разнобой).

    Args:
        offsets: Смещения ``номер − индекс`` ближайших полос стороны, от ближней к дальней.
    """
    if len(offsets) < AGREEING:
        return None
    value, votes = Counter(offsets).most_common(1)[0]
    return value if votes >= AGREEING else None


def _neighbors(numbered: list[tuple[int, int]], position: int, forward: bool) -> list[int]:
    """Смещения до ``NEIGHBORS`` ближайших нумерованных полос по одну сторону от полосы.

    Args:
        numbered: ``[(индекс полосы, смещение)]`` всех нумерованных полос по порядку.
        position: Индекс полосы, вокруг которой смотрим (сама она не берётся).
        forward: ``True`` — полосы после неё, ``False`` — до неё.
    """
    if forward:
        picked = [offset for index, offset in numbered if index > position]
        return picked[:NEIGHBORS]
    picked = [offset for index, offset in numbered if index < position]
    return picked[::-1][:NEIGHBORS]


def _distance(numbered: list[tuple[int, int]], position: int, forward: bool) -> int:
    """Расстояние до ближайшей нумерованной полосы по одну сторону; без таких — бесконечность.

    Args:
        numbered: ``[(индекс полосы, смещение)]``.
        position: Индекс полосы.
        forward: Сторона.
    """
    if forward:
        return next((index - position for index, _ in numbered if index > position), 10**9)
    return next((position - index for index, _ in reversed(numbered) if index < position), 10**9)


def suggest_page_numbers(printed: list[str | None]) -> list[PageNumberGuess]:
    """Номера всех полос выпуска: напечатанные проверены по соседям, отсутствующие выведены.

    Для каждой полосы обе стороны дают смещение (:func:`_side_offset`) по своим ближайшим
    нумерованным полосам (сама полоса не в счёт). Полоса без номера: стороны согласны между собой
    или надёжна одна — её смещение; надёжны обе, но разные (между ними вклейка или разворот) —
    ближняя по расстоянию, при равном расстоянии номер не выводится. Полоса с номером: если обе
    стороны надёжны, согласны между собой и расходятся с напечатанным — номер подозрителен
    (опечатка модели), пользоваться выведенным; иначе — напечатанным.

    Args:
        printed: Поле ``page_number`` каждой полосы выпуска по порядку имён файлов.

    Returns:
        По оценке на полосу, в том же порядке.
    """
    parsed = [parse_printed(text) for text in printed]
    numbered = [(index, number - index) for index, number in enumerate(parsed) if number is not None]
    guesses: list[PageNumberGuess] = []
    for index, number in enumerate(parsed):
        back = _side_offset(_neighbors(numbered, index, forward=False))
        ahead = _side_offset(_neighbors(numbered, index, forward=True))
        if back is not None and ahead is not None:
            if back == ahead:
                offset = back
            else:
                # Между сторонами вклейка или разворот: верим той, что ближе; равные — не знаем.
                back_distance, ahead_distance = _distance(numbered, index, False), _distance(numbered, index, True)
                offset = back if back_distance < ahead_distance else ahead if ahead_distance < back_distance else None
        else:
            offset = back if back is not None else ahead
        suggested = index + offset if offset is not None else None
        if number is None:
            source = NumberSource.SUGGESTED if suggested is not None else NumberSource.NONE
            guesses.append(PageNumberGuess(None, suggested, source))
            continue
        # Напечатанный номер подозрителен только при двух надёжных и согласных сторонах против него.
        suspect = back is not None and ahead is not None and back == ahead and number != suggested
        source = NumberSource.SUGGESTED if suspect else NumberSource.PRINTED
        guesses.append(PageNumberGuess(number, suggested, source, suspect))
    return guesses
