"""Что именно съел корешок в конце строки.

СПОСОБ. Строка обрывается на полуслове, но вторая половина слова стоит в начале
следующей строки. Значит, слово вычисляется, а не угадывается: ищется такая вставка X
длиной 0…3 буквы, что «видимый хвост + X + начало следующей строки» — слово выпуска.

ПОЧЕМУ ТОЛЬКО ОДНОЗНАЧНЫЕ СЛУЧАИ. Восстановление дописывает буквы в архивный скан.
Ошибка здесь — не опечатка, а подделка источника, и стоит она дороже, чем пропуск.
Поэтому строка чинится, только если:

* видимый хвост САМ ПО СЕБЕ не слово (иначе строка могла кончиться и без переноса,
  и дорисованный дефис был бы выдумкой);
* вставка X определяется единственным образом.

Во всех остальных случаях строка остаётся как есть и попадает в отчёт как непочиненная.
"""

import re
from dataclasses import dataclass

from ocr_utils.gutter_loss_restoration.lexicon import normalize
from ocr_utils.gutter_loss_restoration.pageocr import Line

# Буквы, которыми пробуем закрыть разрыв.
ALPHABET = "абвгдежзийклмнопрстуфхцчшщъыьэюя"

# Максимальная длина вставки. Больше трёх букв корешок съедает редко, а перебор
# растёт как степень алфавита.
MAX_INSERT = 3


def letters_only(token: str) -> str:
    """Оставляет от слова только буквы и внутренние дефисы.

    Распознаватель охотно вешает на обрезанное слово лишнюю точку или запятую
    («производ.»), а иногда читает уже видимый дефис переноса. Набирать этот мусор
    заново нельзя, поэтому слово чистится до букв.

    Args:
        token: Слово как его прочитал распознаватель.

    Returns:
        Слово без краевых знаков препинания.
    """
    return token.strip().strip("«»\"'()[]{}.,;:!?—–-… ")


@dataclass(frozen=True)
class Tail:
    """Решение по хвосту одной строки.

    Attributes:
        index: Номер строки в полосе.
        word: Слово целиком, как его надо набрать (с дефисом переноса, если он нужен).
        visible: Видимая часть слова, как её прочитал распознаватель.
        x_start: Левый край последнего слова в координатах кадра.
        added: Что дописано сверх видимого.
        reason: Почему решено так; для непочиненных — почему не вышло.
        ok: Чинится ли строка.
    """

    index: int
    word: str = ""
    visible: str = ""
    x_start: int = 0
    added: str = ""
    reason: str = ""
    ok: bool = False


def _candidates(prefix: str, suffix: str, lexicon: set[str]) -> list[str]:
    """Вставки X, замыкающие «prefix + X + suffix» в слово словаря.

    Возвращаются вставки НАИМЕНЬШЕЙ длины, при которой хоть что-то нашлось: корешок
    съедает знак-другой, и предпочитать более длинную догадку не за что.
    """
    for length in range(0, MAX_INSERT + 1):
        level = [combo for combo in _combos(length) if normalize(prefix + combo + suffix) in lexicon]
        if level:
            return level
    return []


def _combos(length: int):
    """Все сочетания букв заданной длины."""
    if length == 0:
        yield ""
        return
    if length == 1:
        yield from ALPHABET
        return
    for first in ALPHABET:
        for rest in _combos(length - 1):
            yield first + rest


def resolve_line(index: int, line: Line, nxt: Line | None, lexicon: set[str], hyphen_room: float = 0.0) -> Tail:
    """Решает, что дописать в конце строки.

    Args:
        index: Номер строки в полосе.
        line: Строка, у которой срезан хвост.
        nxt: Следующая строка полосы либо None.
        lexicon: Словарь выпуска.
        hyphen_room: Сколько пикселей осталось между концом слова и сгибом, в долях
            шага строк. Если дефис туда помещался, а его нет — значит, его и не было,
            и строка кончилась без переноса.

    Returns:
        ``Tail`` с решением.
    """
    if not line.words:
        return Tail(index, reason="строка без слов")
    last = line.words[-1]
    visible_raw = letters_only(last.text)
    if last.text.strip().endswith("-"):
        return Tail(index, visible=visible_raw, x_start=last.x0, reason="дефис переноса уже виден")
    visible = normalize(visible_raw)
    if not visible:
        return Tail(index, reason="хвост не слово")
    if visible in lexicon:
        return Tail(index, visible=visible_raw, x_start=last.x0, reason="хвост сам по себе слово — не трогаем")
    if nxt is None or not nxt.words:
        return Tail(index, visible=visible_raw, x_start=last.x0, reason="нет следующей строки")
    head = normalize(nxt.words[0].text)
    if not head:
        return Tail(index, visible=visible_raw, x_start=last.x0, reason="начало следующей строки не слово")

    found = _candidates(visible, head, lexicon)
    if not found:
        return Tail(index, visible=visible_raw, x_start=last.x0, reason="слово не собралось по словарю")
    if len(found) > 1:
        return Tail(
            index, visible=visible_raw, x_start=last.x0, reason=f"вставка неоднозначна ({len(found)} вариантов)"
        )
    insert = found[0]
    if not insert and hyphen_room > 0.35:
        return Tail(index, visible=visible_raw, x_start=last.x0, reason="дефису хватало места — переноса не было")
    return Tail(
        index=index,
        word=visible_raw + insert + "-",
        visible=visible_raw,
        x_start=last.x0,
        added=insert + "-",
        reason="перенос собран по словарю",
        ok=True,
    )


def resolve_side(lines: list[Line], lexicon: set[str]) -> list[Tail]:
    """Решает хвосты всех строк полосы.

    Args:
        lines: Строки полосы сверху вниз.
        lexicon: Словарь выпуска.

    Returns:
        Список решений по строкам.
    """
    return [
        resolve_line(i, line, lines[i + 1] if i + 1 < len(lines) else None, lexicon) for i, line in enumerate(lines)
    ]
