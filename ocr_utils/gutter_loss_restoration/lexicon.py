"""Словарь выпуска, собранный из распознанных полос.

ЗАЧЕМ СВОЙ СЛОВАРЬ. Чтобы понять, какое слово разорвано переносом, нужен ответ на
вопрос «слово ли это». Русского словаря в системе нет, а если бы и был — общий словарь
плохо знает лексику отраслевого журнала 1926 года («хозрасчёт», «Госплан», «НКПС»,
«Эмбанефть»). Зато сам выпуск — идеальный корпус: сто пятьдесят разворотов одной темы,
и почти всякое слово встречается в нём не один раз.

ЧТО В СЛОВАРЬ НЕ КЛАДЁТСЯ. Первое и последнее слово каждой строки. Последнее срезано
корешком, первое — вторая половина переноса; и то и другое не слова, а обрывки, и
попади они в словарь, любой обрывок стал бы «подтверждаться» сам собой.
"""

import json
import re
from collections import Counter
from pathlib import Path

from ocr_utils.gutter_loss_restoration.pageocr import Line

# Слово словаря: кириллица, при желании с дефисом внутри, не короче двух букв.
WORD_RE = re.compile(r"^[а-яёА-ЯЁ]+(?:-[а-яёА-ЯЁ]+)*$")

# Сколько раз слово должно встретиться, чтобы ему верить. Единичное вхождение слишком
# часто оказывается опечаткой распознавания.
MIN_COUNT = 2


def normalize(token: str) -> str:
    """Приводит слово к словарному виду: без знаков препинания, в нижнем регистре.

    Args:
        token: Слово как его прочитал распознаватель.

    Returns:
        Нормализованное слово либо пустая строка, если это не слово.
    """
    token = token.strip().strip("«»\"'()[]{}.,;:!?—–-…").replace("ё", "е")
    return token.lower() if WORD_RE.match(token) else ""


def collect(halves: dict[str, list[Line]], counter: Counter) -> None:
    """Добавляет слова полос в счётчик, пропуская краевые слова строк.

    Args:
        halves: Распознанные полосы кадра.
        counter: Счётчик, который пополняется на месте.
    """
    for lines in halves.values():
        for line in lines:
            for word in line.words[1:-1]:
                normalized = normalize(word.text)
                if normalized:
                    counter[normalized] += 1


def build(cache_dir: Path, loader) -> Counter:
    """Собирает счётчик слов по всему кэшу распознавания.

    Args:
        cache_dir: Папка кэша.
        loader: Функция чтения одного файла кэша (путь -> полосы).

    Returns:
        Счётчик слов.
    """
    counter: Counter = Counter()
    for path in sorted(cache_dir.glob("*.json")):
        halves = loader(path)
        if halves:
            collect(halves, counter)
    return counter


def save(path: Path, counter: Counter) -> None:
    """Пишет словарь на диск.

    Args:
        path: Куда писать.
        counter: Счётчик слов.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {word: count for word, count in counter.items() if count >= MIN_COUNT}
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def load(path: Path) -> set[str]:
    """Читает словарь с диска.

    Args:
        path: Путь к JSON словаря.

    Returns:
        Множество слов.
    """
    return set(json.loads(path.read_text(encoding="utf-8")))
