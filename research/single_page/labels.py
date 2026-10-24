"""Разметка стенда: файл ``sets/*.txt`` — строка ``набор/ключ_кадра # single|sliver|no_page|spread|maybe [— заметка]`` (одна страница / страница с полоской соседней / страницы нет / разворот / спорно)."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path


class Label(StrEnum):
    """Сколько страниц в кадре по отсмотру глазами."""

    SINGLE = "single"  # одна страница целиком: обложка журнала, крышка подшивки с бумажной обтяжкой, одиночный лист
    SLIVER = "sliver"  # одна страница и полоска соседней у настоящего корешка (съёмка «по страницам»)
    NO_PAGE = "no_page"  # страницы нет: закрытая крышка подшивки без бумаги (ярлык не в счёт)
    SPREAD = "spread"  # раскрытый разворот (в том числе с узкой страницей: поднятая, вклейка, карточка)
    MAYBE = "maybe"  # спорно


def read_labels(path: Path) -> dict[str, Label]:
    """Разметка из файла: ключ кадра → метка; пустые строки и строки-комментарии (``#`` в начале) пропускаются.

    Args:
        path: Файл разметки.

    Returns:
        Словарь ``набор/ключ → метка``; при повторе ключа берётся последняя строка.

    Raises:
        ValueError: Строка без метки или с неизвестной меткой.
    """
    out: dict[str, Label] = {}
    for n, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "#" not in line:
            raise ValueError(f"{path}:{n}: нет метки после «#»")
        key, rest = (s.strip() for s in line.split("#", 1))
        out[key] = Label(rest.split()[0])
    return out
