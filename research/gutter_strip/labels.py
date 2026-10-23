"""Разметка стенда: файл ``sets/*.txt`` — строка ``ключ_кадра # strip|none|maybe [— заметка]`` (есть полоса-вклейка у корешка / нет / спорно)."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path


class Label(StrEnum):
    """Есть ли на развороте полоса-вклейка у корешка."""

    STRIP = "strip"
    NONE = "none"
    MAYBE = "maybe"


def read_labels(path: Path) -> dict[str, Label]:
    """Разметка из файла: ключ кадра → метка; пустые строки и строки-комментарии (``#`` в начале) пропускаются.

    Args:
        path: Файл разметки.

    Returns:
        Словарь; при повторе ключа берётся последняя строка.

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
