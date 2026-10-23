"""Разметка сторон стенда: файлы ``sets/*.txt`` — строки ``<прогон> <ключ кадра> <L|R> <метка>``, ``#`` — комментарий."""

from __future__ import annotations

from enum import Enum
from pathlib import Path


class Label(Enum):
    """Метка стороны."""

    RAGGED = "ragged"  # рваный край
    SMOOTH = "smooth"  # ровный
    UNSURE = "unsure"  # не понять


def read_labels(path: Path) -> dict[tuple[str, str, str], Label]:
    """Разметка из файла.

    Args:
        path: Файл ``sets/*.txt``.

    Returns:
        ``(прогон, ключ, страница)`` → метка; при повторе — последняя строка.
    """
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        run, key, page, label = line.split()
        out[(run, key, page)] = Label(label)
    return out
