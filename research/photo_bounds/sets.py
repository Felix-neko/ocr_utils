"""Разметка набора глазами (``sets/*.csv`` стенда): класс каждого эталонного фото (чёткая граница, нечёткая, с рамкой-линейкой, вплотную к тексту, цветное) и пометки о грубом эталоне."""

from __future__ import annotations

import csv
from enum import StrEnum
from pathlib import Path

SETS_DIR = Path(__file__).parent / "sets"
# Файл разметки классов фото: ``key,set_class,ref_note,note``.
CLASSES_CSV = SETS_DIR / "photo_classes.csv"


class SetClass(StrEnum):
    """Класс фото по разметке глазами."""

    SHARP = "sharp"  # чёткая граница «бумага → фото» по всем сторонам
    SOFT = "soft"  # нечёткая: виньетка, растворение в фон, светлый край без рамки
    RULED = "ruled"  # фото в рамке-линейке (тонкая черта вокруг, между ней и снимком бумага или нет)
    TEXT_ADJACENT = "text_adjacent"  # вплотную к тексту (обтекание, подпись ближе 1–2 мм)
    COLOR = "color"  # цветное
    NOT_PHOTO = "not_photo"  # в эталоне растр, а на деле штрих, плашка, рисунок


class RefNote(StrEnum):
    """Пометка о качестве эталонной рамки CVAT."""

    OK = "ok"  # рамка по краю снимка (до 1 мм)
    LOOSE = "loose"  # рамка шире снимка (захвачены поля, подпись)
    TIGHT = "tight"  # рамка срезает снимок
    MERGED = "merged"  # одна рамка на несколько снимков


def read_classes(path: Path = CLASSES_CSV) -> dict[str, dict]:
    """Разметка классов: ключ фото → строка CSV.

    Args:
        path: CSV разметки.

    Returns:
        Словарь; пустой, если файла нет.
    """
    if not path.exists():
        return {}
    with path.open() as fh:
        return {row["key"]: row for row in csv.DictReader(fh)}


def attach_sets(rows: list[dict], path: Path = CLASSES_CSV) -> list[dict]:
    """Добавляет к строкам мер колонки ``set_class`` и ``ref_note`` из разметки глазами.

    Args:
        rows: Строки мер (ключ — ``key``).
        path: CSV разметки.

    Returns:
        Новые строки с колонками разметки (пусто — фото не размечено).
    """
    classes = read_classes(path)
    out = []
    for row in rows:
        mark = classes.get(row["key"], {})
        out.append({**row, "set_class": mark.get("set_class", ""), "ref_note": mark.get("ref_note", "")})
    return out
