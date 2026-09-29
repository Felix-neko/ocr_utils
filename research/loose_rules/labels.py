"""Разметка сирот глазами: классы (enum) и чтение ``sets/labels.csv`` (``rule_id,label,note``)."""

from __future__ import annotations

import csv
from enum import Enum
from pathlib import Path

# Разметка стенда лежит в репо рядом с кодом: она маленькая и нужна для воспроизведения порогов.
LABELS_PATH = Path(__file__).parent / "sets" / "labels.csv"


class RuleLabel(str, Enum):
    """Что на самом деле нашёл детектор под видом линейки-сироты."""

    # Настоящая линейка: отбивка, линейка сноски, межколонник, плашка врезки, линия бланка, подчёркивание.
    RULE = "rule"
    # Стволы букв нескольких строк, сшитые через межстрочье («Н» над «П» заголовка).
    GLYPH_STACK = "glyph_stack"
    # Низы, верхи или перекладины букв одной строки, сшитые через просветы между буквами.
    GLYPH_ROW = "glyph_row"
    # Скобка или черта дроби формулы.
    FORMULA = "formula"
    # Тень кромки листа или корешка, край кадра.
    EDGE_SHADOW = "edge_shadow"
    # Карандашная или чернильная пометка читателя.
    HANDWRITING = "handwriting"
    # Прочее ложное (рамка растра, край рисунка и т. п.).
    OTHER = "other"

    @property
    def is_false(self) -> bool:
        """Ложная ли это линейка: всё, кроме настоящей."""
        return self is not RuleLabel.RULE


def read_labels(path: Path = LABELS_PATH) -> dict[str, RuleLabel]:
    """Разметка: ``rule_id`` → класс.

    Args:
        path: CSV с колонками ``rule_id``, ``label`` (значение :class:`RuleLabel`), ``note``.

    Returns:
        Словарь; нет файла — пустой.
    """
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {row["rule_id"]: RuleLabel(row["label"]) for row in csv.DictReader(handle)}
