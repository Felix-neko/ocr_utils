"""Эталонная разметка полос окна: ``rel_path,label[,note]`` с метками contents / index / none.

Тот же формат, что у ``curved_lines/labels.py``: путь без расширения (размечаются
оригиналы ``.tif``, а меряться могут заострённые ``.jpg``), строки с ``#`` и пустые
пропускаются. Метка ``none`` — полоса окна, которая НЕ оглавление; она нужна не меньше
положительных: без неё не посчитать ложные срабатывания.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ocr_utils.scan_markup.toc import KIND_CONTENTS, KIND_INDEX

NONE = "none"
LABELS = (KIND_CONTENTS, KIND_INDEX, NONE)


@dataclass(frozen=True)
class Label:
    label: str
    note: str = ""

    @property
    def is_toc(self) -> bool:
        return self.label != NONE


def key(rel_path: str) -> str:
    return str(Path(rel_path).with_suffix("")).replace("\\", "/")


def load_labels(path: Path) -> dict[str, Label]:
    """Метки по ключу «путь без расширения»."""
    labels: dict[str, Label] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue
            if row[0].strip() == "rel_path":
                continue  # заголовок
            if len(row) < 2 or row[1].strip() not in LABELS:
                raise ValueError(f"метки: ожидалось «путь,{'|'.join(LABELS)}[,заметка]», получено {row!r}")
            labels[key(row[0].strip())] = Label(row[1].strip(), row[2].strip() if len(row) > 2 else "")
    return labels


def lookup(labels: dict[str, Label], rel_path: str) -> Label | None:
    return labels.get(key(rel_path))


__all__ = ["LABELS", "NONE", "Label", "key", "load_labels", "lookup"]
