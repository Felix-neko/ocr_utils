"""Общие типы стенда: рамка, таблица, ячейка, сетка.

ЖИВОЙ КОД ПЕРЕЕХАЛ в ``ocr_utils.scan_markup.table_detection.geometry``: здесь остался реэкспорт, чтобы стенд исследований
(сравнение версий, добыча, отчёты) мерил тот же детектор, что стоит в конвейере.
"""

from ocr_utils.scan_markup.table_detection.geometry import (  # noqa: F401 — реэкспорт для стенда
    Box,
    union,
    intersection,
    iou,
    KIND_TABLE,
    KIND_DIAGRAM,
    KIND_DRAWING,
    KINDS,
    TableBox,
    Edge,
    Cell,
    Grid,
    match_boxes,
    boxes_from_rows,
)

__all__ = [
    "Box",
    "union",
    "intersection",
    "iou",
    "KIND_TABLE",
    "KIND_DIAGRAM",
    "KIND_DRAWING",
    "KINDS",
    "TableBox",
    "Edge",
    "Cell",
    "Grid",
    "match_boxes",
    "boxes_from_rows",
]
