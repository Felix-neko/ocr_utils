"""Детектор таблиц и блок-схем на полосе: линейки, вид объекта, рамка без разрезанных букв.

Вход — серая копия полосы в рабочем разрешении (150 dpi, копия 1/4 от 600-dpi скана) и,
если есть, разметка surya layout той же копии. Выход — прямоугольники ``rect_regions``
в пикселях ОРИГИНАЛА с видом ``table`` или ``line_art_schema`` и JSON подробностей.

``TABLE_DETECTOR_VERSION`` — версия АЛГОРИТМА, как ``detection.DETECTOR_VERSION`` у растра
и ``orientation.ORIENTATION_VERSION`` у ориентации, и с тем же назначением: пишется в
``Page.table_detector_version`` и решает при ``--skip-detected``, надо ли перечитывать
полосу. Поднимать при любом изменении, меняющем результат: порог, правило роста, вид.
Версия 1 — детектор v4.2 исследований (``research/legacy/table_processing``), которым
прогнан пак-1 и чьи оверлеи принял человек.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from ocr_utils.scan_markup.db.models import KIND_LINE_ART_SCHEMA, KIND_TABLE
from ocr_utils.scan_markup.table_detection.geometry import KIND_TABLE as KIND_TABLE_RU
from ocr_utils.scan_markup.table_detection.geometry import TableBox
from ocr_utils.scan_markup.table_detection.layout import PageLayout

TABLE_DETECTOR_VERSION = 1


@dataclass(frozen=True)
class Region:
    """Находка в координатах ОРИГИНАЛА и в терминах базы: вид из ``TABLE_KINDS`` и JSON."""

    x1: int
    y1: int
    x2: int
    y2: int
    kind: str
    detector_info: str


def db_kind(finding_kind: str) -> str:
    """Вид находки детектора («таблица», «схема», «рисунок») -> вид региона в базе.

    Схема и рисунок — один вид ``line_art_schema``: лечат их одинаково (блок остаётся на
    месте, повёрнутый текст вписывают в него), а тонкий вид остаётся в ``detector_info``.
    """
    return KIND_TABLE if finding_kind == KIND_TABLE_RU else KIND_LINE_ART_SCHEMA


def to_regions(findings: list[TableBox], scale: float, page_size: tuple[int, int]) -> list[Region]:
    """Находки в пикселях рабочей копии -> регионы в пикселях оригинала (``scale`` — во сколько раз оригинал крупнее).

    ``page_size`` — ``(ширина, высота)`` оригинала: рамка зажимается в кадр, потому что
    округление на границе копии даёт координату на пиксель за краем, а CVAT такой шейп
    отвергает.
    """
    width, height = page_size
    regions: list[Region] = []
    for table in findings:
        box = table.box.scaled(scale).clipped(width, height)
        if box.width < 1 or box.height < 1:
            continue
        info = {
            "kind": table.kind,
            "score": round(float(table.score), 4),
            "skew_deg": round(float(table.skew_deg), 3),
            "rule_box": list(table.rules.scaled(scale).clipped(width, height).as_tuple()),
            "metrics": {key: round(float(value), 4) for key, value in sorted(table.metrics.items())},
        }
        regions.append(
            Region(box.x0, box.y0, box.x1, box.y1, db_kind(table.kind), json.dumps(info, ensure_ascii=False))
        )
    return regions


def detect_regions(
    gray: np.ndarray, dpi: int, layout: "PageLayout | None", scale: float, page_size: tuple[int, int]
) -> list[Region]:
    """Полный ход по одной полосе: детектор на рабочей копии -> регионы оригинала.

    Импорт детектора отложен: он тянет OpenCV-морфологию и не нужен тем, кто берёт отсюда
    только версию и типы.
    """
    from ocr_utils.scan_markup.table_detection.detector import detect

    return to_regions(detect(gray, dpi, layout=layout), scale, page_size)


__all__ = ["TABLE_DETECTOR_VERSION", "Region", "db_kind", "detect_regions", "to_regions"]
