"""Кто искорёжил таблицу: типография, скан или коррекция геометрии FineReader.

ТРИ ИСТОЧНИКА ОДНОЙ И ТОЙ ЖЕ ТАБЛИЦЫ. Скан (то, что есть у нас), страница распознанного PDF
с коррекцией геометрии и она же без коррекции. Третий вариант собран не для всех выпусков:
112 из 123, но у всех 112 есть и DOCX.

ЧТО ЗАМЕРЕНО ЗАРАНЕЕ И ЧТО ИЗ ЭТОГО СЛЕДУЕТ. Кривизна у таблиц, которые FineReader увидел,
и у тех, что пропустил, различается слабо: медиана сагитты 0.18 против 0.21 мм, p90 0.33
против 0.60. То есть геометрия — не главная причина пропусков, а лишь одна из; главные
различия структурные (у пропущенных вдвое меньше заполненных ячеек: 0.75 против 1.00).
Поэтому вердикт «геометрия ни при чём» здесь полноправный, а не отговорка.

СОПОСТАВЛЕНИЕ ПО ДОЛЯМ СТРАНИЦЫ. Страницы трёх источников разного размера (скан без полей,
PDF с полями, да ещё и распрямление меняет содержимое), поэтому таблица ищется на каждой
странице своим детектором, а сопоставляется по НОРМИРОВАННОЙ рамке — доле ширины и высоты
страницы. Это грубее, чем пересчёт координат, но не зависит от того, что FineReader сделал
со страницей, а именно это и проверяется.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from research.legacy.table_processing.detection import ruling, ruling_v4
from research.legacy.table_processing.geometry import Box, iou
from research.legacy.table_processing.warping.metrics import EMPTY, Geometry, measure

logger = logging.getLogger(__name__)

# Разрешение, в котором сравниваются все три источника.
WORK_DPI = 300

# Сагитта, начиная с которой таблица считается кривой: 0.5 мм. Замер по паку: у таблиц,
# которые FineReader увидел, p90 сагитты 0.33 мм, у пропущенных — 0.60. Порог посередине
# отделяет хвост распределения, а не его тело.
CURVED_MM = 0.5

# Насколько должны совпасть нормированные рамки, чтобы считать, что это одна таблица.
MIN_IOU = 0.3

VERDICT_SCAN = "крива уже на скане"
VERDICT_FIXED = "FineReader выпрямил"
VERDICT_BROKEN = "FineReader искорёжил"
VERDICT_FLAT = "геометрия ни при чём"
VERDICT_UNKNOWN = "не с чем сравнить"


@dataclass
class Diagnosis:
    """Одна таблица в трёх источниках."""

    issue: str
    page_number: int
    scan_rel_path: str
    verdict: str
    scan: Geometry = EMPTY
    corrected: Geometry = EMPTY
    uncorrected: Geometry = EMPTY
    note: str = ""
    boxes: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)

    def as_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "issue": self.issue,
            "page_number": self.page_number,
            "scan_rel_path": self.scan_rel_path,
            "verdict": self.verdict,
            "note": self.note,
        }
        for name, geometry in (("scan", self.scan), ("corr", self.corrected), ("nocorr", self.uncorrected)):
            for key, value in geometry.as_row().items():
                row[f"{name}_{key}"] = value
        return row


def render_page(pdf_path: Path, page_number: int, dpi: int = WORK_DPI) -> "np.ndarray | None":
    """Страница PDF в оттенках серого."""
    import fitz

    try:
        with fitz.open(pdf_path) as document:
            if not 1 <= page_number <= document.page_count:
                return None
            pixmap = document[page_number - 1].get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
            return np.frombuffer(pixmap.samples, np.uint8).reshape(pixmap.height, pixmap.width).copy()
    except Exception as error:
        logger.warning("Не отрисовалась страница %s:%d — %s", pdf_path.name, page_number, error)
        return None


def _normalized(box: Box, shape: tuple[int, int]) -> Box:
    """Рамка в тысячных долях страницы — общая валюта для источников разного размера."""
    height, width = shape
    return Box(
        round(box.x0 * 1000 / max(1, width)),
        round(box.y0 * 1000 / max(1, height)),
        round(box.x1 * 1000 / max(1, width)),
        round(box.y1 * 1000 / max(1, height)),
    )


def pick_same_table(page: np.ndarray, wanted: Box, dpi: int = WORK_DPI) -> "Box | None":
    """Та же таблица на другой странице: по совпадению нормированных рамок."""
    found = [table for table in ruling_v4.detect(page, dpi) if table.is_table]
    best, best_score = None, 0.0
    for table in found:
        score = iou(_normalized(table.box, page.shape[:2]), wanted)
        if score > best_score:
            best, best_score = table.box, score
    return best if best_score >= MIN_IOU else None


def diagnose(
    scan_gray: np.ndarray,
    scan_box: Box,
    corrected: "np.ndarray | None",
    uncorrected: "np.ndarray | None",
    dpi: int = WORK_DPI,
) -> tuple[str, Geometry, Geometry, Geometry, str, dict[str, tuple[int, int, int, int]]]:
    """Вердикт и три набора мер."""
    boxes = {"scan": scan_box.as_tuple()}
    scan_geometry = measure(scan_gray[scan_box.clipped(scan_gray.shape[1], scan_gray.shape[0]).slice], dpi)
    wanted = _normalized(scan_box, scan_gray.shape[:2])

    def side(page: "np.ndarray | None", name: str) -> Geometry:
        if page is None:
            return EMPTY
        box = pick_same_table(page, wanted, dpi)
        if box is None:
            return EMPTY
        boxes[name] = box.as_tuple()
        return measure(page[box.clipped(page.shape[1], page.shape[0]).slice], dpi)

    corrected_geometry = side(corrected, "corr")
    uncorrected_geometry = side(uncorrected, "nocorr")

    if not scan_geometry.measurable:
        return (
            VERDICT_UNKNOWN,
            scan_geometry,
            corrected_geometry,
            uncorrected_geometry,
            "на скане линеек не нашлось",
            boxes,
        )
    if not corrected_geometry.measurable:
        note = "в распознанном PDF та же таблица не нашлась — FineReader её не сохранил как таблицу"
        verdict = VERDICT_SCAN if scan_geometry.sagitta_max_mm >= CURVED_MM else VERDICT_FLAT
        return verdict, scan_geometry, corrected_geometry, uncorrected_geometry, note, boxes

    scan_curved = scan_geometry.sagitta_max_mm >= CURVED_MM
    corrected_curved = corrected_geometry.sagitta_max_mm >= CURVED_MM
    if scan_curved and corrected_curved:
        verdict = VERDICT_SCAN
    elif scan_curved and not corrected_curved:
        verdict = VERDICT_FIXED
    elif not scan_curved and corrected_curved:
        verdict = VERDICT_BROKEN
    else:
        verdict = VERDICT_FLAT
    return verdict, scan_geometry, corrected_geometry, uncorrected_geometry, "", boxes
