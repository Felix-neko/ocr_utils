"""Зоны повёрнутого текста на растре страницы PDF: ячейки таблиц, подписи на line art, отдельные.

Таблицы разбираются тем же ходом, что и в ``ocr_utils.rotated_text.tables``: детектор на
копии 150 dpi → вырезка 600 dpi по рамке → выравнивание по линейкам → сетка на копии
300 dpi → склейка разрезанных ячеек → ориентация ячейки (ось по форме глифов, сторона по
буквам tesseract, приор по таблице). Разница одна: растр берётся из самого PDF, а не из
заострённой копии скана, потому что FineReader поправил геометрию страницы и координаты
базы разметки на неё не ложатся.

Вне таблиц боковой текст ищется по соседям глифов (``docstrum``); сторону называет tesseract
(``side_of``). Все рамки — в пикселях родного растра страницы (600 dpi).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np

from ocr_utils.rotated_text.tables.ocr import LANGUAGES
from ocr_utils.rotated_text.tables.orientation import (
    MIN_LETTERS,
    CellOrientation,
    evidence_at,
    is_sideways_table,
    orient_table,
)
from ocr_utils.rotated_text.tables.source import CROP_PAD_MM, deskew_by_rules
from ocr_utils.rotated_text.tables.structure import analyse_structure, interior_box, merge_split_cells, work_copy
from ocr_utils.scan_markup.rotation import rotate_cw as rotate_image
from ocr_utils.scan_markup.table_detection.detector import detect
from ocr_utils.page_layout.geometry import Box, TableBox

from ocr_utils.text_layer_fix import WORK_DPI, mm_to_px
from ocr_utils.text_layer_fix.docstrum import cluster_lines, cluster_rotated, glyph_components
from ocr_utils.text_layer_fix.raster import downscale

logger = logging.getLogger(__name__)

# Рабочее разрешение сетки и ориентации ячеек — как в rotated_text (порог сетки калиброван на нём).
GRID_DPI = 300

# Поле вокруг вырезки бокового текста при чтении стороны, мм: захватить выносные, не соседей.
SIDE_PAD_MM = 0.8


class ZoneKind(StrEnum):
    """Откуда зона: ячейка таблицы, смешанная ячейка, подпись на line art, отдельный текст."""

    TABLE_CELL = "table_cell"
    TABLE_CELL_MIXED = "table_cell_mixed"
    TABLE_CELL_UPRIGHT = "table_cell_upright"  # прямая ячейка с краской, но без единого слова в слое
    LINE_ART_UPRIGHT = "line_art_upright"  # прямая подпись на картинке/схеме, под которой слоя нет
    STANDALONE_UPRIGHT = "standalone_upright"  # прямой текст вне таблиц и схем, под которым слоя нет
    LINE_ART_LABEL = "line_art_label"
    STANDALONE = "standalone"


@dataclass
class RotatedZone:
    """Зона повёрнутого текста в пикселях растра страницы."""

    box: Box
    kind: ZoneKind
    # На сколько повернуть по часовой, чтобы текст стал прямым; None — ось лежит, сторона не ясна.
    rotate_cw: "int | None"
    confidence: float = 0.0
    table_index: "int | None" = None
    cell_key: "tuple[int, int] | None" = None
    letters: dict[int, int] = field(default_factory=dict)
    note: str = ""

    def to_json(self) -> dict:
        return {
            "box": self.box.as_tuple(),
            "kind": str(self.kind),
            "rotate_cw": self.rotate_cw,
            "confidence": round(self.confidence, 3),
            "table_index": self.table_index,
            "cell_key": list(self.cell_key) if self.cell_key else None,
            "letters": {str(k): v for k, v in self.letters.items()},
            "note": self.note,
        }

    @staticmethod
    def from_json(payload: dict) -> "RotatedZone":
        return RotatedZone(
            Box(*payload["box"]),
            ZoneKind(payload["kind"]),
            payload.get("rotate_cw"),
            float(payload.get("confidence", 0.0)),
            payload.get("table_index"),
            tuple(payload["cell_key"]) if payload.get("cell_key") else None,
            {int(k): v for k, v in payload.get("letters", {}).items()},
            payload.get("note", ""),
        )


@dataclass
class CellInfo:
    """Ячейка таблицы на растре страницы: рамка, внутренность, поворот текста."""

    key: tuple[int, int]
    box: Box
    inner: Box
    rotate_cw: "int | None"
    axis_sideways: "bool | None"
    confidence: float
    is_header: bool
    note: str = ""


@dataclass
class TableInfo:
    """Таблица (или схема/рисунок по мнению детектора) на растре страницы."""

    index: int
    box: Box
    kind: str
    deskew_deg: float = 0.0
    cells: list[CellInfo] = field(default_factory=list)
    sideways_table: bool = False
    error: str = ""

    def to_json(self) -> dict:
        return {
            "index": self.index,
            "box": self.box.as_tuple(),
            "kind": self.kind,
            "deskew_deg": round(self.deskew_deg, 3),
            "sideways_table": self.sideways_table,
            "error": self.error,
            "cells": [
                {
                    "key": list(c.key),
                    "box": c.box.as_tuple(),
                    "inner": c.inner.as_tuple(),
                    "rotate_cw": c.rotate_cw,
                    "axis_sideways": c.axis_sideways,
                    "confidence": round(c.confidence, 3),
                    "is_header": c.is_header,
                    "note": c.note,
                }
                for c in self.cells
            ],
        }


def _undo_deskew(box: Box, angle_cw: float, shape: tuple[int, int]) -> Box:
    """Рамка из выровненной вырезки → в исходную (невыровненную) вырезку: обратный поворот углов."""
    if abs(angle_cw) < 1e-6:
        return box
    height, width = shape[:2]
    forward = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), -angle_cw, 1.0)
    inverse = cv2.invertAffineTransform(forward)
    corners = np.array([[box.x0, box.y0], [box.x1, box.y0], [box.x0, box.y1], [box.x1, box.y1]], float)
    moved = corners @ inverse[:, :2].T + inverse[:, 2]
    return Box(
        int(np.floor(moved[:, 0].min())),
        int(np.floor(moved[:, 1].min())),
        int(np.ceil(moved[:, 0].max())),
        int(np.ceil(moved[:, 1].max())),
    )


def detect_tables(gray600: np.ndarray, dpi: int = 600) -> list[TableBox]:
    """Таблицы, схемы и рисунки детектора на копии 150 dpi; рамки — в пикселях 600 dpi.

    Args:
        gray600: Растр страницы.
        dpi: Его разрешение.

    Returns:
        Находки детектора, пересчитанные в пиксели поданного растра.
    """
    factor = dpi / WORK_DPI
    small = downscale(gray600, factor)
    found = detect(small, WORK_DPI)
    height, width = gray600.shape[:2]
    return [
        TableBox(
            box=t.box.scaled(factor).clipped(width, height),
            score=t.score,
            source=t.source,
            origin=t.origin,
            skew_deg=t.skew_deg,
            metrics=t.metrics,
            rule_box=t.rule_box.scaled(factor).clipped(width, height) if t.rule_box is not None else None,
            kind=t.kind,
        )
        for t in found
    ]


def analyse_table(
    gray600: np.ndarray,
    index: int,
    table: TableBox,
    dpi: int = 600,
    allowed: tuple[int, ...] = (0, 90, 180, 270),
    lang: str = LANGUAGES,
) -> tuple[TableInfo, list[RotatedZone]]:
    """Сетка и ориентация ячеек одной таблицы на растре страницы.

    Args:
        gray600: Растр страницы.
        index: Порядковый номер таблицы на странице.
        table: Находка детектора в пикселях растра.
        dpi: Разрешение растра.
        allowed: Допустимые повороты текста ячеек.
        lang: Языки tesseract для определения стороны.

    Returns:
        Описание таблицы со всеми ячейками и зоны её боковых ячеек.
    """
    info = TableInfo(index, table.box, table.kind)
    height, width = gray600.shape[:2]
    crop_box = table.box.padded(mm_to_px(CROP_PAD_MM, dpi)).clipped(width, height)
    crop = np.ascontiguousarray(gray600[crop_box.slice])
    if crop.size == 0:
        info.error = "пустая вырезка"
        return info, []
    deskewed, angle = deskew_by_rules(crop, dpi)
    info.deskew_deg = angle
    work = work_copy(deskewed, dpi, GRID_DPI)
    structure = analyse_structure(work, GRID_DPI)
    grid = structure.grid
    if not grid.cells:
        info.error = "сетка ячеек не построена"
        return info, []
    grid = merge_split_cells(grid, structure.lines, work, GRID_DPI)
    verdicts: dict[tuple[int, int], CellOrientation] = orient_table(work, grid, GRID_DPI, allowed, lang)
    info.sideways_table = is_sideways_table(verdicts)
    factor = dpi / GRID_DPI
    zones: list[RotatedZone] = []
    for cell in grid.cells:
        verdict = verdicts[cell.key]
        inner_work = interior_box(cell, GRID_DPI)
        box_page = (
            _undo_deskew(cell.box.scaled(factor), angle, deskewed.shape)
            .shifted(crop_box.x0, crop_box.y0)
            .clipped(width, height)
        )
        inner_page = (
            _undo_deskew(inner_work.scaled(factor), angle, deskewed.shape)
            .shifted(crop_box.x0, crop_box.y0)
            .clipped(width, height)
        )
        info.cells.append(
            CellInfo(
                cell.key,
                box_page,
                inner_page,
                verdict.rotate_cw,
                verdict.axis_sideways,
                verdict.confidence,
                cell.is_header,
                verdict.note,
            )
        )
        mixed = "смешанная" in verdict.note
        if mixed:
            # Смешанная ячейка (сетка склеила боковую шапку с прямой): боковая часть ищется по
            # соседям глифов внутри ячейки и становится своей зоной; остальное — как было.
            for sub in _mixed_subzones(gray600, inner_page, dpi, lang):
                sub.table_index, sub.cell_key = index, cell.key
                zones.append(sub)
        if verdict.rotate_cw is None and verdict.axis_sideways and not mixed:
            # Ось лежит уверенно, но букв нет и приора по таблице не нашлось (боковые числа):
            # зона есть, сторону назовёт чтение.
            zones.append(
                RotatedZone(
                    inner_page,
                    ZoneKind.TABLE_CELL,
                    None,
                    0.0,
                    index,
                    cell.key,
                    dict(verdict.letters),
                    verdict.note or "ось лежит, сторона не ясна",
                )
            )
        elif verdict.rotate_cw not in (None, 0) or (mixed and verdict.axis_sideways):
            zones.append(
                RotatedZone(
                    inner_page,
                    ZoneKind.TABLE_CELL_MIXED if mixed else ZoneKind.TABLE_CELL,
                    verdict.rotate_cw if verdict.rotate_cw not in (None, 0) else None,
                    verdict.confidence,
                    index,
                    cell.key,
                    dict(verdict.letters),
                    verdict.note,
                )
            )
        elif mixed:
            zones.append(
                RotatedZone(
                    inner_page,
                    ZoneKind.TABLE_CELL_MIXED,
                    None,
                    verdict.confidence,
                    index,
                    cell.key,
                    dict(verdict.letters),
                    verdict.note,
                )
            )
    return info, zones


def _mixed_subzones(gray600: np.ndarray, inner: Box, dpi: int, lang: str) -> list[RotatedZone]:
    """Боковые куски смешанной ячейки по соседям глифов (в пикселях страницы)."""
    if inner.width <= 0 or inner.height <= 0:
        return []
    crop = np.ascontiguousarray(gray600[inner.slice])
    factor = dpi / WORK_DPI
    stats = glyph_components(downscale(crop, factor), WORK_DPI)
    zones: list[RotatedZone] = []
    for small_box in cluster_rotated(stats, WORK_DPI):
        box = small_box.scaled(factor).shifted(inner.x0, inner.y0).clipped(gray600.shape[1], gray600.shape[0])
        rotate, letters = side_of(gray600, box, dpi, lang)
        zones.append(
            RotatedZone(
                box,
                ZoneKind.TABLE_CELL,
                rotate,
                0.0 if rotate is None else 1.0,
                letters=letters,
                note="боковая часть смешанной ячейки",
            )
        )
    return zones


def side_of(
    gray600: np.ndarray, box: Box, dpi: int = 600, lang: str = LANGUAGES
) -> tuple["int | None", dict[int, int]]:
    """Сторона бокового текста (90 или 270) по числу букв, прочитанных tesseract под каждым углом.

    Args:
        gray600: Растр страницы.
        box: Зона в пикселях растра.
        dpi: Разрешение растра.
        lang: Языки tesseract.

    Returns:
        Поворот по часовой до прямого текста (или None, если букв нет ни так, ни так) и
        число букв под каждым углом.
    """
    height, width = gray600.shape[:2]
    padded = box.padded(mm_to_px(SIDE_PAD_MM, dpi)).clipped(width, height)
    crop = np.ascontiguousarray(gray600[padded.slice])
    if crop.size == 0:
        return None, {}
    # ``evidence_at`` считает буквы, а при их отсутствии — уверенно прочитанные цифры: боковые
    # размеры на чертеже («36000») букв не содержат.
    letters = {angle: evidence_at(crop, angle, False, lang) for angle in (90, 270)}
    best = max(letters, key=letters.get)
    if letters[best] < MIN_LETTERS:
        return None, letters
    if letters[90] == letters[270]:
        return None, letters
    return best, letters


def free_zones(
    gray600: np.ndarray,
    exclude: list[Box],
    kind: ZoneKind,
    dpi: int = 600,
    lang: str = LANGUAGES,
    with_side: bool = True,
    inside: "Box | None" = None,
) -> list[RotatedZone]:
    """Зоны бокового текста вне таблиц (по соседям глифов) с определением стороны.

    Args:
        gray600: Растр страницы.
        exclude: Рамки, внутри которых искать не надо (таблицы, растр).
        kind: Какой вид присвоить найденным зонам.
        dpi: Разрешение растра.
        lang: Языки tesseract.
        with_side: Звать ли tesseract за стороной (дорого: две попытки на зону).
        inside: Искать только внутри этой рамки (подписи на конкретной схеме).

    Returns:
        Зоны в пикселях растра.
    """
    factor = dpi / WORK_DPI
    small = downscale(gray600, factor)
    stats = glyph_components(small, WORK_DPI, [b.scaled(1.0 / factor) for b in exclude])
    height, width = gray600.shape[:2]
    zones: list[RotatedZone] = []
    for small_box in cluster_rotated(stats, WORK_DPI):
        box = small_box.scaled(factor).clipped(width, height)
        if inside is not None and (
            box.x0 < inside.x0 - box.width
            or box.x1 > inside.x1 + box.width
            or box.y0 < inside.y0 - box.height
            or box.y1 > inside.y1 + box.height
        ):
            continue
        rotate, letters = (None, {})
        if with_side:
            rotate, letters = side_of(gray600, box, dpi, lang)
        zones.append(RotatedZone(box, kind, rotate, 0.0 if rotate is None else 1.0, letters=letters))
    return zones


def rotate_crop(gray600: np.ndarray, box: Box, rotate: int, pad_mm: float = SIDE_PAD_MM, dpi: int = 600) -> np.ndarray:
    """Вырезка зоны с полем, повёрнутая так, чтобы текст читался.

    Args:
        gray600: Растр страницы.
        box: Зона.
        rotate: Поворот по часовой (0/90/180/270).
        pad_mm: Поле вокруг зоны.
        dpi: Разрешение растра.

    Returns:
        Серый массив выпрямленной вырезки.
    """
    height, width = gray600.shape[:2]
    padded = box.padded(mm_to_px(pad_mm, dpi)).clipped(width, height)
    crop = np.ascontiguousarray(gray600[padded.slice])
    return rotate_image(crop, rotate) if rotate else crop


# Прямая цепочка считается «без слоя», если слова слоя накрывают её рамку меньше чем на эту долю.
LAYER_COVER_MAX = 0.2


def upright_zones(
    gray600: np.ndarray, exclude: list[Box], layer_boxes: list[Box], dpi: int = 600, inside: "list[Box] | None" = None
) -> list[RotatedZone]:
    """Прямые строки и подписи вне таблиц, под которыми FineReader не оставил слоя.

    Args:
        gray600: Растр страницы.
        exclude: Рамки, внутри которых искать не надо (таблицы, найденные боковые зоны).
        layer_boxes: Рамки слов слоя в пикселях растра; цепочка, накрытая ими, пропускается.
        dpi: Разрешение растра.
        inside: Рамки картинок и схем: цепочка внутри них — ``LINE_ART_UPRIGHT``, иначе ``STANDALONE_UPRIGHT``.

    Returns:
        Зоны с ``rotate_cw = 0`` (только для чтения и вставки; слов слоя они не удаляют).
    """
    factor = dpi / WORK_DPI
    small = downscale(gray600, factor)
    stats = glyph_components(small, WORK_DPI, [b.scaled(1.0 / factor) for b in exclude])
    height, width = gray600.shape[:2]
    zones: list[RotatedZone] = []
    for small_box in cluster_lines(stats, WORK_DPI, vertical=False):
        box = small_box.scaled(factor).clipped(width, height)
        if box.area <= 0:
            continue
        covered = 0
        for word in layer_boxes:
            ix0, iy0 = max(box.x0, word.x0), max(box.y0, word.y0)
            ix1, iy1 = min(box.x1, word.x1), min(box.y1, word.y1)
            if ix1 > ix0 and iy1 > iy0:
                covered += (ix1 - ix0) * (iy1 - iy0)
        if covered >= LAYER_COVER_MAX * box.area:
            continue
        art = any(
            a.x0 - box.height <= box.x0
            and box.x1 <= a.x1 + box.height
            and a.y0 - box.height <= box.y0
            and box.y1 <= a.y1 + box.height
            for a in (inside or [])
        )
        kind = ZoneKind.LINE_ART_UPRIGHT if art else ZoneKind.STANDALONE_UPRIGHT
        zones.append(RotatedZone(box, kind, 0, 1.0, note="прямой текст без слоя"))
    return zones
