"""Крупный штрих на этапе ``detect``: детектор ``line_art_detection`` по копии 1/4 -> регионы базы.

ЗАЧЕМ. При сборке финальных PDF страница берётся из прогона FineReader без коррекции
геометрии, если детектор порчи геометрии (``ocr_utils.geometry_regression``) сказал «bad»;
а тот, решая, где на странице рисунок, зовёт ``line_art_detection.features.analyse_gray``
на бинарном рендере 150 dpi (``geometry_regression.regions.lineart_boxes``). Качество этого
детектора до сих пор было видно только косвенно — по вердиктам. Здесь его находки кладутся
в базу разметки двумя видами (``STROKE_KINDS``) и уходят в CVAT: разметчик их правит, и
уточнённая база становится эталоном для оценки детектора.

ТОТ ЖЕ АЛГОРИТМ, ДРУГОЙ ВХОД. Детектор порчи геометрии подаёт детектору страницу,
бинаризованную FineReader; здесь FineReader ещё не было, и бинаризуется серая копия 1/4
скана порогом Оцу — тем же, каким детектор таблиц (``page_layout.tables.ruling.binarize``)
на той же копии ищет линейки. Разрешение то же, 150 dpi при 600 у скана
(``geometry_regression.WORK_DPI``), пороги — те же ``params_for_dpi``; surya и
исключения растра не подаются, как и в ``lineart_boxes``. Расхождения с прогоном по PDF
возможны там, где бинаризация Оцу и бинаризация FineReader разойдутся: тени у корешка,
просвет с оборота, пересвеченные полосы.

ВИДЫ. Объединённые рамки детектора одни и те же (``PageFindings.boxes``, их видит детектор
геометрии), а вид приписан каждой по источникам её кандидатов (``PageFindings.kinds``):
скопление линеек — ``stroke_table``, связное пятно — ``stroke_drawing``.

ВЕРСИЯ. ``STROKE_DETECTOR_VERSION`` пишется в ``Page.stroke_detector_version`` и решает при
``--skip-detected``, пересчитывать ли полосу. Поднимать при любой правке, меняющей рамки
или виды: как здесь (бинаризация, пересчёт), так и в ``line_art_detection.features``.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from ocr_utils.db.models import KIND_STROKE_DRAWING, KIND_STROKE_TABLE
from ocr_utils.line_art_detection.features import BOX_KIND_TABLE, PageFindings, analyse_gray, members_of, params_for_dpi
from ocr_utils.page_layout.tables import Region
from ocr_utils.page_layout.tables.ruling import binarize

STROKE_DETECTOR_VERSION = 1

# Псевдосерый битональный кадр для детектора: он берёт краску порогом ``gray < 128``.
PSEUDO_INK = 0
PSEUDO_PAPER = 255


def db_kind(box_kind: str) -> str:
    """Вид рамки детектора (``BOX_KIND_*``) -> вид региона в базе (``STROKE_KINDS``)."""
    return KIND_STROKE_TABLE if box_kind == BOX_KIND_TABLE else KIND_STROKE_DRAWING


def bitonal(gray: np.ndarray) -> np.ndarray:
    """Серая копия -> битональный кадр в шкале детектора: краска 0, бумага 255.

    Порог Оцу по всей копии — тот же, что у детектора таблиц на той же копии, поэтому
    линейки, которые видит один, видит и другой.

    Args:
        gray: Серая копия 1/4 полосы.

    Returns:
        Кадр той же формы из двух значений.
    """
    ink = binarize(gray)  # 255 там, где краска
    return np.where(ink > 0, PSEUDO_INK, PSEUDO_PAPER).astype(np.uint8)


def find_strokes(gray: np.ndarray, dpi: int) -> PageFindings:
    """Крупный штрих на серой копии полосы: бинаризация и детектор с порогами под ``dpi``.

    Args:
        gray: Серая копия 1/4 полосы.
        dpi: Её разрешение (150 при 600 у скана).

    Returns:
        Находки детектора в пикселях копии, с видом на каждую рамку.
    """
    return analyse_gray(bitonal(gray), params_for_dpi(dpi))


def to_regions(findings: PageFindings, scale: float, page_size: tuple[int, int]) -> list[Region]:
    """Рамки детектора в пикселях копии -> регионы в пикселях оригинала.

    Args:
        findings: Находки по копии.
        scale: Во сколько раз оригинал крупнее копии.
        page_size: ``(ширина, высота)`` оригинала — рамка зажимается в кадр, потому что
            округление на границе копии даёт координату за краем, а CVAT такой шейп отвергает.

    Returns:
        По региону на рамку; JSON ``detector_info`` — вид рамки и сводка её кандидатов.
    """
    width, height = page_size
    regions: list[Region] = []
    for box, kind in zip(findings.boxes, findings.kinds):
        x1 = max(0, min(width, int(round(box[0] * scale))))
        y1 = max(0, min(height, int(round(box[1] * scale))))
        x2 = max(0, min(width, int(round(box[2] * scale))))
        y2 = max(0, min(height, int(round(box[3] * scale))))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        members = members_of(box, findings.candidates)
        info = {
            "kind": kind,
            "sources": sorted({c.source for c in members}),
            "candidates": len(members),
            "area_px": int(sum(c.area for c in members)),  # краска кандидатов в пикселях копии
            "fill": round(float(np.mean([c.fill for c in members])), 4) if members else 0.0,
            "long_run_frac": round(float(np.mean([c.long_run_frac for c in members])), 4) if members else 0.0,
        }
        regions.append(Region(x1, y1, x2, y2, db_kind(kind), json.dumps(info, ensure_ascii=False)))
    return regions


def detect_stroke_regions(gray: np.ndarray, dpi: int, scale: float, page_size: tuple[int, int]) -> list[Region]:
    """Полный ход по одной полосе: детектор на серой копии -> регионы оригинала.

    Args:
        gray: Серая копия 1/4 полосы.
        dpi: Разрешение копии.
        scale: Во сколько раз оригинал крупнее копии.
        page_size: ``(ширина, высота)`` оригинала.

    Returns:
        Регионы видов ``STROKE_KINDS`` в пикселях оригинала.
    """
    if gray is None or gray.size == 0:
        return []
    return to_regions(find_strokes(gray, dpi), scale, page_size)


__all__ = ["STROKE_DETECTOR_VERSION", "bitonal", "db_kind", "detect_stroke_regions", "find_strokes", "to_regions"]
