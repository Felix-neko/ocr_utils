"""Зоны повёрнутого текста вне таблиц: цепочки глифов Docstrum по битональной копии, без tesseract.

ЧТО ИЩЕТСЯ. Боковой текст, набранный «лёжа»: подписи осей на графиках, надписи на схемах,
колонтитулы боком, отдельные повёрнутые врезки. Внутри таблиц он тоже бывает (шапки граф), но
таблицы разбираются своим ходом (``rotated_text.tables``) и сюда подаются исключениями — как и
растр, где глифов нет вовсе, зато полно пятен размером с букву.

КАК. ``docstrum.glyph_components`` берёт связные компоненты размера глифа (0.8–7 мм), а
``cluster_rotated`` — вертикальные цепочки ближайших соседей (kNN, k = 3, досягаемость 1.15
высоты глифа, не короче трёх глифов). Это ровно то, чем ``text_layer_fix.zones.free_zones``
находит зоны на растре PDF; сторона поворота (90 или 270) там определяется tesseract, здесь
не определяется: на этапе ``detect`` нужны только области, а две попытки OCR на зону — дорого.
"""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.geometry import Box
from ocr_utils.page_layout.regions import Region, RegionKind
from ocr_utils.page_layout.rotated_text.docstrum import cluster_rotated, glyph_components


def rotated_zones(bitonal: np.ndarray, dpi: int, exclude: list[Box], line_art: list[Box] | None = None) -> list[Region]:
    """Зоны повёрнутого текста в пикселях поданной копии.

    Args:
        bitonal: Битональная копия (краска 0, бумага 255).
        dpi: Её разрешение.
        exclude: Рамки, внутри которых глифы не считаются (таблицы, растр).
        line_art: Рамки line art — зона внутри них помечается ``inside_line_art`` (подпись на схеме).

    Returns:
        Области ``ROTATED_TEXT``; ``info["glyphs"]`` — число глифов в цепочке по оценке площади.
    """
    if bitonal is None or bitonal.size == 0:
        return []
    height, width = bitonal.shape[:2]
    stats = glyph_components(bitonal, dpi, exclude or None)
    regions: list[Region] = []
    for box in cluster_rotated(stats, dpi):
        box = box.clipped(width, height)
        if box.area <= 0:
            continue
        inside = any(_inside(box, art) for art in line_art or [])
        regions.append(Region(box, RegionKind.ROTATED_TEXT, None, "rotated_text", False, {"inside_line_art": inside}))
    return regions


def _inside(box: Box, outer: Box) -> bool:
    """Лежит ли зона внутри рамки с допуском в её собственную ширину/высоту (как в ``free_zones``)."""
    return (
        box.x0 >= outer.x0 - box.width
        and box.x1 <= outer.x1 + box.width
        and box.y0 >= outer.y0 - box.height
        and box.y1 <= outer.y1 + box.height
    )


__all__ = ["rotated_zones"]
