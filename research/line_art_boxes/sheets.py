"""Листы отсмотра рамок line art: миниатюры страниц с рамками (одним вариантом или «было | стало»), сеткой с легендой в поле."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, header_strip, legend_strip
from research.line_art_boxes.detect import PageArt

# Цвета BGR (палитра проекта, навык draw-overlay).
COLOUR_ART = (0, 140, 255)  # рамка line art — оранжевый, как на листах 08_rectify
COLOUR_SEED = (0, 150, 0)  # затравка роста схемы/рисунка детектора таблиц — «принято»
COLOUR_RASTER = (200, 60, 200)  # растр
COLOUR_TABLE = (200, 140, 60)  # таблица — вспомогательное
COLOUR_NUMBER = (20, 20, 20)
ART_ALPHA = 0.18  # заливка рамки line art
# Высота миниатюры страницы, px: рамка рисунка и строки текста под ней различимы глазом.
TILE_HEIGHT = 760
GAP = 12


def legend_entries(seeds: bool = True) -> list[LegendEntry]:
    """Легенда листов.

    Args:
        seeds: Показывать строку затравок роста.

    Returns:
        Строки легенды.
    """
    entries = [
        LegendEntry("рамка line art (номер — порядок в разборе)", COLOUR_ART, alpha=ART_ALPHA, style=SampleStyle.BOX),
        LegendEntry("растр", COLOUR_RASTER),
        LegendEntry("таблица", COLOUR_TABLE),
    ]
    if seeds:
        entries.insert(1, LegendEntry("затравка схемы/рисунка детектора таблиц (до роста)", COLOUR_SEED, alpha=0.9,
                                      style=SampleStyle.DASHED))  # fmt: skip
    return entries


def _dashed_rect(canvas: np.ndarray, box, colour, thickness: int = 2, dash: int = 10) -> None:
    """Пунктирная рамка (рисует на ``canvas`` на месте: это холст оверлея, а не аргумент-данные)."""
    x0, y0, x1, y1 = (int(v) for v in box)
    for xa in range(x0, x1, 2 * dash):
        cv2.line(canvas, (xa, y0), (min(xa + dash, x1), y0), colour, thickness)
        cv2.line(canvas, (xa, y1), (min(xa + dash, x1), y1), colour, thickness)
    for ya in range(y0, y1, 2 * dash):
        cv2.line(canvas, (x0, ya), (x0, min(ya + dash, y1)), colour, thickness)
        cv2.line(canvas, (x1, ya), (x1, min(ya + dash, y1)), colour, thickness)


def page_tile(image: np.ndarray, art: PageArt, title: list[str], height: int = TILE_HEIGHT) -> np.ndarray:
    """Миниатюра страницы с рамками line art, затравками роста, растром и таблицами.

    Args:
        image: Вырезка страницы BGR (родное разрешение).
        art: Разбор страницы одним вариантом.
        title: Строки подписи над миниатюрой.
        height: Высота миниатюры, px.

    Returns:
        Миниатюра с подписью (BGR).
    """
    scale = height / image.shape[0]
    tile = cv2.resize(image, (max(1, int(image.shape[1] * scale)), height), interpolation=cv2.INTER_AREA)
    layer = tile.copy()
    for box in art.boxes:
        x0, y0, x1, y1 = (int(v * scale) for v in box.box)
        cv2.rectangle(layer, (x0, y0), (x1, y1), COLOUR_ART, -1)
    cv2.addWeighted(layer, ART_ALPHA, tile, 1 - ART_ALPHA, 0, tile)
    for box in art.raster:
        cv2.rectangle(tile, *_corners(box, scale), COLOUR_RASTER, 2)
    for box in art.tables:
        cv2.rectangle(tile, *_corners(box, scale), COLOUR_TABLE, 2)
    for growth in art.growths:
        _dashed_rect(tile, [v * art.scale * scale for v in growth.seed], COLOUR_SEED)
    for number, box in enumerate(art.boxes, 1):
        p0, p1 = _corners(box.box, scale)
        cv2.rectangle(tile, p0, p1, COLOUR_ART, 3)
        cv2.putText(tile, str(number), (p0[0] + 4, p0[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOUR_NUMBER, 2)
    return np.vstack([header_strip(title, tile.shape[1]), tile])


def _corners(box, scale: float) -> tuple[tuple[int, int], tuple[int, int]]:
    """Углы рамки на миниатюре."""
    x0, y0, x1, y1 = (int(v * scale) for v in box)
    return (x0, y0), (x1, y1)


def pair_tile(image: np.ndarray, before: PageArt, after: PageArt, title: list[str]) -> np.ndarray:
    """«Было | стало» одной страницы: две миниатюры рядом под общей подписью.

    Args:
        image: Вырезка страницы BGR.
        before: Разбор боевым кодом.
        after: Разбор прототипом.
        title: Подпись страницы.

    Returns:
        Склейка BGR.
    """
    left = page_tile(image, before, [f"было ({before.variant}): рамок {len(before.boxes)}"])
    right = page_tile(image, after, [f"стало ({after.variant}): рамок {len(after.boxes)}"])
    gap = np.full((left.shape[0], GAP, 3), 255, np.uint8)
    body = np.hstack([left, gap, right])
    return np.vstack([header_strip(title, body.shape[1]), body])


def grid(tiles: list[np.ndarray], columns: int, header: list[str], legend: list[LegendEntry]) -> np.ndarray:
    """Сетка миниатюр с шапкой и легендой в полях листа.

    Args:
        tiles: Миниатюры (разной ширины и высоты — добиваются белым).
        columns: Миниатюр в ряду.
        header: Строки шапки листа.
        legend: Легенда.

    Returns:
        Лист BGR.
    """
    cell_w = max(t.shape[1] for t in tiles)
    cell_h = max(t.shape[0] for t in tiles)
    rows = []
    for start in range(0, len(tiles), columns):
        cells = []
        for tile in tiles[start : start + columns]:
            cell = np.full((cell_h, cell_w, 3), 255, np.uint8)
            cell[: tile.shape[0], : tile.shape[1]] = tile
            cells.append(cell)
        while len(cells) < columns:
            cells.append(np.full((cell_h, cell_w, 3), 255, np.uint8))
        row = []
        for cell in cells:
            row += [cell, np.full((cell_h, GAP, 3), 255, np.uint8)]
        rows += [np.hstack(row[:-1]), np.full((GAP, columns * cell_w + (columns - 1) * GAP, 3), 255, np.uint8)]
    body = np.vstack(rows[:-1])
    return np.vstack([header_strip(header, body.shape[1]), body, legend_strip(legend, body.shape[1])])


def box_tile(image: np.ndarray, box: tuple[int, int, int, int], dpi: float, title: list[str],
             margin_mm: float = 12.0, side: int = 520) -> np.ndarray:  # fmt: skip
    """Вырезка вокруг одной рамки line art с полем ``margin_mm``, ужатая или увеличенная до ``side`` по большей стороне.

    Args:
        image: Вырезка страницы BGR.
        box: Рамка в px вырезки.
        dpi: Плотность вырезки.
        title: Подпись.
        margin_mm: Поле вокруг рамки, мм бумаги.
        side: Большая сторона миниатюры, px.

    Returns:
        Миниатюра с подписью (BGR).
    """
    margin = int(round(margin_mm * dpi / 25.4))
    height, width = image.shape[:2]
    x0, y0 = max(0, box[0] - margin), max(0, box[1] - margin)
    x1, y1 = min(width, box[2] + margin), min(height, box[3] + margin)
    crop = image[y0:y1, x0:x1].copy()
    scale = side / max(crop.shape[:2])
    tile = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    p0 = (int((box[0] - x0) * scale), int((box[1] - y0) * scale))
    p1 = (int((box[2] - x0) * scale), int((box[3] - y0) * scale))
    layer = tile.copy()
    cv2.rectangle(layer, p0, p1, COLOUR_ART, -1)
    cv2.addWeighted(layer, ART_ALPHA, tile, 1 - ART_ALPHA, 0, tile)
    cv2.rectangle(tile, p0, p1, COLOUR_ART, 2)
    return np.vstack([header_strip(title, tile.shape[1]), tile])
