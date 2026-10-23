"""Оверлеи срезанных строк: кадр с краем листа и рамками строк, увеличенные вырезки строк, листы вырезок по паку."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed, header_strip
from research.edge_cut_pages.detect import CutLine, FrameResult, Side

# Цвета BGR (палитра ``.claude/skills/draw-overlay``): край листа — розовый «линия отсечки», глиф у края без
# среза — оранжевый, глиф срезанной строки — синий «главная граница».
COLOUR_EDGE = (180, 105, 255)
COLOUR_TOUCH = (0, 165, 255)
COLOUR_CUT = (220, 90, 20)
# Рамки срезанных строк по стороне.
SIDE_COLOUR = {
    Side.BOTTOM: (200, 60, 200),
    Side.TOP: (200, 60, 200),
    Side.OUTER: (0, 0, 220),
    Side.SPINE: (140, 140, 0),
}
SIDE_TEXT = {Side.BOTTOM: "низ", Side.TOP: "верх", Side.OUTER: "наружный край", Side.SPINE: "корешок"}
# Прозрачность края листа и закраски глифов.
EDGE_ALPHA = 0.6
GLYPH_ALPHA = 0.7
# Ширина кадра на оверлее, px; высота вырезки строки, px; запас вокруг строки на вырезке, px ½ кадра.
OVERLAY_WIDTH = 1800
CROP_HEIGHT = 200
CROP_PAD = 50
# Сколько вырезок строк под кадром.
MAX_CROPS = 8


def paint(canvas: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int], alpha: float) -> np.ndarray:
    """Полупрозрачная закраска маски цветом.

    Args:
        canvas: Картинка BGR.
        mask: Булева маска того же размера.
        colour: Цвет BGR.
        alpha: Доля цвета.

    Returns:
        Новая картинка.
    """
    out = canvas.copy()
    out[mask] = (out[mask] * (1 - alpha) + np.array(colour) * alpha).astype(np.uint8)
    return out


def marked(bgr: np.ndarray, result: FrameResult) -> np.ndarray:
    """Кадр ½ с разметкой: край листа, глифы у края, глифы срезанных строк, рамки строк по стороне.

    Args:
        bgr: Кадр BGR ½.
        result: Разбор с масками (``keep_masks=True``).

    Returns:
        Картинка BGR того же размера.
    """
    masks = result.masks
    canvas = bgr.copy()
    if masks is not None:
        canvas = paint(
            canvas, cv2.dilate(masks.edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0, COLOUR_EDGE, EDGE_ALPHA
        )
        canvas = paint(canvas, masks.touching & ~masks.cut, COLOUR_TOUCH, GLYPH_ALPHA)
        canvas = paint(canvas, masks.cut, COLOUR_CUT, GLYPH_ALPHA)
    for line in result.lines:
        x, y, w, h = line.box
        cv2.rectangle(canvas, (x - 6, y - 6), (x + w + 6, y + h + 6), SIDE_COLOUR[line.side], 3)
    return canvas


def line_crop(picture: np.ndarray, line: CutLine, height: int = CROP_HEIGHT) -> np.ndarray:
    """Увеличенная вырезка строки с запасом :data:`CROP_PAD` (видно, что за краем).

    Args:
        picture: Размеченный кадр ½.
        line: Строка.
        height: Высота вырезки, px.

    Returns:
        Картинка BGR высоты ``height``.
    """
    x, y, w, h = line.box
    rows, cols = picture.shape[:2]
    crop = picture[
        max(0, y - CROP_PAD) : min(rows, y + h + CROP_PAD), max(0, x - CROP_PAD) : min(cols, x + w + CROP_PAD)
    ]
    scale = height / crop.shape[0]
    crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), height), interpolation=cv2.INTER_AREA)
    # Ширина вырезки ограничена: длинная строка ужимается по ширине кадра оверлея.
    if crop.shape[1] > OVERLAY_WIDTH:
        crop = cv2.resize(
            crop, (OVERLAY_WIDTH, round(height * OVERLAY_WIDTH / crop.shape[1])), interpolation=cv2.INTER_AREA
        )
    return crop


def pack_row(crops: list[np.ndarray], width: int, gap: int = 8) -> list[np.ndarray]:
    """Раскладывает вырезки по рядам ширины ``width`` (белое поле между ними).

    Args:
        crops: Вырезки.
        width: Ширина ряда, px.
        gap: Зазор, px.

    Returns:
        Ряды BGR ширины ``width``.
    """
    rows, current, used = [], [], 0
    for crop in crops:
        if current and used + crop.shape[1] > width:
            rows.append(current)
            current, used = [], 0
        current.append(crop)
        used += crop.shape[1] + gap
    if current:
        rows.append(current)
    out = []
    for row in rows:
        height = max(c.shape[0] for c in row)
        strip = np.full((height + gap, width, 3), 255, np.uint8)
        x = 0
        for crop in row:
            strip[: crop.shape[0], x : x + crop.shape[1]] = crop
            x += crop.shape[1] + gap
        out.append(strip)
    return out


def legend() -> list[LegendEntry]:
    """Легенда оверлея.

    Returns:
        Строки легенды.
    """
    return [
        LegendEntry("край листа (ступенька бумаги, граница коврика/крышки)", COLOUR_EDGE, alpha=EDGE_ALPHA),
        LegendEntry("глиф у края, строка не срезана", COLOUR_TOUCH, alpha=GLYPH_ALPHA, style=SampleStyle.BOX),
        LegendEntry("глиф срезанной строки", COLOUR_CUT, alpha=GLYPH_ALPHA, style=SampleStyle.BOX),
        LegendEntry("срезанная строка: низ / верх", SIDE_COLOUR[Side.BOTTOM]),
        LegendEntry("срезанная строка: наружный край", SIDE_COLOUR[Side.OUTER]),
        LegendEntry("срезанная строка: корешок", SIDE_COLOUR[Side.SPINE]),
    ]


def frame_overlay(bgr: np.ndarray, result: FrameResult, title: str) -> np.ndarray:
    """Оверлей кадра: шапка, кадр с разметкой, вырезки срезанных строк (сначала не у корешка), легенда — в полях.

    Args:
        bgr: Кадр BGR ½.
        result: Разбор с масками.
        title: Строка шапки (выпуск, кадр, числа).

    Returns:
        Картинка BGR.
    """
    picture = marked(bgr, result)
    scale = OVERLAY_WIDTH / picture.shape[1]
    canvas = cv2.resize(picture, (OVERLAY_WIDTH, round(picture.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    # Вырезки: сначала низ/верх/наружный край (главное), потом корешок.
    ordered = sorted(result.lines, key=lambda line: (line.side is Side.SPINE, -line.touching))[:MAX_CROPS]
    crops = []
    for line in ordered:
        crop = line_crop(picture, line)
        label = header_strip([SIDE_TEXT[line.side] + f": {line.touching}/{line.glyphs} глифов у края"], crop.shape[1])
        crops.append(np.vstack([label, crop]))
    parts = [canvas]
    if crops:
        parts += pack_row(crops, OVERLAY_WIDTH)
    return framed(np.vstack(parts), [title], legend())


def sheet(tiles: list[tuple[np.ndarray, str]], width: int = OVERLAY_WIDTH, title: str = "") -> np.ndarray:
    """Лист отсмотра: подписанные вырезки рядами.

    Args:
        tiles: Пары (вырезка BGR, подпись).
        width: Ширина листа, px.
        title: Шапка листа.

    Returns:
        Картинка BGR.
    """
    crops = []
    for crop, label in tiles:
        crops.append(
            np.vstack([header_strip([label], max(crop.shape[1], 360)), pad_width(crop, max(crop.shape[1], 360))])
        )
    body = np.vstack(pack_row(crops, width)) if crops else np.full((10, width, 3), 255, np.uint8)
    return np.vstack([header_strip([title], width), body]) if title else body


def pad_width(picture: np.ndarray, width: int) -> np.ndarray:
    """Добивает картинку белым справа до ширины ``width``.

    Args:
        picture: Картинка BGR.
        width: Нужная ширина.

    Returns:
        Картинка BGR.
    """
    if picture.shape[1] >= width:
        return picture
    pad = np.full((picture.shape[0], width - picture.shape[1], 3), 255, np.uint8)
    return np.hstack([picture, pad])
