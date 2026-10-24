"""Оверлеи стенда: карточка фото (эталон CVAT, прежняя рамка детектора, новая рамка, кривой четырёхугольник, точки края) с увеличенными углами и листы карточек с легендой в поле."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, framed, header_strip
from ocr_utils.units import mm_to_px_float

# Палитра (BGR) — по навыку draw-overlay: главное насыщенным, справочное приглушённым.
COLOUR_REF = (0, 150, 0)  # эталон CVAT — «принято», зелёный
COLOUR_DET = (0, 165, 255)  # прежняя рамка детектора page_layout — оранжевый
COLOUR_NEW = (220, 90, 20)  # новая осевая рамка — синий (главная граница)
COLOUR_QUAD = (200, 60, 200)  # кривой четырёхугольник — фиолетовый
COLOUR_EDGE = (140, 140, 0)  # годные точки края — бирюзовый
COLOUR_EDGE_BAD = (0, 0, 220)  # отброшенные точки края — красный
LINE_ALPHA = 0.85

# Поле карточки вокруг рамок, мм; окно увеличенного угла, мм.
CARD_MARGIN_MM = 5.0
ZOOM_MM = 6.0
CARD_WIDTH = 760

LEGEND = [
    LegendEntry("эталон CVAT (pack1_reviewed)", COLOUR_REF, LINE_ALPHA),
    LegendEntry("прежняя рамка детектора page_layout", COLOUR_DET, LINE_ALPHA, SampleStyle.DASHED),
    LegendEntry("новая осевая рамка", COLOUR_NEW, LINE_ALPHA),
    LegendEntry("кривой четырёхугольник (все стороны чёткие)", COLOUR_QUAD, LINE_ALPHA),
    LegendEntry("точки края в подгонке", COLOUR_EDGE),
    LegendEntry("точки края вне подгонки", COLOUR_EDGE_BAD),
]


def dashed_rect(img: np.ndarray, box, colour, thickness: int, dash: int = 10) -> None:
    """Пунктирная рамка (OpenCV пунктира не умеет).

    Args:
        img: Холст BGR (рисуется на месте).
        box: ``(x0, y0, x1, y1)`` px холста.
        colour: Цвет BGR.
        thickness: Толщина.
        dash: Длина штриха, px.
    """
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    for (ax, ay), (bx, by) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        length = max(abs(bx - ax), abs(by - ay))
        for s in range(0, length, 2 * dash):
            e = min(s + dash, length)
            p = (ax + (bx - ax) * s // max(length, 1), ay + (by - ay) * s // max(length, 1))
            q = (ax + (bx - ax) * e // max(length, 1), ay + (by - ay) * e // max(length, 1))
            cv2.line(img, p, q, colour, thickness, cv2.LINE_AA)


def draw_marks(canvas: np.ndarray, offset: np.ndarray, scale: float, ref, det, photo: dict | None, thick: int) -> None:
    """Рисует рамки, четырёхугольник и точки края на холсте (в слое с прозрачностью :data:`LINE_ALPHA`).

    Args:
        canvas: Холст BGR — вырезка полосы, увеличенная в ``scale`` раз (рисуется на месте).
        offset: Левый верхний угол вырезки, px полосы.
        scale: Увеличение вырезки.
        ref: Эталон или None.
        det: Прежняя рамка или None.
        photo: JSON фото или None.
        thick: Толщина линий.
    """

    def tr(box):
        return [(box[0] - offset[0]) * scale, (box[1] - offset[1]) * scale,
                (box[2] - offset[0]) * scale, (box[3] - offset[1]) * scale]  # fmt: skip

    layer = canvas.copy()
    if ref is not None:
        x0, y0, x1, y1 = (int(round(v)) for v in tr(ref))
        cv2.rectangle(layer, (x0, y0), (x1, y1), COLOUR_REF, thick, cv2.LINE_AA)
    if det is not None:
        dashed_rect(layer, tr(det), COLOUR_DET, thick)
    if photo is not None:
        x0, y0, x1, y1 = (int(round(v)) for v in tr(photo["box"]))
        cv2.rectangle(layer, (x0, y0), (x1, y1), COLOUR_NEW, thick, cv2.LINE_AA)
        if photo["corners"] is not None and all(photo["sharp_sides"]):
            poly = (np.array(photo["polygon"]) - offset) * scale
            cv2.polylines(layer, [np.round(poly).astype(np.int32)], True, COLOUR_QUAD, thick, cv2.LINE_AA)
    cv2.addWeighted(layer, LINE_ALPHA, canvas, 1 - LINE_ALPHA, 0, canvas)
    if photo is not None:
        r = max(1, thick)
        for side in photo["sides"]:
            horizontal = side["side"] in ("top", "bottom")
            for a, c, ok in zip(side["along"], side["across"], side["inlier"]):
                x, y = (a, c) if horizontal else (c, a)
                p = (int(round((x - offset[0]) * scale)), int(round((y - offset[1]) * scale)))
                cv2.circle(canvas, p, r, COLOUR_EDGE if ok else COLOUR_EDGE_BAD, -1, cv2.LINE_AA)


def union_box(*boxes):
    """Охватывающая рамка непустых рамок."""
    boxes = [b for b in boxes if b is not None]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def photo_card(
    gray: np.ndarray, dpi: float, title: list[str], ref, det, photo: dict | None, zoom: bool = True
) -> np.ndarray:
    """Карточка фото: вырезка с рамками и под ней четыре увеличенных угла (по рамке эталона или новой).

    Args:
        gray: Серая полоса стенда.
        dpi: Её разрешение.
        title: Строки шапки карточки.
        ref, det: Эталон и прежняя рамка, px полосы (или None).
        photo: JSON нового фото или None.
        zoom: Добавить увеличенные углы (для листов разметки классов — без них).

    Returns:
        Карточка BGR шириной :data:`CARD_WIDTH`.
    """
    h, w = gray.shape
    margin = mm_to_px_float(CARD_MARGIN_MM, dpi)
    box = union_box(ref, det, None if photo is None else photo["box"])
    x0, y0 = int(max(box[0] - margin, 0)), int(max(box[1] - margin, 0))
    x1, y1 = int(min(box[2] + margin, w)), int(min(box[3] + margin, h))
    crop = cv2.cvtColor(gray[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    scale = CARD_WIDTH / crop.shape[1]
    view = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    draw_marks(view, np.array([x0, y0], float), scale, ref, det, photo, 2)
    if not zoom:
        return np.vstack([header_strip(title, CARD_WIDTH), view])
    # Увеличенные углы сеткой 2×2 под вырезкой: по углам новой рамки (если её нет — эталона).
    anchor = photo["box"] if photo is not None else (ref if ref is not None else det)
    half = mm_to_px_float(ZOOM_MM, dpi) / 2
    size = CARD_WIDTH // 2
    tiles = []
    for cx, cy in ((anchor[0], anchor[1]), (anchor[2], anchor[1]), (anchor[0], anchor[3]), (anchor[2], anchor[3])):
        zx0, zy0 = int(max(cx - half, 0)), int(max(cy - half, 0))
        zx1, zy1 = int(min(cx + half, w)), int(min(cy + half, h))
        factor = size / max(zx1 - zx0, zy1 - zy0, 1)
        z = cv2.cvtColor(gray[zy0:zy1, zx0:zx1], cv2.COLOR_GRAY2BGR)
        z = cv2.resize(z, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)
        draw_marks(z, np.array([zx0, zy0], float), factor, ref, det, photo, 2)
        tile = np.full((size, size, 3), 255, np.uint8)
        zh, zw = min(size, z.shape[0]), min(size, z.shape[1])
        tile[:zh, :zw] = z[:zh, :zw]
        # Тонкая серая рамка плитки, чтобы углы не сливались.
        cv2.rectangle(tile, (0, 0), (size - 1, size - 1), (170, 170, 170), 1)
        tiles.append(tile)
    zoom_row = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    return np.vstack([header_strip(title, CARD_WIDTH), view, zoom_row])


def grid(cards: list[np.ndarray], cols: int, gap: int = 12) -> np.ndarray:
    """Сетка карточек: ряды по ``cols``, высота ряда — по самой высокой, промежутки белые.

    Args:
        cards: Карточки одной ширины.
        cols: Карточек в ряду.
        gap: Промежуток, px.

    Returns:
        Лист BGR.
    """
    width = max(c.shape[1] for c in cards)
    rows = []
    for i in range(0, len(cards), cols):
        row_cards = cards[i : i + cols]
        height = max(c.shape[0] for c in row_cards)
        padded = [cv2.copyMakeBorder(c, 0, height - c.shape[0], 0, width - c.shape[1] + gap, cv2.BORDER_CONSTANT,
                                     value=(255, 255, 255)) for c in row_cards]  # fmt: skip
        while len(padded) < cols:
            padded.append(np.full((height, width + gap, 3), 255, np.uint8))
        rows.append(np.hstack(padded))
        rows.append(np.full((gap, rows[-1].shape[1], 3), 255, np.uint8))
    return np.vstack(rows)


def sheet(cards: list[np.ndarray], header: list[str], cols: int = 3) -> np.ndarray:
    """Лист карточек со шапкой и легендой в поле.

    Args:
        cards: Карточки.
        header: Строки шапки листа.
        cols: Карточек в ряду.

    Returns:
        Лист BGR.
    """
    return framed(grid(cards, cols), header, LEGEND)
