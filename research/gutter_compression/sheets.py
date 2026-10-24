"""Листы стенда: пояса сжатия (вырезки у корешка из выпрямленных итогов с мерой в подписи) и «было | стало» растяжения с увеличением у корешка по трём высотам страницы."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ocr_utils.page_layout.overlay_frame import LegendEntry, SampleStyle, _font, framed

# Цвета (BGR, палитра ``/draw-overlay``): линия корешка — синяя пунктиром, край текста у корешка — оранжевый.
COLOUR_SPINE = (200, 80, 0)
COLOUR_EDGE = (0, 140, 255)
# Кегль подписи вырезки, px.
CAPTION_SIZE = 18


def caption(image: np.ndarray, text: str) -> np.ndarray:
    """Вырезка с подписью над ней в белом поле (не поверх страницы).

    Args:
        image: Вырезка BGR.
        text: Подпись.

    Returns:
        Новая картинка: полоса подписи и вырезка.
    """
    strip = Image.new("RGB", (image.shape[1], CAPTION_SIZE + 8), (255, 255, 255))
    ImageDraw.Draw(strip).text((4, 3), text, fill=(0, 0, 0), font=_font(CAPTION_SIZE))
    return np.vstack([cv2.cvtColor(np.asarray(strip), cv2.COLOR_RGB2BGR), image])


def gutter_crop(image: np.ndarray, spine_x: float, y_centre: float, px_per_mm: float, half_width_mm: float,
                half_height_mm: float, zoom: float) -> np.ndarray:  # fmt: skip
    """Вырезка у корешка с увеличением и линией корешка пунктиром по краям (вне текста — короткие риски).

    Args:
        image: Холст BGR.
        spine_x: X линии корешка, px.
        y_centre: Центр вырезки по высоте, px.
        px_per_mm: Плотность, px/мм.
        half_width_mm: Половина ширины вырезки, мм.
        half_height_mm: Половина высоты, мм.
        zoom: Увеличение.

    Returns:
        Вырезка BGR (за краем холста — белое).
    """
    hw, hh = int(half_width_mm * px_per_mm), int(half_height_mm * px_per_mm)
    x0, y0 = int(spine_x) - hw, int(y_centre) - hh
    canvas = np.full((2 * hh, 2 * hw, 3), 255, np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(image.shape[1], x0 + 2 * hw), min(image.shape[0], y0 + 2 * hh)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = image[sy0:sy1, sx0:sx1]
    out = cv2.resize(canvas, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC if zoom > 1 else cv2.INTER_AREA)
    # Риски линии корешка сверху и снизу (не поверх букв).
    xs = int(round((spine_x - x0) * zoom))
    tick = max(6, int(1.5 * px_per_mm * zoom))
    cv2.line(out, (xs, 0), (xs, tick), COLOUR_SPINE, 2)
    cv2.line(out, (xs, out.shape[0] - tick), (xs, out.shape[0] - 1), COLOUR_SPINE, 2)
    return out


def band_sheet(crops: list[tuple[np.ndarray, str]], columns: int, header: list[str]) -> np.ndarray:
    """Лист пояса: вырезки у корешка с подписями сеткой.

    Args:
        crops: Вырезки и подписи (одного размера).
        columns: Вырезок в ряду.
        header: Строки шапки.

    Returns:
        Лист BGR.
    """
    tiles = [caption(c, t) for c, t in crops]
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.full((rows * (h + 6), columns * (w + 6), 3), 255, np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, columns)
        sheet[r * (h + 6) : r * (h + 6) + h, c * (w + 6) : c * (w + 6) + w] = tile
    legend = [LegendEntry("линия корешка (риски у краёв вырезки)", COLOUR_SPINE, style=SampleStyle.LINE)]
    return framed(sheet, header, legend)


def before_after_sheet(before: np.ndarray, after: np.ndarray, spine_x: float, ys: list[float], px_per_mm: float,
                       header: list[str], notes: list[str], spine_after: float | None = None) -> np.ndarray:  # fmt: skip
    """Лист «было | стало»: разворот целиком уменьшенным и вырезки у корешка ×2 на трёх высотах.

    Args:
        before: Холст без растяжения (Кунс + поле).
        after: Холст с растяжением у корешка.
        spine_x: X линии корешка, px.
        ys: Высоты центров вырезок, px.
        px_per_mm: Плотность, px/мм.
        header: Строки шапки.
        notes: Подписи к вырезкам по высотам (меры до и после).
        spine_after: X линии корешка на холсте «стало» (он шире при расширении страниц) или None — как у «было».

    Returns:
        Лист BGR.
    """
    zoom = 2.0
    spine_b = spine_x if spine_after is None else spine_after
    rows = []
    for y, note in zip(ys, notes):
        a = gutter_crop(before, spine_x, y, px_per_mm, 22.0, 9.0, zoom)
        b = gutter_crop(after, spine_b, y, px_per_mm, 22.0, 9.0, zoom)
        gap = np.full((a.shape[0], 12, 3), 255, np.uint8)
        rows.append(caption(np.hstack([a, gap, b]), f"было | стало, ×2 у корешка — {note}"))
    width = rows[0].shape[1]
    scale = width / (before.shape[1] + after.shape[1] + 12)
    small = [cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for im in (before, after)]
    gap_w = max(width - small[0].shape[1] - small[1].shape[1], 0)
    overview = np.hstack([small[0], np.full((small[0].shape[0], gap_w, 3), 255, np.uint8), small[1]])
    # Округление размеров уменьшенных холстов: обзор доводится до ширины вырезок (обрезка или белое поле справа).
    overview = overview[:, :width]
    if overview.shape[1] < width:
        overview = np.hstack([overview, np.full((overview.shape[0], width - overview.shape[1], 3), 255, np.uint8)])
    for x in (int(spine_x * scale), int(spine_b * scale) + small[0].shape[1] + gap_w):
        cv2.line(overview, (x, 0), (x, 8), COLOUR_SPINE, 2)
    for y in ys:
        yy = int(y * scale)
        cv2.line(overview, (0, yy), (6, yy), COLOUR_EDGE, 2)
    body = np.vstack([caption(overview, "разворот: было | стало (оранжевые риски слева — высоты вырезок)")] + rows)
    legend = [LegendEntry("линия корешка (риски у краёв)", COLOUR_SPINE),
              LegendEntry("высоты вырезок (риски у левого края обзора)", COLOUR_EDGE)]  # fmt: skip
    return framed(body, header, legend)
