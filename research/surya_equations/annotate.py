"""Картинки для разметки глазами: полоса или её область с сеткой в координатах кадра кэша (боксов surya не видно)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from research.surya_equations.overlay import load_font

# Полоса целиком — не шире/выше этого (px); область — шире: координаты читаются по подписям сетки.
PAGE_WIDTH_PX, PAGE_HEIGHT_PX = 1100, 1900
ZOOM_WIDTH_PX = 1400
# Шаг сетки в единицах кадра: 50 для полосы, 25 для области уже 450 единиц.
GRID_STEP, GRID_STEP_ZOOM, ZOOM_SPAN = 50, 25, 450
GRID_COLOR, GRID_COLOR_MAJOR = (0, 150, 255), (0, 90, 255)  # RGB (PIL)


def grid_view(
    gray: np.ndarray, frame_width: int, region: tuple[float, float, float, float] | None = None
) -> Image.Image:
    """Полоса (или область) с сеткой, подписанной координатами кадра кэша.

    Картинка масштабируется под экран, поэтому координаты разметчик читает только по подписям
    сетки, а не пересчётом пикселей.

    Args:
        gray: серый JPEG полосы.
        frame_width: ширина кадра кэша (единиц).
        region: область ``(x0, y0, x1, y1)`` в кадре; ``None`` — вся полоса.

    Returns:
        RGB-картинка PIL.
    """
    scale = gray.shape[1] / frame_width
    frame_height = gray.shape[0] / scale
    x0, y0, x1, y1 = region if region is not None else (0, 0, frame_width, frame_height)
    crop = Image.fromarray(gray[int(y0 * scale) : int(y1 * scale), int(x0 * scale) : int(x1 * scale)]).convert("RGB")
    target = ZOOM_WIDTH_PX if region is not None else PAGE_WIDTH_PX
    zoom = min(target / crop.width, PAGE_HEIGHT_PX / crop.height)
    crop = crop.resize((int(crop.width * zoom), int(crop.height * zoom)))
    draw = ImageDraw.Draw(crop)
    font = load_font(14)
    step = GRID_STEP_ZOOM if (x1 - x0) < ZOOM_SPAN else GRID_STEP
    k = scale * zoom
    # Вертикальные линии с подписью сверху, горизонтальные — с подписью слева; сотни — темнее.
    for value in range(int(x0 // step + 1) * step, int(x1), step):
        x = (value - x0) * k
        color = GRID_COLOR_MAJOR if value % 100 == 0 else GRID_COLOR
        draw.line((x, 0, x, crop.height), fill=color, width=1)
        draw.text((x + 2, 2), str(value), fill=GRID_COLOR_MAJOR, font=font)
    for value in range(int(y0 // step + 1) * step, int(y1), step):
        y = (value - y0) * k
        color = GRID_COLOR_MAJOR if value % 100 == 0 else GRID_COLOR
        draw.line((0, y, crop.width, y), fill=color, width=1)
        draw.text((2, y + 2), str(value), fill=GRID_COLOR_MAJOR, font=font)
    return crop
