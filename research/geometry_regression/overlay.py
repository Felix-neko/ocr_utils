"""Картинка «было | стало» для просмотра находок глазами.

Слева страница без коррекции (B), справа с коррекцией (A), обе с реперной сеткой через 1/12
ширины и высоты (по образцу ``curved_lines.finereader_compare.compose``, но и с вертикалями:
наклон линейки или кромки колонки виден только по ним). Рамка виновника — область, по которой
страница получила свой score, — рисуется на обеих панелях. Третья панель по запросу: стрелки
поля смещений (остаток после аффинной части, ×5) и тайлы без пары, чтобы видеть, ЧТО именно
FineReader сделал со страницей.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GUIDES = 12
GAP_PX = 12
CAPTION_PX = 36
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
GRID_COLOUR = (255, 0, 0)
CULPRIT_COLOUR = (0, 90, 255)
ARROW_SCALE = 5.0


def _font(size: int = 20) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _panel(gray: np.ndarray, boxes: list, height: int) -> Image.Image:
    image = Image.fromarray(gray).convert("RGB")
    if image.height != height:
        image = image.resize((max(1, round(image.width * height / image.height)), height), Image.LANCZOS)
    draw = ImageDraw.Draw(image)
    for step in range(1, GUIDES):
        y = image.height * step // GUIDES
        x = image.width * step // GUIDES
        draw.line([(0, y), (image.width, y)], fill=GRID_COLOUR, width=1)
        draw.line([(x, 0), (x, image.height)], fill=GRID_COLOUR, width=1)
    k = height / gray.shape[0]
    for x0, y0, x1, y1 in boxes:
        draw.rectangle([x0 * k - 6, y0 * k - 6, x1 * k + 6, y1 * k + 6], outline=CULPRIT_COLOUR, width=3)
    return image


def _field_panel(gray: np.ndarray, field_raw: dict | None, height: int) -> Image.Image:
    image = Image.fromarray(gray).convert("RGB")
    draw = ImageDraw.Draw(image)
    if field_raw:
        affine = np.array(field_raw["affine"])
        for (cx, cy, ux, uy, _), weight in zip(field_raw["tiles"], field_raw["weight"]):
            predicted = affine @ np.array([cx, cy, 1.0]) - np.array([cx, cy])
            rx, ry = ux - predicted[0], uy - predicted[1]
            colour = (220, 0, 0) if weight > 0 else (255, 160, 0)
            draw.line([(cx, cy), (cx + rx * ARROW_SCALE, cy + ry * ARROW_SCALE)], fill=colour, width=2)
            draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(0, 0, 255))
        for cx, cy, kind in field_raw.get("failed", []):
            colour = (200, 0, 200) if kind == 0 else (120, 120, 120)
            draw.rectangle([cx - 8, cy - 8, cx + 8, cy + 8], outline=colour, width=2)
    if image.height != height:
        image = image.resize((max(1, round(image.width * height / image.height)), height), Image.LANCZOS)
    return image


def pair_image(
    before: np.ndarray,
    after: np.ndarray,
    culprit: dict | None,
    caption_before: str,
    caption_after: str,
    field_raw: dict | None = None,
    height: int | None = None,
) -> Image.Image:
    """Панели B | A (| поле) одной высоты с сеткой, рамками виновника и подписями.

    ``culprit`` — ``{"b": box, "a": box}`` в пикселях серых копий (или None); ``height`` —
    высота панелей (по умолчанию высота B).
    """
    height = height or before.shape[0]
    panels = [
        _panel(before, [culprit["b"]] if culprit else [], height),
        _panel(after, [culprit["a"]] if culprit else [], height),
    ]
    captions = [caption_before, caption_after]
    if field_raw is not None:
        panels.append(_field_panel(before, field_raw, height))
        captions.append("поле смещений: остаток ×5, без пары: лиловый — слабый пик, серый — неотчётливый")
    width = sum(p.width for p in panels) + GAP_PX * (len(panels) - 1)
    canvas = Image.new("RGB", (width, height + CAPTION_PX), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _font()
    x = 0
    for panel, caption in zip(panels, captions):
        canvas.paste(panel, (x, CAPTION_PX))
        draw.text((x + 6, 8), caption, fill=(0, 0, 0), font=font)
        x += panel.width
        if x < width:
            draw.rectangle([x, CAPTION_PX, x + GAP_PX - 1, height + CAPTION_PX], fill=(120, 120, 120))
        x += GAP_PX
    return canvas
