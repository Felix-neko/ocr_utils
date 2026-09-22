"""Картинка «было | стало» для просмотра находок глазами.

Слева страница без коррекции (B), справа с коррекцией (A), обе с реперной сеткой через 1/12
ширины и высоты (по образцу ``curved_lines.finereader_compare.compose``, но и с вертикалями:
наклон линейки или кромки колонки виден только по ним). Рамка виновника — область, по которой
страница получила свой score, — рисуется на обеих панелях. Третья панель по запросу: стрелки
поля смещений (остаток после аффинной части, ×5) и тайлы без пары, чтобы видеть, ЧТО именно
FineReader сделал со страницей.

Кромка блока (метрики ``edge_*`` стенда v15) рисуется не ломаной по сырым краям строк, а так,
как её видит расчёт: кружки — края строк, вошедших в кромку (те самые точки), прямая — подгонка
Тейла–Сена по ним (та же функция, что в ``blocks.py``), продлённая на высоту блока. Ломаная по
сырым точкам проваливалась в отступы и вихляла на каждой строке, хотя подгонка это сглаживает.
"""

from __future__ import annotations

import numpy as np
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_utils.geometry_regression.edges import _theil_sen

GUIDES = 12
GAP_PX = 12
CAPTION_PX = 36
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
GRID_COLOUR = (255, 0, 0)
CULPRIT_COLOUR = (0, 90, 255)
POINT_COLOUR = (255, 140, 0)
ARROW_SCALE = 5.0
# Метрики, у которых ``segments_*`` виновника — ряд точек (края строк кромки), а не отрезки.
POINT_METRICS = ("edge_shear_delta_mm", "edge_rough_delta_mm")
POINT_RADIUS = 5


def _font(size: int = 20) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _points_of(segments: list) -> np.ndarray:
    """Вершины ломаной ``segments`` (N отрезков → N + 1 точек)."""
    points = [(x0, y0) for x0, y0, _, _ in segments]
    points.append((segments[-1][2], segments[-1][3]))
    return np.array(points, dtype=np.float64)


def _draw_edge_fit(draw: ImageDraw.ImageDraw, segments: list, k: float) -> None:
    """Кружки по краям строк кромки и прямая Тейла–Сена x = s·y + c по ним, как в ``blocks.py``."""
    points = _points_of(segments)
    xs, ys = points[:, 0], points[:, 1]
    if len(points) >= 2:
        slope = _theil_sen(xs, ys)
        intercept = float(np.median(xs - slope * ys))
        y0, y1 = float(ys.min()), float(ys.max())
        draw.line(
            [((slope * y0 + intercept) * k, y0 * k), ((slope * y1 + intercept) * k, y1 * k)],
            fill=CULPRIT_COLOUR,
            width=4,
        )
    for x, y in points:
        draw.ellipse(
            [x * k - POINT_RADIUS, y * k - POINT_RADIUS, x * k + POINT_RADIUS, y * k + POINT_RADIUS],
            outline=POINT_COLOUR,
            width=2,
        )


def _panel(
    gray: np.ndarray, boxes: list, height: int, segments: list | None = None, points: bool = False
) -> Image.Image:
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
    if segments and points:
        _draw_edge_fit(draw, segments, k)
    elif segments:
        # Группа параллельных штрихов: сами отрезки, а не рамка вокруг всей группы.
        for x0, y0, x1, y1 in segments:
            draw.line([(x0 * k, y0 * k), (x1 * k, y1 * k)], fill=CULPRIT_COLOUR, width=4)
    else:
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

    ``culprit`` — ``{"b": box, "a": box}`` в пикселях серых копий (или None), с необязательными
    ``segments_b/a`` (отрезки или, для метрик из ``POINT_METRICS``, ряд точек) и ``metric`` — имя
    метрики, давшей вердикт; ``height`` — высота панелей (по умолчанию высота B).
    """
    height = height or before.shape[0]
    points = bool(culprit) and culprit.get("metric") in POINT_METRICS
    panels = [
        _panel(
            before, [culprit["b"]] if culprit else [], height, culprit.get("segments_b") if culprit else None, points
        ),
        _panel(
            after, [culprit["a"]] if culprit else [], height, culprit.get("segments_a") if culprit else None, points
        ),
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
