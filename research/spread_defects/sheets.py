"""Листы для отсмотра глазами: крупы кадров (у сгиба, у зоны смаза, целиком) сеткой с подписями."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from research.spread_defects.frames import read_rgb

# Шрифт подписей с кириллицей.
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
# Цвет подписи (RGB) и подложки под неё.
LABEL_COLOR = (255, 255, 255)
LABEL_BACK = (40, 40, 40)
# Высота полосы подписи, px листа.
LABEL_HEIGHT = 34
# Разделитель между ячейками листа, px: без него соседние крупы текста сливаются в один.
GAP = 16
GAP_COLOR = (120, 120, 120)


@dataclass(frozen=True)
class Crop:
    """Заказ одного крупа на лист.

    Attributes:
        path: Файл кадра.
        box: (x0, y0, x1, y1) в px полного кадра; None — весь кадр.
        label: Подпись под крупом.
        marks: Вертикали для отрисовки (x в px полного кадра) — например, сгиб.
    """

    path: Path
    box: tuple[int, int, int, int] | None
    label: str
    marks: tuple[float, ...] = ()


def render_crop(crop: Crop, cell: tuple[int, int]) -> np.ndarray:
    """Рисует один круп в ячейку листа с подписью.

    Args:
        crop: Заказ крупа.
        cell: (ширина, высота) ячейки без подписи, px.

    Returns:
        Ячейка RGB uint8 (высота + ``LABEL_HEIGHT``).
    """
    rgb = read_rgb(crop.path)
    x0, y0, x1, y1 = crop.box if crop.box else (0, 0, rgb.shape[1], rgb.shape[0])
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(rgb.shape[1], x1), min(rgb.shape[0], y1)
    part = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    # Отметки (сгиб) — тонкая полупрозрачная красная линия, чтобы не закрывать текст.
    for x in crop.marks:
        xc = int(x) - x0
        if 0 <= xc < part.shape[1]:
            band = part[:, max(0, xc - 1) : xc + 2].astype(np.float32)
            part[:, max(0, xc - 1) : xc + 2] = (0.5 * band + 0.5 * np.array([230, 30, 30])).astype(np.uint8)
    scale = min(cell[0] / part.shape[1], cell[1] / part.shape[0])
    resized = cv2.resize(part, (int(part.shape[1] * scale), int(part.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    out = np.full((cell[1] + LABEL_HEIGHT, cell[0], 3), 255, np.uint8)
    out[: resized.shape[0], : resized.shape[1]] = resized
    out[cell[1] :] = LABEL_BACK
    return draw_label(out, crop.label, (6, cell[1] + 6))


def draw_label(image: np.ndarray, text: str, origin: tuple[int, int]) -> np.ndarray:
    """Подпись с кириллицей (PIL, шрифт DejaVu Sans): ``cv2.putText`` кириллицу не рисует.

    Args:
        image: Картинка RGB uint8.
        text: Подпись.
        origin: Левый верхний угол текста, px.

    Returns:
        Картинка с подписью.
    """
    canvas = Image.fromarray(image)
    ImageDraw.Draw(canvas).text(origin, text, font=_font(), fill=LABEL_COLOR)
    return np.asarray(canvas)


@lru_cache(maxsize=1)
def _font() -> ImageFont.FreeTypeFont:
    """Шрифт подписей листов (DejaVu Sans 18 px; при отсутствии — встроенный PIL)."""
    try:
        return ImageFont.truetype(FONT_PATH, 18)
    except OSError:
        return ImageFont.load_default()


def contact_sheet(crops: list[Crop], target: Path, columns: int = 2, cell: tuple[int, int] = (1000, 700)) -> Path:
    """Собирает лист из крупов сеткой и пишет JPEG.

    Args:
        crops: Заказы крупов по порядку (слева направо, сверху вниз).
        target: Файл листа (папка создаётся).
        columns: Крупов в ряду.
        cell: (ширина, высота) ячейки без подписи, px.

    Returns:
        Путь записанного листа.
    """
    cells = [render_crop(c, cell) for c in crops]
    blank = np.full_like(cells[0], 255)
    vgap = np.full((cells[0].shape[0], GAP, 3), GAP_COLOR, np.uint8)
    rows = []
    for start in range(0, len(cells), columns):
        row = cells[start : start + columns]
        row += [blank] * (columns - len(row))
        joined = [row[0]]
        for item in row[1:]:
            joined += [vgap, item]
        rows.append(np.hstack(joined))
    hgap = np.full((GAP, rows[0].shape[1], 3), GAP_COLOR, np.uint8)
    stacked = [rows[0]]
    for item in rows[1:]:
        stacked += [hgap, item]
    sheet = np.vstack(stacked)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return target
