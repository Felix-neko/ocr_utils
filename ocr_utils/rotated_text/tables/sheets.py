"""Листы «было — стало»: проверка глазами того, ради чего всё и затевалось.

СЛЕВА — исходная вырезка с сеткой: границы ячеек зелёным, ячейки с повёрнутым текстом,
которые набираются заново, — красным (с углом поворота в углу), ячейки, где текст повёрнут,
но подменять его нечем (мало букв), — оранжевым. СПРАВА — результат. Панели приводятся к
одной ширине, а не высоте: после поворота таблицы или расширения колонок пропорции
меняются, и общая ширина оставляет обе панели читаемыми.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ocr_utils.scan_markup.table_detection.geometry import Grid

from ocr_utils.rotated_text.tables.fit import load_font
from ocr_utils.rotated_text.tables.pipeline import CellRecord

GREEN = (0, 150, 0)
RED = (210, 0, 0)
ORANGE = (230, 130, 0)
GAP_PX = 12
CAPTION_PX = 28
BACKGROUND = (255, 255, 255)


def overlay_before(gray: np.ndarray, grid: Grid, cells: Sequence[CellRecord], line_px: int = 2) -> np.ndarray:
    """Исходная вырезка с сеткой и выделенными повёрнутыми ячейками."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    by_key = {cell.key: cell for cell in grid.cells}
    thick = max(1, line_px * 3)
    for record in cells:
        cell = by_key.get(record.key)
        if cell is None:
            continue
        box = cell.box
        colour = None
        # После поворота таблицы записи хранят новый угол; «было» показывает прежний.
        angle = record.rotate_before if record.rotate_before is not None else record.rotate_cw
        if record.candidate or (record.replaced and record.rotate_before is None):
            colour = RED
            patch = rgb[box.y0 : box.y1, box.x0 : box.x1]
            tint = np.zeros_like(patch)
            tint[..., 0] = 255
            rgb[box.y0 : box.y1, box.x0 : box.x1] = cv2.addWeighted(patch, 0.82, tint, 0.18, 0)
        elif angle not in (None, 0):
            colour = ORANGE
        cv2.rectangle(rgb, (box.x0, box.y0), (box.x1 - 1, box.y1 - 1), GREEN, line_px)
        if colour is not None:
            cv2.rectangle(rgb, (box.x0, box.y0), (box.x1 - 1, box.y1 - 1), colour, thick)
            label = str(angle)
            font_scale = max(0.6, line_px * 0.5)
            cv2.putText(
                rgb,
                label,
                (box.x0 + thick + 2, box.y0 + thick + int(20 * font_scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                colour,
                max(1, line_px),
            )
    return rgb


def _fit_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.LANCZOS)


def pair_image(
    before: np.ndarray, after: np.ndarray, caption_left: str, caption_right: str, panel_width: int = 1400
) -> Image.Image:
    """Две панели одной ширины с подписями сверху."""
    left = _fit_width(Image.fromarray(before).convert("RGB"), panel_width)
    right = _fit_width(Image.fromarray(after).convert("RGB"), panel_width)
    height = max(left.height, right.height)
    canvas = Image.new("RGB", (panel_width * 2 + GAP_PX, height + CAPTION_PX), BACKGROUND)
    canvas.paste(left, (0, CAPTION_PX))
    canvas.paste(right, (panel_width + GAP_PX, CAPTION_PX))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([panel_width, CAPTION_PX, panel_width + GAP_PX - 1, height + CAPTION_PX], fill=(150, 150, 150))
    font = load_font(18)
    draw.text((4, 5), caption_left, fill=(160, 0, 0), font=font)
    draw.text((panel_width + GAP_PX + 4, 5), caption_right, fill=(0, 110, 0), font=font)
    return canvas


def build_sheets(
    pairs: Sequence[Path], out_dir: Path, prefix: str = "sheet", per_sheet: int = 4, width_px: int = 2000
) -> list[Path]:
    """Собрать пары в листы по несколько штук, чтобы листать, а не открывать по одной."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index in range(0, len(pairs), per_sheet):
        chunk = [Image.open(path).convert("RGB") for path in pairs[index : index + per_sheet]]
        scaled = [_fit_width(image, min(width_px, image.width)) for image in chunk]
        canvas = Image.new("RGB", (max(i.width for i in scaled), sum(i.height + 8 for i in scaled)), BACKGROUND)
        top = 0
        for image in scaled:
            canvas.paste(image, (0, top))
            top += image.height + 8
        path = out_dir / f"{prefix}_{index // per_sheet + 1:02d}.png"
        canvas.save(path)
        written.append(path)
    return written
