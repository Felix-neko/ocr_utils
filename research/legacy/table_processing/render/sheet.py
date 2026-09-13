"""Листы «было — стало»: то, ради чего всё и затевалось.

Панели ставятся рядом и приводятся к общей высоте, между ними серая полоса. Реперных
горизонталей, как в ``curved_lines.finereader_compare``, здесь нет намеренно: там сравнивали
ГЕОМЕТРИЮ и надо было видеть, куда уехала строка, а здесь сравнивают ТЕКСТ, и линейки
поперёк только мешают.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from research.legacy.table_processing.report import load_font

GAP_PX = 12
CAPTION_PX = 26
BACKGROUND = (255, 255, 255)


def pair(before: np.ndarray, after: np.ndarray, caption_left: str, caption_right: str) -> Image.Image:
    """Две панели одной высоты с подписями сверху."""
    left = Image.fromarray(before).convert("RGB")
    right = Image.fromarray(after).convert("RGB")
    height = max(left.height, right.height)
    for image_name in ("left", "right"):
        image = left if image_name == "left" else right
        if image.height != height:
            resized = image.resize((max(1, round(image.width * height / image.height)), height), Image.LANCZOS)
            if image_name == "left":
                left = resized
            else:
                right = resized

    canvas = Image.new("RGB", (left.width + GAP_PX + right.width, height + CAPTION_PX), BACKGROUND)
    canvas.paste(left, (0, CAPTION_PX))
    canvas.paste(right, (left.width + GAP_PX, CAPTION_PX))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([left.width, CAPTION_PX, left.width + GAP_PX - 1, height + CAPTION_PX], fill=(150, 150, 150))
    font = load_font(16)
    draw.text((4, 5), caption_left, fill=(160, 0, 0), font=font)
    draw.text((left.width + GAP_PX + 4, 5), caption_right, fill=(0, 110, 0), font=font)
    return canvas


def write_pair(before: np.ndarray, after: np.ndarray, caption: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pair(before, after, f"было: {caption}", f"стало: {caption}").save(path)


def sheets(pairs: Sequence[Path], out_dir: Path, prefix: str, per_sheet: int = 4, width_px: int = 2000) -> list[Path]:
    """Собрать готовые пары в листы по несколько штук — чтобы листать, а не открывать по одной."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index in range(0, len(pairs), per_sheet):
        chunk = [Image.open(path).convert("RGB") for path in pairs[index : index + per_sheet]]
        scaled = []
        for image in chunk:
            scale = min(1.0, width_px / image.width)
            scaled.append(image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS))
        canvas = Image.new("RGB", (max(i.width for i in scaled), sum(i.height + 8 for i in scaled)), BACKGROUND)
        top = 0
        for image in scaled:
            canvas.paste(image, (0, top))
            top += image.height + 8
        path = out_dir / f"{prefix}_sheet_{index // per_sheet + 1:02d}.png"
        canvas.save(path)
        written.append(path)
    return written
