"""Отчёты: CSV, markdown и контактные листы.

Формат тот же, что у остальных исследовательских пакетов проекта: тяжёлая команда пишет
CSV, а дешёвые отчёты собираются из него без повторного прогона. Это не украшательство —
пороги калибруются итеративно, и пересчитывать полосы ради новой таблицы в отчёте нельзя.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Плитка контактного листа и сколько их на лист. Числа подобраны под экран: на листе
# 5x4 плитки по 420 px читаются заголовки таблиц, а сам лист остаётся меньше 10 МБ.
SHEET_TILE_PX = 420
SHEET_COLUMNS = 5
SHEET_ROWS = 4
CAPTION_PX = 22

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
)


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Простая markdown-таблица без выравнивания по ширине: её читают в готовом виде."""
    lines = ["| " + " | ".join(str(cell) for cell in header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def contact_sheets(images: Sequence[tuple[str, np.ndarray]], out_dir: Path, prefix: str) -> list[Path]:
    """Контактные листы: подписанные плитки, по ``SHEET_COLUMNS x SHEET_ROWS`` на лист."""
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sheet = SHEET_COLUMNS * SHEET_ROWS
    written: list[Path] = []
    font = load_font(14)
    for sheet_index in range(0, max(1, (len(images) + per_sheet - 1) // per_sheet)):
        chunk = images[sheet_index * per_sheet : (sheet_index + 1) * per_sheet]
        if not chunk:
            break
        canvas = Image.new(
            "RGB", (SHEET_COLUMNS * SHEET_TILE_PX, SHEET_ROWS * (SHEET_TILE_PX + CAPTION_PX)), (255, 255, 255)
        )
        draw = ImageDraw.Draw(canvas)
        for position, (caption, gray) in enumerate(chunk):
            tile = Image.fromarray(gray).convert("L")
            tile.thumbnail((SHEET_TILE_PX, SHEET_TILE_PX), Image.LANCZOS)
            x = (position % SHEET_COLUMNS) * SHEET_TILE_PX
            y = (position // SHEET_COLUMNS) * (SHEET_TILE_PX + CAPTION_PX)
            canvas.paste(tile, (x, y + CAPTION_PX))
            draw.text((x + 3, y + 3), caption[:60], fill=(180, 0, 0), font=font)
        path = out_dir / f"{prefix}_sheet_{sheet_index + 1:02d}.png"
        canvas.save(path)
        written.append(path)
    return written


def downscale(gray: np.ndarray, long_side: int) -> np.ndarray:
    scale = long_side / max(gray.shape[:2])
    if scale >= 1.0:
        return gray
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
