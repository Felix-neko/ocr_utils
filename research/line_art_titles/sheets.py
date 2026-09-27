"""Контактные листы вырезок line art по поясам признака: слова tesseract поверх, номер и числа под каждой."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from research.line_art_titles.features import is_good_word

# Клетка листа: сторона картинки и высота подписи, px.
CELL = 420
CAPTION = 44
COLUMNS = 5
ROWS = 4

# Цвета (BGR) — палитра проекта (навык draw-overlay): принято / отвергнуто, подсказка — рамка области.
COLOR_GOOD = (0, 150, 0)
COLOR_BAD = (0, 0, 220)
COLOR_BOX = (200, 140, 60)
COLOR_TEXT = (20, 20, 20)

# Шрифт подписей с кириллицей.
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Прозрачность заливки уверенных слов: буквы под ней должны читаться.
WORD_ALPHA = 0.4


def _on_paper(color: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    """Цвет полупрозрачной заливки на белой бумаге — для образца в легенде."""
    return tuple(int(round(alpha * c + (1 - alpha) * 255)) for c in color)


def draw_cell(gray: np.ndarray, row: dict, words: list[dict], number: int, caption: str) -> np.ndarray:
    """Одна клетка листа: вырезка, рамка области, слова (уверенные — заливкой, прочие — контуром), подпись.

    Args:
        gray: Серая вырезка.
        row: Строка ``regions.jsonl`` (нужна ``crop_inner``).
        words: Слова лучшего режима tesseract.
        number: Номер клетки на листе — по нему ведётся ручная разметка.
        caption: Числа под картинкой.

    Returns:
        Клетка BGR размером ``CELL`` × (``CELL`` + ``CAPTION``).
    """
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    layer = canvas.copy()
    for word in words:
        if is_good_word(word):
            cv2.rectangle(layer, (word["x0"], word["y0"]), (word["x1"], word["y1"]), COLOR_GOOD, -1)
    cv2.addWeighted(layer, WORD_ALPHA, canvas, 1 - WORD_ALPHA, 0, canvas)
    for word in words:
        if not is_good_word(word):
            cv2.rectangle(canvas, (word["x0"], word["y0"]), (word["x1"], word["y1"]), COLOR_BAD, 1)
    x0, y0, x1, y1 = row["crop_inner"]
    cv2.rectangle(canvas, (x0, y0), (x1, y1), COLOR_BOX, 1)
    scale = min(CELL / canvas.shape[1], CELL / canvas.shape[0])
    small = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    cell = np.full((CELL + CAPTION, CELL, 3), 255, np.uint8)
    cell[: small.shape[0], : small.shape[1]] = small
    cv2.rectangle(cell, (0, 0), (CELL - 1, CELL + CAPTION - 1), (200, 200, 200), 1)
    _put(cell, f"#{number} {caption}", (4, CELL + 18))
    _put(cell, row["page"], (4, CELL + 38), 0.45)
    return cell


def _put(image: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.5) -> None:
    """Подпись кириллицей шрифтом DejaVu (шрифты OpenCV кириллицу не рисуют); ``origin`` — левый нижний угол."""
    size = int(round(32 * scale))
    pil = Image.fromarray(image)
    ImageDraw.Draw(pil).text((origin[0], origin[1] - size), text, font=_font(size), fill=COLOR_TEXT[::-1])
    image[:] = np.asarray(pil)


def _font(size: int):
    """Шрифт DejaVu Sans нужного кегля (или встроенный PIL, если DejaVu нет)."""
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def legend(width: int) -> np.ndarray:
    """Полоса легенды над листом; образцы заливки — с той же прозрачностью, что на вырезках."""
    strip = np.full((34, width, 3), 255, np.uint8)
    items = [
        (_on_paper(COLOR_GOOD, WORD_ALPHA), True, "уверенное слово (conf ≥ 60, буквы)"),
        (COLOR_BAD, False, "прочее слово tesseract"),
        (COLOR_BOX, False, "рамка области line art"),
    ]
    x = 8
    for color, filled, label in items:
        cv2.rectangle(strip, (x, 9), (x + 26, 25), color, -1 if filled else 2)
        _put(strip, label, (x + 34, 23))
        x += 60 + 10 * len(label)
    return strip


def write_sheets(items: list[tuple[np.ndarray, dict, list[dict], str]], out_dir: Path, prefix: str) -> list[Path]:
    """Разложить клетки по листам ``COLUMNS`` × ``ROWS``.

    Args:
        items: Четвёрки (вырезка, строка regions.jsonl, слова, подпись) в порядке листа.
        out_dir: Куда писать JPEG.
        prefix: Начало имени файла листа (пояс признака).

    Returns:
        Пути записанных листов; номер клетки на листе = порядковый номер в ``items`` с единицы.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sheet = COLUMNS * ROWS
    paths = []
    for start in range(0, len(items), per_sheet):
        chunk = items[start : start + per_sheet]
        cells = [draw_cell(g, r, w, start + i + 1, c) for i, (g, r, w, c) in enumerate(chunk)]
        blank = np.full_like(cells[0], 255)
        cells += [blank] * (per_sheet - len(cells))
        rows = [np.hstack(cells[i * COLUMNS : (i + 1) * COLUMNS]) for i in range(ROWS)]
        sheet = np.vstack([legend(COLUMNS * CELL)] + rows)
        path = out_dir / f"{prefix}_{start // per_sheet + 1:02d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(path)
    return paths
