"""Листы кандидатов для разметки глазами: плитка — серый 300 dpi без утолщения вокруг кандидата (рамка вокруг компоненты, за краем вырезки — серое поле), подпись с номером и признаками в поле под плиткой."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from ocr_utils.page_layout.overlay_frame import FONT_PATH, LegendEntry, SampleStyle, framed
from ocr_utils.page_layout.text_blocks import RENDER_DPI
from ocr_utils.units import mm_to_px
from research.pseudo_glyphs.crops import HP1991_CANVAS, load_crop

# Поле вокруг кандидата, мм: по ширине и по высоте.
CONTEXT_X_MM, CONTEXT_Y_MM = 9.0, 4.0
# Окно плитки по высоте не больше стольких мм (высокие полосы тени ужимаются целиком).
MAX_WINDOW_MM = 60.0
# Размер плитки, px, и высота подписи под ней.
TILE_W, TILE_H, CAPTION_H = 340, 240, 34
# Цвет рамки кандидата (BGR) и поля за краем вырезки (серый — чтобы край картинки был виден).
COLOUR_BOX = (0, 0, 220)
OUTSIDE_GRAY = 170
# Плиток в ряду листа.
COLUMNS = 5


def raw300_of(png: Path) -> np.ndarray:
    """Серый 300 dpi вырезки без утолщения (как ``crops.prepare().raw300``, но без гашения линеек)."""
    bgr, crop = load_crop(png)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    f = RENDER_DPI / crop.dpi
    return cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


def crop_png(canvas_root: Path, key: str) -> Path:
    """PNG вырезки по ключу ``<номер выпуска>/<кадр>_<часть>``."""
    issue, stem = key.split("/")
    folder = next(p for p in canvas_root.iterdir() if p.name.split()[0] == issue)
    return folder / f"{stem}.png"


def tile(raw300: np.ndarray, row: pd.Series, caption: str) -> np.ndarray:
    """Плитка одного кандидата с подписью.

    Args:
        raw300: Серый 300 dpi страницы.
        row: Кандидат (``x0, y0, x1, y1`` — пиксели 300 dpi).
        caption: Подпись (номер и признаки).

    Returns:
        BGR ``(TILE_H + CAPTION_H, TILE_W)``.
    """
    x0, y0, x1, y1 = int(row.x0), int(row.y0), int(row.x1), int(row.y1)
    mx, my = mm_to_px(CONTEXT_X_MM, RENDER_DPI), mm_to_px(CONTEXT_Y_MM, RENDER_DPI)
    cy = (y0 + y1) // 2
    half = min(max((y1 - y0) // 2 + my, my), mm_to_px(MAX_WINDOW_MM, RENDER_DPI) // 2)
    wx0, wx1, wy0, wy1 = x0 - mx, x1 + mx, cy - half, cy + half
    # Окно с серым полем за краем картинки.
    height, width = raw300.shape
    window = np.full((wy1 - wy0, wx1 - wx0), OUTSIDE_GRAY, np.uint8)
    sx0, sy0, sx1, sy1 = max(wx0, 0), max(wy0, 0), min(wx1, width), min(wy1, height)
    window[sy0 - wy0 : sy1 - wy0, sx0 - wx0 : sx1 - wx0] = raw300[sy0:sy1, sx0:sx1]
    scale = min(TILE_W / window.shape[1], TILE_H / window.shape[0])
    view = cv2.resize(window, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    # Рамка вокруг компоненты с отступом в 3 px плитки — краску не закрывает.
    bx0, by0 = int((x0 - wx0) * scale) - 3, int((y0 - wy0) * scale) - 3
    bx1, by1 = int((x1 - wx0) * scale) + 3, int((y1 - wy0) * scale) + 3
    cv2.rectangle(view, (bx0, by0), (bx1, by1), COLOUR_BOX, 1)
    out = np.full((TILE_H + CAPTION_H, TILE_W, 3), 255, np.uint8)
    out[: view.shape[0], : view.shape[1]] = view
    picture = Image.fromarray(out[:, :, ::-1])
    draw = ImageDraw.Draw(picture)
    from PIL import ImageFont

    font = ImageFont.truetype(FONT_PATH, 13)
    draw.text((4, TILE_H + 2), caption, font=font, fill=(20, 20, 20))
    return np.asarray(picture)[:, :, ::-1].copy()


def caption_of(number: int, row: pd.Series) -> str:
    """Подпись плитки: номер на листе, вид кандидата и главные признаки в две строки."""
    kind = {"row_left": "L", "row_right": "R", "edge": "E"}.get(row.kind, "?")
    first = f"{number} {kind} h{row.h_mm:.1f} w{row.w_mm:.1f} край{row.edge_mm:.1f}"
    gap = "" if pd.isna(row.gap_rel) else f" отр{row.gap_rel:.2f}"
    hb = "" if pd.isna(row.h_rel_block) else f" hb{row.h_rel_block:.2f}"
    return first + f"\nтон{row.tone_rel:.2f} шт{row.stroke_mm:.2f}{hb}{gap}"


def write_sheets(frame: pd.DataFrame, out_dir: Path, title: str, canvas_root: Path = HP1991_CANVAS,
                 per_sheet: int = 40) -> list[Path]:  # fmt: skip
    """Листы по ``per_sheet`` плиток и опись ``index.csv`` (номер на листе → ``cid``).

    Args:
        frame: Кандидаты по порядку показа.
        out_dir: Папка листов.
        title: Заголовок листов.
        canvas_root: Корень вырезок.
        per_sheet: Плиток на лист.

    Returns:
        Пути листов.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles: dict[str, np.ndarray] = {}
    # Плитки — по страницам, чтобы каждую вырезку читать один раз.
    for key, part in frame.groupby("key"):
        raw = raw300_of(crop_png(canvas_root, key))
        for cid, row in part.iterrows():
            tiles[cid] = (raw, row)
    paths, index = [], []
    ids = list(frame.index)
    for sheet_no, start in enumerate(range(0, len(ids), per_sheet), 1):
        chunk = ids[start : start + per_sheet]
        rendered = []
        for offset, cid in enumerate(chunk):
            number = start + offset + 1
            raw, row = tiles[cid]
            rendered.append(tile(raw, row, caption_of(number, row)))
            index.append({"number": number, "cid": row.cid, "sheet": sheet_no})
        while len(rendered) % COLUMNS:
            rendered.append(np.full_like(rendered[0], 255))
        rows = [np.hstack(rendered[i : i + COLUMNS]) for i in range(0, len(rendered), COLUMNS)]
        body = np.vstack(rows)
        picture = framed(
            body,
            [f"{title} — лист {sheet_no}, кандидаты {start + 1}–{start + len(chunk)}",
             "L/R — крайняя компонента ряда слева/справа, E — у края вырезки; h, w, край — мм; отр — отрыв / p99 "
             "пробела; тон — 0 краска, 1 бумага; шт — штрих, мм; hb — высота / p95 букв блока"],  # fmt: skip
            [LegendEntry("рамка кандидата", COLOUR_BOX, style=SampleStyle.BOX),
             LegendEntry("за краем вырезки", (OUTSIDE_GRAY,) * 3, style=SampleStyle.BOX)],  # fmt: skip
        )
        path = out_dir / f"sheet_{sheet_no:03d}.jpg"
        cv2.imwrite(str(path), picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(path)
    pd.DataFrame(index).to_csv(out_dir / "index.csv", index=False)
    return paths


__all__ = ["caption_of", "crop_png", "raw300_of", "tile", "write_sheets"]
