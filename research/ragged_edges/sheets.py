"""Листы отсмотра дальних сторон: вырезка вдоль стороны с контуром сырой маски SAM 3 и итоговой страницы, номер плитки
для разметки; плитки по поясам меры."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ocr_utils.page_layout.overlay_frame import LegendEntry, framed
from ocr_utils.scan_cropping.v2.page_pose import SEG_PX_PER_MM
from ocr_utils.scan_cropping.v2.page_sides import Side, quad_sides
from research.ragged_edges.features import load_frame, stage_page_mask

# Вырезка вдоль дальней стороны: столько наружу и внутрь от стороны трапеции, мм; запас по высоте, мм.
OUT_MM, IN_MM, PAD_MM = 15.0, 25.0, 5.0
TILE_PX_PER_MM = 4.0  # масштаб плитки на листе, px/мм (сегментация — 6.9): сторона ~230 мм — ~920 px
LINE_PX = 2  # толщина контуров на плитке
COLS, ROWS = 8, 1  # плиток на листе
RAW_COLOUR = (0, 0, 220)  # контур сырой маски SAM 3 (то, что меряет детектор)
PAPER_COLOUR = (220, 90, 20)  # контур итоговой страницы


def side_tile(work: Path, page: str, number: int, caption: str) -> np.ndarray:
    """Плитка дальней стороны страницы кадра кэша.

    Args:
        work: Папка кадра в кэше v2.
        page: ``L`` — левая страница (сторона слева), ``R`` — правая.
        number: Номер плитки (для ответа разметкой).
        caption: Подпись под номером (ключ, мера).

    Returns:
        Плитка BGR в масштабе :data:`TILE_PX_PER_MM` с шапкой.
    """
    frame = load_frame(work)
    index = 0 if page == "L" else 1
    quad = frame.pages[index].quad
    a, b = quad_sides(quad, index == 0)[Side.FAR]
    mm = SEG_PX_PER_MM
    xs = (a[0], b[0])
    # У левой страницы наружу — влево, у правой — вправо.
    x0, x1 = (
        (min(xs) - OUT_MM * mm, max(xs) + IN_MM * mm) if index == 0 else (min(xs) - IN_MM * mm, max(xs) + OUT_MM * mm)
    )
    h, w = frame.img.shape[:2]
    x0, x1 = int(max(0, x0)), int(min(w, x1))
    y0, y1 = int(max(0, min(a[1], b[1]) - PAD_MM * mm)), int(min(h, max(a[1], b[1]) + PAD_MM * mm))
    s = TILE_PX_PER_MM / mm
    crop = cv2.resize(frame.img[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    # Контуры — по уменьшенной вырезке, линией LINE_PX: в 1 px после уменьшения их не видно.
    for mask, colour in ((frame.pages[index].mask, PAPER_COLOUR), (stage_page_mask(frame, "raw", index), RAW_COLOUR)):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        moved = [
            np.round((c[:, 0, :].astype(np.float64) - (x0, y0)) * s).astype(np.int32)[:, None, :] for c in contours
        ]
        cv2.drawContours(crop, moved, -1, colour, LINE_PX)
    head = np.full((46, crop.shape[1], 3), 255, np.uint8)
    cv2.putText(head, str(number), (4, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(head, caption, (4, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([head, crop])


def write_sheets(
    tiles: list[np.ndarray], out: Path, prefix: str, title: str, legend: list[LegendEntry] | None = None
) -> list[Path]:
    """Листы из плиток по :data:`COLS` × :data:`ROWS`.

    Args:
        tiles: Плитки (номера уже напечатаны).
        out: Папка листов.
        prefix: Начало имени файла листа.
        title: Шапка листа.
        legend: Легенда; None — контуры плиток разметки (сырая маска SAM 3 и итоговая страница).

    Returns:
        Пути листов.
    """
    out.mkdir(parents=True, exist_ok=True)
    if legend is None:
        legend = [LegendEntry("сырая маска SAM 3 страницы (её меряет детектор)", RAW_COLOUR),
                  LegendEntry("итоговая страница v2", PAPER_COLOUR)]  # fmt: skip
    per = COLS * ROWS
    paths = []
    for k in range(0, len(tiles), per):
        part = tiles[k : k + per]
        width, height = max(t.shape[1] for t in part), max(t.shape[0] for t in part)
        # Плитки разной высоты и ширины (длина стороны, наклон) — добить белым до общей рамки.
        cells = [
            np.pad(t, ((0, height - t.shape[0]), (0, width - t.shape[1]), (0, 0)), constant_values=255) for t in part
        ]
        cells += [np.full_like(cells[0], 255)] * (-len(cells) % COLS)
        rows = [np.hstack(sum(([c, np.full((c.shape[0], 6, 3), 60, np.uint8)] for c in cells[i : i + COLS]), [])[:-1])
                for i in range(0, len(cells), COLS)]  # fmt: skip
        pic = np.vstack(sum(([r, np.full((8, r.shape[1], 3), 255, np.uint8)] for r in rows), [])[:-1])
        path = out / f"{prefix}_{k // per + 1:02d}.jpg"
        cv2.imwrite(str(path), framed(pic, [title], legend), [cv2.IMWRITE_JPEG_QUALITY, 88])
        paths.append(path)
    return paths
