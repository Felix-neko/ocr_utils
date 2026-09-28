"""Вырезки концов строк из бинаризованного PDF и контакт-листы с номерами — для разметки глазами."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import fitz
import numpy as np

from ocr_utils.geometry_regression.render import render_gray
from ocr_utils.page_layout.text_blocks import RENDER_DPI, WORK_DPI

# Пиксель рабочей копии в мм и во сколько раз рендер крупнее рабочей копии.
MM_PER_PX = 25.4 / WORK_DPI
K = RENDER_DPI / WORK_DPI
# Поле вырезки, мм: вглубь текста от конца строки, наружу за конец строки и по вертикали от оси.
CROP_IN_MM, CROP_OUT_MM, CROP_HALF_MM = 16.0, 8.0, 5.0
# Увеличение вырезки на листе.
ZOOM = 1.5
# Цвета разметки вырезки (BGR): конец строки по детектору и кривая стороны блока.
COLOUR_END = (0, 0, 230)
COLOUR_SIDE = (0, 170, 0)
COLOUR_TEXT = (20, 20, 20)
# Стрелки проверяемой строки.
COLOUR_ROW = (220, 120, 0)


def side_x(x: float, resid_mm: float, side: str) -> float:
    """Абсцисса кривой стороны у ряда по концу строки и отклонению («внутрь блока — плюс»).

    Args:
        x: Конец строки, пиксели рабочей копии.
        resid_mm: Отклонение конца от стороны.
        side: ``left`` или ``right``.

    Returns:
        Абсцисса стороны в пикселях рабочей копии.
    """
    shift = resid_mm / MM_PER_PX
    return x + shift if side == "right" else x - shift


def end_crop(gray: np.ndarray, record: dict) -> np.ndarray:
    """Цветная вырезка конца строки ``RENDER_DPI`` с отметками конца (красная) и стороны блока (зелёная).

    Args:
        gray: Серый рендер страницы ``RENDER_DPI``.
        record: Строка ``ends.csv`` (словарь, числа — float).

    Returns:
        Вырезка BGR; вне страницы добивается белым.
    """
    px = RENDER_DPI / 25.4
    x, y = float(record["x"]) * K, float(record["y"]) * K
    right = record["side"] == "right"
    # Границы вырезки — целые пиксели рендера, чтобы размер окна и куска страницы совпадали.
    x0 = int(round(x - (CROP_IN_MM if right else CROP_OUT_MM) * px))
    x1 = int(round(x + (CROP_OUT_MM if right else CROP_IN_MM) * px))
    y0, y1 = int(round(y - CROP_HALF_MM * px)), int(round(y + CROP_HALF_MM * px))
    # Вырезка с полями: координаты вне страницы заполняются белым.
    out = np.full((y1 - y0, x1 - x0), 255, dtype=np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = gray[sy0:sy1, sx0:sx1]
    colour = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    # Отметки: конец строки по детектору — красный штрих над и под строкой, сторона блока — зелёный.
    ex = int(x - x0)
    sx = int(side_x(float(record["x"]), float(record["resid_mm"]), record["side"]) * K - x0)
    h = colour.shape[0]
    for mark, tone in ((ex, COLOUR_END), (sx, COLOUR_SIDE)):
        cv2.line(colour, (mark, 0), (mark, h // 4), tone, 2)
        cv2.line(colour, (mark, 3 * h // 4), (mark, h - 1), tone, 2)
    # Проверяемая строка — стрелки по её оси у обоих краёв вырезки: на вырезке видны и соседние строки.
    row_y, w = int(y - y0), colour.shape[1]
    for tip, base in ((12, 0), (w - 13, w - 1)):
        cv2.fillPoly(colour, [np.array([[base, row_y - 7], [base, row_y + 7], [tip, row_y]], dtype=np.int32)], COLOUR_ROW)
    return cv2.resize(colour, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_AREA)


def labelled(image: np.ndarray, text: str) -> np.ndarray:
    """Вырезка с белой полосой-подписью сверху (латиница и цифры — cv2 не рисует кириллицу)."""
    band = np.full((26, image.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(band, text, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOUR_TEXT, 1, cv2.LINE_AA)
    return np.vstack([band, image])


def crops_for(records: list[dict], pdf_dir: Path) -> dict[str, np.ndarray]:
    """Вырезки для набора концов строк, по одному рендеру на страницу.

    Args:
        records: Строки ``ends.csv`` с полем ``id`` (подпись на листе).
        pdf_dir: Каталог PDF варианта.

    Returns:
        ``id → вырезка`` (BGR, с подписью).
    """
    by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        by_page[(record["pdf"], int(record["page"]))].append(record)
    out = {}
    for (pdf, page), items in by_page.items():
        with fitz.open(pdf_dir / pdf) as document:
            gray = render_gray(document, page - 1)
        for record in items:
            caption = f"{record['id']}  {record['key'][:7]} {record['key'][8:]} r{float(record['resid_mm']):+.1f}"
            out[record["id"]] = labelled(end_crop(gray, record), caption)
    return out


def contact_sheets(crops: list[np.ndarray], columns: int = 4, rows: int = 6) -> list[np.ndarray]:
    """Контакт-листы: вырезки сеткой ``columns × rows``, по листу на каждые ``columns·rows`` вырезок.

    Args:
        crops: Вырезки одного размера (BGR).
        columns: Столбцов на листе.
        rows: Строк на листе.

    Returns:
        Листы BGR.
    """
    if not crops:
        return []
    h = max(crop.shape[0] for crop in crops)
    w = max(crop.shape[1] for crop in crops)
    sheets = []
    per = columns * rows
    for start in range(0, len(crops), per):
        chunk = crops[start : start + per]
        sheet = np.full((rows * (h + 6), columns * (w + 6), 3), 200, dtype=np.uint8)
        for index, crop in enumerate(chunk):
            r, c = divmod(index, columns)
            sheet[r * (h + 6) : r * (h + 6) + crop.shape[0], c * (w + 6) : c * (w + 6) + crop.shape[1]] = crop
        used_rows = (len(chunk) + columns - 1) // columns
        sheets.append(sheet[: used_rows * (h + 6)])
    return sheets





def before_after(records: list[dict], pdf_dir: Path) -> dict[str, np.ndarray]:
    """Вырезки «до | после» для сдвинутых концов: слева — конец строки до правила, справа — после.

    Args:
        records: Строки ``changes_<правило>.csv`` (``key``, ``side``, ``x_before``, ``x_after``, ``y``,
            ``resid_before``, ``resid_after``) с полями ``id``, ``pdf``, ``page``.
        pdf_dir: Каталог PDF.

    Returns:
        ``id → вырезка`` (две вырезки рядом с подписью).
    """
    by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        by_page[(record["pdf"], int(record["page"]))].append(record)
    out = {}
    for (pdf, page), items in by_page.items():
        with fitz.open(pdf_dir / pdf) as document:
            gray = render_gray(document, page - 1)
        for record in items:
            pair = []
            for which in ("before", "after"):
                x = float(record[f"x_{which}"])
                resid = float(record[f"resid_{which}"])
                pair.append(end_crop(gray, {"x": x, "y": record["y"], "side": record["side"], "resid_mm": resid}))
            gap = np.full((pair[0].shape[0], 8, 3), 160, dtype=np.uint8)
            caption = f"{record['id']} {record['key'][:7]} {record['key'][8:]} {record['side'][0]} {float(record['moved_mm']):+.1f}mm"
            out[record["id"]] = labelled(np.hstack([pair[0], gap, pair[1]]), caption)
    return out
__all__ = ["before_after", "contact_sheets", "crops_for", "end_crop", "labelled", "side_x"]
