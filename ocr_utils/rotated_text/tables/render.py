"""Закраска старого текста и набор нового.

ЗАКРАСКА — плоская заливка внутренности ячейки уровнем бумаги ЭТОЙ ячейки (90-й процентиль
её пикселей): фон полос выровнен предыдущими шагами конвейера, и заплата не видна;
инпейнтинг здесь ни к чему. Линейки не трогаются: внутренность отступает от них на
измеренную толщину плюс 0.3 мм.

НАБОР — PIL по готовой укладке (``fit.Fit``), по центру ячейки, цветом краски всей таблицы
(5-й процентиль): у закрашенной ячейки собственный квантиль уплыл бы к бумаге и буквы
вышли бы серыми.

УВЕЛИЧЕНИЕ — до набора, а не после: буквы, набранные в исходном разрешении и растянутые
потом, размылись бы, а весь смысл увеличения в том, чтобы распознаватель их прочитал.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ocr_utils.page_layout.geometry import Box

from ocr_utils.rotated_text.tables.fit import Fit, line_height, load_font, text_width


@dataclass(frozen=True)
class Replacement:
    """Что набрать в ячейке: внутренность, укладка, уровень бумаги для закраски."""

    inner: Box
    fit: Fit
    paper: int


def paper_of(gray: np.ndarray, box: Box) -> int:
    """Уровень бумаги внутри рамки; 255, если рамка пуста."""
    safe = box.clipped(gray.shape[1], gray.shape[0])
    if safe.width <= 0 or safe.height <= 0:
        return 255
    return int(np.percentile(gray[safe.slice], 90))


def upscale(gray: np.ndarray, factor: float) -> np.ndarray:
    """Увеличение с бикубической интерполяцией; единичный множитель — копия."""
    if abs(factor - 1.0) < 1e-6:
        return gray.copy()
    return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def render(gray: np.ndarray, replacements: list[Replacement], ink: int) -> np.ndarray:
    """Закрасить и набрать все замены; возвращает новую картинку."""
    image = Image.fromarray(gray).copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for item in replacements:
        box = item.inner.clipped(width, height)
        if box.width <= 0 or box.height <= 0:
            continue
        draw.rectangle([box.x0, box.y0, box.x1 - 1, box.y1 - 1], fill=item.paper)
        draw_fit(draw, box, item.fit, ink)
    return np.asarray(image)


def draw_fit(draw: ImageDraw.ImageDraw, box: Box, fit: Fit, colour: int) -> None:
    """Строки укладки по центру рамки, каждая строка по центру по горизонтали."""
    font = load_font(fit.font_px)
    step = line_height(fit.font_px)
    total = step * len(fit.lines)
    top = box.y0 + max(0, (box.height - total) // 2)
    for index, line in enumerate(fit.lines):
        left = box.x0 + max(0, (box.width - text_width(font, line)) // 2)
        draw.text((left, top + index * step), line, fill=colour, font=font)
