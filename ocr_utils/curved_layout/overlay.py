"""Оверлей разбора: огибающие блоков, края рядов и осевые кривые строк поверх страницы."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ocr_utils.curved_layout.alignment import Alignment
from ocr_utils.curved_layout.page import PageAnalysis

# Цвета BGR: огибающая — синяя, крупная огибающая — фиолетовая, оси строк — зелёные,
# найденные края рядов — оранжевые кружки, границы колонок — серые пунктиры.
COLOUR_ENVELOPE = (220, 90, 20)
COLOUR_COARSE = (200, 60, 200)
COLOUR_AXIS = (40, 170, 40)
COLOUR_POINT = (30, 140, 240)
COLOUR_COLUMN = (170, 170, 170)
COLOUR_TEXT = (20, 20, 20)


def _polyline(
    canvas: np.ndarray, points: np.ndarray, colour, thickness: int, scale: float, closed: bool = False
) -> None:
    """Ломаная поверх холста с масштабом ``scale`` (рабочая копия → холст)."""
    if points is None or len(points) < 2:
        return
    pts = np.round(np.asarray(points, dtype=np.float64) * scale).astype(np.int32)
    cv2.polylines(canvas, [pts], closed, colour, thickness, cv2.LINE_AA)


def draw(analysis: PageAnalysis, gray: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Нарисовать разбор поверх серого изображения страницы.

    Args:
        analysis: Результат разбора (координаты — в пикселях рабочей копии).
        gray: Серое изображение страницы, в котором рисуем.
        scale: Пикселей изображения на пиксель рабочей копии.

    Returns:
        Цветной холст BGR.
    """
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for gutter in analysis.gutters:
        # Межколонник — ломаная: на трапеции он уезжает вбок вместе с колонками.
        for side in (1, 2):
            points = np.array([[point[side], point[0]] for point in gutter.points], dtype=np.float64)
            _polyline(canvas, points, COLOUR_COLUMN, 1, scale)
    for axis in analysis.axes:
        _polyline(canvas, axis.points, COLOUR_AXIS, 2, scale)
    for block, alignment in zip(analysis.blocks, analysis.alignments):
        _polyline(canvas, block.envelope.polygon, COLOUR_ENVELOPE, 2, scale, closed=True)
        _polyline(canvas, block.envelope_coarse.left, COLOUR_COARSE, 1, scale)
        _polyline(canvas, block.envelope_coarse.right, COLOUR_COARSE, 1, scale)
        for row in block.rows:
            for x in (row.x0, row.x1):
                cv2.circle(canvas, (int(x * scale), int(row.y * scale)), 3, COLOUR_POINT, 1, cv2.LINE_AA)
        _caption(canvas, block, alignment, scale)
    return canvas


def _caption(canvas: np.ndarray, block, alignment: Alignment, scale: float) -> None:
    """Подпись блока: колонка, строки, шаг, выключка и меры кромок."""
    lines = [
        f"кол.{block.column}.{block.index}: строк {block.lines}, шаг {block.pitch_mm:.1f} мм, выключка {alignment.kind.value}",
        f"L core {alignment.left.core_share:.2f} dev {alignment.left.envelope_dev_mm:.2f} "
        f"bend {alignment.left.bend_mm:.2f} отступов {alignment.left.indent_rows}",
        f"R core {alignment.right.core_share:.2f} dev {alignment.right.envelope_dev_mm:.2f} "
        f"bend {alignment.right.bend_mm:.2f} отступов {alignment.right.indent_rows}",
    ]
    x = int(block.span[0] * scale) + 4
    y = max(30, int(block.envelope.polygon[:, 1].min() * scale) - 8 - 12 * len(lines))
    for i, text in enumerate(lines):
        # Подложка под подпись: на тексте страницы чёрные буквы иначе не читаются.
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.33, 1)
        cv2.rectangle(canvas, (x - 2, y + i * 12 - h - 2), (x + w + 2, y + i * 12 + 3), (255, 255, 255), -1)
        cv2.putText(canvas, text, (x, y + i * 12), cv2.FONT_HERSHEY_COMPLEX, 0.33, COLOUR_TEXT, 1, cv2.LINE_AA)


def write(analysis: PageAnalysis, gray300: np.ndarray, path: Path, width: int = 1400) -> Path:
    """Записать оверлей в файл, ужав страницу до ширины ``width``."""
    scale_page = width / gray300.shape[1]
    page = cv2.resize(gray300, (width, int(gray300.shape[0] * scale_page)), interpolation=cv2.INTER_AREA)
    canvas = draw(analysis, page, scale=width / analysis.width)
    header = f"{analysis.name} с.{analysis.page} [{analysis.variant}] движок {analysis.engine}"
    cv2.putText(canvas, header, (10, 18), cv2.FONT_HERSHEY_COMPLEX, 0.5, COLOUR_TEXT, 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return path


__all__ = ["draw", "write"]
