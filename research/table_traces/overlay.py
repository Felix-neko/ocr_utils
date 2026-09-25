"""Оверлей «сетка по осям | сетка по кривым» для одной таблицы: палитра проекта, полупрозрачные ячейки, легенда."""

from __future__ import annotations

import cv2
import numpy as np

from ocr_utils.page_layout.geometry import Grid
from ocr_utils.page_layout.tables.curved_grid import CurvedGrid
from ocr_utils.page_layout.tables.ruling import Lines
from ocr_utils.page_layout.tables.traces import RuleTraces

# Палитра оверлеев проекта (BGR, навык draw-overlay).
COLOR_CELL = (0, 165, 255)  # ячейка тела — оранжевый
COLOR_HEADER = (200, 60, 200)  # ячейка шапки — фиолетовый
COLOR_HORIZONTAL = (40, 170, 40)  # горизонтальная линейка — зелёный
COLOR_VERTICAL = (0, 0, 220)  # вертикальная линейка — красный
COLOR_CHORD = (20, 20, 20)  # отрезок «начало–конец»
COLOR_BOX = (220, 90, 20)  # рамка ячейки сетки по осям — синий
COLOR_TEXT = (20, 20, 20)
CELL_ALPHA = 0.35  # заливка ячеек: буквы под ней должны читаться

LEGEND = (
    ("ячейка тела (заливка)", COLOR_CELL, True),
    ("ячейка шапки (заливка)", COLOR_HEADER, True),
    ("горизонтальная линейка", COLOR_HORIZONTAL, False),
    ("вертикальная линейка", COLOR_VERTICAL, False),
    ("отрезок начало-конец", COLOR_CHORD, False),
    ("ячейка сетки по осям", COLOR_BOX, False),
)


def _on_paper(color: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    """Цвет, смешанный с белым так же, как полупрозрачная заливка ложится на бумагу."""
    return tuple(int(round(alpha * c + (1 - alpha) * 255)) for c in color)


def _legend(canvas: np.ndarray, scale: float) -> None:
    """Легенда в левом верхнем углу: образец цвета (заливка — с той же прозрачностью, что на странице) и подпись.

    Подписи рисует PIL: ``cv2.putText`` кириллицу не умеет.
    """
    from PIL import Image, ImageDraw, ImageFont

    font_px = max(14, int(16 * scale))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_px)
    except OSError:
        font = ImageFont.load_default()
    line = int(font_px * 1.5)
    width = int(font_px * 16)
    height = line * len(LEGEND) + 10
    box = np.full((height, width, 3), 255, np.uint8)
    for index, (label, color, filled) in enumerate(LEGEND):
        y = 5 + index * line + line // 2
        sample = _on_paper(color, CELL_ALPHA) if filled else color
        if filled:
            cv2.rectangle(box, (6, y - line // 3), (36, y + line // 3), sample, -1)
        else:
            cv2.line(box, (6, y), (36, y), sample, 2)
    image = Image.fromarray(cv2.cvtColor(box, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for index, (label, _color, _filled) in enumerate(LEGEND):
        draw.text((44, 5 + index * line + (line - font_px) // 2), label, fill=(20, 20, 20), font=font)
    box = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    h, w = min(height, canvas.shape[0]), min(width, canvas.shape[1])
    canvas[:h, :w] = box[:h, :w]


def _polyline(canvas: np.ndarray, points: np.ndarray, color, thickness: int) -> None:
    cv2.polylines(canvas, [np.round(points).astype(np.int32)], False, color, thickness, cv2.LINE_AA)


def draw_pair(gray: np.ndarray, axis_grid: Grid, lines: Lines, curved: CurvedGrid, traces: RuleTraces) -> np.ndarray:
    """Две панели одной вырезки: слева сетка по осям и её линейки, справа ячейки по кривым и сплайны.

    Args:
        gray: Вырезка таблицы (рабочее разрешение).
        axis_grid: Сетка по осям в пикселях вырезки.
        lines: Линейки сетки по осям (``find_lines``).
        curved: Сетка по кривым в пикселях вырезки.
        traces: Трассы линеек в пикселях вырезки.

    Returns:
        Картинка BGR: две панели рядом.
    """
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    left = base.copy()
    layer = left.copy()
    for cell in axis_grid.cells:
        b = cell.box
        color = COLOR_HEADER if cell.is_header else COLOR_CELL
        cv2.rectangle(layer, (b.x0 + 2, b.y0 + 2), (b.x1 - 3, b.y1 - 3), color, -1)
    cv2.addWeighted(layer, CELL_ALPHA, left, 1 - CELL_ALPHA, 0, left)
    for cell in axis_grid.cells:
        b = cell.box
        cv2.rectangle(left, (b.x0, b.y0), (b.x1 - 1, b.y1 - 1), COLOR_BOX, 1)
    for segment in lines.horizontal + lines.vertical:
        b = segment.box
        color = COLOR_HORIZONTAL if segment.horizontal else COLOR_VERTICAL
        cv2.rectangle(left, (b.x0, b.y0), (b.x1 - 1, b.y1 - 1), color, 1)

    right = base.copy()
    layer = right.copy()
    for cell in curved.cells:
        color = COLOR_HEADER if cell.is_header else COLOR_CELL
        cv2.fillPoly(layer, [np.round(cell.inner_polygon).astype(np.int32)], color)
    cv2.addWeighted(layer, CELL_ALPHA, right, 1 - CELL_ALPHA, 0, right)
    for trace in traces.all:
        color = COLOR_HORIZONTAL if trace.horizontal else COLOR_VERTICAL
        (x0, y0), (x1, y1) = trace.chord
        cv2.line(right, (int(x0), int(y0)), (int(x1), int(y1)), COLOR_CHORD, 1, cv2.LINE_AA)
        _polyline(right, trace.sample(0.5), color, 2)
    scale = max(1.0, gray.shape[1] / 1200)
    _legend(right, scale)
    gap = np.full((gray.shape[0], 12, 3), 255, np.uint8)
    return np.hstack([left, gap, right])
