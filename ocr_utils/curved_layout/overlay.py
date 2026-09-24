"""Оверлей разбора: огибающие блоков, края рядов и осевые кривые строк поверх страницы."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ocr_utils.curved_layout.alignment import Alignment
from ocr_utils.curved_layout.blocks import dilate_polygon
from ocr_utils.curved_layout.leaders import inside_spans
from ocr_utils.curved_layout.page import PageAnalysis

# Цвета BGR: огибающая — синяя, крупная огибающая — фиолетовая, оси строк — зелёные,
# найденные края рядов — оранжевые кружки, границы колонок — серые пунктиры.
COLOUR_ENVELOPE = (220, 90, 20)
COLOUR_COARSE = (200, 60, 200)
COLOUR_AXIS = (40, 170, 40)
COLOUR_POINT = (30, 140, 240)
COLOUR_COLUMN = (170, 170, 170)
# Участок оси над точкой или запятой: там ось провисает к базовой линии, в меры формы строки он
# не входит и на оверлее рисуется отдельным цветом, чтобы провисание не принимали за дефект.
COLOUR_MARK = (0, 165, 255)
COLOUR_TEXT = (20, 20, 20)
# Раздутые границы: полсимвола — жёлтая, символ — красная, прочие доли — серая.
COLOUR_DILATE = {0.5: (0, 200, 255), 1.0: (40, 40, 220)}
COLOUR_DILATE_OTHER = (120, 120, 120)


def _polyline(
    canvas: np.ndarray, points: np.ndarray, colour, thickness: int, scale: float, closed: bool = False
) -> None:
    """Ломаная поверх холста с масштабом ``scale`` (рабочая копия → холст)."""
    if points is None or len(points) < 2:
        return
    pts = np.round(np.asarray(points, dtype=np.float64) * scale).astype(np.int32)
    cv2.polylines(canvas, [pts], closed, colour, thickness, cv2.LINE_AA)


def draw(
    analysis: PageAnalysis, gray: np.ndarray, scale: float = 1.0, dilate_extra: tuple[float, ...] = ()
) -> np.ndarray:
    """Нарисовать разбор поверх серого изображения страницы.

    Args:
        analysis: Результат разбора (координаты — в пикселях рабочей копии).
        gray: Серое изображение страницы, в котором рисуем.
        scale: Пикселей изображения на пиксель рабочей копии.
        dilate_extra: Доли размера символа, на которые дополнительно раздуть границу блока и
            нарисовать её же: так видно, как меняется контур от дилатации.

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
        _axis_line(canvas, axis, scale)
    for block, alignment in zip(analysis.blocks, analysis.alignments):
        _polyline(canvas, block.envelope.polygon, COLOUR_ENVELOPE, 2, scale, closed=True)
        _polyline(canvas, block.envelope_coarse.left, COLOUR_COARSE, 1, scale)
        _polyline(canvas, block.envelope_coarse.right, COLOUR_COARSE, 1, scale)
        glyph = block.glyph_size
        for share in dilate_extra:
            outline = dilate_polygon(block.envelope.polygon, share * glyph[0], share * glyph[1], block.dpi)
            _polyline(canvas, outline, COLOUR_DILATE.get(share, COLOUR_DILATE_OTHER), 1, scale, closed=True)
        for row in block.rows:
            for x in (row.x0, row.x1):
                cv2.circle(canvas, (int(x * scale), int(row.y * scale)), 3, COLOUR_POINT, 1, cv2.LINE_AA)
        _caption(canvas, block, alignment, scale)
    return canvas


def _axis_line(canvas: np.ndarray, axis, scale: float) -> None:
    """Ось строки: участки над точками и запятыми — своим цветом.

    Такой участок провисает к базовой линии (знак стоит на ней, а не на оси строки), поэтому он
    и выделяется: в меры наклона и формы строки он не входит, и принимать его за дефект не надо.
    """
    points = np.asarray(axis.points, dtype=np.float64)
    spans = list(getattr(axis, "mark_spans", ()) or ())
    if not spans or points.shape[0] < 2:
        _polyline(canvas, points, COLOUR_AXIS, 2, scale)
        return
    # Звено считается «над меткой», если под метку попадает любой из его концов.
    marked = inside_spans(points[:, 0], spans)
    start = 0
    for index in range(1, points.shape[0]):
        own = bool(marked[index - 1] or marked[index])
        last = index == points.shape[0] - 1
        following = bool(marked[index] or marked[index + 1]) if not last else not own
        if own != following or last:
            piece = points[start : index + 1]
            _polyline(canvas, piece, COLOUR_MARK if own else COLOUR_AXIS, 2, scale)
            start = index
    for x0, x1 in spans:
        middle = (x0 + x1) / 2.0
        y = float(np.interp(middle, points[:, 0], points[:, 1]))
        cv2.circle(canvas, (int(middle * scale), int(y * scale)), 3, COLOUR_MARK, -1, cv2.LINE_AA)


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


def _legend(canvas: np.ndarray, dilate_extra: tuple[float, ...]) -> None:
    """Легенда в правом верхнем углу: что означает каждая кривая."""
    lines = [("кромка по краске", COLOUR_ENVELOPE), ("ось строки", COLOUR_AXIS)]
    lines += [(f"+{share:g} символа", COLOUR_DILATE.get(share, COLOUR_DILATE_OTHER)) for share in dilate_extra]
    x = canvas.shape[1] - 150
    for index, (text, colour) in enumerate(lines):
        y = 16 + 14 * index
        cv2.rectangle(canvas, (x - 4, y - 10), (canvas.shape[1] - 4, y + 3), (255, 255, 255), -1)
        cv2.line(canvas, (x, y - 3), (x + 18, y - 3), colour, 2, cv2.LINE_AA)
        cv2.putText(canvas, text, (x + 22, y), cv2.FONT_HERSHEY_COMPLEX, 0.33, COLOUR_TEXT, 1, cv2.LINE_AA)


def write(
    analysis: PageAnalysis, gray300: np.ndarray, path: Path, width: int = 1400, dilate_extra: tuple[float, ...] = ()
) -> Path:
    """Записать оверлей в файл, ужав страницу до ширины ``width``."""
    scale_page = width / gray300.shape[1]
    page = cv2.resize(gray300, (width, int(gray300.shape[0] * scale_page)), interpolation=cv2.INTER_AREA)
    canvas = draw(analysis, page, scale=width / analysis.width, dilate_extra=dilate_extra)
    header = f"{analysis.name} с.{analysis.page} [{analysis.variant}] движок {analysis.engine}"
    cv2.putText(canvas, header, (10, 18), cv2.FONT_HERSHEY_COMPLEX, 0.5, COLOUR_TEXT, 1, cv2.LINE_AA)
    if dilate_extra:
        _legend(canvas, dilate_extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return path


__all__ = ["draw", "write"]
