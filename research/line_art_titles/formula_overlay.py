"""Оверлеи формул для просмотра глазами: окрестность рамки, рамка surya до и после достройки, блоки DeepSeek-OCR-2 с типом и текстом."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_utils.page_layout.geometry import Box

# Окрестность рамки на оверлее, мм, и разрешение картинки.
CONTEXT_MM = 8.0
VIEW_DPI = 300

# Разметка формулы в тексте DeepSeek — та же, что у слияния (merge.MATH).
from research.line_art_titles.merge import MATH  # noqa: E402

# Цвета (BGR): рамка surya до достройки — приглушённая, после — главная; блоки DeepSeek — как в overlay.py.
COLOR_SURYA_RAW = (160, 160, 160)
COLOR_GROWN = (220, 90, 20)
COLOR_BLOCK = (200, 60, 200)
COLOR_TEXT = (20, 20, 20)
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


class FormulaVerdict(str, Enum):
    """Вердикт DeepSeek-OCR-2 по вырезке формулы (промпт ``markdown``)."""

    EQUATION = "equation"  # есть блок equation
    INLINE = "текст_с_формулой"  # только текстовые блоки, но в тексте LaTeX-разметка
    NONE = "без_формулы"  # ни блока equation, ни разметки формулы


def verdict_of(elements: list[dict], raw: str) -> FormulaVerdict:
    """Вердикт DeepSeek по блокам вырезки.

    Args:
        elements: Блоки DeepSeek (``label``, ``text``).
        raw: Сырой вывод модели.

    Returns:
        :class:`FormulaVerdict`.
    """
    if any(e["label"] == "equation" for e in elements):
        return FormulaVerdict.EQUATION
    if MATH.search(raw or "") or any(MATH.search(e["text"]) for e in elements):
        return FormulaVerdict.INLINE
    return FormulaVerdict.NONE


def _font(size: int):
    """Шрифт DejaVu Sans нужного кегля."""
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def draw_formula(
    gray: np.ndarray, dpi: int, row: dict, elements: list[dict], raw: str, label: str | None = None
) -> np.ndarray:
    """Оверлей одной формулы.

    Args:
        gray: Серая полоса в родном разрешении.
        dpi: Разрешение полосы.
        row: Строка ``formulas/regions.jsonl`` (``box``, ``surya_box``, ``crop_inner`` — рамка в вырезке 300 dpi).
        elements: Блоки DeepSeek в пикселях вырезки (300 dpi, с полем).
        raw: Сырой вывод DeepSeek.
        label: Ручная разметка, если есть.

    Returns:
        Картинка BGR: шапка с вердиктом и текстом DeepSeek, окрестность с рамками, легенда.
    """
    height, width = gray.shape[:2]
    box = Box(*row["box"])
    margin = int(round(CONTEXT_MM / 25.4 * dpi))
    view = Box(box.x0 - margin, box.y0 - margin, box.x1 + margin, box.y1 + margin).clipped(width, height)
    scale = VIEW_DPI / dpi
    canvas = cv2.cvtColor(
        cv2.resize(gray[view.slice], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR
    )

    def to_view(x: float, y: float) -> tuple[int, int]:
        return int(round((x - view.x0) * scale)), int(round((y - view.y0) * scale))

    raw_box = Box(*row["surya_box"])
    cv2.rectangle(canvas, to_view(raw_box.x0, raw_box.y0), to_view(raw_box.x1, raw_box.y1), COLOR_SURYA_RAW, 1)
    cv2.rectangle(canvas, to_view(box.x0, box.y0), to_view(box.x1, box.y1), COLOR_GROWN, 2)
    # Блоки DeepSeek: из пикселей вырезки (300 dpi, начало — рамка минус поле) в пиксели полосы.
    ix0, iy0 = row["crop_inner"][:2]
    origin_x = box.x0 - ix0 / VIEW_DPI * dpi
    origin_y = box.y0 - iy0 / VIEW_DPI * dpi
    for element in elements:
        p0 = to_view(origin_x + element["x0"] / VIEW_DPI * dpi, origin_y + element["y0"] / VIEW_DPI * dpi)
        p1 = to_view(origin_x + element["x1"] / VIEW_DPI * dpi, origin_y + element["y1"] / VIEW_DPI * dpi)
        cv2.rectangle(canvas, (p0[0] + 2, p0[1] + 2), (p1[0] - 2, p1[1] - 2), COLOR_BLOCK, 2)
        cv2.putText(canvas, element["label"], (p0[0] + 4, p0[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLOCK, 1)
    # Картинка не уже 700 px: иначе шапка с текстом не помещается.
    if canvas.shape[1] < 700:
        canvas = np.hstack([canvas, np.full((canvas.shape[0], 700 - canvas.shape[1], 3), 255, np.uint8)])
    verdict = verdict_of(elements, raw)
    text = " ".join(" ".join(e["text"].split()) for e in elements)[:150] or "—"
    manual = f"   разметка: {label}" if label else ""
    lines = [
        f"{row['page']}, {row['id'].rsplit('_', 1)[-1]}   вердикт DeepSeek: {verdict.value}{manual}",
        f"блоки: {', '.join(e['label'] for e in elements) or '—'};  уверенность surya {row.get('confidence')}",
        f"DeepSeek: {text[:75]}",
        f"          {text[75:150]}",
    ]
    legend = [
        (COLOR_SURYA_RAW, "рамка surya Equation (исходная)"),
        (COLOR_GROWN, "рамка после достройки (выход детектора)"),
        (COLOR_BLOCK, "блок DeepSeek-OCR-2 с типом"),
    ]
    head = Image.new("RGB", (canvas.shape[1], 12 + 22 * len(lines)), (255, 255, 255))
    draw = ImageDraw.Draw(head)
    for index, line in enumerate(lines):
        draw.text((8, 6 + 22 * index), line, font=_font(15), fill=COLOR_TEXT[::-1])
    foot = Image.new("RGB", (canvas.shape[1], 12 + 20 * len(legend)), (255, 255, 255))
    draw = ImageDraw.Draw(foot)
    for index, (color, name) in enumerate(legend):
        y = 6 + 20 * index
        draw.rectangle((8, y + 3, 34, y + 15), outline=color[::-1], width=2)
        draw.text((42, y), name, font=_font(13), fill=COLOR_TEXT[::-1])
    to_bgr = lambda image: cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)  # noqa: E731
    return np.vstack([to_bgr(head), canvas, to_bgr(foot)])


def formula_path(root: Path, row: dict, verdict: FormulaVerdict) -> Path:
    """Путь оверлея формулы: ``<root>/<вердикт>/<id>.jpg``."""
    return root / verdict.value / f"{row['id']}.jpg"


__all__ = ["FormulaVerdict", "draw_formula", "formula_path", "verdict_of"]
