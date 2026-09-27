"""Поля отладочных оверлеев: шапка и легенда рисуются в полосах ВНЕ страницы, а не поверх неё.

Легенда поверх страницы закрывает то, что на оверлее надо разглядеть (угол полосы, колонтитул,
край блока), а на вырезке с увеличением она и вовсе съедает поле зрения. Поэтому шапка (что за
полоса и что на картинке) и легенда (что значит каждый цвет) собираются отдельными белыми
полосами и приклеиваются к холсту сверху и снизу: :func:`framed`.

Текст — шрифтом DejaVu через PIL: векторный шрифт OpenCV не знает части кириллицы и типографских
знаков («—», «×», «°»).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Шрифт подписей и его кегли (пиксели картинки): шапка и строки легенды.
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
HEADER_SIZE = 18
LEGEND_SIZE = 15
# Высота строки шапки и легенды, поля вокруг текста (пиксели картинки).
HEADER_ROW = 26
LEGEND_ROW = 22
PAD = 8
# Ширина образца цвета в легенде и зазор от образца до подписи.
SAMPLE_WIDTH = 30
SAMPLE_GAP = 8
# Наименьшая ширина колонки легенды (шире — по самой длинной подписи): по ней считается, сколько
# колонок влезает в ширину картинки.
LEGEND_COLUMN_WIDTH = 360
COLOUR_TEXT = (20, 20, 20)
PAPER = (255, 255, 255)


class SampleStyle(Enum):
    """Как рисуется образец в легенде: линией, пунктиром или залитым прямоугольником с рамкой."""

    LINE = "линия"
    DASHED = "пунктир"
    BOX = "область"


@dataclass(frozen=True)
class LegendEntry:
    """Строка легенды: подпись, цвет BGR, прозрачность разметки на странице и вид образца."""

    text: str
    colour: tuple[int, int, int]
    alpha: float = 1.0
    style: SampleStyle = SampleStyle.LINE


@lru_cache(maxsize=4)
def _font(size: int) -> ImageFont.FreeTypeFont:
    """Шрифт DejaVu кегля ``size`` (кэшируется: полос на прогон — тысячи)."""
    return ImageFont.truetype(FONT_PATH, size)


def on_paper(colour: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
    """Цвет полупрозрачной разметки, как он ложится на белую бумагу.

    Образец в легенде обязан выглядеть так же, как линия на странице: полупрозрачная линия на
    бумаге светлее своего цвета, и непрозрачный образец сбивал бы с толку.

    Args:
        colour: Цвет BGR.
        alpha: Прозрачность, с которой разметка подмешана к странице.

    Returns:
        Цвет BGR после смешения с белым.
    """
    return tuple(int(round(alpha * own + (1.0 - alpha) * 255)) for own in colour)


def _to_bgr(picture: Image.Image) -> np.ndarray:
    """Картинку PIL (RGB) — в массив BGR для OpenCV."""
    return cv2.cvtColor(np.asarray(picture), cv2.COLOR_RGB2BGR)


def _rgb(colour: tuple[int, int, int]) -> tuple[int, int, int]:
    """Цвет BGR — в RGB для PIL."""
    return colour[2], colour[1], colour[0]


def header_strip(lines: list[str], width: int) -> np.ndarray:
    """Шапка: белая полоса с текстом по строке на элемент ``lines``.

    Args:
        lines: Строки шапки (выпуск, полоса, вариант, что на картинке, счёт объектов).
        width: Ширина полосы — ширина холста.

    Returns:
        Полоса BGR ``(высота, width, 3)``.
    """
    picture = Image.new("RGB", (width, PAD + HEADER_ROW * max(1, len(lines))), PAPER)
    draw = ImageDraw.Draw(picture)
    for index, line in enumerate(lines):
        draw.text((PAD, PAD // 2 + HEADER_ROW * index), line, font=_font(HEADER_SIZE), fill=_rgb(COLOUR_TEXT))
    return _to_bgr(picture)


def _sample(draw: ImageDraw.ImageDraw, x: int, y: int, entry: LegendEntry) -> None:
    """Образец цвета строки легенды в точке ``(x, y)`` (верх строки)."""
    paper = _rgb(on_paper(entry.colour, entry.alpha))
    middle = y + LEGEND_ROW // 2
    if entry.style is SampleStyle.BOX:
        draw.rectangle((x, middle - 7, x + SAMPLE_WIDTH, middle + 7), fill=paper, outline=_rgb(entry.colour), width=2)
    elif entry.style is SampleStyle.DASHED:
        for start in range(0, SAMPLE_WIDTH, 8):
            draw.line((x + start, middle, x + min(start + 4, SAMPLE_WIDTH), middle), fill=paper, width=3)
    else:
        draw.line((x, middle, x + SAMPLE_WIDTH, middle), fill=paper, width=3)


def legend_strip(entries: list[LegendEntry], width: int, title: str | None = None) -> np.ndarray:
    """Легенда: белая полоса с образцами цвета и подписями в несколько колонок.

    Args:
        entries: Строки легенды.
        width: Ширина полосы — ширина холста.
        title: Необязательная подпись над легендой (чья это легенда).

    Returns:
        Полоса BGR; пустой массив высоты 0, если строк нет.
    """
    if not entries and not title:
        return np.zeros((0, width, 3), dtype=np.uint8)
    # Колонка не уже самой длинной подписи: иначе текст залезает на соседнюю колонку.
    longest = max((_font(LEGEND_SIZE).getlength(entry.text) for entry in entries), default=0.0)
    column_need = max(LEGEND_COLUMN_WIDTH, int(longest) + SAMPLE_WIDTH + SAMPLE_GAP + 2 * PAD)
    columns = max(1, min(len(entries), (width - PAD) // column_need)) if entries else 1
    rows = (len(entries) + columns - 1) // columns
    top = LEGEND_ROW if title else 0
    picture = Image.new("RGB", (width, PAD + top + LEGEND_ROW * rows + PAD // 2), PAPER)
    draw = ImageDraw.Draw(picture)
    if title:
        draw.text((PAD, PAD // 2), title, font=_font(LEGEND_SIZE), fill=_rgb(COLOUR_TEXT))
    column_width = width // columns
    for index, entry in enumerate(entries):
        # Строки идут по колонкам сверху вниз: соседние по смыслу строки остаются рядом.
        column, row = divmod(index, rows)
        x = PAD + column * column_width
        y = PAD // 2 + top + row * LEGEND_ROW
        _sample(draw, x, y, entry)
        draw.text((x + SAMPLE_WIDTH + SAMPLE_GAP, y + 2), entry.text, font=_font(LEGEND_SIZE), fill=_rgb(COLOUR_TEXT))
    return _to_bgr(picture)


def framed(
    canvas: np.ndarray, header: list[str], legend: list[LegendEntry], legend_title: str | None = None
) -> np.ndarray:
    """Холст со шапкой сверху и легендой снизу — обе в полях, страница остаётся открытой целиком.

    Args:
        canvas: Холст BGR с разметкой.
        header: Строки шапки.
        legend: Строки легенды.
        legend_title: Подпись над легендой.

    Returns:
        Новая картинка BGR: шапка, холст, легенда.
    """
    width = canvas.shape[1]
    return np.vstack([header_strip(header, width), canvas, legend_strip(legend, width, legend_title)])


__all__ = ["LegendEntry", "SampleStyle", "framed", "header_strip", "legend_strip", "on_paper"]
