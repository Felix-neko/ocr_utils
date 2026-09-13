"""Шрифт замены и укладка текста в ячейку.

ШРИФТ. Исходный набор — рубленый гротеск с плотным очком; из установленных ближе всего
DejaVu Sans Condensed: он узкий, а узость здесь важнее рисунка — заголовок и повёрнутым-то
ставили ради того, чтобы влез по ширине. Liberation в системе нет.

УКЛАДКА. Кегль подбирается сверху вниз от потолка (``dpi.font_cap_px``) до ``min_px``, пока
текст не влезет в рамку с полями. Перебираются и число строк, и разрывы: сперва ищется
самый крупный кегль, при котором целиком влезает САМОЕ ДЛИННОЕ СЛОВО, и только если
такого нет — разрешаются разрывы по дефису и по буквам. «Специфицированна-я» читается хуже,
чем то же слово кеглем на два пункта мельче.

Кегль ниже ``FIT_MIN_PX`` не рассматривается: это уже не текст, а штрих, и такой ответ
означает «не влезло» — дальше решают увеличение страницы и перекройка таблицы.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed.ttf",
)

# Межстрочный интервал в долях кегля.
LINE_SPACING = 1.15

# Внутреннее поле ячейки в долях кегля: текст не должен липнуть к линейке.
CELL_PADDING = 0.35

# Больше строк в ячейке не набираем: четыре и более строк в шапке узкой графы — это уже
# не заголовок, а абзац, и FineReader склеит его с соседями.
MAX_LINES = 3

# Ниже этого кегля подбор не идёт.
FIT_MIN_PX = 6


@dataclass(frozen=True)
class Fit:
    """Уложенный текст: каким кеглем, какими строками и сколько места занял."""

    font_px: int
    lines: tuple[str, ...]
    width: int
    height: int


def font_path() -> Path:
    for candidate in FONT_CANDIDATES:
        path = Path(candidate)
        if path.is_file():
            return path
    raise FileNotFoundError("не нашлось ни одного шрифта с кириллицей из списка")


@lru_cache(maxsize=512)
def load_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font_path()), max(1, int(size)))


def text_width(font: ImageFont.FreeTypeFont, text: str) -> int:
    return int(round(font.getlength(text)))


def line_height(font_px: int) -> int:
    return int(round(font_px * LINE_SPACING))


def wrap(text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    """Разбиение по ширине: сперва по словам, потом по дефису, потом по буквам."""
    lines: list[str] = []
    for word in text.split():
        if not lines:
            lines.append(word)
            continue
        candidate = f"{lines[-1]} {word}"
        if text_width(font, candidate) <= width:
            lines[-1] = candidate
        else:
            lines.append(word)
    result: list[str] = []
    for line in lines:
        result.extend(_split_long(line, font, width))
    return result or [""]


def _split_long(line: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    if not line or text_width(font, line) <= width:
        return [line]
    for position in range(len(line) - 1, 0, -1):
        if line[position] == "-" and text_width(font, line[: position + 1]) <= width:
            return [line[: position + 1]] + _split_long(line[position + 1 :], font, width)
    for position in range(len(line) - 1, 0, -1):
        if text_width(font, line[:position] + "-") <= width:
            return [line[:position] + "-"] + _split_long(line[position:], font, width)
    return [line]


def longest_word_width(text: str, font: ImageFont.FreeTypeFont) -> int:
    return max((text_width(font, word) for word in text.split()), default=0)


def layout(text: str, font_px: int, box_width: int, box_height: int, whole_words: bool) -> "Fit | None":
    """Уложить текст заданным кеглем; None — не влезает."""
    font = load_font(font_px)
    padding = int(round(font_px * CELL_PADDING))
    inner_width, inner_height = box_width - 2 * padding, box_height - 2 * padding
    if inner_width <= 0 or inner_height <= 0:
        return None
    if whole_words and longest_word_width(text, font) > inner_width:
        return None
    lines = wrap(text, font, inner_width)
    height = line_height(font_px) * len(lines)
    widest = max((text_width(font, line) for line in lines), default=0)
    if len(lines) > MAX_LINES or height > inner_height or widest > inner_width:
        return None
    return Fit(font_px, tuple(lines), widest, height)


def fit_text(text: str, box_width: int, box_height: int, max_px: int, min_px: int = FIT_MIN_PX) -> "Fit | None":
    """Самый крупный кегль, при котором текст влезает в рамку. None — не влез и на минимуме."""
    if not text.strip() or box_width <= 0 or box_height <= 0 or max_px < min_px:
        return None
    for whole_words in (True, False):
        for size in range(int(max_px), int(min_px) - 1, -1):
            fit = layout(text, size, box_width, box_height, whole_words)
            if fit is not None:
                return fit
    return None


def required_width(text: str, font_px: int, box_height: int, max_lines: int = MAX_LINES) -> int:
    """Сколько ширины нужно тексту этого кегля в рамке такой высоты — чтобы расширение
    колонки знало, НАСКОЛЬКО раздвигать. Ширина ищется удвоением и делением пополам:
    формулы для переносов нет, а счёт дешёвый."""
    font = load_font(font_px)
    padding = int(round(font_px * CELL_PADDING))
    inner_height = box_height - 2 * padding
    step = line_height(font_px)
    if inner_height < step:
        return text_width(font, text) + 2 * padding
    allowed_lines = max(1, min(max_lines, inner_height // step))

    low, high = 1, max(64, text_width(font, text) + 1)
    while len(wrap(text, font, high)) > allowed_lines:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if len(wrap(text, font, middle)) <= allowed_lines:
            high = middle
        else:
            low = middle + 1
    # Самое длинное слово обязано влезать целиком, иначе набор пойдёт с разрывами по буквам,
    # ради избавления от которых колонка и раздвигается.
    return max(low, longest_word_width(text, font)) + 2 * padding
