"""Шрифт замены и укладка текста в ячейку.

ШРИФТ. Исходный набор — рубленый гротеск шестидесятых с плотным очком; из установленных
в системе ближе всего DejaVu Sans Condensed: он узкий, а узость здесь важнее рисунка —
заголовок и повёрнутым-то ставили ради того, чтобы влез по ширине. Serif Condensed идёт
вторым: у части таблиц шапка набрана с засечками.

УКЛАДКА. Кегль подбирается сверху вниз, пока текст не влезет в отведённую рамку. Ниже
``min_px`` не опускаемся: смысл всей затеи в том, чтобы FineReader прочитал результат, а
не в том, чтобы он поместился любой ценой. Не влезло — значит, колонку надо расширять,
и это делает ``compose``, а не шрифт.

ПЕРЕНОСЫ. Слово переносится целиком; если целиком не влезает даже одно слово, разрешается
разрыв по дефису, а затем — жёсткий разрыв по букве. Так набирали и в исходнике:
«количество / в сутко-комп- / лекте».

Но разрыв ПОСЕРЕДИНЕ слова — крайняя мера, а не первая: «Специфицированна-я» читается
хуже, чем то же слово кеглем на два пункта мельче. Поэтому подбор идёт в два прохода:
сперва ищется самый крупный кегль, при котором целиком влезает САМОЕ ДЛИННОЕ СЛОВО, и
только если такого нет — разрешаются разрывы по буквам.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import ImageFont

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)

# Межстрочный интервал в долях кегля.
LINE_SPACING = 1.15

# Внутреннее поле ячейки в долях кегля: текст не должен липнуть к линейке.
CELL_PADDING = 0.35


@dataclass(frozen=True)
class Fit:
    """Уложенный текст: каким кеглем, какими строками и сколько места занял."""

    font_px: int
    lines: list[str]
    width: int
    height: int


def font_path(preferred: "tuple[str, ...]" = FONT_CANDIDATES) -> Path:
    for candidate in preferred:
        path = Path(candidate)
        if path.is_file():
            return path
    raise FileNotFoundError("не нашлось ни одного шрифта с кириллицей из списка")


def load_font(size: int, path: "Path | None" = None) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path or font_path()), size)


def _measure(font: ImageFont.FreeTypeFont, text: str) -> int:
    return int(round(font.getlength(text)))


def wrap(text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    """Разбиение на строки по ширине: сперва по словам, потом по дефису, потом по буквам."""
    lines: list[str] = []
    for word in text.split():
        if not lines:
            lines.append(word)
            continue
        candidate = f"{lines[-1]} {word}"
        if _measure(font, candidate) <= width:
            lines[-1] = candidate
        else:
            lines.append(word)
    result: list[str] = []
    for line in lines:
        result.extend(_split_long(line, font, width))
    return result or [""]


def _split_long(line: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    if _measure(font, line) <= width or not line:
        return [line]
    for position in range(len(line) - 1, 0, -1):
        if line[position] == "-" and _measure(font, line[: position + 1]) <= width:
            return [line[: position + 1]] + _split_long(line[position + 1 :], font, width)
    for position in range(len(line) - 1, 0, -1):
        if _measure(font, line[:position] + "-") <= width:
            return [line[:position] + "-"] + _split_long(line[position:], font, width)
    return [line]


def longest_word_width(text: str, font: ImageFont.FreeTypeFont) -> int:
    """Ширина самого длинного слова: ниже неё набор пойдёт с разрывами по буквам."""
    return max((_measure(font, word) for word in text.split()), default=0)


def fit_text(
    text: str, box_width: int, box_height: int, max_px: int, min_px: int, path: "Path | None" = None
) -> "Fit | None":
    """Самый крупный кегль, при котором текст влезает в рамку. None — не влез и на минимуме."""
    if not text.strip() or box_width <= 0 or box_height <= 0:
        return None
    for whole_words in (True, False):
        for size in range(int(max_px), int(min_px) - 1, -1):
            font = load_font(size, path)
            padding = int(round(size * CELL_PADDING))
            inner_width = box_width - 2 * padding
            inner_height = box_height - 2 * padding
            if inner_width <= 0 or inner_height <= 0:
                continue
            if whole_words and longest_word_width(text, font) > inner_width:
                continue
            lines = wrap(text, font, inner_width)
            line_height = int(round(size * LINE_SPACING))
            height = line_height * len(lines)
            widest = max((_measure(font, line) for line in lines), default=0)
            if height <= inner_height and widest <= inner_width:
                return Fit(size, lines, widest, height)
    return None


def required_width(text: str, font_px: int, box_height: int, max_lines: int = 3, path: "Path | None" = None) -> int:
    """Сколько ширины нужно тексту этого кегля, чтобы влезть в рамку такой высоты.

    Нужна расширению колонки: оно должно знать, НАСКОЛЬКО раздвигать, а не подбирать
    вслепую. Ширина ищется удвоением, потом уточняется делением пополам — счёт дешёвый,
    а формулы для переносов нет.
    """
    font = load_font(font_px, path)
    padding = int(round(font_px * CELL_PADDING))
    inner_height = box_height - 2 * padding
    line_height = int(round(font_px * LINE_SPACING))
    if inner_height < line_height:
        return _measure(font, text) + 2 * padding
    allowed_lines = max(1, min(max_lines, inner_height // line_height))

    low, high = 1, max(64, _measure(font, text) + 1)
    while len(wrap(text, font, high)) > allowed_lines:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if len(wrap(text, font, middle)) <= allowed_lines:
            high = middle
        else:
            low = middle + 1
    # Ширины должно хватить и на самое длинное слово целиком, иначе набор пойдёт с
    # разрывами по буквам, ради избавления от которых колонка и раздвигается.
    return max(low, longest_word_width(text, font)) + 2 * padding
