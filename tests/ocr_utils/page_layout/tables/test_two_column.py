"""Таблица в две графы: одна внутренняя вертикаль, но рамка или шапка, разделитель во всю высоту и текст."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from ocr_utils.page_layout.tables import ruling, verify
from tests.ocr_utils.page_layout.tables import synthetic
from tests.ocr_utils.page_layout.tables.synthetic import INK, PAPER, load_font

DPI = synthetic.DPI
WIDTH, DIVIDER = 1000, 640
LEFT = [
    "Издержки, связанные с вложением капитала",
    "Потери в связи с моральным износом",
    "Потери из-за порчи запасов . . . . .",
    "Расходы на перемещение запасов . .",
    "Содержание складских помещений . .",
    "Всего издержек в процентах . . . .",
]
RIGHT = ["12,00 %", "3,00 %", "3,00 %", "2,00 %", "0,25 %", "21,00 %"]


def _two_column(frame: bool = True, header: bool = True, rows: int = 8, font_px: int = 26) -> np.ndarray:
    """Таблица в две графы на 300 dpi.

    Args:
        frame: Рисовать ли боковые стороны и низ рамки (иначе — открытая таблица).
        header: Рисовать ли верхнюю линейку и отбивку шапки.
        rows: Сколько строк текста в теле.
        font_px: Кегль, px.

    Returns:
        Серый кадр вплотную к линейкам, как вырезка затравки.
    """
    height = 110 + rows * 45 + 20
    canvas = Image.new("L", (WIDTH, height), PAPER)
    draw = ImageDraw.Draw(canvas)
    font = load_font(font_px)
    if header or frame:
        draw.rectangle([0, 0, WIDTH - 1, 3], fill=INK)
    if header:
        draw.rectangle([0, 100, WIDTH - 1, 103], fill=INK)
        draw.text((40, 35), "Наименование статей", font=font, fill=INK)
        draw.text((DIVIDER + 40, 35), "Процент", font=font, fill=INK)
    if frame:
        draw.rectangle([0, height - 4, WIDTH - 1, height - 1], fill=INK)
        draw.rectangle([0, 0, 3, height - 1], fill=INK)
        draw.rectangle([WIDTH - 4, 0, WIDTH - 1, height - 1], fill=INK)
    draw.rectangle([DIVIDER, 104 if header else 0, DIVIDER + 3, height - 1], fill=INK)
    for row in range(rows):
        y = 120 + row * 45
        draw.text((40, y), LEFT[row % len(LEFT)], font=font, fill=INK)
        draw.text((DIVIDER + 60, y), RIGHT[row % len(RIGHT)], font=font, fill=INK)
    return np.asarray(canvas)


def _verdict(image: np.ndarray) -> tuple[bool, str]:
    found = verify.features(image, ruling.find_lines(image, DPI), DPI)
    return verify.is_table_v3(found)


def test_framed_two_column_table_is_a_table() -> None:
    """Замкнутая рамка, шапка, разделитель, шесть строк текста по обе стороны — таблица."""
    assert _verdict(_two_column())[0]


def test_open_two_column_table_with_header_is_a_table() -> None:
    """Без боковин и низа, но с верхней линейкой и отбивкой шапки (1974/01 с.21) — таблица."""
    assert _verdict(_two_column(frame=False))[0]


def test_single_line_in_a_box_stays_a_banner() -> None:
    """Одна строка крупного текста в рамке с разделителем — заголовок рубрики, не таблица."""
    ok, reason = _verdict(_two_column(header=False, rows=1, font_px=40))
    assert not ok, reason


def test_divider_without_frame_or_header_is_not_a_two_column_table() -> None:
    """Две колонки текста под разделителем без рамки и шапки (колонтитул с содержанием) — не таблица."""
    found = verify.features(
        *(lambda img: (img, ruling.find_lines(img, DPI)))(_two_column(frame=False, header=False)), DPI
    )
    assert not verify.is_two_column_table(found)
