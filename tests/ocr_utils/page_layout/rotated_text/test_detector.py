"""Зоны повёрнутого текста с проверкой чтением: синтетика 150 dpi (цепочки) и 600 dpi (tesseract)."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_utils.page_layout.geometry import Box
from ocr_utils.page_layout.rotated_text.detector import read_evidence, reads_sideways, rotated_zones, speck_density
from ocr_utils.page_layout.rotated_text.docstrum import cluster_rotated, glyph_components

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
WORK_DPI = 150
READ_DPI = 600


def _page600() -> np.ndarray:
    """Полоса 600 dpi: подпись «Техническое снабжение» боком (снизу вверх) и столбик чисел прямо."""
    image = Image.new("L", (2400, 3000), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, 60)
    label = Image.new("L", (900, 80), 255)
    ImageDraw.Draw(label).text((5, 5), "Техническое снабжение", fill=0, font=font)
    image.paste(label.rotate(90, expand=True), (1800, 600))
    # Столбик чисел с шагом строки: первые цифры стоят друг над другом, цепочка по вертикали есть.
    for i, number in enumerate(("86", "81", "96", "92", "94", "94", "89")):
        draw.text((400, 600 + i * 75), number, fill=0, font=font)
    return np.asarray(image)


def _bitonal150(gray600: np.ndarray) -> np.ndarray:
    """Битональная рабочая копия 150 dpi."""
    small = cv2.resize(gray600, None, fx=WORK_DPI / READ_DPI, fy=WORK_DPI / READ_DPI, interpolation=cv2.INTER_AREA)
    return np.where(small < 160, 0, 255).astype(np.uint8)


def test_чтение_оставляет_подпись_и_снимает_столбик_чисел() -> None:
    gray = _page600()
    bitonal = _bitonal150(gray)
    checked = rotated_zones(bitonal, WORK_DPI, [], gray=gray, gray_dpi=READ_DPI)
    assert len(checked) == 1 and checked[0].box.x0 * 4 >= 1700, [zone.box.as_tuple() for zone in checked]
    evidence = checked[0].info["evidence"]
    assert max(evidence[90], evidence[270]) > evidence[0]


def test_в_line_art_нечитаемая_зона_остаётся_а_прямое_чтение_снимает() -> None:
    assert reads_sideways({0: 0, 90: 0, 270: 0}, inside_line_art=True)
    assert not reads_sideways({0: 0, 90: 0, 270: 0}, inside_line_art=False)
    assert not reads_sideways({0: 4, 90: 2, 270: 0}, inside_line_art=True)
    assert not reads_sideways({}, inside_line_art=True)


def test_пустая_полоса_не_читается() -> None:
    gray = np.full((400, 400), 255, np.uint8)
    assert read_evidence(gray, Box(100, 100, 140, 300), READ_DPI) == {0: 0, 90: 0, 270: 0}


def test_тонкий_пунктир_не_цепочка() -> None:
    bitonal = np.full((1400, 1000), 255, np.uint8)
    for y in range(100, 1300, 30):
        cv2.rectangle(bitonal, (500, y), (501, y + 18), 0, -1)  # штрих 0.3 мм × 3 мм
    assert cluster_rotated(glyph_components(bitonal, WORK_DPI), WORK_DPI) == []


def test_толстый_наклонный_пунктир_не_цепочка() -> None:
    """Штрихи толще THIN_MM, наклонные — отсекаются долей прямых штрихов в цепочке."""
    bitonal = np.full((1400, 1000), 255, np.uint8)
    for y in range(100, 1300, 40):
        cv2.line(bitonal, (500, y), (506, y + 24), 0, 4)
    assert cluster_rotated(glyph_components(bitonal, WORK_DPI), WORK_DPI) == []


def test_точки_над_буквами_не_удлиняют_цепочку() -> None:
    """Две «буквы» соседних строк, связанные точками «ё» между ними, — не повёрнутый текст."""
    bitonal = np.full((600, 600), 255, np.uint8)
    for y in (200, 245):
        cv2.rectangle(bitonal, (300, y), (318, y + 24), 0, -1)
        cv2.rectangle(bitonal, (303, y - 9), (306, y - 6), 0, -1)
        cv2.rectangle(bitonal, (312, y - 9), (315, y - 6), 0, -1)
    assert cluster_rotated(glyph_components(bitonal, WORK_DPI), WORK_DPI) == []


def test_полутоновый_растр_даёт_много_точек_а_подпись_нет() -> None:
    gray = _page600()
    label = Box(1790, 590, 1880, 1500)
    assert speck_density(gray, label, READ_DPI) < 1.0
    screen = np.full((600, 600), 255, np.uint8)
    for y in range(0, 600, 8):
        for x in range(0, 600, 8):
            screen[y : y + 3, x : x + 3] = 0  # растр 75 lpi, точки 0.13 мм
    assert speck_density(screen, Box(250, 150, 300, 450), READ_DPI) > 1.0
