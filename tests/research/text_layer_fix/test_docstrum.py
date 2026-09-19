"""Боковой текст по соседям глифов: синтетика с прямым и повёрнутым текстом."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from research.text_layer_fix import WORK_DPI
from research.text_layer_fix.docstrum import cluster_rotated, glyph_components

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _page(dpi: int = WORK_DPI) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Страница 150 dpi: абзац прямого текста и одна вертикальная подпись справа."""
    image = Image.new("L", (1000, 1400), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, 22)
    for i in range(12):
        draw.text((80, 100 + i * 34), "Материально техническое снабжение и сбыт", fill=0, font=font)
    # Колонка чисел с шагом строки (первые знаки строк стоят друг над другом — не боковой текст).
    for i in range(12):
        draw.text((40, 100 + i * 34), str(i + 1), fill=0, font=font)
    label = Image.new("L", (300, 30), 255)
    ImageDraw.Draw(label).text((2, 2), "рампа с навесом", fill=0, font=font)
    rotated = label.rotate(90, expand=True)  # читается снизу вверх
    image.paste(rotated, (900, 500))
    box = (900, 500, 930, 800)
    return np.asarray(image), box


def test_finds_vertical_label_only() -> None:
    gray, expected = _page()
    stats = glyph_components(gray, WORK_DPI)
    assert len(stats) > 100
    zones = cluster_rotated(stats, WORK_DPI)
    assert len(zones) == 1, [z.as_tuple() for z in zones]
    zone = zones[0]
    assert abs(zone.x0 - expected[0]) < 12 and abs(zone.x1 - expected[2]) < 12
    assert zone.y0 >= expected[1] - 5 and zone.y1 <= expected[3] + 5
    assert zone.height > 3 * zone.width


def test_upright_lines_by_transposition() -> None:
    gray, _ = _page()
    stats = glyph_components(gray, WORK_DPI)
    from research.text_layer_fix.docstrum import cluster_lines

    lines = cluster_lines(stats, WORK_DPI, vertical=False)
    # Двенадцать строк абзаца (могут склеиться в один блок) и ни одной вертикальной подписи.
    assert lines, "прямые строки не найдены"
    for line in lines:
        assert line.width > line.height, line.as_tuple()
        assert line.x1 < 890, "вертикальная подпись не должна попасть в прямые строки"
    covered = sum(z.width * z.height for z in lines)
    assert covered > 300 * 100
