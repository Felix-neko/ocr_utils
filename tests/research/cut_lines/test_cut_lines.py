"""Тесты стенда дорисовки срезанных строк на синтетике: продление поля, признак строки текста, корешок, цвет вклейки."""

import numpy as np

from research.cut_lines.compose import compose, extend_background
from research.cut_lines.lines import is_text_line, spine_zone
from research.cut_lines.register import extend_field


def text_strip(n_glyphs: int = 12, height: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Синтетическая строка: ``n_glyphs`` прямоугольных «букв» с промежутками.

    Args:
        n_glyphs: Число букв.
        height: Высота букв, px.

    Returns:
        ``(маска сгустка-строки, краска)`` в рамке строки.
    """
    ink = np.zeros((height + 4, n_glyphs * 12 + 4), bool)
    for i in range(n_glyphs):
        ink[2 : 2 + height, 2 + i * 12 : 2 + i * 12 + 7] = True
    return np.ones_like(ink), ink


def test_text_line_accepted() -> None:
    """Строка из 12 букв высотой 14 px — строка текста."""
    comp, ink = text_strip()
    assert is_text_line(comp, ink, 14.0)


def test_stack_edge_rejected() -> None:
    """Длинная сплошная полоса (линия между листами торца) — не строка: глифов нет."""
    comp = np.ones((10, 400), bool)
    ink = np.zeros_like(comp)
    ink[4:6, :] = True
    assert not is_text_line(comp, ink, 14.0)


def test_vertical_stripe_rejected() -> None:
    """Вертикальный сгусток (торец стопки вдоль страницы) — не строка."""
    comp, ink = text_strip()
    assert not is_text_line(comp.T.copy(), ink.T.copy(), 14.0)


def test_spine_zone_on_side_of_neighbour() -> None:
    """Полоса корешка — у края страницы, обращённого к соседней."""
    page = np.zeros((100, 300), bool)
    page[10:90, 160:290] = True
    other = np.zeros_like(page)
    other[10:90, 10:150] = True
    zone = spine_zone(page, other)
    assert zone[50, 160] and not zone[50, 280]


def test_extend_field_continues_shift() -> None:
    """Постоянный сдвиг, измеренный в надёжной зоне, продлевается за её край."""
    h, w = 120, 160
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    reliable = np.zeros((h, w), bool)
    reliable[:70] = True
    # Измеренное поле: сдвиг +5 по y в надёжной зоне, мусор вне её.
    map_x, map_y = gx.copy(), gy + 5
    map_y[70:] = gy[70:] + 40
    out_x, out_y = extend_field(gx, gy, map_x, map_y, reliable)
    assert abs(float(out_y[110, 80] - gy[110, 80]) - 5) < 1.0
    assert abs(float(out_x[110, 80] - gx[110, 80])) < 1.0


def test_extend_background_nearest() -> None:
    """Фон продлевается за пределы надёжной зоны значением соседней бумаги."""
    bg = np.zeros((40, 40, 3), np.float32)
    bg[:20] = 200
    valid = np.zeros((40, 40), bool)
    valid[:20] = True
    out = extend_background(bg, valid)
    assert out[35, 20].mean() > 150


def test_compose_takes_paper_tone_from_base() -> None:
    """Вклейка: бумага донора получает тон бумаги основы, буквы донора остаются тёмными."""
    h, w = 200, 200
    base = np.zeros((h, w, 3), np.uint8)
    base[:120] = (170, 190, 210)  # бумага основы, ниже 120 — коврик
    donor = np.full((h, w, 3), 240, np.uint8)  # донор светлее
    donor[140:150, 40:160:10] = 40  # «буквы» срезанной строки
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    alpha = np.zeros((h, w), np.float32)
    alpha[130:160, 30:170] = 1
    paper = np.zeros((h, w), bool)
    paper[:120] = True
    cut = np.zeros((h, w), bool)
    cut[138:152, 38:162] = True
    out = compose(base, donor, gx, gy, alpha, paper, cut, np.zeros_like(cut), 1.0, 1.0)
    assert out is not None
    img = out.image
    # Бумага вклейки — тон основы (а не белый донора), буква — тёмная.
    assert np.abs(img[155, 45].astype(int) - (170, 190, 210)).max() < 25
    assert img[145, 40].mean() < 80
    # Вне альфы — без изменений.
    assert (img[50, 50] == base[50, 50]).all()
