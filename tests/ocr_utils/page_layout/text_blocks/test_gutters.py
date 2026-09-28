"""Межколонники как запреты сцепки: наклонная полоса, запреты по отрезкам ломаной и мера «ось через межколонник»."""

from __future__ import annotations

import random

import cv2
import numpy as np

from ocr_utils.page_layout.orientation.detectors.ink_axis import glyph_mask
from ocr_utils.page_layout.text_blocks import RENDER_DPI, WORK_DPI
from ocr_utils.page_layout.text_blocks.columns import Gutter, GutterMode, separators_for_segmentation
from ocr_utils.page_layout.text_blocks.engines.ink import InkEngine
from ocr_utils.page_layout.text_blocks.metrics import gutter_crossings_of
from ocr_utils.page_layout.text_blocks.page import analyse_gray
from ocr_utils.page_layout.text_blocks.report import page_json
from tests.ocr_utils.page_layout.orientation.synthetic import LINE_STEP, MARGIN, PAGE_SHAPE
from tests.ocr_utils.page_layout.text_blocks.synthetic import GUTTER_PX, _draw_line, column_page
from tests.ocr_utils.scan_markup.synthetic import INK, PAPER, paper

# Наклон полосы (градусы): на высоте синтетической страницы межколонник уезжает вбок больше своей
# ширины — ровно случай 1966/02 IMG_0076_1L (~1°, ломаная 426..451 → 452..479).
TILT_DEG = 1.5

# Межколонник-ломаная с наклоном, как на 1966/02 IMG_0076_1L (пиксели рабочей копии): узлы через ленту.
SLANTED = Gutter(
    points=(
        (0.0, 426.0, 451.0),
        (118.0, 426.0, 451.0),
        (472.0, 433.0, 459.0),
        (944.0, 443.0, 470.0),
        (1416.0, 452.0, 479.0),
        (1583.0, 452.0, 479.0),
    )
)


# Заголовок во всю ширину: высота букв, ширина слов и пробелов (пиксели 300 dpi).
HEADING_H = 60
HEADING_WORD = (140, 260)
HEADING_SPACE = 40


def _heading(image: np.ndarray, y: int, generator: random.Random) -> None:
    """Строка заголовка крупным кеглем во всю ширину; одно слово нарочно ложится на межколонник."""
    middle = image.shape[1] // 2
    x = MARGIN
    while x < image.shape[1] - MARGIN:
        width = generator.randint(*HEADING_WORD)
        # Слово, накрывающее межколонник: заголовок перегораживает его, как на 1969/03 IMG_0147_1L.
        if x < middle < x + width + HEADING_SPACE:
            x0 = middle - width // 2
            image[y : y + HEADING_H, x0 : x0 + width] = INK
            x = x0 + width + HEADING_SPACE
            continue
        image[y : y + HEADING_H, x : min(x + width, image.shape[1] - MARGIN)] = INK
        x += width + HEADING_SPACE


def fragment_page(fragment_lines: int = 5, seed: int = 3) -> np.ndarray:
    """Две колонки, заголовок во всю ширину и под ним короткий двухколонный фрагмент в ``fragment_lines`` строк.

    Межколонник над заголовком длинный и находится прежним ходом; под заголовком он короче двух
    лент по 40 мм и прежним ходом не находится (1969/03 IMG_0147_1L).
    """
    image = paper(PAGE_SHAPE)
    generator = random.Random(seed)
    width = (PAGE_SHAPE[1] - 2 * MARGIN - GUTTER_PX) // 2
    columns = ((MARGIN, MARGIN + width), (MARGIN + width + GUTTER_PX, PAGE_SHAPE[1] - MARGIN))
    heading_y = 1900
    for y in range(MARGIN, heading_y - 2 * LINE_STEP, LINE_STEP):
        for x0, x1 in columns:
            _draw_line(image, generator, x0, x1, y, True)
    _heading(image, heading_y, generator)
    start = heading_y + HEADING_H + 2 * LINE_STEP
    for index in range(fragment_lines):
        for x0, x1 in columns:
            _draw_line(image, generator, x0, x1, start + index * LINE_STEP, True)
    return image


def _tilted(image: np.ndarray, degrees: float) -> np.ndarray:
    """Страница, повёрнутая на ``degrees`` вокруг центра; углы добиты бумагой."""
    height, width = image.shape
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=int(PAPER))


def _inside(gutter: Gutter, band: tuple[int, int, int, int]) -> bool:
    """Лежит ли полоса запрета внутри межколонника на всей своей высоте."""
    x0, x1, y0, y1 = band
    return all(gutter.x0_at(y) <= x0 and x1 <= gutter.x1_at(y) for y in np.linspace(y0, y1, 9))


def test_legacy_separator_drops_slanted_gutter():
    """Прежний ход: у наклонной ломаной общей части по x нет — межколонник выпадает из запретов."""
    assert separators_for_segmentation([SLANTED], GutterMode.LEGACY) == []


def test_segmented_separators_cover_slanted_gutter():
    """По отрезкам: запреты есть, лежат внутри межколонника и покрывают всю его высоту без дыр."""
    bands = separators_for_segmentation([SLANTED], GutterMode.SEGMENTED)
    assert bands
    assert all(_inside(SLANTED, band) for band in bands)
    spans = sorted((band[2], band[3]) for band in bands)
    assert spans[0][0] <= SLANTED.y0 and spans[-1][1] >= SLANTED.y1
    assert all(later[0] <= earlier[1] for earlier, later in zip(spans, spans[1:]))


def test_segment_without_common_part_is_halved():
    """Отрезок, у концов которого нет общей пустоты, делится по высоте, а не выбрасывается."""
    gutter = Gutter(points=((0.0, 100.0, 110.0), (200.0, 120.0, 130.0)))
    bands = separators_for_segmentation([gutter], GutterMode.SEGMENTED)
    assert len(bands) >= 2
    assert all(_inside(gutter, band) for band in bands)


def _through(analysis, middle: float) -> int:
    """Сколько осей заходит за середину межколонника с обеих сторон больше чем на 5 мм."""
    reach = 5.0 * WORK_DPI / 25.4
    return sum(1 for axis in analysis.axes if axis.x0 < middle - reach and axis.x1 > middle + reach)


def _crossings(analysis, gray300: np.ndarray) -> int:
    """Мера «ось через межколонник» по разбору, как на стенде."""
    work = cv2.resize(gray300, (analysis.width, analysis.height), interpolation=cv2.INTER_AREA)
    return len(gutter_crossings_of(page_json(analysis)["axes"], glyph_mask(work), WORK_DPI))


def test_tilted_two_columns_do_not_stitch():
    """Две колонки с наклоном 1.5°: по отрезкам строки не идут через межколонник, блоков два.

    На прежнем ходе та же полоса сшивается через межколонник — мера это видит.
    """
    page = _tilted(column_page(columns=2), TILT_DEG)
    middle = page.shape[1] / 2.0 * WORK_DPI / RENDER_DPI
    legacy = analyse_gray(page, InkEngine(gutter_mode=GutterMode.LEGACY))
    fixed = analyse_gray(page, InkEngine(gutter_mode=GutterMode.SEGMENTED))
    assert _through(legacy, middle) > 0, "прежний ход не воспроизводит сшивку — тест ничего не проверяет"
    assert _crossings(legacy, page) > 0
    assert _through(fixed, middle) == 0
    assert _crossings(fixed, page) == 0
    assert len(fixed.blocks) == 2


def test_straight_two_columns_same_in_both_modes():
    """Ровная полоса в две колонки: режимы дают одно и то же — оси и блоки совпадают."""
    page = column_page(columns=2)
    legacy = analyse_gray(page, InkEngine(gutter_mode=GutterMode.LEGACY))
    fixed = analyse_gray(page, InkEngine(gutter_mode=GutterMode.SEGMENTED))
    assert len(legacy.axes) == len(fixed.axes)
    assert len(legacy.blocks) == len(fixed.blocks) == 2
    assert _crossings(fixed, page) == 0


def _through_at(analysis, middle: float, y0: float, y1: float) -> int:
    """Сколько осей на высотах ``[y0, y1]`` (пиксели рабочей копии) идёт через середину межколонника."""
    reach = 5.0 * WORK_DPI / 25.4
    return sum(
        1 for axis in analysis.axes if y0 <= axis.cy <= y1 and axis.x0 < middle - reach and axis.x1 > middle + reach
    )


def test_short_fragment_under_heading_gets_its_gutter():
    """Короткий двухколонный фрагмент под заголовком: в режиме ``SHORT`` строки не идут через межколонник.

    Прежний и ``SEGMENTED`` режимы межколонника под заголовком не находят и сшивают строки фрагмента.
    Заголовок при этом не режется: его строка, набранная через межколонник, остаётся целой.
    """
    page = fragment_page()
    k = WORK_DPI / RENDER_DPI
    middle = page.shape[1] / 2.0 * k
    heading = (1900 * k, (1900 + HEADING_H) * k)
    fragment = ((1900 + HEADING_H + LINE_STEP) * k, page.shape[0] * k)
    segmented = analyse_gray(page, InkEngine(gutter_mode=GutterMode.SEGMENTED))
    short = analyse_gray(page, InkEngine(gutter_mode=GutterMode.SHORT))
    assert _through_at(segmented, middle, *fragment) > 0, "фрагмент не сшивается — тест ничего не проверяет"
    assert _through_at(short, middle, *fragment) == 0
    assert _through_at(short, middle, *heading) == 1


def test_short_mode_leaves_plain_columns_alone():
    """Обычная полоса в две колонки без заголовков: режим ``SHORT`` не находит лишних межколонников."""
    page = column_page(columns=2)
    segmented = analyse_gray(page, InkEngine(gutter_mode=GutterMode.SEGMENTED))
    short = analyse_gray(page, InkEngine(gutter_mode=GutterMode.SHORT))
    assert len(short.gutters) == len(segmented.gutters)
    assert len(short.blocks) == len(segmented.blocks) == 2
