"""Синтетические полосы для проверки детекторов ориентации.

Полосы рисуются в масштабе 300 dpi: детекторы калиброваны по копии 150 dpi, а ``read_frame``
делает её уменьшением вдвое — значит, на вход надо подавать вдвое крупнее.

ГЛАВНОЕ ТРЕБОВАНИЕ к такой полосе — она не должна быть решёткой. Буквы одинаковой ширины,
расставленные с постоянным шагом, дают профиль ПОПЕРЁК строки периодичнее, чем вдоль неё,
и детектор ``profile`` на такой полосе честно находит межбуквенный период и объявляет текст
вертикальным. У настоящего набора ширина букв разная, а между словами дыры, поэтому
переменные ширины и пробелы здесь — условие того, чтобы тест мерил детектор, а не рисовалку.
"""

from __future__ import annotations

import random

import numpy as np

from tests.ocr_utils.scan_markup.synthetic import INK, paper

# Размер полосы и набора — по настоящей полосе пака-1: копия 300 dpi там 1710x3036,
# межстрочный шаг около 22 px на копии 150 dpi, строк на полосе 60-140. Число строк здесь
# важно не для красоты: уверенность ink_axis в СТОРОНЕ поворота гасится по числу строк
# (SIGN_FULL_LINES=60), и на вчетверо меньшей полосе детектор честно отвечал бы вполсилы —
# тест ловил бы этот эффект вместо того, что проверяет.
PAGE_SHAPE = (3036, 1710)
LINE_STEP = 44
GLYPH_H = 22
MARGIN = 120

# Ширина буквы и межбуквенный просвет, в пикселях масштаба 300 dpi. Разброс взят таким,
# чтобы никакого одного периода в строке не было вовсе.
GLYPH_W_RANGE = (6, 15)
GAP_RANGE = (3, 9)

# Раз в столько букв — пробел между словами. Ширина замерена по настоящему набору: пробел
# кегля 10 пунктов это около 0.9 мм, то есть 5 px на копии 150 dpi и 10 px здесь. Ставить
# его шире нельзя: смыкание RLSA работает с зазором 8 px на копии 150 dpi, и от слишком
# широкого пробела строка распадается на отдельные слова, которые ни один детектор строкой
# уже не считает — тест мерил бы рисовалку, а не детектор.
SPACE_EVERY = 6
SPACE_W = 10

# Каждая пятая буква с нижним выносным — как «р», «у», «д», «ц» в кириллице. Без выносных
# асимметрия строки равна нулю, и сторону поворота проверять было бы не на чем.
DESCENDER_EVERY = 5
DESCENDER_H = 10


def text_page(shape: tuple[int, int] = PAGE_SHAPE, descenders: bool = True, seed: int = 0) -> np.ndarray:
    """Полоса сплошного текста; при ``descenders`` — с нижними выносными."""
    image = paper(shape)
    generator = random.Random(seed)
    right = shape[1] - MARGIN
    index = 0
    for y in range(MARGIN, shape[0] - MARGIN - GLYPH_H - DESCENDER_H, LINE_STEP):
        x = MARGIN
        while True:
            width = generator.randint(*GLYPH_W_RANGE)
            if x + width > right:
                break
            image[y : y + GLYPH_H, x : x + width] = INK
            if descenders and index % DESCENDER_EVERY == 0:
                # Вниз уходит хвост, а не всё очко: от его ширины и зависит асимметрия.
                image[y + GLYPH_H : y + GLYPH_H + DESCENDER_H, x : x + max(2, width // 3)] = INK
            index += 1
            x += width + generator.randint(*GAP_RANGE) + (SPACE_W if index % SPACE_EVERY == 0 else 0)
    return image


def blank_page(shape: tuple[int, int] = PAGE_SHAPE) -> np.ndarray:
    return paper(shape)


def line_art_page(shape: tuple[int, int] = PAGE_SHAPE) -> np.ndarray:
    """Чертёж без текста: длинные штрихи, ни одной компоненты размера буквы."""
    image = paper(shape)
    for y in range(MARGIN, shape[0] - MARGIN, 130):
        image[y : y + 3, MARGIN : shape[1] - MARGIN] = INK
    for x in range(MARGIN, shape[1] - MARGIN, 130):
        image[MARGIN : shape[0] - MARGIN, x : x + 3] = INK
    return image
