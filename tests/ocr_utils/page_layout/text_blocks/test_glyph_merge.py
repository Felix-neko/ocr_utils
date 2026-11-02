"""Склейка перекрытых по x компонент маски глифов (``glyph_merge``): точка «i», кратка «й», оторванная ветвь «к» — одна буква; буквы соседних строк и соседние буквы ряда — нет."""

from __future__ import annotations

import numpy as np

from ocr_utils.page_layout.text_blocks import glyph_merge
from ocr_utils.page_layout.text_blocks.blocks import _split_by_measure
from ocr_utils.page_layout.text_blocks.glyph_merge import GlyphMerge, merged_components
from ocr_utils.page_layout.text_blocks.segment import SCALES, _cut_tall_glyphs, _glyph_rects
from ocr_utils.page_layout.text_blocks.typeset import glyphs_kegl
from tests.ocr_utils.page_layout.text_blocks.test_typeset import axis_of, word

# «ко-» из строки «…автомобильных путей, ко-» полосы 1968/05 IMG_0074_1L (маска Оцу рабочей копии 150 dpi,
# y 785–794): тело «к» (10 px), оторванная верхняя правая ветвь «к» (3 px, x внутри тела), «о» (10 px),
# дефис (3 px). Ветвь — столбцы 8–11 строк 0–2, отделённые от стойки двумя пустыми столбцами.
KO_HYPHEN = (
    "..#####..###....######..........",
    "..#####.####...########.........",
    "...###..####...##....###........",
    "...###........###....###..######",
    "...######.....###....###..######",
    "...###.###....###....###..####..",
    "...###..###...###....###........",
    "...###..###....###...##.........",
    "...###..####...########.........",
    "..####...###.....####...........",
)
# Где рисуется «ко-» на синтетической странице (пиксели рабочей копии): левый верхний угол.
KO_X0, KO_Y0 = 120, 40


def _pattern(rows: tuple[str, ...]) -> np.ndarray:
    """Маска ``uint8`` из строк ASCII: ``#`` — краска (255), ``.`` — бумага."""
    return np.array([[255 if char == "#" else 0 for char in row] for row in rows], dtype=np.uint8)


def _ko_mask() -> np.ndarray:
    """Маска 100 × 200 px с одним сгустком «ко-» в точке ``(KO_X0, KO_Y0)``."""
    mask = np.zeros((100, 200), dtype=np.uint8)
    ko = _pattern(KO_HYPHEN)
    mask[KO_Y0 : KO_Y0 + ko.shape[0], KO_X0 : KO_X0 + ko.shape[1]] = ko
    return mask


def _heights(mask: np.ndarray, mode: GlyphMerge) -> list[int]:
    """Высоты букв маски слева направо при заданном способе склейки."""
    count, _, stats, _ = merged_components(mask, mode)
    order = np.argsort(stats[1:, 0])
    return [int(value) for value in stats[1:, 3][order]]


def test_broken_k_branch_is_one_letter():
    """Ветвь «к» внутри тела по x склеивается с ним: у «ко-» три буквы 10, 10, 3 вместо четырёх."""
    assert _heights(_ko_mask(), GlyphMerge.NONE) == [10, 3, 10, 3]
    assert _heights(_ko_mask(), GlyphMerge.OVERLAP) == [10, 10, 3]


def test_dot_and_breve_join_letter():
    """Точка над «i» и кратка над «й» — одна буква с ней; бокс — от верха метки до низа буквы."""
    mask = np.zeros((40, 60), dtype=np.uint8)
    mask[16:26, 10:13] = 255  # стойка «i», 10 px
    mask[12:14, 10:13] = 255  # точка над ней, просвет 2 px
    mask[16:26, 30:38] = 255  # тело «й»
    mask[13:15, 31:37] = 255  # кратка, просвет 1 px
    count, labels, stats, _ = merged_components(mask, GlyphMerge.OVERLAP)
    assert count - 1 == 2
    assert sorted(int(h) for h in stats[1:, 3]) == [13, 14]
    # Пиксели маски не меняются: каждый пиксель краски — в своей букве, площадь сохраняется.
    assert int(stats[1:, 4].sum()) == int((mask > 0).sum())
    assert np.array_equal(labels > 0, mask > 0)


def test_side_by_side_parts_are_not_glued():
    """«ы» (две компоненты рядом, не перекрыты по x) и слегка налезающие буквы курсива не склеиваются."""
    mask = np.zeros((40, 80), dtype=np.uint8)
    mask[16:26, 10:16] = 255  # «ь» из «ы»
    mask[16:26, 18:20] = 255  # палочка «ы» рядом
    # Две буквы курсива: верх левой и низ правой заходят друг за друга по x на 2 px, краски не касаются.
    mask[16:26, 40:46] = 255
    mask[16, 46:49] = 255
    mask[16:26, 50:56] = 255
    mask[25, 47:50] = 255
    assert len(_heights(mask, GlyphMerge.NONE)) == 4
    assert len(_heights(mask, GlyphMerge.OVERLAP)) == 4


def test_letters_of_neighbor_rows_are_not_glued():
    """Буква над буквой соседней строки и дефис над буквой строки ниже остаются отдельными."""
    mask = np.zeros((80, 60), dtype=np.uint8)
    # Шаг строк 18 px при букве 10 px (как на корпусе пака-1): «о» над «о».
    mask[10:20, 10:17] = 255
    mask[28:38, 10:17] = 255
    # Дефис строки выше (3 px на середине икса) над буквой строки ниже.
    mask[14:17, 30:36] = 255
    mask[28:38, 30:37] = 255
    # Выносной «р» строки выше (14 px) над дефисом строки ниже.
    mask[10:24, 45:51] = 255
    mask[32:35, 45:51] = 255
    assert len(_heights(mask, GlyphMerge.OVERLAP)) == 6


def test_ko_is_not_cut_as_two_row_glyph(monkeypatch):
    """Без склейки кегль сгустка «ко-» — медиана [3, 10, 3] = 3, и «к» режется как сросшийся глиф; со склейкой — нет."""
    mask = _ko_mask()
    monkeypatch.setattr(glyph_merge, "DEFAULT_GLYPH_MERGE", GlyphMerge.NONE)
    assert not np.array_equal(_cut_tall_glyphs(mask, SCALES[0]), mask)
    monkeypatch.setattr(glyph_merge, "DEFAULT_GLYPH_MERGE", GlyphMerge.OVERLAP)
    assert np.array_equal(_cut_tall_glyphs(mask, SCALES[0]), mask)


def _ko_axis_glyphs() -> np.ndarray:
    """Боксы глифов оси «ко-», как их строит разбор: резка сросшихся глифов, затем ``_glyph_rects``."""
    mask = _cut_tall_glyphs(_ko_mask(), SCALES[0])
    labels = np.ones(mask.shape, dtype=np.int32)
    return _glyph_rects(mask, labels, [1], 0, 0, mask.shape[1], mask.shape[0])


def test_ko_keeps_row_with_its_line(monkeypatch):
    """«ко-» в 10 px от «путей,» (кегль 10): со склейкой кегль оси 10 и ряд не режется; без неё — 3 и режется."""
    bottom = float(KO_Y0 + len(KO_HYPHEN))
    line = axis_of(word(10.0, 12, 10.0, bottom))
    gap_x = float(line.glyphs[:, 2].max()) + 10.0
    monkeypatch.setattr(glyph_merge, "DEFAULT_GLYPH_MERGE", GlyphMerge.NONE)
    before = _ko_axis_glyphs()
    monkeypatch.setattr(glyph_merge, "DEFAULT_GLYPH_MERGE", GlyphMerge.OVERLAP)
    after = _ko_axis_glyphs()
    assert glyphs_kegl(before) <= 4.0
    assert glyphs_kegl(after) == 10.0
    for glyphs, parts in ((before, 2), (after, 1)):
        shifted = glyphs + np.array([gap_x - KO_X0, 0.0, gap_x - KO_X0, 0.0])
        group = [line, axis_of(shifted)]
        assert len(_split_by_measure(group, None, 2.0, 150.0, by_boxes=True)) == parts


def test_real_two_row_glyph_is_still_cut():
    """Хвост буквы верхней строки, сросшийся с буквой нижней, режется и со склейкой (части — полноразмерные буквы)."""
    mask = np.zeros((140, 400), dtype=np.uint8)
    for top in (20, 43):
        for x in range(10, 382, 11):
            mask[top : top + 12, x : x + 8] = 255
    mask[32:43, 153:155] = 255  # тонкая перемычка между буквами двух рядов
    cut = _cut_tall_glyphs(mask, SCALES[0])
    assert merged_components(cut)[0] == merged_components(mask)[0] + 1
