"""Тесты карты смаза по зонам: смазанный угол даёт зону там, где он есть; растр и цветная плашка отсеиваются."""

import cv2
import numpy as np

from research.spread_defects.blur_zones import (
    Band,
    Side,
    blur_zones,
    issue_midtone_level,
    print_flags,
    zone_mask,
    zone_maps,
)
from research.spread_defects.motion import TILE, MidtoneTiles, MotionTiles


def _grids(rows: int = 12, cols: int = 16) -> tuple[MotionTiles, MidtoneTiles]:
    """Сетки кадра-разворота: текст по всем плиткам, кроме рамки в одну плитку; ореол 0.2, кепстр молчит."""
    text = np.zeros((rows, cols), bool)
    text[1:-1, 1:-1] = True
    ink = np.where(text, 0.15, 0.0).astype(np.float32)
    lines = np.zeros((rows, cols), np.float32)
    mid = np.where(text, 0.2, 0.0).astype(np.float32)
    zeros = np.zeros((rows, cols), np.float32)
    tiles = MotionTiles(
        text, ink, lines, zeros, zeros.astype(np.int16), zeros.astype(np.int16), zeros, np.zeros((4, 4), np.float16)
    )
    return tiles, MidtoneTiles(text, ink, lines, mid)


def test_sharp_frame_has_no_zones():
    """Ровный ореол по всему кадру — зон нет."""
    tiles, mid = _grids()
    fold = 8 * TILE
    level = issue_midtone_level([mid], [zone_mask(mid, fold)])
    assert blur_zones(zone_maps(tiles, mid, fold, level), fold) == []


def test_blurred_corner_found_at_corner():
    """Ореол втрое выше в верхнем правом углу текста — одна зона: правая полоса, верх."""
    tiles, mid = _grids()
    blurred = mid.mid.copy()
    blurred[1:4, 11:15] = 0.6
    mid = MidtoneTiles(mid.text, mid.ink, mid.lines, blurred)
    fold = 8 * TILE
    # Уровень выпуска — по резким кадрам (смаз одного кадра медиану выпуска не сдвигает).
    _, sharp = _grids()
    level = issue_midtone_level([sharp], [zone_mask(sharp, fold)])
    zones = blur_zones(zone_maps(tiles, mid, fold, level), fold)
    assert len(zones) == 1
    zone = zones[0]
    assert zone.side is Side.RIGHT and zone.band is Band.TOP
    assert zone.row0 <= 2 and zone.col1 >= 13
    assert zone.mid_peak > 2.0 and zone.by_midtone and not zone.by_cepstrum


def test_excluded_tiles_drop_zone():
    """Те же плитки, помеченные полиграфией, — зоны нет."""
    tiles, mid = _grids()
    blurred = mid.mid.copy()
    blurred[1:4, 11:15] = 0.6
    mid = MidtoneTiles(mid.text, mid.ink, mid.lines, blurred)
    fold = 8 * TILE
    excluded = np.zeros(mid.text.shape, bool)
    excluded[1:4, 11:15] = True
    assert blur_zones(zone_maps(tiles, mid, fold, 0.2, excluded), fold) == []


def test_print_flags_halftone_colour_text():
    """Растр — мелкие точки, голубая плашка — цвет, текст на кремовой бумаге — ни то ни другое."""
    paper = np.full((TILE, TILE, 3), (238, 232, 215), np.uint8)
    halftone = paper.copy()
    for y in range(4, TILE, 8):
        for x in range(4, TILE, 8):
            cv2.circle(halftone, (x, y), 2, (30, 30, 30), -1)
    plate = np.full((TILE, TILE, 3), (150, 200, 240), np.uint8)
    text = paper.copy()
    for y in range(40, TILE, 50):
        cv2.putText(text, "word text", (5, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3, cv2.LINE_AA)
    assert print_flags(halftone).is_print()
    assert print_flags(plate).is_print()
    assert not print_flags(text).is_print()


def test_heading_zone_against_ring():
    """Зона с кеглем вдвое крупнее кольца вокруг — заголовок; ровно крупный кегль по всей полосе (перспектива) — нет."""
    from research.spread_defects.blur_zones import BlurZone, heading_ratio, is_heading

    zone = BlurZone(4, 5, 4, 5, 4, Side.LEFT, Band.MIDDLE, 2.0, 2.0, 0.0, True, False, 4, 4)
    heights = np.full((10, 10), 30.0, np.float32)
    heights[4:6, 4:6] = 60.0
    assert is_heading(heading_ratio(zone, heights, float("nan")))
    # Поднятая страница: кегль плавно растёт к краю, зона не выделяется ни из кольца, ни из полосы.
    ramp = np.tile(np.linspace(30, 39, 10, dtype=np.float32), (10, 1))
    assert not is_heading(heading_ratio(zone, ramp, float("nan")))
    # Крупным кеглем набрана половина полосы: кольцо тоже крупное, но основной набор полосы мелкий.
    half = np.full((10, 10), 28.0, np.float32)
    half[:7] = 41.0
    to_ring, to_body = heading_ratio(zone, half, float("nan"))
    assert to_ring < 1.2 and is_heading((to_ring, to_body))


def test_print_mask_returns_letter_heights():
    """Высота букв: у крупного кегля больше, чем у мелкого; растр и цвет от высоты не зависят."""
    from research.spread_defects.blur_zones import print_mask

    rows, cols = 2, 2
    page = np.full((rows * TILE, cols * TILE, 3), (238, 232, 215), np.uint8)
    for col, (scale, thick, step) in enumerate(((3.0, 6, 110), (1.0, 2, 40))):
        for y in range(step, rows * TILE - 10, step):
            cv2.putText(page, "text word", (col * TILE + 4, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), thick)
    excluded, heights = print_mask(page, np.ones((rows, cols), bool))
    assert not excluded.any()
    assert np.nanmedian(heights[:, 0]) > 2 * np.nanmedian(heights[:, 1])
