"""Сетка тайлов: число, порядок, перекрытие, покрытие кадра и уменьшение под модель."""

import base64

import pytest
from PIL import Image

from ocr_utils.external_ocr_services.tiling import (
    DEFAULT_MAX_SRC_TILE_MM,
    TILE_OVERLAP,
    describe,
    grid,
    grid_shape,
    prepare_tiles,
)
from ocr_utils.units import mm_to_px, px_to_mm


@pytest.mark.parametrize(
    "size, expected",
    [
        ((3600, 5900), (1, 2)),  # обычная полоса пака-1
        ((4499, 6692), (1, 2)),  # самая широкая и самая высокая обычная
        ((6444, 5336), (2, 2)),  # склеенный разворот 1967/10/IMG_0041
        ((6000, 3300), (2, 1)),  # полоса, повёрнутая на 90°
        ((300, 400), (1, 1)),
    ],
)
def test_grid_shape_for_pack1_sizes(size, expected):
    assert grid_shape(*size, 4500) == expected


def test_grid_order_is_column_major_and_covers_frame():
    boxes = grid(6444, 5336, 4500)
    assert [(b.col, b.row) for b in boxes] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert boxes[0].left == 0 and boxes[0].top == 0
    assert boxes[-1].right == 6444 and boxes[-1].bottom == 5336
    # Соседи по вертикали и по горизонтали перекрываются не меньше константы.
    assert boxes[0].bottom - boxes[1].top >= int(5336 * TILE_OVERLAP) - 1
    assert boxes[0].right - boxes[2].left >= int(6444 * TILE_OVERLAP) - 1


def test_no_overlap_on_axis_with_single_tile():
    boxes = grid(3600, 5900, 4500)
    assert all(b.left == 0 and b.right == 3600 for b in boxes)
    assert boxes[0].bottom - boxes[1].top == 2 * int(5900 * TILE_OVERLAP / 2)


def test_prepare_tiles_downscales_and_keeps_boxes(tmp_path):
    path = tmp_path / "IMG_0001_1L.jpg"
    Image.new("RGB", (1000, 3000), (200, 190, 170)).save(path, quality=90, dpi=(600, 600))
    tiles = prepare_tiles(path, max_src_tile_mm=px_to_mm(2000, 600), max_model_tile=600)
    assert [(t.box.col, t.box.row) for t in tiles] == [(0, 0), (0, 1)]
    assert all(max(t.width, t.height) == 600 for t in tiles)
    assert tiles[0].data_url().startswith("data:image/jpeg;base64,")
    assert base64.b64decode(tiles[0].data_url().split(",", 1)[1]) == tiles[0].data
    info = describe(tiles)
    assert (info.ncols, info.nrows) == (1, 2) and info.tiles[1].src[3] == 3000
    assert info.as_dict()["tiles"][1]["src"][3] == 3000  # в meta уходит плоский словарь


def test_small_image_single_tile_not_upscaled(tmp_path):
    path = tmp_path / "small.png"
    Image.new("L", (300, 400), 230).save(path, dpi=(600, 600))
    (tile,) = prepare_tiles(path, max_model_tile=2200)
    assert (tile.width, tile.height) == (300, 400) and tile.box.size == (300, 400)


def test_src_tile_step_is_paper_size_not_pixels():
    """Шаг сетки задан в мм бумаги: при 600 dpi — прежние 4500 px, при 450 — 3375 px.

    Обычная полоса пака-1 (≈ 3450×6000 при 600 dpi) режется 1×2; та же полоса в 450 dpi
    (≈ 2588×4500) обязана резаться так же, а не уйти в модель одним куском.
    """
    assert mm_to_px(DEFAULT_MAX_SRC_TILE_MM, 600) == 4500
    assert mm_to_px(DEFAULT_MAX_SRC_TILE_MM, 450) == 3375
    assert grid_shape(3450, 6000, mm_to_px(DEFAULT_MAX_SRC_TILE_MM, 600)) == (1, 2)
    assert grid_shape(2588, 4500, mm_to_px(DEFAULT_MAX_SRC_TILE_MM, 450)) == (1, 2)


def test_prepare_tiles_reads_dpi_from_tag_and_rejects_untagged(tmp_path):
    """Без тега dpi и без ``default_dpi`` полоса не режется: молча считать её 600 dpi нельзя."""
    path = tmp_path / "IMG_0002_1L.jpg"
    Image.new("L", (2588, 4500), 230).save(path, dpi=(450, 450))
    assert [(t.box.col, t.box.row) for t in prepare_tiles(path, max_model_tile=600)] == [(0, 0), (0, 1)]

    untagged = tmp_path / "untagged.png"
    Image.new("L", (2588, 4500), 230).save(untagged)
    with pytest.raises(ValueError):
        prepare_tiles(untagged, max_model_tile=600)
    assert len(prepare_tiles(untagged, max_model_tile=600, default_dpi=450)) == 2
