"""Уменьшение и нарезка полосы на куски."""

import base64

from PIL import Image

from research.external_ocr_models.imaging import prepare


def _page(tmp_path, size=(1000, 3000)):
    path = tmp_path / "IMG_0001_1L.jpg"
    Image.new("RGB", size, (200, 190, 170)).save(path, quality=90)
    return path


def test_downscale_to_max_side(tmp_path):
    (image,) = prepare(_page(tmp_path), max_side=600)
    assert (image.width, image.height) == (200, 600)
    assert image.data_url().startswith("data:image/jpeg;base64,")
    assert base64.b64decode(image.data_url().split(",", 1)[1]) == image.data


def test_small_image_not_upscaled(tmp_path):
    (image,) = prepare(_page(tmp_path, (300, 400)), max_side=2200)
    assert (image.width, image.height) == (300, 400)


def test_strips_overlap_and_cover_page(tmp_path):
    pieces = prepare(_page(tmp_path), max_side=10000, strips=3, overlap=0.1)
    assert len(pieces) == 3
    heights = [piece.height for piece in pieces]
    # крайние куски получают перекрытие с одной стороны, средний — с двух
    assert heights[0] == 1000 + 150 and heights[1] == 1000 + 300 and heights[2] == 1000 + 150
    assert all(piece.width == 1000 for piece in pieces)
