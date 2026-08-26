"""Круговорот маски печати: CVAT -> база -> потребитель.

Формат хранения (RLE + bbox) выбран ради того, чтобы маска читалась обратно тем же
декодером CVAT, каким она пришла. Тесты проверяют, что после пересчёта в разрешение
оригинала это по-прежнему так.
"""

import numpy as np
from cvat_sdk.masks import decode_mask, encode_mask

from ocr_utils.scan_markup.cvat.export import mask_from_row, shape_to_mask
from ocr_utils.scan_markup.db.models import MASK_LIBRARY_STAMP, Page
from ocr_utils.scan_markup.geometry import mask_to_original

W, H, D = 3492, 6051, 8
CVAT_W, CVAT_H = 436, 756


class _Shape:
    """Минимальный дублёр шейпа CVAT: только то, что читает shape_to_mask."""

    def __init__(self, points, label_name: str = "Библиотечная печать", shape_id: int = 7) -> None:
        self.points = points
        self._label_name = label_name
        self.id = shape_id


def _page() -> Page:
    page = Page(
        issue_id=1,
        file_name="IMG_0004.tif",
        rel_path="1974/01/IMG_0004.tif",
        order_index=0,
        width=W,
        height=H,
        dpi=600,
        divisor=D,
        cvat_width=CVAT_W,
        cvat_height=CVAT_H,
    )
    page.id = 42
    return page


def test_stamp_round_trip_matches_upscaled_mask() -> None:
    """Маска из CVAT, сохранённая в базу и прочитанная обратно, равна апскейлу исходной."""
    drawn = np.zeros((CVAT_H, CVAT_W), bool)
    drawn[100:140, 200:260] = True
    drawn[110:130, 190:200] = True  # хвостик, чтобы форма не была прямоугольником

    row = shape_to_mask(_Shape(encode_mask(drawn)), _page())
    assert row is not None
    assert row.kind == MASK_LIBRARY_STAMP
    assert row.source_divisor == D

    restored = mask_from_row(row, W, H)
    assert np.array_equal(restored, mask_to_original(drawn, D, W, H))


def test_stored_bbox_is_in_original_coordinates() -> None:
    """bbox в базе — координаты оригинала, а не кадра CVAT."""
    drawn = np.zeros((CVAT_H, CVAT_W), bool)
    drawn[100:140, 200:260] = True

    row = shape_to_mask(_Shape(encode_mask(drawn)), _page())
    assert (row.left, row.top) == (200 * D, 100 * D)
    assert (row.width, row.height) == (60 * D, 40 * D)


def test_stamp_touching_right_edge_reaches_original_width() -> None:
    """Печать, доведённая до правого края кадра, в базе достаёт до края оригинала."""
    drawn = np.zeros((CVAT_H, CVAT_W), bool)
    drawn[100:140, CVAT_W - 20 :] = True

    row = shape_to_mask(_Shape(encode_mask(drawn)), _page())
    assert row.left + row.width == W
    assert mask_from_row(row, W, H)[100 * D, W - 1]


def test_empty_mask_is_dropped() -> None:
    """Пустая маска в базу не пишется: строка без единого пикселя бессмысленна."""
    drawn = np.zeros((CVAT_H, CVAT_W), bool)
    drawn[0, 0] = True
    points = encode_mask(drawn)
    drawn[0, 0] = False
    # Подсовываем RLE от пустой маски с тем же bbox.
    assert shape_to_mask(_Shape([0, 1, *points[-4:]]), _page()) is not None
    assert shape_to_mask(_Shape([1, *points[-4:]]), _page()) is None


def test_rle_column_is_plain_decimal_text() -> None:
    """RLE хранится как строка целых через запятую и читается без cvat_sdk."""
    drawn = np.zeros((CVAT_H, CVAT_W), bool)
    drawn[5:8, 5:8] = True
    row = shape_to_mask(_Shape(encode_mask(drawn)), _page())

    runs = [int(value) for value in row.rle.split(",")]
    points = [*runs, row.left, row.top, row.left + row.width - 1, row.top + row.height - 1]
    assert decode_mask(points, image_width=W, image_height=H).sum() == (3 * D) ** 2
