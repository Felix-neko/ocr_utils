"""Синтетический пак: база разметки и файлы полос на диске.

Настоящие оригиналы (25-45 МБ TIFF на медленном диске) в тестах не нужны: проверяется
наша половина работы — что читается из базы, что попадает в маски и что пишется на
выход. Полосы поэтому маленькие и рисуются numpy.
"""

import cv2
import numpy as np
import pytest
from cvat_sdk.masks import encode_mask

from ocr_utils.scan_markup.db.models import (
    KIND_COLOR,
    KIND_GRAYSCALE,
    MASK_LIBRARY_STAMP,
    SOURCE_CVAT,
    MaskAnnotation,
    RasterRegion,
)
from ocr_utils.scan_markup.db.repo import upsert_pack
from ocr_utils.scan_markup.db.session import open_db
from ocr_utils.scan_markup.scan_tree import ScannedIssue, ScannedPage, ScannedYear

PACK_NAME = "пак-тест"
W, H = 400, 600
PAPER = 250
INK = 40


def text_page(w: int = W, h: int = H, seed: int = 0) -> np.ndarray:
    """Бумага со «строками текста» — цветной кадр BGR."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), PAPER, np.uint8)
    img = np.clip(img.astype(np.int16) + rng.integers(-2, 3, size=img.shape), 0, 255).astype(np.uint8)
    for y in range(40, h - 40, 40):
        img[y : y + 6, 30 : w - 30] = INK
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def mask_rle(shape, box) -> str:
    """RLE в формате CVAT для прямоугольной маски ``box`` = (x1, y1, x2, y2).

    ``encode_mask`` ждёт bool-массив и ПОЛУИНТЕРВАЛЬНУЮ рамку, а в хвост кладёт её
    же во включительном виде — те четыре числа база хранит отдельными колонками,
    поэтому здесь они отбрасываются.
    """
    x1, y1, x2, y2 = box
    mask = np.zeros(shape, bool)
    mask[y1:y2, x1:x2] = True
    points = encode_mask(mask, [x1, y1, x2, y2])
    return ",".join(str(int(v)) for v in points[:-4])


@pytest.fixture
def pack(tmp_path):
    """Пак из двух полос: обычная текстовая и обложка с печатью.

    Возвращает ``(db_path, pack_dir, имя пака)``.
    """
    pack_dir = tmp_path / "pack"
    names = ["0010.tif", "0020.tif"]
    for name in names:
        path = pack_dir / "1970" / "01" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), text_page())

    tree = [
        ScannedYear(
            name="1970",
            year=1970,
            rel_path="1970",
            issues=[
                ScannedIssue(
                    name="01",
                    number=1,
                    rel_path="1970/01",
                    pages=[
                        ScannedPage(
                            path=pack_dir / "1970/01" / name, file_name=name, rel_path=f"1970/01/{name}", order_index=i
                        )
                        for i, name in enumerate(names)
                    ],
                )
            ],
        )
    ]

    db_path = tmp_path / "markup.sqlite"
    with open_db(db_path)() as session:
        pack_row = upsert_pack(session, PACK_NAME, pack_dir, tree)
        pages = {p.file_name: p for p in pack_row.year_packages[0].issues[0].pages}
        for page in pages.values():
            page.width, page.height, page.dpi, page.divisor = W, H, 600, 8

        # Первая полоса: обычная, с иллюстрацией в углу.
        pages["0010.tif"].raster_regions = [
            RasterRegion(x1=250, y1=50, x2=380, y2=200, kind=KIND_GRAYSCALE, full_page=False, source=SOURCE_CVAT)
        ]
        # Вторая: обложка (картинка во всю полосу) с библиотечной печатью.
        pages["0020.tif"].raster_regions = [
            RasterRegion(x1=0, y1=0, x2=W, y2=H, kind=KIND_COLOR, full_page=True, source=SOURCE_CVAT)
        ]
        pages["0020.tif"].masks = [
            MaskAnnotation(
                kind=MASK_LIBRARY_STAMP,
                left=100,
                top=100,
                width=60,
                height=40,
                rle=mask_rle((H, W), (100, 100, 160, 140)),
                source_divisor=8,
                source=SOURCE_CVAT,
            )
        ]
        session.commit()

    return db_path, pack_dir, PACK_NAME
