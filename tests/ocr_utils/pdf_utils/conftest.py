"""Синтетический пак под сборку PDF: база разметки, оригиналы и заострённые копии.

Настоящие полосы (25-45 МБ TIFF при 600 dpi) в тестах не нужны: проверяется наша половина
работы — что попадает на страницу, в каком месте и с какими номерами. Кадры поэтому
маленькие, а «заострённая» копия отличается от оригинала только цветом, чтобы по одному
пикселю было видно, откуда он взялся.
"""

import numpy as np
import pytest
from PIL import Image

from ocr_utils.scan_markup.db.models import KIND_COLOR, KIND_GRAYSCALE, SOURCE_CVAT, RectRegion
from ocr_utils.scan_markup.db.repo import upsert_pack
from ocr_utils.scan_markup.db.session import open_db
from ocr_utils.scan_markup.scan_tree import ScannedIssue, ScannedPage, ScannedYear

PACK_NAME = "пак-тест"
W, H = 400, 600
DPI = 600

# Цвета, по которым в тесте узнаётся источник пикселя.
ORIGINAL_COLOR = (200, 30, 30)  # оригинал: красный
SHARPENED_COLOR = (30, 30, 200)  # заострённая копия: синий

# Полосы выпуска: имя -> размеченные иллюстрации.
PAGES = {
    "0010.tif": (),
    "0020.tif": ((100, 200, 300, 400, KIND_GRAYSCALE),),  # врезка внутри полосы
    "0030.tif": ((0, 0, W, H, KIND_COLOR),),  # иллюстрация во весь кадр
}


def _write(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((H, W, 3), np.uint8)
    array[:] = color
    Image.fromarray(array).save(path, dpi=(DPI, DPI))


@pytest.fixture
def pack(tmp_path):
    """Пак из трёх полос. Возвращает ``(db_path, originals_dir, sharpened_dir, имя пака)``."""
    originals = tmp_path / "blurred"
    sharpened = tmp_path / "sharpened"
    for name in PAGES:
        _write(originals / "1970" / "01" / name, ORIGINAL_COLOR)
        _write(sharpened / "1970" / "01" / f"{name[:-4]}.jpg", SHARPENED_COLOR)

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
                            path=originals / "1970/01" / name,
                            file_name=name,
                            rel_path=f"1970/01/{name}",
                            order_index=index,
                        )
                        for index, name in enumerate(PAGES)
                    ],
                )
            ],
        )
    ]

    db_path = tmp_path / "markup.sqlite"
    with open_db(db_path)() as session:
        pack_row = upsert_pack(session, PACK_NAME, originals, tree)
        rows = {p.source_file_name: p for p in pack_row.year_packages[0].issues[0].pages}
        for name, page in rows.items():
            page.width, page.height, page.dpi, page.divisor = W, H, DPI, 8
            page.sharpened_text_pic_file_name = f"{name[:-4]}.jpg"
            page.sharpened_text_pic_rel_path = f"1970/01/{name[:-4]}.jpg"
            page.rect_regions = [
                RectRegion(x1=x1, y1=y1, x2=x2, y2=y2, kind=kind, full_page=False, source=SOURCE_CVAT)
                for x1, y1, x2, y2, kind in PAGES[name]
            ]
        session.commit()

    return db_path, originals, sharpened, PACK_NAME
