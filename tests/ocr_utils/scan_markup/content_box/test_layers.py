"""Команда page-layers на синтетическом паке: обложка по паспорту — зона во всю полосу и закрас пальцев; страница без паспорта — отступы; ручная зона из CVAT не перезаписывается."""

import cv2
import numpy as np

from ocr_utils.db.models import SOURCE_CVAT, ContentBox
from ocr_utils.db.repo import require_pack, upsert_pack
from ocr_utils.db.session import open_db
from ocr_utils.scan_cropping.v2.rectify.passport import PagePassport, write_passport
from ocr_utils.scan_markup.content_box.layers import LayersParams, part_of, run_layers
from ocr_utils.scan_markup.scan_tree import ScannedIssue, ScannedPage, ScannedYear

DPI = 254  # 10 px/мм
W, H = 600, 900


def build(tmp_path):
    """Пак 1974/01 из трёх белых страниц, паспорт у обложки, база."""
    pack_dir, meta = tmp_path / "pages", tmp_path / "meta"
    names = ["IMG_0001_R.tif", "IMG_0002_L.tif", "IMG_0003_R.tif"]
    (pack_dir / "1974" / "01").mkdir(parents=True)
    for name in names:
        cv2.imwrite(str(pack_dir / "1974" / "01" / name), np.full((H, W, 3), 235, np.uint8))
    zone = [[10, 10], [110, 10], [110, 110], [10, 110]]
    write_passport(meta / "1974" / "01" / "IMG_0001_R",
                   PagePassport("01/IMG_0001.jpg", "R", ["cover-right"], True, DPI, W, H, 0.5, False, 100.0, [zone]),
                   None)  # fmt: skip
    factory = open_db(tmp_path / "markup.sqlite")
    with factory() as session:
        pages = [ScannedPage(pack_dir / "1974/01" / n, n, f"1974/01/{n}", i) for i, n in enumerate(names)]
        pack = upsert_pack(
            session, "пак-4", pack_dir, [ScannedYear("1974", 1974, "1974", [ScannedIssue("01", 1, "1974/01", pages)])]
        )
        for page in pack.year_packages[0].issues[0].pages:
            page.width, page.height, page.dpi = W, H, DPI
        # Третья полоса уже правлена руками в CVAT.
        pack.year_packages[0].issues[0].pages[2].content_box = ContentBox(x1=5, y1=5, x2=50, y2=50, source=SOURCE_CVAT)
        session.commit()
    return factory, pack_dir, meta


def test_part_of_names():
    assert [part_of(n) for n in ("a/IMG_1_L.tiff", "IMG_1_R.tif", "IMG_1_LR.tif", "IMG_1_P.tif", "IMG_1.tif")] == [
        "L",
        "R",
        "LR",
        "P",
        "LR",
    ]


def test_layers(tmp_path):
    factory, pack_dir, meta = build(tmp_path)
    stats = run_layers(LayersParams(tmp_path / "markup.sqlite", "пак-4", pack_dir, meta, jobs=2), factory)
    assert stats.failed == 0 and stats.covers == 1 and stats.kept_manual == 1 and stats.finger_pages == 1
    with factory() as session:
        cover, plain, manual = require_pack(session, "пак-4").year_packages[0].issues[0].pages
        assert (cover.content_box.x2, cover.content_box.y2, cover.content_box.method) == (W, H, "cover")
        assert len(cover.finger_zones) == 1 and abs(cover.finger_zones[0].area_mm2 - 100.0) < 1.0
        # Пустая левая страница без паспорта: отступы A, у разреза (справа) — 5 мм.
        assert (plain.content_box.x2, plain.content_box.y1) == (W - 50, 40)
        assert (manual.content_box.source, manual.content_box.x2) == (SOURCE_CVAT, 50)
