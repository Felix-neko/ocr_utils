"""PageImage: копии по разрешениям, кадр surya, битональная копия, pickle."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from ocr_utils.page_layout import WORK_DPI
from ocr_utils.page_layout.image import SURYA_MAX_SIDE, PageImage, Variant


def _scan(tmp_path: Path, size=(3492, 6051), dpi: int = 600) -> Path:
    array = np.full((size[1], size[0], 3), 240, np.uint8)
    array[1000:1400, 300:2000] = 30
    path = tmp_path / "IMG_0003_2R.tif"
    Image.fromarray(array).save(path, dpi=(dpi, dpi))
    return path


def test_from_file_is_lazy_and_frames_scale(tmp_path: Path) -> None:
    page = PageImage.from_file(_scan(tmp_path), Variant.SCAN, cache_name="1966/01/IMG_0003_2R")
    assert (page.width, page.height, page.dpi) == (3492, 6051, 600)
    assert page._bgr is None, "заголовка достаточно, пиксели не читались"
    gray150 = page.gray_at(WORK_DPI)
    assert gray150.shape == (1513, 873)
    assert page.bgr_at(WORK_DPI).shape == (1513, 873, 3)
    frame = page.surya_frame
    assert frame.shape == (1513, 873, 3) and page.surya_dpi == WORK_DPI
    bitonal = page.bitonal_at(WORK_DPI)
    assert set(np.unique(bitonal)) == {0, 255} and bitonal[300, 300] == 0 and bitonal[10, 10] == 255
    assert page.size_at(300) == (1746, 3026) and page.scale_to_native(150) == 4.0


def test_no_dpi_tag_requires_default(tmp_path: Path) -> None:
    array = np.full((100, 80, 3), 200, np.uint8)
    path = tmp_path / "x.png"
    Image.fromarray(array).save(path)
    try:
        PageImage.from_file(path, Variant.CAMERA)
    except ValueError as error:
        assert "default_dpi" in str(error)
    else:
        raise AssertionError("без тега и умолчания должно падать")
    page = PageImage.from_file(path, Variant.CAMERA, default_dpi=300)
    assert page.dpi == 300


def test_huge_frame_is_capped_for_surya() -> None:
    array = np.full((4000, 8000), 250, np.uint8)  # разворот 300 dpi: 150 dpi было бы 4000 px
    page = PageImage.from_array(array, 300, Variant.CAMERA)
    assert max(page.surya_frame.shape[:2]) <= SURYA_MAX_SIDE and page.surya_dpi < WORK_DPI
    assert not page.has_color and page.bgr_at(page.surya_dpi).ndim == 3


def test_pickle_restores_file_source(tmp_path: Path) -> None:
    page = PageImage.from_file(_scan(tmp_path), Variant.SCAN, cache_name="a")
    clone = pickle.loads(pickle.dumps(page))
    assert clone._loader is not None and clone.gray_at(WORK_DPI).shape == (1513, 873)
    page.surya_frame
    page.drop_pixels()
    clone = pickle.loads(pickle.dumps(page))
    assert clone._surya_frame is not None and clone._bgr is None
    assert clone.surya_frame_digest() == page.surya_frame_digest()


def test_pdf_page_renders_directly_at_requested_dpi(tmp_path: Path) -> None:
    import fitz

    pdf = tmp_path / "doc.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=595, height=842)  # A4 в пунктах
        page.draw_rect(fitz.Rect(100, 100, 300, 200), color=(0, 0, 0), fill=(0, 0, 0))
        doc.save(str(pdf))
    with fitz.open(str(pdf)) as doc:
        image = PageImage.from_pdf_page(doc, 0, Variant.FR_GEO, native_dpi=600)
        assert (image.width, image.height) == (4958, 7017)
        gray150 = image.gray_at(150)
        assert (
            gray150.shape == (1755, 1240) == image.size_at(150)[::-1]
            and gray150[300, 400] < 50
            and gray150[10, 10] > 200
        )
        assert image._gray is None, "рендер в 150 dpi не требует полного кадра"
        clone = pickle.loads(pickle.dumps(image))
        assert clone.gray_at(150).shape == gray150.shape
