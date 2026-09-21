"""Набивка кэша surya: страницы PDF и картинки, пропуск уже записанных, модель-заглушка в родителе."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ocr_utils.page_layout.image import Variant
from ocr_utils.page_layout.prefill import pdf_requests, prefill, scan_requests
from ocr_utils.page_layout.surya.blocks import from_boxes
from ocr_utils.page_layout.surya.cache import SuryaCache


class _Model:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, frames):
        self.calls += len(frames)
        return [from_boxes([], f.shape[1], f.shape[0]) for f in frames]

    @staticmethod
    def name() -> str:
        return "stub"


def test_prefill_pdf_pages_and_scans(tmp_path: Path) -> None:
    import fitz

    pdf = tmp_path / "pdf" / "full_1966_01.pdf"
    pdf.parent.mkdir()
    with fitz.open() as doc:
        for _ in range(3):
            doc.new_page(width=300, height=400)
        doc.save(str(pdf))
    scans = tmp_path / "scans" / "1966" / "01"
    scans.mkdir(parents=True)
    Image.fromarray(np.full((800, 600, 3), 240, np.uint8)).save(scans / "IMG_0001_1L.tif", dpi=(600, 600))

    cache = SuryaCache(tmp_path / "cache")
    model = _Model()
    requests = list(pdf_requests([pdf], Variant.FR_GEO)) + list(scan_requests(tmp_path / "scans", Variant.SCAN))
    assert [r.cache_name for r in requests] == [
        "full_1966_01/p0000",
        "full_1966_01/p0001",
        "full_1966_01/p0002",
        "1966/01/IMG_0001_1L",
    ]
    stats = prefill(requests, cache, model, jobs=2, chunk=2, progress=False)
    assert (stats.requested, stats.cached, stats.done, stats.failed) == (4, 0, 4, 0) and model.calls == 4
    assert cache.has(Variant.FR_GEO, "full_1966_01/p0002") and cache.has(Variant.SCAN, "1966/01/IMG_0001_1L")

    again = prefill(requests, cache, model, jobs=2, chunk=2, progress=False)
    assert (again.cached, again.done) == (4, 0) and model.calls == 4, "второй раз — всё из кэша, модель не звалась"
