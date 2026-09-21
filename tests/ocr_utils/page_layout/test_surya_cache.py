"""Кэш surya: попадание по отпечатку и по дайджесту, промах на чужой картинке, битый файл, импорт старого pickle."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ocr_utils.page_layout.geometry import Box
from ocr_utils.page_layout.image import PageImage, Variant
from ocr_utils.page_layout.surya import OnMiss, SuryaCache, SuryaMissing, SuryaSource
from ocr_utils.page_layout.surya.blocks import LayoutBlocks, from_boxes
from ocr_utils.page_layout.surya.cache import CacheEntry, import_legacy

NAME = "1966/01/IMG_0017_2R"


def _blocks(width: int = 873, height: int = 1512) -> LayoutBlocks:
    return from_boxes([("Table", 0.9, Box(10, 20, 300, 400)), ("Picture", 0.8, Box(0, 500, 200, 700))], width, height)


def _page(tmp_path: Path, seed: int = 0, name: str = NAME) -> PageImage:
    """Скан 600 dpi 3492x6051 → кадр surya 873x1513; содержимое зависит от seed."""
    rng = np.random.default_rng(seed)
    array = np.full((6051, 3492, 3), 240, np.uint8)
    array[500:900, 300:2000] = rng.integers(0, 60, (400, 1700, 3), dtype=np.uint8)
    path = tmp_path / "scan" / f"{seed}.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, dpi=(600, 600))
    return PageImage.from_file(path, Variant.SCAN, cache_name=name)


class _Model:
    """Модель-заглушка: считает вызовы, отдаёт заранее известные блоки."""

    calls = 0

    def predict_one(self, frame):
        _Model.calls += 1
        return _blocks(frame.shape[1], frame.shape[0])

    @staticmethod
    def name() -> str:
        return "stub"


def test_round_trip_and_hit_by_source_stat(tmp_path: Path) -> None:
    cache = SuryaCache(tmp_path / "cache")
    page = _page(tmp_path)
    assert cache.load(page) is None
    saved = cache.save(page, _blocks(), "stub")
    assert saved == tmp_path / "cache" / "scan" / f"{NAME}.json"
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["variant"] == "scan" and payload["source"]["size"] > 0 and payload["digest"]

    again = PageImage.from_file(page.source.path, Variant.SCAN, cache_name=NAME)
    got = cache.load(again)
    assert got is not None and len(got.blocks) == 2
    assert again._surya_frame is None, "попадание по отпечатку файла — пиксели не читались"


def test_rewritten_file_misses_by_stat_but_hits_by_digest(tmp_path: Path) -> None:
    cache = SuryaCache(tmp_path / "cache")
    page = _page(tmp_path)
    cache.save(page, _blocks())
    # Тот же файл переписан тем же содержимым: stat разошёлся, дайджест кадра совпал.
    import os, time

    path = Path(page.source.path)
    time.sleep(0.01)
    os.utime(path, None)
    data = path.read_bytes()
    path.write_bytes(data + b"\x00")  # размер другой, картинка та же
    again = PageImage.from_file(path, Variant.SCAN, cache_name=NAME)
    assert cache.load(again) is not None
    # А другая картинка под тем же именем — промах.
    other = _page(tmp_path, seed=1)
    assert cache.load(other) is None


def test_broken_and_foreign_entries_are_misses(tmp_path: Path) -> None:
    cache = SuryaCache(tmp_path / "cache")
    page = _page(tmp_path)
    path = cache.save(page, _blocks())
    path.write_text("{ не json", encoding="utf-8")
    assert cache.load(page) is None
    cache.save(page, _blocks())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cache_name"] = "1966/01/другая"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert cache.load(page) is None


def test_variant_is_part_of_the_key(tmp_path: Path) -> None:
    cache = SuryaCache(tmp_path / "cache")
    page = _page(tmp_path)
    cache.save(page, _blocks())
    sharpened = PageImage.from_file(page.source.path, Variant.SHARPENED, cache_name=NAME)
    assert cache.load(sharpened) is None


def test_source_model_saves_and_readonly_fails_or_skips(tmp_path: Path) -> None:
    cache = SuryaCache(tmp_path / "cache")
    page = _page(tmp_path)
    _Model.calls = 0
    source = SuryaSource(cache, _Model())
    assert source.resolve(page) is not None and _Model.calls == 1
    assert source.resolve(_page(tmp_path)) is not None and _Model.calls == 1, "второй раз — из кэша"

    worker = SuryaSource(SuryaCache(tmp_path / "cache", readonly=True))
    assert worker.resolve(_page(tmp_path)) is not None
    with pytest.raises(SuryaMissing):
        worker.resolve(_page(tmp_path, seed=2, name="1966/01/none"))
    lenient = SuryaSource(SuryaCache(tmp_path / "cache", readonly=True), on_miss=OnMiss.SKIP)
    assert lenient.resolve(_page(tmp_path, seed=2, name="1966/01/none")) is None
    assert not SuryaSource(None).enabled


def test_pdf_page_and_array_pages_use_digest(tmp_path: Path) -> None:
    import fitz

    pdf = tmp_path / "full_1967_01.pdf"
    with fitz.open() as doc:
        page = doc.new_page(width=420, height=595)
        page.insert_text((72, 72), "текст", fontsize=20)
        doc.save(str(pdf))
    with fitz.open(str(pdf)) as doc:
        image = PageImage.from_pdf_page(doc, 0, Variant.FR_NOGEO, native_dpi=600)
        assert image.cache_name == "full_1967_01/p0000" and image.source.page_index == 0
        cache = SuryaCache(tmp_path / "cache")
        cache.save(image, _blocks(*image.surya_frame.shape[1::-1]))
        assert cache.load(PageImage.from_pdf_page(doc, 0, Variant.FR_NOGEO, native_dpi=600)) is not None
        assert cache.path(Variant.FR_NOGEO, image.cache_name).is_file()

    array = np.full((3000, 2000), 250, np.uint8)
    array[100:400, 100:900] = 20
    page = PageImage.from_array(array, 600, Variant.BLURRED, cache_name="1966/01/x")
    cache.save(page, _blocks(*page.surya_frame.shape[1::-1]))
    assert cache.load(PageImage.from_array(array.copy(), 600, Variant.BLURRED, cache_name="1966/01/x")) is not None
    array[100:400, 100:900] = 200
    assert cache.load(PageImage.from_array(array, 600, Variant.BLURRED, cache_name="1966/01/x")) is None


class _ForeignThing:
    def __init__(self) -> None:
        self.payload = [1, 2, 3]


def test_import_legacy_pickles(tmp_path: Path, monkeypatch) -> None:
    """Старый pickle с сырым LayoutResult читается без surya; запись — legacy, принимается на веру."""
    import surya.layout.schema as schema

    monkeypatch.setattr(_ForeignThing, "__module__", schema.__name__)
    monkeypatch.setattr(schema, "_ForeignThing", _ForeignThing, raising=False)
    src = tmp_path / "old" / "1966" / "01"
    src.mkdir(parents=True)
    layout = {
        "width": 873,
        "height": 1512,
        "blocks": [{"label": "Table", "confidence": 0.9, "box": [10, 20, 300, 400]}],
    }
    with (src / "IMG_0017_2R.pkl").open("wb") as handle:
        pickle.dump(
            {"scan_rel_path": "1966/01/IMG_0017_2R.tif", "dpi": 150, "layout": layout, "surya": _ForeignThing()}, handle
        )
    (src / "битый.pkl").write_bytes(b"not a pickle")

    cache = SuryaCache(tmp_path / "cache")
    done, skipped, broken = import_legacy(tmp_path / "old", Variant.SCAN, cache)
    assert (done, skipped, broken) == (1, 0, 1)
    entry = cache.read(Variant.SCAN, NAME)
    assert entry is not None and entry.legacy and entry.frame_dpi == 150 and entry.blocks.blocks[0].label == "Table"
    assert cache.load(_page(tmp_path)) is not None, "legacy-запись подходит любой картинке под этим именем"
    assert import_legacy(tmp_path / "old", Variant.SCAN, cache) == (0, 1, 1)


def test_entry_json_round_trip() -> None:
    entry = CacheEntry(Variant.FR_GEO, "a/p0001", _blocks(), 150, None, "abc", False, "stub")
    assert CacheEntry.from_json(entry.to_json()) == entry
