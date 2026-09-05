"""Чтение разметки из базы в плоские структуры (``scan_cleanup.source``)."""

import pickle

import numpy as np

from ocr_utils.scan_cleanup.source import decode_mask_rows, load_markup
from ocr_utils.scan_markup.db.models import KIND_COLOR, KIND_GRAYSCALE, MASK_LIBRARY_STAMP


def test_loads_pages_with_their_markup(pack):
    db_path, _pack_dir, pack_name = pack
    pages = load_markup(db_path, pack_name)

    assert [p.rel_path for p in pages] == ["1970/01/0010.tif", "1970/01/0020.tif"]
    text, cover = pages
    assert (text.width, text.height, text.divisor) == (400, 600, 8)
    assert [r.kind for r in text.regions] == [KIND_GRAYSCALE]
    assert not text.masks and not text.needs_inpaint
    assert [r.kind for r in cover.regions] == [KIND_COLOR]
    assert cover.regions[0].full_page
    assert [m.kind for m in cover.masks] == [MASK_LIBRARY_STAMP]
    assert cover.needs_inpaint


def test_markup_survives_pickle(pack):
    """На этом держится пул: воркер получает разметку через pickle."""
    db_path, _pack_dir, pack_name = pack
    pages = load_markup(db_path, pack_name)
    assert pickle.loads(pickle.dumps(pages)) == pages


def test_only_rel_filters_pages(pack):
    db_path, _pack_dir, pack_name = pack
    pages = load_markup(db_path, pack_name, only_rel={"1970/01/0020.tif"})
    assert [p.rel_path for p in pages] == ["1970/01/0020.tif"]


def test_only_year_and_limit(pack):
    db_path, _pack_dir, pack_name = pack
    assert len(load_markup(db_path, pack_name, only_year="1970")) == 2
    assert load_markup(db_path, pack_name, only_year="1999") == []
    assert len(load_markup(db_path, pack_name, limit=1)) == 1


def test_decode_mask_rows_returns_full_frame_mask(pack):
    db_path, _pack_dir, pack_name = pack
    cover = load_markup(db_path, pack_name)[1]

    mask = decode_mask_rows(cover.masks, cover.width, cover.height)

    assert mask.shape == (cover.height, cover.width)
    assert mask.dtype == np.bool_
    assert mask[100:140, 100:160].all()
    assert int(mask.sum()) == 60 * 40


def test_source_path_joins_pack_root(pack):
    db_path, pack_dir, pack_name = pack
    page = load_markup(db_path, pack_name)[0]
    assert page.source_path(pack_dir).exists()
