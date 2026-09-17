from __future__ import annotations

import numpy as np
import pytest

from ocr_utils.scan_markup.curved_lines import labels
from ocr_utils.scan_markup.curved_lines.cache import PageCache


def test_cache_roundtrip_and_version_key(tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"x")
    cache = PageCache(tmp_path / "cache")
    assert cache.load(image, "det", "v1.1") is None
    cache.store(image, "det", "v1.1", {"metrics": {"a": 1.0}, "raw": {"k": [1, 2]}})
    assert cache.load(image, "det", "v1.1") == {"metrics": {"a": 1.0}, "raw": {"k": [1, 2]}}
    assert cache.has(image, "det", "v1.1")
    assert cache.load(image, "det", "v1.2") is None
    # Правка файла делает запись невидимой.
    image.write_bytes(b"xy")
    assert cache.load(image, "det", "v1.1") is None


def test_labels_load_and_separation(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text(
        "rel_path,label,note\n# комментарий\n1966/01/a.tif,curved,низ\n1970/01/b.tif,straight\n", encoding="utf-8"
    )
    loaded = labels.load_labels(path)
    assert labels.lookup(loaded, "1966/01/a.jpg").label == "curved"
    assert labels.lookup(loaded, "1970/01/b.png").label == "straight"
    assert labels.lookup(loaded, "1970/01/c.png") is None
    rows = [("curved", {"d": {"m": 2.0}}), ("curved", {"d": {"m": 3.0}}), ("straight", {"d": {"m": 1.0}})]
    items = labels.separation(rows, {"d": {"m": 1.5}}, ["d"])
    assert len(items) == 1
    assert items[0].gap == pytest.approx(1.0)
    assert items[0].suggested == pytest.approx(1.5)
    assert items[0].confusion() == (2, 0, 0, 1)
    assert "| d | m | 1.5 |" in labels.separation_table(items)


def test_labels_reject_unknown_label(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("1966/01/a.tif,bent\n", encoding="utf-8")
    with pytest.raises(ValueError):
        labels.load_labels(path)
