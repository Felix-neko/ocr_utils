"""Отпечаток файла: хеш, дешёвая проверка по ``stat`` и признак расхождения с CVAT."""

import hashlib
import subprocess
from pathlib import Path

import pytest

from ocr_utils.scan_markup.db.models import Page
from ocr_utils.scan_markup.hashing import (
    HASH_ALGO,
    apply_stamp,
    file_digest,
    full_stamp,
    is_stale_in_cvat,
    stat_matches,
    stat_stamp,
)


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "IMG_0004_1L.tif"
    path.write_bytes(b"\x49\x49\x2a\x00" + b"scan" * 5000)
    return path


def test_digest_matches_sha256sum(sample):
    """Хеш совпадает с системной утилитой: строку из базы можно проверить руками."""
    expected = hashlib.sha256(sample.read_bytes()).hexdigest()
    assert file_digest(sample) == expected

    tool = subprocess.run(["sha256sum", str(sample)], capture_output=True, text=True)
    if tool.returncode == 0:  # на машине без coreutils проверка просто пропускается
        assert tool.stdout.split()[0] == expected


def test_full_stamp_carries_size_and_digest(sample):
    stamp = full_stamp(sample)
    assert stamp.size == sample.stat().st_size
    assert stamp.digest == file_digest(sample)


def test_stat_stamp_does_not_hash(sample):
    assert stat_stamp(sample).digest is None


def test_apply_stamp_without_digest_keeps_old_hash(sample):
    """Обновление одного лишь ``stat`` не должно затирать посчитанный раньше хеш."""
    page = Page(file_name="a.tif", rel_path="1974/01/a.tif", order_index=0)
    apply_stamp(page, full_stamp(sample))
    original = page.file_hash

    sample.touch()
    apply_stamp(page, stat_stamp(sample))
    assert page.file_hash == original
    assert page.hash_algo == HASH_ALGO


def test_stat_matches_only_on_full_coincidence(sample):
    page = Page(file_name="a.tif", rel_path="1974/01/a.tif", order_index=0)
    apply_stamp(page, full_stamp(sample))
    assert stat_matches(page, stat_stamp(sample))

    sample.write_bytes(sample.read_bytes() + b"more")
    assert not stat_matches(page, stat_stamp(sample))


def test_page_without_stamp_never_matches():
    """Полоса из базы, заведённой до появления хешей, должна перечитываться."""
    page = Page(file_name="a.tif", rel_path="1974/01/a.tif", order_index=0)
    assert not stat_matches(page, stat_stamp(Path(__file__)))


@pytest.mark.parametrize(
    "file_hash, cvat_hash, stale",
    [
        ("aa", "aa", False),  # залито то же, что на диске
        ("bb", "aa", True),  # файл подменили после заливки
        ("aa", None, False),  # ещё не заливали — это не расхождение, а новая полоса
        (None, "aa", False),  # не считали хеш — судить не о чем
    ],
)
def test_is_stale_in_cvat(file_hash, cvat_hash, stale):
    page = Page(file_name="a.tif", rel_path="1974/01/a.tif", order_index=0)
    page.file_hash, page.cvat_file_hash = file_hash, cvat_hash
    assert is_stale_in_cvat(page) is stale
