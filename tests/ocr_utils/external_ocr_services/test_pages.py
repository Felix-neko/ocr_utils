"""Обход входа, флаги из базы (только чтение) и из списков, сопоставление .tif ↔ .jpg."""

from pathlib import Path

import pytest
from PIL import Image

from ocr_utils.external_ocr_services.pages import (
    UNKNOWN,
    PageFlags,
    flags_for,
    flags_from_db,
    flags_from_lists,
    group_by_issue,
    list_pages,
    read_hints,
)
from ocr_utils.scan_markup.db.repo import require_pack, upsert_pack
from ocr_utils.scan_markup.db.session import open_db
from ocr_utils.scan_markup.scan_tree import ScannedIssue, ScannedPage, ScannedYear


def _make_pages(root: Path, rels=("1966/03/IMG_0104_2R.jpg", "1966/03/IMG_0105_1L.jpg", "1967/10/IMG_0041.jpg")):
    for rel in rels:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (40, 60), 230).save(path)


def test_list_pages_filters_and_groups(tmp_path):
    _make_pages(tmp_path)
    (tmp_path / "1966/_служебная").mkdir()
    Image.new("L", (4, 4)).save(tmp_path / "1966/_служебная/x.jpg")
    rels = list_pages(tmp_path)
    assert [rel.as_posix() for rel in rels] == [
        "1966/03/IMG_0104_2R.jpg",
        "1966/03/IMG_0105_1L.jpg",
        "1967/10/IMG_0041.jpg",
    ]
    assert [rel.as_posix() for rel in list_pages(tmp_path, only_year="1967")] == ["1967/10/IMG_0041.jpg"]
    assert list_pages(tmp_path, only_issue="03", limit=1) == [Path("1966/03/IMG_0104_2R.jpg")]
    assert list(group_by_issue(rels)) == ["1966/03", "1967/10"]

    listed = tmp_path / "pages.txt"
    listed.write_text("# комментарий\n1967/10/IMG_0041.jpg\n", encoding="utf-8")
    assert list_pages(tmp_path, listed) == [Path("1967/10/IMG_0041.jpg")]
    listed.write_text("1967/10/нет.jpg\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        list_pages(tmp_path, listed)


def _db_with_flags(tmp_path):
    db = tmp_path / "pack.sqlite"
    factory = open_db(db)
    with factory() as session:
        pages = [
            ScannedPage(tmp_path / "a.tif", "IMG_0104_2R.tif", "1966/03/IMG_0104_2R.tif", 0),
            ScannedPage(tmp_path / "b.tif", "IMG_0105_1L.tif", "1966/03/IMG_0105_1L.tif", 1),
            ScannedPage(tmp_path / "c.tif", "IMG_0106_2R.tif", "1966/03/IMG_0106_2R.tif", 2),
        ]
        years = [ScannedYear("1966", 1966, "1966", [ScannedIssue("03", 3, "1966/03", pages)])]
        upsert_pack(session, "пак-1", tmp_path, years)
        pack = require_pack(session, "пак-1")
        rows = pack.year_packages[0].issues[0].pages
        rows[0].is_toc = True
        rows[1].is_year_index, rows[1].force_is_not_toc = True, True
        session.commit()
    return db


def test_flags_from_db_match_jpg_input_and_veto_wins(tmp_path):
    db = _db_with_flags(tmp_path)
    table = flags_from_db(db, "пак-1")
    assert flags_for(Path("1966/03/IMG_0104_2R.jpg"), table).toc_kind == "contents"
    vetoed = flags_for(Path("1966/03/IMG_0105_1L.jpg"), table)
    assert vetoed.is_year_index and vetoed.force_is_not_toc and vetoed.toc_kind is None
    assert flags_for(Path("1966/03/IMG_0106_2R.jpg"), table) == PageFlags(False, False, False)
    assert flags_for(Path("1966/03/нет.jpg"), table) is UNKNOWN and flags_for(Path("x.jpg"), None) is UNKNOWN
    with pytest.raises(LookupError):
        flags_from_db(db, "другой")


def test_flags_from_lists(tmp_path):
    issue = tmp_path / "1975/12"
    issue.mkdir(parents=True)
    (issue / "toc_pages.txt").write_text(
        "IMG_0002.jpg  # contents 1.00 cvat\nIMG_0090.jpg  # index 0.80 auto\n\n", encoding="utf-8"
    )
    table = flags_from_lists(tmp_path)
    assert flags_for(Path("1975/12/IMG_0002.jpg"), table).toc_kind == "contents"
    assert flags_for(Path("1975/12/IMG_0090.jpg"), table).toc_kind == "index"
    assert not flags_for(Path("1975/12/IMG_0003.jpg"), table).known


def test_read_hints(tmp_path):
    path = tmp_path / "hints.txt"
    path.write_text("# комментарий\nа/b.jpg\tсрезан правый край\nпусто\t\n", encoding="utf-8")
    assert read_hints(path) == {"а/b.jpg": "срезан правый край"}
