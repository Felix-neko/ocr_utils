"""Сбор выгрузки Capture One: выдержка по времени, полнота выпуска, запись в базу."""

import os
import time

import numpy as np
from PIL import Image

from ocr_utils.pdf_utils.collect_sharpened import collect, fill_database
from ocr_utils.scan_markup.db.repo import require_pack
from ocr_utils.scan_markup.db.session import open_db

from .conftest import PAGES

OLD_ENOUGH = time.time() - 3600  # час назад: заведомо дописан


def _age(path, when=OLD_ENOUGH):
    os.utime(path, (when, when))


def _make_export(root, names, aged=True):
    """Кладёт «выгрузку Capture One» в root/1970/01/sharpened."""
    subdir = root / "1970" / "01" / "sharpened"
    subdir.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = subdir / f"{name[:-4]}.jpg"
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(path)
        if aged:
            _age(path)
    return subdir


def test_complete_issue_is_moved_and_the_subdir_disappears(pack, tmp_path):
    """Выпуск переносится целиком, а опустевшая sharpened убирается за собой."""
    _, originals, _sharpened, _name = pack
    subdir = _make_export(originals, PAGES)
    dest = tmp_path / "collected"

    report, moved = collect(originals, dest, min_age_minutes=10)

    assert (report.moved, report.issues_done, report.issues_skipped) == (3, 1, 0)
    assert not subdir.exists()
    assert sorted(p.name for p in (dest / "1970" / "01").iterdir()) == ["0010.jpg", "0020.jpg", "0030.jpg"]
    assert moved["1970/01/0010"] == "0010.jpg"


def test_a_file_still_being_written_holds_the_whole_issue(pack, tmp_path):
    """Свежий файл не трогается вовсе, и выпуск с ним ждёт следующего запуска.

    Наполовину записанный JPEG на диске неотличим от готового: имя есть, размер ненулевой.
    Единственный дешёвый признак — время правки, по нему и держим паузу.
    """
    _, originals, _sharpened, _name = pack
    subdir = _make_export(originals, PAGES)
    _age(subdir / "0020.jpg", time.time())  # только что дописан
    dest = tmp_path / "collected"

    report, moved = collect(originals, dest, min_age_minutes=10)

    assert (report.moved, report.issues_done, report.issues_skipped, report.too_fresh) == (0, 0, 1, 1)
    assert moved == {}
    assert subdir.exists() and len(list(subdir.iterdir())) == 3
    assert not dest.exists() or not list(dest.rglob("*.jpg"))


def test_the_same_issue_is_collected_on_the_next_run(pack, tmp_path):
    """Скрипт идемпотентен: доехавший выпуск забирается повторным запуском."""
    _, originals, _sharpened, _name = pack
    subdir = _make_export(originals, PAGES)
    _age(subdir / "0020.jpg", time.time())
    dest = tmp_path / "collected"

    collect(originals, dest, min_age_minutes=10)
    _age(subdir / "0020.jpg")  # файл дописался и отлежался
    report, _moved = collect(originals, dest, min_age_minutes=10)

    assert (report.moved, report.issues_done) == (3, 1)
    assert not subdir.exists()


def test_incomplete_issue_is_moved_only_when_asked(pack, tmp_path):
    """Половинный выпуск переносится лишь по явному --no-require-complete."""
    _, originals, _sharpened, _name = pack
    _make_export(originals, ["0010.tif", "0020.tif"])  # третьей полосы нет
    dest = tmp_path / "collected"

    report, _ = collect(originals, dest, min_age_minutes=10)
    assert (report.moved, report.issues_skipped, report.missing) == (0, 1, 1)

    report, _ = collect(originals, dest, min_age_minutes=10, require_complete=False)
    assert (report.moved, report.issues_done) == (2, 1)


def test_database_gets_the_root_and_the_file_names(pack, tmp_path):
    db_path, originals, _sharpened, name = pack
    _make_export(originals, PAGES)
    dest = tmp_path / "collected"

    report, moved = collect(originals, dest, min_age_minutes=10)
    # Стираем то, что положила фикстура, — иначе проверялась бы она, а не сбор.
    with open_db(db_path)() as session:
        for page in require_pack(session, name).year_packages[0].issues[0].pages:
            page.sharpened_text_pic_file_name = None
            page.sharpened_text_pic_rel_path = None
        session.commit()

    fill_database(db_path, name, dest, moved, report)

    with open_db(db_path)() as session:
        pack_row = require_pack(session, name)
        assert pack_row.sharpened_text_pics_root == str(dest)
        pages = {p.source_file_name: p for p in pack_row.year_packages[0].issues[0].pages}
        assert pages["0010.tif"].sharpened_text_pic_file_name == "0010.jpg"
        assert pages["0010.tif"].sharpened_text_pic_rel_path == "1970/01/0010.jpg"
    assert report.db_filled == 3


def test_copy_mode_leaves_the_source_in_place(pack, tmp_path):
    _, originals, _sharpened, _name = pack
    subdir = _make_export(originals, PAGES)
    dest = tmp_path / "collected"

    report, _ = collect(originals, dest, min_age_minutes=10, move=False)

    assert report.moved == 3
    assert len(list(subdir.iterdir())) == 3
    assert len(list((dest / "1970" / "01").iterdir())) == 3
