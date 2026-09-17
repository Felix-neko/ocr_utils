"""Ориентация внутри шага 1: своя версия, своя запись, свой набор углов.

Главное, что здесь проверяется, — НЕЗАВИСИМОСТЬ двух версий. Слитые в одну, они означали бы
перечитывание полутерабайта после каждой правки порога ориентации.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from ocr_utils.scan_markup.cli import main
from ocr_utils.db.models import SOURCE_AUTO, SOURCE_CVAT
from ocr_utils.db.repo import iter_pages, require_pack
from ocr_utils.db.session import open_db
from ocr_utils.scan_markup.detection import DETECTOR_VERSION
from ocr_utils.scan_markup.orientation import ORIENTATION_VERSION

# Синтетический пак берём готовый: рисовать вторую такую же полосу незачем.
from tests.ocr_utils.scan_markup.test_cli import pack_dir  # noqa: F401


def run(pack_dir: Path, db: Path, *extra: str):
    result = CliRunner().invoke(
        main,
        [
            "detect",
            "--pack-dir",
            str(pack_dir),
            "--db",
            str(db),
            "--no-use-surya-layout",
            "--no-first-page-is-cover",
            "--orientation-detectors",
            "ink_axis",
            *extra,
        ],
    )
    assert result.exit_code == 0, result.output + str(result.exception)
    return result


def pages(db: Path):
    with open_db(db)() as session:
        pack = require_pack(session, "пак-1")
        return {page.source_rel_path: page for _y, _i, page in iter_pages(pack)}


def test_orientation_is_written_by_the_same_run(pack_dir, tmp_path):
    """Отдельный прогон ориентации читал бы пак второй раз — полтерабайта впустую."""
    db = tmp_path / "m.sqlite"
    run(pack_dir, db)
    page = next(iter(pages(db).values()))
    assert page.rotate_cw is not None, "ориентация обязана посчитаться"
    assert page.orientation_version == ORIENTATION_VERSION
    assert page.orientation_source == SOURCE_AUTO
    assert page.orientation_detected_at is not None


def test_orientation_can_be_switched_off(pack_dir, tmp_path):
    db = tmp_path / "m.sqlite"
    run(pack_dir, db, "--no-orientation")
    page = next(iter(pages(db).values()))
    assert page.rotate_cw is None, "не считали — значит NULL, а не 0"
    assert page.orientation_version is None


def test_allowed_angles_are_remembered_by_the_pack(pack_dir, tmp_path):
    """Условие живёт в базе, а не в памяти запускающего — как и корни путей."""
    db = tmp_path / "m.sqlite"
    run(pack_dir, db, "--angles", "0,90")
    with open_db(db)() as session:
        assert require_pack(session, "пак-1").allowed_rotations == "0,90"
    # Повторный прогон без ключа берёт набор оттуда же и не затирает его умолчанием.
    run(pack_dir, db)
    with open_db(db)() as session:
        assert require_pack(session, "пак-1").allowed_rotations == "0,90"


def test_default_angles_exclude_180(pack_dir, tmp_path):
    """На паке-1 все размеченные боковые полосы требуют поворота по часовой; 180 не встретилось."""
    db = tmp_path / "m.sqlite"
    run(pack_dir, db)
    with open_db(db)() as session:
        assert require_pack(session, "пак-1").allowed_rotations == "0,90,270"


def test_stale_orientation_alone_does_not_force_a_full_redetect(pack_dir, tmp_path):
    """Версии независимы: устаревшая ориентация не отменяет свежую растровую разметку."""
    db = tmp_path / "m.sqlite"
    run(pack_dir, db)
    with open_db(db)() as session:
        pack = require_pack(session, "пак-1")
        for _y, _i, page in iter_pages(pack):
            page.orientation_version = ORIENTATION_VERSION - 1
        session.commit()

    run(pack_dir, db, "--skip-detected")
    for page in pages(db).values():
        assert page.orientation_version == ORIENTATION_VERSION, "ориентация обязана пересчитаться"
        assert page.detector_version == DETECTOR_VERSION


def test_manual_orientation_from_cvat_is_not_overwritten(pack_dir, tmp_path):
    """Решение человека дороже автоматического: пересчёт затёр бы ручную правку."""
    db = tmp_path / "m.sqlite"
    run(pack_dir, db)
    with open_db(db)() as session:
        pack = require_pack(session, "пак-1")
        for _y, _i, page in iter_pages(pack):
            page.rotate_cw = 180
            page.orientation_source = SOURCE_CVAT
            page.orientation_version = None
        session.commit()

    run(pack_dir, db, "--skip-detected")
    for page in pages(db).values():
        assert (page.rotate_cw, page.orientation_source) == (180, SOURCE_CVAT)
