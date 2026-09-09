"""Имена симлинков и защита каталога находок."""

from __future__ import annotations

from pathlib import Path

import pytest

from ocr_utils.scan_markup.orientation.analysis import PageResult
from ocr_utils.scan_markup.orientation.detectors import Verdict
from ocr_utils.scan_markup.orientation.pdf_pages import PdfPage
from ocr_utils.scan_markup.orientation.report import LinkDirError, link_name, write_link_dir


def result_of(rel_path: str, path: Path, rotate: int = 90) -> PageResult:
    result = PageResult(rel_path=rel_path, path=path, verdicts={"ink_axis": Verdict(rotate, 0.9)})
    result.combo = Verdict(rotate, 0.9)
    return result


def test_link_name_carries_year_issue_name_page_and_rotation():
    result = result_of("1967/01/IMG_0043_2R.tif", Path("/x/IMG_0043_2R.tif"))
    name = link_name(result, PdfPage("full_1967_01.pdf", 80), 90)
    assert name == "1967_01_IMG_0043_2R_p080_cw90.tif"


def test_link_name_without_a_database_falls_back_to_the_ordinal():
    result = result_of("1968/01/IMG_0031_2R.jpg", Path("/x/IMG_0031_2R.jpg"))
    assert link_name(result, None, 270, ordinal=7) == "1968_01_IMG_0031_2R_p007_ccw90.jpg"


def test_link_name_survives_a_rescan_folder_with_spaces():
    """Пересъёмка выпуска лежит в каталоге вида «05 (2)» — пробел в имя файла попасть не должен."""
    result = result_of("1975/05 (2)/IMG_0004_1L.tif", Path("/x/IMG_0004_1L.tif"))
    assert link_name(result, None, 180, ordinal=3) == "1975_05_(2)_IMG_0004_1L_p003_180.tif"


def test_write_link_dir_replaces_only_its_own_symlinks(tmp_path):
    source = tmp_path / "IMG_0001_1L.tif"
    source.write_bytes(b"")
    results = [result_of("1967/01/IMG_0001_1L.tif", source)]
    root = tmp_path / "links"
    counts = write_link_dir(root, results, ["ink_axis"], {})
    assert counts["ink_axis"] == 1 and counts["combo"] == 1
    assert (root / "ink_axis" / "1967_01_IMG_0001_1L_p001_cw90.tif").is_symlink()
    # Повторный прогон не должен ни падать, ни копить дубли.
    assert write_link_dir(root, results, ["ink_axis"], {})["ink_axis"] == 1


def test_write_link_dir_refuses_to_delete_real_files(tmp_path):
    """Каталог мог оказаться не тем; восстанавливать удалённое было бы неоткуда."""
    root = tmp_path / "links"
    (root / "ink_axis").mkdir(parents=True)
    stranger = root / "ink_axis" / "важный.tif"
    stranger.write_bytes(b"")
    with pytest.raises(LinkDirError):
        write_link_dir(root, [], ["ink_axis"], {})
    assert stranger.exists()


def test_link_root_finds_the_original_under_a_different_extension(tmp_path):
    """Разбираем заострённый JPEG, а смотреть глазами удобнее оригинальный TIFF."""
    analysed = tmp_path / "sharpened" / "1967" / "01"
    analysed.mkdir(parents=True)
    (analysed / "IMG_0043_2R.jpg").write_bytes(b"")
    originals = tmp_path / "pack" / "1967" / "01"
    originals.mkdir(parents=True)
    original = originals / "IMG_0043_2R.tif"
    original.write_bytes(b"")

    results = [result_of("1967/01/IMG_0043_2R.jpg", analysed / "IMG_0043_2R.jpg")]
    root = tmp_path / "links"
    write_link_dir(root, results, ["ink_axis"], {}, link_root=tmp_path / "pack")
    link = root / "combo" / "1967_01_IMG_0043_2R_p001_cw90.tif"
    assert link.is_symlink() and link.resolve() == original.resolve()


def test_contact_sheet_shows_pages_already_turned(tmp_path):
    """Миниатюра рисуется ПОВЁРНУТОЙ: проверка сводится к «читается или нет»."""
    import cv2

    from ocr_utils.scan_markup.orientation.report import SHEET_COLUMNS, SHEET_TILE_PX, contact_sheet
    from tests.ocr_utils.scan_markup.orientation.synthetic import text_page

    page = tmp_path / "IMG_0002_1L.png"
    cv2.imwrite(str(page), text_page())
    results = [result_of("1967/01/IMG_0002_1L.png", page, rotate=270)]

    sheets = contact_sheet(tmp_path / "sheet.png", results, {})
    assert [path.name for path in sheets] == ["sheet_01.png"]
    image = cv2.imread(str(sheets[0]))
    # Полоса книжная, после поворота на 270 становится альбомной — и плитка это показывает.
    assert image.shape[0] < image.shape[1]
    assert image.shape[1] == SHEET_COLUMNS * (SHEET_TILE_PX + 2)


def test_contact_sheet_skips_pages_that_need_no_rotation(tmp_path):
    from ocr_utils.scan_markup.orientation.report import contact_sheet

    results = [result_of("1967/01/IMG_0001_1L.png", tmp_path / "нет.png", rotate=0)]
    assert contact_sheet(tmp_path / "sheet.png", results, {}) == []


def test_candidates_dir_keeps_pages_the_arbiter_rejected(tmp_path):
    """Каталог кандидатов нужен ради ОТВЕРГНУТЫХ: по ним видно, не потерялось ли настоящее."""
    from ocr_utils.scan_markup.orientation.report import CANDIDATES_DIR, COMBO_DIR

    source = tmp_path / "IMG_0022_2R.tif"
    source.write_bytes(b"")
    rejected = PageResult(
        rel_path="1970/04/IMG_0022_2R.tif", path=source, verdicts={"profile": Verdict(90, 1.0, axis_only=True)}
    )
    rejected.candidate = True
    rejected.combo = Verdict(0, 0.6)  # арбитр отменил находку

    root = tmp_path / "links"
    counts = write_link_dir(root, [rejected], ["profile"], {})
    assert counts[CANDIDATES_DIR] == 1, "отвергнутая полоса обязана остаться среди кандидатов"
    assert counts[COMBO_DIR] == 0, "и не попасть в принятые"
    # В имя идёт поворот, из-за которого полоса и попала в кандидаты.
    assert (root / CANDIDATES_DIR / "1970_04_IMG_0022_2R_p001_cw90.tif").is_symlink()


def test_pages_nobody_flagged_stay_out_of_the_candidates_dir(tmp_path):
    from ocr_utils.scan_markup.orientation.report import CANDIDATES_DIR

    quiet = result_of("1967/01/IMG_0001_1L.tif", tmp_path / "нет.tif", rotate=0)
    assert write_link_dir(tmp_path / "links", [quiet], ["ink_axis"], {})[CANDIDATES_DIR] == 0
