"""Оснастка валидации: разбор имён оверлеев и таблица ожиданий."""

import pytest

from ocr_utils.scan_markup.db.models import KIND_COLOR, KIND_GRAYSCALE
from ocr_utils.scan_markup.detection.overlay import overlay_name, overlay_to_rel_path
from ocr_utils.scan_markup.detection.page import DetectedRegion
from ocr_utils.scan_markup.validation.cases import BY_FOLDER, DEFECTS, collect_cases
from ocr_utils.scan_markup.validation.checks import expectation_holds

REL_PATH = "1969/12/IMG_0115_2R.tif"


def _region(kind: str = KIND_GRAYSCALE) -> DetectedRegion:
    return DetectedRegion(box=(0, 0, 10, 10), kind=kind, full_page=False)


def test_overlay_name_round_trip() -> None:
    """Имя оверлея — контракт с папками-эталонами, и разбор обязан быть обратным сборке."""
    assert overlay_name(REL_PATH) == "1969__12__IMG_0115_2R.tif.jpg"
    assert overlay_to_rel_path(overlay_name(REL_PATH)) == REL_PATH
    assert overlay_to_rel_path(f"/куда-то/{overlay_name(REL_PATH)}") == REL_PATH


def test_every_defect_has_a_unique_folder() -> None:
    """Две записи с одним именем папки молча потеряли бы половину выборки."""
    assert len(BY_FOLDER) == len(DEFECTS)


@pytest.mark.parametrize(
    "key,regions,expected",
    [
        ("color_on_gray", [_region(KIND_GRAYSCALE)], True),
        ("color_on_gray", [_region(KIND_COLOR)], False),
        ("color_on_gray", [], False),  # потерять картинку — тоже не починка
        ("lineart", [], True),
        ("lineart", [_region()], False),
        ("false_positive", [], True),
        ("merged", [_region(), _region()], True),
        ("merged", [_region()], False),
        ("split", [_region()], True),
        ("split", [_region(), _region()], False),
        ("split", [], False),
    ],
)
def test_expectations(key: str, regions, expected: bool) -> None:
    """Каждое ожидание проверяется на обоих исходах, включая вырожденный «областей нет»."""
    assert expectation_holds(key, regions) is expected


def test_unknown_folder_is_reported_not_swallowed(tmp_path) -> None:
    """Опечатка в имени папки не должна выглядеть как «дефектов не осталось»."""
    (tmp_path / "cases" / "опечатка").mkdir(parents=True)
    (tmp_path / "cases" / "опечатка" / "a.jpg").write_bytes(b"")
    _cases, notes = collect_cases(tmp_path / "cases", tmp_path / "pack")
    assert notes and "опечатка" in notes[0]


def test_missing_original_is_reported(tmp_path) -> None:
    """Файл из выборки, которого нет в паке, попадает в замечания, а не в зачёт."""
    folder = tmp_path / "cases" / DEFECTS[0].folder
    folder.mkdir(parents=True)
    (folder / overlay_name(REL_PATH)).write_bytes(b"")
    cases, notes = collect_cases(tmp_path / "cases", tmp_path / "pack")
    assert not cases
    assert notes and REL_PATH in notes[0]
