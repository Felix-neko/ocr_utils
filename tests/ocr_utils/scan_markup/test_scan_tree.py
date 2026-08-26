"""Обход пака: что считается полосой, а что мусором."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ocr_utils.scan_markup.scan_tree import count_pages, issue_images, scan_pack


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(path)


@pytest.fixture
def pack(tmp_path: Path) -> Path:
    """Мини-пак со всем мусором, который встречается в реальном паке-1."""
    root = tmp_path / "пак-1"
    for name in ("IMG_0003.tif", "IMG_0004.tif"):
        _make_image(root / "1974" / "01" / name)
    _make_image(root / "1974" / "02" / "IMG_0100.tif")
    # Мусор ScanTailor и Windows.
    _make_image(root / "1974" / "01" / "cache" / "thumbs" / "IMG_0003_q500.png")
    _make_image(root / "1974" / "01" / ".cache" / "preview.png")
    (root / "1974" / "01" / "74_01.ScanTailor").write_text("<project/>")
    (root / "1974" / "01" / "Thumbs.db").write_bytes(b"\x00")
    # Перескан того же выпуска — отдельный выпуск.
    _make_image(root / "1975" / "05" / "IMG_0200.tif")
    _make_image(root / "1975" / "05 (2)" / "IMG_0201.tif")
    # Папка без года в имени и пустой выпуск — не попадают в дерево.
    _make_image(root / "черновики" / "01" / "IMG_0300.tif")
    (root / "1975" / "06").mkdir(parents=True)
    return root


def test_cache_dirs_ignored(pack: Path) -> None:
    """Миниатюры ScanTailor из cache/ и .cache/ полосами не считаются."""
    names = [page.file_name for page in scan_pack(pack)[0].issues[0].pages]
    assert names == ["IMG_0003.tif", "IMG_0004.tif"]


def test_cache_dirs_ignored_even_recursively(pack: Path) -> None:
    """Правило держится и при рекурсивном обходе, а не только за счёт его отсутствия.

    Иначе достаточно однажды включить рекурсию — и 252 миниатюры пака-1 молча уедут в базу
    как полосы, причём с виду правдоподобные.
    """
    found = issue_images(pack / "1974" / "01", recursive=True)
    assert [path.name for path in found] == ["IMG_0003.tif", "IMG_0004.tif"]


def test_non_image_files_ignored(pack: Path) -> None:
    """Проект ScanTailor и Thumbs.db отбрасываются по расширению."""
    names = {page.file_name for year in scan_pack(pack) for issue in year.issues for page in issue.pages}
    assert "74_01.ScanTailor" not in names
    assert "Thumbs.db" not in names


def test_rescan_issue_is_separate(pack: Path) -> None:
    """«05 (2)» — отдельный выпуск, номер тот же, имя разное."""
    year = next(year for year in scan_pack(pack) if year.name == "1975")
    assert [issue.name for issue in year.issues] == ["05", "05 (2)"]
    assert [issue.number for issue in year.issues] == [5, 5]


def test_non_year_dirs_and_empty_issues_skipped(pack: Path) -> None:
    """Папка без года в имени и выпуск без картинок в дерево не попадают."""
    years = scan_pack(pack)
    assert [year.name for year in years] == ["1974", "1975"]
    assert "06" not in {issue.name for issue in years[1].issues}


def test_rel_paths_and_order(pack: Path) -> None:
    """Пути хранятся относительно корня пака, порядок полос — по имени файла."""
    pages = scan_pack(pack)[0].issues[0].pages
    assert [page.rel_path for page in pages] == ["1974/01/IMG_0003.tif", "1974/01/IMG_0004.tif"]
    assert [page.order_index for page in pages] == [0, 1]
    assert count_pages(scan_pack(pack)) == 5
