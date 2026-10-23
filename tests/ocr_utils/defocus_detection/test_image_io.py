"""Тесты сбора файлов детектора расфокуса: скрытые подпапки не обходятся."""

from pathlib import Path

from ocr_utils.defocus_detection.image_io import collect_images, in_hidden_subfolder, unsupported_files


def make_tree(root: Path) -> None:
    """Дерево: кадры в корне и в обычной подпапке, копии и мусор — в скрытых подпапках.

    Args:
        root: Пустая папка, в которой создаётся дерево.

    Returns:
        None.
    """
    for rel in [
        "DSCF0001.RAF",
        "batch/DSCF0002.RAF",
        ".defocus_for_debugging/DSCF0001.RAF",
        "batch/.trash/deep/DSCF0003.RAF",
        ".defocus_for_debugging/notes.txt",
        "batch/readme.txt",
    ]:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def test_collect_images_skips_hidden_subfolders(tmp_path: Path) -> None:
    """При рекурсивном обходе кадры из скрытых подпапок (на любой глубине) не берутся."""
    make_tree(tmp_path)
    found = [p.relative_to(tmp_path).as_posix() for p in collect_images(tmp_path, recursive=True)]
    assert found == ["DSCF0001.RAF", "batch/DSCF0002.RAF"]


def test_unsupported_files_ignores_hidden_subfolders(tmp_path: Path) -> None:
    """Предупреждение о неподдерживаемых файлах не жалуется на содержимое скрытых подпапок."""
    make_tree(tmp_path)
    skipped = [p.relative_to(tmp_path).as_posix() for p in unsupported_files(tmp_path, recursive=True)]
    assert skipped == ["batch/readme.txt"]


def test_hidden_root_itself_is_allowed(tmp_path: Path) -> None:
    """Корень обхода, даже со скрытым именем, указан явно и не отбрасывается."""
    root = tmp_path / ".shoot"
    make_tree(root)
    assert not in_hidden_subfolder(root / "DSCF0001.RAF", root)
    assert [p.name for p in collect_images(root)] == ["DSCF0001.RAF"]
