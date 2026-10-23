"""Тесты обхода папок select_best_raws.py: скрытые подпапки не обходятся."""

from pathlib import Path

from select_best_raws import find_raf_folders


def test_find_raf_folders_skips_hidden_subfolders(tmp_path: Path) -> None:
    """Рекурсивный обход берёт обычные подпапки и пропускает скрытые вместе со всем содержимым."""
    for rel in [
        "DSCF0001.RAF",
        "276/DSCF0001.RAF",
        ".defocus_for_debugging/DSCF0001.RAF",
        "276/.trash/inner/DSCF0002.RAF",
    ]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    folders = [folder.relative_to(tmp_path).as_posix() for folder, _ in find_raf_folders(tmp_path, recursive=True)]
    assert folders == [".", "276"]
