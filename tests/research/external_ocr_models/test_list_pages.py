"""Отбор полос прогона: список --pages и вычитание --skip-pages (оглавление — отдельным запросом)."""

from __future__ import annotations

from pathlib import Path

from research.external_ocr_models.cli import list_pages


def test_skip_pages_removes_listed_regardless_of_suffix(tmp_path: Path) -> None:
    in_dir = tmp_path / "1975" / "12"
    in_dir.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        (in_dir / name).write_bytes(b"")
    skip = tmp_path / "toc_pages.txt"
    skip.write_text("b.tif  # contents 1.00 auto\n# комментарий\n\n", encoding="utf-8")

    assert list_pages(in_dir, None, None) == [Path("a.jpg"), Path("b.jpg"), Path("c.jpg")]
    assert list_pages(in_dir, None, None, skip) == [Path("a.jpg"), Path("c.jpg")]
    assert list_pages(in_dir, None, 1, skip) == [Path("a.jpg")]
