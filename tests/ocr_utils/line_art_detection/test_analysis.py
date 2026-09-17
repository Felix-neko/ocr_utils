"""Отбор файлов: из идущей выгрузки нельзя читать то, что прямо сейчас пишут."""

import os
import time

from ocr_utils.line_art_detection.analysis import collect_pdfs


def make(path, minutes_old: float):
    path.write_bytes(b"%PDF-1.5\n")
    stamp = time.time() - minutes_old * 60.0
    os.utime(path, (stamp, stamp))
    return path


def test_свежий_файл_откладывается(tmp_path):
    old = make(tmp_path / "old.pdf", 30)
    new = make(tmp_path / "new.pdf", 1)
    ready, fresh = collect_pdfs(tmp_path, min_age_minutes=10)
    assert ready == [old]
    assert fresh == [new]


def test_нулевой_возраст_берёт_всё(tmp_path):
    make(tmp_path / "a.pdf", 30)
    make(tmp_path / "b.pdf", 0)
    ready, fresh = collect_pdfs(tmp_path, min_age_minutes=0)
    assert len(ready) == 2 and fresh == []


def test_одиночный_файл_берётся_как_есть(tmp_path):
    """Явно названный файл — сознательный выбор человека, возраст ему не судья."""
    path = make(tmp_path / "one.pdf", 0)
    assert collect_pdfs(path) == ([path], [])
