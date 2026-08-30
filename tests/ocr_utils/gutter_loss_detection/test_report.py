"""Отчёты детектора."""

import csv
from pathlib import Path

from ocr_utils.gutter_loss_detection.analysis import FileResult, SideResult, sort_worst_first
from ocr_utils.gutter_loss_detection.report import console_table, markdown_report, write_csv, write_link_dir


def _result(name: str, score: float, code: str = "текст") -> FileResult:
    return FileResult(
        path=Path("/пак") / name,
        score=score,
        code=code,
        why="",
        sides=[SideResult("L", score, 0.05, code == "таблица"), SideResult("R", score, 0.02, False)],
        pitch=64.0,
        lines=45,
    )


def test_сортировка_худшие_первыми():
    results = [_result("a.jpg", 0.2), _result("b.jpg", 0.9), _result("c.jpg", 0.5)]
    assert [r.path.name for r in sort_worst_first(results)] == ["b.jpg", "c.jpg", "a.jpg"]


def test_неизмеренные_уходят_в_конец():
    bad = FileResult(path=Path("/пак/x.jpg"), problem="мало строк")
    results = sort_worst_first([bad, _result("a.jpg", 0.2)])
    assert results[-1].path.name == "x.jpg"


def test_таблица_в_консоли_содержит_имя():
    text = console_table([_result("a.jpg", 0.9)], limit=10)
    assert "a.jpg" in text and "0.90" in text


def test_csv_пишется(tmp_path):
    path = tmp_path / "отчёт.csv"
    write_csv(path, [_result("a.jpg", 0.9, "таблица")])
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows[0]["вердикт"] == "таблица"
    assert rows[0]["балл"] == "0.90"


def test_markdown_делит_на_таблицы_и_текст():
    text = markdown_report([_result("a.jpg", 0.9, "таблица"), _result("b.jpg", 0.8)], limit=10, threshold=0.35)
    assert "Пересканировать обязательно" in text
    assert "a.jpg" in text and "b.jpg" in text


def test_симлинки_нумеруются_по_рейтингу(tmp_path):
    src = tmp_path / "кадры"
    src.mkdir()
    files = []
    for name in ("a.jpg", "b.jpg"):
        (src / name).write_bytes(b"x")
        files.append(name)
    results = [FileResult(path=src / n, score=s, code="текст") for n, s in zip(files, (0.9, 0.4))]
    root, made = write_link_dir(tmp_path / "худшие", results)
    assert made == 2
    names = sorted(p.name for p in root.iterdir())
    assert names[0].startswith("01_0.90") and names[1].startswith("02_0.40")
