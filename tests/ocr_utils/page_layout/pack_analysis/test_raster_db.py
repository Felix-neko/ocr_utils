"""Растр из базы разметки (``pack_analysis.raster_db``) и перенос неизменённых полос из прошлого прогона (``run.stage_reuse``)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ocr_utils.page_layout.pack_analysis.raster_db import DB_SOURCE, load_raster, same_raster
from ocr_utils.page_layout.pack_analysis.run import stage_reuse
from ocr_utils.page_layout.pack_analysis.stages import PageTask


def _make_db(path: Path) -> Path:
    """Крошечная база со схемой, нужной :func:`load_raster`: две полосы, одна повёрнута, растр, печать, таблица."""
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        create table packs (id integer primary key, name text);
        create table year_packages (id integer primary key, pack_id integer, name text);
        create table issues (id integer primary key, year_package_id integer, name text);
        create table pages (id integer primary key, issue_id integer, source_rel_path text, width integer,
                            height integer, rotate_cw integer);
        create table rect_regions (id integer primary key, page_id integer, x1 integer, y1 integer, x2 integer,
                                   y2 integer, kind text);
        insert into packs values (1, 'пак-1'), (2, 'пак-2');
        insert into year_packages values (1, 1, '1973'), (2, 2, '1980');
        insert into issues values (1, 1, '06'), (2, 2, '01');
        insert into pages values (1, 1, '1973/06/IMG_0001_1L.tif', 100, 200, 0),
                                 (2, 1, '1973/06/IMG_0002_2R.tif', 100, 200, 90),
                                 (3, 2, '1980/01/IMG_0001_1L.tif', 100, 200, 0);
        insert into rect_regions values
            (1, 1, 10, 20, 30, 40, 'color'),
            (2, 1, 50, 60, 70, 80, 'stamp_suspect'),
            (3, 1, 0, 0, 100, 100, 'table'),
            (4, 1, 5, 5, 15, 15, 'color_text'),
            (5, 2, 10, 20, 30, 40, 'grayscale'),
            (6, 3, 1, 1, 2, 2, 'color');
        """
    )
    connection.commit()
    connection.close()
    return path


def test_load_raster_takes_pictures_only_and_rotates_boxes(tmp_path):
    """Берутся цветной, серый растр и цветной текст своего пака; печати и таблицы — нет; рамка поворачивается."""
    raster = load_raster(_make_db(tmp_path / "db.sqlite"), "пак-1")
    assert set(raster) == {"1973/06/IMG_0001_1L", "1973/06/IMG_0002_2R"}
    first = raster["1973/06/IMG_0001_1L"]
    assert [(r["kind"], r["box"]) for r in first] == [("color", [10, 20, 30, 40]), ("color_text", [5, 5, 15, 15])]
    assert all(r["info"] == {"source": DB_SOURCE} and r["confidence"] is None for r in first)
    # Поворот на 90° по часовой в кадре 100×200: x' = H − y, y' = x.
    assert raster["1973/06/IMG_0002_2R"] == [
        {"kind": "grayscale", "box": [160, 10, 180, 30], "confidence": None, "info": {"source": DB_SOURCE}}
    ]


def _region(kind: str, box: list[int]) -> dict:
    """Область в формате записи полосы."""
    return {"kind": kind, "box": box, "confidence": 1.0, "info": {}}


def test_same_raster_ignores_order_but_not_any_change():
    """Порядок областей не важен; смена вида, сдвиг на пиксель, лишняя печать — «растр изменился»."""
    old = [_region("color", [1, 2, 3, 4]), _region("grayscale", [5, 6, 7, 8])]
    assert same_raster(old, list(reversed(old)))
    assert not same_raster(old, [_region("color", [1, 2, 3, 4]), _region("color", [5, 6, 7, 8])])
    assert not same_raster(old, [_region("color", [1, 2, 3, 4]), _region("grayscale", [5, 6, 7, 9])])
    assert not same_raster(old + [_region("stamp_suspect", [0, 0, 1, 1])], old)
    assert same_raster([], [])


@pytest.fixture()
def previous(tmp_path):
    """Прошлый прогон: две полосы с кандидатами, вырезки и вывод DeepSeek обоих проходов."""
    root = tmp_path / "v2"
    work = root / "work"
    pages = {
        "1973_06_A": {"raster": [_region("color", [1, 2, 3, 4])], "candidates": [{"id": "1973_06_A_0"}]},
        "1973_06_B": {"raster": [_region("color", [1, 2, 3, 4])], "candidates": [{"id": "1973_06_B_0"}]},
    }
    for key, record in pages.items():
        (work / "pages").mkdir(parents=True, exist_ok=True)
        (work / "pages" / f"{key}.json").write_text(json.dumps({"page": key, **record}))
        for stage in ("pass1", "pass2"):
            (work / "crops" / stage).mkdir(parents=True, exist_ok=True)
            (work / "crops" / stage / f"{key}_0.png").write_bytes(b"png")
    for stage, prompts in (("pass1", ("markdown", "ocr")), ("pass2", ("markdown",))):
        (work / "deepseek" / stage).mkdir(parents=True, exist_ok=True)
        for prompt in prompts:
            lines = [json.dumps({"id": f"{key}_0", "elements": [prompt]}) for key in pages]
            (work / "deepseek" / stage / f"{prompt}.jsonl").write_text("\n".join(lines) + "\n")
    return root


def test_stage_reuse_copies_unchanged_pages_only_once(tmp_path, previous):
    """Полоса с тем же растром переносится с вырезками и выводом DeepSeek; изменённая — нет; повтор без дублей."""
    tasks = [PageTask("1973/06/A", Path("a.jpg")), PageTask("1973/06/B", Path("b.jpg"))]
    # У полосы A растр тот же (вид и рамка; уверенность и info не сравниваются), у B — сдвинут.
    raster = {
        "1973/06/A": [{"kind": "color", "box": [1, 2, 3, 4], "confidence": None, "info": {"source": "db"}}],
        "1973/06/B": [{"kind": "color", "box": [1, 2, 3, 5], "confidence": None, "info": {"source": "db"}}],
    }
    work = tmp_path / "v3" / "work"
    for _ in range(2):
        stage_reuse(tasks, previous, work, raster)
    assert (work / "pages" / "1973_06_A.json").is_file()
    assert not (work / "pages" / "1973_06_B.json").exists()
    assert (work / "crops" / "pass1" / "1973_06_A_0.png").is_file()
    assert (work / "crops" / "pass2" / "1973_06_A_0.png").is_file()
    assert not (work / "crops" / "pass1" / "1973_06_B_0.png").exists()
    for stage, prompt in (("pass1", "markdown"), ("pass1", "ocr"), ("pass2", "markdown")):
        lines = (work / "deepseek" / stage / f"{prompt}.jsonl").read_text().splitlines()
        assert [json.loads(line)["id"] for line in lines] == ["1973_06_A_0"]
    assert (work / "raster_changed.txt").read_text() == "1973/06/B\n"
