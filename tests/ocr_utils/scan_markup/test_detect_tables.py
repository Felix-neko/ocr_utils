"""Таблицы на этапе detect: своя версия, своя замена в базе, растр и ручная разметка целы."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from click.testing import CliRunner
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ocr_utils.scan_markup.cli import main
from ocr_utils.db.models import (
    KIND_GRAYSCALE,
    KIND_TABLE,
    SOURCE_AUTO,
    SOURCE_CVAT,
    TABLE_KINDS,
    Page,
    RectRegion,
)
from ocr_utils.db.session import open_db
from ocr_utils.scan_markup.detection import DETECTOR_VERSION
from ocr_utils.scan_markup.table_detection import TABLE_DETECTOR_VERSION
from tests.ocr_utils.scan_markup import synthetic

DPI = 600
SIZE = (2400, 1800)  # полоса 600 dpi: копия 1/4 — 600x450, на ней и ищутся линейки
# Таблица 4x3 в пикселях оригинала: 5 горизонталей и 4 вертикали, ячейки с «буквами».
TABLE = (300, 400, 1500, 1400)
INK = 25


def _table_page(path: Path) -> None:
    gray = synthetic.paper(SIZE)
    gray[1700:2300] = synthetic.text_page((600, SIZE[1]), line_step=90, glyph_w=24, glyph_h=48, char_step=44, margin=80)
    x0, y0, x1, y1 = TABLE
    xs = np.linspace(x0, x1, 4).astype(int)
    ys = np.linspace(y0, y1, 5).astype(int)
    for y in ys:
        gray[y : y + 8, x0 : x1 + 8] = INK
    for x in xs:
        gray[y0 : y1 + 8, x : x + 8] = INK
    # «Буквы» в каждой ячейке — иначе проверка сочтёт сетку графиком.
    for top, bottom in zip(ys[:-1], ys[1:]):
        for left, right in zip(xs[:-1], xs[1:]):
            for cx in range(left + 60, right - 60, 44):
                gray[top + 90 : top + 138, cx : cx + 24] = INK
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.stack([gray] * 3, -1)).save(path, dpi=(DPI, DPI))


def _run(pack_dir: Path, db: Path, *extra: str):
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
            "--no-orientation",
            *extra,
        ],
    )
    assert result.exit_code == 0, result.output + str(result.exception)
    return result


def _page(db: Path) -> Page:
    with open_db(db)() as session:
        return session.scalars(select(Page).options(selectinload(Page.rect_regions))).one()


def test_detect_writes_table_regions_with_their_own_version(tmp_path: Path) -> None:
    pack = tmp_path / "пак-1"
    _table_page(pack / "1974" / "01" / "a.tif")
    db = tmp_path / "m.sqlite"
    result = _run(pack, db)
    assert "Таблиц: 1" in result.output

    page = _page(db)
    tables = [r for r in page.rect_regions if r.kind in TABLE_KINDS]
    assert [r.kind for r in tables] == [KIND_TABLE]
    region = tables[0]
    assert region.source == SOURCE_AUTO and region.detector_info and '"kind": "таблица"' in region.detector_info
    x0, y0, x1, y1 = TABLE
    assert abs(region.x1 - x0) <= 40 and abs(region.y1 - y0) <= 40
    assert abs(region.x2 - x1 - 8) <= 40 and abs(region.y2 - y1 - 8) <= 40
    assert page.table_detector_version == TABLE_DETECTOR_VERSION and page.tables_detected_at is not None
    assert page.detector_version == DETECTOR_VERSION


def test_stale_tables_alone_recompute_only_tables(tmp_path: Path) -> None:
    """Версии независимы: устаревшие таблицы не заставляют пересчитывать растр — и наоборот."""
    pack = tmp_path / "пак-1"
    _table_page(pack / "1974" / "01" / "a.tif")
    db = tmp_path / "m.sqlite"
    _run(pack, db)
    with open_db(db)() as session:
        page = session.scalars(select(Page)).one()
        page.table_detector_version = TABLE_DETECTOR_VERSION - 1
        page.detected_at = None  # растр как будто не считали — но растр в этом прогоне выключен
        session.commit()

    result = _run(pack, db, "--skip-detected", "--no-raster")
    page = _page(db)
    assert "пропущено: 0" in result.output
    assert page.table_detector_version == TABLE_DETECTOR_VERSION
    assert page.detected_at is None, "растр выключен — его колонки не трогаются"

    result = _run(pack, db, "--skip-detected", "--no-raster")
    assert "пропущено: 1" in result.output, "таблицы свежие, растр не просили — полоса пропускается"


def test_tables_do_not_touch_manual_raster_markup(tmp_path: Path) -> None:
    """Таблицы дописываются в базу с уточнённым человеком растром, и растр остаётся его."""
    pack = tmp_path / "пак-1"
    _table_page(pack / "1974" / "01" / "a.tif")
    db = tmp_path / "m.sqlite"
    _run(pack, db, "--no-tables")
    with open_db(db)() as session:
        page = session.scalars(select(Page)).one()
        assert page.tables_detected_at is None
        page.rect_regions = [RectRegion(x1=1, y1=2, x2=300, y2=400, kind=KIND_GRAYSCALE, source=SOURCE_CVAT)]
        session.commit()

    _run(pack, db, "--skip-detected", "--no-raster")
    page = _page(db)
    kinds = sorted((r.kind, r.source) for r in page.rect_regions)
    assert kinds == [(KIND_GRAYSCALE, SOURCE_CVAT), (KIND_TABLE, SOURCE_AUTO)]

    # Повторный полный пересчёт таблиц не удваивает их.
    _run(pack, db, "--no-raster")
    assert sorted(r.kind for r in _page(db).rect_regions) == [KIND_GRAYSCALE, KIND_TABLE]


def test_cover_page_gets_an_empty_table_set(tmp_path: Path) -> None:
    """Обложка считается «без таблиц», а не «таблицы не искали»: иначе прогон возвращался бы к ней вечно."""
    pack = tmp_path / "пак-1"
    _table_page(pack / "1974" / "01" / "a.tif")
    db = tmp_path / "m.sqlite"
    result = CliRunner().invoke(
        main, ["detect", "--pack-dir", str(pack), "--db", str(db), "--no-use-surya-layout", "--no-orientation"]
    )
    assert result.exit_code == 0, result.output + str(result.exception)
    page = _page(db)
    assert page.tables_detected_at is not None
    assert [r.kind for r in page.rect_regions if r.kind in TABLE_KINDS] == []


def test_tables_only_rerun_does_not_wake_the_arbiter(tmp_path: Path, monkeypatch) -> None:
    """Полоса, которой нужны одни таблицы, к арбитру ориентации не идёт: её ответ уже принят."""
    from ocr_utils.scan_markup.detection import run as run_module

    seen: list[dict] = []
    original = run_module._run_arbiter

    def spy(session, params, by_rel, stats):
        seen.append(dict(by_rel))
        return original(session, params, by_rel, stats)

    monkeypatch.setattr(run_module, "_run_arbiter", spy)
    pack = tmp_path / "пак-1"
    _table_page(pack / "1974" / "01" / "a.tif")
    db = tmp_path / "m.sqlite"
    common = ["detect", "--pack-dir", str(pack), "--db", str(db), "--no-use-surya-layout", "--no-first-page-is-cover"]
    result = CliRunner().invoke(main, [*common, "--orientation-detectors", "ink_axis,ocr_vote"])
    assert result.exit_code == 0, result.output + str(result.exception)
    assert len(seen) == 1 and len(seen[0]) == 1, "первый прогон считал ориентацию — полоса у арбитра"

    with open_db(db)() as session:
        page = session.scalars(select(Page)).one()
        page.table_detector_version = TABLE_DETECTOR_VERSION - 1
        session.commit()
    result = CliRunner().invoke(main, [*common, "--orientation-detectors", "ink_axis,ocr_vote", "--skip-detected"])
    assert result.exit_code == 0, result.output + str(result.exception)
    assert len(seen) == 2 and seen[1] == {}, "пересчитывались одни таблицы — арбитру смотреть нечего"
