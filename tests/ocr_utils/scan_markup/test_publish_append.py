"""Дозаливка находок детектора таблиц в уже размеченные задачи: PATCH, а не замена, и без дублей."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ocr_utils.scan_markup.cvat import publish
from ocr_utils.scan_markup.cvat.project import LABEL_LINE_ART, LABEL_RASTER_COLOR, LABEL_TABLE
from ocr_utils.db.models import KIND_COLOR, KIND_LINE_ART_SCHEMA, KIND_TABLE, SOURCE_CVAT, RectRegion
from ocr_utils.db.repo import require_pack
from tests.ocr_utils.scan_markup.test_publish_drift import _FullTask, _Shape, _fake_cvat, _params, pack_db  # noqa: F401

LABEL_IDS = {LABEL_RASTER_COLOR: 11, LABEL_TABLE: 17, LABEL_LINE_ART: 18}


class _PatchingTask(_FullTask):
    """Задача, которая помнит дозалитое и НИКОГДА не принимает замену разметки."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.appended = []

    def update_annotations(self, data, *, action):
        self.appended.append((str(getattr(action, "value", action)), list(data.shapes)))
        self._shapes.extend(SimpleNamespace(frame=s.frame, label_id=s.label_id, points=s.points) for s in data.shapes)

    def set_annotations(self, data):
        raise AssertionError("замена разметки затёрла бы ручную правку — дозаливка обязана идти PATCH-ем")


def _published_pack(monkeypatch, pack_db):
    """Пак уже в CVAT: задача-год есть, на первом кадре — ручной растр; в базе — таблицы и схемы."""
    db, factory, tmp_path = pack_db
    # Имена кадров — пути уменьшенных копий в share: .jpg, а не .tif оригинала.
    frame_names = [f"пак-1/1974/{issue}/{name}" for issue in ("01", "02") for name in ("a.jpg", "b.jpg")]
    manual = _Shape(0, [1, 2, 30, 40], label_id=11)
    log: list = []
    task = _PatchingTask(5, "1974 (2 вып., 4 пол.)", frame_names, [manual], log, [0, 2])
    tasks = _fake_cvat(monkeypatch, {task.name: task}, log)
    monkeypatch.setattr(publish, "project_label_ids", lambda client, pid: LABEL_IDS)

    with factory() as session:
        pack = require_pack(session, "пак-1")
        pack.year_packages[0].cvat_task_id = task.id
        pages = [p for issue in pack.year_packages[0].issues for p in issue.pages]
        for page in pages:
            page.cvat_rel_path = f"пак-1/{page.source_rel_path[:-4]}.jpg"
            page.cvat_file_hash = page.file_hash
        pages[0].rect_regions = [
            RectRegion(x1=8, y1=16, x2=240, y2=320, kind=KIND_COLOR, source=SOURCE_CVAT),
            RectRegion(x1=800, y1=1600, x2=3000, y2=3000, kind=KIND_TABLE),
        ]
        pages[1].rect_regions = [RectRegion(x1=400, y1=3200, x2=3200, y2=5600, kind=KIND_LINE_ART_SCHEMA)]
        pages[3].rect_regions = [RectRegion(x1=100, y1=100, x2=2000, y2=2000, kind=KIND_COLOR)]
        session.commit()
    return db, factory, tmp_path, task, tasks


def test_append_adds_only_wanted_kinds_by_patch(monkeypatch, pack_db):
    db, factory, tmp_path, task, _tasks = _published_pack(monkeypatch, pack_db)

    stats = publish.run_publish(_params(db, tmp_path, append_kinds=(KIND_TABLE, KIND_LINE_ART_SCHEMA)), factory)

    assert stats.tasks_existing == 1 and stats.shapes == 0, "обычной заливки в существующую задачу нет"
    assert (stats.shapes_appended, stats.frames_appended) == (2, 2)
    assert [action for action, _ in task.appended] == ["create"]
    added = task.appended[0][1]
    assert sorted((s.frame, s.label_id) for s in added) == [(0, 17), (1, 18)]
    assert all(s.label_id != 11 for s in added), "растр в дозаливку не попадает"


def test_append_skips_frames_that_already_have_such_labels(monkeypatch, pack_db):
    """Второй прогон ничего не удваивает; рамку, снятую разметчиком, назад не возвращает."""
    db, factory, tmp_path, task, _tasks = _published_pack(monkeypatch, pack_db)
    params = _params(db, tmp_path, append_kinds=(KIND_TABLE, KIND_LINE_ART_SCHEMA))

    publish.run_publish(params, factory)
    # Разметчик убрал схему на втором кадре, но оставил свою собственную таблицу там же.
    task._shapes = [s for s in task._shapes if not (s.frame == 1 and s.label_id == 18)]
    task._shapes.append(SimpleNamespace(frame=1, label_id=17, points=[5, 5, 50, 50]))

    stats = publish.run_publish(params, factory)
    assert stats.shapes_appended == 0
    assert len(task.appended) == 1, "повторный прогон не шлёт даже пустого PATCH"


def test_append_without_flag_touches_nothing(monkeypatch, pack_db):
    db, factory, tmp_path, task, _tasks = _published_pack(monkeypatch, pack_db)
    stats = publish.run_publish(_params(db, tmp_path), factory)
    assert stats.shapes_appended == 0 and task.appended == []


def test_append_needs_labels_in_the_project(monkeypatch, pack_db, caplog):
    """Нет метки в проекте — дозаливка не идёт, а говорит об этом; тихо потерять вид нельзя."""
    db, factory, tmp_path, task, _tasks = _published_pack(monkeypatch, pack_db)
    monkeypatch.setattr(publish, "project_label_ids", lambda client, pid: {LABEL_RASTER_COLOR: 11})
    with caplog.at_level("WARNING"):
        stats = publish.run_publish(_params(db, tmp_path, append_kinds=(KIND_TABLE,)), factory)
    assert stats.shapes_appended == 0 and task.appended == []
    assert "нет меток" in caplog.text


def test_cli_rejects_unknown_kind() -> None:
    from click.testing import CliRunner

    from ocr_utils.scan_markup.cli import main

    result = CliRunner().invoke(
        main, ["to-cvat", "--db", __file__, "--pack-name", "x", "--share-root", "/tmp", "--append-kinds", "tables"]
    )
    assert result.exit_code != 0 and "неизвестные виды" in result.output


@pytest.mark.parametrize("kinds", ["table,line_art_schema", "table"])
def test_copy_regions_moves_only_wanted_kinds_and_keeps_manual_raster(tmp_path, kinds) -> None:
    """copy-regions: таблицы едут в уточнённую базу, ручной растр в ней остаётся, второй прогон — без дублей."""
    from click.testing import CliRunner
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from ocr_utils.scan_markup.cli import main
    from ocr_utils.scan_markup.cvat.export import copy_tree
    from ocr_utils.db.models import KIND_GRAYSCALE, SOURCE_AUTO, Page
    from ocr_utils.db.repo import upsert_pack
    from ocr_utils.db.session import open_db
    from ocr_utils.scan_markup.scan_tree import ScannedIssue, ScannedPage, ScannedYear

    src_db, dst_db = tmp_path / "src.sqlite", tmp_path / "dst.sqlite"
    pages = [ScannedPage(tmp_path / n, n, f"1974/01/{n}", i) for i, n in enumerate(("a.tif", "b.tif"))]
    tree = [ScannedYear("1974", 1974, "1974", [ScannedIssue("01", 1, "1974/01", pages)])]
    with open_db(src_db)() as src:
        pack = upsert_pack(src, "пак-1", tmp_path, tree)
        first, second = pack.year_packages[0].issues[0].pages
        first.table_detector_version = 1
        first.rect_regions = [
            RectRegion(x1=1, y1=1, x2=50, y2=50, kind=KIND_COLOR),
            RectRegion(x1=100, y1=100, x2=900, y2=900, kind=KIND_TABLE, detector_info='{"kind": "таблица"}'),
            RectRegion(x1=100, y1=1000, x2=900, y2=1900, kind=KIND_LINE_ART_SCHEMA),
        ]
        second.rect_regions = [RectRegion(x1=5, y1=5, x2=60, y2=60, kind=KIND_TABLE)]
        src.commit()
        with open_db(dst_db)() as dst:
            copy_tree(src, dst, "пак-1")
            target = require_pack(dst, "пак-1").year_packages[0].issues[0].pages[0]
            target.rect_regions = [RectRegion(x1=2, y1=2, x2=40, y2=40, kind=KIND_GRAYSCALE, source=SOURCE_CVAT)]
            dst.commit()

    args = ["copy-regions", "--db", str(src_db), "--out-db", str(dst_db), "--pack-name", "пак-1", "--kinds", kinds]
    for _ in range(2):  # второй прогон — те же области, не вдвое больше
        result = CliRunner().invoke(main, args)
        assert result.exit_code == 0, result.output + str(result.exception)

    wanted = tuple(kinds.split(","))
    with open_db(dst_db)() as dst:
        rows = dst.scalars(select(Page).options(selectinload(Page.rect_regions)).order_by(Page.source_file_name)).all()
        first_kinds = sorted((r.kind, r.source) for r in rows[0].rect_regions)
        expected = [(KIND_GRAYSCALE, SOURCE_CVAT)] + sorted((k, SOURCE_AUTO) for k in wanted)
        assert first_kinds == expected
        assert rows[0].table_detector_version == 1
        table = next(r for r in rows[0].rect_regions if r.kind == KIND_TABLE)
        assert (table.x1, table.y2, table.detector_info) == (100, 900, '{"kind": "таблица"}')
        assert [r.kind for r in rows[1].rect_regions] == [KIND_TABLE]

    result = CliRunner().invoke(
        main, ["copy-regions", "--db", str(src_db), "--out-db", str(src_db), "--pack-name", "x"]
    )
    assert result.exit_code != 0 and "совпадает" in result.output


def test_append_goes_into_a_drifted_year_but_skips_changed_pages(monkeypatch, pack_db):
    """Год с изменившимся файлом без --recreate-stale: дозаливка идёт мимо изменившейся полосы,
    отметка о заливке у неё не трогается — расхождение остаётся видно."""
    db, factory, tmp_path, task, _tasks = _published_pack(monkeypatch, pack_db)
    with factory() as session:
        pack = require_pack(session, "пак-1")
        pages = [p for issue in pack.year_packages[0].issues for p in issue.pages]
        pages[0].file_hash = "новый-файл"  # первая полоса разошлась с CVAT; на ней же таблица
        session.commit()

    stats = publish.run_publish(_params(db, tmp_path, append_kinds=(KIND_TABLE, KIND_LINE_ART_SCHEMA)), factory)

    assert stats.stale_years == ["1974"] and stats.tasks_rebuilt == 0
    added = task.appended[0][1]
    assert [(s.frame, s.label_id) for s in added] == [(1, 18)], "таблица изменившейся полосы не дозаливается"
    with factory() as session:
        pack = require_pack(session, "пак-1")
        first = pack.year_packages[0].issues[0].pages[0]
        assert first.cvat_file_hash != first.file_hash, "расхождение не замазано"
