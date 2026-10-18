"""Первичная заливка с отбором видов (``to-cvat --kinds``) и удаление проекта с отвязкой базы (``delete-project``)."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

from ocr_utils.db.models import KIND_COLOR, KIND_ROTATED_TEXT, KIND_STAMP_SUSPECT, KIND_TABLE, RectRegion
from ocr_utils.db.repo import require_pack
from ocr_utils.scan_markup.cvat import delete, publish
from ocr_utils.scan_markup.cvat.project import LABEL_RASTER_COLOR, LABEL_ROTATED_TEXT, LABEL_STAMP_SUSPECT, LABEL_TABLE
from tests.ocr_utils.scan_markup.test_publish_drift import _fake_cvat, _params, _year_task, pack_db  # noqa: F401

LABEL_IDS = {LABEL_RASTER_COLOR: 11, LABEL_TABLE: 17, LABEL_ROTATED_TEXT: 19, LABEL_STAMP_SUSPECT: 13}


def _add_regions(factory) -> None:
    """На первой полосе — растр, таблица, повёрнутый текст и подозрение на печать."""
    with factory() as session:
        pack = require_pack(session, "пак-1")
        page = pack.year_packages[0].issues[0].pages[0]
        page.rect_regions = [
            RectRegion(x1=8, y1=16, x2=240, y2=320, kind=KIND_COLOR),
            RectRegion(x1=800, y1=1600, x2=3000, y2=3000, kind=KIND_TABLE),
            RectRegion(x1=100, y1=3200, x2=400, y2=5000, kind=KIND_ROTATED_TEXT),
            RectRegion(x1=2000, y1=100, x2=2600, y2=500, kind=KIND_STAMP_SUSPECT),
        ]
        session.commit()


def test_first_publish_uploads_only_wanted_kinds(monkeypatch, pack_db):
    """``kinds`` отбирает виды первичной предразметки: повёрнутый текст и печать не заливаются."""
    db, factory, tmp_path = pack_db
    _add_regions(factory)
    tasks: dict = {}
    _fake_cvat(monkeypatch, tasks, [])
    monkeypatch.setattr(publish, "project_label_ids", lambda client, pid: LABEL_IDS)

    stats = publish.run_publish(_params(db, tmp_path, kinds=(KIND_COLOR, KIND_TABLE)), factory)

    uploaded_shapes = _year_task(tasks, "1974").uploaded
    assert stats.shapes == 2
    assert sorted(shape.label_id for shape in uploaded_shapes) == [11, 17]


def test_first_publish_without_kinds_uploads_everything(monkeypatch, pack_db):
    """Пустой ``kinds`` — прежнее поведение: все виды из базы."""
    db, factory, tmp_path = pack_db
    _add_regions(factory)
    tasks: dict = {}
    _fake_cvat(monkeypatch, tasks, [])
    monkeypatch.setattr(publish, "project_label_ids", lambda client, pid: LABEL_IDS)

    stats = publish.run_publish(_params(db, tmp_path), factory)
    assert stats.shapes == 4


class _Job:
    def __init__(self, state="new", stage="annotation"):
        self.state, self.stage = state, stage


class _Task:
    """Задача проекта: два кадра, один растр на первом, тег на втором."""

    def __init__(self, task_id, project_id, jobs):
        self.id, self.project_id, self.name = task_id, project_id, "1974 · выпусков 2 · полос 4"
        self._jobs = jobs

    def get_jobs(self):
        return self._jobs

    def get_frames_info(self):
        return [SimpleNamespace(name="пак-1/1974/01/a.jpg"), SimpleNamespace(name="пак-1/1974/01/b.jpg")]

    def get_annotations(self):
        shape = SimpleNamespace(frame=0, label_id=11, to_dict=lambda: {"frame": 0, "label_id": 11})
        tag = SimpleNamespace(frame=1, label_id=21, to_dict=lambda: {"frame": 1, "label_id": 21})
        return SimpleNamespace(shapes=[shape], tags=[tag])


def _fake_client(monkeypatch, projects, tasks, destroyed):
    """Клиент CVAT без сети: списки проектов и задач, журнал удалений."""
    client = SimpleNamespace(
        projects=SimpleNamespace(list=lambda: projects),
        tasks=SimpleNamespace(list=lambda: tasks),
        api_client=SimpleNamespace(projects_api=SimpleNamespace(destroy=destroyed.append)),
    )

    @contextlib.contextmanager
    def make(settings):
        yield client

    monkeypatch.setattr(delete, "make_cvat_client", make)
    monkeypatch.setattr(delete, "project_label_ids", lambda c, pid: {LABEL_RASTER_COLOR: 11, "Оглавление": 21})


def _link_pack(factory) -> None:
    """Пак будто бы уже залит: id проекта, задачи, джобов, кадры и отпечатки."""
    with factory() as session:
        pack = require_pack(session, "пак-1")
        pack.cvat_project_id = 6
        for year in pack.year_packages:
            year.cvat_task_id = 43
            for index, issue in enumerate(year.issues):
                issue.cvat_job_id = 450 + index
                for frame, page in enumerate(issue.pages):
                    page.cvat_rel_path = f"пак-1/{page.source_rel_path[:-4]}.jpg"
                    page.cvat_frame, page.cvat_file_hash = frame, page.file_hash
        session.commit()


def test_dry_run_counts_and_deletes_nothing(monkeypatch, pack_db):
    db, factory, tmp_path = pack_db
    _link_pack(factory)
    destroyed: list = []
    project = SimpleNamespace(id=6, name="пак-1")
    _fake_client(monkeypatch, [project], [_Task(43, 6, [_Job(), _Job("completed")])], destroyed)

    stats = delete.run_delete(delete.DeleteParams(db_path=db, pack_name="пак-1"), factory)

    assert (stats.project_id, stats.tasks, stats.jobs, stats.jobs_touched) == (6, 1, 2, 1)
    assert stats.shapes_by_label == {LABEL_RASTER_COLOR: 1} and stats.tags_by_label == {"Оглавление": 1}
    assert destroyed == [] and not stats.deleted and stats.backups == []
    with factory() as session:
        assert require_pack(session, "пак-1").cvat_project_id == 6, "сухой прогон базу не трогает"


def test_confirmed_delete_backs_up_destroys_and_unlinks(monkeypatch, pack_db):
    db, factory, tmp_path = pack_db
    _link_pack(factory)
    destroyed: list = []
    # Чужой проект с тем же именем, но другим id не трогается: сперва ищем по id из базы.
    projects = [SimpleNamespace(id=2, name="пак-1"), SimpleNamespace(id=6, name="пак-1 (переименован)")]
    _fake_client(monkeypatch, projects, [_Task(43, 6, [_Job()]), _Task(44, 2, [_Job()])], destroyed)

    stats = delete.run_delete(delete.DeleteParams(db_path=db, pack_name="пак-1", confirm=True), factory)

    assert destroyed == [6] and stats.deleted and stats.tasks == 1
    assert len(stats.backups) == 1 and stats.backups[0].exists()
    with factory() as session:
        pack = require_pack(session, "пак-1")
        assert pack.cvat_project_id is None
        year = pack.year_packages[0]
        assert year.cvat_task_id is None
        assert all(issue.cvat_job_id is None for issue in year.issues)
        pages = [page for issue in year.issues for page in issue.pages]
        assert all(page.cvat_frame is None and page.cvat_file_hash is None for page in pages)
        assert all(page.cvat_rel_path is not None for page in pages), "пути к превью в share остаются"
