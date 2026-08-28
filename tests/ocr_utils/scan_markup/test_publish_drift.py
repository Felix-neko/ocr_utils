"""Обновление пака по частям: поиск разошедшихся полос и пересоздание задачи-года.

Настоящий CVAT не нужен: проверяется наша половина — что считается расхождением, какая
разметка переносится в новую задачу и в каком порядке трогается сервер. Порядок здесь не
косметика: старую задачу удаляют последней, чтобы сбой не унёс разметку.
"""

import pytest
from cvat_sdk import models

from ocr_utils.scan_markup.cvat import publish
from ocr_utils.scan_markup.cvat.project import shapes_to_requests
from ocr_utils.scan_markup.db.models import Issue, Page


def make_page(name: str, file_hash: str | None, cvat_hash: str | None) -> Page:
    page = Page(file_name=name, rel_path=f"1974/01/{name}", order_index=0)
    page.file_hash = file_hash
    page.cvat_file_hash = cvat_hash
    page.cvat_rel_path = f"пак-1/1974/01/{name}"
    return page


def make_issue(pages: list[Page], name: str = "01") -> Issue:
    issue = Issue(name=name, number=1, rel_path=f"1974/{name}")
    issue.pages = pages
    issue.cvat_job_id = 42
    return issue


def test_issue_drift_splits_changed_and_added():
    """Изменившиеся и ни разу не залитые — разные беды и лечатся по-разному."""
    same = make_page("a.tif", "h1", "h1")
    changed = make_page("b.tif", "h2new", "h2old")
    added = make_page("c.tif", "h3", None)

    got_changed, got_added = publish.issue_drift(make_issue([same, changed, added]))
    assert [p.file_name for p in got_changed] == ["b.tif"]
    assert [p.file_name for p in got_added] == ["c.tif"]


def test_issue_without_drift_is_dropped():
    issue = make_issue([make_page("a.tif", "h1", "h1")])
    assert publish.year_drift([issue]) == []


class _Frame:
    def __init__(self, name):
        self.name = name


class _Attr:
    def __init__(self):
        self.spec_id, self.value = 1, "x"


class _Shape:
    def __init__(self, frame, points, label_id=11, type_="rectangle"):
        self.frame, self.points, self.label_id, self.type = frame, points, label_id, type_
        self.occluded, self.outside, self.z_order, self.group, self.rotation = False, False, 0, 0, 0.0
        self.attributes = []

    def to_dict(self):
        return {"frame": self.frame, "points": self.points, "label_id": self.label_id, "type": self.type}


class _Annotations:
    def __init__(self, shapes):
        self.shapes = shapes


class _Task:
    """Поддельная задача, ведущая журнал того, что с ней сделали."""

    def __init__(self, task_id, name, frame_names, shapes, log):
        self.id, self.name, self._log = task_id, name, log
        self._frames = [_Frame(n) for n in frame_names]
        self._shapes = shapes
        self.uploaded = None

    def get_frames_info(self):
        return self._frames

    def get_annotations(self):
        return _Annotations(self._shapes)

    def set_annotations(self, data):
        self.uploaded = data.shapes
        self._log.append(("upload", self.id, len(data.shapes)))

    def remove(self):
        self._log.append(("remove", self.id))

    def update(self, spec):
        self.name = spec.name
        self._log.append(("rename", self.id, spec.name))


def test_shapes_are_carried_by_frame_name_not_number():
    """Новая задача нумерует кадры заново, поэтому переносим по именам."""
    by_frame = {"пак-1/1974/01/a.tif": [_Shape(0, [1, 2, 3, 4])], "пак-1/1974/01/b.tif": [_Shape(1, [5, 6, 7, 8])]}
    # В новой задаче те же кадры идут в обратном порядке.
    frames = {"пак-1/1974/01/a.tif": 1, "пак-1/1974/01/b.tif": 0}

    requests = shapes_to_requests(by_frame, frames, skip_names=set())
    by_points = {tuple(r.points): r.frame for r in requests}
    assert by_points == {(1, 2, 3, 4): 1, (5, 6, 7, 8): 0}


def test_changed_frames_lose_their_manual_markup():
    """Разметка с изменившегося кадра не переносится: она обводит уже не тот файл."""
    by_frame = {"пак-1/1974/01/a.tif": [_Shape(0, [1, 2, 3, 4])], "пак-1/1974/01/b.tif": [_Shape(1, [5, 6, 7, 8])]}
    frames = {"пак-1/1974/01/a.tif": 0, "пак-1/1974/01/b.tif": 1}

    requests = shapes_to_requests(by_frame, frames, skip_names={"пак-1/1974/01/b.tif"})
    assert [tuple(r.points) for r in requests] == [(1, 2, 3, 4)]


def test_rebuild_deletes_old_task_only_after_new_one_is_filled(monkeypatch, tmp_path):
    """Порядок операций: создать -> залить -> удалить старую -> переименовать.

    Обратный порядок означал бы окно, в котором разметки нет уже нигде.
    """
    log = []
    old = _Task(7, "1974", ["пак-1/1974/01/a.tif"], [_Shape(0, [1, 2, 3, 4])], log)
    new = _Task(8, "1974 (пересоздание)", ["пак-1/1974/01/a.tif"], [], log)

    def fake_create(client, project_id, name, job_files):
        log.append(("create", name))
        return new

    monkeypatch.setattr(publish, "create_year_task", fake_create)

    result = publish.rebuild_year_task(
        client=None,
        project_id=1,
        year_name="1974",
        old_task=old,
        job_files=[["пак-1/1974/01/a.tif"]],
        carry=lambda frames: [models.LabeledShapeRequest(type="rectangle", frame=0, label_id=11, points=[1, 2, 3, 4])],
    )

    assert result is new
    assert log == [("create", "1974 (пересоздание)"), ("upload", 8, 1), ("remove", 7), ("rename", 8, "1974")]
    assert new.name == "1974"


def test_backup_is_written_before_anything_is_removed(tmp_path):
    """Бэкап — единственное, что остаётся, если перенос окажется неполным."""
    task = _Task(7, "1974", ["пак-1/1974/01/a.tif"], [], [])
    by_frame = {"пак-1/1974/01/a.tif": [_Shape(0, [1, 2, 3, 4])]}

    path = publish._backup_annotations(tmp_path, "пак-1", "1974", task, by_frame)
    assert path.exists()

    import json

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["task_id"] == 7
    assert saved["frames"]["пак-1/1974/01/a.tif"][0]["points"] == [1, 2, 3, 4]


class _Job:
    def __init__(self, job_id, start_frame):
        self.id, self.start_frame = job_id, start_frame

    def update(self, spec):
        pass


class _FullTask(_Task):
    """Задача с джобами — то, чего просит assign_jobs_to_issues."""

    def __init__(self, task_id, name, frame_names, shapes, log, job_starts):
        super().__init__(task_id, name, frame_names, shapes, log)
        self._jobs = [_Job(100 + task_id * 10 + i, start) for i, start in enumerate(job_starts)]

    def get_jobs(self):
        return self._jobs


def _fake_cvat(monkeypatch, tasks_by_name, log):
    """Подменяет всё, что ходит по сети, оставляя настоящей логику publish."""
    import contextlib

    from pathlib import Path as _Path

    class _Project:
        id = 1

    @contextlib.contextmanager
    def fake_client(settings):
        yield object()

    monkeypatch.setattr(publish, "make_cvat_client", fake_client)
    monkeypatch.setattr(publish, "ensure_project", lambda client, name: _Project())
    monkeypatch.setattr(publish, "project_label_ids", lambda client, pid: {})
    monkeypatch.setattr(publish, "check_share_root", lambda root: None)
    monkeypatch.setattr(publish, "share_prefix", lambda root: _Path(""))
    monkeypatch.setattr(publish, "prepare_images", lambda jobs, root, workers, force: [])
    monkeypatch.setattr(publish, "assign_annotator", lambda client, task, name: None)
    monkeypatch.setattr(publish, "find_task", lambda client, pid, name: tasks_by_name.get(name))

    def fake_create(client, project_id, name, job_files):
        log.append(("create", name))
        starts, offset = [], 0
        for files in job_files:
            starts.append(offset)
            offset += len(files)
        task = _FullTask(len(tasks_by_name) + 7, name, [f for files in job_files for f in files], [], log, starts)
        tasks_by_name[name] = task
        return task

    monkeypatch.setattr(publish, "create_year_task", fake_create)
    return tasks_by_name


@pytest.fixture
def pack_db(tmp_path):
    """Пак из одного года, двух выпусков по две полосы, уже прошедший detect."""
    from ocr_utils.scan_markup.db.repo import upsert_pack
    from ocr_utils.scan_markup.db.session import open_db
    from ocr_utils.scan_markup.scan_tree import ScannedIssue, ScannedPage, ScannedYear

    db = tmp_path / "markup.sqlite"
    factory = open_db(db)
    issues = []
    for issue_name in ("01", "02"):
        pages = [
            ScannedPage(path=tmp_path / name, file_name=name, rel_path=f"1974/{issue_name}/{name}", order_index=index)
            for index, name in enumerate(("a.tif", "b.tif"))
        ]
        issues.append(ScannedIssue(name=issue_name, number=int(issue_name), rel_path=f"1974/{issue_name}", pages=pages))
    tree = [ScannedYear(name="1974", year=1974, rel_path="1974", issues=issues)]

    with factory() as session:
        pack = upsert_pack(session, "пак-1", tmp_path / "пак-1", tree)
        for year in pack.year_packages:
            for issue in year.issues:
                for page in issue.pages:
                    page.width, page.height, page.dpi = 3492, 6051, 600
                    page.divisor, page.crop_width, page.crop_height = 8, 3488, 6048
                    page.cvat_width, page.cvat_height = 436, 756
                    page.file_hash = f"hash-{issue.name}-{page.file_name}"
                    page.file_size, page.file_mtime = 100, 1.0
        session.commit()
    return db, factory, tmp_path


def _params(db, tmp_path, **extra):
    return publish.PublishParams(
        db_path=db, pack_name="пак-1", share_root=tmp_path / "share", pack_dir=tmp_path / "пак-1", **extra
    )


def test_first_publish_records_what_was_uploaded(monkeypatch, pack_db):
    """После заливки у каждой полосы запомнен хеш файла, который лёг под кадр."""
    db, factory, tmp_path = pack_db
    _fake_cvat(monkeypatch, {}, [])

    stats = publish.run_publish(_params(db, tmp_path), factory)
    assert stats.tasks_created == 1
    assert stats.stale_years == []

    with factory() as session:
        from ocr_utils.scan_markup.db.repo import require_pack

        pages = [p for y in require_pack(session, "пак-1").year_packages for i in y.issues for p in i.pages]
        assert all(p.cvat_file_hash == p.file_hash for p in pages)
        assert all(p.cvat_frame is not None for p in pages)


def test_changed_file_is_reported_and_task_left_alone_without_flag(monkeypatch, pack_db):
    """Без --recreate-stale команда только показывает, какие джобы задеты."""
    db, factory, tmp_path = pack_db
    tasks, log = {}, []
    _fake_cvat(monkeypatch, tasks, log)
    publish.run_publish(_params(db, tmp_path), factory)

    with factory() as session:
        from ocr_utils.scan_markup.db.repo import require_pack

        page = require_pack(session, "пак-1").year_packages[0].issues[1].pages[0]
        page.file_hash = "hash-новый"
        session.commit()

    log.clear()
    stats = publish.run_publish(_params(db, tmp_path), factory)
    assert stats.stale_years == ["1974"]
    assert stats.pages_changed == 1
    assert stats.tasks_rebuilt == 0
    assert log == [], "без флага сервер трогать нельзя"


def test_recreate_stale_keeps_markup_of_untouched_pages(monkeypatch, pack_db):
    """Пересоздание года: разметка неизменившихся полос переезжает, изменившейся — нет."""
    db, factory, tmp_path = pack_db
    tasks, log = {}, []
    _fake_cvat(monkeypatch, tasks, log)
    publish.run_publish(_params(db, tmp_path), factory)

    # Разметчик обвёл по объекту на каждом кадре.
    task = tasks["1974"]
    task._shapes = [_Shape(index, [1, 2, 3, 4]) for index in range(4)]
    changed_name = task.get_frames_info()[2].name  # первая полоса второго выпуска

    with factory() as session:
        from ocr_utils.scan_markup.db.repo import require_pack

        pack = require_pack(session, "пак-1")
        page = next(p for y in pack.year_packages for i in y.issues for p in i.pages if p.cvat_rel_path == changed_name)
        page.file_hash = "hash-новый"
        session.commit()

    log.clear()
    stats = publish.run_publish(_params(db, tmp_path, recreate_stale=True), factory)

    assert stats.tasks_rebuilt == 1
    # Переехали три шейпа из четырёх — тот, что на подменённой полосе, отброшен.
    assert stats.shapes_carried == 3
    assert [entry[0] for entry in log] == ["create", "upload", "remove", "rename"]

    with factory() as session:
        from ocr_utils.scan_markup.db.repo import require_pack

        pack = require_pack(session, "пак-1")
        pages = [p for y in pack.year_packages for i in y.issues for p in i.pages]
        assert all(p.cvat_file_hash == p.file_hash for p in pages), "после пересоздания расхождений быть не должно"
        # Год теперь указывает на НОВУЮ задачу, а не на удалённую.
        new_task = next(t for t in tasks.values() if t.name == "1974" and t.id != 7)
        assert pack.year_packages[0].cvat_task_id == new_task.id


def test_leftover_temp_task_is_cleaned_up(monkeypatch, pack_db):
    """Прерванное пересоздание оставляет двойника — следующий прогон его сносит."""
    db, factory, tmp_path = pack_db
    log = []
    leftover = _FullTask(99, "1974 (пересоздание)", [], [], log, [])
    tasks = {"1974 (пересоздание)": leftover}
    _fake_cvat(monkeypatch, tasks, log)

    publish.run_publish(_params(db, tmp_path), factory)
    assert ("remove", 99) in log
