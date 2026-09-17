"""Оглавление в CVAT: теги «Оглавление» и «Годовой указатель» туда (заливка и дозаливка) и обратно.

Как и у поворота, тег не виден на холсте и теряется молча, поэтому проверяется и «доехал»,
и «снятый разметчиком тег означает False», и что дозаливка идёт PATCH-ем без дублей.
"""

from __future__ import annotations

from types import SimpleNamespace

from ocr_utils.scan_markup.cvat import publish
from ocr_utils.scan_markup.cvat.export import ExportParams, ExportStats, _toc_from_tags, import_task
from ocr_utils.scan_markup.cvat.project import (
    LABEL_NOT_TOC,
    LABEL_RASTER_COLOR,
    LABEL_ROTATE_CW90,
    LABEL_TOC,
    LABEL_YEAR_INDEX,
    LABELS,
    TOC_LABEL_BY_FIELD,
    page_tags,
    toc_tags,
)
from ocr_utils.db.models import SOURCE_CVAT
from ocr_utils.db.repo import require_pack
from ocr_utils.scan_markup.toc import TOC_VERSION
from tests.ocr_utils.scan_markup.test_cvat_roundtrip import _Task as _ExportTask, page_and_session  # noqa: F401
from tests.ocr_utils.scan_markup.test_publish_drift import (
    _Annotations,
    _FullTask,
    _fake_cvat,
    _params,
    pack_db,
)  # noqa: F401

FRAMES = {"пак-1/1975/12/a.jpg": 0, "пак-1/1975/12/b.jpg": 1, "пак-1/1975/12/c.jpg": 2}
LABEL_IDS = {LABEL_RASTER_COLOR: 11, LABEL_ROTATE_CW90: 101, LABEL_TOC: 201, LABEL_YEAR_INDEX: 202, LABEL_NOT_TOC: 203}
LABEL_NAMES = {value: key for key, value in LABEL_IDS.items()}


def _page(rel_path, is_toc=None, is_year_index=None, rotate_cw=0, force_is_not_toc=None):
    return SimpleNamespace(
        cvat_rel_path=rel_path,
        is_toc=is_toc,
        is_year_index=is_year_index,
        rotate_cw=rotate_cw,
        force_is_not_toc=force_is_not_toc,
    )


def test_toc_labels_are_tags():
    by_name = {label["name"]: label for label in LABELS}
    assert by_name[LABEL_TOC]["type"] == "tag" and by_name[LABEL_YEAR_INDEX]["type"] == "tag"
    assert by_name[LABEL_NOT_TOC]["type"] == "tag"
    assert set(TOC_LABEL_BY_FIELD) == {"is_toc", "is_year_index", "force_is_not_toc"}


def test_veto_tag_travels_to_cvat_and_back(page_and_session):
    """«Не оглавление» ставится только руками, но из базы переливается и из CVAT читается."""
    pages = [_page("пак-1/1975/12/a.jpg", is_toc=True, force_is_not_toc=True), _page("пак-1/1975/12/b.jpg")]
    assert sorted((t.frame, t.label_id) for t in toc_tags(pages, FRAMES, LABEL_IDS)) == [(0, 201), (0, 203)]

    page, session = page_and_session
    task = _ExportTask([page.cvat_rel_path], [])
    task.get_annotations = lambda: _Annotations([], [SimpleNamespace(frame=0, label_id=203)])
    stats = ExportStats()
    import_task(task, LABEL_NAMES, {0: page}, session, ExportParams(None, None, "пак-1"), stats)
    assert (page.is_toc, page.is_year_index, page.force_is_not_toc) == (False, False, True)
    assert stats.not_toc_pages == 1
    task.get_annotations = lambda: _Annotations([], [])
    import_task(task, LABEL_NAMES, {0: page}, session, ExportParams(None, None, "пак-1"), ExportStats())
    assert page.force_is_not_toc is False, "снятое вето — False, снимок тегов"


def test_only_flagged_pages_get_tags_and_both_flags_give_two():
    pages = [
        _page("пак-1/1975/12/a.jpg", is_toc=True),
        _page("пак-1/1975/12/b.jpg", is_year_index=True, is_toc=True),
        _page("пак-1/1975/12/c.jpg", is_toc=False),
        _page("пак-1/1999/01/z.jpg", is_toc=True),  # чужая полоса
    ]
    tags = toc_tags(pages, FRAMES, LABEL_IDS)
    assert sorted((t.frame, t.label_id) for t in tags) == [(0, 201), (1, 201), (1, 202)]
    # Без метки в проекте тег не строится, а не падает.
    assert toc_tags(pages, FRAMES, {LABEL_RASTER_COLOR: 11}) == []


def test_page_tags_join_rotation_and_toc():
    pages = [_page("пак-1/1975/12/a.jpg", is_toc=True, rotate_cw=90)]
    assert sorted(t.label_id for t in page_tags(pages, FRAMES, LABEL_IDS)) == [101, 201]


def test_import_reads_flags_from_tags_and_absent_tag_means_false(page_and_session):
    page, session = page_and_session
    page.is_toc, page.is_year_index, page.toc_score, page.toc_version = False, False, 0.7, TOC_VERSION
    tag = SimpleNamespace(frame=0, label_id=LABEL_IDS[LABEL_YEAR_INDEX])
    task = _ExportTask([page.cvat_rel_path], [])
    task.get_annotations = lambda: _Annotations([], [tag])
    stats = ExportStats()
    import_task(task, LABEL_NAMES, {0: page}, session, ExportParams(None, None, "пак-1"), stats)
    assert (page.is_toc, page.is_year_index) == (False, True)
    assert page.toc_source == SOURCE_CVAT and page.toc_score is None and page.toc_version is None
    assert (stats.toc_pages, stats.year_index_pages) == (0, 1)

    task.get_annotations = lambda: _Annotations([], [])
    import_task(task, LABEL_NAMES, {0: page}, session, ExportParams(None, None, "пак-1"), ExportStats())
    assert (page.is_toc, page.is_year_index) == (False, False), "снятый тег — это False, а не «как было»"


def test_toc_from_tags_ignores_foreign_labels():
    tags = [SimpleNamespace(frame=0, label_id=11), SimpleNamespace(frame=0, label_id=201)]
    assert _toc_from_tags(tags, LABEL_NAMES) == {"is_toc": True, "is_year_index": False, "force_is_not_toc": False}


class _TaggingTask(_FullTask):
    """Задача, которая помнит дозалитые теги и не принимает замену разметки."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tags: list = []
        self.appended: list = []

    def get_annotations(self):
        return _Annotations(self._shapes, self._tags)

    def update_annotations(self, data, *, action):
        tags = list(getattr(data, "tags", None) or [])
        self.appended.append((str(getattr(action, "value", action)), tags))
        self._tags.extend(SimpleNamespace(frame=t.frame, label_id=t.label_id, group=0) for t in tags)

    def set_annotations(self, data):
        raise AssertionError("замена разметки затёрла бы ручную правку — дозаливка обязана идти PATCH-ем")


def _published_pack(monkeypatch, pack_db):
    """Пак уже в CVAT; в базе детектор пометил первую полосу «Содержанием», последнюю — указателем."""
    db, factory, tmp_path = pack_db
    frame_names = [f"пак-1/1974/{issue}/{name}" for issue in ("01", "02") for name in ("a.jpg", "b.jpg")]
    log: list = []
    task = _TaggingTask(5, "1974 (2 вып., 4 пол.)", frame_names, [], log, [0, 2])
    _fake_cvat(monkeypatch, {task.name: task}, log)
    monkeypatch.setattr(publish, "project_label_ids", lambda client, pid: LABEL_IDS)
    with factory() as session:
        pack = require_pack(session, "пак-1")
        pack.year_packages[0].cvat_task_id = task.id
        pages = [p for issue in pack.year_packages[0].issues for p in issue.pages]
        for page in pages:
            page.cvat_rel_path = f"пак-1/{page.source_rel_path[:-4]}.jpg"
            page.cvat_file_hash = page.file_hash
            page.is_toc, page.is_year_index = False, False
        pages[0].is_toc = True
        pages[3].is_year_index = True
        session.commit()
    return db, factory, tmp_path, task


def test_append_tags_goes_by_patch_and_is_idempotent(monkeypatch, pack_db):
    db, factory, tmp_path, task = _published_pack(monkeypatch, pack_db)

    stats = publish.run_publish(_params(db, tmp_path, append_tags=True), factory)
    assert stats.tasks_existing == 1 and stats.tags == 0 and stats.tags_appended == 2
    assert [action for action, _ in task.appended] == ["create"]
    assert sorted((t.frame, t.label_id) for t in task.appended[0][1]) == [(0, 201), (3, 202)]

    # Разметчик снял тег с последнего кадра и поставил «Оглавление» на второй — второй прогон
    # ничего не возвращает и не удваивает.
    task._tags = [t for t in task._tags if t.frame != 3] + [SimpleNamespace(frame=1, label_id=201, group=0)]
    with factory() as session:
        pack = require_pack(session, "пак-1")
        pages = [p for issue in pack.year_packages[0].issues for p in issue.pages]
        pages[1].is_toc = True
        session.commit()
    stats = publish.run_publish(_params(db, tmp_path, append_tags=True), factory)
    assert stats.tags_appended == 1, "только кадр 3: у него теперь нет тега, а в базе признак есть"
    assert sorted((t.frame, t.label_id) for t in task.appended[1][1]) == [(3, 202)]


def test_append_tags_without_flag_touches_nothing(monkeypatch, pack_db):
    db, factory, tmp_path, task = _published_pack(monkeypatch, pack_db)
    stats = publish.run_publish(_params(db, tmp_path), factory)
    assert stats.tags_appended == 0 and task.appended == []
