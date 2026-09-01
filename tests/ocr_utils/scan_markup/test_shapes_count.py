"""Сверка числа шейпов до и после пересоздания задачи."""

from ocr_utils.scan_markup.cvat.shapes import compare, count_task_shapes, read_snapshot, report, write_snapshot


class _Frame:
    def __init__(self, name: str) -> None:
        self.name = name


class _Shape:
    def __init__(self, frame: int) -> None:
        self.frame = frame


class _Annotations:
    def __init__(self, shapes) -> None:
        self.shapes = shapes


class _Task:
    def __init__(self, names, frames) -> None:
        self._names = [_Frame(n) for n in names]
        self._shapes = [_Shape(f) for f in frames]

    def get_frames_info(self):
        return self._names

    def get_annotations(self):
        return _Annotations(self._shapes)


def test_counts_by_frame_name() -> None:
    """Считаем по именам кадров: номера у пересозданной задачи другие."""
    task = _Task(["a.jpg", "b.jpg", "c.jpg"], [0, 0, 2])
    assert count_task_shapes(task) == {"a.jpg": 2, "c.jpg": 1}


def test_equal_snapshots_have_no_changes() -> None:
    snap = {"1973": {"a.jpg": 2, "b.jpg": 1}}
    assert compare(snap, snap) == []
    lines, lost = report(snap, snap)
    assert not lost and "сошлось" in lines[-1]


def test_lost_shape_is_reported() -> None:
    """Ради этого всё и затевалось: пропажу одного шейпа сумма по году не показывает."""
    before = {"1973": {"a.jpg": 1, "b.jpg": 1}}
    after = {"1973": {"a.jpg": 2, "b.jpg": 0}}

    assert compare(before, after) == [("1973", "a.jpg", 1, 2), ("1973", "b.jpg", 1, 0)]
    _lines, lost = report(before, after)
    assert lost, "по сумме год не изменился, а шейп потерян — сверка обязана это поймать"


def test_moved_page_is_followed_by_file_name() -> None:
    """Полоса переехала в другой выпуск: путь другой, имя файла то же."""
    before = {"1971": {"пак-1/1971/05/IMG_0053_1L.jpg": 1}}
    after = {"1971": {"пак-1/1971/04/IMG_0053_1L.jpg": 1}}

    assert compare(before, after) == []
    assert not report(before, after)[1]


def test_year_absent_from_the_new_snapshot_is_not_a_loss() -> None:
    """Снимок снимают по нескольким годам, а сверяют один — это не пропажа."""
    before = {"1972": {"a.jpg": 5}, "1973": {"b.jpg": 1}}
    after = {"1973": {"b.jpg": 1}}

    lines, lost = report(before, after)
    assert not lost
    assert any("Не проверялись" in line and "1972" in line for line in lines)


def test_snapshot_round_trip(tmp_path) -> None:
    snap = {"1973": {"пак-1/1973/01/a.jpg": 2}}
    path = tmp_path / "sub" / "snap.json"
    write_snapshot(snap, path)
    assert read_snapshot(path) == snap
