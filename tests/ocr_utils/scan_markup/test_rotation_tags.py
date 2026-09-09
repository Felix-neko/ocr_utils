"""Ориентация полосы в CVAT: тег кадра туда и обратно.

Тег не виден на холсте, и потерять его легче всего именно молча — поэтому проверяется не
только «доехал», но и «снятый разметчиком тег означает отсутствие поворота».
"""

from __future__ import annotations

import pytest

from ocr_utils.scan_markup.cvat.export import _rotation_from_tags
from ocr_utils.scan_markup.cvat.project import (
    LABEL_BY_ROTATION,
    LABEL_ROTATE_CCW90,
    LABEL_ROTATE_CW90,
    LABELS,
    ROTATION_BY_LABEL,
    rotation_tags,
    tags_to_requests,
    upload_preannotations,
)


class _Page:
    def __init__(self, rel_path, rotate_cw):
        self.cvat_rel_path = rel_path
        self.rotate_cw = rotate_cw


class _Tag:
    def __init__(self, frame, label_id, group=0):
        self.frame, self.label_id, self.group = frame, label_id, group


class _Stats:
    conflicting_rotations = 0


FRAMES = {"пак-1/1967/01/a.jpg": 0, "пак-1/1967/01/b.jpg": 1, "пак-1/1967/01/c.jpg": 2}
LABEL_IDS = {label: 100 + index for index, label in enumerate(LABEL_BY_ROTATION.values())}


def test_tag_labels_are_declared_as_tags_not_shapes():
    """Тег обязан быть типа tag: фигура на холсте мешала бы рисовать и могла бы уехать."""
    by_name = {label["name"]: label for label in LABELS}
    for label in LABEL_BY_ROTATION.values():
        assert by_name[label]["type"] == "tag"


def test_only_rotated_pages_get_a_tag():
    """Полосе без поворота тег не вешается: отсутствие тега и есть «поворот не нужен»."""
    pages = [_Page("пак-1/1967/01/a.jpg", 90), _Page("пак-1/1967/01/b.jpg", 0), _Page("пак-1/1967/01/c.jpg", None)]
    tags = rotation_tags(pages, FRAMES, LABEL_IDS)
    assert [(tag.frame, tag.label_id) for tag in tags] == [(0, LABEL_IDS[LABEL_ROTATE_CW90])]


def test_pages_outside_the_task_are_skipped():
    """Прогон по подмножеству лет не должен падать на чужих полосах."""
    assert rotation_tags([_Page("пак-1/1999/01/z.jpg", 90)], FRAMES, LABEL_IDS) == []


@pytest.mark.parametrize("rotation", sorted(LABEL_BY_ROTATION))
def test_every_rotation_survives_the_round_trip(rotation):
    tags = rotation_tags([_Page("пак-1/1967/01/a.jpg", rotation)], FRAMES, LABEL_IDS)
    names = {label_id: label for label, label_id in LABEL_IDS.items()}
    back = _rotation_from_tags([_Tag(0, tags[0].label_id)], names, _Stats())
    assert back == rotation


def test_a_frame_without_tags_means_no_rotation_needed():
    """Снятый разметчиком тег обязан читаться как «поворот не нужен», а не «не знаем»."""
    assert _rotation_from_tags([], {}, _Stats()) == 0


def test_conflicting_tags_are_counted_and_resolved_deterministically():
    """Два взаимоисключающих тега — ошибка разметки; молча брать любой нельзя."""
    names = {LABEL_IDS[LABEL_ROTATE_CW90]: LABEL_ROTATE_CW90, LABEL_IDS[LABEL_ROTATE_CCW90]: LABEL_ROTATE_CCW90}
    stats = _Stats()
    got = _rotation_from_tags(
        [_Tag(0, LABEL_IDS[LABEL_ROTATE_CCW90]), _Tag(0, LABEL_IDS[LABEL_ROTATE_CW90])], names, stats
    )
    assert got == 270 and stats.conflicting_rotations == 1
    # Порядок выдачи сервера не должен менять ответ.
    assert (
        _rotation_from_tags(
            [_Tag(0, LABEL_IDS[LABEL_ROTATE_CW90]), _Tag(0, LABEL_IDS[LABEL_ROTATE_CCW90])], names, _Stats()
        )
        == 270
    )


def test_foreign_labels_do_not_look_like_a_rotation():
    assert _rotation_from_tags([_Tag(0, 999)], {999: "Растр цветной"}, _Stats()) == 0


def test_tags_are_carried_into_a_rebuilt_task():
    """Без переноса --recreate-stale терял бы ручную разметку ориентации молча."""
    by_frame = {"пак-1/1967/01/a.jpg": [_Tag(0, 101)], "пак-1/1967/01/b.jpg": [_Tag(1, 102)]}
    carried = tags_to_requests(by_frame, {"пак-1/1967/01/a.jpg": 5, "пак-1/1967/01/b.jpg": 6}, set())
    assert sorted((tag.frame, tag.label_id) for tag in carried) == [(5, 101), (6, 102)]


def test_upload_sends_tags_together_with_shapes():
    """set_annotations — это PUT: залить одни шейпы значило бы стереть все теги."""
    sent = {}

    class _Task:
        def set_annotations(self, request):
            sent["shapes"] = list(request.shapes)
            sent["tags"] = list(request.tags)

    total = upload_preannotations(_Task(), [], rotation_tags([_Page("пак-1/1967/01/a.jpg", 90)], FRAMES, LABEL_IDS))
    assert total == 1, "год без единого прямоугольника, но с тегами, обязан залиться"
    assert sent["shapes"] == [] and len(sent["tags"]) == 1
