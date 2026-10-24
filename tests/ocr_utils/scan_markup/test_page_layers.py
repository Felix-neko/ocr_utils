"""Слои страниц v2 в CVAT: зона обрезки (прямоугольник, туда и обратно) и закрас пальцев (замкнутый контур, только туда; в reviewed-базу переносится из исходной)."""

import json

from ocr_utils.db.models import ContentBox, FingerZone, SOURCE_CVAT
from ocr_utils.db.session import open_db
from ocr_utils.scan_markup.cvat.export import ExportParams, ExportStats, copy_tree, import_task
from ocr_utils.scan_markup.cvat.project import (
    LABEL_CONTENT_BOX,
    LABEL_FINGER_ZONE,
    content_box_shapes,
    finger_zone_shapes,
    polygon_to_cvat,
)
from tests.ocr_utils.scan_markup.test_cvat_roundtrip import (  # noqa: F401 — фикстура
    CVAT_H,
    CVAT_W,
    D,
    _Shape,
    _Task,
    page_and_session,
)

LABEL_IDS = {LABEL_CONTENT_BOX: 31, LABEL_FINGER_ZONE: 32}
LABEL_NAMES = {value: key for key, value in LABEL_IDS.items()}


def test_polygon_to_cvat_is_closed_and_scaled() -> None:
    points = polygon_to_cvat([[80, 160], [800, 160], [800, 960]], D, CVAT_W, CVAT_H)
    assert points == [10.0, 20.0, 100.0, 20.0, 100.0, 120.0, 10.0, 20.0]


def test_layers_go_to_cvat(page_and_session) -> None:
    page, session = page_and_session
    page.content_box = ContentBox(x1=800, y1=1600, x2=3000, y2=5000, method="hybrid")
    page.finger_zones = [FingerZone(polygon=json.dumps([[80, 160], [800, 160], [800, 960]]), area_mm2=12.0)]
    session.commit()
    frames = {page.cvat_rel_path: 0}
    boxes = content_box_shapes([page], frames, LABEL_IDS)
    zones = finger_zone_shapes([page], frames, LABEL_IDS)
    assert len(boxes) == 1 and boxes[0].points == [100.0, 200.0, 375.0, 625.0]
    assert len(zones) == 1 and str(getattr(zones[0].type, "value", zones[0].type)) == "polyline"
    assert len(zones[0].points) == 8


def test_content_box_round_trip_and_finger_zone_ignored(page_and_session) -> None:
    page, session = page_and_session
    task = _Task(
        [page.cvat_rel_path],
        [
            _Shape("rectangle", 0, LABEL_IDS[LABEL_CONTENT_BOX], [100.0, 200.0, 375.0, 625.0], 5),
            _Shape("polyline", 0, LABEL_IDS[LABEL_FINGER_ZONE], [1, 2, 3, 4, 1, 2], 6),
        ],
    )
    stats = ExportStats()
    import_task(task, LABEL_NAMES, {0: page}, session, ExportParams(None, None, "пак-1"), stats)
    box = page.content_box
    assert stats.content_boxes == 1 and stats.unknown_labels == 0
    assert box.source == SOURCE_CVAT and box.cvat_shape_id == 5
    assert (box.x1, box.y1, box.x2, box.y2) == (800, 1600, 3000, 5000)
    # Снимок состояния: без прямоугольника в CVAT зоны нет.
    import_task(_Task([page.cvat_rel_path], []), LABEL_NAMES, {0: page}, session, ExportParams(None, None, "пак-1"),
                ExportStats())  # fmt: skip
    assert page.content_box is None


def test_two_boxes_merge_to_union(page_and_session) -> None:
    page, session = page_and_session
    task = _Task(
        [page.cvat_rel_path],
        [
            _Shape("rectangle", 0, LABEL_IDS[LABEL_CONTENT_BOX], [10, 10, 100, 100], 1),
            _Shape("rectangle", 0, LABEL_IDS[LABEL_CONTENT_BOX], [50, 50, 200, 300], 2),
        ],
    )
    stats = ExportStats()
    import_task(task, LABEL_NAMES, {0: page}, session, ExportParams(None, None, "пак-1"), stats)
    assert stats.content_box_conflicts == 1
    assert (page.content_box.x1, page.content_box.y2) == (10 * D, 300 * D)


def test_copy_tree_carries_finger_zones(page_and_session, tmp_path) -> None:
    page, session = page_and_session
    page.finger_zones = [FingerZone(polygon="[[1, 2], [3, 4], [5, 6]]", area_mm2=3.5)]
    session.commit()
    dst = open_db(tmp_path / "reviewed.sqlite")()
    pack = copy_tree(session, dst, "пак-1")
    copied = pack.year_packages[0].issues[0].pages[0]
    assert [(z.polygon, z.area_mm2) for z in copied.finger_zones] == [("[[1, 2], [3, 4], [5, 6]]", 3.5)]
