"""Круговорот разметки: база -> шейпы CVAT -> база, на поддельной задаче.

Настоящий CVAT здесь не нужен: проверяется наша половина стыка — сборка шейпов,
сопоставление кадров с полосами и пересчёт координат. Всё, что делает сервер, сведено к
списку шейпов и списку кадров.
"""

import numpy as np
import pytest
from cvat_sdk.masks import encode_mask

from ocr_utils.scan_markup.cvat.export import ExportParams, ExportStats, import_task
from ocr_utils.scan_markup.cvat.project import (
    LABEL_COLOR_TEXT,
    LABEL_EXLIBRIS,
    LABEL_HANDWRITING,
    LABEL_LINE_ART,
    LABEL_RASTER_COLOR,
    LABEL_RASTER_GRAY,
    LABEL_STAMP,
    LABEL_TABLE,
    frame_index_by_name,
    rect_shapes,
)
from ocr_utils.scan_markup.db.models import (
    KIND_COLOR,
    KIND_COLOR_TEXT,
    KIND_GRAYSCALE,
    KIND_LINE_ART_SCHEMA,
    KIND_TABLE,
    MASK_HANDWRITING,
    POINT_EXLIBRIS,
    SOURCE_CVAT,
    Page,
    RectRegion,
)
from ocr_utils.scan_markup.db.repo import upsert_pack
from ocr_utils.scan_markup.db.session import open_db
from ocr_utils.scan_markup.scan_tree import ScannedIssue, ScannedPage, ScannedYear

W, H, D = 3492, 6051, 8
CVAT_W, CVAT_H = 436, 756
LABEL_IDS = {
    LABEL_RASTER_COLOR: 11,
    LABEL_RASTER_GRAY: 12,
    LABEL_STAMP: 13,
    LABEL_HANDWRITING: 14,
    LABEL_EXLIBRIS: 15,
    LABEL_COLOR_TEXT: 16,
    LABEL_TABLE: 17,
    LABEL_LINE_ART: 18,
}
LABEL_NAMES = {value: key for key, value in LABEL_IDS.items()}


class _Frame:
    def __init__(self, name: str) -> None:
        self.name = name


class _Shape:
    """Шейп в том виде, в каком его отдаёт SDK: тип, кадр, метка, points."""

    def __init__(self, type_: str, frame: int, label_id: int, points, shape_id: int = 1) -> None:
        self.type = type_
        self.frame = frame
        self.label_id = label_id
        self.points = points
        self.id = shape_id


class _Annotations:
    """Двойник ответа CVAT. Теги есть ВСЕГДА, пусть и пустые: у настоящего ответа поле
    ``tags`` есть всегда, и двойник без него прятал бы работу с ориентацией от тестов."""

    def __init__(self, shapes, tags=()) -> None:
        self.shapes = shapes
        self.tags = list(tags)


class _Task:
    """Поддельная задача: кадры и разметка заданы списками."""

    def __init__(self, frame_names, shapes) -> None:
        self._frames = [_Frame(name) for name in frame_names]
        self._shapes = shapes

    def get_frames_info(self):
        return self._frames

    def get_annotations(self):
        return _Annotations(self._shapes)


@pytest.fixture
def page_and_session(tmp_path):
    """Одна полоса в базе с реальными параметрами уменьшения пака-1."""
    session_factory = open_db(tmp_path / "markup.sqlite")
    tree = [
        ScannedYear(
            name="1974",
            year=1974,
            rel_path="1974",
            issues=[
                ScannedIssue(
                    name="01",
                    number=1,
                    rel_path="1974/01",
                    pages=[
                        ScannedPage(path=tmp_path / "a.tif", file_name="a.tif", rel_path="1974/01/a.tif", order_index=0)
                    ],
                )
            ],
        )
    ]
    session = session_factory()
    pack = upsert_pack(session, "пак-1", tmp_path, tree)
    page = pack.year_packages[0].issues[0].pages[0]
    page.width, page.height, page.dpi, page.divisor = W, H, 600, D
    page.crop_width, page.crop_height = 3488, 6048
    page.cvat_width, page.cvat_height = CVAT_W, CVAT_H
    page.cvat_rel_path = "пак-1/1974/01/a.jpg"
    page.cvat_frame = 0
    session.commit()
    yield page, session
    session.close()


def test_frame_index_by_name_uses_server_numbering() -> None:
    """Индекс кадра берётся у сервера, а не из позиции файла в нашем списке."""
    task = _Task(["p/a.jpg", "p/b.jpg", "p/c.jpg"], [])
    assert frame_index_by_name(task) == {"p/a.jpg": 0, "p/b.jpg": 1, "p/c.jpg": 2}


def test_preannotation_shapes_are_scaled_down(page_and_session) -> None:
    """Найденные области уезжают в CVAT поделёнными на делитель и с нужной меткой."""
    page, _session = page_and_session
    region = RectRegion(x1=800, y1=1600, x2=3000, y2=5000, kind=KIND_COLOR)

    shapes = rect_shapes([(page, [region])], {page.cvat_rel_path: 0}, LABEL_IDS)
    assert len(shapes) == 1
    assert shapes[0].label_id == LABEL_IDS[LABEL_RASTER_COLOR]
    assert shapes[0].points == [100.0, 200.0, 375.0, 625.0]


def test_preannotation_skips_pages_outside_this_task(page_and_session) -> None:
    """Полоса, которой нет среди кадров задачи, шейпов не даёт (прогон по одному году)."""
    page, _session = page_and_session
    region = RectRegion(x1=800, y1=1600, x2=3000, y2=5000, kind=KIND_COLOR)
    assert rect_shapes([(page, [region])], {"чужой/кадр.jpg": 0}, LABEL_IDS) == []


def test_full_round_trip_keeps_geometry(page_and_session) -> None:
    """Область оригинала -> CVAT -> обратно: промах не больше делителя."""
    page, session = page_and_session
    original = RectRegion(x1=800, y1=1600, x2=3000, y2=5000, kind=KIND_COLOR)
    shapes = rect_shapes([(page, [original])], {page.cvat_rel_path: 0}, LABEL_IDS)

    task = _Task([page.cvat_rel_path], [_Shape("rectangle", 0, LABEL_IDS[LABEL_RASTER_COLOR], list(shapes[0].points))])
    stats = ExportStats()
    import_task(task, LABEL_NAMES, {0: page}, session, ExportParams(None, None, "пак-1"), stats)

    assert stats.regions == 1 and stats.color == 1
    back = page.rect_regions[0]
    assert back.source == SOURCE_CVAT
    for got, want in zip((back.x1, back.y1, back.x2, back.y2), (800, 1600, 3000, 5000)):
        assert abs(got - want) <= D


def test_import_reads_kind_from_label(page_and_session) -> None:
    """Смена метки в CVAT — это и есть смена типа color/grayscale."""
    page, session = page_and_session
    task = _Task([page.cvat_rel_path], [_Shape("rectangle", 0, LABEL_IDS[LABEL_RASTER_GRAY], [10, 10, 100, 100])])
    import_task(task, LABEL_NAMES, {0: page}, session, ExportParams(None, None, "пак-1"), ExportStats())
    assert page.rect_regions[0].kind == KIND_GRAYSCALE


def test_import_marks_full_page_by_area(page_and_session) -> None:
    """Обложка, обведённая во всю полосу, получает признак full_page."""
    page, session = page_and_session
    task = _Task([page.cvat_rel_path], [_Shape("rectangle", 0, LABEL_IDS[LABEL_RASTER_COLOR], [0, 0, CVAT_W, CVAT_H])])
    import_task(task, LABEL_NAMES, {0: page}, session, ExportParams(None, None, "пак-1"), ExportStats())

    region = page.rect_regions[0]
    assert region.full_page
    assert (region.x2, region.y2) == (W, H)  # растянулось в обрезанную полоску


def test_import_stores_stamp_mask(page_and_session) -> None:
    """Маска печати попадает в свою таблицу, а не в растровые области."""
    page, session = page_and_session
    drawn = np.zeros((CVAT_H, CVAT_W), bool)
    drawn[100:140, 200:260] = True
    task = _Task([page.cvat_rel_path], [_Shape("mask", 0, LABEL_IDS[LABEL_STAMP], encode_mask(drawn))])

    stats = ExportStats()
    import_task(task, LABEL_NAMES, {0: page}, session, ExportParams(None, None, "пак-1"), stats)

    assert stats.masks == 1 and stats.regions == 0
    assert page.rect_regions == []
    assert (page.masks[0].left, page.masks[0].top) == (200 * D, 100 * D)


def test_import_replaces_previous_markup(page_and_session) -> None:
    """Повторная выгрузка — снимок состояния: удалённое разметчиком исчезает из базы."""
    page, session = page_and_session
    params, stats = ExportParams(None, None, "пак-1"), ExportStats()

    two = _Task(
        [page.cvat_rel_path],
        [
            _Shape("rectangle", 0, LABEL_IDS[LABEL_RASTER_COLOR], [10, 10, 100, 100], 1),
            _Shape("rectangle", 0, LABEL_IDS[LABEL_RASTER_GRAY], [110, 110, 200, 200], 2),
        ],
    )
    import_task(two, LABEL_NAMES, {0: page}, session, params, stats)
    assert len(page.rect_regions) == 2

    one = _Task([page.cvat_rel_path], [_Shape("rectangle", 0, LABEL_IDS[LABEL_RASTER_COLOR], [10, 10, 100, 100])])
    import_task(one, LABEL_NAMES, {0: page}, session, params, ExportStats())
    assert len(page.rect_regions) == 1


def test_page_without_shapes_is_marked_reviewed(page_and_session) -> None:
    """«Растра нет» — тоже результат, и он должен отличаться от «сюда не смотрели»."""
    page, session = page_and_session
    assert page.reviewed_at is None
    import_task(
        _Task([page.cvat_rel_path], []),
        LABEL_NAMES,
        {0: page},
        session,
        ExportParams(None, None, "пак-1"),
        ExportStats(),
    )
    assert page.reviewed_at is not None and page.rect_regions == []


def test_foreign_labels_are_counted_not_imported(page_and_session) -> None:
    """Шейп с чужой меткой не роняет прогон и попадает в счётчик."""
    page, session = page_and_session
    stats = ExportStats()
    task = _Task([page.cvat_rel_path], [_Shape("polygon", 0, 99, [1, 2, 3, 4])])
    import_task(task, LABEL_NAMES, {0: page}, session, ExportParams(None, None, "пак-1"), stats)
    assert stats.unknown_labels == 1 and page.rect_regions == []


def test_import_stores_handwriting_with_its_own_kind(page_and_session) -> None:
    """Второй вид маски отличается от печати только значением ``kind``, таблица одна."""
    page, session = page_and_session
    drawn = np.zeros((CVAT_H, CVAT_W), bool)
    drawn[10:30, 40:90] = True
    task = _Task([page.cvat_rel_path], [_Shape("mask", 0, LABEL_IDS[LABEL_HANDWRITING], encode_mask(drawn))])

    stats = ExportStats()
    import_task(task, LABEL_NAMES, {0: page}, session, ExportParams(None, None, "пак-1"), stats)

    assert stats.masks == 1
    assert page.masks[0].kind == MASK_HANDWRITING


def test_import_stores_exlibris_point(page_and_session) -> None:
    """Точка экслибриса едет в свою таблицу и пересчитывается в координаты оригинала."""
    page, session = page_and_session
    task = _Task([page.cvat_rel_path], [_Shape("points", 0, LABEL_IDS[LABEL_EXLIBRIS], [120.0, 300.0])])

    stats = ExportStats()
    import_task(task, LABEL_NAMES, {0: page}, session, ExportParams(None, None, "пак-1"), stats)

    assert stats.points == 1 and stats.regions == 0 and stats.masks == 0
    point = page.points[0]
    assert (point.x, point.y) == (120 * D, 300 * D)
    assert point.kind == POINT_EXLIBRIS
    assert point.source == SOURCE_CVAT and point.source_divisor == D


def test_import_replaces_points_too(page_and_session) -> None:
    """Точки — такой же снимок состояния, как области и маски: снятое исчезает."""
    page, session = page_and_session
    params = ExportParams(None, None, "пак-1")
    first = _Task([page.cvat_rel_path], [_Shape("points", 0, LABEL_IDS[LABEL_EXLIBRIS], [10.0, 20.0])])
    import_task(first, LABEL_NAMES, {0: page}, session, params, ExportStats())
    assert len(page.points) == 1

    import_task(_Task([page.cvat_rel_path], []), LABEL_NAMES, {0: page}, session, params, ExportStats())
    assert page.points == []


def test_import_stores_colour_text_region(page_and_session) -> None:
    """Область цветного набора — обычный прямоугольник со своим ``kind``."""
    page, session = page_and_session
    task = _Task([page.cvat_rel_path], [_Shape("rectangle", 0, LABEL_IDS[LABEL_COLOR_TEXT], [10, 20, 110, 220])])

    stats = ExportStats()
    import_task(task, LABEL_NAMES, {0: page}, session, ExportParams(None, None, "пак-1"), stats)

    assert stats.regions == 1 and stats.color == 0 and stats.grayscale == 0
    region = page.rect_regions[0]
    assert region.kind == KIND_COLOR_TEXT
    assert (region.x1, region.y1) == (10 * D, 20 * D)


def test_table_and_line_art_round_trip(page_and_session) -> None:
    """Таблица и схема уезжают своими метками и возвращаются своими видами.

    ``detector_info`` назад не едет: разметчик его не видел и не правил, а ``from-cvat`` —
    снимок разметки, не слияние с автоматикой.
    """
    page, session = page_and_session
    table = RectRegion(x1=800, y1=1600, x2=3000, y2=3000, kind=KIND_TABLE, detector_info='{"kind_fine": "таблица"}')
    schema = RectRegion(x1=400, y1=3200, x2=3200, y2=5600, kind=KIND_LINE_ART_SCHEMA)
    shapes = rect_shapes([(page, [table, schema])], {page.cvat_rel_path: 0}, LABEL_IDS)
    assert [shape.label_id for shape in shapes] == [LABEL_IDS[LABEL_TABLE], LABEL_IDS[LABEL_LINE_ART]]

    task = _Task(
        [page.cvat_rel_path],
        [
            _Shape("rectangle", 0, shape.label_id, list(shape.points), shape_id=index)
            for index, shape in enumerate(shapes)
        ],
    )
    stats = ExportStats()
    import_task(task, LABEL_NAMES, {0: page}, session, ExportParams(None, None, "пак-1"), stats)

    assert (stats.regions, stats.table, stats.line_art, stats.color) == (2, 1, 1, 0)
    by_kind = {region.kind: region for region in page.rect_regions}
    assert set(by_kind) == {KIND_TABLE, KIND_LINE_ART_SCHEMA}
    assert abs(by_kind[KIND_TABLE].x1 - 800) <= D and abs(by_kind[KIND_LINE_ART_SCHEMA].y2 - 5600) <= D
    assert by_kind[KIND_TABLE].source == SOURCE_CVAT and by_kind[KIND_TABLE].detector_info is None


def test_rect_shapes_can_be_limited_to_kinds(page_and_session) -> None:
    """Дозаливка таблиц берёт только свои виды, растр в PATCH не попадает."""
    page, _session = page_and_session
    regions = [
        RectRegion(x1=800, y1=1600, x2=3000, y2=3000, kind=KIND_COLOR),
        RectRegion(x1=800, y1=3200, x2=3000, y2=5000, kind=KIND_TABLE),
    ]
    shapes = rect_shapes([(page, regions)], {page.cvat_rel_path: 0}, LABEL_IDS, kinds=(KIND_TABLE,))
    assert [shape.label_id for shape in shapes] == [LABEL_IDS[LABEL_TABLE]]
