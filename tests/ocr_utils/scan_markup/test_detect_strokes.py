"""Крупный штрих на этапе detect: детектор line_art_detection, свои виды, своя версия, CVAT-метки."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from click.testing import CliRunner
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from ocr_utils.db.models import (
    KIND_GRAYSCALE,
    KIND_STROKE_DRAWING,
    KIND_STROKE_TABLE,
    SOURCE_AUTO,
    SOURCE_CVAT,
    STROKE_KINDS,
    TABLE_KINDS,
    Page,
    RectRegion,
)
from ocr_utils.db.session import open_db
from ocr_utils.line_art_detection.features import BOX_KIND_DRAWING, BOX_KIND_TABLE
from ocr_utils.scan_markup.cli import main
from ocr_utils.scan_markup.cvat.project import LABEL_BY_KIND, LABELS
from ocr_utils.scan_markup.detection.stroke_regions import STROKE_DETECTOR_VERSION, bitonal, find_strokes
from tests.ocr_utils.scan_markup import synthetic

DPI = 600
SIZE = (6000, 4000)  # полоса 600 dpi; копия 1/4 — 1500x1000 при 150 dpi, как у детектора геометрии
# Разлинованная таблица в пикселях оригинала — линейки НЕ касаются друг друга, как в печати.
TABLE = (600, 800, 3400, 3500)
# Штриховой рисунок — ломаная в нижней половине.
DRAWING = (600, 4000, 3400, 5600)
INK = 25


def _ruled_table(gray: np.ndarray) -> None:
    x0, y0, x1, y1 = TABLE
    for y in range(y0, y1 + 1, 300):
        gray[y : y + 10, x0:x1] = INK
    for x in range(x0, x1 + 1, 700):
        gray[y0:y1, x : x + 10] = INK
    for y in range(y0, y1 + 1, 300):
        for x in range(x0, x1 + 1, 700):
            gray[y - 12 : y + 22, x - 12 : x + 22] = synthetic.PAPER


def _drawing(gray: np.ndarray) -> None:
    x0, y0, x1, y1 = DRAWING
    rng = np.random.default_rng(0)
    points = rng.integers([x0, y0], [x1, y1], size=(60, 2)).reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(gray, [points], False, INK, 7)


def _page(path: Path, table: bool = True, drawing: bool = True) -> None:
    gray = synthetic.paper(SIZE)
    if table:
        _ruled_table(gray)
    if drawing:
        _drawing(gray)
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
            "--no-tables",
            *extra,
        ],
    )
    assert result.exit_code == 0, result.output + str(result.exception)
    return result


def _db_page(db: Path) -> Page:
    with open_db(db)() as session:
        return session.scalars(select(Page).options(selectinload(Page.rect_regions))).one()


def _work_gray() -> np.ndarray:
    gray = synthetic.paper(SIZE)
    _ruled_table(gray)
    _drawing(gray)
    return cv2.resize(gray, (SIZE[1] // 4, SIZE[0] // 4), interpolation=cv2.INTER_AREA)


def test_bitonal_is_two_valued_ink_zero() -> None:
    frame = bitonal(_work_gray())
    assert set(np.unique(frame)) == {0, 255}
    assert frame[TABLE[1] // 4 + 1, (TABLE[0] + 100) // 4] == 0, "линейка таблицы — краска"
    assert frame[10, 10] == 255, "поле — бумага"


def test_find_strokes_separates_table_from_drawing() -> None:
    findings = find_strokes(_work_gray(), DPI // 4)
    kinds = dict(zip(findings.kinds, findings.boxes))
    assert set(kinds) == {BOX_KIND_TABLE, BOX_KIND_DRAWING}
    table_box, drawing_box = kinds[BOX_KIND_TABLE], kinds[BOX_KIND_DRAWING]
    assert abs(table_box[0] * 4 - TABLE[0]) <= 60 and abs(table_box[1] * 4 - TABLE[1]) <= 60
    assert drawing_box[1] * 4 >= DRAWING[1] - 60 and drawing_box[3] * 4 <= DRAWING[3] + 60


def test_detect_writes_stroke_regions_with_their_own_version(tmp_path: Path) -> None:
    pack = tmp_path / "пак-2"
    _page(pack / "1977" / "01" / "0010_2R.tif")
    db = tmp_path / "m.sqlite"
    result = _run(pack, db)
    assert "Крупный штрих: таблиц 1, рисунков 1" in result.output

    page = _db_page(db)
    strokes = sorted((r for r in page.rect_regions if r.kind in STROKE_KINDS), key=lambda r: r.y1)
    assert [r.kind for r in strokes] == [KIND_STROKE_TABLE, KIND_STROKE_DRAWING]
    table, drawing = strokes
    assert table.source == SOURCE_AUTO and '"kind": "table"' in table.detector_info and '"rules"' in table.detector_info
    assert '"kind": "drawing"' in drawing.detector_info and '"ink"' in drawing.detector_info
    assert abs(table.x1 - TABLE[0]) <= 60 and abs(table.y1 - TABLE[1]) <= 60
    assert abs(table.x2 - TABLE[2]) <= 60 and abs(table.y2 - TABLE[3]) <= 60
    assert page.stroke_detector_version == STROKE_DETECTOR_VERSION and page.strokes_detected_at is not None
    assert page.tables_detected_at is None, "детектор таблиц выключен — его колонки не трогаются"
    assert not [r for r in page.rect_regions if r.kind in TABLE_KINDS]


def test_stale_strokes_alone_recompute_only_strokes(tmp_path: Path) -> None:
    """Четвёртая версия независима: устаревший штрих не пересчитывает растр, а свежий — пропускается."""
    pack = tmp_path / "пак-2"
    _page(pack / "1977" / "01" / "0010_2R.tif")
    db = tmp_path / "m.sqlite"
    _run(pack, db)
    with open_db(db)() as session:
        page = session.scalars(select(Page)).one()
        page.stroke_detector_version = STROKE_DETECTOR_VERSION - 1
        page.detected_at = None
        session.commit()

    result = _run(pack, db, "--skip-detected", "--no-raster")
    page = _db_page(db)
    assert "пропущено: 0" in result.output
    assert page.stroke_detector_version == STROKE_DETECTOR_VERSION
    assert page.detected_at is None

    result = _run(pack, db, "--skip-detected", "--no-raster")
    assert "пропущено: 1" in result.output


def test_strokes_do_not_touch_other_kinds_and_do_not_duplicate(tmp_path: Path) -> None:
    pack = tmp_path / "пак-2"
    _page(pack / "1977" / "01" / "0010_2R.tif")
    db = tmp_path / "m.sqlite"
    _run(pack, db, "--no-strokes")
    with open_db(db)() as session:
        page = session.scalars(select(Page)).one()
        assert page.strokes_detected_at is None
        page.rect_regions = [RectRegion(x1=1, y1=2, x2=300, y2=400, kind=KIND_GRAYSCALE, source=SOURCE_CVAT)]
        session.commit()

    _run(pack, db, "--skip-detected", "--no-raster")
    kinds = sorted((r.kind, r.source) for r in _db_page(db).rect_regions)
    assert kinds == [
        (KIND_GRAYSCALE, SOURCE_CVAT),
        (KIND_STROKE_DRAWING, SOURCE_AUTO),
        (KIND_STROKE_TABLE, SOURCE_AUTO),
    ]

    _run(pack, db, "--no-raster")
    assert sorted(r.kind for r in _db_page(db).rect_regions) == [KIND_GRAYSCALE, KIND_STROKE_DRAWING, KIND_STROKE_TABLE]


def test_cover_page_gets_an_empty_stroke_set(tmp_path: Path) -> None:
    pack = tmp_path / "пак-2"
    _page(pack / "1977" / "01" / "0010_2R.tif")
    db = tmp_path / "m.sqlite"
    result = CliRunner().invoke(
        main, ["detect", "--pack-dir", str(pack), "--db", str(db), "--no-use-surya-layout", "--no-orientation"]
    )
    assert result.exit_code == 0, result.output + str(result.exception)
    page = _db_page(db)
    assert page.strokes_detected_at is not None
    assert [r.kind for r in page.rect_regions if r.kind in STROKE_KINDS] == []


def test_stroke_kinds_have_cvat_rectangle_labels() -> None:
    names = {label["name"]: label for label in LABELS}
    for kind in STROKE_KINDS:
        assert LABEL_BY_KIND[kind] in names and names[LABEL_BY_KIND[kind]]["type"] == "rectangle"
