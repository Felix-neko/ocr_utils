"""Line art и повёрнутый текст на этапе detect: свои виды, свои версии, замена только своих видов."""

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
    KIND_LINE_ART_SCHEMA,
    KIND_ROTATED_TEXT,
    KIND_TABLE,
    LINE_ART_KINDS,
    ROTATED_TEXT_KINDS,
    SOURCE_AUTO,
    SOURCE_CVAT,
    TABLE_KINDS,
    Page,
    RectRegion,
)
from ocr_utils.db.session import open_db
from ocr_utils.page_layout import LINE_ART_VERSION, RASTER_VERSION, ROTATED_TEXT_VERSION, TABLES_VERSION
from ocr_utils.scan_markup.cli import main
from ocr_utils.scan_markup.cvat.project import LABEL_BY_KIND, LABELS
from tests.ocr_utils.page_layout.test_analysis import DRAWING, PHOTO, TABLE, _page_array
from tests.ocr_utils.scan_markup import synthetic

DPI = 600


def _write(path: Path, rotated=None) -> None:
    gray = _page_array(rotated=rotated)
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


def _iou(region: RectRegion, target) -> float:
    x0, y0, x1, y1 = (
        max(region.x1, target[0]),
        max(region.y1, target[1]),
        min(region.x2, target[2]),
        min(region.y2, target[3]),
    )
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (
        (region.x2 - region.x1) * (region.y2 - region.y1) + (target[2] - target[0]) * (target[3] - target[1]) - inter
    )
    return inter / union


def test_detect_writes_all_families_with_their_versions(tmp_path: Path) -> None:
    pack = tmp_path / "пак-2"
    _write(pack / "1977" / "01" / "0010_2R.tif", rotated=(3000, 2600))
    db = tmp_path / "m.sqlite"
    result = _run(pack, db)
    assert "Таблиц: 1" in result.output and "line art: " in result.output

    page = _page(db)
    by_kind = {}
    for region in page.rect_regions:
        by_kind.setdefault(region.kind, []).append(region)
    assert [r.kind for r in by_kind[KIND_GRAYSCALE]] == [KIND_GRAYSCALE] and _iou(
        by_kind[KIND_GRAYSCALE][0], PHOTO
    ) > 0.5
    assert len(by_kind[KIND_TABLE]) == 1 and '"kind": "таблица"' in by_kind[KIND_TABLE][0].detector_info
    arts = by_kind[KIND_LINE_ART_SCHEMA]
    assert arts and all(r.source == SOURCE_AUTO and '"sources"' in r.detector_info for r in arts)
    assert any(_iou(r, DRAWING) > 0.4 for r in arts)
    assert all(_iou(r, PHOTO) < 0.1 and _iou(r, TABLE) < 0.1 for r in arts), "line art не внутри растра и таблицы"
    assert by_kind[KIND_ROTATED_TEXT] and all(
        '"inside_line_art"' in r.detector_info for r in by_kind[KIND_ROTATED_TEXT]
    )
    assert (page.detector_version, page.table_detector_version) == (RASTER_VERSION, TABLES_VERSION)
    assert (page.line_art_version, page.rotated_text_version) == (LINE_ART_VERSION, ROTATED_TEXT_VERSION)
    assert page.line_art_detected_at is not None and page.rotated_text_detected_at is not None


def test_stale_line_art_alone_recomputes_only_line_art(tmp_path: Path) -> None:
    pack = tmp_path / "пак-2"
    _write(pack / "1977" / "01" / "0010_2R.tif")
    db = tmp_path / "m.sqlite"
    _run(pack, db)
    with open_db(db)() as session:
        page = session.scalars(select(Page)).one()
        page.line_art_version = LINE_ART_VERSION - 1
        page.detected_at = None
        session.commit()

    result = _run(pack, db, "--skip-detected", "--no-raster")
    page = _page(db)
    assert "пропущено: 0" in result.output
    assert page.line_art_version == LINE_ART_VERSION and page.detected_at is None

    result = _run(pack, db, "--skip-detected", "--no-raster")
    assert "пропущено: 1" in result.output


def test_line_art_rerun_keeps_manual_raster_and_does_not_duplicate(tmp_path: Path) -> None:
    pack = tmp_path / "пак-2"
    _write(pack / "1977" / "01" / "0010_2R.tif")
    db = tmp_path / "m.sqlite"
    _run(pack, db, "--no-line-art", "--no-rotated-text", "--no-tables")
    with open_db(db)() as session:
        page = session.scalars(select(Page)).one()
        assert page.line_art_detected_at is None and page.rotated_text_detected_at is None
        page.rect_regions = [RectRegion(x1=1, y1=2, x2=300, y2=400, kind=KIND_GRAYSCALE, source=SOURCE_CVAT)]
        session.commit()

    _run(pack, db, "--skip-detected", "--no-raster")
    kinds = sorted({(r.kind, r.source) for r in _page(db).rect_regions})
    assert (KIND_GRAYSCALE, SOURCE_CVAT) in kinds and (KIND_LINE_ART_SCHEMA, SOURCE_AUTO) in kinds
    before = sorted(r.kind for r in _page(db).rect_regions)
    _run(pack, db, "--no-raster")
    assert sorted(r.kind for r in _page(db).rect_regions) == before


def test_cover_page_gets_empty_families(tmp_path: Path) -> None:
    pack = tmp_path / "пак-2"
    _write(pack / "1977" / "01" / "0010_2R.tif")
    db = tmp_path / "m.sqlite"
    result = CliRunner().invoke(
        main, ["detect", "--pack-dir", str(pack), "--db", str(db), "--no-use-surya-layout", "--no-orientation"]
    )
    assert result.exit_code == 0, result.output + str(result.exception)
    page = _page(db)
    assert page.line_art_detected_at is not None and page.tables_detected_at is not None
    assert [r.kind for r in page.rect_regions] == ["color"] and page.rect_regions[0].full_page


def test_rect_kinds_have_cvat_rectangle_labels() -> None:
    names = {label["name"]: label for label in LABELS}
    for kind in TABLE_KINDS + LINE_ART_KINDS + ROTATED_TEXT_KINDS:
        assert LABEL_BY_KIND[kind] in names and names[LABEL_BY_KIND[kind]]["type"] == "rectangle"


def test_line_art_rerun_excludes_raster_known_from_db(tmp_path: Path) -> None:
    """Пересчёт одного line art видит растр из базы: штриха внутри фотографии не появляется."""
    pack = tmp_path / "пак-2"
    _write(pack / "1977" / "01" / "0010_2R.tif")
    db = tmp_path / "m.sqlite"
    _run(pack, db)
    with open_db(db)() as session:
        page = session.scalars(select(Page)).one()
        # Растр «уточнён человеком» и накрывает рисунок целиком; line art помечен устаревшим.
        page.rect_regions = [r for r in page.rect_regions if r.kind not in ("grayscale", "color")] + [
            RectRegion(
                x1=DRAWING[0], y1=DRAWING[1], x2=DRAWING[2], y2=DRAWING[3], kind=KIND_GRAYSCALE, source=SOURCE_CVAT
            )
        ]
        page.line_art_version = LINE_ART_VERSION - 1
        session.commit()
    _run(pack, db, "--skip-detected")
    page = _page(db)
    arts = [r for r in page.rect_regions if r.kind == KIND_LINE_ART_SCHEMA]
    assert all(_iou(r, DRAWING) < 0.1 for r in arts), "рисунок накрыт ручным растром — line art там не ищется"
    assert any(r.kind == KIND_GRAYSCALE and r.source == SOURCE_CVAT for r in page.rect_regions)
