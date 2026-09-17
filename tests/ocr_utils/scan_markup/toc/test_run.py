"""Прогон по базе: окно, запись колонок, --skip-detected, ручное решение из CVAT, выгрузка списков."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import select

from ocr_utils.scan_markup.db.models import Page
from ocr_utils.scan_markup.db.repo import upsert_pack
from ocr_utils.scan_markup.db.session import open_db
from ocr_utils.scan_markup.scan_tree import scan_pack
from ocr_utils.scan_markup.toc import KIND_CONTENTS, KIND_INDEX, SOURCE_AUTO, SOURCE_CVAT, TOC_VERSION, kind_from_flags
from ocr_utils.scan_markup.toc import run as toc_run
from ocr_utils.scan_markup.db.repo import require_pack
from ocr_utils.scan_markup.toc.export import LIST_NAME, export_lists
from ocr_utils.scan_markup.toc.features import PageFeatures
from ocr_utils.scan_markup.toc.run import TocParams, run_toc

N_PAGES = 30
PACK = "пак-т"


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 6), 255, np.uint8)).save(path)


@pytest.fixture
def pack_dir(tmp_path: Path) -> Path:
    root = tmp_path / PACK
    for index in range(N_PAGES):
        _make_image(root / "1975" / "12" / f"IMG_{index:04d}.tif")
    return root


@pytest.fixture
def session_factory(tmp_path: Path, pack_dir: Path):
    factory = open_db(tmp_path / "markup.sqlite")
    with factory() as session:
        upsert_pack(session, PACK, pack_dir, scan_pack(pack_dir))
        session.commit()
    return factory


@pytest.fixture
def fake_features(monkeypatch):
    """Вместо tesseract: полосы 2-3 — содержание, три последние — указатель."""
    measured: list[str] = []

    def fake(path, rel_path, order_index, idx_from_end, layout_cache_dir, want_thumbnail=False, default_dpi=600):
        measured.append(rel_path)
        features = PageFeatures(rel_path, order_index, idx_from_end)
        if order_index == 2:
            return PageFeatures(
                rel_path, order_index, idx_from_end, kw_contents=True, num_tail_lines=9, num_tail_ratio=0.3
            )
        if order_index == 3:
            return PageFeatures(
                rel_path, order_index, idx_from_end, num_tail_lines=2, num_tail_ratio=0.06, kw_imprint=True
            )
        if idx_from_end == 3:
            return PageFeatures(
                rel_path, order_index, idx_from_end, kw_index=True, num_tail_lines=30, num_tail_ratio=0.6
            )
        if idx_from_end in (1, 2):
            return PageFeatures(
                rel_path, order_index, idx_from_end, num_tail_lines=30, num_tail_ratio=0.6, surya_table_area=0.7
            )
        return features

    monkeypatch.setattr(toc_run, "page_features", fake)
    return measured


def _params(pack_dir: Path, tmp_path: Path, **kwargs) -> TocParams:
    return TocParams(
        db_path=tmp_path / "markup.sqlite", pack_name=PACK, pack_dir=pack_dir, jobs=1, progress=False, **kwargs
    )


def test_run_measures_only_the_window_and_writes_columns(pack_dir, tmp_path, session_factory, fake_features) -> None:
    stats, results = run_toc(_params(pack_dir, tmp_path), session_factory)
    assert stats.issues == 1 and stats.contents == 2 and stats.index == 3 and not stats.issues_without_contents
    assert stats.pages_measured == 5 + 12 == len(fake_features)
    with session_factory() as session:
        pages = session.scalars(select(Page).order_by(Page.order_index)).all()
        assert all(p.toc_version == TOC_VERSION and p.toc_source == SOURCE_AUTO for p in pages)
        kinds = {p.order_index: kind_from_flags(p.is_toc, p.is_year_index) for p in pages}
        assert all(p.is_toc is not None and p.is_year_index is not None for p in pages), "искали везде"
        assert kinds[2] == KIND_CONTENTS and kinds[3] == KIND_CONTENTS, "пара 3+4"
        assert (
            kinds[N_PAGES - 4] == KIND_INDEX and kinds[N_PAGES - 3] == KIND_INDEX and kinds[N_PAGES - 2] == KIND_INDEX
        )
        assert kinds[N_PAGES - 1] is None and kinds[10] is None
        assert pages[2].toc_score == 1.0 and pages[N_PAGES - 2].toc_score == 0.6


def test_skip_detected_and_cvat_source_are_respected(pack_dir, tmp_path, session_factory, fake_features) -> None:
    run_toc(_params(pack_dir, tmp_path), session_factory)
    with session_factory() as session:
        manual = session.scalar(select(Page).where(Page.order_index == 10))
        manual.is_toc, manual.toc_source = True, SOURCE_CVAT
        session.commit()

    fake_features.clear()
    stats, _ = run_toc(_params(pack_dir, tmp_path, skip_detected=True), session_factory)
    assert stats.skipped == 1 and not fake_features, "выпуск посчитан текущей версией — не трогать"

    stats, _ = run_toc(_params(pack_dir, tmp_path), session_factory)
    assert stats.issues == 1
    with session_factory() as session:
        manual = session.scalar(select(Page).where(Page.order_index == 10))
        assert manual.is_toc and manual.toc_source == SOURCE_CVAT, "ручное решение не затёрто"


def test_dry_run_and_csv(pack_dir, tmp_path, session_factory, fake_features) -> None:
    csv_path = tmp_path / "toc.csv"
    run_toc(_params(pack_dir, tmp_path, dry_run=True, csv_path=csv_path), session_factory)
    with session_factory() as session:
        assert all(p.toc_version is None for p in session.scalars(select(Page)).all())
    text = csv_path.read_text(encoding="utf-8")
    assert text.count("\n") == 1 + 17 and "contents" in text and "index" in text


def test_export_lists(pack_dir, tmp_path, session_factory, fake_features) -> None:
    out = tmp_path / "lists"
    stats = export_lists(session_factory, PACK, out)
    assert stats.issues == 0 and stats.not_detected == ["1975/12"]

    run_toc(_params(pack_dir, tmp_path), session_factory)
    stats = export_lists(session_factory, PACK, out)
    assert stats.issues == 1 and stats.pages == 5 and not stats.empty
    lines = (out / "1975" / "12" / LIST_NAME).read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("IMG_0002.jpg  # contents 1.00 auto")
    assert [line.split()[0] for line in lines] == [
        f"IMG_{i:04d}.jpg" for i in (2, 3, N_PAGES - 4, N_PAGES - 3, N_PAGES - 2)
    ]

    stats = export_lists(session_factory, PACK, out, kinds=(KIND_INDEX,))
    assert stats.pages == 3

    # Вето «Не оглавление» сильнее признака: полоса выпадает из списка.
    with session_factory() as session:
        pack = require_pack(session, PACK)
        page = next(p for p in pack.year_packages[0].issues[0].pages if p.order_index == 2)
        page.force_is_not_toc = True
        session.commit()
    assert export_lists(session_factory, PACK, out).pages == 4
